// Decoder timestamps and pinned JSON fixtures intentionally use the model's f32/usize types.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{cmp::Ordering, collections::HashMap};

use anyhow::{Result, anyhow, bail};

use crate::AsrTranscript;

const NVIDIA_BATCHED_BEAM_THRESHOLD: f32 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtcDecodingStrategy {
    Greedy,
    BatchedBeam { beam_size: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CtcPath {
    pub token_ids: Vec<usize>,
    pub frame_indices: Vec<usize>,
    pub score: f32,
}

/// Decodes a frame-major CTC log-probability tensor.
///
/// # Errors
///
/// Returns an error when the tensor shape, blank identifier, or beam size is invalid.
pub fn decode_ctc(
    log_probs: &[f32],
    num_frames: usize,
    vocab_size: usize,
    blank_id: usize,
    strategy: CtcDecodingStrategy,
) -> Result<CtcPath> {
    validate_logits(log_probs, num_frames, vocab_size, blank_id)?;
    match strategy {
        CtcDecodingStrategy::Greedy => Ok(greedy_path(log_probs, num_frames, vocab_size, blank_id)),
        CtcDecodingStrategy::BatchedBeam { beam_size } => {
            batched_beam_path(log_probs, num_frames, vocab_size, blank_id, beam_size)
        }
    }
}

/// Converts a decoded CTC path into the shared transcript representation.
///
/// # Errors
///
/// Returns an error when a decoded token identifier is missing from `tokens`.
pub fn transcript_from_path(
    path: &CtcPath,
    tokens: &[String],
    frame_shift_sec: f64,
) -> Result<AsrTranscript> {
    let token_texts = path
        .token_ids
        .iter()
        .map(|&token_id| {
            tokens
                .get(token_id)
                .map(|token| token.replace('▁', " "))
                .ok_or_else(|| anyhow!("CTC token id {token_id} is outside the token table"))
        })
        .collect::<Result<Vec<_>>>()?;
    let timestamps = path
        .frame_indices
        .iter()
        .map(|&frame| (frame as f64 * frame_shift_sec) as f32)
        .collect::<Vec<_>>();
    let text = token_texts.concat();
    Ok(AsrTranscript::from_parts(
        text,
        token_texts,
        Some(&timestamps),
        None,
    ))
}

fn validate_logits(
    log_probs: &[f32],
    num_frames: usize,
    vocab_size: usize,
    blank_id: usize,
) -> Result<()> {
    if vocab_size == 0 {
        bail!("CTC vocabulary must not be empty");
    }
    if blank_id >= vocab_size {
        bail!("CTC blank id {blank_id} is outside vocabulary size {vocab_size}");
    }
    let expected = num_frames
        .checked_mul(vocab_size)
        .ok_or_else(|| anyhow!("CTC logit shape overflow"))?;
    if log_probs.len() != expected {
        bail!(
            "CTC logit length {} does not match [{num_frames}, {vocab_size}]",
            log_probs.len()
        );
    }
    Ok(())
}

fn greedy_path(
    log_probs: &[f32],
    num_frames: usize,
    vocab_size: usize,
    blank_id: usize,
) -> CtcPath {
    let mut token_ids = Vec::new();
    let mut frame_indices = Vec::new();
    let mut score = 0.0;
    let mut previous = blank_id;
    for (frame_index, frame) in log_probs
        .chunks_exact(vocab_size)
        .take(num_frames)
        .enumerate()
    {
        let (token_id, &token_score) = frame
            .iter()
            .enumerate()
            .max_by(|left, right| total_cmp(*left.1, *right.1))
            .expect("a validated vocabulary has at least one token");
        score += token_score;
        if token_id != blank_id && token_id != previous {
            token_ids.push(token_id);
            frame_indices.push(frame_index);
        }
        previous = token_id;
    }
    CtcPath {
        token_ids,
        frame_indices,
        score,
    }
}

#[derive(Debug, Clone)]
struct BeamState {
    token_ids: Vec<usize>,
    frame_indices: Vec<usize>,
    last_label: usize,
    score: f32,
}

fn batched_beam_path(
    log_probs: &[f32],
    num_frames: usize,
    vocab_size: usize,
    blank_id: usize,
    beam_size: usize,
) -> Result<CtcPath> {
    if beam_size == 0 {
        bail!("CTC beam size must be greater than zero");
    }
    let mut beam = vec![BeamState {
        token_ids: Vec::new(),
        frame_indices: Vec::new(),
        last_label: blank_id,
        score: 0.0,
    }];

    for (frame_index, frame) in log_probs
        .chunks_exact(vocab_size)
        .take(num_frames)
        .enumerate()
    {
        let mut candidates = Vec::with_capacity(beam.len() * vocab_size);
        for state in &beam {
            for (token_id, &token_score) in frame.iter().enumerate().take(vocab_size) {
                let mut token_ids = state.token_ids.clone();
                let mut frames = state.frame_indices.clone();
                if token_id != blank_id && token_id != state.last_label {
                    token_ids.push(token_id);
                    frames.push(frame_index);
                }
                candidates.push(BeamState {
                    token_ids,
                    frame_indices: frames,
                    last_label: token_id,
                    score: state.score + token_score,
                });
            }
        }
        candidates.sort_by(|left, right| total_cmp(right.score, left.score));
        candidates.truncate(beam_size);
        let best_score = candidates
            .first()
            .map_or(f32::NEG_INFINITY, |candidate| candidate.score);
        candidates.retain(|candidate| candidate.score > best_score - NVIDIA_BATCHED_BEAM_THRESHOLD);

        // Pinned NeMo recombines only after top-k and uses max (not
        // log-sum-exp) for CTC-equivalent transcript/last-label states.
        let mut recombined = HashMap::<(Vec<usize>, usize), BeamState>::new();
        for candidate in candidates {
            let key = (candidate.token_ids.clone(), candidate.last_label);
            let entry = recombined.entry(key).or_insert_with(|| candidate.clone());
            if candidate.score > entry.score {
                *entry = candidate;
            }
        }
        beam = recombined.into_values().collect();
    }

    beam.into_iter()
        .max_by(|left, right| total_cmp(left.score, right.score))
        .map(|state| CtcPath {
            token_ids: state.token_ids,
            frame_indices: state.frame_indices,
            score: state.score,
        })
        .ok_or_else(|| anyhow!("CTC beam search produced no hypothesis"))
}

fn total_cmp(left: f32, right: f32) -> Ordering {
    left.total_cmp(&right)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use serde_json::Value;

    use super::{CtcDecodingStrategy, decode_ctc, transcript_from_path};

    fn one_hot_log_probs(ids: &[usize], vocab_size: usize) -> Vec<f32> {
        ids.iter()
            .flat_map(|&id| {
                (0..vocab_size).map(move |candidate| if candidate == id { 0.0 } else { -20.0 })
            })
            .collect()
    }

    #[test]
    fn ctc_blank_resets_repetition_and_preserves_emission_frames() {
        let ids = [3, 2, 2, 3, 2, 1, 1, 3];
        let logits = one_hot_log_probs(&ids, 4);

        for strategy in [
            CtcDecodingStrategy::Greedy,
            CtcDecodingStrategy::BatchedBeam { beam_size: 4 },
        ] {
            let path = decode_ctc(&logits, ids.len(), 4, 3, strategy).unwrap();
            assert_eq!(path.token_ids, vec![2, 2, 1], "strategy={strategy:?}");
            assert_eq!(path.frame_indices, vec![1, 4, 5], "strategy={strategy:?}");
        }
    }

    #[test]
    fn sentencepiece_word_boundary_becomes_a_trimmed_space_with_token_ranges() {
        let path = super::CtcPath {
            token_ids: vec![2, 1],
            frame_indices: vec![1, 4],
            score: 0.0,
        };
        let transcript = transcript_from_path(
            &path,
            &["<unk>".into(), "語".into(), "▁日".into(), "<blk>".into()],
            0.08,
        )
        .unwrap();

        assert_eq!(transcript.text, "日語");
        assert_eq!(transcript.tokens[0].text, " 日");
        assert_eq!(transcript.tokens[0].char_range, Some(0..1));
        assert_eq!(transcript.tokens[0].start_sec, Some(0.08));
        assert_eq!(transcript.tokens[1].start_sec, Some(0.32));
    }

    #[test]
    fn batched_beam_matches_pinned_nvidia_without_fusion_models() {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../diagnostics/nemo-reference/fixtures/nemo-ctc-batched-beam.json");
        let fixture: Value =
            serde_json::from_str(&fs::read_to_string(fixture_path).unwrap()).unwrap();
        let shape = fixture["input"]["shape"].as_array().unwrap();
        let frames = shape[0].as_u64().unwrap() as usize;
        let vocab_size = shape[1].as_u64().unwrap() as usize;
        let blank_id = fixture["reference"]["parameters"]["blank_id"]
            .as_u64()
            .unwrap() as usize;
        let beam_size = fixture["reference"]["parameters"]["beam_size"]
            .as_u64()
            .unwrap() as usize;
        let log_probs = fixture["input"]["log_probs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();

        let path = decode_ctc(
            &log_probs,
            frames,
            vocab_size,
            blank_id,
            CtcDecodingStrategy::BatchedBeam { beam_size },
        )
        .unwrap();
        let expected_tokens = fixture["output"]["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let expected_score = fixture["output"]["score"].as_f64().unwrap() as f32;

        assert_eq!(path.token_ids, expected_tokens);
        assert!(
            (path.score - expected_score).abs() < 1.0e-5,
            "Rust score={} NVIDIA score={expected_score}",
            path.score
        );
    }
}
