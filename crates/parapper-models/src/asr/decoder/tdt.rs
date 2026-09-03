// Score normalization converts bounded hypothesis lengths to the model's f32 score type.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use anyhow::{Result, bail};

pub const NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP: usize = 10;
pub const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

pub trait TdtNetwork {
    type State: Clone;

    fn initial_state(&self) -> Self::State;
    /// Runs one predictor step.
    ///
    /// # Errors
    ///
    /// Returns an error when the model-specific predictor cannot run.
    fn predictor(&mut self, token: usize, state: &Self::State) -> Result<(Vec<f32>, Self::State)>;

    /// Combines one encoder frame with a predictor output.
    ///
    /// # Errors
    ///
    /// Returns an error when the model-specific joiner cannot run.
    fn joiner(&mut self, encoder_frame: &[f32], predictor: &[f32]) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TdtHypothesis<S> {
    pub score: f32,
    pub token_ids: Vec<usize>,
    pub timestamps: Vec<usize>,
    pub durations: Vec<usize>,
    pub state: S,
    pub last_frame: usize,
}

/// Runs NVIDIA-compatible greedy TDT decoding.
///
/// # Errors
///
/// Returns an error for an invalid encoder shape or a predictor/joiner inference failure.
pub fn greedy_tdt<N: TdtNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
) -> Result<TdtHypothesis<N::State>> {
    validate_encoder(encoder, frames, encoder_dim)?;
    let mut hypothesis = TdtHypothesis {
        score: 0.0,
        token_ids: Vec::new(),
        timestamps: Vec::new(),
        durations: Vec::new(),
        state: network.initial_state(),
        last_frame: 0,
    };
    let mut last_token = None;
    let mut time = 0;
    while time < frames {
        let frame = encoder_frame(encoder, frames, encoder_dim, time);
        let mut symbols_added = 0;
        let mut skip;
        loop {
            // Pinned GreedyTDTInfer uses SOS (the blank id) only for the first
            // predictor call. A blank prediction does not commit hidden_prime.
            let label = last_token.unwrap_or(blank_id);
            let (prediction, hidden_prime) = network.predictor(label, &hypothesis.state)?;
            let logits = network.joiner(&frame, &prediction)?;
            let (token_logits, duration_logits) = split_logits(&logits, blank_id + 1)?;
            let (token, token_score) = argmax(token_logits)?;
            let (duration_index, _) = argmax(duration_logits)?;
            skip = TDT_DURATIONS[duration_index];

            if token != blank_id {
                hypothesis.token_ids.push(token);
                hypothesis.timestamps.push(time);
                hypothesis.durations.push(skip);
                hypothesis.score += token_score;
                hypothesis.state = hidden_prime;
                last_token = Some(token);
            }

            symbols_added += 1;
            time += skip;
            if skip != 0 || symbols_added == NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP {
                break;
            }
        }
        if skip == 0 && symbols_added == NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP {
            // Exact pinned NeMo safeguard: after ten duration-0 predictions,
            // advance one encoder frame even if the last symbol was nonblank.
            time += 1;
        }
    }
    hypothesis.last_frame = time;
    Ok(hypothesis)
}

/// Runs NVIDIA-compatible default TDT beam search.
///
/// # Errors
///
/// Returns an error for an invalid encoder shape, an empty beam, or a model inference failure.
///
/// # Panics
///
/// Panics only if the internal non-empty current-hypothesis invariant is violated.
#[allow(
    clippy::too_many_lines,
    reason = "the beam-search loop is kept together to mirror NVIDIA"
)]
pub fn default_beam_tdt<N: TdtNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    beam_size: usize,
) -> Result<Vec<TdtHypothesis<N::State>>> {
    validate_encoder(encoder, frames, encoder_dim)?;
    if beam_size == 0 {
        bail!("TDT beam size must be positive");
    }
    let initial_state = network.initial_state();
    let mut kept = vec![TdtHypothesis {
        score: 0.0,
        token_ids: Vec::new(),
        timestamps: Vec::new(),
        durations: Vec::new(),
        state: initial_state,
        last_frame: 0,
    }];

    for time in 0..frames {
        let frame = encoder_frame(encoder, frames, encoder_dim, time);
        let mut current = kept
            .extract_if(.., |hypothesis| hypothesis.last_frame == time)
            .collect::<Vec<_>>();
        let mut expansions = 0_usize;
        while !current.is_empty() {
            let best_index = current
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.score.total_cmp(&right.score))
                .map(|(index, _)| index)
                .expect("current hypotheses are non-empty");
            let hypothesis = current.swap_remove(best_index);
            let last_token = hypothesis.token_ids.last().copied().unwrap_or(blank_id);
            let (prediction, predicted_state) = network.predictor(last_token, &hypothesis.state)?;
            let logits = network.joiner(&frame, &prediction)?;
            let (token_logits, duration_logits) = split_logits(&logits, blank_id + 1)?;
            let token_log_probs = log_softmax(token_logits);
            let duration_log_probs = log_softmax(duration_logits);
            let token_candidates = top_k(&token_log_probs[..blank_id], beam_size);
            let duration_candidates = top_k(&duration_log_probs, beam_size);

            let mut pairs = token_candidates
                .iter()
                .flat_map(|&(token, token_score)| {
                    duration_candidates
                        .iter()
                        .map(move |&(duration_index, duration_score)| {
                            (token, duration_index, token_score + duration_score)
                        })
                })
                .collect::<Vec<_>>();
            pairs.sort_by(|left, right| right.2.total_cmp(&left.2));
            pairs.truncate(beam_size.min(pairs.len()));
            for (token, duration_index, score) in pairs {
                let duration = TDT_DURATIONS[duration_index];
                let mut next = hypothesis.clone();
                next.score += score;
                next.token_ids.push(token);
                // Pinned default BeamTDTInfer records the end frame, unlike
                // GreedyTDTInfer which records the start frame.
                next.timestamps.push(time + duration);
                next.durations.push(duration);
                next.state = predicted_state.clone();
                next.last_frame += duration;
                if duration == 0 {
                    current.push(next);
                } else {
                    kept.push(next);
                }
            }

            for &(duration_index, duration_score) in &duration_candidates {
                let mut duration = TDT_DURATIONS[duration_index];
                if duration == 0 {
                    if duration_candidates.len() == 1 {
                        duration = 1;
                    } else {
                        continue;
                    }
                }
                let mut next = hypothesis.clone();
                let duration_score = if TDT_DURATIONS[duration_index] == 0 {
                    let replacement = TDT_DURATIONS
                        .iter()
                        .position(|&candidate| candidate == duration)
                        .expect("replacement TDT duration must exist");
                    duration_log_probs[replacement]
                } else {
                    duration_score
                };
                next.score += token_log_probs[blank_id] + duration_score;
                next.last_frame += duration;
                kept.push(next);
            }
            merge_duplicates(&mut kept);

            if let Some(current_best) = current
                .iter()
                .map(|hypothesis| hypothesis.score)
                .max_by(f32::total_cmp)
            {
                let better = kept
                    .iter()
                    .filter(|hypothesis| hypothesis.score > current_best)
                    .count();
                if better >= beam_size {
                    kept.retain(|hypothesis| hypothesis.score > current_best);
                    break;
                }
            } else {
                kept.sort_by(|left, right| right.score.total_cmp(&left.score));
                kept.truncate(beam_size);
            }
            expansions += 1;
            if expansions > 100_000 {
                bail!("TDT beam search exceeded its duration-0 safety budget");
            }
        }
    }
    kept.sort_by(|left, right| {
        normalized_score(right)
            .total_cmp(&normalized_score(left))
            .then_with(|| right.score.total_cmp(&left.score))
    });
    Ok(kept)
}

fn normalized_score<S>(hypothesis: &TdtHypothesis<S>) -> f32 {
    // Pinned BeamTDTInfer normalizes before pack_hypotheses removes SOS.
    hypothesis.score / (hypothesis.token_ids.len() + 1) as f32
}

fn merge_duplicates<S: Clone>(hypotheses: &mut Vec<TdtHypothesis<S>>) {
    let mut merged: HashMap<(Vec<usize>, usize), TdtHypothesis<S>> = HashMap::new();
    hypotheses.sort_by(|left, right| right.score.total_cmp(&left.score));
    for hypothesis in hypotheses.drain(..) {
        let key = (hypothesis.token_ids.clone(), hypothesis.last_frame);
        if let Some(existing) = merged.get_mut(&key) {
            existing.score = log_add_exp(existing.score, hypothesis.score);
        } else {
            merged.insert(key, hypothesis);
        }
    }
    hypotheses.extend(merged.into_values());
}

fn log_add_exp(left: f32, right: f32) -> f32 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn split_logits(logits: &[f32], token_classes: usize) -> Result<(&[f32], &[f32])> {
    if logits.len() != token_classes + TDT_DURATIONS.len() {
        bail!(
            "TDT joiner returned {} logits, expected {} token and {} duration logits",
            logits.len(),
            token_classes,
            TDT_DURATIONS.len()
        );
    }
    Ok(logits.split_at(token_classes))
}

fn argmax(values: &[f32]) -> Result<(usize, f32)> {
    let (&first, rest) = values
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("cannot take argmax of an empty tensor"))?;
    Ok(rest
        .iter()
        .copied()
        .enumerate()
        .fold((0, first), |best, (index, value)| {
            if value > best.1 {
                (index + 1, value)
            } else {
                best
            }
        }))
}

fn log_softmax(values: &[f32]) -> Vec<f32> {
    let maximum = values
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .unwrap_or(f32::NEG_INFINITY);
    let log_sum = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f32>()
        .ln()
        + maximum;
    values.iter().map(|value| value - log_sum).collect()
}

fn top_k(values: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));
    indexed.truncate(count.min(indexed.len()));
    indexed
}

fn validate_encoder(encoder: &[f32], frames: usize, encoder_dim: usize) -> Result<()> {
    if frames == 0 || encoder_dim == 0 || encoder.len() != frames * encoder_dim {
        bail!("invalid TDT encoder shape");
    }
    if encoder.iter().any(|value| !value.is_finite()) {
        bail!("TDT encoder contains a non-finite value");
    }
    Ok(())
}

fn encoder_frame(encoder: &[f32], frames: usize, encoder_dim: usize, time: usize) -> Vec<f32> {
    (0..encoder_dim)
        .map(|feature| encoder[feature * frames + time])
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{
        NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP, TdtHypothesis, TdtNetwork, greedy_tdt, merge_duplicates,
    };

    #[derive(Clone, Default, Debug, PartialEq)]
    struct State(usize);

    struct ScriptedNetwork {
        logits: Vec<Vec<f32>>,
        calls: usize,
    }

    impl TdtNetwork for ScriptedNetwork {
        type State = State;

        fn initial_state(&self) -> Self::State {
            State::default()
        }

        fn predictor(
            &mut self,
            _token: usize,
            state: &Self::State,
        ) -> Result<(Vec<f32>, Self::State)> {
            Ok((vec![0.0], State(state.0 + 1)))
        }

        fn joiner(&mut self, _encoder_frame: &[f32], _predictor: &[f32]) -> Result<Vec<f32>> {
            let value = self.logits[self.calls.min(self.logits.len() - 1)].clone();
            self.calls += 1;
            Ok(value)
        }
    }

    fn logits(token: usize, duration: usize) -> Vec<f32> {
        let mut values = vec![-10.0; 3 + 5];
        values[token] = 5.0;
        values[3 + duration] = 5.0;
        values
    }

    #[test]
    fn greedy_uses_nvidia_ten_symbol_guard_and_commits_only_nonblank_state() {
        let mut network = ScriptedNetwork {
            logits: (0..NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP)
                .map(|index| {
                    if index == 4 {
                        logits(2, 0)
                    } else {
                        logits(0, 0)
                    }
                })
                .collect(),
            calls: 0,
        };
        let decoded = greedy_tdt(&mut network, &[0.0], 1, 1, 2).unwrap();

        assert_eq!(decoded.token_ids, vec![0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(decoded.timestamps, vec![0; 9]);
        assert_eq!(decoded.durations, vec![0; 9]);
        assert_eq!(decoded.state, State(9));
        assert_eq!(decoded.last_frame, 1);
        assert_eq!(network.calls, NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP);
    }

    #[test]
    fn greedy_duration_skips_encoder_frames_and_reports_start_timestamp() {
        let mut network = ScriptedNetwork {
            logits: vec![logits(1, 2), logits(2, 1)],
            calls: 0,
        };
        let decoded = greedy_tdt(&mut network, &[0.0, 0.0, 0.0], 3, 1, 2).unwrap();
        assert_eq!(decoded.token_ids, vec![1]);
        assert_eq!(decoded.timestamps, vec![0]);
        assert_eq!(decoded.durations, vec![2]);
        assert_eq!(decoded.last_frame, 3);
        assert_eq!(decoded.state, State(1));
    }

    #[test]
    fn default_beam_uses_end_timestamps_and_sos_in_score_normalization() {
        let mut network = ScriptedNetwork {
            logits: vec![
                vec![3.0, 1.0, 2.0, -10.0, 3.0, -10.0, -10.0, -10.0],
                vec![3.0, 1.0, 2.0, -10.0, 3.0, -10.0, -10.0, -10.0],
            ],
            calls: 0,
        };
        let decoded = super::default_beam_tdt(&mut network, &[0.0, 0.0], 2, 1, 2, 1).unwrap();

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].token_ids, vec![0, 0]);
        assert_eq!(decoded[0].timestamps, vec![1, 2]);
        assert_eq!(decoded[0].durations, vec![1, 1]);
        assert_eq!(decoded[0].state, State(2));
        assert_eq!(decoded[0].last_frame, 2);
        assert!((super::normalized_score(&decoded[0]) - decoded[0].score / 3.0).abs() < 1.0e-7);
    }

    #[test]
    fn duplicate_merge_keeps_metadata_from_the_highest_score_path() {
        let mut hypotheses = vec![
            TdtHypothesis {
                score: -5.0,
                token_ids: vec![1],
                timestamps: vec![8],
                durations: vec![2],
                state: State(1),
                last_frame: 10,
            },
            TdtHypothesis {
                score: -1.0,
                token_ids: vec![1],
                timestamps: vec![9],
                durations: vec![1],
                state: State(2),
                last_frame: 10,
            },
        ];

        merge_duplicates(&mut hypotheses);

        assert_eq!(hypotheses.len(), 1);
        assert_eq!(hypotheses[0].timestamps, vec![9]);
        assert_eq!(hypotheses[0].durations, vec![1]);
        assert_eq!(hypotheses[0].state, State(2));
        assert!((hypotheses[0].score - super::log_add_exp(-5.0, -1.0)).abs() < 1.0e-7);
    }
}
