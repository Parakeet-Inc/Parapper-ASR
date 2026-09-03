use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::Path,
};

use anyhow::{Context, Result, anyhow, bail};
use unicode_normalization::UnicodeNormalization;

pub(super) const STATIC_EMBEDDING_DIR_NAME: &str = "static-embedding-japanese";
pub(super) const STATIC_EMBEDDING_REQUIRED_FILES: &[&str] = &[
    "0_StaticEmbedding/tokenizer.json",
    "0_StaticEmbedding/model.safetensors",
];

const STATIC_COHERENCE_WEIGHT: f64 = 0.1;
const LENGTH_EXPONENT: f64 = 0.5;

#[derive(Debug, Clone, Copy)]
pub(super) struct EvidenceSeed<'a> {
    pub raw_score: f32,
    pub token_ids: &'a [usize],
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct EvidenceSelection {
    pub token_ids: Vec<usize>,
    pub source_seed: Option<usize>,
}

#[derive(Debug)]
struct EvidenceCandidate {
    token_ids: Vec<usize>,
    base_score: f64,
    source_seed: Option<usize>,
}

type State = (usize, usize, usize);

struct EvidenceMasks {
    visitors: HashMap<State, u64>,
    edges: HashMap<(State, usize), u64>,
    terminals: HashMap<State, u64>,
}

/// Selects the production width-4 / one-splice / retained-width-2 candidate.
///
/// Candidate generation and evidence scoring intentionally mirror the pinned
/// JSUT experiment. The caller supplies static-embedding coherence so the
/// lattice logic can be tested independently from the model artifact.
pub(super) fn select_one_splice_candidate<F>(
    seeds: &[EvidenceSeed<'_>],
    retained_candidates: usize,
    mut coherence: F,
) -> Result<EvidenceSelection>
where
    F: FnMut(&[usize]) -> Result<f64>,
{
    if seeds.is_empty() {
        bail!("ReazonSpeech one-splice reranking requires at least one seed");
    }
    if retained_candidates == 0 {
        bail!("ReazonSpeech one-splice retained width must be positive");
    }
    if seeds.len() > 64 {
        bail!("ReazonSpeech one-splice evidence supports at most 64 seeds");
    }

    let weights = seed_weights(seeds);
    let masks = evidence_masks(seeds);
    let mut candidates = one_splice_sequences(seeds)
        .into_iter()
        .map(|token_ids| {
            let score = evidence_score(&token_ids, &weights, &masks)?;
            let length = f64::from(
                u32::try_from(token_ids.len() + 2)
                    .context("one-splice candidate is too long to normalize")?,
            );
            let source_seed = seeds.iter().position(|seed| seed.token_ids == token_ids);
            Ok(EvidenceCandidate {
                token_ids,
                base_score: score / length.powf(LENGTH_EXPONENT),
                source_seed,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    candidates.sort_by(|left, right| {
        right
            .base_score
            .total_cmp(&left.base_score)
            .then_with(|| left.token_ids.cmp(&right.token_ids))
    });
    candidates.truncate(retained_candidates);

    let coherence_scores = candidates
        .iter()
        .map(|candidate| coherence(&candidate.token_ids))
        .collect::<Result<Vec<_>>>()?;
    let retained_count = f64::from(
        u32::try_from(coherence_scores.len())
            .context("one-splice retained candidate count is too large")?,
    );
    let mean = coherence_scores.iter().sum::<f64>() / retained_count;
    let deviation = (coherence_scores
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / retained_count)
        .sqrt();

    let selected = candidates
        .into_iter()
        .zip(coherence_scores)
        .max_by(|(left, left_coherence), (right, right_coherence)| {
            let standardize = |value: f64| {
                if deviation > 1.0e-12 {
                    (value - mean) / deviation
                } else {
                    0.0
                }
            };
            let left_score =
                left.base_score + STATIC_COHERENCE_WEIGHT * standardize(*left_coherence);
            let right_score =
                right.base_score + STATIC_COHERENCE_WEIGHT * standardize(*right_coherence);
            left_score
                .total_cmp(&right_score)
                .then_with(|| right.token_ids.cmp(&left.token_ids))
        })
        .map(|(candidate, _)| candidate)
        .ok_or_else(|| anyhow!("ReazonSpeech one-splice reranking produced no candidates"))?;
    Ok(EvidenceSelection {
        token_ids: selected.token_ids,
        source_seed: selected.source_seed,
    })
}

fn state_at(token_ids: &[usize], count: usize) -> State {
    match count {
        0 => (0, 0, 0),
        1 => (1, 0, token_ids[0]),
        _ => (count, token_ids[count - 2], token_ids[count - 1]),
    }
}

fn one_splice_sequences(seeds: &[EvidenceSeed<'_>]) -> BTreeSet<Vec<usize>> {
    let mut sequences = seeds
        .iter()
        .map(|seed| seed.token_ids.to_vec())
        .collect::<BTreeSet<_>>();
    for prefix in seeds {
        for suffix in seeds {
            let shared_length = prefix.token_ids.len().min(suffix.token_ids.len());
            for count in 0..=shared_length {
                if state_at(prefix.token_ids, count) != state_at(suffix.token_ids, count) {
                    continue;
                }
                let mut sequence = prefix.token_ids[..count].to_vec();
                sequence.extend_from_slice(&suffix.token_ids[count..]);
                sequences.insert(sequence);
            }
        }
    }
    sequences
}

fn seed_weights(seeds: &[EvidenceSeed<'_>]) -> Vec<f64> {
    let maximum = seeds
        .iter()
        .map(|seed| f64::from(seed.raw_score))
        .fold(f64::NEG_INFINITY, f64::max);
    seeds
        .iter()
        .map(|seed| (f64::from(seed.raw_score) - maximum).exp())
        .collect()
}

fn evidence_masks(seeds: &[EvidenceSeed<'_>]) -> EvidenceMasks {
    let mut visitors = HashMap::new();
    let mut edges = HashMap::new();
    let mut terminals = HashMap::new();
    for (seed_index, seed) in seeds.iter().enumerate() {
        let bit = 1_u64 << seed_index;
        let mut state = state_at(seed.token_ids, 0);
        *visitors.entry(state).or_default() |= bit;
        for (index, &token_id) in seed.token_ids.iter().enumerate() {
            *edges.entry((state, token_id)).or_default() |= bit;
            state = state_at(seed.token_ids, index + 1);
            *visitors.entry(state).or_default() |= bit;
        }
        *terminals.entry(state).or_default() |= bit;
    }
    EvidenceMasks {
        visitors,
        edges,
        terminals,
    }
}

fn evidence_score(token_ids: &[usize], weights: &[f64], masks: &EvidenceMasks) -> Result<f64> {
    let mut state = state_at(token_ids, 0);
    let mut score = 0.0;
    for (index, &token_id) in token_ids.iter().enumerate() {
        let denominator = masks
            .visitors
            .get(&state)
            .copied()
            .ok_or_else(|| anyhow!("one-splice candidate reached an unsupported state"))?;
        let numerator = masks
            .edges
            .get(&(state, token_id))
            .copied()
            .ok_or_else(|| anyhow!("one-splice candidate used an unsupported edge"))?;
        score += (masked_mass(numerator, weights) / masked_mass(denominator, weights)).ln();
        state = state_at(token_ids, index + 1);
    }
    let denominator = masks
        .visitors
        .get(&state)
        .copied()
        .ok_or_else(|| anyhow!("one-splice candidate ended at an unsupported state"))?;
    let numerator = masks
        .terminals
        .get(&state)
        .copied()
        .ok_or_else(|| anyhow!("one-splice candidate used an unsupported terminal"))?;
    score += (masked_mass(numerator, weights) / masked_mass(denominator, weights)).ln();
    Ok(score)
}

fn masked_mass(mask: u64, weights: &[f64]) -> f64 {
    weights
        .iter()
        .enumerate()
        .filter_map(|(index, weight)| ((mask & (1_u64 << index)) != 0).then_some(weight))
        .sum()
}

#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<char, usize>,
    terminals: Vec<(usize, f64)>,
}

#[derive(Debug)]
struct UnigramTokenizer {
    unknown_id: usize,
    unknown_score: f64,
    nodes: Vec<TrieNode>,
}

#[derive(Debug)]
pub(super) struct StaticEmbeddingModel {
    tokenizer: UnigramTokenizer,
    weights: Vec<u8>,
    rows: usize,
    dimensions: usize,
    data_offset: usize,
    cache: HashMap<String, f64>,
}

impl StaticEmbeddingModel {
    pub(super) fn load(snapshot: &Path) -> Result<Self> {
        let module = snapshot.join("0_StaticEmbedding");
        let tokenizer = load_tokenizer(&module.join("tokenizer.json"))?;
        let weights_path = module.join("model.safetensors");
        let weights = fs::read(&weights_path).with_context(|| {
            format!(
                "failed to read ReazonSpeech static embedding weights: {}",
                weights_path.display()
            )
        })?;
        let (rows, dimensions, data_offset, data_length) = parse_embedding_tensor(&weights)?;
        if rows <= tokenizer.unknown_id {
            bail!("static embedding vocabulary is smaller than its unknown token id");
        }
        if data_offset
            .checked_add(data_length)
            .is_none_or(|end| end > weights.len())
        {
            bail!("static embedding tensor extends beyond model.safetensors");
        }
        Ok(Self {
            tokenizer,
            weights,
            rows,
            dimensions,
            data_offset,
            cache: HashMap::new(),
        })
    }

    pub(super) fn piece_mean(&mut self, text: &str) -> Result<f64> {
        if let Some(score) = self.cache.get(text) {
            return Ok(*score);
        }
        let pieces = self.tokenizer.tokenize(text)?;
        if pieces.iter().any(|&(token_id, _)| token_id >= self.rows) {
            bail!("static tokenizer returned an embedding row outside the model");
        }
        let mut sentence = vec![0.0_f32; self.dimensions];
        for &(token_id, _) in &pieces {
            for (dimension, value) in sentence.iter_mut().enumerate() {
                *value += self.weight(token_id, dimension);
            }
        }
        let piece_count = f32::from(
            u16::try_from(pieces.len()).context("static tokenized sentence is too long")?,
        );
        for value in &mut sentence {
            *value /= piece_count;
        }
        let sentence_norm = sentence
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if sentence_norm > 0.0 {
            for value in &mut sentence {
                *value /= sentence_norm;
            }
        }

        let mut sum = 0.0_f32;
        let mut content_pieces = 0_usize;
        for &(token_id, covered) in &pieces {
            if covered == 0 {
                continue;
            }
            let mut dot = 0.0_f32;
            let mut norm_squared = 0.0_f32;
            for (dimension, &sentence_value) in sentence.iter().enumerate() {
                let value = self.weight(token_id, dimension);
                dot += value * sentence_value;
                norm_squared += value * value;
            }
            let norm = norm_squared.sqrt();
            if norm > 0.0 {
                sum += dot / norm;
            }
            content_pieces += 1;
        }
        let score = if content_pieces == 0 {
            0.0
        } else {
            f64::from(
                sum / f32::from(
                    u16::try_from(content_pieces)
                        .context("static content piece count is too large")?,
                ),
            )
        };
        self.cache.insert(text.to_owned(), score);
        Ok(score)
    }

    pub(super) fn sentence_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let pieces = self.tokenizer.tokenize(text)?;
        if pieces.iter().any(|&(token_id, _)| token_id >= self.rows) {
            bail!("static tokenizer returned an embedding row outside the model");
        }
        let mut sentence = vec![0.0_f32; self.dimensions];
        for &(token_id, _) in &pieces {
            for (dimension, value) in sentence.iter_mut().enumerate() {
                *value += self.weight(token_id, dimension);
            }
        }
        let piece_count = f32::from(
            u16::try_from(pieces.len()).context("static tokenized sentence is too long")?,
        );
        for value in &mut sentence {
            *value /= piece_count;
        }
        let norm = sentence
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for value in &mut sentence {
                *value /= norm;
            }
        }
        Ok(sentence)
    }

    fn weight(&self, row: usize, dimension: usize) -> f32 {
        let index = self.data_offset + (row * self.dimensions + dimension) * size_of::<f32>();
        f32::from_le_bytes(
            self.weights[index..index + size_of::<f32>()]
                .try_into()
                .unwrap(),
        )
    }
}

pub(super) fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f64> {
    if left.is_empty()
        || left.len() != right.len()
        || left.iter().chain(right).any(|value| !value.is_finite())
    {
        bail!("static sentence embeddings must be finite, non-empty, and equally sized");
    }
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(0.0);
    }
    Ok(f64::from(dot / (left_norm * right_norm)))
}

fn load_tokenizer(path: &Path) -> Result<UnigramTokenizer> {
    let document: serde_json::Value = serde_json::from_slice(
        &fs::read(path)
            .with_context(|| format!("failed to read static tokenizer: {}", path.display()))?,
    )
    .context("failed to parse static tokenizer.json")?;
    let model = document
        .get("model")
        .ok_or_else(|| anyhow!("static tokenizer has no model"))?;
    if model.get("type").and_then(serde_json::Value::as_str) != Some("Unigram")
        || model
            .get("byte_fallback")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    {
        bail!("static tokenizer must be a non-byte-fallback Unigram model");
    }
    let unknown_id = model
        .get("unk_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("static tokenizer has no valid unk_id"))?;
    let vocabulary = model
        .get("vocab")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("static tokenizer has no vocabulary"))?;
    let mut nodes = vec![TrieNode::default()];
    let mut minimum_score = f64::INFINITY;
    for (token_id, entry) in vocabulary.iter().enumerate() {
        let pair = entry
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| anyhow!("static tokenizer vocabulary entry is invalid"))?;
        let piece = pair[0]
            .as_str()
            .ok_or_else(|| anyhow!("static tokenizer piece is not text"))?;
        let score = pair[1]
            .as_f64()
            .ok_or_else(|| anyhow!("static tokenizer score is not numeric"))?;
        minimum_score = minimum_score.min(score);
        if token_id <= unknown_id || piece.is_empty() {
            continue;
        }
        let mut node_index = 0;
        for character in piece.chars() {
            let next = if let Some(&existing) = nodes[node_index].children.get(&character) {
                existing
            } else {
                nodes.push(TrieNode::default());
                let created = nodes.len() - 1;
                nodes[node_index].children.insert(character, created);
                created
            };
            node_index = next;
        }
        nodes[node_index].terminals.push((token_id, score));
    }
    Ok(UnigramTokenizer {
        unknown_id,
        unknown_score: minimum_score - 10.0,
        nodes,
    })
}

impl UnigramTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        let mut normalized = vec!['▁'];
        for character in text.nfkc().flat_map(char::to_lowercase) {
            normalized.push(if character.is_whitespace() {
                '▁'
            } else {
                character
            });
        }
        let mut best = vec![f64::NEG_INFINITY; normalized.len() + 1];
        let mut backpointers = vec![None; normalized.len() + 1];
        best[0] = 0.0;
        for start in 0..normalized.len() {
            if !best[start].is_finite() {
                continue;
            }
            let mut node_index = 0;
            let mut matched = false;
            for end in start..normalized.len() {
                let Some(&next) = self.nodes[node_index].children.get(&normalized[end]) else {
                    break;
                };
                node_index = next;
                for &(token_id, token_score) in &self.nodes[node_index].terminals {
                    matched = true;
                    update_unigram_path(
                        &normalized,
                        start,
                        end + 1,
                        token_id,
                        best[start] + token_score,
                        &mut best,
                        &mut backpointers,
                    );
                }
            }
            if !matched {
                update_unigram_path(
                    &normalized,
                    start,
                    start + 1,
                    self.unknown_id,
                    best[start] + self.unknown_score,
                    &mut best,
                    &mut backpointers,
                );
            }
        }
        let mut pieces = Vec::new();
        let mut position = normalized.len();
        while position > 0 {
            let (previous, token_id, covered) = backpointers[position]
                .ok_or_else(|| anyhow!("static Unigram tokenizer lost position {position}"))?;
            pieces.push((token_id, covered));
            position = previous;
        }
        pieces.reverse();
        Ok(pieces)
    }
}

fn update_unigram_path(
    normalized: &[char],
    start: usize,
    end: usize,
    token_id: usize,
    score: f64,
    best: &mut [f64],
    backpointers: &mut [Option<(usize, usize, usize)>],
) {
    if score <= best[end] {
        return;
    }
    best[end] = score;
    let covered = normalized[start..end]
        .iter()
        .filter(|&&character| character != '▁')
        .count();
    backpointers[end] = Some((start, token_id, covered));
}

fn parse_embedding_tensor(bytes: &[u8]) -> Result<(usize, usize, usize, usize)> {
    let header_length = bytes
        .get(..8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("static embedding safetensors header is invalid"))?;
    let header_end = 8_usize
        .checked_add(header_length)
        .ok_or_else(|| anyhow!("static embedding safetensors header is too large"))?;
    let header: serde_json::Value = serde_json::from_slice(
        bytes
            .get(8..header_end)
            .ok_or_else(|| anyhow!("static embedding safetensors header is truncated"))?,
    )
    .context("failed to parse static embedding safetensors header")?;
    let tensor = header
        .get("embedding.weight")
        .ok_or_else(|| anyhow!("static embedding has no embedding.weight tensor"))?;
    if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("F32") {
        bail!("static embedding.weight must use F32");
    }
    let shape = tensor
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .filter(|shape| shape.len() == 2)
        .ok_or_else(|| anyhow!("static embedding.weight must be a matrix"))?;
    let rows = shape[0]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("static embedding row count is invalid"))?;
    let dimensions = shape[1]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("static embedding dimension is invalid"))?;
    let offsets = tensor
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(|| anyhow!("static embedding data offsets are invalid"))?;
    let start = offsets[0]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("static embedding data start is invalid"))?;
    let end = offsets[1]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("static embedding data end is invalid"))?;
    let expected = rows
        .checked_mul(dimensions)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .ok_or_else(|| anyhow!("static embedding matrix is too large"))?;
    if end.checked_sub(start) != Some(expected) {
        bail!("static embedding tensor byte count does not match its shape");
    }
    let data_offset = header_end
        .checked_add(start)
        .ok_or_else(|| anyhow!("static embedding data offset is too large"))?;
    Ok((rows, dimensions, data_offset, expected))
}

#[cfg(test)]
mod tests {
    use super::{
        EvidenceSeed, StaticEmbeddingModel, cosine_similarity, select_one_splice_candidate,
    };

    #[test]
    fn cosine_similarity_compares_sentence_vectors_without_length_bias() {
        assert!((cosine_similarity(&[2.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1.0e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap()).abs() < 1.0e-6);
        assert!(cosine_similarity(&[1.0], &[1.0, 0.0]).is_err());
    }

    #[test]
    fn one_splice_candidates_can_recombine_prefix_and_suffix_once() {
        let seeds = [
            EvidenceSeed {
                raw_score: 0.0,
                token_ids: &[1, 2, 3, 4],
            },
            EvidenceSeed {
                raw_score: -1.0,
                token_ids: &[5, 2, 3, 6],
            },
        ];
        let candidates = super::one_splice_sequences(&seeds);

        assert_eq!(
            candidates,
            [[1, 2, 3, 4], [1, 2, 3, 6], [5, 2, 3, 4], [5, 2, 3, 6]]
                .into_iter()
                .map(Vec::from)
                .collect()
        );
    }

    #[test]
    fn retained_width_two_uses_standardized_static_coherence_for_final_choice() {
        let seeds = [
            EvidenceSeed {
                raw_score: 0.0,
                token_ids: &[1, 2, 3, 4],
            },
            EvidenceSeed {
                raw_score: 0.0,
                token_ids: &[5, 2, 3, 6],
            },
        ];
        let selection =
            select_one_splice_candidate(&seeds, 2, |tokens| Ok(f64::from(tokens == [1, 2, 3, 6])))
                .expect("one-splice selection should succeed");

        assert_eq!(selection.token_ids, [1, 2, 3, 6]);
        assert_eq!(selection.source_seed, None);
    }

    #[test]
    #[ignore = "requires the pinned hotchpotch/static-embedding-japanese snapshot"]
    fn pinned_static_embedding_matches_the_python_tuning_implementation() {
        let snapshot = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/hf-cache/models--hotchpotch--static-embedding-japanese/snapshots/95b3d9c80a7ccf604e2b5daee7b1b3eed6b1a9d3");
        let mut model = StaticEmbeddingModel::load(&snapshot)
            .expect("pinned static embedding snapshot should load");

        let score = model
            .piece_mean("今日は良い天気です")
            .expect("Japanese text should tokenize and score");

        assert!((score - 0.354_502_010_345_459).abs() < 1.0e-6, "{score}");
    }
}
