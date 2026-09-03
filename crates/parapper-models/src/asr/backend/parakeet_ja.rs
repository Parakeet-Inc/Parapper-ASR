// Pinned tensor dimensions fit i64, and frame timestamps are represented as f32 by the ASR API.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

//! Japanese Parakeet hybrid backend over Parapper's pinned ONNX release.
//!
//! Fast mode runs the extracted CTC head after the same shared encoder used by
//! accuracy mode. Accuracy mode adds the fused TDT `decoder_joint` graph; its
//! CTC head is used for optional hotword admission. There is deliberately no
//! fallback to the legacy monolithic sherpa-layout CTC artifact.
//!
//! Expected model directory layout (from
//! `nadare/parakeet-tdt_ctc-0.6b-ja-onnx-dynamic-int8@ab9073e4`):
//! `encoder-model.int8.onnx`, `decoder_joint-model.onnx`,
//! `ctc-head-model.onnx`, their external tensor data, and `vocab.txt`.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    AsrEngine, AsrTranscript,
    asr::{
        decoder::{
            ctc::{CtcDecodingStrategy, decode_ctc, transcript_from_path},
            hotword::{HotwordContextGraph, HotwordTokenPath},
            tdt::{NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP, TDT_DURATIONS},
        },
        frontend::NemoMelFrontend,
    },
    init_onnx_runtime,
};
use anyhow::{Context, Result, anyhow, bail};
use ort::{
    session::Session,
    value::{Tensor, ValueType},
};

use super::JapaneseStaticEmbeddingModel;

// Preference order: quantized artifact names first (q4 MatMulNBits, then int8),
// then the fp32 export name, so one engine can load int8 releases, q4
// experiments, and self-exported fp32 baselines without per-variant renames.
const ENCODER_FILES: &[&str] = &[
    "encoder-model.q4.onnx",
    "encoder-model.q8.onnx",
    "encoder-model.int8.onnx",
    "encoder-model.onnx",
];
const DECODER_JOINT_FILES: &[&str] = &["decoder_joint-model.int8.onnx", "decoder_joint-model.onnx"];
const CTC_MODEL_FILES: &[&str] = &[
    "model.q4.onnx",
    "model.q8.onnx",
    "model.int8.onnx",
    "model.onnx",
];
const CTC_HEAD_FILES: &[&str] = &["ctc-head-model.onnx"];
const VOCAB_FILE: &str = "vocab.txt";
const MEL_BINS: usize = 80;
const ENCODER_DIM: usize = 1_024;
const PREDICTOR_DIM: usize = 640;
const PREDICTOR_LAYERS: usize = 2;
const VOCAB_SIZE: usize = 3_073;
const BLANK_ID: usize = 3_072;
const JOINT_WIDTH: usize = VOCAB_SIZE + TDT_DURATIONS.len();
const FEATURE_FRAME_SHIFT_SEC: f32 = 0.01;
const SUBSAMPLING_FACTOR: f32 = 8.0;

/// Required base artifacts for the Japanese Parakeet accuracy-mode export.
pub const HYBRID_REQUIRED_FILES: &[&str] = &[
    "encoder-model.int8.onnx",
    "encoder-model.int8.onnx.data",
    "decoder_joint-model.onnx",
    "vocab.txt",
];

/// Additional artifact used only when a CTC-gated hotword search is enabled.
pub const HYBRID_CTC_GATE_REQUIRED_FILES: &[&str] =
    &["ctc-head-model.onnx", "ctc-head-model.onnx_data"];

/// Product artifacts required by the fast shared-encoder CTC path.
///
/// This replaces the legacy monolithic sherpa-layout `model.int8.onnx` graph.
pub const SHARED_CTC_REQUIRED_FILES: &[&str] = &[
    "encoder-model.int8.onnx",
    "encoder-model.int8.onnx.data",
    "ctc-head-model.onnx",
    "ctc-head-model.onnx_data",
    "vocab.txt",
];

/// Decoder branch selected after the shared Japanese Parakeet encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParakeetJaDecodingStrategy {
    CtcGreedy,
    TdtGreedy,
    TdtVariableDag(FusedTdtDagConfig),
    TdtVariableDagStaticEmbedding(FusedTdtDagConfig),
}

/// One trimmed FP32 output from the shared Japanese Parakeet encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetJaEncodedAudio {
    values: Vec<f32>,
    frames: usize,
}

/// One time-major CTC posterior matrix produced from the shared encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetJaCtcOutput {
    values: Vec<f32>,
    frames: usize,
}

/// One completed TDT path retained for inexpensive second-pass reranking.
#[derive(Debug, Clone)]
pub struct ParakeetJaTdtCandidate {
    pub transcript: AsrTranscript,
    pub acoustic_score: f32,
}

/// Selects a TDT candidate by acoustic score plus standardized similarity to
/// a CTC transcript anchor.
///
/// # Errors
///
/// Returns an error for empty/mismatched candidates or invalid scores.
pub fn select_ctc_anchor_candidate(
    acoustic_scores: &[f32],
    similarities: &[f64],
    similarity_weight: f32,
) -> Result<usize> {
    if acoustic_scores.is_empty() || acoustic_scores.len() != similarities.len() {
        bail!("CTC-anchor reranking requires matching non-empty candidate scores");
    }
    if !similarity_weight.is_finite() || similarity_weight < 0.0 {
        bail!("CTC-anchor similarity weight must be finite and non-negative");
    }
    if acoustic_scores.iter().any(|value| !value.is_finite())
        || similarities.iter().any(|value| !value.is_finite())
    {
        bail!("CTC-anchor candidate scores must be finite");
    }
    let count = similarities.len() as f64;
    let mean = similarities.iter().sum::<f64>() / count;
    let deviation = (similarities
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    let standardized = |value: f64| {
        if deviation > 1.0e-12 {
            (value - mean) / deviation
        } else {
            0.0
        }
    };
    acoustic_scores
        .iter()
        .zip(similarities)
        .enumerate()
        .max_by(
            |(left_index, (left_score, left_similarity)),
             (right_index, (right_score, right_similarity))| {
                let combined = |score: f32, similarity: f64| {
                    f64::from(score) + f64::from(similarity_weight) * standardized(similarity)
                };
                combined(**left_score, **left_similarity)
                    .total_cmp(&combined(**right_score, **right_similarity))
                    .then_with(|| right_index.cmp(left_index))
            },
        )
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow!("CTC-anchor reranking produced no candidate"))
}

impl ParakeetJaCtcOutput {
    /// Scores a token path by its best local monotonic CTC alignment.
    ///
    /// Each selected token frame contributes its log-probability margin from
    /// that frame's acoustic best token. Consecutive keyword tokens must occur
    /// within `max_gap_frames`; repeated token ids additionally require an
    /// intervening CTC blank. The returned token-mean score is at most zero.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/invalid token path or invalid gap.
    pub fn local_keyword_score(&self, token_ids: &[usize], max_gap_frames: usize) -> Result<f32> {
        ctc_local_keyword_score(
            &self.values,
            self.frames,
            VOCAB_SIZE,
            BLANK_ID,
            token_ids,
            max_gap_frames,
        )
    }
}

/// Keeps complete hotword entries whose best registered token path has enough
/// local monotonic CTC evidence.
///
/// # Errors
///
/// Returns an error for an invalid threshold, duplicate entry surfaces, or an
/// invalid token path.
pub fn ctc_gate_hotword_paths(
    output: &ParakeetJaCtcOutput,
    paths: &[HotwordTokenPath],
    threshold: f32,
    max_gap_frames: usize,
) -> Result<Vec<HotwordTokenPath>> {
    if !threshold.is_finite() || threshold > 0.0 {
        bail!("CTC hotword gate threshold must be finite and non-positive");
    }
    let mut entry_scores = std::collections::HashMap::<usize, f32>::new();
    let mut entry_surfaces = std::collections::HashMap::<usize, &str>::new();
    for path in paths {
        if let Some(previous) = entry_surfaces.insert(path.entry_id, &path.surface)
            && previous != path.surface
        {
            bail!("CTC hotword gate entry maps to multiple surfaces");
        }
        let score = output.local_keyword_score(&path.tokens, max_gap_frames)?;
        entry_scores
            .entry(path.entry_id)
            .and_modify(|current| *current = current.max(score))
            .or_insert(score);
    }
    Ok(paths
        .iter()
        .filter(|path| {
            entry_scores
                .get(&path.entry_id)
                .is_some_and(|&score| score >= threshold)
        })
        .cloned()
        .collect())
}

/// Scores a keyword using a bounded monotonic alignment over CTC frames.
///
/// # Errors
///
/// Returns an error when the posterior shape, keyword, or gap is invalid.
pub fn ctc_local_keyword_score(
    log_probs: &[f32],
    frames: usize,
    vocab_size: usize,
    blank_id: usize,
    token_ids: &[usize],
    max_gap_frames: usize,
) -> Result<f32> {
    if frames == 0
        || vocab_size == 0
        || blank_id >= vocab_size
        || log_probs.len() != frames * vocab_size
    {
        bail!("invalid CTC keyword posterior shape");
    }
    if token_ids.is_empty()
        || token_ids
            .iter()
            .any(|&token| token >= vocab_size || token == blank_id)
    {
        bail!("CTC keyword tokens must be non-empty, in-vocabulary, and non-blank");
    }
    if max_gap_frames == 0 {
        bail!("CTC keyword max frame gap must be positive");
    }
    if log_probs.iter().any(|value| !value.is_finite()) {
        bail!("CTC keyword posterior contains a non-finite value");
    }

    let frame_best = log_probs
        .chunks_exact(vocab_size)
        .map(|frame| frame.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect::<Vec<_>>();
    let margin =
        |frame: usize, token: usize| log_probs[frame * vocab_size + token] - frame_best[frame];

    let mut previous = (0..frames)
        .map(|frame| margin(frame, token_ids[0]))
        .collect::<Vec<_>>();
    for pair in token_ids.windows(2) {
        let repeated = pair[0] == pair[1];
        let token = pair[1];
        let mut current = vec![f32::NEG_INFINITY; frames];
        for (frame, score) in current.iter_mut().enumerate().take(frames).skip(1) {
            let latest_previous = if repeated {
                frame.checked_sub(2)
            } else {
                frame.checked_sub(1)
            };
            let Some(latest_previous) = latest_previous else {
                continue;
            };
            let earliest_previous = frame.saturating_sub(max_gap_frames);
            if earliest_previous > latest_previous {
                continue;
            }
            let best_previous = previous[earliest_previous..=latest_previous]
                .iter()
                .copied()
                .max_by(f32::total_cmp)
                .unwrap_or(f32::NEG_INFINITY);
            let blank_score = if repeated {
                margin(frame - 1, blank_id)
            } else {
                0.0
            };
            *score = best_previous + blank_score + margin(frame, token);
        }
        previous = current;
    }
    let best = previous
        .into_iter()
        .max_by(f32::total_cmp)
        .unwrap_or(f32::NEG_INFINITY);
    Ok(best / token_ids.len() as f32)
}

/// One fused prediction-network/joint step of an onnx-asr TDT export.
///
/// The fused graph consumes one encoder frame, the previous label, and the
/// predictor LSTM state in a single call, returning the joint logits together
/// with the state after consuming that label. This shape cannot implement the
/// pinned `TdtNetwork` trait (its `predictor` must return the successor state
/// before the joiner runs), so greedy decoding is mirrored here instead.
pub trait FusedTdtStep {
    type State: Clone;

    fn initial_state(&self) -> Self::State;

    /// Runs one fused step.
    ///
    /// # Errors
    ///
    /// Returns an error when the model-specific fused graph cannot run.
    fn step(
        &mut self,
        encoder_frame: &[f32],
        token: usize,
        state: &Self::State,
    ) -> Result<(Vec<f32>, Self::State)>;

    /// Runs one sparse frontier as a batch.
    ///
    /// The default keeps scripted/reference networks small; the ORT adapter
    /// overrides it with one real batched session invocation.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched batch cardinalities or a scalar step
    /// failure.
    fn step_batch(
        &mut self,
        encoder_frames: &[Vec<f32>],
        tokens: &[usize],
        states: &[&Self::State],
    ) -> Result<Vec<(Vec<f32>, Self::State)>> {
        if encoder_frames.len() != tokens.len() || tokens.len() != states.len() {
            bail!("fused TDT batch inputs have mismatched lengths");
        }
        encoder_frames
            .iter()
            .zip(tokens)
            .zip(states)
            .map(|((frame, &token), state)| self.step(frame, token, state))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusedTdtHypothesis<S> {
    pub score: f32,
    pub token_ids: Vec<usize>,
    pub timestamps: Vec<usize>,
    pub durations: Vec<usize>,
    pub state: S,
    pub last_frame: usize,
}

/// Runs NVIDIA-compatible greedy TDT decoding over a fused decoder/joint graph.
///
/// This mirrors `parapper_models::asr::decoder::tdt::greedy_tdt` exactly: the
/// blank id acts as SOS for the first step only, the predictor state commits
/// only on a non-blank emission, and the pinned `NeMo` ten-symbol safeguard
/// advances one frame after ten duration-0 predictions.
///
/// # Errors
///
/// Returns an error for an invalid encoder shape or a fused inference failure.
pub fn greedy_fused_tdt<N: FusedTdtStep>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
) -> Result<FusedTdtHypothesis<N::State>> {
    if frames == 0 || encoder_dim == 0 || encoder.len() != frames * encoder_dim {
        bail!("invalid fused TDT encoder shape");
    }
    if encoder.iter().any(|value| !value.is_finite()) {
        bail!("fused TDT encoder contains a non-finite value");
    }
    let mut hypothesis = FusedTdtHypothesis {
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
        let frame = (0..encoder_dim)
            .map(|feature| encoder[feature * frames + time])
            .collect::<Vec<_>>();
        let mut symbols_added = 0;
        let mut skip;
        loop {
            let label = last_token.unwrap_or(blank_id);
            let (logits, hidden_prime) = network.step(&frame, label, &hypothesis.state)?;
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
            time += 1;
        }
    }
    hypothesis.last_frame = time;
    Ok(hypothesis)
}

/// CPU-oriented sparse frame-bucket TDT beam configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedTdtDagConfig {
    /// Maximum number of merged hypotheses retained in one encoder-frame bucket.
    pub beam_size: usize,
    /// Safety guard for consecutive duration-0 emissions at one frame.
    pub max_symbols_per_step: usize,
    /// Whether every duration is expanded or only the most likely duration is retained.
    pub duration_expansion: FusedTdtDurationExpansion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedTdtDurationExpansion {
    All,
    Argmax,
}

/// Experimental ordering for frame-bucket pruning and path merging.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FusedTdtDagMergeOrder {
    /// Retain the acoustic top-k first, then merge duplicates without refilling.
    #[default]
    PruneThenMerge,
    /// Merge every duplicate in the bucket first, then retain the top-k paths.
    MergeThenPrune,
}

/// Diagnostic switches for validating the TDT DAG merge invariant.
///
/// The default exactly preserves the production search behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FusedTdtDagMergeAblation {
    pub order: FusedTdtDagMergeOrder,
    pub include_symbols_since_advance_in_key: bool,
}

/// Controls how hotword token arcs enter each TDT hypothesis expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedTdtHotwordCandidatePolicy {
    /// Existing diagnostic behavior: retain acoustic top-k and append every
    /// reachable hotword continuation before the frame bucket is pruned.
    InjectedAll,
    /// NeMo-style fixed token width: form a wider acoustic shortlist, add only
    /// direct trie children, apply context scores, then retain `beam_size`
    /// tokens including blank before duration expansion.
    DirectPreTopK { acoustic_top_k: usize },
}

/// Search-management counters independent of encoder inference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FusedTdtDagStats {
    pub network_batches: usize,
    pub network_hypotheses: usize,
    pub generated_candidates: usize,
    pub max_active_width: usize,
    pub merge_calls: usize,
    pub merge_input_nodes: usize,
    pub duplicate_nodes_merged: usize,
    pub nodes_pruned: usize,
    pub underfilled_merge_calls: usize,
    pub symbol_budget_conflicts: usize,
}

/// N-best output and sparse-frontier counters from one DAG search.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedTdtDagResult<S> {
    pub hypotheses: Vec<FusedTdtHypothesis<S>>,
    pub stats: FusedTdtDagStats,
}

#[derive(Clone)]
struct FusedTdtDagNode<S> {
    score: f32,
    sequence_hash: u64,
    emitted_count: usize,
    last_token: Option<usize>,
    state: Arc<S>,
    frame: usize,
    history: Option<usize>,
    symbols_since_advance: usize,
    hotword_state: usize,
}

struct FusedTdtHistoryNode {
    parent: Option<usize>,
    token: usize,
    timestamp: usize,
    duration: usize,
}

/// Runs a variable-width frame-bucket DAG over the configured TDT durations.
///
/// `beam_size` is a per-bucket upper bound, not a fixed tensor width. Duplicate
/// `(token-sequence hash, frame)` nodes are log-sum-exp merged and a bucket is
/// never refilled after merging. `All` branches every retained non-blank token
/// over all model durations and blank over every positive duration. `Argmax`
/// retains one duration per token candidate; blank uses the best positive
/// duration because blank with duration zero cannot advance the search.
///
/// # Errors
///
/// Returns an error for an invalid encoder/configuration or a fused network
/// failure.
#[allow(
    clippy::too_many_lines,
    reason = "candidate generation, same-frame waves, and future buckets form one search invariant"
)]
pub fn variable_width_dag_fused_tdt<N: FusedTdtStep>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    config: FusedTdtDagConfig,
) -> Result<FusedTdtDagResult<N::State>> {
    variable_width_dag_fused_tdt_inner(
        network,
        encoder,
        frames,
        encoder_dim,
        blank_id,
        config,
        None,
        FusedTdtHotwordCandidatePolicy::InjectedAll,
        FusedTdtDagMergeAblation::default(),
    )
}

/// Runs the variable-width TDT DAG with diagnostic merge semantics.
///
/// This entry point exists for accuracy and cost ablations; ordinary product
/// decoding continues to use [`FusedTdtDagMergeAblation::default`].
///
/// # Errors
///
/// Returns an error for an invalid encoder/configuration or a fused network
/// failure.
pub fn variable_width_dag_fused_tdt_with_merge_ablation<N: FusedTdtStep>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    config: FusedTdtDagConfig,
    merge_ablation: FusedTdtDagMergeAblation,
) -> Result<FusedTdtDagResult<N::State>> {
    variable_width_dag_fused_tdt_inner(
        network,
        encoder,
        frames,
        encoder_dim,
        blank_id,
        config,
        None,
        FusedTdtHotwordCandidatePolicy::InjectedAll,
        merge_ablation,
    )
}

/// Runs the variable-width TDT DAG while scoring and retaining configured
/// hotword continuations that fall below the normal acoustic token top-k.
///
/// # Errors
///
/// Returns an error for an invalid encoder/configuration or a fused network
/// failure.
pub fn variable_width_dag_fused_tdt_with_hotwords<N: FusedTdtStep>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    config: FusedTdtDagConfig,
    hotwords: &HotwordContextGraph,
) -> Result<FusedTdtDagResult<N::State>> {
    variable_width_dag_fused_tdt_inner(
        network,
        encoder,
        frames,
        encoder_dim,
        blank_id,
        config,
        Some(hotwords),
        FusedTdtHotwordCandidatePolicy::InjectedAll,
        FusedTdtDagMergeAblation::default(),
    )
}

/// Runs fixed-width pre-Top-K hotword candidate admission.
///
/// # Errors
///
/// Returns an error for invalid widths, encoder/configuration mismatches, or
/// a fused network failure.
#[allow(
    clippy::too_many_arguments,
    reason = "the diagnostic decoder mirrors the existing tensor-shape API and adds one search policy"
)]
pub fn variable_width_dag_fused_tdt_with_hotword_policy<N: FusedTdtStep>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    config: FusedTdtDagConfig,
    hotwords: &HotwordContextGraph,
    policy: FusedTdtHotwordCandidatePolicy,
) -> Result<FusedTdtDagResult<N::State>> {
    variable_width_dag_fused_tdt_inner(
        network,
        encoder,
        frames,
        encoder_dim,
        blank_id,
        config,
        Some(hotwords),
        policy,
        FusedTdtDagMergeAblation::default(),
    )
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "candidate generation, same-frame waves, and future buckets form one search invariant"
)]
fn variable_width_dag_fused_tdt_inner<N: FusedTdtStep>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    config: FusedTdtDagConfig,
    hotwords: Option<&HotwordContextGraph>,
    hotword_candidate_policy: FusedTdtHotwordCandidatePolicy,
    merge_ablation: FusedTdtDagMergeAblation,
) -> Result<FusedTdtDagResult<N::State>> {
    if frames == 0 || encoder_dim == 0 || encoder.len() != frames * encoder_dim {
        bail!("invalid fused TDT encoder shape");
    }
    if encoder.iter().any(|value| !value.is_finite()) {
        bail!("fused TDT encoder contains a non-finite value");
    }
    if config.beam_size == 0 {
        bail!("fused TDT DAG beam size must be positive");
    }
    if config.max_symbols_per_step == 0 {
        bail!("fused TDT DAG max symbols per step must be positive");
    }
    if let FusedTdtHotwordCandidatePolicy::DirectPreTopK { acoustic_top_k } =
        hotword_candidate_policy
    {
        if hotwords.is_none() {
            bail!("direct pre-Top-K hotword policy requires a context graph");
        }
        if acoustic_top_k < config.beam_size {
            bail!("direct pre-Top-K acoustic width must be at least the retained beam size");
        }
    }

    let root = FusedTdtDagNode {
        score: 0.0,
        sequence_hash: FNV_OFFSET_BASIS,
        emitted_count: 0,
        last_token: None,
        state: Arc::new(network.initial_state()),
        frame: 0,
        history: None,
        symbols_since_advance: 0,
        hotword_state: hotwords.map_or(0, HotwordContextGraph::root),
    };
    let mut buckets = vec![Vec::new(); frames];
    buckets[0].push(root);
    let mut completed = Vec::new();
    let mut history = Vec::new();
    let mut stats = FusedTdtDagStats::default();

    for time in 0..frames {
        let mut wave = merge_and_prune_dag_nodes(
            std::mem::take(&mut buckets[time]),
            config.beam_size,
            merge_ablation,
            Some(&mut stats),
        );
        while !wave.is_empty() {
            stats.max_active_width = stats.max_active_width.max(wave.len());
            stats.network_batches += 1;
            stats.network_hypotheses += wave.len();
            let encoder_frames = wave
                .iter()
                .map(|_| {
                    (0..encoder_dim)
                        .map(|feature| encoder[feature * frames + time])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let tokens = wave
                .iter()
                .map(|node| node.last_token.unwrap_or(blank_id))
                .collect::<Vec<_>>();
            let frontier_states = wave
                .iter()
                .map(|node| node.state.as_ref())
                .collect::<Vec<_>>();
            let outputs = network.step_batch(&encoder_frames, &tokens, &frontier_states)?;
            if outputs.len() != wave.len() {
                bail!("fused TDT batch returned the wrong hypothesis count");
            }

            let mut same_frame = Vec::new();
            for (node, (logits, predicted_state)) in wave.into_iter().zip(outputs) {
                let (token_logits, duration_logits) = split_logits(&logits, blank_id + 1)?;
                let token_log_probs = tdt_log_softmax(token_logits);
                let duration_log_probs = tdt_log_softmax(duration_logits);
                let token_candidates = match hotword_candidate_policy {
                    FusedTdtHotwordCandidatePolicy::InjectedAll => {
                        let mut acoustic =
                            tdt_top_k(&token_log_probs[..blank_id], config.beam_size);
                        if let Some(graph) = hotwords {
                            for &token in graph.continuation_tokens(node.hotword_state) {
                                if token < blank_id
                                    && !acoustic.iter().any(|&(candidate, _)| candidate == token)
                                {
                                    acoustic.push((token, token_log_probs[token]));
                                }
                            }
                        }
                        acoustic
                            .into_iter()
                            .map(|(token, token_score)| {
                                let (hotword_score, hotword_state) = hotwords
                                    .map_or((0.0, node.hotword_state), |graph| {
                                        graph.forward(node.hotword_state, token)
                                    });
                                (token, token_score, hotword_score, hotword_state)
                            })
                            .collect::<Vec<_>>()
                    }
                    FusedTdtHotwordCandidatePolicy::DirectPreTopK { acoustic_top_k } => {
                        let graph = hotwords.expect("validated direct pre-Top-K graph");
                        let mut acoustic = tdt_top_k(&token_log_probs, acoustic_top_k);
                        for token in graph.direct_continuation_tokens(node.hotword_state) {
                            if token < blank_id
                                && !acoustic.iter().any(|&(candidate, _)| candidate == token)
                            {
                                acoustic.push((token, token_log_probs[token]));
                            }
                        }
                        let mut adjusted = acoustic
                            .into_iter()
                            .map(|(token, token_score)| {
                                let (hotword_score, hotword_state) = if token == blank_id {
                                    (0.0, node.hotword_state)
                                } else {
                                    graph.forward(node.hotword_state, token)
                                };
                                (token, token_score, hotword_score, hotword_state)
                            })
                            .collect::<Vec<_>>();
                        adjusted.sort_by(|left, right| {
                            (right.1 + right.2)
                                .total_cmp(&(left.1 + left.2))
                                .then_with(|| left.0.cmp(&right.0))
                        });
                        adjusted.truncate(config.beam_size);
                        adjusted
                    }
                };
                let nonblank_durations =
                    tdt_duration_candidates(&duration_log_probs, config.duration_expansion, true);
                let blank_durations =
                    tdt_duration_candidates(&duration_log_probs, config.duration_expansion, false);
                let predicted_state = Arc::new(predicted_state);

                for (token, token_score, hotword_score, hotword_state) in token_candidates {
                    if token == blank_id {
                        for &(duration_index, duration_score) in &blank_durations {
                            let duration = TDT_DURATIONS[duration_index];
                            let child = FusedTdtDagNode {
                                score: node.score + token_score + duration_score,
                                sequence_hash: node.sequence_hash,
                                emitted_count: node.emitted_count,
                                last_token: node.last_token,
                                state: Arc::clone(&node.state),
                                frame: time + duration,
                                history: node.history,
                                symbols_since_advance: 0,
                                hotword_state: node.hotword_state,
                            };
                            stats.generated_candidates += 1;
                            push_dag_child(
                                child,
                                frames,
                                time,
                                &mut same_frame,
                                &mut buckets,
                                &mut completed,
                            );
                        }
                        continue;
                    }
                    for &(duration_index, duration_score) in &nonblank_durations {
                        let duration = TDT_DURATIONS[duration_index];
                        let symbols_since_advance = if duration == 0 {
                            node.symbols_since_advance + 1
                        } else {
                            0
                        };
                        if duration == 0 && symbols_since_advance >= config.max_symbols_per_step {
                            continue;
                        }
                        let next_frame = time + duration;
                        let history_id = history.len();
                        history.push(FusedTdtHistoryNode {
                            parent: node.history,
                            token,
                            timestamp: time,
                            duration,
                        });
                        let child = FusedTdtDagNode {
                            score: node.score + token_score + duration_score + hotword_score,
                            sequence_hash: update_sequence_hash(node.sequence_hash, token),
                            emitted_count: node.emitted_count + 1,
                            last_token: Some(token),
                            state: Arc::clone(&predicted_state),
                            frame: next_frame,
                            history: Some(history_id),
                            symbols_since_advance,
                            hotword_state,
                        };
                        stats.generated_candidates += 1;
                        push_dag_child(
                            child,
                            frames,
                            time,
                            &mut same_frame,
                            &mut buckets,
                            &mut completed,
                        );
                    }
                }

                if hotword_candidate_policy == FusedTdtHotwordCandidatePolicy::InjectedAll {
                    for &(duration_index, duration_score) in &blank_durations {
                        let duration = TDT_DURATIONS[duration_index];
                        let child = FusedTdtDagNode {
                            score: node.score + token_log_probs[blank_id] + duration_score,
                            sequence_hash: node.sequence_hash,
                            emitted_count: node.emitted_count,
                            last_token: node.last_token,
                            state: Arc::clone(&node.state),
                            frame: time + duration,
                            history: node.history,
                            symbols_since_advance: 0,
                            hotword_state: node.hotword_state,
                        };
                        stats.generated_candidates += 1;
                        push_dag_child(
                            child,
                            frames,
                            time,
                            &mut same_frame,
                            &mut buckets,
                            &mut completed,
                        );
                    }
                }
            }
            wave = merge_and_prune_dag_nodes(
                same_frame,
                config.beam_size,
                merge_ablation,
                Some(&mut stats),
            );
        }
    }

    if let Some(graph) = hotwords {
        for node in &mut completed {
            node.score += graph.finalize(node.hotword_state);
            node.hotword_state = graph.root();
        }
    }
    let mut finalists = merge_and_prune_dag_nodes(
        completed,
        config.beam_size,
        merge_ablation,
        Some(&mut stats),
    );
    finalists.sort_by(|left, right| {
        dag_normalized_score(right)
            .total_cmp(&dag_normalized_score(left))
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.sequence_hash.cmp(&right.sequence_hash))
    });
    let hypotheses = finalists
        .iter()
        .map(|node| dag_node_to_hypothesis(node, &history))
        .collect();
    Ok(FusedTdtDagResult { hypotheses, stats })
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn update_sequence_hash(hash: u64, token: usize) -> u64 {
    (hash ^ (token as u64).wrapping_add(1)).wrapping_mul(FNV_PRIME)
}

fn push_dag_child<S>(
    child: FusedTdtDagNode<S>,
    frames: usize,
    current_frame: usize,
    same_frame: &mut Vec<FusedTdtDagNode<S>>,
    buckets: &mut [Vec<FusedTdtDagNode<S>>],
    completed: &mut Vec<FusedTdtDagNode<S>>,
) {
    if child.frame == current_frame {
        same_frame.push(child);
    } else if child.frame < frames {
        buckets[child.frame].push(child);
    } else {
        completed.push(child);
    }
}

fn merge_and_prune_dag_nodes<S>(
    mut nodes: Vec<FusedTdtDagNode<S>>,
    beam_size: usize,
    ablation: FusedTdtDagMergeAblation,
    mut stats: Option<&mut FusedTdtDagStats>,
) -> Vec<FusedTdtDagNode<S>> {
    let input_len = nodes.len();
    nodes.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.sequence_hash.cmp(&right.sequence_hash))
            .then_with(|| left.emitted_count.cmp(&right.emitted_count))
            .then_with(|| left.symbols_since_advance.cmp(&right.symbols_since_advance))
    });
    if ablation.order == FusedTdtDagMergeOrder::PruneThenMerge {
        nodes.truncate(beam_size);
    }
    let mut merged: Vec<FusedTdtDagNode<S>> = Vec::with_capacity(nodes.len());
    let mut duplicate_nodes_merged = 0;
    let mut symbol_budget_conflicts = 0;
    for node in nodes {
        if merged.iter().any(|candidate| {
            candidate.sequence_hash == node.sequence_hash
                && candidate.frame == node.frame
                && candidate.symbols_since_advance != node.symbols_since_advance
        }) {
            symbol_budget_conflicts += 1;
        }
        if let Some(existing) = merged.iter_mut().find(|candidate| {
            candidate.sequence_hash == node.sequence_hash
                && candidate.frame == node.frame
                && (!ablation.include_symbols_since_advance_in_key
                    || candidate.symbols_since_advance == node.symbols_since_advance)
        }) {
            debug_assert_eq!(existing.emitted_count, node.emitted_count);
            debug_assert_eq!(existing.hotword_state, node.hotword_state);
            existing.score = dag_log_add_exp(existing.score, node.score);
            existing.symbols_since_advance = existing
                .symbols_since_advance
                .min(node.symbols_since_advance);
            duplicate_nodes_merged += 1;
        } else {
            merged.push(node);
        }
    }
    if ablation.order == FusedTdtDagMergeOrder::MergeThenPrune {
        merged.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.sequence_hash.cmp(&right.sequence_hash))
                .then_with(|| left.emitted_count.cmp(&right.emitted_count))
                .then_with(|| left.symbols_since_advance.cmp(&right.symbols_since_advance))
        });
        merged.truncate(beam_size);
    }
    if let Some(stats) = stats.as_mut() {
        stats.merge_calls += 1;
        stats.merge_input_nodes += input_len;
        stats.duplicate_nodes_merged += duplicate_nodes_merged;
        stats.nodes_pruned += input_len.saturating_sub(duplicate_nodes_merged + merged.len());
        stats.symbol_budget_conflicts += symbol_budget_conflicts;
        if input_len >= beam_size && merged.len() < beam_size {
            stats.underfilled_merge_calls += 1;
        }
    }
    merged
}

fn dag_node_to_hypothesis<S: Clone>(
    node: &FusedTdtDagNode<S>,
    history: &[FusedTdtHistoryNode],
) -> FusedTdtHypothesis<S> {
    let mut token_ids = Vec::with_capacity(node.emitted_count);
    let mut timestamps = Vec::with_capacity(node.emitted_count);
    let mut durations = Vec::with_capacity(node.emitted_count);
    let mut cursor = node.history;
    while let Some(index) = cursor {
        let edge = &history[index];
        token_ids.push(edge.token);
        timestamps.push(edge.timestamp);
        durations.push(edge.duration);
        cursor = edge.parent;
    }
    token_ids.reverse();
    timestamps.reverse();
    durations.reverse();
    FusedTdtHypothesis {
        score: node.score,
        token_ids,
        timestamps,
        durations,
        state: node.state.as_ref().clone(),
        last_frame: node.frame,
    }
}

fn dag_normalized_score<S>(node: &FusedTdtDagNode<S>) -> f32 {
    node.score / (node.emitted_count + 1) as f32
}

const TDT_STATIC_COHERENCE_WEIGHT: f64 = 0.1;
const TDT_STATIC_LENGTH_EXPONENT: f64 = 0.5;

fn select_tdt_static_embedding_candidate<S>(
    hypotheses: &[FusedTdtHypothesis<S>],
    coherences: &[f64],
) -> Result<usize> {
    if hypotheses.is_empty() || hypotheses.len() != coherences.len() {
        bail!("TDT static embedding reranking requires one coherence per hypothesis");
    }
    if coherences.iter().any(|value| !value.is_finite()) {
        bail!("TDT static embedding coherence must be finite");
    }
    let count = f64::from(
        u16::try_from(coherences.len()).context("too many TDT static embedding candidates")?,
    );
    let mean = coherences.iter().sum::<f64>() / count;
    let deviation = (coherences
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    let standardized = |value: f64| {
        if deviation > 1.0e-12 {
            (value - mean) / deviation
        } else {
            0.0
        }
    };

    hypotheses
        .iter()
        .zip(coherences)
        .enumerate()
        .max_by(
            |(left_index, (left, left_coherence)), (right_index, (right, right_coherence))| {
                let score = |hypothesis: &FusedTdtHypothesis<S>, coherence: f64| {
                    let length = (hypothesis.token_ids.len() + 1) as f64;
                    f64::from(hypothesis.score) / length.powf(TDT_STATIC_LENGTH_EXPONENT)
                        + TDT_STATIC_COHERENCE_WEIGHT * standardized(coherence)
                };
                score(left, **left_coherence)
                    .total_cmp(&score(right, **right_coherence))
                    .then_with(|| right_index.cmp(left_index))
            },
        )
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow!("TDT static embedding reranking produced no candidate"))
}

fn dag_log_add_exp(left: f32, right: f32) -> f32 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn tdt_log_softmax(values: &[f32]) -> Vec<f32> {
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

fn tdt_top_k(values: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    indexed.truncate(count.min(indexed.len()));
    indexed
}

fn tdt_duration_candidates(
    log_probs: &[f32],
    expansion: FusedTdtDurationExpansion,
    allow_zero: bool,
) -> Vec<(usize, f32)> {
    let start = usize::from(!allow_zero);
    match expansion {
        FusedTdtDurationExpansion::All => {
            log_probs.iter().copied().enumerate().skip(start).collect()
        }
        FusedTdtDurationExpansion::Argmax => log_probs
            .iter()
            .copied()
            .enumerate()
            .skip(start)
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .into_iter()
            .collect(),
    }
}

fn split_logits(logits: &[f32], token_classes: usize) -> Result<(&[f32], &[f32])> {
    if logits.len() != token_classes + TDT_DURATIONS.len() {
        bail!(
            "fused TDT joint returned {} logits, expected {} token and {} duration logits",
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
        .ok_or_else(|| anyhow!("cannot take argmax of an empty tensor"))?;
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

#[derive(Clone)]
pub struct FusedPredictorState {
    hidden: Vec<f32>,
    cell: Vec<f32>,
}

struct OrtFusedTdtNetwork<'a> {
    decoder_joint: &'a mut Session,
}

impl FusedTdtStep for OrtFusedTdtNetwork<'_> {
    type State = FusedPredictorState;

    fn initial_state(&self) -> Self::State {
        let elements = PREDICTOR_LAYERS * PREDICTOR_DIM;
        FusedPredictorState {
            hidden: vec![0.0; elements],
            cell: vec![0.0; elements],
        }
    }

    fn step(
        &mut self,
        encoder_frame: &[f32],
        token: usize,
        state: &Self::State,
    ) -> Result<(Vec<f32>, Self::State)> {
        if encoder_frame.len() != ENCODER_DIM {
            bail!("invalid fused TDT encoder frame length");
        }
        let token = i32::try_from(token).context("fused TDT token id exceeds i32")?;
        let encoder =
            Tensor::from_array((vec![1_i64, ENCODER_DIM as i64, 1], encoder_frame.to_vec()))?;
        let targets = Tensor::from_array((vec![1_i64, 1], vec![token]))?;
        let target_length = Tensor::from_array((vec![1_i64], vec![1_i32]))?;
        let hidden =
            Tensor::from_array((vec![2_i64, 1, PREDICTOR_DIM as i64], state.hidden.clone()))?;
        let cell = Tensor::from_array((vec![2_i64, 1, PREDICTOR_DIM as i64], state.cell.clone()))?;
        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => encoder,
            "targets" => targets,
            "target_length" => target_length,
            "input_states_1" => hidden,
            "input_states_2" => cell,
        ])?;
        let logits = extract_f32(&outputs, "outputs", JOINT_WIDTH)?;
        let next_hidden = extract_f32(
            &outputs,
            "output_states_1",
            PREDICTOR_LAYERS * PREDICTOR_DIM,
        )?;
        let next_cell = extract_f32(
            &outputs,
            "output_states_2",
            PREDICTOR_LAYERS * PREDICTOR_DIM,
        )?;
        Ok((
            logits,
            FusedPredictorState {
                hidden: next_hidden,
                cell: next_cell,
            },
        ))
    }

    fn step_batch(
        &mut self,
        encoder_frames: &[Vec<f32>],
        tokens: &[usize],
        states: &[&Self::State],
    ) -> Result<Vec<(Vec<f32>, Self::State)>> {
        let batch = encoder_frames.len();
        if batch == 0 || batch != tokens.len() || batch != states.len() {
            bail!("fused TDT batch inputs have mismatched or empty lengths");
        }
        if encoder_frames
            .iter()
            .any(|frame| frame.len() != ENCODER_DIM)
        {
            bail!("invalid fused TDT batched encoder frame length");
        }
        let batch_i64 = i64::try_from(batch).context("fused TDT batch exceeds i64")?;
        let encoder_values = encoder_frames
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect::<Vec<_>>();
        let target_values = tokens
            .iter()
            .map(|&token| i32::try_from(token).context("fused TDT token id exceeds i32"))
            .collect::<Result<Vec<_>>>()?;
        let mut hidden_values = Vec::with_capacity(PREDICTOR_LAYERS * batch * PREDICTOR_DIM);
        let mut cell_values = Vec::with_capacity(PREDICTOR_LAYERS * batch * PREDICTOR_DIM);
        for layer in 0..PREDICTOR_LAYERS {
            let start = layer * PREDICTOR_DIM;
            let end = start + PREDICTOR_DIM;
            for state in states {
                hidden_values.extend_from_slice(&state.hidden[start..end]);
                cell_values.extend_from_slice(&state.cell[start..end]);
            }
        }
        let encoder = Tensor::from_array((vec![batch_i64, ENCODER_DIM as i64, 1], encoder_values))?;
        let targets = Tensor::from_array((vec![batch_i64, 1], target_values))?;
        let target_length = Tensor::from_array((vec![batch_i64], vec![1_i32; batch]))?;
        let hidden = Tensor::from_array((
            vec![PREDICTOR_LAYERS as i64, batch_i64, PREDICTOR_DIM as i64],
            hidden_values,
        ))?;
        let cell = Tensor::from_array((
            vec![PREDICTOR_LAYERS as i64, batch_i64, PREDICTOR_DIM as i64],
            cell_values,
        ))?;
        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => encoder,
            "targets" => targets,
            "target_length" => target_length,
            "input_states_1" => hidden,
            "input_states_2" => cell,
        ])?;
        let logits = extract_f32(&outputs, "outputs", batch * JOINT_WIDTH)?;
        let next_hidden = extract_f32(
            &outputs,
            "output_states_1",
            PREDICTOR_LAYERS * batch * PREDICTOR_DIM,
        )?;
        let next_cell = extract_f32(
            &outputs,
            "output_states_2",
            PREDICTOR_LAYERS * batch * PREDICTOR_DIM,
        )?;

        let mut results = Vec::with_capacity(batch);
        for batch_index in 0..batch {
            let mut hidden = Vec::with_capacity(PREDICTOR_LAYERS * PREDICTOR_DIM);
            let mut cell = Vec::with_capacity(PREDICTOR_LAYERS * PREDICTOR_DIM);
            for layer in 0..PREDICTOR_LAYERS {
                let start = (layer * batch + batch_index) * PREDICTOR_DIM;
                let end = start + PREDICTOR_DIM;
                hidden.extend_from_slice(&next_hidden[start..end]);
                cell.extend_from_slice(&next_cell[start..end]);
            }
            let logits_start = batch_index * JOINT_WIDTH;
            results.push((
                logits[logits_start..logits_start + JOINT_WIDTH].to_vec(),
                FusedPredictorState { hidden, cell },
            ));
        }
        Ok(results)
    }
}

/// Fast Japanese Parakeet CTC engine using the same encoder artifact as TDT.
///
/// The CTC head is kept in a separate one-thread session because it is small;
/// the configured ASR thread count is reserved for the compute-bound encoder.
pub struct SharedEncoderCtcJaOrtEngine {
    encoder: Session,
    ctc_head: Session,
    frontend: NemoMelFrontend,
    tokens: Vec<String>,
}

impl SharedEncoderCtcJaOrtEngine {
    /// Loads the pinned shared encoder and extracted CTC head.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid threads, missing split artifacts, an I/O
    /// contract mismatch, or an ONNX Runtime initialization failure.
    pub fn new(model_dir: &Path, encoder_threads: i32) -> Result<Self> {
        if encoder_threads <= 0 {
            bail!("encoder thread count must be greater than zero");
        }
        for name in SHARED_CTC_REQUIRED_FILES {
            let path = model_dir.join(name);
            if !path.is_file() {
                bail!(
                    "Japanese Parakeet shared CTC required artifact not found: {}",
                    path.display()
                );
            }
        }
        let tokens = load_vocab(&model_dir.join(VOCAB_FILE))?;
        init_onnx_runtime();
        let encoder_threads =
            usize::try_from(encoder_threads).context("invalid encoder thread count")?;
        let encoder = load_session(
            &model_dir.join("encoder-model.int8.onnx"),
            encoder_threads,
            "shared CTC encoder",
        )?;
        let ctc_head = load_session(&model_dir.join("ctc-head-model.onnx"), 1, "shared CTC head")?;
        validate_names(
            &encoder,
            &["audio_signal", "length"],
            &["outputs", "encoded_lengths"],
            "shared CTC encoder",
        )?;
        validate_names(
            &ctc_head,
            &["encoder_outputs", "encoded_lengths"],
            &["logprobs"],
            "shared CTC head",
        )?;
        Ok(Self {
            encoder,
            ctc_head,
            frontend: NemoMelFrontend::new(),
            tokens,
        })
    }

    fn encode(&mut self, samples: &[f32]) -> Result<ParakeetJaEncodedAudio> {
        let features = self.frontend.process(samples)?;
        let feature_frames =
            i64::try_from(features.frames).context("too many shared CTC feature frames")?;
        let valid_frames = i64::try_from(features.valid_frames)
            .context("too many valid shared CTC feature frames")?;
        let signal = Tensor::from_array((
            vec![1_i64, MEL_BINS as i64, feature_frames],
            features.values,
        ))?;
        let length = Tensor::from_array((vec![1_i64], vec![valid_frames]))?;
        let outputs = self.encoder.run(ort::inputs![
            "audio_signal" => signal,
            "length" => length,
        ])?;
        let (shape, values) = outputs
            .get("outputs")
            .ok_or_else(|| anyhow!("shared CTC encoder did not return outputs"))?
            .try_extract_tensor::<f32>()
            .context("failed to extract shared CTC encoder output")?;
        let encoded_length = outputs
            .get("encoded_lengths")
            .ok_or_else(|| anyhow!("shared CTC encoder did not return encoded_lengths"))?
            .try_extract_tensor::<i64>()
            .context("failed to extract shared CTC encoded length")?
            .1
            .first()
            .copied()
            .ok_or_else(|| anyhow!("shared CTC encoded length is empty"))?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != ENCODER_DIM as i64 {
            bail!("unexpected shared CTC encoder output shape: {shape:?}");
        }
        let all_frames = usize::try_from(shape[2]).context("invalid shared CTC encoder frames")?;
        let frames =
            usize::try_from(encoded_length).context("invalid shared CTC encoded length")?;
        if frames == 0 || frames > all_frames {
            bail!("shared CTC encoded length {frames} exceeds output frames {all_frames}");
        }
        let mut trimmed = vec![0.0; ENCODER_DIM * frames];
        for feature in 0..ENCODER_DIM {
            trimmed[feature * frames..(feature + 1) * frames]
                .copy_from_slice(&values[feature * all_frames..feature * all_frames + frames]);
        }
        Ok(ParakeetJaEncodedAudio {
            values: trimmed,
            frames,
        })
    }

    fn recognize_encoded(&mut self, encoded: ParakeetJaEncodedAudio) -> Result<AsrTranscript> {
        let frames = i64::try_from(encoded.frames).context("too many encoded CTC frames")?;
        let encoder_tensor =
            Tensor::from_array((vec![1_i64, ENCODER_DIM as i64, frames], encoded.values))?;
        let lengths = Tensor::from_array((vec![1_i64], vec![frames]))?;
        let outputs = self.ctc_head.run(ort::inputs![
            "encoder_outputs" => encoder_tensor,
            "encoded_lengths" => lengths,
        ])?;
        let (shape, log_probs) = outputs
            .get("logprobs")
            .ok_or_else(|| anyhow!("shared CTC head did not return logprobs"))?
            .try_extract_tensor::<f32>()
            .context("failed to extract shared CTC head logprobs")?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] != VOCAB_SIZE as i64 {
            bail!("unexpected shared CTC head output shape: {shape:?}");
        }
        let output_frames =
            usize::try_from(shape[1]).context("invalid shared CTC output frame count")?;
        let path = decode_ctc(
            log_probs,
            output_frames,
            VOCAB_SIZE,
            BLANK_ID,
            CtcDecodingStrategy::Greedy,
        )?;
        transcript_from_path(
            &path,
            &self.tokens,
            f64::from(FEATURE_FRAME_SHIFT_SEC) * f64::from(SUBSAMPLING_FACTOR),
        )
    }
}

impl AsrEngine for SharedEncoderCtcJaOrtEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let encoded = self.encode(samples)?;
        self.recognize_encoded(encoded)
    }
}

/// Japanese Parakeet hybrid engine with one encoder shared by the CTC and TDT heads.
///
/// The encoder and decoder sessions intentionally have independent intra-op
/// thread counts. The encoder is the large compute-bound graph, while both
/// decoder heads are small enough that one intra-op thread can be faster.
pub struct HybridParakeetJaOrtEngine {
    encoder: Session,
    ctc_head: Option<Session>,
    decoder_joint: Session,
    frontend: NemoMelFrontend,
    tokens: Vec<String>,
    strategy: ParakeetJaDecodingStrategy,
    static_embedding: Option<JapaneseStaticEmbeddingModel>,
}

impl HybridParakeetJaOrtEngine {
    /// Loads the shared encoder and both decoder heads.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid thread counts, missing artifacts, an I/O
    /// contract mismatch, or an ONNX Runtime initialization failure.
    pub fn new(
        model_dir: &Path,
        encoder_threads: i32,
        decoder_threads: i32,
        strategy: ParakeetJaDecodingStrategy,
    ) -> Result<Self> {
        Self::new_inner(
            model_dir,
            encoder_threads,
            decoder_threads,
            strategy,
            None,
            true,
        )
    }

    /// Loads the hybrid model together with the Japanese static embedding scorer.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid model or embedding artifact contract.
    pub fn new_with_static_embedding(
        model_dir: &Path,
        encoder_threads: i32,
        decoder_threads: i32,
        strategy: ParakeetJaDecodingStrategy,
        static_embedding_dir: &Path,
    ) -> Result<Self> {
        Self::new_inner(
            model_dir,
            encoder_threads,
            decoder_threads,
            strategy,
            Some(static_embedding_dir),
            true,
        )
    }

    /// Loads only the TDT accuracy path, optionally adding the CTC head used
    /// to admit hotword paths. The ordinary TDT DAG does not require CTC.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid thread counts or a missing artifact that
    /// is required by the selected path.
    pub fn new_tdt_dag(
        model_dir: &Path,
        encoder_threads: i32,
        decoder_threads: i32,
        config: FusedTdtDagConfig,
        load_ctc_head: bool,
    ) -> Result<Self> {
        Self::new_inner(
            model_dir,
            encoder_threads,
            decoder_threads,
            ParakeetJaDecodingStrategy::TdtVariableDag(config),
            None,
            load_ctc_head,
        )
    }

    fn new_inner(
        model_dir: &Path,
        encoder_threads: i32,
        decoder_threads: i32,
        strategy: ParakeetJaDecodingStrategy,
        static_embedding_dir: Option<&Path>,
        load_ctc_head: bool,
    ) -> Result<Self> {
        if encoder_threads <= 0 {
            bail!("encoder thread count must be greater than zero");
        }
        if decoder_threads <= 0 {
            bail!("decoder thread count must be greater than zero");
        }
        let tokens = load_vocab(&model_dir.join(VOCAB_FILE))?;
        let static_embedding = static_embedding_dir
            .map(JapaneseStaticEmbeddingModel::load)
            .transpose()?;
        if matches!(
            strategy,
            ParakeetJaDecodingStrategy::TdtVariableDagStaticEmbedding(_)
        ) && static_embedding.is_none()
        {
            bail!("static-embedding TDT DAG strategy requires a static embedding directory");
        }

        init_onnx_runtime();
        let encoder_threads =
            usize::try_from(encoder_threads).context("invalid encoder thread count")?;
        let decoder_threads =
            usize::try_from(decoder_threads).context("invalid decoder thread count")?;
        let encoder_path = resolve_model_file(model_dir, ENCODER_FILES, "shared encoder")?;
        let decoder_joint_path =
            resolve_model_file(model_dir, DECODER_JOINT_FILES, "fused TDT decoder_joint")?;
        let encoder = load_session(&encoder_path, encoder_threads, "shared encoder")?;
        let ctc_head = if load_ctc_head {
            let ctc_head_path = resolve_model_file(model_dir, CTC_HEAD_FILES, "CTC head")?;
            Some(load_session(&ctc_head_path, decoder_threads, "CTC head")?)
        } else {
            None
        };
        let decoder_joint = load_session(
            &decoder_joint_path,
            decoder_threads,
            "fused TDT decoder_joint",
        )?;
        validate_names(
            &encoder,
            &["audio_signal", "length"],
            &["outputs", "encoded_lengths"],
            "shared encoder",
        )?;
        if let Some(ctc_head) = &ctc_head {
            validate_names(
                ctc_head,
                &["encoder_outputs", "encoded_lengths"],
                &["logprobs"],
                "CTC head",
            )?;
        }
        validate_names(
            &decoder_joint,
            &[
                "encoder_outputs",
                "targets",
                "target_length",
                "input_states_1",
                "input_states_2",
            ],
            &[
                "outputs",
                "prednet_lengths",
                "output_states_1",
                "output_states_2",
            ],
            "decoder_joint",
        )?;

        Ok(Self {
            encoder,
            ctc_head,
            decoder_joint,
            frontend: NemoMelFrontend::new(),
            tokens,
            strategy,
            static_embedding,
        })
    }

    /// Runs the shared frontend and encoder exactly once.
    ///
    /// # Errors
    ///
    /// Returns an error when feature extraction or encoder inference fails.
    pub fn encode(&mut self, samples: &[f32]) -> Result<ParakeetJaEncodedAudio> {
        let features = self.frontend.process(samples)?;
        let feature_frames =
            i64::try_from(features.frames).context("too many shared encoder feature frames")?;
        let valid_frames = i64::try_from(features.valid_frames)
            .context("too many valid shared encoder feature frames")?;
        let signal = Tensor::from_array((
            vec![1_i64, MEL_BINS as i64, feature_frames],
            features.values,
        ))?;
        let length = Tensor::from_array((vec![1_i64], vec![valid_frames]))?;
        let outputs = self.encoder.run(ort::inputs![
            "audio_signal" => signal,
            "length" => length,
        ])?;
        let (shape, values) = outputs
            .get("outputs")
            .ok_or_else(|| anyhow!("shared encoder did not return outputs"))?
            .try_extract_tensor::<f32>()
            .context("failed to extract shared encoder output")?;
        let encoded_length = outputs
            .get("encoded_lengths")
            .ok_or_else(|| anyhow!("shared encoder did not return encoded_lengths"))?
            .try_extract_tensor::<i64>()
            .context("failed to extract shared encoder length")?
            .1
            .first()
            .copied()
            .ok_or_else(|| anyhow!("shared encoder length is empty"))?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != ENCODER_DIM as i64 {
            bail!("unexpected shared encoder output shape: {shape:?}");
        }
        let all_frames = usize::try_from(shape[2]).context("invalid shared encoder frames")?;
        let frames = usize::try_from(encoded_length).context("invalid shared encoded length")?;
        if frames == 0 || frames > all_frames {
            bail!("shared encoded length {frames} exceeds output frames {all_frames}");
        }
        let mut trimmed = vec![0.0; ENCODER_DIM * frames];
        for feature in 0..ENCODER_DIM {
            trimmed[feature * frames..(feature + 1) * frames]
                .copy_from_slice(&values[feature * all_frames..feature * all_frames + frames]);
        }
        Ok(ParakeetJaEncodedAudio {
            values: trimmed,
            frames,
        })
    }

    /// Decodes one shared encoder output through either hybrid-model head.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected decoder head fails or emits an
    /// invalid tensor or token id.
    pub fn decode_encoded(
        &mut self,
        encoded: &ParakeetJaEncodedAudio,
        strategy: ParakeetJaDecodingStrategy,
    ) -> Result<AsrTranscript> {
        match strategy {
            ParakeetJaDecodingStrategy::CtcGreedy => self.decode_ctc(encoded),
            ParakeetJaDecodingStrategy::TdtGreedy => self.decode_tdt(encoded),
            ParakeetJaDecodingStrategy::TdtVariableDag(config) => {
                self.decode_tdt_dag(encoded, config)
            }
            ParakeetJaDecodingStrategy::TdtVariableDagStaticEmbedding(config) => {
                self.decode_tdt_dag_static_embedding(encoded, config)
            }
        }
    }

    /// Decodes one shared encoder output with experimental DAG merge semantics.
    ///
    /// The returned counters make it possible to distinguish output changes
    /// from a merge-key or frontier-width change during an offline ablation.
    ///
    /// # Errors
    ///
    /// Returns an error when the fused decoder fails or emits an invalid token.
    pub fn decode_encoded_tdt_dag_with_merge_ablation(
        &mut self,
        encoded: &ParakeetJaEncodedAudio,
        config: FusedTdtDagConfig,
        merge_ablation: FusedTdtDagMergeAblation,
    ) -> Result<(AsrTranscript, FusedTdtDagStats)> {
        let mut network = OrtFusedTdtNetwork {
            decoder_joint: &mut self.decoder_joint,
        };
        let result = variable_width_dag_fused_tdt_with_merge_ablation(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
            config,
            merge_ablation,
        )?;
        let hypothesis = result
            .hypotheses
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("fused TDT DAG returned no hypothesis"))?;
        let transcript = self.tdt_hypothesis_to_transcript(&hypothesis)?;
        Ok((transcript, result.stats))
    }

    /// Decodes one shared encoder output with the variable-width TDT DAG and
    /// a pre-tokenized hotword context graph.
    ///
    /// # Errors
    ///
    /// Returns an error when the fused decoder fails or emits an invalid token.
    pub fn decode_encoded_tdt_dag_with_hotwords(
        &mut self,
        encoded: &ParakeetJaEncodedAudio,
        config: FusedTdtDagConfig,
        hotwords: &HotwordContextGraph,
    ) -> Result<AsrTranscript> {
        self.decode_encoded_tdt_dag_with_hotword_policy(
            encoded,
            config,
            hotwords,
            FusedTdtHotwordCandidatePolicy::InjectedAll,
        )
    }

    /// Decodes one shared encoder output with an explicit hotword candidate
    /// admission policy.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid search widths, fused decoder failures, or
    /// an invalid emitted token.
    pub fn decode_encoded_tdt_dag_with_hotword_policy(
        &mut self,
        encoded: &ParakeetJaEncodedAudio,
        config: FusedTdtDagConfig,
        hotwords: &HotwordContextGraph,
        policy: FusedTdtHotwordCandidatePolicy,
    ) -> Result<AsrTranscript> {
        self.decode_encoded_tdt_dag_hotword_candidates(encoded, config, hotwords, policy)?
            .into_iter()
            .next()
            .map(|candidate| candidate.transcript)
            .ok_or_else(|| anyhow!("hotword TDT DAG returned no hypothesis"))
    }

    /// Retains the completed hotword TDT paths for a cheap second-pass
    /// language-model reranker without rerunning the encoder or decoder.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid search widths, fused decoder failures, or
    /// invalid emitted token ids.
    pub fn decode_encoded_tdt_dag_hotword_candidates(
        &mut self,
        encoded: &ParakeetJaEncodedAudio,
        config: FusedTdtDagConfig,
        hotwords: &HotwordContextGraph,
        policy: FusedTdtHotwordCandidatePolicy,
    ) -> Result<Vec<ParakeetJaTdtCandidate>> {
        let mut network = OrtFusedTdtNetwork {
            decoder_joint: &mut self.decoder_joint,
        };
        let result = variable_width_dag_fused_tdt_with_hotword_policy(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
            config,
            hotwords,
            policy,
        )?;
        result
            .hypotheses
            .into_iter()
            .map(|hypothesis| {
                let acoustic_score = hypothesis.score / (hypothesis.token_ids.len() + 1) as f32;
                Ok(ParakeetJaTdtCandidate {
                    transcript: tdt_hypothesis_to_transcript_with_hotwords(
                        &self.tokens,
                        &hypothesis,
                        hotwords,
                    )?,
                    acoustic_score,
                })
            })
            .collect()
    }

    /// Runs the shared CTC head once and retains its frame posteriors for
    /// keyword gating and transcript decoding.
    ///
    /// # Errors
    ///
    /// Returns an error when ORT inference fails or the output contract does
    /// not match the pinned Japanese Parakeet vocabulary.
    pub fn ctc_output(&mut self, encoded: &ParakeetJaEncodedAudio) -> Result<ParakeetJaCtcOutput> {
        let frames = i64::try_from(encoded.frames).context("too many encoded CTC frames")?;
        let encoder_tensor = Tensor::from_array((
            vec![1_i64, ENCODER_DIM as i64, frames],
            encoded.values.clone(),
        ))?;
        let lengths = Tensor::from_array((vec![1_i64], vec![frames]))?;
        let ctc_head = self.ctc_head.as_mut().ok_or_else(|| {
            anyhow!("CTC head was not loaded; CTC decoding and hotword gating are unavailable")
        })?;
        let outputs = ctc_head.run(ort::inputs![
            "encoder_outputs" => encoder_tensor,
            "encoded_lengths" => lengths,
        ])?;
        let (shape, log_probs) = outputs
            .get("logprobs")
            .ok_or_else(|| anyhow!("CTC head did not return logprobs"))?
            .try_extract_tensor::<f32>()
            .context("failed to extract CTC head logprobs")?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] != VOCAB_SIZE as i64 {
            bail!("unexpected CTC head output shape: {shape:?}");
        }
        let output_frames =
            usize::try_from(shape[1]).context("invalid CTC head output frame count")?;
        Ok(ParakeetJaCtcOutput {
            values: log_probs.to_vec(),
            frames: output_frames,
        })
    }

    /// Greedily decodes a previously computed CTC posterior matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid CTC path or token id.
    pub fn decode_ctc_output(&self, output: &ParakeetJaCtcOutput) -> Result<AsrTranscript> {
        let path = decode_ctc(
            &output.values,
            output.frames,
            VOCAB_SIZE,
            BLANK_ID,
            CtcDecodingStrategy::Greedy,
        )?;
        transcript_from_path(
            &path,
            &self.tokens,
            f64::from(FEATURE_FRAME_SHIFT_SEC) * f64::from(SUBSAMPLING_FACTOR),
        )
    }

    fn decode_ctc(&mut self, encoded: &ParakeetJaEncodedAudio) -> Result<AsrTranscript> {
        let output = self.ctc_output(encoded)?;
        self.decode_ctc_output(&output)
    }

    fn decode_tdt(&mut self, encoded: &ParakeetJaEncodedAudio) -> Result<AsrTranscript> {
        let mut network = OrtFusedTdtNetwork {
            decoder_joint: &mut self.decoder_joint,
        };
        let hypothesis = greedy_fused_tdt(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
        )?;
        let token_texts = hypothesis
            .token_ids
            .iter()
            .map(|&id| {
                self.tokens
                    .get(id)
                    .map(|token| token.replace('▁', " "))
                    .ok_or_else(|| anyhow!("fused TDT decoder emitted unknown token id {id}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let text = token_texts.concat();
        let timestamps = hypothesis
            .timestamps
            .iter()
            .map(|&frame| frame as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
            .collect::<Vec<_>>();
        let durations = hypothesis
            .durations
            .iter()
            .map(|&frames| frames as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
            .collect::<Vec<_>>();
        Ok(AsrTranscript::from_parts(
            text,
            token_texts,
            Some(&timestamps),
            Some(&durations),
        ))
    }

    fn decode_tdt_dag(
        &mut self,
        encoded: &ParakeetJaEncodedAudio,
        config: FusedTdtDagConfig,
    ) -> Result<AsrTranscript> {
        let mut network = OrtFusedTdtNetwork {
            decoder_joint: &mut self.decoder_joint,
        };
        let result = variable_width_dag_fused_tdt(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
            config,
        )?;
        let hypothesis = result
            .hypotheses
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("fused TDT DAG returned no hypothesis"))?;
        self.tdt_hypothesis_to_transcript(&hypothesis)
    }

    fn decode_tdt_dag_static_embedding(
        &mut self,
        encoded: &ParakeetJaEncodedAudio,
        config: FusedTdtDagConfig,
    ) -> Result<AsrTranscript> {
        let mut network = OrtFusedTdtNetwork {
            decoder_joint: &mut self.decoder_joint,
        };
        let result = variable_width_dag_fused_tdt(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
            config,
        )?;
        let texts = result
            .hypotheses
            .iter()
            .map(|hypothesis| self.tdt_hypothesis_text(hypothesis))
            .collect::<Result<Vec<_>>>()?;
        let static_embedding = self.static_embedding.as_mut().ok_or_else(|| {
            anyhow!("static-embedding TDT DAG strategy has no static embedding model")
        })?;
        let coherences = texts
            .iter()
            .map(|text| static_embedding.piece_mean(text))
            .collect::<Result<Vec<_>>>()?;
        let selected = select_tdt_static_embedding_candidate(&result.hypotheses, &coherences)?;
        self.tdt_hypothesis_to_transcript(&result.hypotheses[selected])
    }

    fn tdt_hypothesis_text(
        &self,
        hypothesis: &FusedTdtHypothesis<FusedPredictorState>,
    ) -> Result<String> {
        hypothesis
            .token_ids
            .iter()
            .map(|&id| {
                self.tokens
                    .get(id)
                    .map(|token| token.replace('▁', " "))
                    .ok_or_else(|| anyhow!("fused TDT DAG emitted unknown token id {id}"))
            })
            .collect::<Result<String>>()
    }

    fn tdt_hypothesis_to_transcript(
        &self,
        hypothesis: &FusedTdtHypothesis<FusedPredictorState>,
    ) -> Result<AsrTranscript> {
        let token_texts = hypothesis
            .token_ids
            .iter()
            .map(|&id| {
                self.tokens
                    .get(id)
                    .map(|token| token.replace('▁', " "))
                    .ok_or_else(|| anyhow!("fused TDT DAG emitted unknown token id {id}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let text = token_texts.concat();
        let timestamps = hypothesis
            .timestamps
            .iter()
            .map(|&frame| frame as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
            .collect::<Vec<_>>();
        let durations = hypothesis
            .durations
            .iter()
            .map(|&frames| frames as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
            .collect::<Vec<_>>();
        Ok(AsrTranscript::from_parts(
            text,
            token_texts,
            Some(&timestamps),
            Some(&durations),
        ))
    }
}

impl AsrEngine for HybridParakeetJaOrtEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let encoded = self.encode(samples)?;
        self.decode_encoded(&encoded, self.strategy)
    }
}

/// Product TDT-DAG engine for the Japanese hybrid Parakeet artifact.
///
/// It keeps the CTC head entirely out of the ordinary accuracy path. When
/// hotwords and a gate threshold are both supplied, the CTC head runs once
/// after the shared encoder solely to select which registered phrases may be
/// injected into the TDT search.
pub struct ParakeetJaTdtDagOrtAsrEngine {
    hybrid: HybridParakeetJaOrtEngine,
    dag_config: FusedTdtDagConfig,
    ctc_gate_threshold: Option<f32>,
    hotword_paths: Vec<HotwordTokenPath>,
}

impl ParakeetJaTdtDagOrtAsrEngine {
    /// Loads the accuracy-mode hybrid contract.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid config, missing required hybrid
    /// artifact, or a hotword surface/reading that cannot be represented by
    /// the pinned vocabulary.
    pub fn new(
        model_dir: &Path,
        encoder_threads: i32,
        decoder_threads: i32,
        dag_config: FusedTdtDagConfig,
        ctc_gate_threshold: Option<f32>,
        hotwords: &[super::HotwordEntry],
    ) -> Result<Self> {
        if dag_config.beam_size == 0 {
            bail!("Parakeet TDT DAG beam size must be positive");
        }
        if dag_config.max_symbols_per_step == 0 {
            bail!("Parakeet TDT DAG max symbols per step must be positive");
        }
        if ctc_gate_threshold.is_some_and(|threshold| !threshold.is_finite() || threshold > 0.0) {
            bail!("Parakeet CTC hotword gate threshold must be finite and non-positive");
        }
        let load_ctc_head = uses_ctc_hotword_gate(hotwords.len(), ctc_gate_threshold);
        let required_files = HYBRID_REQUIRED_FILES.iter().copied().chain(
            load_ctc_head
                .then_some(HYBRID_CTC_GATE_REQUIRED_FILES)
                .into_iter()
                .flatten()
                .copied(),
        );
        for name in required_files {
            let path = model_dir.join(name);
            if !path.is_file() {
                bail!(
                    "Japanese Parakeet hybrid required artifact not found: {}",
                    path.display()
                );
            }
        }
        let hybrid = HybridParakeetJaOrtEngine::new_tdt_dag(
            model_dir,
            encoder_threads,
            decoder_threads,
            dag_config,
            load_ctc_head,
        )?;
        let hotword_paths = tokenize_parakeet_hotword_entries(hotwords, &hybrid.tokens)?;
        Ok(Self {
            hybrid,
            dag_config,
            ctc_gate_threshold,
            hotword_paths,
        })
    }
}

const fn uses_ctc_hotword_gate(hotword_count: usize, threshold: Option<f32>) -> bool {
    hotword_count > 0 && threshold.is_some()
}

impl AsrEngine for ParakeetJaTdtDagOrtAsrEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let encoded = self.hybrid.encode(samples)?;
        if self.hotword_paths.is_empty() {
            return self.hybrid.decode_encoded(
                &encoded,
                ParakeetJaDecodingStrategy::TdtVariableDag(self.dag_config),
            );
        }

        let paths = if let Some(threshold) = self.ctc_gate_threshold {
            let ctc = self.hybrid.ctc_output(&encoded)?;
            ctc_gate_hotword_paths(
                &ctc,
                &self.hotword_paths,
                threshold,
                super::PARAKEET_JA_HOTWORD_MAX_GAP_FRAMES,
            )?
        } else {
            self.hotword_paths.clone()
        };
        if paths.is_empty() {
            return self.hybrid.decode_encoded(
                &encoded,
                ParakeetJaDecodingStrategy::TdtVariableDag(self.dag_config),
            );
        }
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            paths,
            super::PARAKEET_JA_HOTWORD_PHRASE_MULTIPLIER,
        )?;
        self.hybrid.decode_encoded_tdt_dag_with_hotword_policy(
            &encoded,
            self.dag_config,
            &graph,
            FusedTdtHotwordCandidatePolicy::DirectPreTopK {
                acoustic_top_k: super::PARAKEET_JA_HOTWORD_ACOUSTIC_TOP_K,
            },
        )
    }
}

fn tokenize_parakeet_hotword_entries(
    entries: &[super::HotwordEntry],
    tokens: &[String],
) -> Result<Vec<HotwordTokenPath>> {
    let token_ids = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| (token.as_str(), id))
        .collect::<HashMap<_, _>>();
    entries
        .iter()
        .enumerate()
        .try_fold(Vec::new(), |mut paths, (entry_id, entry)| {
            if entry.surface.trim().is_empty() {
                bail!("hotword surface must not be empty");
            }
            match tokenize_parakeet_hotword_text(&entry.surface, tokens, &token_ids, "surface") {
                Ok(surface_paths) => {
                    paths.extend(surface_paths.into_iter().map(|tokens| HotwordTokenPath {
                        tokens,
                        entry_id,
                        surface: entry.surface.clone(),
                        kind: super::HotwordPathKind::Surface,
                        phrase_score: entry.phrase_score,
                    }));
                }
                Err(error) if entry.readings.is_empty() => return Err(error),
                Err(_) => {}
            }
            for reading in &entry.readings {
                let normalized = super::normalize_reading(reading);
                if normalized.trim().is_empty() {
                    bail!("hotword reading must not be empty");
                }
                let hiragana_paths =
                    tokenize_parakeet_hotword_text(&normalized, tokens, &token_ids, "reading")?;
                paths.extend(hiragana_paths.into_iter().map(|tokens| HotwordTokenPath {
                    tokens,
                    entry_id,
                    surface: entry.surface.clone(),
                    kind: super::HotwordPathKind::Reading,
                    phrase_score: entry.phrase_score,
                }));
                let katakana = normalized
                    .chars()
                    .map(|character| match character {
                        '\u{3041}'..='\u{3096}' => {
                            char::from_u32(character as u32 + 0x60).unwrap_or(character)
                        }
                        _ => character,
                    })
                    .collect::<String>();
                if katakana != normalized {
                    let katakana_paths =
                        tokenize_parakeet_hotword_text(&katakana, tokens, &token_ids, "reading")?;
                    paths.extend(katakana_paths.into_iter().map(|tokens| HotwordTokenPath {
                        tokens,
                        entry_id,
                        surface: entry.surface.clone(),
                        kind: super::HotwordPathKind::Reading,
                        phrase_score: entry.phrase_score,
                    }));
                }
            }
            Ok(paths)
        })
}

const MAX_PARAKEET_HOTWORD_PATHS_PER_TEXT: usize = 8;
const MAX_PARAKEET_HOTWORD_SEGMENTATION_EXPANSIONS: usize = 4_096;
type ParakeetPieceCandidate = (usize, usize, usize);

fn parakeet_piece_candidates(
    normalized: &str,
    vocabulary: &[String],
    boundaries: &[usize],
) -> Vec<Vec<ParakeetPieceCandidate>> {
    let mut pieces_at = vec![Vec::new(); boundaries.len() - 1];
    for (token_id, raw_piece) in vocabulary.iter().enumerate() {
        let piece = raw_piece.strip_prefix('▁').unwrap_or(raw_piece);
        if piece.is_empty() || piece.starts_with('<') || piece.chars().any(char::is_whitespace) {
            continue;
        }
        for start in 0..boundaries.len() - 1 {
            if raw_piece.starts_with('▁') && start != 0 {
                continue;
            }
            let suffix = &normalized[boundaries[start]..];
            if !suffix.starts_with(piece) {
                continue;
            }
            let end_offset = boundaries[start] + piece.len();
            if let Ok(end) = boundaries.binary_search(&end_offset) {
                pieces_at[start].push((end, token_id, piece.chars().count()));
            }
        }
    }
    for pieces in &mut pieces_at {
        pieces.sort_unstable_by(|left, right| {
            right.2.cmp(&left.2).then_with(|| left.1.cmp(&right.1))
        });
    }
    pieces_at
}

fn visit_parakeet_hotword_segmentations(
    position: usize,
    terminal: usize,
    pieces_at: &[Vec<ParakeetPieceCandidate>],
    current: &mut Vec<usize>,
    completed: &mut Vec<Vec<usize>>,
    expansions: &mut usize,
) {
    if position == terminal {
        completed.push(current.clone());
        return;
    }
    for &(end, token_id, _) in &pieces_at[position] {
        if *expansions >= MAX_PARAKEET_HOTWORD_SEGMENTATION_EXPANSIONS {
            return;
        }
        *expansions += 1;
        current.push(token_id);
        visit_parakeet_hotword_segmentations(
            end, terminal, pieces_at, current, completed, expansions,
        );
        current.pop();
    }
}

fn tokenize_parakeet_hotword_text(
    text: &str,
    vocabulary: &[String],
    token_ids: &HashMap<&str, usize>,
    kind: &str,
) -> Result<Vec<Vec<usize>>> {
    let normalized = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    if normalized.is_empty() {
        bail!(
            "Parakeet Japanese hotword {kind} must contain at least one non-whitespace character"
        );
    }

    // Preserve the character path used by the initial implementation even
    // when a shorter SentencePiece-style segmentation is also available.
    let character_path = normalized
        .chars()
        .map(|character| {
            let token = character.to_string();
            token_ids.get(token.as_str()).copied()
        })
        .collect::<Option<Vec<_>>>();

    let boundaries = normalized
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(normalized.len()))
        .collect::<Vec<_>>();
    let pieces_at = parakeet_piece_candidates(&normalized, vocabulary, &boundaries);
    // Longer pieces are explored first; token id makes ties deterministic.
    let mut completed = Vec::new();
    visit_parakeet_hotword_segmentations(
        0,
        boundaries.len() - 1,
        &pieces_at,
        &mut Vec::new(),
        &mut completed,
        &mut 0,
    );
    completed
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    completed.dedup();

    let mut selected = Vec::with_capacity(MAX_PARAKEET_HOTWORD_PATHS_PER_TEXT);
    if let Some(character_path) = character_path {
        selected.push(character_path.clone());
        completed.retain(|path| path != &character_path);
    }
    selected.extend(
        completed
            .into_iter()
            .take(MAX_PARAKEET_HOTWORD_PATHS_PER_TEXT.saturating_sub(selected.len())),
    );
    if selected.is_empty() {
        bail!("Parakeet Japanese hotword {kind} cannot be segmented by the model vocabulary");
    }
    Ok(selected)
}

fn tdt_hypothesis_to_transcript_with_hotwords<S>(
    tokens: &[String],
    hypothesis: &FusedTdtHypothesis<S>,
    graph: &HotwordContextGraph,
) -> Result<AsrTranscript> {
    if hypothesis.token_ids.len() != hypothesis.timestamps.len()
        || hypothesis.token_ids.len() != hypothesis.durations.len()
    {
        bail!("hotword TDT hypothesis token, timestamp, and duration counts differ");
    }
    let matches = graph.find_matches(&hypothesis.token_ids);
    let frame_seconds = FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR;
    let mut token_texts = Vec::new();
    let mut timestamps = Vec::new();
    let mut durations = Vec::new();
    let mut cursor = 0;

    for matched in matches {
        while cursor < matched.start_token {
            let token_id = hypothesis.token_ids[cursor];
            let text = tokens
                .get(token_id)
                .with_context(|| format!("fused TDT DAG emitted unknown token id {token_id}"))?;
            token_texts.push(text.replace('▁', " "));
            timestamps.push(hypothesis.timestamps[cursor] as f32 * frame_seconds);
            durations.push(hypothesis.durations[cursor] as f32 * frame_seconds);
            cursor += 1;
        }

        let last = matched.end_token - 1;
        let start = hypothesis.timestamps[matched.start_token] as f32 * frame_seconds;
        let end = (hypothesis.timestamps[last] + hypothesis.durations[last]) as f32 * frame_seconds;
        token_texts.push(matched.surface);
        timestamps.push(start);
        durations.push((end - start).max(0.0));
        cursor = matched.end_token;
    }

    while cursor < hypothesis.token_ids.len() {
        let token_id = hypothesis.token_ids[cursor];
        let text = tokens
            .get(token_id)
            .with_context(|| format!("fused TDT DAG emitted unknown token id {token_id}"))?;
        token_texts.push(text.replace('▁', " "));
        timestamps.push(hypothesis.timestamps[cursor] as f32 * frame_seconds);
        durations.push(hypothesis.durations[cursor] as f32 * frame_seconds);
        cursor += 1;
    }

    Ok(AsrTranscript::from_parts(
        token_texts.concat(),
        token_texts,
        Some(&timestamps),
        Some(&durations),
    ))
}

/// Greedy TDT decoding of the onnx-asr fused Japanese Parakeet export.
pub struct FusedTdtJaOrtEngine {
    encoder: Session,
    decoder_joint: Session,
    frontend: NemoMelFrontend,
    tokens: Vec<String>,
}

impl FusedTdtJaOrtEngine {
    /// Loads the fused TDT export with greedy decoding.
    ///
    /// # Errors
    ///
    /// Returns an error for missing or invalid model artifacts, or an ONNX
    /// Runtime session initialization failure.
    pub fn new(model_dir: &Path, num_threads: i32) -> Result<Self> {
        if num_threads <= 0 {
            bail!("ASR thread count must be greater than zero");
        }
        let tokens = load_vocab(&model_dir.join(VOCAB_FILE))?;

        init_onnx_runtime();
        let threads = usize::try_from(num_threads).context("invalid ASR thread count")?;
        let encoder_path = resolve_model_file(model_dir, ENCODER_FILES, "fused TDT encoder")?;
        let decoder_joint_path =
            resolve_model_file(model_dir, DECODER_JOINT_FILES, "fused TDT decoder_joint")?;
        let encoder = load_session(&encoder_path, threads, "fused TDT encoder")?;
        let decoder_joint = load_session(&decoder_joint_path, threads, "fused TDT decoder_joint")?;
        validate_names(
            &encoder,
            &["audio_signal", "length"],
            &["outputs", "encoded_lengths"],
            "encoder",
        )?;
        validate_names(
            &decoder_joint,
            &[
                "encoder_outputs",
                "targets",
                "target_length",
                "input_states_1",
                "input_states_2",
            ],
            &[
                "outputs",
                "prednet_lengths",
                "output_states_1",
                "output_states_2",
            ],
            "decoder_joint",
        )?;

        Ok(Self {
            encoder,
            decoder_joint,
            frontend: NemoMelFrontend::new(),
            tokens,
        })
    }
}

impl AsrEngine for FusedTdtJaOrtEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let features = self.frontend.process(samples)?;
        let feature_frames =
            i64::try_from(features.frames).context("too many fused TDT feature frames")?;
        let valid_frames = i64::try_from(features.valid_frames)
            .context("too many valid fused TDT feature frames")?;
        let signal = Tensor::from_array((
            vec![1_i64, MEL_BINS as i64, feature_frames],
            features.values,
        ))?;
        let length = Tensor::from_array((vec![1_i64], vec![valid_frames]))?;
        let encoder_outputs = self.encoder.run(ort::inputs![
            "audio_signal" => signal,
            "length" => length,
        ])?;
        let (encoder_shape, encoder_values) = encoder_outputs
            .get("outputs")
            .ok_or_else(|| anyhow!("fused TDT encoder did not return outputs"))?
            .try_extract_tensor::<f32>()
            .context("failed to extract fused TDT encoder output")?;
        let encoded_length = encoder_outputs
            .get("encoded_lengths")
            .ok_or_else(|| anyhow!("fused TDT encoder did not return encoded_lengths"))?
            .try_extract_tensor::<i64>()
            .context("failed to extract fused TDT encoded length")?
            .1
            .first()
            .copied()
            .ok_or_else(|| anyhow!("fused TDT encoded length is empty"))?;
        if encoder_shape.len() != 3
            || encoder_shape[0] != 1
            || encoder_shape[1] != ENCODER_DIM as i64
        {
            bail!("unexpected fused TDT encoder output shape: {encoder_shape:?}");
        }
        let all_frames =
            usize::try_from(encoder_shape[2]).context("invalid fused TDT encoder frames")?;
        let frames = usize::try_from(encoded_length).context("invalid fused TDT encoded length")?;
        if frames == 0 || frames > all_frames {
            bail!("fused TDT encoded length {frames} exceeds output frames {all_frames}");
        }
        let mut trimmed = vec![0.0; ENCODER_DIM * frames];
        for feature in 0..ENCODER_DIM {
            trimmed[feature * frames..(feature + 1) * frames].copy_from_slice(
                &encoder_values[feature * all_frames..feature * all_frames + frames],
            );
        }
        drop(encoder_outputs);

        let mut network = OrtFusedTdtNetwork {
            decoder_joint: &mut self.decoder_joint,
        };
        let hypothesis = greedy_fused_tdt(&mut network, &trimmed, frames, ENCODER_DIM, BLANK_ID)?;
        let token_texts = hypothesis
            .token_ids
            .iter()
            .map(|&id| {
                self.tokens
                    .get(id)
                    .map(|token| token.replace('▁', " "))
                    .ok_or_else(|| anyhow!("fused TDT decoder emitted unknown token id {id}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let text = token_texts.concat();
        let timestamps = hypothesis
            .timestamps
            .iter()
            .map(|&frame| frame as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
            .collect::<Vec<_>>();
        let durations = hypothesis
            .durations
            .iter()
            .map(|&frames| frames as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
            .collect::<Vec<_>>();
        Ok(AsrTranscript::from_parts(
            text,
            token_texts,
            Some(&timestamps),
            Some(&durations),
        ))
    }
}

/// Greedy CTC decoding of the onnx-asr Japanese Parakeet CTC export.
///
/// The graph contract matches the production sherpa CTC export
/// (`audio_signal`/`length` in, `logprobs [1, T, 3073]` out) but carries no
/// sherpa metadata, so the production backend's metadata pins cannot apply.
/// Running this next to [`FusedTdtJaOrtEngine`] compares both decoder branches
/// through one export/quantization pipeline.
pub struct OnnxAsrCtcJaOrtEngine {
    session: Session,
    frontend: NemoMelFrontend,
    tokens: Vec<String>,
}

impl OnnxAsrCtcJaOrtEngine {
    /// Loads the onnx-asr CTC export with greedy decoding.
    ///
    /// # Errors
    ///
    /// Returns an error for missing or invalid model artifacts, or an ONNX
    /// Runtime session initialization failure.
    pub fn new(model_dir: &Path, num_threads: i32) -> Result<Self> {
        if num_threads <= 0 {
            bail!("ASR thread count must be greater than zero");
        }
        let tokens = load_vocab(&model_dir.join(VOCAB_FILE))?;

        init_onnx_runtime();
        let threads = usize::try_from(num_threads).context("invalid ASR thread count")?;
        let ctc_path = resolve_model_file(model_dir, CTC_MODEL_FILES, "onnx-asr CTC")?;
        let session = load_session(&ctc_path, threads, "onnx-asr CTC")?;
        validate_names(&session, &["audio_signal", "length"], &["logprobs"], "CTC")?;

        Ok(Self {
            session,
            frontend: NemoMelFrontend::new(),
            tokens,
        })
    }
}

impl AsrEngine for OnnxAsrCtcJaOrtEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let features = self.frontend.process(samples)?;
        let frames = i64::try_from(features.frames).context("too many CTC feature frames")?;
        let valid_frames =
            i64::try_from(features.valid_frames).context("too many valid CTC feature frames")?;
        let audio_signal =
            Tensor::from_array((vec![1_i64, MEL_BINS as i64, frames], features.values))?;
        let length = Tensor::from_array((vec![1_i64], vec![valid_frames]))?;
        let outputs = self.session.run(ort::inputs![
            "audio_signal" => audio_signal,
            "length" => length,
        ])?;
        let (shape, log_probs) = outputs
            .get("logprobs")
            .ok_or_else(|| anyhow!("onnx-asr CTC model did not return logprobs"))?
            .try_extract_tensor::<f32>()
            .context("failed to extract onnx-asr CTC logprobs")?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] != VOCAB_SIZE as i64 {
            bail!("unexpected onnx-asr CTC output shape: {shape:?}");
        }
        let output_frames = usize::try_from(shape[1]).context("invalid CTC output frame count")?;
        let path = decode_ctc(
            log_probs,
            output_frames,
            VOCAB_SIZE,
            BLANK_ID,
            CtcDecodingStrategy::Greedy,
        )?;
        transcript_from_path(
            &path,
            &self.tokens,
            f64::from(FEATURE_FRAME_SHIFT_SEC) * f64::from(SUBSAMPLING_FACTOR),
        )
    }
}

fn extract_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let (_, values) = outputs
        .get(name)
        .ok_or_else(|| anyhow!("ONNX session did not return {name}"))?
        .try_extract_tensor::<f32>()
        .with_context(|| format!("failed to extract ONNX output {name}"))?;
    if values.len() != expected_elements {
        bail!(
            "ONNX output {name} has {} elements, expected {expected_elements}",
            values.len()
        );
    }
    Ok(values.to_vec())
}

fn resolve_model_file(model_dir: &Path, candidates: &[&str], label: &str) -> Result<PathBuf> {
    candidates
        .iter()
        .map(|name| model_dir.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| {
            anyhow!(
                "{label} not found in {}: tried {candidates:?}",
                model_dir.display()
            )
        })
}

fn load_session(path: &Path, threads: usize, label: &str) -> Result<Session> {
    if !path.is_file() {
        bail!("{label} file not found: {}", path.display());
    }
    Session::builder()
        .map_err(|error| anyhow!("failed to create {label} session builder: {error}"))?
        .with_intra_threads(threads)
        .map_err(|error| anyhow!("failed to configure {label} intra-op threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow!("failed to configure {label} inter-op threads: {error}"))?
        .commit_from_file(path)
        .map_err(|error| anyhow!("failed to load {label} {}: {error}", path.display()))
}

fn load_vocab(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read vocabulary: {}", path.display()))?;
    let mut tokens = vec![None; VOCAB_SIZE];
    for (line_index, line) in contents.lines().enumerate() {
        let (token, id) = line.rsplit_once(' ').ok_or_else(|| {
            anyhow!(
                "invalid vocabulary line {} in {}",
                line_index + 1,
                path.display()
            )
        })?;
        let id = id.parse::<usize>().with_context(|| {
            format!(
                "invalid vocabulary token id on line {} in {}",
                line_index + 1,
                path.display()
            )
        })?;
        let slot = tokens
            .get_mut(id)
            .ok_or_else(|| anyhow!("vocabulary token id {id} exceeds model vocabulary"))?;
        if slot.replace(token.to_string()).is_some() {
            bail!("duplicate vocabulary token id {id}");
        }
    }
    if tokens.iter().any(Option::is_none) {
        bail!("vocabulary must contain contiguous ids through blank {BLANK_ID}");
    }
    let tokens = tokens.into_iter().flatten().collect::<Vec<_>>();
    if tokens[BLANK_ID] != "<blk>" {
        bail!("vocabulary blank token must be <blk> at id {BLANK_ID}");
    }
    Ok(tokens)
}

fn validate_names(
    session: &Session,
    expected_inputs: &[&str],
    expected_outputs: &[&str],
    label: &str,
) -> Result<()> {
    let inputs = session.inputs();
    let outputs = session.outputs();
    if inputs
        .iter()
        .map(ort::value::Outlet::name)
        .collect::<Vec<_>>()
        != expected_inputs
        || outputs
            .iter()
            .map(ort::value::Outlet::name)
            .collect::<Vec<_>>()
            != expected_outputs
    {
        bail!("onnx-asr {label} I/O contract changed");
    }
    for input in inputs {
        if !matches!(input.dtype(), ValueType::Tensor { .. }) {
            bail!("onnx-asr {label} input {} is not a tensor", input.name());
        }
    }
    for output in outputs {
        if !matches!(output.dtype(), ValueType::Tensor { .. }) {
            bail!("onnx-asr {label} output {} is not a tensor", output.name());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::Path, sync::Arc};

    use crate::asr::decoder::{
        hotword::{HotwordContextGraph, HotwordPathKind, HotwordTokenPath},
        tdt::NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP,
    };
    use anyhow::Result;

    use super::{
        BLANK_ID, FNV_OFFSET_BASIS, FusedTdtDagConfig, FusedTdtDagMergeAblation,
        FusedTdtDagMergeOrder, FusedTdtDagNode, FusedTdtDurationExpansion,
        FusedTdtHotwordCandidatePolicy, FusedTdtHypothesis, FusedTdtStep,
        HybridParakeetJaOrtEngine, ParakeetJaCtcOutput, ParakeetJaDecodingStrategy, VOCAB_SIZE,
        ctc_gate_hotword_paths, ctc_local_keyword_score, greedy_fused_tdt,
        merge_and_prune_dag_nodes, select_ctc_anchor_candidate,
        select_tdt_static_embedding_candidate, tdt_duration_candidates,
        tdt_hypothesis_to_transcript_with_hotwords, tdt_log_softmax,
        tokenize_parakeet_hotword_entries, tokenize_parakeet_hotword_text, uses_ctc_hotword_gate,
        variable_width_dag_fused_tdt, variable_width_dag_fused_tdt_with_hotword_policy,
        variable_width_dag_fused_tdt_with_hotwords,
    };
    use crate::asr::backend::HotwordEntry;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct State(usize);

    struct ScriptedNetwork {
        logits: Vec<Vec<f32>>,
        calls: usize,
    }

    impl FusedTdtStep for ScriptedNetwork {
        type State = State;

        fn initial_state(&self) -> Self::State {
            State::default()
        }

        fn step(
            &mut self,
            _encoder_frame: &[f32],
            _token: usize,
            state: &Self::State,
        ) -> Result<(Vec<f32>, Self::State)> {
            let value = self.logits[self.calls.min(self.logits.len() - 1)].clone();
            self.calls += 1;
            Ok((value, State(state.0 + 1)))
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
        let decoded = greedy_fused_tdt(&mut network, &[0.0], 1, 1, 2).unwrap();

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
        let decoded = greedy_fused_tdt(&mut network, &[0.0, 0.0, 0.0], 3, 1, 2).unwrap();
        assert_eq!(decoded.token_ids, vec![1]);
        assert_eq!(decoded.timestamps, vec![0]);
        assert_eq!(decoded.durations, vec![2]);
        assert_eq!(decoded.last_frame, 3);
        assert_eq!(decoded.state, State(1));
    }

    #[test]
    fn greedy_rejects_a_mismatched_joint_width() {
        struct WrongWidth;
        impl FusedTdtStep for WrongWidth {
            type State = ();

            fn initial_state(&self) -> Self::State {}

            fn step(
                &mut self,
                _encoder_frame: &[f32],
                _token: usize,
                (): &Self::State,
            ) -> Result<(Vec<f32>, Self::State)> {
                Ok((vec![0.0; 4], ()))
            }
        }
        let error = greedy_fused_tdt(&mut WrongWidth, &[0.0], 1, 1, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected 3 token and 5 duration logits"));
    }

    #[test]
    fn dag_keeps_every_positive_duration_instead_of_duration_argmax() {
        let mut values = vec![-100.0; 3 + 5];
        values[0] = 5.0;
        values[1] = -100.0;
        values[2] = -100.0;
        values[3] = -10.0;
        values[4] = 4.0;
        values[5] = 3.9;
        values[6] = 3.8;
        values[7] = 3.7;
        let mut network = ScriptedNetwork {
            logits: vec![values],
            calls: 0,
        };

        let result = variable_width_dag_fused_tdt(
            &mut network,
            &[0.0],
            1,
            1,
            2,
            FusedTdtDagConfig {
                beam_size: 4,
                max_symbols_per_step: 10,
                duration_expansion: FusedTdtDurationExpansion::All,
            },
        )
        .unwrap();

        assert_eq!(
            result
                .hypotheses
                .iter()
                .map(|hypothesis| hypothesis.durations.as_slice())
                .collect::<Vec<_>>(),
            vec![&[1][..], &[2][..], &[3][..], &[4][..]]
        );
        assert!(
            result
                .hypotheses
                .iter()
                .all(|hypothesis| hypothesis.token_ids == [0] && hypothesis.timestamps == [0])
        );
    }

    #[test]
    fn dag_hotword_continuation_can_recover_a_token_below_acoustic_top_k() {
        let mut values = vec![-10.0; 3 + 5];
        values[0] = 3.0;
        values[1] = 0.0;
        values[2] = -10.0;
        values[3 + 1] = 5.0;
        let config = FusedTdtDagConfig {
            beam_size: 1,
            max_symbols_per_step: 10,
            duration_expansion: FusedTdtDurationExpansion::Argmax,
        };
        let mut baseline_network = ScriptedNetwork {
            logits: vec![values.clone()],
            calls: 0,
        };
        let baseline =
            variable_width_dag_fused_tdt(&mut baseline_network, &[0.0], 1, 1, 2, config).unwrap();
        assert_eq!(baseline.hypotheses[0].token_ids, [0]);

        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![HotwordTokenPath {
                tokens: vec![1],
                entry_id: 0,
                surface: "語".to_string(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            }],
            100.0,
        )
        .unwrap();
        let mut hotword_network = ScriptedNetwork {
            logits: vec![values],
            calls: 0,
        };
        let biased = variable_width_dag_fused_tdt_with_hotwords(
            &mut hotword_network,
            &[0.0],
            1,
            1,
            2,
            config,
            &graph,
        )
        .unwrap();
        assert_eq!(biased.hypotheses[0].token_ids, [1]);
    }

    #[test]
    fn completed_reading_path_is_rendered_as_the_registered_kanji_surface() {
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 0,
                surface: "国頭".to_owned(),
                kind: HotwordPathKind::Reading,
                phrase_score: None,
            }],
            1.0,
        )
        .unwrap();
        let hypothesis = FusedTdtHypothesis {
            score: 0.0,
            token_ids: vec![3, 1, 2, 4],
            timestamps: vec![0, 5, 6, 8],
            durations: vec![1, 1, 2, 1],
            state: State::default(),
            last_frame: 9,
        };

        let transcript = tdt_hypothesis_to_transcript_with_hotwords(
            &["", "くに", "がみ", "沖縄県", "村"].map(str::to_owned),
            &hypothesis,
            &graph,
        )
        .unwrap();

        assert_eq!(transcript.text, "沖縄県国頭村");
        assert_eq!(
            transcript
                .tokens
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["沖縄県", "国頭", "村"]
        );
        assert!((transcript.tokens[1].start_sec.unwrap() - 0.4).abs() < 1.0e-6);
        assert!((transcript.tokens[1].duration_sec.unwrap() - 0.24).abs() < 1.0e-6);
    }

    #[test]
    fn direct_pre_topk_recovers_a_hotword_without_expanding_the_retained_width() {
        let mut values = vec![-10.0; 4 + 5];
        values[0] = 2.0;
        values[1] = 0.0;
        values[3] = 3.0;
        values[4 + 1] = 5.0;
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![HotwordTokenPath {
                tokens: vec![1],
                entry_id: 0,
                surface: "固有名詞".to_owned(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            }],
            100.0,
        )
        .unwrap();
        let mut network = ScriptedNetwork {
            logits: vec![values],
            calls: 0,
        };

        let result = variable_width_dag_fused_tdt_with_hotword_policy(
            &mut network,
            &[0.0],
            1,
            1,
            3,
            FusedTdtDagConfig {
                beam_size: 1,
                max_symbols_per_step: 10,
                duration_expansion: FusedTdtDurationExpansion::Argmax,
            },
            &graph,
            FusedTdtHotwordCandidatePolicy::DirectPreTopK { acoustic_top_k: 2 },
        )
        .unwrap();

        assert_eq!(result.hypotheses[0].token_ids, [1]);
        assert_eq!(result.stats.generated_candidates, 1);
        assert_eq!(result.stats.max_active_width, 1);
    }

    #[test]
    fn direct_pre_topk_injects_only_paths_with_multiplier_above_one() {
        let path = |token, surface: &str, phrase_score| HotwordTokenPath {
            tokens: vec![token],
            entry_id: token,
            surface: surface.to_owned(),
            kind: HotwordPathKind::Surface,
            phrase_score,
        };
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![
                path(0, "neutral", Some(1.0)),
                path(1, "suppressed", Some(0.5)),
                path(2, "boosted", None),
            ],
            100.0,
        )
        .unwrap();
        assert_eq!(
            graph
                .direct_continuation_tokens(graph.root())
                .collect::<Vec<_>>(),
            vec![2]
        );
        let mut values = vec![-10.0; 4 + 5];
        values[0] = -2.0;
        values[1] = -3.0;
        values[2] = -1.0;
        values[3] = 0.0;
        values[4 + 1] = 5.0;
        let mut network = ScriptedNetwork {
            logits: vec![values],
            calls: 0,
        };

        let result = variable_width_dag_fused_tdt_with_hotword_policy(
            &mut network,
            &[0.0],
            1,
            1,
            3,
            FusedTdtDagConfig {
                beam_size: 1,
                max_symbols_per_step: 10,
                duration_expansion: FusedTdtDurationExpansion::Argmax,
            },
            &graph,
            FusedTdtHotwordCandidatePolicy::DirectPreTopK { acoustic_top_k: 1 },
        )
        .unwrap();

        let emitted = result
            .hypotheses
            .iter()
            .filter_map(|hypothesis| hypothesis.token_ids.first().copied())
            .collect::<Vec<_>>();
        assert_eq!(emitted, vec![2]);
        assert_eq!(result.stats.generated_candidates, 1);
    }

    #[test]
    fn dag_removes_an_incomplete_hotword_prefix_bonus_at_utterance_end() {
        let mut values = vec![-10.0; 3 + 5];
        values[0] = 3.0;
        values[1] = 0.0;
        values[2] = -10.0;
        values[3 + 1] = 5.0;
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            vec![HotwordTokenPath {
                tokens: vec![1, 0],
                entry_id: 0,
                surface: "word".into(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            }],
            100.0,
        )
        .unwrap();
        let mut network = ScriptedNetwork {
            logits: vec![values],
            calls: 0,
        };
        let result = variable_width_dag_fused_tdt_with_hotwords(
            &mut network,
            &[0.0],
            1,
            1,
            2,
            FusedTdtDagConfig {
                beam_size: 2,
                max_symbols_per_step: 10,
                duration_expansion: FusedTdtDurationExpansion::Argmax,
            },
            &graph,
        )
        .unwrap();

        let incomplete = result
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.token_ids == [1])
            .unwrap();
        let expected_without_hotword = tdt_log_softmax(&[3.0, 0.0, -10.0])[1];
        assert!((incomplete.score - expected_without_hotword).abs() < 1.0e-5);
    }

    #[test]
    fn dag_merges_after_pruning_and_does_not_refill_the_freed_slot() {
        let node = |score, sequence_hash| FusedTdtDagNode {
            score,
            sequence_hash,
            emitted_count: 1,
            last_token: Some(0),
            state: Arc::new(State::default()),
            frame: 1,
            history: None,
            symbols_since_advance: 0,
            hotword_state: 0,
        };
        let merged = merge_and_prune_dag_nodes(
            vec![
                node(-1.0, FNV_OFFSET_BASIS),
                node(-2.0, FNV_OFFSET_BASIS),
                node(-3.0, FNV_OFFSET_BASIS + 1),
            ],
            2,
            FusedTdtDagMergeAblation::default(),
            None,
        );

        assert_eq!(merged.len(), 1, "merge must not refill from the third path");
        let expected = -1.0 + (-1.0_f32).exp().ln_1p();
        assert!((merged[0].score - expected).abs() < 1.0e-6);
    }

    #[test]
    fn dag_merge_then_prune_refills_a_slot_freed_by_duplicate_paths() {
        let node = |score, sequence_hash| FusedTdtDagNode {
            score,
            sequence_hash,
            emitted_count: 1,
            last_token: Some(0),
            state: Arc::new(State::default()),
            frame: 1,
            history: None,
            symbols_since_advance: 0,
            hotword_state: 0,
        };
        let merged = merge_and_prune_dag_nodes(
            vec![
                node(-1.0, FNV_OFFSET_BASIS),
                node(-2.0, FNV_OFFSET_BASIS),
                node(-3.0, FNV_OFFSET_BASIS + 1),
            ],
            2,
            FusedTdtDagMergeAblation {
                order: FusedTdtDagMergeOrder::MergeThenPrune,
                ..FusedTdtDagMergeAblation::default()
            },
            None,
        );

        assert_eq!(
            merged.len(),
            2,
            "the third path must refill the merged slot"
        );
        assert!(
            merged
                .iter()
                .any(|node| node.sequence_hash == FNV_OFFSET_BASIS + 1)
        );
    }

    #[test]
    fn dag_merge_key_can_keep_distinct_zero_duration_budgets() {
        let node = |score, symbols_since_advance| FusedTdtDagNode {
            score,
            sequence_hash: FNV_OFFSET_BASIS,
            emitted_count: 2,
            last_token: Some(1),
            state: Arc::new(State::default()),
            frame: 1,
            history: None,
            symbols_since_advance,
            hotword_state: 0,
        };
        let merged = merge_and_prune_dag_nodes(
            vec![node(-1.0, 1), node(-2.0, 9)],
            2,
            FusedTdtDagMergeAblation {
                include_symbols_since_advance_in_key: true,
                ..FusedTdtDagMergeAblation::default()
            },
            None,
        );

        assert_eq!(
            merged
                .iter()
                .map(|node| node.symbols_since_advance)
                .collect::<Vec<_>>(),
            [1, 9]
        );
    }

    #[test]
    fn duration_argmax_keeps_one_duration_and_blank_falls_back_to_best_positive() {
        let log_probs = vec![0.0, -3.0, -2.0, -1.0, -4.0];

        assert_eq!(
            tdt_duration_candidates(&log_probs, FusedTdtDurationExpansion::Argmax, true),
            vec![(0, 0.0)]
        );
        assert_eq!(
            tdt_duration_candidates(&log_probs, FusedTdtDurationExpansion::Argmax, false),
            vec![(3, -1.0)]
        );
        assert_eq!(
            tdt_duration_candidates(&log_probs, FusedTdtDurationExpansion::All, false),
            vec![(1, -3.0), (2, -2.0), (3, -1.0), (4, -4.0)]
        );
    }

    #[test]
    fn static_embedding_can_select_a_close_lower_acoustic_candidate() {
        let candidate = |score, token| FusedTdtHypothesis {
            score,
            token_ids: vec![token],
            timestamps: vec![0],
            durations: vec![1],
            state: State::default(),
            last_frame: 1,
        };
        let hypotheses = [candidate(-1.0, 0), candidate(-1.1, 1)];

        assert_eq!(
            select_tdt_static_embedding_candidate(&hypotheses, &[0.0, 1.0]).unwrap(),
            1
        );
        assert_eq!(
            select_tdt_static_embedding_candidate(&hypotheses, &[0.5, 0.5]).unwrap(),
            0
        );
    }

    #[test]
    fn ctc_keyword_gate_requires_local_order_and_a_blank_between_repeated_tokens() {
        let frames = 5;
        let vocab = 4;
        let blank = 3;
        let mut posteriors = vec![-5.0; frames * vocab];
        for frame in 0..frames {
            posteriors[frame * vocab + 2] = 0.0;
        }
        posteriors[vocab] = -0.1;
        posteriors[2 * vocab + blank] = -0.2;
        posteriors[3 * vocab + 1] = -0.1;
        posteriors[3 * vocab] = -0.1;

        let ordered =
            ctc_local_keyword_score(&posteriors, frames, vocab, blank, &[0, 1], 3).unwrap();
        let reversed =
            ctc_local_keyword_score(&posteriors, frames, vocab, blank, &[1, 0], 3).unwrap();
        let too_distant =
            ctc_local_keyword_score(&posteriors, frames, vocab, blank, &[0, 1], 1).unwrap();
        let repeated =
            ctc_local_keyword_score(&posteriors, frames, vocab, blank, &[0, 0], 3).unwrap();

        assert!(ordered > reversed);
        assert!(ordered > too_distant);
        assert!((repeated - -0.2).abs() < 1.0e-6);
    }

    #[test]
    fn ctc_anchor_similarity_can_select_a_close_second_best_path() {
        assert_eq!(
            select_ctc_anchor_candidate(&[-1.0, -1.1], &[0.0, 1.0], 0.1).unwrap(),
            1
        );
        assert_eq!(
            select_ctc_anchor_candidate(&[-1.0, -1.1], &[0.5, 0.5], 1.0).unwrap(),
            0
        );
        assert_eq!(
            select_ctc_anchor_candidate(&[-1.0, -1.1], &[0.0, 1.0], 0.0).unwrap(),
            0
        );
    }

    #[test]
    fn ctc_gate_retains_all_aliases_of_an_acoustically_supported_entry() {
        let mut values = vec![-5.0; 3 * VOCAB_SIZE];
        for frame in 0..3 {
            values[frame * VOCAB_SIZE + BLANK_ID] = 0.0;
        }
        values[10] = -0.1;
        values[VOCAB_SIZE + 11] = -0.1;
        let output = ParakeetJaCtcOutput { values, frames: 3 };
        let paths = vec![
            HotwordTokenPath {
                tokens: vec![10, 11],
                entry_id: 0,
                surface: "supported".into(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            },
            HotwordTokenPath {
                tokens: vec![12, 13],
                entry_id: 0,
                surface: "supported".into(),
                kind: HotwordPathKind::Reading,
                phrase_score: None,
            },
            HotwordTokenPath {
                tokens: vec![20, 21],
                entry_id: 1,
                surface: "unsupported".into(),
                kind: HotwordPathKind::Surface,
                phrase_score: None,
            },
        ];

        let retained = ctc_gate_hotword_paths(&output, &paths, -1.0, 2).unwrap();

        assert_eq!(retained, paths[..2]);
    }

    #[test]
    fn dag_rejects_zero_width_and_zero_duration_zero_guard() {
        let mut network = ScriptedNetwork {
            logits: vec![logits(0, 1)],
            calls: 0,
        };
        let zero_width = variable_width_dag_fused_tdt(
            &mut network,
            &[0.0],
            1,
            1,
            2,
            FusedTdtDagConfig {
                beam_size: 0,
                max_symbols_per_step: 10,
                duration_expansion: FusedTdtDurationExpansion::All,
            },
        )
        .unwrap_err();
        let zero_guard = variable_width_dag_fused_tdt(
            &mut network,
            &[0.0],
            1,
            1,
            2,
            FusedTdtDagConfig {
                beam_size: 2,
                max_symbols_per_step: 0,
                duration_expansion: FusedTdtDurationExpansion::All,
            },
        )
        .unwrap_err();

        assert!(
            zero_width
                .to_string()
                .contains("beam size must be positive")
        );
        assert!(
            zero_guard
                .to_string()
                .contains("max symbols per step must be positive")
        );
    }

    #[test]
    fn hybrid_rejects_encoder_and_decoder_thread_counts_before_loading_artifacts() {
        let missing = Path::new("missing-parakeet-ja-model");
        let encoder_error =
            HybridParakeetJaOrtEngine::new(missing, 0, 1, ParakeetJaDecodingStrategy::CtcGreedy)
                .err()
                .expect("zero encoder threads must be rejected")
                .to_string();
        let decoder_error =
            HybridParakeetJaOrtEngine::new(missing, 1, 0, ParakeetJaDecodingStrategy::TdtGreedy)
                .err()
                .expect("zero decoder threads must be rejected")
                .to_string();

        assert!(encoder_error.contains("encoder thread count must be greater than zero"));
        assert!(decoder_error.contains("decoder thread count must be greater than zero"));
    }

    #[test]
    fn ctc_hotword_gate_is_not_required_without_an_enabled_hotword() {
        assert!(!uses_ctc_hotword_gate(0, Some(-5.0)));
        assert!(!uses_ctc_hotword_gate(1, None));
        assert!(uses_ctc_hotword_gate(1, Some(-5.0)));
    }

    #[test]
    fn hotword_tokenization_preserves_surface_and_normalized_reading_paths() {
        let tokens = ["斎", "藤", "さ", "い", "と", "う", "サ", "イ", "ト", "ウ"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let paths = tokenize_parakeet_hotword_entries(
            &[HotwordEntry {
                surface: "斎藤".into(),
                readings: vec!["サイﾄウ".into()],
                phrase_score: None,
            }],
            &tokens,
        )
        .unwrap();

        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0].tokens, vec![0, 1]);
        assert_eq!(paths[1].tokens, vec![2, 3, 4, 5]);
        assert_eq!(paths[2].tokens, vec![6, 7, 8, 9]);
    }

    #[test]
    fn hotword_tokenization_keeps_character_and_multi_character_piece_paths() {
        let mut tokens = (0..424)
            .map(|id| format!("unused-{id}"))
            .collect::<Vec<_>>();
        tokens[10] = "か".into();
        tokens[11] = "な".into();
        tokens[174] = "かな".into();
        tokens[423] = "ざ".into();
        tokens[52] = "わ".into();
        tokens[20] = "カ".into();
        tokens[21] = "ナ".into();
        tokens[22] = "ザ".into();
        tokens[23] = "ワ".into();
        tokens[175] = "カナ".into();
        let paths = tokenize_parakeet_hotword_entries(
            &[HotwordEntry {
                surface: "かなざわ".into(),
                readings: vec!["かなざわ".into()],
                phrase_score: None,
            }],
            &tokens,
        )
        .unwrap();

        let surface_paths = paths
            .iter()
            .filter(|path| path.kind == HotwordPathKind::Surface)
            .map(|path| path.tokens.as_slice())
            .collect::<Vec<_>>();
        let reading_paths = paths
            .iter()
            .filter(|path| path.kind == HotwordPathKind::Reading)
            .map(|path| path.tokens.as_slice())
            .collect::<Vec<_>>();
        assert!(surface_paths.contains(&[10, 11, 423, 52].as_slice()));
        assert!(surface_paths.contains(&[174, 423, 52].as_slice()));
        assert!(reading_paths.contains(&[174, 423, 52].as_slice()));
        assert!(reading_paths.contains(&[175, 22, 23].as_slice()));
    }

    #[test]
    fn hotword_piece_segmentation_is_bounded_and_deterministic() {
        let tokens = ["あ", "ああ", "あああ", "ああああ"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let token_ids = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_str(), id))
            .collect::<HashMap<_, _>>();

        let first =
            tokenize_parakeet_hotword_text("ああああああああ", &tokens, &token_ids, "surface")
                .unwrap();
        let second =
            tokenize_parakeet_hotword_text("ああああああああ", &tokens, &token_ids, "surface")
                .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.len(), 8);
        assert_eq!(first[0], vec![0; 8]);
    }

    #[test]
    fn sentencepiece_boundary_piece_is_admitted_only_at_hotword_start() {
        // The pinned vocabulary contains both marker-free Japanese pieces
        // (`かな 174`) and word-boundary pieces (`▁この 114`). The latter may
        // begin a hotword but must not be injected into its textual interior.
        let tokens = ["あ", "い", "▁い"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let token_ids = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_str(), id))
            .collect::<HashMap<_, _>>();

        let internal =
            tokenize_parakeet_hotword_text("あい", &tokens, &token_ids, "surface").unwrap();
        let at_start =
            tokenize_parakeet_hotword_text("い", &tokens, &token_ids, "surface").unwrap();

        assert!(!internal.contains(&vec![0, 2]));
        assert!(at_start.contains(&vec![2]));
    }
}
