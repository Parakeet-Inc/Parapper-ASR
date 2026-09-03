use std::{
    collections::BinaryHeap,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};

use super::hotword::HotwordContextGraph;

pub trait RnntNetwork {
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
pub struct RnntHypothesis<S> {
    pub token_ids: Vec<usize>,
    pub timestamps: Vec<usize>,
    pub state: S,
    pub last_token: Option<usize>,
    pub frame_offset: usize,
}

/// A frame-synchronous hypothesis for a stateless two-token RNN-T predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct StatelessRnntHypothesis {
    pub score: f32,
    pub token_ids: Vec<usize>,
    pub timestamps: Vec<usize>,
}

/// Accumulated wall-clock timings and operation counts for stateless RNN-T search.
///
/// The ordinary decoder does not read the clock. Call the explicitly profiled
/// entry point only from diagnostics so production inference keeps its original
/// hot path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatelessRnntSearchProfile {
    pub frames: usize,
    pub active_hypotheses: usize,
    pub network_context_rows: usize,
    pub logit_values: usize,
    pub network_output_values: usize,
    pub network_output_bytes: usize,
    pub scalar_exp_terms_evaluated: usize,
    pub scalar_exp_terms_skipped: usize,
    pub candidates_generated: usize,
    pub states_after_dominance: usize,
    pub context_layout: Duration,
    pub network_logits: Duration,
    pub log_softmax: Duration,
    pub top_token_selection: Duration,
    pub candidate_generation: Duration,
    pub state_dominance: Duration,
    pub score_pruning: Duration,
    pub materialization: Duration,
    pub final_ranking_and_reconstruction: Duration,
}

/// Compact log-probability data returned by a network-side TopK/Gather path.
#[derive(Debug, Clone, PartialEq)]
pub struct StatelessRnntCompactScores {
    /// Top emitting tokens for each unique predictor context row.
    pub top_tokens: Vec<Vec<(usize, f32)>>,
    /// Scores parallel to the requested token IDs for each row.
    pub requested_scores: Vec<Vec<f32>>,
    /// Total scalar values and indices transferred from the network session.
    pub network_output_values: usize,
    /// Total bytes transferred from the network session.
    pub network_output_bytes: usize,
}

/// Selects the bounded top-token data structure for diagnostic ablation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StatelessRnntTopTokenAlgorithm {
    /// Search the sorted top list for every vocabulary item.
    BinarySearch,
    /// Reject items below the current floor before searching the sorted list.
    #[default]
    CutoffBinarySearch,
    /// Keep the current worst retained item at a binary-heap root and replace it.
    WorstFirstHeap,
}

/// Selects where joiner logits become log probabilities.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StatelessRnntScoreNormalization {
    /// Normalize and overwrite every vocabulary value inside the search.
    FullLogSoftmax,
    /// Retain raw logits and subtract one scalar normalizer only on access.
    #[default]
    SparseLogNormalizer,
    /// Skip terms whose `exp(logit - row_max)` is below `1e-8`.
    SparseLogNormalizerEpsilon1e8,
    /// Skip terms whose `exp(logit - row_max)` is below `1e-7`.
    SparseLogNormalizerEpsilon1e7,
    /// Skip terms whose `exp(logit - row_max)` is below `1e-6`.
    SparseLogNormalizerEpsilon1e6,
    /// The network already returned one log-softmax row per context.
    PrecomputedLogProbabilities,
}

/// Controls which hypothesis identity owns one stateless RNN-T beam slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StatelessRnntPruning {
    /// Keep one acoustic top-1 history for each `(token count, last two tokens)` state.
    #[default]
    StateDominance,
    /// Merge exact full prefixes first, then refill the beam to its requested width.
    FullPrefix,
    /// Select global top-K paths first and merge exact prefixes without refilling, as sherpa does.
    Sherpa,
}

/// Length normalization applied when ranking partial or final hypotheses.
///
/// The initial two-token predictor context remains part of the denominator so
/// `PerToken` preserves the icefall/sherpa final-ranking convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StatelessRnntLengthNormalization {
    #[default]
    Raw,
    Quarter,
    Half,
    ThreeQuarters,
    PerToken,
}

impl StatelessRnntLengthNormalization {
    #[must_use]
    pub const fn exponent_quarters(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::Quarter => 1,
            Self::Half => 2,
            Self::ThreeQuarters => 3,
            Self::PerToken => 4,
        }
    }
}

/// Diagnostic controls for the stateless RNN-T modified beam search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatelessRnntBeamConfig {
    pub beam_size: usize,
    pub initial_context: [i64; 2],
    pub additional_non_emitting_id: Option<usize>,
    pub pruning: StatelessRnntPruning,
    pub deduplicate_contexts: bool,
    pub search_length_normalization: StatelessRnntLengthNormalization,
    pub final_length_normalization: StatelessRnntLengthNormalization,
}

impl StatelessRnntBeamConfig {
    /// The contract used by the `ReazonSpeech` ONNX-producing reference.
    #[must_use]
    pub fn model_reference(beam_size: usize, blank_id: usize) -> Self {
        let blank = token_context_id(blank_id);
        Self {
            beam_size,
            initial_context: [blank, blank],
            additional_non_emitting_id: None,
            pruning: StatelessRnntPruning::StateDominance,
            deduplicate_contexts: true,
            search_length_normalization: StatelessRnntLengthNormalization::Raw,
            final_length_normalization: StatelessRnntLengthNormalization::PerToken,
        }
    }
}

fn token_context_id(token_id: usize) -> i64 {
    i64::try_from(token_id).unwrap_or(i64::MAX)
}

#[derive(Debug, Clone, Copy)]
struct ActiveStatelessHypothesis {
    score: f32,
    prefix_id: usize,
    alignment_id: usize,
    hotword_state: usize,
}

#[derive(Debug)]
struct TokenPrefixNode {
    parent: Option<usize>,
    token_id: Option<usize>,
    token_count: usize,
    context: [i64; 2],
    children: Vec<(usize, usize)>,
}

#[derive(Debug)]
struct TokenPrefixArena {
    nodes: Vec<TokenPrefixNode>,
}

impl TokenPrefixArena {
    fn new(initial_context: [i64; 2]) -> Self {
        Self {
            nodes: vec![TokenPrefixNode {
                parent: None,
                token_id: None,
                token_count: 0,
                context: initial_context,
                children: Vec::new(),
            }],
        }
    }

    fn child(&self, parent: usize, token_id: usize) -> Option<usize> {
        self.nodes[parent]
            .children
            .iter()
            .find_map(|&(token, child)| (token == token_id).then_some(child))
    }

    fn intern_child(&mut self, parent: usize, token_id: usize) -> usize {
        if let Some(child) = self.child(parent, token_id) {
            return child;
        }
        let token_count = self.nodes[parent].token_count + 1;
        let context = [self.nodes[parent].context[1], token_context_id(token_id)];
        let child = self.nodes.len();
        self.nodes.push(TokenPrefixNode {
            parent: Some(parent),
            token_id: Some(token_id),
            token_count,
            context,
            children: Vec::new(),
        });
        self.nodes[parent].children.push((token_id, child));
        child
    }

    fn token_ids(&self, mut prefix_id: usize) -> Vec<usize> {
        let mut token_ids = Vec::with_capacity(self.nodes[prefix_id].token_count);
        while let Some(parent) = self.nodes[prefix_id].parent {
            token_ids.push(
                self.nodes[prefix_id]
                    .token_id
                    .expect("non-root prefix has a token"),
            );
            prefix_id = parent;
        }
        token_ids.reverse();
        token_ids
    }
}

#[derive(Debug)]
struct AlignmentNode {
    parent: Option<usize>,
    frame: Option<usize>,
}

#[derive(Debug)]
struct AlignmentArena {
    nodes: Vec<AlignmentNode>,
}

impl AlignmentArena {
    fn new() -> Self {
        Self {
            nodes: vec![AlignmentNode {
                parent: None,
                frame: None,
            }],
        }
    }

    fn push(&mut self, parent: usize, frame: usize) -> usize {
        let child = self.nodes.len();
        self.nodes.push(AlignmentNode {
            parent: Some(parent),
            frame: Some(frame),
        });
        child
    }

    fn timestamps(&self, mut alignment_id: usize) -> Vec<usize> {
        let mut timestamps = Vec::new();
        while let Some(parent) = self.nodes[alignment_id].parent {
            timestamps.push(
                self.nodes[alignment_id]
                    .frame
                    .expect("non-root alignment has a frame"),
            );
            alignment_id = parent;
        }
        timestamps.reverse();
        timestamps
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PendingPrefix {
    Existing(usize),
    New { parent: usize, token_id: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StatelessSearchState {
    token_count: usize,
    context: [i64; 2],
    hotword_state: usize,
}

#[derive(Debug, Clone, Copy)]
struct PendingStatelessHypothesis {
    prefix: PendingPrefix,
    state: StatelessSearchState,
    score: f32,
    best_path_score: f32,
    source_alignment_id: usize,
    emitted_frame: Option<usize>,
    source_index: usize,
    token_id: usize,
}

/// Scores a batch of two-token predictor contexts against one encoder frame.
///
/// Implementations should return frame-major logits with shape
/// `[contexts.len(), vocab_size]`. Batching is useful on CPU as it avoids one
/// ONNX Runtime call per active hypothesis; it does not assume a GPU provider.
pub trait StatelessRnntNetwork {
    /// Reports whether this network can serve the compact TopK/Gather contract.
    ///
    /// This avoids building dynamic gather requests on the ordinary dense
    /// production path.
    fn supports_compact_log_probabilities(&self) -> bool {
        false
    }

    /// Runs the predictor and joiner for every active context.
    ///
    /// # Errors
    ///
    /// Returns an error when model inference fails.
    fn logits(&mut self, encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>>;

    /// Optionally computes exact `TopK` and selected-token log probabilities
    /// without returning a full vocabulary row to the search implementation.
    ///
    /// `requested_token_ids` includes blank and every token needed for exact
    /// prefix recombination. The default keeps the dense-logit path.
    ///
    /// # Errors
    ///
    /// Returns an error when compact model inference fails.
    fn compact_log_probabilities(
        &mut self,
        _encoder_frame: &[f32],
        _contexts: &[[i64; 2]],
        _requested_token_ids: &[Vec<usize>],
        _emitting_limit: usize,
        _blank_id: usize,
        _additional_non_emitting_id: Option<usize>,
    ) -> Result<Option<StatelessRnntCompactScores>> {
        Ok(None)
    }
}

/// Supplies selected, already normalized token scores for fixed-sequence
/// alignment rescoring.
///
/// Unlike beam search, a fixed transcript only needs the blank score and the
/// score of its next token. Implementations can therefore keep the full
/// vocabulary inside the model runtime and return only requested columns.
pub trait StatelessRnntSequenceScoreNetwork {
    /// Returns one score row per context and one value per requested token ID.
    ///
    /// # Errors
    ///
    /// Returns an error when model inference fails or its selected-score
    /// output no longer matches the requested row layout.
    fn requested_log_probabilities(
        &mut self,
        encoder_frame: &[f32],
        contexts: &[[i64; 2]],
        requested_token_ids: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>>;
}

/// Exact scores for one fixed transcript in the frame-synchronous search
/// graph used by modified beam search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StatelessRnntSequenceScores {
    /// Maximum-score alignment over blank and token emission frames.
    pub viterbi_score: f32,
    /// Log-sum of every valid alignment in the one-symbol-per-frame graph.
    pub forward_score: f32,
}

/// Posterior timing statistics for one occurrence in a fixed transcript.
///
/// Occurrences are indexed by output position, not token ID or decoder
/// context. Repeated text such as `AAAAA` therefore retains five distinct,
/// monotonically ordered alignment variables.
#[derive(Debug, Clone, PartialEq)]
pub struct StatelessRnntTokenAlignment {
    /// Emission frame on the maximum-score complete alignment.
    pub viterbi_frame: usize,
    /// Earliest frame attaining the largest marginal emission probability.
    pub posterior_mode_frame: usize,
    /// Marginal posterior mean emission frame.
    pub expected_frame: f32,
    /// First frame whose cumulative marginal probability reaches 5%.
    pub posterior_lower_frame: usize,
    /// First frame whose cumulative marginal probability reaches 95%.
    pub posterior_upper_frame: usize,
    /// Entropy of the marginal frame distribution, in nats.
    pub entropy: f32,
}

/// Exact monotonic alignment result for one fixed transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct StatelessRnntSequenceAlignment {
    pub scores: StatelessRnntSequenceScores,
    /// One timing result per transcript token occurrence.
    pub tokens: Vec<StatelessRnntTokenAlignment>,
    /// Posterior probability that any transcript token is emitted per frame.
    pub frame_emission_probabilities: Vec<f32>,
}

/// One complete beam candidate and its representative emission alignment.
#[derive(Debug, Clone, Copy)]
pub struct StatelessRnntAlignedSeed<'a> {
    pub token_ids: &'a [usize],
    pub emission_frames: &'a [usize],
}

/// Chooses how occurrences projected onto the same time-free lattice arc are
/// combined.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StatelessRnntLatticeArcMerge {
    /// Keep the strongest representative segment. This is deliberately
    /// optimistic when adjacent arcs originated from incompatible frames.
    #[default]
    Maximum,
    /// Average occurrence probabilities without rewarding duplicate seeds.
    LogMeanExp,
}

/// One terminal path recovered from an approximate time-free lattice.
#[derive(Debug, Clone, PartialEq)]
pub struct StatelessRnntLatticeHypothesis {
    pub score: f32,
    pub token_ids: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ApproximateLatticeState {
    token_count: usize,
    context: [i64; 2],
}

#[derive(Debug, Clone)]
struct ApproximateLatticeArc {
    token_id: usize,
    destination: usize,
    scores: LatticeScoreAggregate,
}

#[derive(Debug, Clone)]
struct ApproximateLatticeNode {
    state: ApproximateLatticeState,
    arcs: Vec<ApproximateLatticeArc>,
    terminal_scores: LatticeScoreAggregate,
}

#[derive(Debug, Clone, Copy)]
struct LatticeScoreAggregate {
    maximum: f32,
    log_sum: f32,
    count: usize,
}

impl Default for LatticeScoreAggregate {
    fn default() -> Self {
        Self {
            maximum: f32::NEG_INFINITY,
            log_sum: f32::NEG_INFINITY,
            count: 0,
        }
    }
}

impl LatticeScoreAggregate {
    fn add(&mut self, score: f32) {
        self.maximum = self.maximum.max(score);
        self.log_sum = log_add_exp(self.log_sum, score);
        self.count += 1;
    }

    fn merged(self, policy: StatelessRnntLatticeArcMerge) -> Option<f32> {
        let count = u16::try_from(self.count).ok()?;
        (count > 0).then(|| match policy {
            StatelessRnntLatticeArcMerge::Maximum => self.maximum,
            StatelessRnntLatticeArcMerge::LogMeanExp => self.log_sum - f32::from(count).ln(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LatticeArcOccurrence {
    source: usize,
    arc: usize,
    segment_start: usize,
    emission_frame: usize,
    score: f32,
}

#[derive(Debug, Clone, Copy)]
struct LatticeTerminalOccurrence {
    node: usize,
    segment_start: usize,
    score: f32,
}

fn find_or_insert_lattice_node(
    nodes: &mut Vec<ApproximateLatticeNode>,
    state: ApproximateLatticeState,
) -> usize {
    nodes
        .iter()
        .position(|node| node.state == state)
        .unwrap_or_else(|| {
            let id = nodes.len();
            nodes.push(ApproximateLatticeNode {
                state,
                arcs: Vec::new(),
                terminal_scores: LatticeScoreAggregate::default(),
            });
            id
        })
}

/// Projects representative beam alignments onto a time-free token lattice.
///
/// A token arc contains the blank scores after the previous emission followed
/// by the token score at the representative emission frame. Nodes are merged
/// by emitted token count and last two tokens. Consequently a recovered path
/// may combine a prefix and suffix whose original frame ranges are
/// incompatible. This approximation is intentional: it measures how much
/// candidate recombination can offer before introducing an external language
/// model.
///
/// # Errors
///
/// Returns an error for malformed alignments, invalid encoder dimensions,
/// blank tokens in a seed, token IDs outside i64, or a selected-score output
/// whose shape differs from the request.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the diagnostic keeps graph projection, score aggregation, and reverse path recovery in one auditable routine"
)]
pub fn approximate_lattice_from_aligned_sequences<N: StatelessRnntSequenceScoreNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    initial_context: [i64; 2],
    seeds: &[StatelessRnntAlignedSeed<'_>],
    merge: StatelessRnntLatticeArcMerge,
) -> Result<Vec<StatelessRnntLatticeHypothesis>> {
    if encoder_dim == 0 || encoder.len() != frames.saturating_mul(encoder_dim) {
        bail!(
            "invalid approximate lattice encoder shape: values={}, frames={frames}, dimension={encoder_dim}",
            encoder.len()
        );
    }
    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    let root_state = ApproximateLatticeState {
        token_count: 0,
        context: initial_context,
    };
    let mut nodes = vec![ApproximateLatticeNode {
        state: root_state,
        arcs: Vec::new(),
        terminal_scores: LatticeScoreAggregate::default(),
    }];
    let mut arc_occurrences = Vec::<LatticeArcOccurrence>::new();
    let mut terminal_occurrences = Vec::<LatticeTerminalOccurrence>::new();

    for seed in seeds {
        if seed.token_ids.len() != seed.emission_frames.len() {
            bail!("approximate lattice seed token/timestamp lengths differ");
        }
        if seed.token_ids.contains(&blank_id) {
            bail!("approximate lattice seed must not contain blank");
        }
        if seed
            .emission_frames
            .iter()
            .enumerate()
            .any(|(index, &frame)| {
                frame >= frames
                    || index
                        .checked_sub(1)
                        .is_some_and(|previous| seed.emission_frames[previous] >= frame)
            })
        {
            bail!("approximate lattice seed emission frames must be increasing and in range");
        }
        let mut context = initial_context;
        let mut source = 0;
        let mut segment_start = 0;
        for (token_count, (&token_id, &emission_frame)) in
            seed.token_ids.iter().zip(seed.emission_frames).enumerate()
        {
            let token =
                i64::try_from(token_id).context("approximate lattice token ID exceeds i64")?;
            let destination_state = ApproximateLatticeState {
                token_count: token_count + 1,
                context: [context[1], token],
            };
            let destination = find_or_insert_lattice_node(&mut nodes, destination_state);
            let arc = nodes[source]
                .arcs
                .iter()
                .position(|candidate| {
                    candidate.token_id == token_id && candidate.destination == destination
                })
                .unwrap_or_else(|| {
                    let index = nodes[source].arcs.len();
                    nodes[source].arcs.push(ApproximateLatticeArc {
                        token_id,
                        destination,
                        scores: LatticeScoreAggregate::default(),
                    });
                    index
                });
            arc_occurrences.push(LatticeArcOccurrence {
                source,
                arc,
                segment_start,
                emission_frame,
                score: 0.0,
            });
            source = destination;
            context = destination_state.context;
            segment_start = emission_frame + 1;
        }
        terminal_occurrences.push(LatticeTerminalOccurrence {
            node: source,
            segment_start,
            score: 0.0,
        });
    }

    let mut contexts = Vec::<[i64; 2]>::new();
    let mut requested_token_ids = Vec::<Vec<usize>>::new();
    let mut node_context_rows = Vec::with_capacity(nodes.len());
    let mut arc_token_columns = Vec::with_capacity(nodes.len());
    for node in &nodes {
        let row = contexts
            .iter()
            .position(|&context| context == node.state.context)
            .unwrap_or_else(|| {
                let row = contexts.len();
                contexts.push(node.state.context);
                requested_token_ids.push(vec![blank_id]);
                row
            });
        node_context_rows.push(row);
        let columns = node
            .arcs
            .iter()
            .map(|arc| {
                requested_token_ids[row]
                    .iter()
                    .position(|&token| token == arc.token_id)
                    .unwrap_or_else(|| {
                        let column = requested_token_ids[row].len();
                        requested_token_ids[row].push(arc.token_id);
                        column
                    })
            })
            .collect::<Vec<_>>();
        arc_token_columns.push(columns);
    }

    for frame in 0..frames {
        let selected = network.requested_log_probabilities(
            &encoder[frame * encoder_dim..(frame + 1) * encoder_dim],
            &contexts,
            &requested_token_ids,
        )?;
        if selected.len() != requested_token_ids.len()
            || selected
                .iter()
                .zip(&requested_token_ids)
                .any(|(actual, requested)| actual.len() != requested.len())
        {
            bail!("approximate lattice selected-score shape changed");
        }
        for occurrence in &mut arc_occurrences {
            let row = node_context_rows[occurrence.source];
            if (occurrence.segment_start..occurrence.emission_frame).contains(&frame) {
                occurrence.score += selected[row][0];
            } else if frame == occurrence.emission_frame {
                occurrence.score +=
                    selected[row][arc_token_columns[occurrence.source][occurrence.arc]];
            }
        }
        for occurrence in &mut terminal_occurrences {
            if frame >= occurrence.segment_start {
                occurrence.score += selected[node_context_rows[occurrence.node]][0];
            }
        }
    }
    for occurrence in arc_occurrences {
        nodes[occurrence.source].arcs[occurrence.arc]
            .scores
            .add(occurrence.score);
    }
    for occurrence in terminal_occurrences {
        nodes[occurrence.node].terminal_scores.add(occurrence.score);
    }

    let mut order = (0..nodes.len()).collect::<Vec<_>>();
    order.sort_by_key(|&node| nodes[node].state.token_count);
    let mut best_scores = vec![f32::NEG_INFINITY; nodes.len()];
    let mut backpointers = vec![None::<(usize, usize)>; nodes.len()];
    best_scores[0] = 0.0;
    for source in order {
        if !best_scores[source].is_finite() {
            continue;
        }
        for arc in &nodes[source].arcs {
            let Some(arc_score) = arc.scores.merged(merge) else {
                continue;
            };
            let score = best_scores[source] + arc_score;
            if score > best_scores[arc.destination] {
                best_scores[arc.destination] = score;
                backpointers[arc.destination] = Some((source, arc.token_id));
            }
        }
    }

    let mut hypotheses = Vec::new();
    for (terminal, node) in nodes.iter().enumerate() {
        let Some(terminal_score) = node.terminal_scores.merged(merge) else {
            continue;
        };
        if !best_scores[terminal].is_finite() {
            continue;
        }
        let mut token_ids = Vec::with_capacity(node.state.token_count);
        let mut current = terminal;
        while current != 0 {
            let Some((parent, token_id)) = backpointers[current] else {
                bail!("approximate lattice terminal lost its backpointer");
            };
            token_ids.push(token_id);
            current = parent;
        }
        token_ids.reverse();
        hypotheses.push(StatelessRnntLatticeHypothesis {
            score: best_scores[terminal] + terminal_score,
            token_ids,
        });
    }
    hypotheses.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.token_ids.cmp(&right.token_ids))
    });
    Ok(hypotheses)
}

#[derive(Debug, Clone, Copy)]
struct FixedSequenceState {
    context_row: usize,
    next_token_column: Option<usize>,
}

type FixedSequencePlan = Vec<FixedSequenceState>;

struct FixedSequenceRequestPlan {
    contexts: Vec<[i64; 2]>,
    requested_token_ids: Vec<Vec<usize>>,
    sequences: Vec<FixedSequencePlan>,
}

fn plan_fixed_sequences(
    blank_id: usize,
    initial_context: [i64; 2],
    token_sequences: &[Vec<usize>],
) -> Result<FixedSequenceRequestPlan> {
    let mut contexts = Vec::<[i64; 2]>::new();
    let mut requested_token_ids = Vec::<Vec<usize>>::new();
    let mut plans = Vec::with_capacity(token_sequences.len());
    for token_ids in token_sequences {
        let mut context = initial_context;
        let mut states = Vec::with_capacity(token_ids.len() + 1);
        for next_token in token_ids.iter().copied().map(Some).chain([None]) {
            let context_row = contexts
                .iter()
                .position(|&existing| existing == context)
                .unwrap_or_else(|| {
                    let row = contexts.len();
                    contexts.push(context);
                    requested_token_ids.push(vec![blank_id]);
                    row
                });
            let next_token_column = next_token.map(|token_id| {
                requested_token_ids[context_row]
                    .iter()
                    .position(|&existing| existing == token_id)
                    .unwrap_or_else(|| {
                        let column = requested_token_ids[context_row].len();
                        requested_token_ids[context_row].push(token_id);
                        column
                    })
            });
            states.push(FixedSequenceState {
                context_row,
                next_token_column,
            });
            if let Some(token_id) = next_token {
                context = [
                    context[1],
                    i64::try_from(token_id).context("fixed stateless RNNT token ID exceeds i64")?,
                ];
            }
        }
        plans.push(states);
    }
    Ok(FixedSequenceRequestPlan {
        contexts,
        requested_token_ids,
        sequences: plans,
    })
}

fn validate_fixed_sequence_inputs(
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    token_sequences: &[Vec<usize>],
) -> Result<()> {
    if encoder_dim == 0 || encoder.len() != frames.saturating_mul(encoder_dim) {
        bail!(
            "invalid stateless RNNT encoder shape: values={}, frames={frames}, dimension={encoder_dim}",
            encoder.len()
        );
    }
    if token_sequences
        .iter()
        .flatten()
        .any(|&token_id| token_id == blank_id)
    {
        bail!("fixed stateless RNNT transcript must not contain blank");
    }
    Ok(())
}

/// Rescores complete transcripts without beam pruning.
///
/// The dynamic program consumes exactly one encoder frame per transition. A
/// transition either emits blank and keeps the output position, or emits the
/// transcript's next token and advances it. Decoder contexts and requested
/// token columns are shared across every supplied transcript, so repeated
/// two-token contexts result in one network row per frame.
///
/// # Errors
///
/// Returns an error for an invalid encoder shape, a transcript containing the
/// blank token, or a selected-score output that does not match its request.
#[allow(
    clippy::too_many_lines,
    reason = "the fixed-sequence DP keeps shared request planning and the paired forward/Viterbi recurrence together"
)]
pub fn score_stateless_sequences<N: StatelessRnntSequenceScoreNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    initial_context: [i64; 2],
    token_sequences: &[Vec<usize>],
) -> Result<Vec<StatelessRnntSequenceScores>> {
    validate_fixed_sequence_inputs(encoder, frames, encoder_dim, blank_id, token_sequences)?;
    if token_sequences.is_empty() {
        return Ok(Vec::new());
    }

    let plan = plan_fixed_sequences(blank_id, initial_context, token_sequences)?;

    let mut forward = token_sequences
        .iter()
        .map(|tokens| {
            let mut scores = vec![f32::NEG_INFINITY; tokens.len() + 1];
            scores[0] = 0.0;
            scores
        })
        .collect::<Vec<_>>();
    let mut viterbi = forward.clone();
    for frame in 0..frames {
        let encoder_frame = &encoder[frame * encoder_dim..(frame + 1) * encoder_dim];
        let selected = network.requested_log_probabilities(
            encoder_frame,
            &plan.contexts,
            &plan.requested_token_ids,
        )?;
        if selected.len() != plan.requested_token_ids.len()
            || selected
                .iter()
                .zip(&plan.requested_token_ids)
                .any(|(actual, requested)| actual.len() != requested.len())
        {
            bail!("stateless RNNT fixed-sequence score shape changed");
        }

        for ((states, forward_scores), viterbi_scores) in
            plan.sequences.iter().zip(&mut forward).zip(&mut viterbi)
        {
            let mut next_forward = vec![f32::NEG_INFINITY; forward_scores.len()];
            let mut next_viterbi = vec![f32::NEG_INFINITY; viterbi_scores.len()];
            for (output_position, state) in states.iter().enumerate() {
                let blank_score = selected[state.context_row][0];
                next_forward[output_position] = log_add_exp(
                    next_forward[output_position],
                    forward_scores[output_position] + blank_score,
                );
                next_viterbi[output_position] = next_viterbi[output_position]
                    .max(viterbi_scores[output_position] + blank_score);
                if let Some(column) = state.next_token_column {
                    let token_score = selected[state.context_row][column];
                    next_forward[output_position + 1] = log_add_exp(
                        next_forward[output_position + 1],
                        forward_scores[output_position] + token_score,
                    );
                    next_viterbi[output_position + 1] = next_viterbi[output_position + 1]
                        .max(viterbi_scores[output_position] + token_score);
                }
            }
            *forward_scores = next_forward;
            *viterbi_scores = next_viterbi;
        }
    }

    Ok(forward
        .into_iter()
        .zip(viterbi)
        .map(|(forward, viterbi)| StatelessRnntSequenceScores {
            viterbi_score: viterbi.last().copied().unwrap_or(f32::NEG_INFINITY),
            forward_score: forward.last().copied().unwrap_or(f32::NEG_INFINITY),
        })
        .collect())
}

#[derive(Debug)]
struct FixedSequenceFrameScores {
    blank: Vec<f32>,
    token: Vec<f32>,
}

fn posterior_quantile(probabilities: &[f32], quantile: f32) -> usize {
    let mut cumulative = 0.0;
    for (frame, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if cumulative >= quantile {
            return frame;
        }
    }
    probabilities.len().saturating_sub(1)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "encoder frame indices are small audio dimensions, and keeping the complete forward-backward recurrence together makes its probability accounting auditable"
)]
fn align_fixed_sequence(
    frames: usize,
    token_count: usize,
    frame_scores: &FixedSequenceFrameScores,
) -> Result<StatelessRnntSequenceAlignment> {
    if token_count > frames {
        bail!("{token_count} tokens cannot align to {frames} frames");
    }
    let width = token_count + 1;
    let matrix_size = (frames + 1).saturating_mul(width);
    let cell = |frame: usize, output_position: usize| frame * width + output_position;
    let score_cell = |frame: usize, output_position: usize| frame * width + output_position;

    let mut forward = vec![f32::NEG_INFINITY; matrix_size];
    let mut viterbi = vec![f32::NEG_INFINITY; matrix_size];
    let mut viterbi_emitted = vec![false; matrix_size];
    forward[cell(0, 0)] = 0.0;
    viterbi[cell(0, 0)] = 0.0;
    for frame in 0..frames {
        for output_position in 0..=token_count {
            let source = cell(frame, output_position);
            if !forward[source].is_finite() {
                continue;
            }
            let blank_destination = cell(frame + 1, output_position);
            let blank_score = frame_scores.blank[score_cell(frame, output_position)];
            forward[blank_destination] =
                log_add_exp(forward[blank_destination], forward[source] + blank_score);
            let blank_viterbi = viterbi[source] + blank_score;
            if blank_viterbi >= viterbi[blank_destination] {
                viterbi[blank_destination] = blank_viterbi;
                viterbi_emitted[blank_destination] = false;
            }
            if output_position < token_count {
                let token_destination = cell(frame + 1, output_position + 1);
                let token_score = frame_scores.token[frame * token_count + output_position];
                forward[token_destination] =
                    log_add_exp(forward[token_destination], forward[source] + token_score);
                let token_viterbi = viterbi[source] + token_score;
                if token_viterbi > viterbi[token_destination] {
                    viterbi[token_destination] = token_viterbi;
                    viterbi_emitted[token_destination] = true;
                }
            }
        }
    }

    let terminal = cell(frames, token_count);
    let forward_score = forward[terminal];
    if !forward_score.is_finite() {
        bail!("fixed transcript has no finite monotonic alignment");
    }
    let mut viterbi_frames = vec![0; token_count];
    let mut output_position = token_count;
    for frame in (1..=frames).rev() {
        if viterbi_emitted[cell(frame, output_position)] {
            output_position -= 1;
            viterbi_frames[output_position] = frame - 1;
        }
    }
    if output_position != 0 {
        bail!("monotonic Viterbi alignment lost a token backpointer");
    }

    let mut backward = vec![f32::NEG_INFINITY; matrix_size];
    backward[terminal] = 0.0;
    for frame in (0..frames).rev() {
        for output_position in 0..=token_count {
            let blank = frame_scores.blank[score_cell(frame, output_position)]
                + backward[cell(frame + 1, output_position)];
            let token = (output_position < token_count).then(|| {
                frame_scores.token[frame * token_count + output_position]
                    + backward[cell(frame + 1, output_position + 1)]
            });
            backward[cell(frame, output_position)] =
                token.map_or(blank, |score| log_add_exp(blank, score));
        }
    }

    let mut frame_emission_probabilities = vec![0.0; frames];
    let mut tokens = Vec::with_capacity(token_count);
    for token_position in 0..token_count {
        let mut probabilities = (0..frames)
            .map(|frame| {
                let log_probability = forward[cell(frame, token_position)]
                    + frame_scores.token[frame * token_count + token_position]
                    + backward[cell(frame + 1, token_position + 1)]
                    - forward_score;
                if log_probability.is_finite() {
                    log_probability.exp()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let probability_sum = probabilities.iter().sum::<f32>();
        if !probability_sum.is_finite() || probability_sum <= 0.0 {
            bail!("token occurrence has no finite posterior alignment");
        }
        for probability in &mut probabilities {
            *probability /= probability_sum;
        }
        for (frame_probability, &token_probability) in
            frame_emission_probabilities.iter_mut().zip(&probabilities)
        {
            *frame_probability += token_probability;
        }
        let posterior_mode_frame = probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map_or(0, |(frame, _)| frame);
        let expected_frame = probabilities
            .iter()
            .enumerate()
            .map(|(frame, probability)| frame as f32 * probability)
            .sum();
        let entropy = probabilities
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.ln())
            .sum();
        tokens.push(StatelessRnntTokenAlignment {
            viterbi_frame: viterbi_frames[token_position],
            posterior_mode_frame,
            expected_frame,
            posterior_lower_frame: posterior_quantile(&probabilities, 0.05),
            posterior_upper_frame: posterior_quantile(&probabilities, 0.95),
            entropy,
        });
    }
    for probability in &mut frame_emission_probabilities {
        *probability = probability.clamp(0.0, 1.0);
    }
    Ok(StatelessRnntSequenceAlignment {
        scores: StatelessRnntSequenceScores {
            viterbi_score: viterbi[terminal],
            forward_score,
        },
        tokens,
        frame_emission_probabilities,
    })
}

/// Computes exact forward-backward monotonic alignments for fixed transcripts.
///
/// This uses the same one-symbol-per-frame graph as modified beam search, but
/// performs no beam pruning once a transcript is fixed. Decoder contexts and
/// selected joiner columns are shared across all supplied transcripts. The
/// returned token marginals distinguish repeated occurrences by output
/// position and expose both a Viterbi timestamp and posterior timing summary.
///
/// # Errors
///
/// Returns an error for invalid shapes, blank tokens in a transcript, a
/// transcript longer than the encoder output, selected-score shape changes,
/// or a transcript with no finite alignment.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "shared model requests and per-transcript forward-backward tables are kept together so the diagnostic graph remains auditable"
)]
pub fn align_stateless_sequences<N: StatelessRnntSequenceScoreNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    initial_context: [i64; 2],
    token_sequences: &[Vec<usize>],
) -> Result<Vec<StatelessRnntSequenceAlignment>> {
    validate_fixed_sequence_inputs(encoder, frames, encoder_dim, blank_id, token_sequences)?;
    if token_sequences.is_empty() {
        return Ok(Vec::new());
    }
    for token_ids in token_sequences {
        if token_ids.len() > frames {
            bail!("{} tokens cannot align to {frames} frames", token_ids.len());
        }
    }
    let plan = plan_fixed_sequences(blank_id, initial_context, token_sequences)?;
    let mut scores = token_sequences
        .iter()
        .map(|tokens| FixedSequenceFrameScores {
            blank: Vec::with_capacity(frames * (tokens.len() + 1)),
            token: Vec::with_capacity(frames * tokens.len()),
        })
        .collect::<Vec<_>>();
    for frame in 0..frames {
        let selected = network.requested_log_probabilities(
            &encoder[frame * encoder_dim..(frame + 1) * encoder_dim],
            &plan.contexts,
            &plan.requested_token_ids,
        )?;
        if selected.len() != plan.requested_token_ids.len()
            || selected
                .iter()
                .zip(&plan.requested_token_ids)
                .any(|(actual, requested)| actual.len() != requested.len())
        {
            bail!("stateless RNNT fixed-sequence alignment score shape changed");
        }
        for (sequence_plan, sequence_scores) in plan.sequences.iter().zip(&mut scores) {
            for state in sequence_plan {
                sequence_scores.blank.push(selected[state.context_row][0]);
                if let Some(column) = state.next_token_column {
                    sequence_scores
                        .token
                        .push(selected[state.context_row][column]);
                }
            }
        }
    }
    token_sequences
        .iter()
        .zip(&scores)
        .map(|(tokens, scores)| align_fixed_sequence(frames, tokens.len(), scores))
        .collect()
}

/// Runs frame-synchronous modified beam search for a stateless two-token RNN-T.
///
/// At most one non-blank symbol is emitted from each encoder frame. Identical
/// token sequences are recombined with log-add-exp, then top-1-equivalent
/// states are dominated by `(emitted token count, last two tokens)`. Only the
/// active leaves are kept in the beam; full tokens and timestamps are restored
/// from shared parent arenas after the last frame. The returned hypotheses are
/// ordered by the configured final length score. The model-reference config
/// matches the sherpa/icefall convention: raw-score pruning and final
/// per-token normalization, while keeping the model-producing reference's
/// `[blank, blank]` initial context.
///
/// State dominance is intentionally a top-1 optimization. A future N-best
/// candidate generator must retain multiple transcript backpointers per state.
///
/// # Errors
///
/// Returns an error for invalid dimensions, a zero beam, or a model inference
/// failure.
pub fn modified_beam_search_stateless<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    beam_size: usize,
) -> Result<Vec<StatelessRnntHypothesis>> {
    modified_beam_search_stateless_with_config(
        network,
        encoder,
        frames,
        encoder_dim,
        vocab_size,
        blank_id,
        StatelessRnntBeamConfig::model_reference(beam_size, blank_id),
    )
}

/// Runs modified beam search with explicit diagnostic compatibility controls.
///
/// # Errors
///
/// Returns an error for invalid dimensions, token IDs, context values, a zero
/// beam, or a model inference failure.
pub fn modified_beam_search_stateless_with_config<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
) -> Result<Vec<StatelessRnntHypothesis>> {
    modified_beam_search_stateless_internal(
        network,
        encoder,
        frames,
        encoder_dim,
        vocab_size,
        blank_id,
        config,
        StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
        StatelessRnntScoreNormalization::SparseLogNormalizer,
        None,
        None,
    )
}

/// Runs unprofiled modified beam search with explicit token-selection and
/// score-normalization implementations.
///
/// # Errors
///
/// Returns an error for invalid dimensions or beam configuration, or when the
/// network fails.
#[allow(
    clippy::too_many_arguments,
    reason = "the diagnostic search options keep model dimensions explicit"
)]
pub fn modified_beam_search_stateless_with_config_and_search_options<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
    top_token_algorithm: StatelessRnntTopTokenAlgorithm,
    score_normalization: StatelessRnntScoreNormalization,
) -> Result<Vec<StatelessRnntHypothesis>> {
    modified_beam_search_stateless_internal(
        network,
        encoder,
        frames,
        encoder_dim,
        vocab_size,
        blank_id,
        config,
        top_token_algorithm,
        score_normalization,
        None,
        None,
    )
}

/// Runs modified beam search with a finite-state hotword context graph.
///
/// Hotword continuations are requested in addition to the acoustic Top-K so a
/// biased token can enter the beam even when it falls just below the ordinary
/// per-context cutoff. Unfinished matches are rolled back at final ranking.
///
/// # Errors
///
/// Returns an error for invalid search dimensions, a sherpa-compatible pruning
/// configuration, or a model inference failure.
#[allow(
    clippy::too_many_arguments,
    reason = "the hotword search mirrors the explicit model and decoder contracts"
)]
pub fn modified_beam_search_stateless_with_hotwords_and_search_options<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
    hotwords: &HotwordContextGraph,
    top_token_algorithm: StatelessRnntTopTokenAlgorithm,
    score_normalization: StatelessRnntScoreNormalization,
) -> Result<Vec<StatelessRnntHypothesis>> {
    modified_beam_search_stateless_internal(
        network,
        encoder,
        frames,
        encoder_dim,
        vocab_size,
        blank_id,
        config,
        top_token_algorithm,
        score_normalization,
        Some(hotwords),
        None,
    )
}

/// Runs the same search while accumulating diagnostic sub-stage timings.
///
/// # Errors
///
/// Returns the same validation and model inference errors as
/// [`modified_beam_search_stateless_with_config`].
#[allow(
    clippy::too_many_arguments,
    reason = "the profiled diagnostic entry point mirrors the model and search contracts"
)]
pub fn modified_beam_search_stateless_with_config_and_profile<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
    profile: &mut StatelessRnntSearchProfile,
) -> Result<Vec<StatelessRnntHypothesis>> {
    modified_beam_search_stateless_internal(
        network,
        encoder,
        frames,
        encoder_dim,
        vocab_size,
        blank_id,
        config,
        StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
        StatelessRnntScoreNormalization::SparseLogNormalizer,
        None,
        Some(profile),
    )
}

/// Profiles explicit top-token and score-normalization implementations.
///
/// # Errors
///
/// Returns the same validation and inference errors as normal modified beam
/// search.
#[allow(
    clippy::too_many_arguments,
    reason = "the search ablation mirrors the model, token selection, normalization, and profile contracts"
)]
pub fn modified_beam_search_stateless_profiled_with_search_options<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
    top_token_algorithm: StatelessRnntTopTokenAlgorithm,
    score_normalization: StatelessRnntScoreNormalization,
    profile: &mut StatelessRnntSearchProfile,
) -> Result<Vec<StatelessRnntHypothesis>> {
    modified_beam_search_stateless_internal(
        network,
        encoder,
        frames,
        encoder_dim,
        vocab_size,
        blank_id,
        config,
        top_token_algorithm,
        score_normalization,
        None,
        Some(profile),
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the shared search keeps the model dimensions and optional diagnostic sink explicit"
)]
fn modified_beam_search_stateless_internal<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
    top_token_algorithm: StatelessRnntTopTokenAlgorithm,
    score_normalization: StatelessRnntScoreNormalization,
    hotwords: Option<&HotwordContextGraph>,
    mut profile: Option<&mut StatelessRnntSearchProfile>,
) -> Result<Vec<StatelessRnntHypothesis>> {
    let beam_size = config.beam_size;
    if beam_size == 0 {
        bail!("RNNT beam size must be positive");
    }
    if frames == 0 || encoder_dim == 0 || encoder.len() != frames * encoder_dim {
        bail!("invalid stateless RNNT encoder shape");
    }
    if vocab_size == 0 || blank_id >= vocab_size {
        bail!("invalid stateless RNNT vocabulary");
    }
    if config
        .initial_context
        .iter()
        .any(|&token| token < -1 || usize::try_from(token).is_ok_and(|token| token >= vocab_size))
    {
        bail!("invalid stateless RNNT initial context");
    }
    if config
        .additional_non_emitting_id
        .is_some_and(|token| token >= vocab_size || token == blank_id)
    {
        bail!("invalid stateless RNNT additional non-emitting token");
    }
    if config.pruning == StatelessRnntPruning::Sherpa
        && config.search_length_normalization != StatelessRnntLengthNormalization::Raw
    {
        bail!("sherpa-compatible pruning supports raw search ranking only");
    }
    if hotwords.is_some() && config.pruning == StatelessRnntPruning::Sherpa {
        bail!("hotword context bias requires refill-capable beam pruning");
    }

    let mut prefix_arena = TokenPrefixArena::new(config.initial_context);
    let mut alignment_arena = AlignmentArena::new();
    let mut beam = vec![ActiveStatelessHypothesis {
        score: 0.0,
        prefix_id: 0,
        alignment_id: 0,
        hotword_state: hotwords.map_or(0, HotwordContextGraph::root),
    }];
    for (frame_index, encoder_frame) in encoder.chunks_exact(encoder_dim).enumerate() {
        beam = expand_stateless_frame(
            network,
            encoder_frame,
            frame_index,
            vocab_size,
            blank_id,
            config,
            &beam,
            &mut prefix_arena,
            &mut alignment_arena,
            top_token_algorithm,
            score_normalization,
            hotwords,
            profile.as_deref_mut(),
        )?;
    }

    let final_started = profile.is_some().then(Instant::now);
    beam.sort_by(|left, right| {
        active_ranking_score(
            right,
            &prefix_arena,
            config.final_length_normalization,
            hotwords,
        )
        .total_cmp(&active_ranking_score(
            left,
            &prefix_arena,
            config.final_length_normalization,
            hotwords,
        ))
        .then_with(|| {
            finalized_hotword_score(right, hotwords)
                .total_cmp(&finalized_hotword_score(left, hotwords))
        })
        .then_with(|| left.prefix_id.cmp(&right.prefix_id))
    });
    let hypotheses = beam
        .into_iter()
        .map(|hypothesis| StatelessRnntHypothesis {
            score: finalized_hotword_score(&hypothesis, hotwords),
            token_ids: prefix_arena.token_ids(hypothesis.prefix_id),
            timestamps: alignment_arena.timestamps(hypothesis.alignment_id),
        })
        .collect();
    if let (Some(profile), Some(started)) = (profile, final_started) {
        profile.final_ranking_and_reconstruction += started.elapsed();
    }
    Ok(hypotheses)
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "frame expansion keeps model dimensions and both immutable-history arenas explicit"
)]
fn expand_stateless_frame<N: StatelessRnntNetwork>(
    network: &mut N,
    encoder_frame: &[f32],
    frame_index: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
    beam: &[ActiveStatelessHypothesis],
    prefix_arena: &mut TokenPrefixArena,
    alignment_arena: &mut AlignmentArena,
    top_token_algorithm: StatelessRnntTopTokenAlgorithm,
    score_normalization: StatelessRnntScoreNormalization,
    hotwords: Option<&HotwordContextGraph>,
    mut profile: Option<&mut StatelessRnntSearchProfile>,
) -> Result<Vec<ActiveStatelessHypothesis>> {
    let beam_size = config.beam_size;
    let context_started = profile.is_some().then(Instant::now);
    let contexts = beam
        .iter()
        .map(|hypothesis| prefix_arena.nodes[hypothesis.prefix_id].context)
        .collect::<Vec<_>>();
    let (network_contexts, context_rows) = if config.deduplicate_contexts {
        unique_context_layout(&contexts)
    } else {
        (contexts, (0..beam.len()).collect())
    };
    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), context_started) {
        profile.frames += 1;
        profile.active_hypotheses += beam.len();
        profile.network_context_rows += network_contexts.len();
        profile.context_layout += started.elapsed();
    }

    let active_prefixes = beam
        .iter()
        .enumerate()
        .map(|(index, hypothesis)| (hypothesis.prefix_id, index))
        .collect::<Vec<_>>();
    let compact_enabled = network.supports_compact_log_probabilities()
        && score_normalization == StatelessRnntScoreNormalization::PrecomputedLogProbabilities
        && config.pruning != StatelessRnntPruning::Sherpa;
    let requested_token_ids = if compact_enabled {
        compact_score_requests(
            beam,
            prefix_arena,
            &context_rows,
            network_contexts.len(),
            &active_prefixes,
            blank_id,
            config.additional_non_emitting_id,
            hotwords,
        )
    } else {
        Vec::new()
    };
    let network_started = profile.is_some().then(Instant::now);
    let expected = network_contexts
        .len()
        .checked_mul(vocab_size)
        .ok_or_else(|| anyhow!("stateless RNNT logit shape overflow"))?;
    let mut compact_scores = if compact_enabled {
        network.compact_log_probabilities(
            encoder_frame,
            &network_contexts,
            &requested_token_ids,
            beam_size,
            blank_id,
            config.additional_non_emitting_id,
        )?
    } else {
        None
    };
    let mut logits = if compact_scores.is_none() {
        network.logits(encoder_frame, &network_contexts)?
    } else {
        Vec::new()
    };
    if compact_scores.is_none() && logits.len() != expected {
        bail!(
            "stateless RNNT logits length {} does not match [{}, {vocab_size}]",
            logits.len(),
            network_contexts.len()
        );
    }
    if let Some(scores) = compact_scores.as_ref() {
        validate_compact_scores(
            scores,
            &requested_token_ids,
            network_contexts.len(),
            vocab_size,
            beam_size,
            blank_id,
            config.additional_non_emitting_id,
        )?;
    }
    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), network_started) {
        profile.logit_values += expected;
        profile.network_output_values += compact_scores
            .as_ref()
            .map_or(logits.len(), |scores| scores.network_output_values);
        profile.network_output_bytes += compact_scores
            .as_ref()
            .map_or(logits.len() * std::mem::size_of::<f32>(), |scores| {
                scores.network_output_bytes
            });
        profile.network_logits += started.elapsed();
    }

    let mut compact_requested_scores = None;
    let (top_tokens, row_log_normalizers) = match score_normalization {
        StatelessRnntScoreNormalization::FullLogSoftmax => {
            let softmax_started = profile.is_some().then(Instant::now);
            for row in logits.chunks_exact_mut(vocab_size) {
                log_softmax_in_place(row);
            }
            if let (Some(profile), Some(started)) = (profile.as_deref_mut(), softmax_started) {
                profile.log_softmax += started.elapsed();
                profile.scalar_exp_terms_evaluated += logits.len();
            }
            if config.pruning == StatelessRnntPruning::Sherpa {
                return expand_full_prefix_frame(
                    frame_index,
                    vocab_size,
                    blank_id,
                    config,
                    beam,
                    &context_rows,
                    prefix_arena,
                    alignment_arena,
                    &logits,
                    None,
                );
            }
            let top_tokens_started = profile.is_some().then(Instant::now);
            let top_tokens = logits
                .chunks_exact(vocab_size)
                .map(|row| {
                    top_emitting_tokens(
                        row,
                        blank_id,
                        config.additional_non_emitting_id,
                        beam_size,
                        top_token_algorithm,
                    )
                })
                .collect::<Vec<_>>();
            if let (Some(profile), Some(started)) = (profile.as_deref_mut(), top_tokens_started) {
                profile.top_token_selection += started.elapsed();
            }
            (top_tokens, None)
        }
        StatelessRnntScoreNormalization::SparseLogNormalizer
        | StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e8
        | StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e7
        | StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e6 => {
            let epsilon = match score_normalization {
                StatelessRnntScoreNormalization::SparseLogNormalizer => None,
                StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e8 => Some(1.0e-8),
                StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e7 => Some(1.0e-7),
                StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e6 => Some(1.0e-6),
                _ => unreachable!("the outer match accepts only sparse normalizers here"),
            };
            if config.pruning == StatelessRnntPruning::Sherpa {
                let softmax_started = profile.is_some().then(Instant::now);
                let (row_log_normalizers, evaluated, skipped) =
                    row_log_normalizers(&logits, vocab_size, epsilon, None);
                if let (Some(profile), Some(started)) = (profile.as_deref_mut(), softmax_started) {
                    profile.log_softmax += started.elapsed();
                    profile.scalar_exp_terms_evaluated += evaluated;
                    profile.scalar_exp_terms_skipped += skipped;
                }
                return expand_full_prefix_frame(
                    frame_index,
                    vocab_size,
                    blank_id,
                    config,
                    beam,
                    &context_rows,
                    prefix_arena,
                    alignment_arena,
                    &logits,
                    Some(&row_log_normalizers),
                );
            }
            let top_tokens_started = profile.is_some().then(Instant::now);
            let analyses = logits
                .chunks_exact(vocab_size)
                .map(|row| {
                    top_emitting_tokens_and_maximum(
                        row,
                        blank_id,
                        config.additional_non_emitting_id,
                        beam_size,
                        top_token_algorithm,
                    )
                })
                .collect::<Vec<_>>();
            if let (Some(profile), Some(started)) = (profile.as_deref_mut(), top_tokens_started) {
                profile.top_token_selection += started.elapsed();
            }
            let softmax_started = profile.is_some().then(Instant::now);
            let maxima = analyses
                .iter()
                .map(|analysis| analysis.maximum)
                .collect::<Vec<_>>();
            let (row_log_normalizers, evaluated, skipped) =
                row_log_normalizers(&logits, vocab_size, epsilon, Some(&maxima));
            if let (Some(profile), Some(started)) = (profile.as_deref_mut(), softmax_started) {
                profile.log_softmax += started.elapsed();
                profile.scalar_exp_terms_evaluated += evaluated;
                profile.scalar_exp_terms_skipped += skipped;
            }
            (
                analyses
                    .into_iter()
                    .map(|analysis| analysis.top_tokens)
                    .collect(),
                Some(row_log_normalizers),
            )
        }
        StatelessRnntScoreNormalization::PrecomputedLogProbabilities => {
            if config.pruning == StatelessRnntPruning::Sherpa {
                return expand_full_prefix_frame(
                    frame_index,
                    vocab_size,
                    blank_id,
                    config,
                    beam,
                    &context_rows,
                    prefix_arena,
                    alignment_arena,
                    &logits,
                    None,
                );
            }
            if let Some(scores) = compact_scores.take() {
                compact_requested_scores = Some(scores.requested_scores);
                (scores.top_tokens, None)
            } else {
                let top_tokens_started = profile.is_some().then(Instant::now);
                let top_tokens = logits
                    .chunks_exact(vocab_size)
                    .map(|row| {
                        top_emitting_tokens(
                            row,
                            blank_id,
                            config.additional_non_emitting_id,
                            beam_size,
                            top_token_algorithm,
                        )
                    })
                    .collect::<Vec<_>>();
                if let (Some(profile), Some(started)) = (profile.as_deref_mut(), top_tokens_started)
                {
                    profile.top_token_selection += started.elapsed();
                }
                (top_tokens, None)
            }
        }
    };

    let candidate_started = profile.is_some().then(Instant::now);
    let frame_scores = compact_requested_scores.as_ref().map_or_else(
        || FrameLogProbabilities::Dense {
            logits: &logits,
            row_log_normalizers: row_log_normalizers.as_deref(),
            vocab_size,
        },
        |scores| FrameLogProbabilities::Compact {
            requested_token_ids: &requested_token_ids,
            requested_scores: scores,
        },
    );
    let mut candidates = Vec::with_capacity(beam.len() * (beam_size + 1));
    for (source_index, hypothesis) in beam.iter().enumerate() {
        let source_prefix = &prefix_arena.nodes[hypothesis.prefix_id];
        let source_row = context_rows[source_index];
        let blank_log_prob = frame_scores.get(source_row, blank_id)?;
        let blank = blank_candidate(
            hypothesis,
            source_prefix,
            source_index,
            blank_log_prob,
            frame_index,
            &frame_scores,
            beam,
            &context_rows,
            &active_prefixes,
            blank_id,
            config.additional_non_emitting_id,
            hotwords,
        )?;
        candidates.push(blank);

        // A row's top N non-blank tokens already occupy N distinct next
        // states. Any lower token from that row therefore cannot enter the
        // global top N states. This is the exact form of score-floor pruning;
        // it avoids materializing H*V prefix candidates.
        let mut emitting_tokens = top_tokens[source_row].clone();
        if let Some(graph) = hotwords {
            for &token_id in graph.continuation_tokens(hypothesis.hotword_state) {
                if token_id == blank_id
                    || config.additional_non_emitting_id == Some(token_id)
                    || emitting_tokens
                        .iter()
                        .any(|&(existing, _)| existing == token_id)
                {
                    continue;
                }
                emitting_tokens.push((token_id, frame_scores.get(source_row, token_id)?));
            }
        }
        for &(token_id, token_score) in &emitting_tokens {
            if prefix_arena
                .child(hypothesis.prefix_id, token_id)
                .is_some_and(|child| active_prefix_index(&active_prefixes, child).is_some())
            {
                // The active child's blank candidate owns this exact-prefix
                // merge, including its representative alignment.
                continue;
            }
            let (prefix, state, emitted_frame, hotword_score) = pending_transition(
                prefix_arena,
                hypothesis,
                source_prefix,
                token_id,
                blank_id,
                frame_index,
                hotwords,
            );
            let token_log_prob = if top_tokens[source_row]
                .iter()
                .any(|&(existing, _)| existing == token_id)
            {
                row_log_normalizers
                    .as_deref()
                    .map_or(token_score, |normalizers| {
                        token_score - normalizers[source_row]
                    })
            } else {
                token_score
            };
            candidates.push(PendingStatelessHypothesis {
                prefix,
                state,
                score: hypothesis.score + token_log_prob + hotword_score,
                best_path_score: hypothesis.score + token_log_prob + hotword_score,
                source_alignment_id: hypothesis.alignment_id,
                emitted_frame,
                source_index,
                token_id,
            });
        }
    }
    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), candidate_started) {
        profile.candidates_generated += candidates.len();
        profile.candidate_generation += started.elapsed();
    }

    if config.pruning == StatelessRnntPruning::FullPrefix {
        let pruning_started = profile.is_some().then(Instant::now);
        candidates.sort_unstable_by(|left, right| {
            pending_ranking_order(left, right, config.search_length_normalization)
        });
        candidates.truncate(beam_size);
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), pruning_started) {
            profile.score_pruning += started.elapsed();
        }
        let materialize_started = profile.is_some().then(Instant::now);
        let result = materialize_survivors(candidates, prefix_arena, alignment_arena);
        if let (Some(profile), Some(started)) = (profile, materialize_started) {
            profile.materialization += started.elapsed();
        }
        return result;
    }

    // Different transcripts with the same state have identical continuations
    // and denominator. Keep the maximum without combining their probability.
    let dominance_started = profile.is_some().then(Instant::now);
    candidates.sort_unstable_by(|left, right| {
        left.state
            .token_count
            .cmp(&right.state.token_count)
            .then_with(|| left.state.context.cmp(&right.state.context))
            .then_with(|| pending_ranking_order(left, right, config.search_length_normalization))
    });
    let mut survivors = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if survivors
            .last()
            .is_some_and(|previous: &PendingStatelessHypothesis| previous.state == candidate.state)
        {
            continue;
        }
        survivors.push(candidate);
    }
    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), dominance_started) {
        profile.states_after_dominance += survivors.len();
        profile.state_dominance += started.elapsed();
    }

    let pruning_started = profile.is_some().then(Instant::now);
    survivors.sort_unstable_by(|left, right| {
        pending_ranking_order(left, right, config.search_length_normalization)
    });
    survivors.truncate(beam_size);
    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), pruning_started) {
        profile.score_pruning += started.elapsed();
    }

    let materialize_started = profile.is_some().then(Instant::now);
    let result = materialize_survivors(survivors, prefix_arena, alignment_arena);
    if let (Some(profile), Some(started)) = (profile, materialize_started) {
        profile.materialization += started.elapsed();
    }
    result
}

#[derive(Debug, Clone, Copy)]
struct RawPrefixCandidate {
    score: f32,
    token_count: usize,
    source_index: usize,
    token_id: usize,
}

#[allow(
    clippy::too_many_arguments,
    reason = "full-prefix expansion keeps search semantics and immutable arenas explicit"
)]
fn expand_full_prefix_frame(
    frame_index: usize,
    vocab_size: usize,
    blank_id: usize,
    config: StatelessRnntBeamConfig,
    beam: &[ActiveStatelessHypothesis],
    context_rows: &[usize],
    prefix_arena: &mut TokenPrefixArena,
    alignment_arena: &mut AlignmentArena,
    logits: &[f32],
    row_log_normalizers: Option<&[f32]>,
) -> Result<Vec<ActiveStatelessHypothesis>> {
    debug_assert_eq!(config.pruning, StatelessRnntPruning::Sherpa);
    let premerge_limit = config.beam_size;
    let mut candidates = Vec::with_capacity(beam.len() * premerge_limit);
    for (source_index, hypothesis) in beam.iter().enumerate() {
        let row_index = context_rows[source_index];
        let row = &logits[row_index * vocab_size..(row_index + 1) * vocab_size];
        for (token_id, token_score) in top_tokens(row, premerge_limit) {
            let token_log_prob = row_log_normalizers.map_or(token_score, |normalizers| {
                token_score - normalizers[row_index]
            });
            candidates.push(RawPrefixCandidate {
                score: hypothesis.score + token_log_prob,
                token_count: prefix_arena.nodes[hypothesis.prefix_id].token_count
                    + usize::from(
                        token_id != blank_id && config.additional_non_emitting_id != Some(token_id),
                    ),
                source_index,
                token_id,
            });
        }
    }
    candidates.sort_unstable_by(|left, right| {
        raw_prefix_ranking_order(left, right, config.search_length_normalization)
    });
    candidates.truncate(premerge_limit);

    // Candidates are score ordered, so the first representative of an exact
    // prefix also owns its stable alignment while all selected path mass is
    // combined into the prefix score.
    let mut merged = Vec::<ActiveStatelessHypothesis>::with_capacity(candidates.len());
    for candidate in candidates {
        let source = beam[candidate.source_index];
        let non_emitting = candidate.token_id == blank_id
            || config.additional_non_emitting_id == Some(candidate.token_id);
        let prefix_id = if non_emitting {
            source.prefix_id
        } else {
            prefix_arena.intern_child(source.prefix_id, candidate.token_id)
        };
        if let Some(existing) = merged
            .iter_mut()
            .find(|hypothesis| hypothesis.prefix_id == prefix_id)
        {
            existing.score = log_add_exp(existing.score, candidate.score);
            continue;
        }
        let alignment_id = if non_emitting {
            source.alignment_id
        } else {
            alignment_arena.push(source.alignment_id, frame_index)
        };
        merged.push(ActiveStatelessHypothesis {
            score: candidate.score,
            prefix_id,
            alignment_id,
            hotword_state: 0,
        });
    }
    if merged.is_empty() {
        bail!("stateless RNNT beam search produced no hypotheses");
    }
    Ok(merged)
}

fn raw_prefix_ranking_order(
    left: &RawPrefixCandidate,
    right: &RawPrefixCandidate,
    normalization: StatelessRnntLengthNormalization,
) -> std::cmp::Ordering {
    ranking_score(right.score, right.token_count, normalization)
        .total_cmp(&ranking_score(left.score, left.token_count, normalization))
        .then_with(|| right.score.total_cmp(&left.score))
        .then_with(|| left.source_index.cmp(&right.source_index))
        .then_with(|| left.token_id.cmp(&right.token_id))
}

fn unique_context_layout(contexts: &[[i64; 2]]) -> (Vec<[i64; 2]>, Vec<usize>) {
    let mut unique = Vec::new();
    let mut rows = Vec::with_capacity(contexts.len());
    for &context in contexts {
        let row = unique
            .iter()
            .position(|&existing| existing == context)
            .unwrap_or_else(|| {
                let row = unique.len();
                unique.push(context);
                row
            });
        rows.push(row);
    }
    (unique, rows)
}

#[allow(
    clippy::too_many_arguments,
    reason = "compact requests encode blank, optional unknown, and exact-prefix dependencies"
)]
fn compact_score_requests(
    beam: &[ActiveStatelessHypothesis],
    prefix_arena: &TokenPrefixArena,
    context_rows: &[usize],
    network_rows: usize,
    active_prefixes: &[(usize, usize)],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    hotwords: Option<&HotwordContextGraph>,
) -> Vec<Vec<usize>> {
    let mut requested = (0..network_rows)
        .map(|_| {
            let mut tokens = vec![blank_id];
            if let Some(token_id) = additional_non_emitting_id {
                tokens.push(token_id);
            }
            tokens
        })
        .collect::<Vec<_>>();
    for (hypothesis_index, hypothesis) in beam.iter().enumerate() {
        let source = &prefix_arena.nodes[hypothesis.prefix_id];
        if let (Some(parent_prefix), Some(last_token)) = (source.parent, source.token_id)
            && let Some(parent_index) = active_prefix_index(active_prefixes, parent_prefix)
        {
            requested[context_rows[parent_index]].push(last_token);
        }
        if let Some(graph) = hotwords {
            requested[context_rows[hypothesis_index]]
                .extend_from_slice(graph.continuation_tokens(hypothesis.hotword_state));
        }
    }
    for tokens in &mut requested {
        tokens.sort_unstable();
        tokens.dedup();
    }
    requested
}

fn validate_compact_scores(
    scores: &StatelessRnntCompactScores,
    requested_token_ids: &[Vec<usize>],
    rows: usize,
    vocab_size: usize,
    emitting_limit: usize,
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
) -> Result<()> {
    if scores.top_tokens.len() != rows
        || scores.requested_scores.len() != rows
        || requested_token_ids.len() != rows
    {
        bail!("stateless RNNT compact score row count changed");
    }
    let excluded = 1 + usize::from(additional_non_emitting_id.is_some());
    let expected_top = emitting_limit.min(vocab_size.saturating_sub(excluded));
    for (row, requested_tokens) in requested_token_ids.iter().enumerate().take(rows) {
        if scores.top_tokens[row].len() != expected_top
            || scores.requested_scores[row].len() != requested_tokens.len()
        {
            bail!("stateless RNNT compact score width changed at row {row}");
        }
        if scores.top_tokens[row].iter().any(|&(token_id, _)| {
            token_id >= vocab_size
                || token_id == blank_id
                || additional_non_emitting_id == Some(token_id)
        }) {
            bail!("stateless RNNT compact TopK returned a non-emitting token");
        }
        if scores.top_tokens[row]
            .windows(2)
            .any(|pair| token_order(&pair[0], &pair[1]).is_gt())
        {
            bail!("stateless RNNT compact TopK ordering changed");
        }
    }
    Ok(())
}

enum FrameLogProbabilities<'a> {
    Dense {
        logits: &'a [f32],
        row_log_normalizers: Option<&'a [f32]>,
        vocab_size: usize,
    },
    Compact {
        requested_token_ids: &'a [Vec<usize>],
        requested_scores: &'a [Vec<f32>],
    },
}

impl FrameLogProbabilities<'_> {
    fn get(&self, row: usize, token_id: usize) -> Result<f32> {
        match self {
            Self::Dense {
                logits,
                row_log_normalizers,
                vocab_size,
            } => Ok(log_probability(
                logits,
                *row_log_normalizers,
                *vocab_size,
                row,
                token_id,
            )),
            Self::Compact {
                requested_token_ids,
                requested_scores,
            } => requested_token_ids[row]
                .iter()
                .position(|&requested| requested == token_id)
                .map(|column| requested_scores[row][column])
                .ok_or_else(|| {
                    anyhow!("stateless RNNT compact scores omitted token {token_id} at row {row}")
                }),
        }
    }
}

fn active_prefix_index(active_prefixes: &[(usize, usize)], prefix_id: usize) -> Option<usize> {
    active_prefixes
        .iter()
        .find_map(|&(prefix, index)| (prefix == prefix_id).then_some(index))
}

#[allow(
    clippy::too_many_arguments,
    reason = "exact-prefix blank merge needs both active-leaf and model-row identities"
)]
fn blank_candidate(
    hypothesis: &ActiveStatelessHypothesis,
    source_prefix: &TokenPrefixNode,
    source_index: usize,
    blank_log_prob: f32,
    frame_index: usize,
    frame_scores: &FrameLogProbabilities<'_>,
    beam: &[ActiveStatelessHypothesis],
    context_rows: &[usize],
    active_prefixes: &[(usize, usize)],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    hotwords: Option<&HotwordContextGraph>,
) -> Result<PendingStatelessHypothesis> {
    let blank_score = hypothesis.score + blank_log_prob;
    let mut candidate = PendingStatelessHypothesis {
        prefix: PendingPrefix::Existing(hypothesis.prefix_id),
        state: StatelessSearchState {
            token_count: source_prefix.token_count,
            context: source_prefix.context,
            hotword_state: hypothesis.hotword_state,
        },
        score: blank_score,
        best_path_score: blank_score,
        source_alignment_id: hypothesis.alignment_id,
        emitted_frame: None,
        source_index,
        token_id: blank_id,
    };
    if let (Some(parent_prefix), Some(last_token)) = (source_prefix.parent, source_prefix.token_id)
        && let Some(parent_index) = active_prefix_index(active_prefixes, parent_prefix)
    {
        let parent_row = context_rows[parent_index];
        let hotword_score = hotwords.map_or(0.0, |graph| {
            graph
                .forward(beam[parent_index].hotword_state, last_token)
                .0
        });
        let emitted_score =
            beam[parent_index].score + frame_scores.get(parent_row, last_token)? + hotword_score;
        candidate.score = log_add_exp(candidate.score, emitted_score);
        if emitted_score > candidate.best_path_score
            || (emitted_score.total_cmp(&candidate.best_path_score).is_eq()
                && (parent_index, last_token) < (candidate.source_index, candidate.token_id))
        {
            candidate.best_path_score = emitted_score;
            candidate.source_alignment_id = beam[parent_index].alignment_id;
            candidate.emitted_frame = Some(frame_index);
            candidate.source_index = parent_index;
            candidate.token_id = last_token;
        }
    }
    if let Some(token_id) = additional_non_emitting_id {
        let source_row = context_rows[source_index];
        let non_emitting_score = hypothesis.score + frame_scores.get(source_row, token_id)?;
        candidate.score = log_add_exp(candidate.score, non_emitting_score);
        if non_emitting_score > candidate.best_path_score
            || (non_emitting_score
                .total_cmp(&candidate.best_path_score)
                .is_eq()
                && (source_index, token_id) < (candidate.source_index, candidate.token_id))
        {
            candidate.best_path_score = non_emitting_score;
            candidate.source_alignment_id = hypothesis.alignment_id;
            candidate.emitted_frame = None;
            candidate.source_index = source_index;
            candidate.token_id = token_id;
        }
    }
    Ok(candidate)
}

fn top_emitting_tokens(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
    algorithm: StatelessRnntTopTokenAlgorithm,
) -> Vec<(usize, f32)> {
    match algorithm {
        StatelessRnntTopTokenAlgorithm::BinarySearch => {
            top_emitting_tokens_binary_search(row, blank_id, additional_non_emitting_id, limit)
        }
        StatelessRnntTopTokenAlgorithm::CutoffBinarySearch => {
            top_emitting_tokens_cutoff_binary_search(
                row,
                blank_id,
                additional_non_emitting_id,
                limit,
            )
        }
        StatelessRnntTopTokenAlgorithm::WorstFirstHeap => {
            top_emitting_tokens_worst_first_heap(row, blank_id, additional_non_emitting_id, limit)
        }
    }
}

struct TokenRowAnalysis {
    maximum: f32,
    top_tokens: Vec<(usize, f32)>,
}

fn top_emitting_tokens_and_maximum(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
    algorithm: StatelessRnntTopTokenAlgorithm,
) -> TokenRowAnalysis {
    match algorithm {
        StatelessRnntTopTokenAlgorithm::BinarySearch => {
            analyze_emitting_tokens_binary_search(row, blank_id, additional_non_emitting_id, limit)
        }
        StatelessRnntTopTokenAlgorithm::CutoffBinarySearch => {
            analyze_emitting_tokens_cutoff_binary_search(
                row,
                blank_id,
                additional_non_emitting_id,
                limit,
            )
        }
        StatelessRnntTopTokenAlgorithm::WorstFirstHeap => analyze_emitting_tokens_worst_first_heap(
            row,
            blank_id,
            additional_non_emitting_id,
            limit,
        ),
    }
}

fn analyze_emitting_tokens_binary_search(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
) -> TokenRowAnalysis {
    let mut maximum = f32::NEG_INFINITY;
    let mut top = Vec::<(usize, f32)>::with_capacity(limit);
    for (token_id, &score) in row.iter().enumerate() {
        if score.total_cmp(&maximum).is_gt() {
            maximum = score;
        }
        if token_id == blank_id || additional_non_emitting_id == Some(token_id) {
            continue;
        }
        let candidate = (token_id, score);
        let position = top
            .binary_search_by(|existing| token_order(existing, &candidate))
            .unwrap_or_else(|position| position);
        if position < limit {
            top.insert(position, candidate);
            top.truncate(limit);
        }
    }
    TokenRowAnalysis {
        maximum,
        top_tokens: top,
    }
}

fn analyze_emitting_tokens_cutoff_binary_search(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
) -> TokenRowAnalysis {
    let mut maximum = f32::NEG_INFINITY;
    let mut top = Vec::<(usize, f32)>::with_capacity(limit);
    for (token_id, &score) in row.iter().enumerate() {
        if score.total_cmp(&maximum).is_gt() {
            maximum = score;
        }
        if token_id == blank_id || additional_non_emitting_id == Some(token_id) {
            continue;
        }
        if limit == 0 {
            continue;
        }
        let candidate = (token_id, score);
        if top.len() == limit
            && !token_order(&candidate, top.last().expect("a full top list has a floor")).is_lt()
        {
            continue;
        }
        let position = top
            .binary_search_by(|existing| token_order(existing, &candidate))
            .unwrap_or_else(|position| position);
        top.insert(position, candidate);
        top.truncate(limit);
    }
    TokenRowAnalysis {
        maximum,
        top_tokens: top,
    }
}

#[derive(Debug, Clone, Copy)]
struct WorstFirstToken {
    token_id: usize,
    score: f32,
}

impl PartialEq for WorstFirstToken {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for WorstFirstToken {}

impl PartialOrd for WorstFirstToken {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorstFirstToken {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        token_order(&(self.token_id, self.score), &(other.token_id, other.score))
    }
}

fn analyze_emitting_tokens_worst_first_heap(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
) -> TokenRowAnalysis {
    let mut maximum = f32::NEG_INFINITY;
    let mut top = BinaryHeap::<WorstFirstToken>::with_capacity(limit);
    for (token_id, &score) in row.iter().enumerate() {
        if score.total_cmp(&maximum).is_gt() {
            maximum = score;
        }
        if token_id == blank_id || additional_non_emitting_id == Some(token_id) {
            continue;
        }
        if limit == 0 {
            continue;
        }
        let candidate = WorstFirstToken { token_id, score };
        if top.len() < limit {
            top.push(candidate);
            continue;
        }
        let mut floor = top.peek_mut().expect("a full top heap has a floor");
        if candidate < *floor {
            *floor = candidate;
        }
    }
    let mut sorted = top
        .into_iter()
        .map(|token| (token.token_id, token.score))
        .collect::<Vec<_>>();
    sorted.sort_unstable_by(token_order);
    TokenRowAnalysis {
        maximum,
        top_tokens: sorted,
    }
}

fn top_emitting_tokens_binary_search(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
) -> Vec<(usize, f32)> {
    let mut top = Vec::<(usize, f32)>::with_capacity(limit);
    for (token_id, &score) in row.iter().enumerate() {
        if token_id == blank_id || additional_non_emitting_id == Some(token_id) {
            continue;
        }
        let candidate = (token_id, score);
        let position = top
            .binary_search_by(|existing| token_order(existing, &candidate))
            .unwrap_or_else(|position| position);
        if position < limit {
            top.insert(position, candidate);
            top.truncate(limit);
        }
    }
    top
}

fn top_emitting_tokens_cutoff_binary_search(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
) -> Vec<(usize, f32)> {
    if limit == 0 {
        return Vec::new();
    }
    let mut top = Vec::<(usize, f32)>::with_capacity(limit);
    for (token_id, &score) in row.iter().enumerate() {
        if token_id == blank_id || additional_non_emitting_id == Some(token_id) {
            continue;
        }
        let candidate = (token_id, score);
        if top.len() == limit
            && !token_order(&candidate, top.last().expect("a full top list has a floor")).is_lt()
        {
            continue;
        }
        let position = top
            .binary_search_by(|existing| token_order(existing, &candidate))
            .unwrap_or_else(|position| position);
        top.insert(position, candidate);
        top.truncate(limit);
    }
    top
}

fn top_emitting_tokens_worst_first_heap(
    row: &[f32],
    blank_id: usize,
    additional_non_emitting_id: Option<usize>,
    limit: usize,
) -> Vec<(usize, f32)> {
    if limit == 0 {
        return Vec::new();
    }
    let mut top = BinaryHeap::<WorstFirstToken>::with_capacity(limit);
    for (token_id, &score) in row.iter().enumerate() {
        if token_id == blank_id || additional_non_emitting_id == Some(token_id) {
            continue;
        }
        let candidate = WorstFirstToken { token_id, score };
        if top.len() < limit {
            top.push(candidate);
            continue;
        }
        let mut floor = top.peek_mut().expect("a full top heap has a floor");
        if candidate < *floor {
            *floor = candidate;
        }
    }
    let mut sorted = top
        .into_iter()
        .map(|token| (token.token_id, token.score))
        .collect::<Vec<_>>();
    sorted.sort_unstable_by(token_order);
    sorted
}

fn top_tokens(row: &[f32], limit: usize) -> Vec<(usize, f32)> {
    let mut top = Vec::<(usize, f32)>::with_capacity(limit);
    for (token_id, &score) in row.iter().enumerate() {
        let candidate = (token_id, score);
        let position = top
            .binary_search_by(|existing| token_order(existing, &candidate))
            .unwrap_or_else(|position| position);
        if position < limit {
            top.insert(position, candidate);
            top.truncate(limit);
        }
    }
    top
}

fn token_order(left: &(usize, f32), right: &(usize, f32)) -> std::cmp::Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}

fn pending_transition(
    prefix_arena: &TokenPrefixArena,
    hypothesis: &ActiveStatelessHypothesis,
    source_prefix: &TokenPrefixNode,
    token_id: usize,
    blank_id: usize,
    frame_index: usize,
    hotwords: Option<&HotwordContextGraph>,
) -> (PendingPrefix, StatelessSearchState, Option<usize>, f32) {
    if token_id == blank_id {
        return (
            PendingPrefix::Existing(hypothesis.prefix_id),
            StatelessSearchState {
                token_count: source_prefix.token_count,
                context: source_prefix.context,
                hotword_state: hypothesis.hotword_state,
            },
            None,
            0.0,
        );
    }
    let (hotword_score, hotword_state) = hotwords.map_or((0.0, 0), |graph| {
        graph.forward(hypothesis.hotword_state, token_id)
    });
    let prefix = prefix_arena.child(hypothesis.prefix_id, token_id).map_or(
        PendingPrefix::New {
            parent: hypothesis.prefix_id,
            token_id,
        },
        PendingPrefix::Existing,
    );
    (
        prefix,
        StatelessSearchState {
            token_count: source_prefix.token_count + 1,
            context: [source_prefix.context[1], token_context_id(token_id)],
            hotword_state,
        },
        Some(frame_index),
        hotword_score,
    )
}

fn materialize_survivors(
    survivors: Vec<PendingStatelessHypothesis>,
    prefix_arena: &mut TokenPrefixArena,
    alignment_arena: &mut AlignmentArena,
) -> Result<Vec<ActiveStatelessHypothesis>> {
    if survivors.is_empty() {
        bail!("stateless RNNT beam search produced no hypotheses");
    }
    Ok(survivors
        .into_iter()
        .map(|candidate| {
            let prefix_id = match candidate.prefix {
                PendingPrefix::Existing(prefix_id) => prefix_id,
                PendingPrefix::New { parent, token_id } => {
                    prefix_arena.intern_child(parent, token_id)
                }
            };
            let alignment_id = candidate
                .emitted_frame
                .map_or(candidate.source_alignment_id, |frame| {
                    alignment_arena.push(candidate.source_alignment_id, frame)
                });
            ActiveStatelessHypothesis {
                score: candidate.score,
                prefix_id,
                alignment_id,
                hotword_state: candidate.state.hotword_state,
            }
        })
        .collect())
}

fn pending_ranking_order(
    left: &PendingStatelessHypothesis,
    right: &PendingStatelessHypothesis,
    normalization: StatelessRnntLengthNormalization,
) -> std::cmp::Ordering {
    ranking_score(right.score, right.state.token_count, normalization)
        .total_cmp(&ranking_score(
            left.score,
            left.state.token_count,
            normalization,
        ))
        .then_with(|| right.score.total_cmp(&left.score))
        .then_with(|| pending_path_order(left, right))
}

fn pending_path_order(
    left: &PendingStatelessHypothesis,
    right: &PendingStatelessHypothesis,
) -> std::cmp::Ordering {
    left.source_index
        .cmp(&right.source_index)
        .then_with(|| left.token_id.cmp(&right.token_id))
}

fn log_softmax_in_place(values: &mut [f32]) {
    let maximum = values
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .unwrap_or(f32::NEG_INFINITY);
    let log_sum = row_log_normalizer_from_maximum(values, maximum);
    for value in values {
        *value -= log_sum;
    }
}

fn row_log_normalizer_from_maximum(values: &[f32], maximum: f32) -> f32 {
    values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f32>()
        .ln()
        + maximum
}

fn row_log_normalizers(
    logits: &[f32],
    vocab_size: usize,
    epsilon: Option<f32>,
    maxima: Option<&[f32]>,
) -> (Vec<f32>, usize, usize) {
    let mut evaluated = 0;
    let mut skipped = 0;
    let normalizers = logits
        .chunks_exact(vocab_size)
        .enumerate()
        .map(|(row_index, row)| {
            let maximum = maxima.map_or_else(
                || {
                    row.iter()
                        .copied()
                        .max_by(f32::total_cmp)
                        .unwrap_or(f32::NEG_INFINITY)
                },
                |maxima| maxima[row_index],
            );
            if let Some(epsilon) = epsilon {
                let cutoff = epsilon.ln();
                let mut sum = 0.0_f32;
                for &value in row {
                    let difference = value - maximum;
                    if difference < cutoff {
                        skipped += 1;
                    } else {
                        sum += difference.exp();
                        evaluated += 1;
                    }
                }
                sum.ln() + maximum
            } else {
                evaluated += row.len();
                row_log_normalizer_from_maximum(row, maximum)
            }
        })
        .collect();
    (normalizers, evaluated, skipped)
}

fn log_probability(
    logits: &[f32],
    row_log_normalizers: Option<&[f32]>,
    vocab_size: usize,
    row: usize,
    token: usize,
) -> f32 {
    let score = logits[row * vocab_size + token];
    row_log_normalizers.map_or(score, |normalizers| score - normalizers[row])
}

fn log_add_exp(left: f32, right: f32) -> f32 {
    let maximum = left.max(right);
    if maximum.is_infinite() {
        maximum
    } else {
        maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "bounded transcript lengths are represented in the model's f32 score domain"
)]
fn active_ranking_score(
    hypothesis: &ActiveStatelessHypothesis,
    prefix_arena: &TokenPrefixArena,
    normalization: StatelessRnntLengthNormalization,
    hotwords: Option<&HotwordContextGraph>,
) -> f32 {
    ranking_score(
        finalized_hotword_score(hypothesis, hotwords),
        prefix_arena.nodes[hypothesis.prefix_id].token_count,
        normalization,
    )
}

fn finalized_hotword_score(
    hypothesis: &ActiveStatelessHypothesis,
    hotwords: Option<&HotwordContextGraph>,
) -> f32 {
    hypothesis.score + hotwords.map_or(0.0, |graph| graph.finalize(hypothesis.hotword_state))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "bounded transcript lengths are represented in the model's f32 score domain"
)]
fn ranking_score(
    score: f32,
    token_count: usize,
    normalization: StatelessRnntLengthNormalization,
) -> f32 {
    let exponent_quarters = normalization.exponent_quarters();
    if exponent_quarters == 0 {
        return score;
    }
    let length = (token_count + 2) as f32;
    score / length.powf(f32::from(exponent_quarters) / 4.0)
}

/// Runs greedy RNN-T decoding, optionally continuing an existing hypothesis.
///
/// # Errors
///
/// Returns an error for an invalid encoder shape or a predictor/joiner inference failure.
pub fn greedy_rnnt<N: RnntNetwork>(
    network: &mut N,
    encoder: &[f32],
    frames: usize,
    encoder_dim: usize,
    blank_id: usize,
    max_symbols_per_frame: usize,
    previous: Option<RnntHypothesis<N::State>>,
) -> Result<RnntHypothesis<N::State>> {
    if frames == 0 || encoder_dim == 0 || encoder.len() != frames * encoder_dim {
        bail!("invalid RNNT encoder shape");
    }
    let mut hypothesis = previous.unwrap_or_else(|| RnntHypothesis {
        token_ids: Vec::new(),
        timestamps: Vec::new(),
        state: network.initial_state(),
        last_token: None,
        frame_offset: 0,
    });
    for time in 0..frames {
        let frame = (0..encoder_dim)
            .map(|feature| encoder[feature * frames + time])
            .collect::<Vec<_>>();
        for _ in 0..max_symbols_per_frame {
            let label = hypothesis.last_token.unwrap_or(blank_id);
            let (prediction, candidate_state) = network.predictor(label, &hypothesis.state)?;
            let logits = network.joiner(&frame, &prediction)?;
            if logits.len() != blank_id + 1 {
                bail!("RNNT joiner vocabulary changed");
            }
            let token = logits
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
                    if value > best.1 { (index, value) } else { best }
                })
                .0;
            if token == blank_id {
                break;
            }
            hypothesis.token_ids.push(token);
            hypothesis.timestamps.push(hypothesis.frame_offset + time);
            hypothesis.state = candidate_state;
            hypothesis.last_token = Some(token);
        }
    }
    hypothesis.frame_offset += frames;
    Ok(hypothesis)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::super::hotword::HotwordContextGraph;

    use super::{
        RnntNetwork, StatelessRnntAlignedSeed, StatelessRnntBeamConfig, StatelessRnntCompactScores,
        StatelessRnntLatticeArcMerge, StatelessRnntLengthNormalization, StatelessRnntNetwork,
        StatelessRnntPruning, StatelessRnntScoreNormalization, StatelessRnntSearchProfile,
        StatelessRnntSequenceScoreNetwork, StatelessRnntTopTokenAlgorithm,
        align_stateless_sequences, approximate_lattice_from_aligned_sequences, greedy_rnnt,
        modified_beam_search_stateless,
        modified_beam_search_stateless_profiled_with_search_options,
        modified_beam_search_stateless_with_config,
        modified_beam_search_stateless_with_config_and_profile,
        modified_beam_search_stateless_with_hotwords_and_search_options, score_stateless_sequences,
        top_emitting_tokens,
    };

    struct Network {
        calls: usize,
    }

    impl RnntNetwork for Network {
        type State = usize;
        fn initial_state(&self) -> usize {
            0
        }
        fn predictor(&mut self, _token: usize, state: &usize) -> Result<(Vec<f32>, usize)> {
            Ok((vec![0.0], state + 1))
        }
        fn joiner(&mut self, _frame: &[f32], _prediction: &[f32]) -> Result<Vec<f32>> {
            self.calls += 1;
            Ok(if self.calls % 2 == 1 {
                vec![2.0, 1.0]
            } else {
                vec![1.0, 2.0]
            })
        }
    }

    #[test]
    fn rnnt_commits_state_only_for_nonblank_and_offsets_streaming_timestamps() {
        let mut network = Network { calls: 0 };
        let first = greedy_rnnt(&mut network, &[0.0, 0.0], 2, 1, 1, 10, None).unwrap();
        assert_eq!(first.token_ids, vec![0, 0]);
        assert_eq!(first.timestamps, vec![0, 1]);
        assert_eq!(first.state, 2);
        let second = greedy_rnnt(&mut network, &[0.0], 1, 1, 1, 10, Some(first)).unwrap();
        assert_eq!(second.token_ids, vec![0, 0, 0]);
        assert_eq!(second.timestamps, vec![0, 1, 2]);
        assert_eq!(second.state, 3);
    }

    struct StatelessNetwork;

    impl StatelessRnntNetwork for StatelessNetwork {
        fn logits(&mut self, encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            let first_frame = encoder_frame[0] == 0.0;
            Ok(contexts
                .iter()
                .flat_map(|context| {
                    if first_frame {
                        [-10.0, 2.0, 1.8]
                    } else if context[1] == 2 {
                        [5.0, 0.0, 0.0]
                    } else {
                        [-5.0, -5.0, -5.0]
                    }
                })
                .collect())
        }
    }

    #[test]
    fn modified_beam_keeps_a_lower_local_token_when_it_wins_after_the_next_frame() {
        let mut network = StatelessNetwork;
        let hypotheses =
            modified_beam_search_stateless(&mut network, &[0.0, 1.0], 2, 1, 3, 0, 4).unwrap();

        assert_eq!(hypotheses[0].token_ids, vec![2]);
        assert_eq!(hypotheses[0].timestamps, vec![0]);
        assert!(hypotheses[0].score > hypotheses[1].score);
    }

    type FixedSequenceRequests = Vec<(Vec<[i64; 2]>, Vec<Vec<usize>>)>;

    #[derive(Default)]
    struct FixedSequenceScoreNetwork {
        frame: usize,
        requests: FixedSequenceRequests,
    }

    impl StatelessRnntSequenceScoreNetwork for FixedSequenceScoreNetwork {
        fn requested_log_probabilities(
            &mut self,
            _encoder_frame: &[f32],
            contexts: &[[i64; 2]],
            requested_token_ids: &[Vec<usize>],
        ) -> Result<Vec<Vec<f32>>> {
            self.requests
                .push((contexts.to_vec(), requested_token_ids.to_vec()));
            let frame = self.frame;
            self.frame += 1;
            Ok(contexts
                .iter()
                .zip(requested_token_ids)
                .map(|(&context, token_ids)| {
                    token_ids
                        .iter()
                        .map(|&token_id| match (frame, context, token_id) {
                            (0, [0, 0], 0) => 0.4_f32.ln(),
                            (0, [0, 0], 1) => 0.6_f32.ln(),
                            (1, [0, 0], 0) => 0.2_f32.ln(),
                            (1, [0, 0], 1) => 0.8_f32.ln(),
                            (1, [0, 1], 0) => 0.7_f32.ln(),
                            _ => 0.01_f32.ln(),
                        })
                        .collect()
                })
                .collect())
        }
    }

    #[test]
    fn fixed_sequence_rescore_sums_all_frame_alignments_and_keeps_the_best_path() {
        let mut network = FixedSequenceScoreNetwork::default();

        let scores =
            score_stateless_sequences(&mut network, &[0.0, 1.0], 2, 1, 0, [0, 0], &[vec![1]])
                .unwrap();

        // Emit at frame 0 then blank: .6 * .7 = .42.
        // Blank at frame 0 then emit: .4 * .8 = .32.
        assert!((scores[0].viterbi_score - 0.42_f32.ln()).abs() < 1.0e-6);
        assert!((scores[0].forward_score - 0.74_f32.ln()).abs() < 1.0e-6);
        assert_eq!(network.requests.len(), 2);
        assert_eq!(network.requests[0].0, vec![[0, 0], [0, 1]]);
        assert_eq!(network.requests[0].1, vec![vec![0, 1], vec![0]]);
    }

    #[test]
    fn fixed_sequence_rescore_shares_a_context_and_requests_each_candidate_token_once() {
        let mut network = FixedSequenceScoreNetwork::default();

        let scores =
            score_stateless_sequences(&mut network, &[0.0], 1, 1, 0, [0, 0], &[vec![1], vec![2]])
                .unwrap();

        assert_eq!(scores.len(), 2);
        assert_eq!(network.requests[0].0[0], [0, 0]);
        assert_eq!(network.requests[0].1[0], vec![0, 1, 2]);
        assert_eq!(
            network.requests[0]
                .0
                .iter()
                .filter(|&&context| context == [0, 0])
                .count(),
            1
        );
    }

    #[test]
    fn monotonic_alignment_sums_every_path_into_token_and_frame_posteriors() {
        let mut network = FixedSequenceScoreNetwork::default();

        let alignments =
            align_stateless_sequences(&mut network, &[0.0, 1.0], 2, 1, 0, [0, 0], &[vec![1]])
                .unwrap();

        let alignment = &alignments[0];
        assert!((alignment.scores.viterbi_score - 0.42_f32.ln()).abs() < 1.0e-6);
        assert!((alignment.scores.forward_score - 0.74_f32.ln()).abs() < 1.0e-6);
        assert_eq!(alignment.tokens.len(), 1);
        let token = &alignment.tokens[0];
        assert_eq!(token.viterbi_frame, 0);
        assert_eq!(token.posterior_mode_frame, 0);
        assert!((token.expected_frame - 0.32 / 0.74).abs() < 1.0e-6);
        assert_eq!(
            (token.posterior_lower_frame, token.posterior_upper_frame),
            (0, 1)
        );
        let expected_probabilities = [0.42_f32 / 0.74, 0.32_f32 / 0.74];
        for (actual, expected) in alignment
            .frame_emission_probabilities
            .iter()
            .zip(expected_probabilities)
        {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        let expected_entropy = -expected_probabilities
            .iter()
            .map(|probability| probability * probability.ln())
            .sum::<f32>();
        assert!((token.entropy - expected_entropy).abs() < 1.0e-6);
    }

    struct UniformSequenceScoreNetwork;

    impl StatelessRnntSequenceScoreNetwork for UniformSequenceScoreNetwork {
        fn requested_log_probabilities(
            &mut self,
            _encoder_frame: &[f32],
            contexts: &[[i64; 2]],
            requested_token_ids: &[Vec<usize>],
        ) -> Result<Vec<Vec<f32>>> {
            assert_eq!(contexts.len(), requested_token_ids.len());
            Ok(requested_token_ids
                .iter()
                .map(|tokens| vec![0.5_f32.ln(); tokens.len()])
                .collect())
        }
    }

    #[test]
    fn repeated_token_occurrences_keep_distinct_monotonic_alignment_positions() {
        let mut network = UniformSequenceScoreNetwork;

        let alignments = align_stateless_sequences(
            &mut network,
            &[0.0, 1.0, 2.0],
            3,
            1,
            0,
            [0, 0],
            &[vec![1, 1]],
        )
        .unwrap();

        let tokens = &alignments[0].tokens;
        assert_eq!(tokens.len(), 2);
        assert!(tokens[0].expected_frame < tokens[1].expected_frame);
        assert!(tokens[0].viterbi_frame < tokens[1].viterbi_frame);
        assert!((tokens[0].expected_frame - 1.0 / 3.0).abs() < 1.0e-6);
        assert!((tokens[1].expected_frame - 5.0 / 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn empty_transcript_alignment_assigns_every_frame_to_blank() {
        let mut network = UniformSequenceScoreNetwork;

        let alignments =
            align_stateless_sequences(&mut network, &[0.0, 1.0], 2, 1, 0, [0, 0], &[vec![]])
                .unwrap();

        assert!(alignments[0].tokens.is_empty());
        assert_eq!(alignments[0].frame_emission_probabilities, vec![0.0, 0.0]);
        assert!((alignments[0].scores.forward_score - 0.25_f32.ln()).abs() < 1.0e-6);
    }

    #[test]
    fn alignment_rejects_a_transcript_longer_than_the_available_frames() {
        let error = align_stateless_sequences(
            &mut UniformSequenceScoreNetwork,
            &[0.0],
            1,
            1,
            0,
            [0, 0],
            &[vec![1, 2]],
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("2 tokens cannot align to 1 frames")
        );
    }

    struct HybridLatticeNetwork;

    impl StatelessRnntSequenceScoreNetwork for HybridLatticeNetwork {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the test encoder stores exact small integer frame IDs as f32"
        )]
        fn requested_log_probabilities(
            &mut self,
            encoder_frame: &[f32],
            contexts: &[[i64; 2]],
            requested_token_ids: &[Vec<usize>],
        ) -> Result<Vec<Vec<f32>>> {
            let frame = encoder_frame[0] as usize;
            Ok(contexts
                .iter()
                .zip(requested_token_ids)
                .map(|(&context, tokens)| {
                    tokens
                        .iter()
                        .map(|&token| match (frame, context, token) {
                            (0, [0, 0], 1) | (1, [0, 1], 2) | (2, [1, 2], 3) | (3, [2, 3], 6) => {
                                -0.1
                            }
                            (1, [0, 5], 2) | (2, [5, 2], 3) => -0.2,
                            (0, [0, 0], 5) | (3, [2, 3], 4) => -2.0,
                            (_, _, 0) => -5.0,
                            _ => -20.0,
                        })
                        .collect()
                })
                .collect())
        }
    }

    #[test]
    fn aligned_lattice_recombines_the_best_prefix_and_suffix_into_a_new_transcript() {
        let first_tokens = [1, 2, 3, 4];
        let second_tokens = [5, 2, 3, 6];
        let frames = [0, 1, 2, 3];
        let mut network = HybridLatticeNetwork;

        let hypotheses = approximate_lattice_from_aligned_sequences(
            &mut network,
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            0,
            [0, 0],
            &[
                StatelessRnntAlignedSeed {
                    token_ids: &first_tokens,
                    emission_frames: &frames,
                },
                StatelessRnntAlignedSeed {
                    token_ids: &second_tokens,
                    emission_frames: &frames,
                },
            ],
            StatelessRnntLatticeArcMerge::Maximum,
        )
        .unwrap();

        assert_eq!(hypotheses[0].token_ids, vec![1, 2, 3, 6]);
        assert!((hypotheses[0].score - -0.4).abs() < 1.0e-6);
        assert!(
            ![first_tokens.as_slice(), second_tokens.as_slice()]
                .contains(&hypotheses[0].token_ids.as_slice())
        );
    }

    #[test]
    fn profiled_search_preserves_hypotheses_and_counts_every_scored_logit() {
        let config = StatelessRnntBeamConfig::model_reference(4, 0);
        let mut reference_network = StatelessNetwork;
        let reference = modified_beam_search_stateless_with_config(
            &mut reference_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
        )
        .unwrap();

        let mut profiled_network = StatelessNetwork;
        let mut profile = StatelessRnntSearchProfile::default();
        let profiled = modified_beam_search_stateless_with_config_and_profile(
            &mut profiled_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
            &mut profile,
        )
        .unwrap();

        assert_eq!(profile.frames, 2);
        assert!(profile.network_context_rows > 0);
        assert_eq!(profile.logit_values, profile.network_context_rows * 3);
        assert!(profile.candidates_generated >= profiled.len());
        assert_eq!(profiled, reference);
    }

    #[test]
    fn cutoff_and_heap_top_tokens_preserve_tied_score_beam_hypotheses() {
        let config = StatelessRnntBeamConfig::model_reference(4, 0);
        let mut reference_network = EqualAlignmentNetwork;
        let reference = modified_beam_search_stateless_with_config(
            &mut reference_network,
            &[0.0, 1.0, 2.0],
            3,
            1,
            2,
            0,
            config,
        )
        .unwrap();

        for algorithm in [
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntTopTokenAlgorithm::WorstFirstHeap,
        ] {
            let mut network = EqualAlignmentNetwork;
            let mut profile = StatelessRnntSearchProfile::default();
            let hypotheses = modified_beam_search_stateless_profiled_with_search_options(
                &mut network,
                &[0.0, 1.0, 2.0],
                3,
                1,
                2,
                0,
                config,
                algorithm,
                StatelessRnntScoreNormalization::SparseLogNormalizer,
                &mut profile,
            )
            .unwrap();

            assert_eq!(hypotheses, reference, "algorithm={algorithm:?}");
        }
    }

    #[test]
    fn every_top_token_algorithm_orders_ties_and_excludes_non_emitting_tokens() {
        let row = [100.0, 3.0, 2.0, 3.0, 1.0, 3.0];
        let expected = vec![(1, 3.0), (3, 3.0), (5, 3.0), (4, 1.0)];

        for algorithm in [
            StatelessRnntTopTokenAlgorithm::BinarySearch,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntTopTokenAlgorithm::WorstFirstHeap,
        ] {
            assert_eq!(
                top_emitting_tokens(&row, 0, Some(2), 4, algorithm),
                expected,
                "algorithm={algorithm:?}"
            );
        }
    }

    #[test]
    fn sparse_log_normalizer_preserves_beam_scores_tokens_and_timestamps() {
        let config = StatelessRnntBeamConfig::model_reference(4, 0);
        let mut full_network = StatelessNetwork;
        let mut full_profile = StatelessRnntSearchProfile::default();
        let full = modified_beam_search_stateless_profiled_with_search_options(
            &mut full_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::FullLogSoftmax,
            &mut full_profile,
        )
        .unwrap();

        let mut sparse_network = StatelessNetwork;
        let mut sparse_profile = StatelessRnntSearchProfile::default();
        let sparse = modified_beam_search_stateless_profiled_with_search_options(
            &mut sparse_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::SparseLogNormalizer,
            &mut sparse_profile,
        )
        .unwrap();

        assert_eq!(sparse, full);
    }

    #[test]
    fn precomputed_log_probabilities_skip_scalar_normalization_and_preserve_search() {
        let config = StatelessRnntBeamConfig::model_reference(4, 0);
        let mut reference_network = StatelessNetwork;
        let mut reference_profile = StatelessRnntSearchProfile::default();
        let reference = modified_beam_search_stateless_profiled_with_search_options(
            &mut reference_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::SparseLogNormalizer,
            &mut reference_profile,
        )
        .unwrap();

        let mut normalized_network = PrecomputedLogProbabilityNetwork;
        let mut precomputed_profile = StatelessRnntSearchProfile::default();
        let precomputed = modified_beam_search_stateless_profiled_with_search_options(
            &mut normalized_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::PrecomputedLogProbabilities,
            &mut precomputed_profile,
        )
        .unwrap();

        assert_eq!(precomputed, reference);
        assert_eq!(precomputed_profile.log_softmax, std::time::Duration::ZERO);
    }

    struct PrecomputedLogProbabilityNetwork;

    impl StatelessRnntNetwork for PrecomputedLogProbabilityNetwork {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the test fixture mirrors the production f64 accumulator followed by f32 scores"
        )]
        fn logits(&mut self, encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            let mut logits = StatelessNetwork.logits(encoder_frame, contexts)?;
            for row in logits.chunks_exact_mut(3) {
                let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let normalizer = row
                    .iter()
                    .map(|score| f64::from(*score - maximum).exp())
                    .sum::<f64>()
                    .ln() as f32
                    + maximum;
                for score in row {
                    *score -= normalizer;
                }
            }
            Ok(logits)
        }
    }

    struct CompactPrecomputedNetwork {
        saw_exact_prefix_gather: bool,
    }

    impl StatelessRnntNetwork for CompactPrecomputedNetwork {
        fn supports_compact_log_probabilities(&self) -> bool {
            true
        }

        fn logits(&mut self, _encoder_frame: &[f32], _contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            anyhow::bail!("dense vocabulary output must not be requested")
        }

        fn compact_log_probabilities(
            &mut self,
            encoder_frame: &[f32],
            contexts: &[[i64; 2]],
            requested_token_ids: &[Vec<usize>],
            emitting_limit: usize,
            blank_id: usize,
            additional_non_emitting_id: Option<usize>,
        ) -> Result<Option<StatelessRnntCompactScores>> {
            self.saw_exact_prefix_gather |= requested_token_ids
                .iter()
                .any(|tokens| tokens.iter().any(|&token| token != blank_id));
            let dense = PrecomputedLogProbabilityNetwork.logits(encoder_frame, contexts)?;
            let top_tokens = dense
                .chunks_exact(3)
                .map(|row| {
                    top_emitting_tokens(
                        row,
                        blank_id,
                        additional_non_emitting_id,
                        emitting_limit,
                        StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
                    )
                })
                .collect::<Vec<_>>();
            let requested_scores = dense
                .chunks_exact(3)
                .zip(requested_token_ids)
                .map(|(row, tokens)| tokens.iter().map(|&token| row[token]).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let top_count = top_tokens.iter().map(Vec::len).sum::<usize>();
            let requested_count = requested_scores.iter().map(Vec::len).sum::<usize>();
            Ok(Some(StatelessRnntCompactScores {
                top_tokens,
                requested_scores,
                network_output_values: top_count * 2 + requested_count,
                network_output_bytes: (top_count + requested_count) * std::mem::size_of::<f32>()
                    + top_count * std::mem::size_of::<i64>(),
            }))
        }
    }

    #[test]
    fn compact_top_k_and_dynamic_gather_preserve_exact_prefix_recombination_without_dense_rows() {
        let config = StatelessRnntBeamConfig::model_reference(4, 0);
        let mut dense_network = PrecomputedLogProbabilityNetwork;
        let mut dense_profile = StatelessRnntSearchProfile::default();
        let dense = modified_beam_search_stateless_profiled_with_search_options(
            &mut dense_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::PrecomputedLogProbabilities,
            &mut dense_profile,
        )
        .unwrap();

        let mut compact_network = CompactPrecomputedNetwork {
            saw_exact_prefix_gather: false,
        };
        let mut compact_profile = StatelessRnntSearchProfile::default();
        let compact = modified_beam_search_stateless_profiled_with_search_options(
            &mut compact_network,
            &[0.0, 1.0],
            2,
            1,
            3,
            0,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::PrecomputedLogProbabilities,
            &mut compact_profile,
        )
        .unwrap();

        assert_eq!(compact, dense);
        assert!(compact_network.saw_exact_prefix_gather);
        assert_eq!(
            compact_profile.top_token_selection,
            std::time::Duration::ZERO
        );
    }

    struct HotwordCompactNetwork {
        requested_below_acoustic_top_k: bool,
    }

    impl StatelessRnntNetwork for HotwordCompactNetwork {
        fn supports_compact_log_probabilities(&self) -> bool {
            true
        }

        fn logits(&mut self, _encoder_frame: &[f32], _contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            anyhow::bail!("hotword compact test must not request dense logits")
        }

        fn compact_log_probabilities(
            &mut self,
            encoder_frame: &[f32],
            contexts: &[[i64; 2]],
            requested_token_ids: &[Vec<usize>],
            emitting_limit: usize,
            blank_id: usize,
            additional_non_emitting_id: Option<usize>,
        ) -> Result<Option<StatelessRnntCompactScores>> {
            let second_frame = encoder_frame[0] > 0.5;
            let dense = contexts
                .iter()
                .flat_map(|context| {
                    if second_frame && context[1] == 3 {
                        [-0.4, -0.5, -0.6, -2.0, -2.0]
                    } else {
                        [-0.2, -0.3, -0.4, -2.0, -2.1]
                    }
                })
                .collect::<Vec<_>>();
            let top_tokens = dense
                .chunks_exact(5)
                .map(|row| {
                    top_emitting_tokens(
                        row,
                        blank_id,
                        additional_non_emitting_id,
                        emitting_limit,
                        StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
                    )
                })
                .collect::<Vec<_>>();
            self.requested_below_acoustic_top_k |=
                requested_token_ids
                    .iter()
                    .zip(&top_tokens)
                    .any(|(requested, top)| {
                        requested.iter().any(|token| {
                            (*token == 3 || *token == 4)
                                && !top.iter().any(|&(top_token, _)| top_token == *token)
                        })
                    });
            let requested_scores = dense
                .chunks_exact(5)
                .zip(requested_token_ids)
                .map(|(row, tokens)| tokens.iter().map(|&token| row[token]).collect())
                .collect::<Vec<Vec<f32>>>();
            Ok(Some(StatelessRnntCompactScores {
                network_output_values: top_tokens.iter().map(Vec::len).sum::<usize>() * 2
                    + requested_scores.iter().map(Vec::len).sum::<usize>(),
                network_output_bytes: 0,
                top_tokens,
                requested_scores,
            }))
        }
    }

    #[test]
    fn hotword_outgoing_tokens_enter_the_beam_below_acoustic_top_k() {
        let graph = HotwordContextGraph::new(vec![vec![3, 4]], 2.0).unwrap();
        let config = StatelessRnntBeamConfig {
            beam_size: 2,
            pruning: StatelessRnntPruning::FullPrefix,
            final_length_normalization: StatelessRnntLengthNormalization::Raw,
            ..StatelessRnntBeamConfig::model_reference(2, 0)
        };
        let mut network = HotwordCompactNetwork {
            requested_below_acoustic_top_k: false,
        };

        let hypotheses = modified_beam_search_stateless_with_hotwords_and_search_options(
            &mut network,
            &[0.0, 1.0],
            2,
            1,
            5,
            0,
            config,
            &graph,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::PrecomputedLogProbabilities,
        )
        .unwrap();

        assert!(network.requested_below_acoustic_top_k);
        assert_eq!(hypotheses[0].token_ids, vec![3, 4]);
    }

    #[test]
    fn epsilon_normalizers_skip_negligible_terms_without_changing_the_transcript() {
        let config = StatelessRnntBeamConfig::model_reference(2, 0);
        let mut reference_network = NegligibleTailNetwork;
        let mut reference_profile = StatelessRnntSearchProfile::default();
        let reference = modified_beam_search_stateless_profiled_with_search_options(
            &mut reference_network,
            &[0.0],
            1,
            1,
            4,
            0,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::SparseLogNormalizer,
            &mut reference_profile,
        )
        .unwrap();

        for normalization in [
            StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e8,
            StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e7,
            StatelessRnntScoreNormalization::SparseLogNormalizerEpsilon1e6,
        ] {
            let mut network = NegligibleTailNetwork;
            let mut profile = StatelessRnntSearchProfile::default();
            let hypotheses = modified_beam_search_stateless_profiled_with_search_options(
                &mut network,
                &[0.0],
                1,
                1,
                4,
                0,
                config,
                StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
                normalization,
                &mut profile,
            )
            .unwrap();

            assert_eq!(
                hypotheses
                    .iter()
                    .map(|hypothesis| (&hypothesis.token_ids, &hypothesis.timestamps))
                    .collect::<Vec<_>>(),
                reference
                    .iter()
                    .map(|hypothesis| (&hypothesis.token_ids, &hypothesis.timestamps))
                    .collect::<Vec<_>>(),
                "normalization={normalization:?}"
            );
            assert!(profile.scalar_exp_terms_skipped > 0);
            assert_eq!(
                profile.scalar_exp_terms_evaluated + profile.scalar_exp_terms_skipped,
                profile.logit_values
            );
        }
    }

    struct NegligibleTailNetwork;

    impl StatelessRnntNetwork for NegligibleTailNetwork {
        fn logits(&mut self, _encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            Ok(contexts
                .iter()
                .flat_map(|_| [-40.0, 0.0, -0.1, -30.0])
                .collect())
        }
    }

    #[test]
    fn modified_beam_rejects_zero_width_before_calling_the_network() {
        let mut network = StatelessNetwork;
        let error =
            modified_beam_search_stateless(&mut network, &[0.0], 1, 1, 3, 0, 0).unwrap_err();

        assert_eq!(error.to_string(), "RNNT beam size must be positive");
    }

    struct EqualAlignmentNetwork;

    impl StatelessRnntNetwork for EqualAlignmentNetwork {
        fn logits(&mut self, _encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            Ok(contexts.iter().flat_map(|_| [0.0, 0.0]).collect())
        }
    }

    struct LengthBiasedNetwork;

    impl StatelessRnntNetwork for LengthBiasedNetwork {
        fn logits(&mut self, _encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            Ok(contexts.iter().flat_map(|_| [0.0, -0.2]).collect())
        }
    }

    #[test]
    fn search_and_final_length_normalization_choose_the_expected_transcript() {
        let cases = [
            (
                1,
                StatelessRnntLengthNormalization::Raw,
                StatelessRnntLengthNormalization::Raw,
                Vec::new(),
            ),
            (
                1,
                StatelessRnntLengthNormalization::PerToken,
                StatelessRnntLengthNormalization::PerToken,
                vec![1],
            ),
            (
                2,
                StatelessRnntLengthNormalization::Raw,
                StatelessRnntLengthNormalization::PerToken,
                vec![1],
            ),
        ];
        for (beam_size, search_normalization, final_normalization, expected) in cases {
            let mut network = LengthBiasedNetwork;
            let config = StatelessRnntBeamConfig {
                beam_size,
                pruning: StatelessRnntPruning::FullPrefix,
                search_length_normalization: search_normalization,
                final_length_normalization: final_normalization,
                ..StatelessRnntBeamConfig::model_reference(beam_size, 0)
            };

            let hypotheses = modified_beam_search_stateless_with_config(
                &mut network,
                &[0.0],
                1,
                1,
                2,
                0,
                config,
            )
            .unwrap();

            assert_eq!(
                hypotheses[0].token_ids, expected,
                "search={search_normalization:?}, final={final_normalization:?}"
            );
        }
    }

    #[test]
    fn same_token_sequence_logadds_early_and_late_emission_and_keeps_stable_alignment() {
        let mut network = EqualAlignmentNetwork;

        let hypotheses =
            modified_beam_search_stateless(&mut network, &[0.0, 1.0], 2, 1, 2, 0, 2).unwrap();

        assert_eq!(hypotheses[0].token_ids, vec![1]);
        assert_eq!(hypotheses[0].timestamps, vec![1]);
        assert!((hypotheses[0].score + std::f32::consts::LN_2).abs() < 1.0e-6);
    }

    struct ConvergingStateNetwork {
        contexts_by_frame: Vec<Vec<[i64; 2]>>,
    }

    impl StatelessRnntNetwork for ConvergingStateNetwork {
        fn logits(&mut self, _encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            let frame = self.contexts_by_frame.len();
            self.contexts_by_frame.push(contexts.to_vec());
            Ok(contexts
                .iter()
                .flat_map(|_| match frame {
                    0 => [-20.0, 2.0, 1.9, -20.0],
                    1 => [-20.0, -20.0, -20.0, 2.0],
                    2 => [-20.0, -20.0, 2.0, -20.0],
                    _ => [2.0, -20.0, -20.0, -20.0],
                })
                .collect())
        }
    }

    #[test]
    fn same_length_histories_with_same_two_token_state_expand_only_the_higher_score_leaf() {
        let mut network = ConvergingStateNetwork {
            contexts_by_frame: Vec::new(),
        };

        let hypotheses =
            modified_beam_search_stateless(&mut network, &[0.0, 1.0, 2.0, 3.0], 4, 1, 4, 0, 4)
                .unwrap();

        assert_eq!(
            network.contexts_by_frame[3]
                .iter()
                .filter(|&&context| context == [3, 2])
                .count(),
            1
        );
        assert_eq!(hypotheses[0].token_ids, vec![1, 3, 2]);
        assert_eq!(hypotheses[0].timestamps, vec![0, 1, 2]);
        assert!(
            !hypotheses
                .iter()
                .any(|hypothesis| hypothesis.token_ids == [2, 3, 2])
        );
    }

    #[test]
    fn hotword_state_keeps_histories_with_the_same_acoustic_state_distinct() {
        let mut network = ConvergingStateNetwork {
            contexts_by_frame: Vec::new(),
        };
        let graph = HotwordContextGraph::new(vec![vec![1, 3, 2, 1]], 0.1).unwrap();
        let config = StatelessRnntBeamConfig {
            beam_size: 8,
            final_length_normalization: StatelessRnntLengthNormalization::Raw,
            ..StatelessRnntBeamConfig::model_reference(8, 0)
        };

        let hypotheses = modified_beam_search_stateless_with_hotwords_and_search_options(
            &mut network,
            &[0.0, 1.0, 2.0],
            3,
            1,
            4,
            0,
            config,
            &graph,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::SparseLogNormalizer,
        )
        .unwrap();

        let token_sequences = hypotheses
            .iter()
            .map(|hypothesis| hypothesis.token_ids.as_slice())
            .collect::<Vec<_>>();
        assert!(token_sequences.contains(&[1, 3, 2].as_slice()));
        assert!(token_sequences.contains(&[2, 3, 2].as_slice()));
    }

    #[test]
    fn repeated_tokens_keep_different_emission_counts_while_sharing_one_network_context() {
        struct RepeatedTokenNetwork {
            contexts_by_frame: Vec<Vec<[i64; 2]>>,
        }

        impl StatelessRnntNetwork for RepeatedTokenNetwork {
            fn logits(
                &mut self,
                _encoder_frame: &[f32],
                contexts: &[[i64; 2]],
            ) -> Result<Vec<f32>> {
                self.contexts_by_frame.push(contexts.to_vec());
                Ok(contexts.iter().flat_map(|_| [0.0, 0.0]).collect())
            }
        }

        let mut network = RepeatedTokenNetwork {
            contexts_by_frame: Vec::new(),
        };
        let hypotheses =
            modified_beam_search_stateless(&mut network, &[0.0; 4], 4, 1, 2, 0, 8).unwrap();

        assert!(
            hypotheses
                .iter()
                .any(|hypothesis| hypothesis.token_ids == [1, 1])
        );
        assert!(
            hypotheses
                .iter()
                .any(|hypothesis| hypothesis.token_ids == [1, 1, 1])
        );
        assert_eq!(
            network.contexts_by_frame[3]
                .iter()
                .filter(|&&context| context == [1, 1])
                .count(),
            1
        );
    }

    struct ContextRecordingNetwork {
        seen: Vec<Vec<[i64; 2]>>,
    }

    impl StatelessRnntNetwork for ContextRecordingNetwork {
        fn logits(&mut self, _encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
            self.seen.push(contexts.to_vec());
            Ok(contexts.iter().flat_map(|_| [2.0, 1.0]).collect())
        }
    }

    #[test]
    fn configured_initial_context_is_the_predictor_input_for_the_first_frame() {
        let mut network = ContextRecordingNetwork { seen: Vec::new() };
        let config = StatelessRnntBeamConfig {
            initial_context: [-1, 0],
            ..StatelessRnntBeamConfig::model_reference(2, 0)
        };

        modified_beam_search_stateless_with_config(&mut network, &[0.0], 1, 1, 2, 0, config)
            .unwrap();

        assert_eq!(network.seen[0], vec![[-1, 0]]);
    }

    #[test]
    fn configured_unknown_token_is_non_emitting_and_combines_with_blank() {
        let mut network = ContextRecordingNetwork { seen: Vec::new() };
        let config = StatelessRnntBeamConfig {
            additional_non_emitting_id: Some(1),
            pruning: StatelessRnntPruning::FullPrefix,
            ..StatelessRnntBeamConfig::model_reference(2, 0)
        };

        let hypotheses =
            modified_beam_search_stateless_with_config(&mut network, &[0.0], 1, 1, 2, 0, config)
                .unwrap();

        assert_eq!(hypotheses.len(), 1);
        assert!(hypotheses[0].token_ids.is_empty());
        assert!((hypotheses[0].score - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn full_prefix_pruning_retains_distinct_histories_with_the_same_predictor_state() {
        let mut network = ConvergingStateNetwork {
            contexts_by_frame: Vec::new(),
        };
        let config = StatelessRnntBeamConfig {
            pruning: StatelessRnntPruning::FullPrefix,
            deduplicate_contexts: false,
            ..StatelessRnntBeamConfig::model_reference(4, 0)
        };

        let hypotheses = modified_beam_search_stateless_with_config(
            &mut network,
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            4,
            0,
            config,
        )
        .unwrap();

        let token_sequences = hypotheses
            .iter()
            .map(|hypothesis| hypothesis.token_ids.as_slice())
            .collect::<Vec<_>>();
        assert!(token_sequences.contains(&[1, 3, 2].as_slice()));
        assert!(token_sequences.contains(&[2, 3, 2].as_slice()));
        assert_eq!(
            network.contexts_by_frame[3]
                .iter()
                .filter(|&&context| context == [3, 2])
                .count(),
            2
        );
    }

    #[test]
    fn sherpa_pruning_does_not_refill_paths_after_blank_and_unknown_merge() {
        struct MergeOrderNetwork;
        impl StatelessRnntNetwork for MergeOrderNetwork {
            fn logits(
                &mut self,
                _encoder_frame: &[f32],
                contexts: &[[i64; 2]],
            ) -> Result<Vec<f32>> {
                Ok(contexts.iter().flat_map(|_| [2.0, 1.9, 1.8]).collect())
            }
        }

        let mut full_prefix_network = MergeOrderNetwork;
        let full_prefix = StatelessRnntBeamConfig {
            additional_non_emitting_id: Some(1),
            pruning: StatelessRnntPruning::FullPrefix,
            ..StatelessRnntBeamConfig::model_reference(2, 0)
        };
        let full_prefix_hypotheses = modified_beam_search_stateless_with_config(
            &mut full_prefix_network,
            &[0.0],
            1,
            1,
            3,
            0,
            full_prefix,
        )
        .unwrap();

        let mut sherpa_network = MergeOrderNetwork;
        let sherpa = StatelessRnntBeamConfig {
            additional_non_emitting_id: Some(1),
            pruning: StatelessRnntPruning::Sherpa,
            ..StatelessRnntBeamConfig::model_reference(2, 0)
        };
        let sherpa_hypotheses = modified_beam_search_stateless_with_config(
            &mut sherpa_network,
            &[0.0],
            1,
            1,
            3,
            0,
            sherpa,
        )
        .unwrap();

        assert_eq!(full_prefix_hypotheses.len(), 2);
        assert_eq!(full_prefix_hypotheses[1].token_ids, vec![2]);
        assert_eq!(sherpa_hypotheses.len(), 1);
        assert!(sherpa_hypotheses[0].token_ids.is_empty());
    }

    #[test]
    fn sherpa_pruning_rejects_search_time_length_ranking_it_cannot_reproduce() {
        let mut network = EqualAlignmentNetwork;
        let config = StatelessRnntBeamConfig {
            pruning: StatelessRnntPruning::Sherpa,
            search_length_normalization: StatelessRnntLengthNormalization::PerToken,
            ..StatelessRnntBeamConfig::model_reference(2, 0)
        };

        let error =
            modified_beam_search_stateless_with_config(&mut network, &[0.0], 1, 1, 2, 0, config)
                .unwrap_err();

        assert_eq!(
            error.to_string(),
            "sherpa-compatible pruning supports raw search ranking only"
        );
    }
}
