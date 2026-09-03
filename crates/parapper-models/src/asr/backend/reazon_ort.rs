// Audio-sized tensor dimensions fit i64, and transcript timestamps use f32 seconds.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    editor::{Graph, Model, Node, ONNX_DOMAIN, Opset},
    operator::Attribute,
    session::{HasSelectedOutputs, OutputSelector, RunOptions, Session, builder::SessionBuilder},
    value::{Outlet, Shape, SymbolicDimensions, Tensor, TensorElementType, ValueType},
};

use crate::decoder::{
    hotword::{
        HotwordContextGraph, HotwordEntry, HotwordPathKind, HotwordTokenPath, normalize_reading,
    },
    rnnt::modified_beam_search_stateless_with_hotwords_and_search_options,
};
use crate::{
    AsrEngine, AsrModel, AsrPrecision, AsrTranscript,
    decoder::rnnt::{
        StatelessRnntAlignedSeed, StatelessRnntBeamConfig, StatelessRnntCompactScores,
        StatelessRnntHypothesis, StatelessRnntLatticeArcMerge, StatelessRnntLengthNormalization,
        StatelessRnntNetwork, StatelessRnntPruning, StatelessRnntScoreNormalization,
        StatelessRnntSearchProfile, StatelessRnntSequenceScoreNetwork,
        StatelessRnntTopTokenAlgorithm, align_stateless_sequences,
        approximate_lattice_from_aligned_sequences,
        modified_beam_search_stateless_profiled_with_search_options,
        modified_beam_search_stateless_with_config_and_search_options, score_stateless_sequences,
    },
    frontend::KaldiFbankFrontend,
    init_onnx_runtime,
};

use super::reazon_rerank::{
    EvidenceSeed, STATIC_EMBEDDING_DIR_NAME, StaticEmbeddingModel, select_one_splice_candidate,
};
use super::{ReazonBeamAblation, ReazonBeamPruning, ReazonInitialContext, ReazonNetworkBatching};

const BLANK_ID: usize = 0;
const UNKNOWN_ID: usize = 5_222;
const SOS_EOS_ID: usize = 5_223;
const VOCAB_SIZE: usize = 5_224;
const ENCODER_DIM: usize = 512;
const MEL_BINS: usize = 80;
const FRAME_SECONDS: f32 = 0.04;
const RAW_JOINER_OUTPUT: &str = "logit";
const LOG_PROBABILITY_JOINER_OUTPUT: &str = "log_probability";
const TOP_K_VALUES_OUTPUT: &str = "top_k_values";
const TOP_K_INDICES_OUTPUT: &str = "top_k_indices";
const TOP_K_COUNT_INPUT: &str = "top_k_count";
const REQUESTED_TOKEN_IDS_INPUT: &str = "requested_token_ids";
const REQUESTED_SCORES_OUTPUT: &str = "requested_scores";
/// Product default: completing a registered hotword multiplies its
/// likelihood by 100, independent of token length.
pub const DEFAULT_HOTWORD_PHRASE_MULTIPLIER: f32 = 100.0;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReazonDecodingStrategy {
    #[default]
    Greedy,
    ModifiedBeam {
        beam_size: usize,
    },
    OneSpliceRerank {
        beam_size: usize,
        retained_candidates: usize,
    },
    ModifiedBeamAblation {
        config: ReazonBeamAblation,
    },
}

/// Encoder output that can be decoded repeatedly with different `ReazonSpeech`
/// search strategies.
///
/// Values are kept private so callers cannot construct an output whose frame
/// count disagrees with the pinned encoder dimension.
#[derive(Debug, Clone)]
pub struct ReazonEncodedAudio {
    values: Vec<f32>,
    frames: usize,
}

/// One reconstructed transcript candidate from diagnostic full-prefix search.
#[derive(Debug, Clone, PartialEq)]
pub struct ReazonNBestCandidate {
    pub raw_score: f32,
    pub token_ids: Vec<usize>,
    pub timestamps: Vec<f32>,
    pub text: String,
}

/// A full-prefix candidate rescored over every alignment in the
/// one-symbol-per-frame graph used by modified beam search.
#[derive(Debug, Clone, PartialEq)]
pub struct ReazonRescoredNBestCandidate {
    pub candidate: ReazonNBestCandidate,
    pub viterbi_score: f32,
    pub forward_score: f32,
}

/// Posterior timing statistics for one token occurrence in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct ReazonTokenAlignment {
    pub viterbi_timestamp: f32,
    pub posterior_mode_timestamp: f32,
    pub expected_timestamp: f32,
    pub posterior_lower_timestamp: f32,
    pub posterior_upper_timestamp: f32,
    /// Entropy of the marginal frame distribution, in nats.
    pub entropy: f32,
}

/// One N-best transcript with an exact fixed-sequence monotonic alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct ReazonAlignedNBestCandidate {
    pub candidate: ReazonNBestCandidate,
    pub viterbi_score: f32,
    pub forward_score: f32,
    pub tokens: Vec<ReazonTokenAlignment>,
    pub frame_emission_probabilities: Vec<f32>,
}

/// One terminal path produced by the approximate time-free lattice.
#[derive(Debug, Clone, PartialEq)]
pub struct ReazonApproximateLatticeCandidate {
    pub score: f32,
    pub token_ids: Vec<usize>,
    pub text: String,
    pub is_seed: bool,
}

/// Width-N seeds and paths recovered after context-state recombination.
#[derive(Debug, Clone, PartialEq)]
pub struct ReazonApproximateLatticeResult {
    pub seeds: Vec<ReazonNBestCandidate>,
    pub candidates: Vec<ReazonApproximateLatticeCandidate>,
}

#[derive(Default)]
struct PredictorCache {
    outputs: HashMap<[i64; 2], Vec<f32>>,
}

impl PredictorCache {
    fn get_or_try_insert_with<F>(&mut self, context: [i64; 2], mut compute: F) -> Result<&[f32]>
    where
        F: FnMut([i64; 2]) -> Result<Vec<f32>>,
    {
        match self.outputs.entry(context) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
            std::collections::hash_map::Entry::Vacant(entry) => Ok(entry.insert(compute(context)?)),
        }
    }
}

pub struct ReazonSpeechOrtAsrEngine {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    frontend: KaldiFbankFrontend,
    tokens: Vec<String>,
    decoding: ReazonDecodingStrategy,
    predictor_cache: PredictorCache,
    joiner_output_name: &'static str,
    joiner_run_options: RunOptions<HasSelectedOutputs>,
    compact_joiner_run_options: Option<RunOptions<HasSelectedOutputs>>,
    rescore_joiner_run_options: Option<RunOptions<HasSelectedOutputs>>,
    static_embedding: Option<StaticEmbeddingModel>,
    hotwords: Vec<HotwordEntry>,
}

impl ReazonSpeechOrtAsrEngine {
    /// Loads the pinned `ReazonSpeech` K2 V2 export.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported model configuration, missing or invalid model
    /// artifacts, or an ONNX Runtime session initialization failure.
    pub fn new(
        model_dir: &Path,
        model: AsrModel,
        precision: AsrPrecision,
        num_threads: i32,
    ) -> Result<Self> {
        Self::new_with_decoding(
            model_dir,
            model,
            precision,
            num_threads,
            ReazonDecodingStrategy::Greedy,
        )
    }

    /// Loads the pinned `ReazonSpeech` K2 V2 export with an explicit decoder.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid beam size, unsupported model
    /// configuration, invalid artifacts, or ONNX Runtime initialization
    /// failure.
    pub fn new_with_decoding(
        model_dir: &Path,
        model: AsrModel,
        precision: AsrPrecision,
        num_threads: i32,
        decoding: ReazonDecodingStrategy,
    ) -> Result<Self> {
        Self::new_with_decoding_and_hotwords(
            model_dir,
            model,
            precision,
            num_threads,
            decoding,
            &[],
        )
    }

    /// Loads `ReazonSpeech` with engine-owned hotwords. Hotwords are applied by
    /// the production `recognize()` path. One-splice decoding is intentionally
    /// kept as an explicit future integration: hotword decoding currently uses
    /// the configured beam width. In one-splice mode the hotword-biased seeds
    /// are passed through the existing static-embedding reranker before the
    /// surface replacement step.
    ///
    /// # Errors
    ///
    /// Returns an error when model loading, hotword validation, or ONNX Runtime
    /// session initialization fails.
    pub fn new_with_decoding_and_hotwords(
        model_dir: &Path,
        model: AsrModel,
        precision: AsrPrecision,
        num_threads: i32,
        decoding: ReazonDecodingStrategy,
        hotwords: &[HotwordEntry],
    ) -> Result<Self> {
        if model != AsrModel::ReazonSpeechK2V2 {
            bail!("direct ReazonSpeech backend does not support {model:?}");
        }
        validate_decoding_strategy(decoding)?;
        let threads = usize::try_from(num_threads)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| anyhow!("ASR thread count must be greater than zero"))?;
        let (encoder_file, decoder_file, joiner_file) = model_files(precision);
        init_onnx_runtime();
        let encoder = load_session(
            &model_dir.join(encoder_file),
            threads,
            "ReazonSpeech encoder",
        )?;
        let decoder = load_session(&model_dir.join(decoder_file), 1, "ReazonSpeech decoder")?;
        let joiner_path = model_dir.join(joiner_file);
        let append_log_softmax = !matches!(decoding, ReazonDecodingStrategy::Greedy);
        let compact_output = matches!(
            decoding,
            ReazonDecodingStrategy::ModifiedBeam { .. }
                | ReazonDecodingStrategy::OneSpliceRerank { .. }
        );
        let (joiner, joiner_output_name) = if append_log_softmax {
            (
                load_joiner_with_log_softmax(
                    &joiner_path,
                    1,
                    "ReazonSpeech joiner",
                    compact_output,
                )?,
                LOG_PROBABILITY_JOINER_OUTPUT,
            )
        } else {
            (
                load_session(&joiner_path, 1, "ReazonSpeech joiner")?,
                RAW_JOINER_OUTPUT,
            )
        };
        // `ort` rc.12 snapshots I/O metadata before an editable session is
        // finalized, so an in-memory extension still reports the original
        // output here. Inference explicitly selects the actual updated output.
        let reported_joiner_output = if append_log_softmax {
            RAW_JOINER_OUTPUT
        } else {
            joiner_output_name
        };
        validate_contract(&encoder, &decoder, &joiner, reported_joiner_output)?;
        let joiner_run_options = RunOptions::new()
            .map_err(|error| anyhow!("failed to create ReazonSpeech joiner run options: {error}"))?
            .with_outputs(OutputSelector::no_default().with(joiner_output_name));
        let compact_joiner_run_options = compact_output
            .then(|| {
                RunOptions::new()
                    .map(|options| {
                        options.with_outputs(
                            OutputSelector::no_default()
                                .with(TOP_K_VALUES_OUTPUT)
                                .with(TOP_K_INDICES_OUTPUT)
                                .with(REQUESTED_SCORES_OUTPUT),
                        )
                    })
                    .map_err(|error| {
                        anyhow!("failed to create ReazonSpeech compact joiner options: {error}")
                    })
            })
            .transpose()?;
        let rescore_joiner_run_options = compact_output
            .then(|| {
                RunOptions::new()
                    .map(|options| {
                        options.with_outputs(
                            OutputSelector::no_default().with(REQUESTED_SCORES_OUTPUT),
                        )
                    })
                    .map_err(|error| {
                        anyhow!("failed to create ReazonSpeech rescore options: {error}")
                    })
            })
            .transpose()?;
        let static_embedding = matches!(decoding, ReazonDecodingStrategy::OneSpliceRerank { .. })
            .then(|| StaticEmbeddingModel::load(&model_dir.join(STATIC_EMBEDDING_DIR_NAME)))
            .transpose()?;
        Ok(Self {
            encoder,
            decoder,
            joiner,
            frontend: KaldiFbankFrontend::new(),
            tokens: load_tokens(&model_dir.join("tokens.txt"))?,
            decoding,
            predictor_cache: PredictorCache::default(),
            joiner_output_name,
            joiner_run_options,
            compact_joiner_run_options,
            rescore_joiner_run_options,
            static_embedding,
            hotwords: hotwords.to_vec(),
        })
    }

    /// Runs the frontend and encoder once so the output can be shared by
    /// several decoder strategies.
    ///
    /// # Errors
    ///
    /// Returns an error when feature extraction or encoder inference fails, or
    /// when the pinned encoder output contract has changed.
    pub fn encode(&mut self, samples: &[f32]) -> Result<ReazonEncodedAudio> {
        let features = self.frontend.process(samples)?;
        if features.frames == 0 {
            return Ok(ReazonEncodedAudio {
                values: Vec::new(),
                frames: 0,
            });
        }
        let signal = Tensor::from_array((
            vec![1_i64, features.frames as i64, MEL_BINS as i64],
            features.values,
        ))?;
        let length = Tensor::from_array((vec![1_i64], vec![features.frames as i64]))?;
        let outputs = self
            .encoder
            .run(ort::inputs!["x" => signal, "x_lens" => length])?;
        let (shape, encoded) = outputs
            .get("encoder_out")
            .ok_or_else(|| anyhow!("ReazonSpeech encoder did not return encoder_out"))?
            .try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] != ENCODER_DIM as i64 {
            bail!("unexpected ReazonSpeech encoder shape: {shape:?}");
        }
        let all_frames = usize::try_from(shape[1]).context("invalid ReazonSpeech output frames")?;
        let frames = outputs
            .get("encoder_out_lens")
            .ok_or_else(|| anyhow!("ReazonSpeech encoder did not return lengths"))?
            .try_extract_tensor::<i64>()?
            .1
            .first()
            .copied()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| anyhow!("invalid ReazonSpeech encoded length"))?;
        if frames > all_frames {
            bail!("ReazonSpeech encoded length exceeds output");
        }
        let values = encoded.to_vec();
        drop(outputs);
        Ok(ReazonEncodedAudio { values, frames })
    }

    /// Decodes a previously encoded utterance without running the frontend or
    /// encoder again.
    ///
    /// Predictor outputs are cached by the two-token stateless context for the
    /// lifetime of this engine and therefore shared across utterances and beam
    /// widths when cached-unique batching is selected.
    ///
    /// # Errors
    ///
    /// Returns an error when decoder or joiner inference fails, the requested
    /// beam configuration is invalid, or an ONNX output contract has changed.
    pub fn decode_encoded(
        &mut self,
        encoded: &ReazonEncodedAudio,
        strategy: ReazonDecodingStrategy,
    ) -> Result<AsrTranscript> {
        if !self.hotwords.is_empty() {
            if let ReazonDecodingStrategy::OneSpliceRerank {
                beam_size,
                retained_candidates,
            } = strategy
            {
                let paths = self.tokenize_hotword_entries(&self.hotwords)?;
                let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
                    paths,
                    DEFAULT_HOTWORD_PHRASE_MULTIPLIER,
                )?;
                return self.decode_one_splice_reranked_with_hotwords(
                    encoded,
                    beam_size,
                    retained_candidates,
                    Some(&graph),
                );
            }
            let beam_size = match strategy {
                ReazonDecodingStrategy::ModifiedBeam { beam_size }
                | ReazonDecodingStrategy::OneSpliceRerank { beam_size, .. } => beam_size,
                ReazonDecodingStrategy::ModifiedBeamAblation { config } => config.beam_size,
                ReazonDecodingStrategy::Greedy => {
                    bail!("ReazonSpeech hotwords require a beam decoding strategy")
                }
            };
            let hotwords = self.hotwords.clone();
            return self.decode_encoded_with_hotword_entries_phrase_multiplier(
                encoded,
                beam_size,
                &hotwords,
                DEFAULT_HOTWORD_PHRASE_MULTIPLIER,
            );
        }
        self.decode_encoded_internal(encoded, strategy, None)
    }

    /// Decodes cached encoder output with per-token context bias for hotwords.
    ///
    /// The configured strings are converted with the pinned `ReazonSpeech`
    /// character token table. Unknown characters are rejected rather than
    /// silently becoming `<unk>`. This method requires an engine constructed
    /// with a beam strategy so the compact LogSoftmax/TopK/Gather outputs are
    /// available.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero beam, an empty or out-of-vocabulary hotword,
    /// an invalid score, a greedy-only joiner session, or inference failure.
    pub fn decode_encoded_with_hotwords(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        hotwords: &[String],
        token_score: f32,
    ) -> Result<AsrTranscript> {
        let entries = hotwords
            .iter()
            .map(|surface| HotwordEntry {
                surface: surface.clone(),
                readings: Vec::new(),
                phrase_score: None,
            })
            .collect::<Vec<_>>();
        self.decode_encoded_with_hotword_entries(encoded, beam_size, &entries, token_score)
    }

    /// Decodes cached encoder output with surface and reading-aware hotwords.
    /// Reading paths are normalized to hiragana and both hiragana and
    /// katakana token paths are added to the context graph. A completed
    /// reading is rewritten to its configured surface after beam search.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid hotwords, unsupported beam configuration,
    /// missing compact joiner outputs, or model inference failure.
    pub fn decode_encoded_with_hotword_entries(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        hotwords: &[HotwordEntry],
        token_score: f32,
    ) -> Result<AsrTranscript> {
        if beam_size == 0 {
            bail!("ReazonSpeech hotword beam size must be positive");
        }
        if hotwords.is_empty() {
            bail!("ReazonSpeech hotwords must not be empty");
        }
        let token_paths = self.tokenize_hotword_entries(hotwords)?;
        let graph = HotwordContextGraph::from_token_paths(token_paths, token_score)?;
        self.decode_encoded_with_hotword_graph(encoded, beam_size, &graph)
    }

    /// Decodes cached encoder output while assigning every completed hotword
    /// one phrase-level likelihood multiplier, independent of token length.
    /// The log multiplier is distributed over the known path length while the
    /// prefix is active, then corrected to the exact phrase total at completion.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid hotwords, a non-positive multiplier,
    /// unsupported beam configuration, or model inference failure.
    pub fn decode_encoded_with_hotwords_phrase_multiplier(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        hotwords: &[String],
        phrase_multiplier: f32,
    ) -> Result<AsrTranscript> {
        if beam_size == 0 {
            bail!("ReazonSpeech hotword beam size must be positive");
        }
        if hotwords.is_empty() {
            bail!("ReazonSpeech hotwords must not be empty");
        }
        let entries = hotwords
            .iter()
            .map(|surface| HotwordEntry {
                surface: surface.clone(),
                readings: Vec::new(),
                phrase_score: None,
            })
            .collect::<Vec<_>>();
        self.decode_encoded_with_hotword_entries_phrase_multiplier(
            encoded,
            beam_size,
            &entries,
            phrase_multiplier,
        )
    }

    /// Decodes cached encoder output with surface/reading hotwords and one
    /// phrase-level likelihood multiplier per completed path.
    ///
    /// Unlike the string-only wrapper, a neutral multiplier still performs
    /// reading-to-surface rendering when the acoustic result already contains
    /// a registered reading.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid entries, a non-positive multiplier,
    /// unsupported beam configuration, or model inference failure.
    pub fn decode_encoded_with_hotword_entries_phrase_multiplier(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        hotwords: &[HotwordEntry],
        phrase_multiplier: f32,
    ) -> Result<AsrTranscript> {
        if beam_size == 0 {
            bail!("ReazonSpeech hotword beam size must be positive");
        }
        if hotwords.is_empty() {
            bail!("ReazonSpeech hotwords must not be empty");
        }
        let token_paths = self.tokenize_hotword_entries(hotwords)?;
        let graph = HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            token_paths,
            phrase_multiplier,
        )?;
        self.decode_encoded_with_hotword_graph(encoded, beam_size, &graph)
    }

    fn decode_encoded_with_hotword_graph(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        graph: &HotwordContextGraph,
    ) -> Result<AsrTranscript> {
        if encoded.frames == 0 {
            return Ok(AsrTranscript::from_text(""));
        }
        let compact_options = self.compact_joiner_run_options.as_ref().ok_or_else(|| {
            anyhow!("ReazonSpeech hotwords require an engine constructed with beam decoding")
        })?;
        let mut network = ReazonBeamNetwork {
            decoder: &mut self.decoder,
            joiner: &mut self.joiner,
            predictor_cache: &mut self.predictor_cache,
            batching: ReazonNetworkBatching::CachedUnique,
            joiner_output_name: self.joiner_output_name,
            joiner_run_options: &self.joiner_run_options,
            compact_joiner_run_options: Some(compact_options),
        };
        let production = ReazonBeamAblation::production(beam_size);
        let config = StatelessRnntBeamConfig {
            pruning: StatelessRnntPruning::FullPrefix,
            ..ablation_decoder_config(production)
        };
        let best = modified_beam_search_stateless_with_hotwords_and_search_options(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            VOCAB_SIZE,
            BLANK_ID,
            config,
            graph,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            StatelessRnntScoreNormalization::PrecomputedLogProbabilities,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("ReazonSpeech hotword beam search returned no hypotheses"))?;
        Ok(transcript_from_tokens_with_hotwords(
            &self.tokens,
            &best.token_ids,
            Some(&best.timestamps),
            graph,
        ))
    }

    fn tokenize_hotword_entries(&self, entries: &[HotwordEntry]) -> Result<Vec<HotwordTokenPath>> {
        let token_ids = self
            .tokens
            .iter()
            .enumerate()
            .map(|(token_id, token)| (token.as_str(), token_id))
            .collect::<HashMap<_, _>>();
        tokenize_reazon_hotword_entries(entries, &token_ids)
    }

    /// Decodes one cached encoder output without state-dominance history loss.
    ///
    /// Exact token prefixes remain distinct beam entries, while equal
    /// predictor contexts still share decoder/joiner rows. This diagnostic API
    /// returns every surviving candidate so final length normalization and
    /// oracle metrics can be evaluated without another model invocation.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero beam, empty model output contract, or model
    /// inference failure.
    pub fn decode_encoded_nbest(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        search_length_normalization: StatelessRnntLengthNormalization,
    ) -> Result<Vec<ReazonNBestCandidate>> {
        self.decode_encoded_nbest_with_hotwords(
            encoded,
            beam_size,
            search_length_normalization,
            None,
        )
    }

    fn decode_encoded_nbest_with_hotwords(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        search_length_normalization: StatelessRnntLengthNormalization,
        hotwords: Option<&HotwordContextGraph>,
    ) -> Result<Vec<ReazonNBestCandidate>> {
        if beam_size == 0 {
            bail!("ReazonSpeech N-best beam size must be positive");
        }
        if encoded.frames == 0 {
            return Ok(vec![ReazonNBestCandidate {
                raw_score: 0.0,
                token_ids: Vec::new(),
                timestamps: Vec::new(),
                text: String::new(),
            }]);
        }
        let mut config = ablation_decoder_config(ReazonBeamAblation::production(beam_size));
        config.pruning = StatelessRnntPruning::FullPrefix;
        config.search_length_normalization = search_length_normalization;
        config.final_length_normalization = search_length_normalization;
        let score_normalization = self.default_score_normalization();
        let hypotheses = search_modified_beam(
            &mut self.decoder,
            &mut self.joiner,
            &mut self.predictor_cache,
            self.joiner_output_name,
            &self.joiner_run_options,
            self.compact_joiner_run_options.as_ref(),
            &encoded.values,
            encoded.frames,
            config,
            ReazonNetworkBatching::CachedUnique,
            score_normalization,
            None,
            hotwords,
        )?;
        Ok(hypotheses
            .into_iter()
            .map(|hypothesis| {
                let text = hypothesis
                    .token_ids
                    .iter()
                    .map(|&id| self.tokens[id].as_str())
                    .collect::<String>();
                ReazonNBestCandidate {
                    raw_score: hypothesis.score,
                    token_ids: hypothesis.token_ids,
                    timestamps: hypothesis
                        .timestamps
                        .into_iter()
                        .map(|frame| frame as f32 * FRAME_SECONDS)
                        .collect(),
                    text,
                }
            })
            .collect())
    }

    /// Generates full-prefix N-best candidates, then recomputes their fixed
    /// transcript scores without beam pruning.
    ///
    /// Decoder outputs are shared by two-token context. At each encoder frame,
    /// the joiner receives every unique context needed by the N-best set and
    /// returns only blank and requested next-token log probabilities. Viterbi
    /// keeps the best alignment; forward log-adds every alignment allowed by
    /// modified beam search's one-symbol-per-frame graph.
    ///
    /// # Errors
    ///
    /// Returns an error when N-best generation fails, the engine was not
    /// constructed with the compact modified-beam graph, or rescoring model
    /// output no longer matches the requested layout.
    pub fn decode_encoded_nbest_rescored(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        search_length_normalization: StatelessRnntLengthNormalization,
    ) -> Result<Vec<ReazonRescoredNBestCandidate>> {
        let candidates =
            self.decode_encoded_nbest(encoded, beam_size, search_length_normalization)?;
        if encoded.frames == 0 {
            return Ok(candidates
                .into_iter()
                .map(|candidate| ReazonRescoredNBestCandidate {
                    candidate,
                    viterbi_score: 0.0,
                    forward_score: 0.0,
                })
                .collect());
        }
        let run_options = self.rescore_joiner_run_options.as_ref().ok_or_else(|| {
            anyhow!("ReazonSpeech fixed-sequence rescoring requires a compact beam joiner")
        })?;
        let sequences = candidates
            .iter()
            .map(|candidate| candidate.token_ids.clone())
            .collect::<Vec<_>>();
        let mut network = ReazonSequenceScoreNetwork {
            decoder: &mut self.decoder,
            joiner: &mut self.joiner,
            predictor_cache: &mut self.predictor_cache,
            run_options,
        };
        let scores = score_stateless_sequences(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
            [BLANK_ID as i64; 2],
            &sequences,
        )?;
        Ok(candidates
            .into_iter()
            .zip(scores)
            .map(|(candidate, scores)| ReazonRescoredNBestCandidate {
                candidate,
                viterbi_score: scores.viterbi_score,
                forward_score: scores.forward_score,
            })
            .collect())
    }

    /// Generates full-prefix N-best candidates and computes exact monotonic
    /// token timing posteriors for each fixed transcript.
    ///
    /// The forward-backward pass sums every alignment in modified beam
    /// search's one-symbol-per-frame graph. Repeated token IDs remain distinct
    /// because each posterior belongs to an output position. Decoder contexts
    /// and requested joiner columns are shared across all candidates.
    ///
    /// # Errors
    ///
    /// Returns an error when N-best generation or selected-score inference
    /// fails, or when a candidate cannot align to the encoder frames.
    pub fn decode_encoded_nbest_aligned(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        search_length_normalization: StatelessRnntLengthNormalization,
    ) -> Result<Vec<ReazonAlignedNBestCandidate>> {
        let candidates =
            self.decode_encoded_nbest(encoded, beam_size, search_length_normalization)?;
        if encoded.frames == 0 {
            return Ok(candidates
                .into_iter()
                .map(|candidate| ReazonAlignedNBestCandidate {
                    candidate,
                    viterbi_score: 0.0,
                    forward_score: 0.0,
                    tokens: Vec::new(),
                    frame_emission_probabilities: Vec::new(),
                })
                .collect());
        }
        let run_options = self.rescore_joiner_run_options.as_ref().ok_or_else(|| {
            anyhow!("ReazonSpeech monotonic alignment requires a compact beam joiner")
        })?;
        let sequences = candidates
            .iter()
            .map(|candidate| candidate.token_ids.clone())
            .collect::<Vec<_>>();
        let mut network = ReazonSequenceScoreNetwork {
            decoder: &mut self.decoder,
            joiner: &mut self.joiner,
            predictor_cache: &mut self.predictor_cache,
            run_options,
        };
        let alignments = align_stateless_sequences(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
            [BLANK_ID as i64; 2],
            &sequences,
        )?;
        Ok(candidates
            .into_iter()
            .zip(alignments)
            .map(|(candidate, alignment)| ReazonAlignedNBestCandidate {
                candidate,
                viterbi_score: alignment.scores.viterbi_score,
                forward_score: alignment.scores.forward_score,
                tokens: alignment
                    .tokens
                    .into_iter()
                    .map(|token| ReazonTokenAlignment {
                        viterbi_timestamp: token.viterbi_frame as f32 * FRAME_SECONDS,
                        posterior_mode_timestamp: token.posterior_mode_frame as f32 * FRAME_SECONDS,
                        expected_timestamp: token.expected_frame * FRAME_SECONDS,
                        posterior_lower_timestamp: token.posterior_lower_frame as f32
                            * FRAME_SECONDS,
                        posterior_upper_timestamp: token.posterior_upper_frame as f32
                            * FRAME_SECONDS,
                        entropy: token.entropy,
                    })
                    .collect(),
                frame_emission_probabilities: alignment.frame_emission_probabilities,
            })
            .collect())
    }

    /// Projects full-prefix N-best alignments onto an approximate time-free
    /// lattice and recovers the best path to every observed terminal state.
    ///
    /// # Errors
    ///
    /// Returns an error when N-best generation fails, timestamps violate the
    /// one-symbol-per-frame contract, the compact joiner is unavailable, or a
    /// model output shape changes.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "timestamps originate from bounded nonnegative encoder frame indices"
    )]
    pub fn decode_encoded_approximate_lattice(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        search_length_normalization: StatelessRnntLengthNormalization,
        merge: StatelessRnntLatticeArcMerge,
    ) -> Result<ReazonApproximateLatticeResult> {
        let seeds = self.decode_encoded_nbest(encoded, beam_size, search_length_normalization)?;
        if encoded.frames == 0 {
            return Ok(ReazonApproximateLatticeResult {
                candidates: vec![ReazonApproximateLatticeCandidate {
                    score: 0.0,
                    token_ids: Vec::new(),
                    text: String::new(),
                    is_seed: true,
                }],
                seeds,
            });
        }
        let run_options = self.rescore_joiner_run_options.as_ref().ok_or_else(|| {
            anyhow!("ReazonSpeech approximate lattice requires a compact beam joiner")
        })?;
        let emission_frames = seeds
            .iter()
            .map(|candidate| {
                candidate
                    .timestamps
                    .iter()
                    .map(|timestamp| (timestamp / FRAME_SECONDS).round() as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let aligned = seeds
            .iter()
            .zip(&emission_frames)
            .map(|(candidate, frames)| StatelessRnntAlignedSeed {
                token_ids: &candidate.token_ids,
                emission_frames: frames,
            })
            .collect::<Vec<_>>();
        let mut network = ReazonSequenceScoreNetwork {
            decoder: &mut self.decoder,
            joiner: &mut self.joiner,
            predictor_cache: &mut self.predictor_cache,
            run_options,
        };
        let hypotheses = approximate_lattice_from_aligned_sequences(
            &mut network,
            &encoded.values,
            encoded.frames,
            ENCODER_DIM,
            BLANK_ID,
            [BLANK_ID as i64; 2],
            &aligned,
            merge,
        )?;
        let candidates = hypotheses
            .into_iter()
            .map(|hypothesis| {
                let is_seed = seeds
                    .iter()
                    .any(|seed| seed.token_ids == hypothesis.token_ids);
                let text = hypothesis
                    .token_ids
                    .iter()
                    .map(|&id| self.tokens[id].as_str())
                    .collect::<String>();
                ReazonApproximateLatticeCandidate {
                    score: hypothesis.score,
                    token_ids: hypothesis.token_ids,
                    text,
                    is_seed,
                }
            })
            .collect();
        Ok(ReazonApproximateLatticeResult { seeds, candidates })
    }

    /// Decodes one cached encoder output and records search sub-stage timings.
    ///
    /// This diagnostic entry point accepts modified beam search only. The
    /// ordinary decoder remains clock-free.
    ///
    /// # Errors
    ///
    /// Returns an error for greedy decoding, an invalid beam, model inference
    /// failure, or an ONNX output contract change.
    pub fn decode_encoded_with_search_profile(
        &mut self,
        encoded: &ReazonEncodedAudio,
        strategy: ReazonDecodingStrategy,
        profile: &mut StatelessRnntSearchProfile,
    ) -> Result<AsrTranscript> {
        if !matches!(
            strategy,
            ReazonDecodingStrategy::ModifiedBeam { .. }
                | ReazonDecodingStrategy::ModifiedBeamAblation { .. }
        ) {
            bail!("ReazonSpeech search profiling requires modified beam decoding");
        }
        // Match production recognize(): predictor entries are shared inside
        // one utterance but never leak into the next measured utterance.
        self.predictor_cache.outputs.clear();
        self.decode_encoded_internal(encoded, strategy, Some(profile))
    }

    fn decode_encoded_internal(
        &mut self,
        encoded: &ReazonEncodedAudio,
        strategy: ReazonDecodingStrategy,
        mut profile: Option<&mut StatelessRnntSearchProfile>,
    ) -> Result<AsrTranscript> {
        if encoded.frames == 0 {
            return Ok(AsrTranscript::from_text(""));
        }
        if let ReazonDecodingStrategy::OneSpliceRerank {
            beam_size,
            retained_candidates,
        } = strategy
        {
            return self.decode_one_splice_reranked(encoded, beam_size, retained_candidates);
        }
        if matches!(
            strategy,
            ReazonDecodingStrategy::ModifiedBeam { beam_size: 0 }
        ) || matches!(
            strategy,
            ReazonDecodingStrategy::ModifiedBeamAblation {
                config: ReazonBeamAblation { beam_size: 0, .. }
            }
        ) {
            bail!("ReazonSpeech beam size must be positive");
        }
        let score_normalization = self.default_score_normalization();

        let (token_ids, timestamp_frames) = match strategy {
            ReazonDecodingStrategy::Greedy => decode_greedy(
                &mut self.decoder,
                &mut self.predictor_cache,
                &mut GreedyJoiner {
                    session: &mut self.joiner,
                    output_name: self.joiner_output_name,
                    run_options: &self.joiner_run_options,
                    compact: self.compact_joiner_run_options.is_some(),
                },
                &encoded.values,
                encoded.frames,
            )?,
            ReazonDecodingStrategy::ModifiedBeam { beam_size } => {
                let config = ReazonBeamAblation::production(beam_size);
                decode_modified_beam(
                    &mut self.decoder,
                    &mut self.joiner,
                    &mut self.predictor_cache,
                    self.joiner_output_name,
                    &self.joiner_run_options,
                    self.compact_joiner_run_options.as_ref(),
                    &encoded.values,
                    encoded.frames,
                    ablation_decoder_config(config),
                    config.network_batching,
                    score_normalization,
                    profile.as_deref_mut(),
                )?
            }
            ReazonDecodingStrategy::ModifiedBeamAblation { config } => decode_modified_beam(
                &mut self.decoder,
                &mut self.joiner,
                &mut self.predictor_cache,
                self.joiner_output_name,
                &self.joiner_run_options,
                self.compact_joiner_run_options.as_ref(),
                &encoded.values,
                encoded.frames,
                ablation_decoder_config(config),
                config.network_batching,
                score_normalization,
                profile,
            )?,
            ReazonDecodingStrategy::OneSpliceRerank { .. } => {
                unreachable!("one-splice reranking returns before acoustic decoding")
            }
        };
        let timestamps = timestamp_frames
            .iter()
            .map(|&frame| frame as f32 * FRAME_SECONDS)
            .collect::<Vec<_>>();
        let token_texts = token_ids
            .iter()
            .map(|&id| self.tokens[id].clone())
            .collect::<Vec<_>>();
        Ok(AsrTranscript::from_parts(
            token_texts.concat(),
            token_texts,
            Some(&timestamps),
            None,
        ))
    }

    fn decode_one_splice_reranked(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        retained_candidates: usize,
    ) -> Result<AsrTranscript> {
        self.decode_one_splice_reranked_with_hotwords(encoded, beam_size, retained_candidates, None)
    }

    fn decode_one_splice_reranked_with_hotwords(
        &mut self,
        encoded: &ReazonEncodedAudio,
        beam_size: usize,
        retained_candidates: usize,
        hotwords: Option<&HotwordContextGraph>,
    ) -> Result<AsrTranscript> {
        let seeds = self.decode_encoded_nbest_with_hotwords(
            encoded,
            beam_size,
            StatelessRnntLengthNormalization::Raw,
            hotwords,
        )?;
        let evidence = seeds
            .iter()
            .map(|seed| EvidenceSeed {
                raw_score: seed.raw_score,
                token_ids: &seed.token_ids,
            })
            .collect::<Vec<_>>();
        let tokens = &self.tokens;
        let static_embedding = self.static_embedding.as_mut().ok_or_else(|| {
            anyhow!("ReazonSpeech one-splice reranking has no static embedding model")
        })?;
        let selection =
            select_one_splice_candidate(&evidence, retained_candidates, |candidate_token_ids| {
                let text = candidate_token_ids
                    .iter()
                    .map(|&id| tokens[id].as_str())
                    .collect::<String>();
                static_embedding.piece_mean(&text)
            })?;
        let token_texts = selection
            .token_ids
            .iter()
            .map(|&id| self.tokens[id].clone())
            .collect::<Vec<_>>();
        let timestamps = selection
            .source_seed
            .and_then(|index| seeds.get(index))
            .map(|seed| seed.timestamps.as_slice());
        if let Some(graph) = hotwords {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "timestamps are generated from bounded nonnegative frame indices"
            )]
            let timestamp_frames = selection
                .source_seed
                .and_then(|index| seeds.get(index))
                .map(|seed| {
                    seed.timestamps
                        .iter()
                        .map(|&timestamp| (timestamp / FRAME_SECONDS).round() as usize)
                        .collect::<Vec<_>>()
                });
            Ok(transcript_from_tokens_with_hotwords(
                &self.tokens,
                &selection.token_ids,
                timestamp_frames.as_deref(),
                graph,
            ))
        } else {
            Ok(AsrTranscript::from_parts(
                token_texts.concat(),
                token_texts,
                timestamps,
                None,
            ))
        }
    }

    fn default_score_normalization(&self) -> StatelessRnntScoreNormalization {
        if self.joiner_output_name == RAW_JOINER_OUTPUT {
            StatelessRnntScoreNormalization::SparseLogNormalizer
        } else {
            StatelessRnntScoreNormalization::PrecomputedLogProbabilities
        }
    }
}

impl AsrEngine for ReazonSpeechOrtAsrEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        // Normal production recognition owns one utterance at a time. Keep its
        // cache bounded to that utterance; accuracy sweeps call encode and
        // decode_encoded directly when they intentionally share it farther.
        self.predictor_cache.outputs.clear();
        let encoded = self.encode(samples)?;
        self.decode_encoded(&encoded, self.decoding)
    }
}

fn transcript_from_tokens_with_hotwords(
    tokens: &[String],
    token_ids: &[usize],
    timestamp_frames: Option<&[usize]>,
    graph: &HotwordContextGraph,
) -> AsrTranscript {
    let matches = graph.find_matches(token_ids);
    let has_timestamps = timestamp_frames.is_some();
    let timestamps = timestamp_frames
        .unwrap_or_default()
        .iter()
        .map(|&frame| frame as f32 * FRAME_SECONDS)
        .collect::<Vec<_>>();
    let mut token_texts = Vec::new();
    let mut output_timestamps = Vec::new();
    let mut output_durations = Vec::new();
    let mut cursor = 0;
    for matched in matches {
        while cursor < matched.start_token {
            token_texts.push(tokens[token_ids[cursor]].clone());
            output_timestamps.push(timestamps.get(cursor).copied().unwrap_or_default());
            output_durations.push(None);
            cursor += 1;
        }
        token_texts.push(matched.surface);
        output_timestamps.push(
            timestamps
                .get(matched.start_token)
                .copied()
                .unwrap_or_default(),
        );
        let start = timestamps
            .get(matched.start_token)
            .copied()
            .unwrap_or_default();
        let end = timestamps
            .get(matched.end_token.saturating_sub(1))
            .copied()
            .unwrap_or(start);
        output_durations.push(Some((end - start).max(0.0) + FRAME_SECONDS));
        cursor = matched.end_token;
    }
    while cursor < token_ids.len() {
        token_texts.push(tokens[token_ids[cursor]].clone());
        output_timestamps.push(timestamps.get(cursor).copied().unwrap_or_default());
        output_durations.push(None);
        cursor += 1;
    }
    let mut transcript = AsrTranscript::from_parts(
        token_texts.concat(),
        token_texts,
        has_timestamps.then_some(output_timestamps.as_slice()),
        None,
    );
    if has_timestamps {
        for (token, duration) in transcript.tokens.iter_mut().zip(output_durations) {
            token.duration_sec = duration;
        }
    }
    transcript
}

fn tokenize_reazon_hotword_entries(
    entries: &[HotwordEntry],
    token_ids: &HashMap<&str, usize>,
) -> Result<Vec<HotwordTokenPath>> {
    entries
        .iter()
        .enumerate()
        .try_fold(Vec::new(), |mut paths, (entry_id, entry)| {
            if entry.surface.trim().is_empty() {
                bail!("hotword surface must not be empty");
            }
            match tokenize_reazon_hotword_text(&entry.surface, token_ids, "surface") {
                Ok(surface_tokens) => paths.push(HotwordTokenPath {
                    tokens: surface_tokens,
                    entry_id,
                    surface: entry.surface.clone(),
                    kind: HotwordPathKind::Surface,
                    phrase_score: entry.phrase_score,
                }),
                Err(error) if entry.readings.is_empty() => return Err(error),
                Err(_) => {}
            }
            for reading in &entry.readings {
                let normalized = normalize_reading(reading);
                if normalized.trim().is_empty() {
                    bail!("hotword reading must not be empty");
                }
                let hiragana_tokens =
                    tokenize_reazon_hotword_text(&normalized, token_ids, "reading")?;
                paths.push(HotwordTokenPath {
                    tokens: hiragana_tokens,
                    entry_id,
                    surface: entry.surface.clone(),
                    kind: HotwordPathKind::Reading,
                    phrase_score: entry.phrase_score,
                });
                let katakana = to_katakana(&normalized);
                if katakana != normalized {
                    paths.push(HotwordTokenPath {
                        tokens: tokenize_reazon_hotword_text(&katakana, token_ids, "reading")?,
                        entry_id,
                        surface: entry.surface.clone(),
                        kind: HotwordPathKind::Reading,
                        phrase_score: entry.phrase_score,
                    });
                }
            }
            Ok(paths)
        })
}

fn tokenize_reazon_hotword_text(
    text: &str,
    token_ids: &HashMap<&str, usize>,
    kind: &str,
) -> Result<Vec<usize>> {
    let tokens = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .map(|character| {
            let token = character.to_string();
            token_ids.get(token.as_str()).copied().ok_or_else(|| {
                anyhow!(
                    "ReazonSpeech hotword {kind} contains out-of-vocabulary character {character:?}"
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if tokens.is_empty() {
        bail!("ReazonSpeech hotword {kind} must contain at least one non-whitespace character");
    }
    Ok(tokens)
}

fn to_katakana(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\u{3041}'..='\u{3096}' => char::from_u32(character as u32 + 0x60).unwrap_or(character),
            _ => character,
        })
        .collect()
}

fn validate_decoding_strategy(decoding: ReazonDecodingStrategy) -> Result<()> {
    if matches!(
        decoding,
        ReazonDecodingStrategy::ModifiedBeam { beam_size: 0 }
            | ReazonDecodingStrategy::OneSpliceRerank { beam_size: 0, .. }
    ) || matches!(
        decoding,
        ReazonDecodingStrategy::ModifiedBeamAblation {
            config: ReazonBeamAblation { beam_size: 0, .. }
        }
    ) {
        bail!("ReazonSpeech beam size must be positive");
    }
    if matches!(
        decoding,
        ReazonDecodingStrategy::OneSpliceRerank {
            retained_candidates: 0,
            ..
        }
    ) {
        bail!("ReazonSpeech one-splice retained width must be positive");
    }
    Ok(())
}

fn ablation_decoder_config(config: ReazonBeamAblation) -> StatelessRnntBeamConfig {
    StatelessRnntBeamConfig {
        beam_size: config.beam_size,
        initial_context: match config.initial_context {
            ReazonInitialContext::ModelReference => [BLANK_ID as i64, BLANK_ID as i64],
            ReazonInitialContext::Sherpa => [-1, BLANK_ID as i64],
        },
        additional_non_emitting_id: config.unknown_is_non_emitting.then_some(UNKNOWN_ID),
        pruning: match config.pruning {
            ReazonBeamPruning::StateDominance => StatelessRnntPruning::StateDominance,
            ReazonBeamPruning::FullPrefix => StatelessRnntPruning::FullPrefix,
            ReazonBeamPruning::Sherpa => StatelessRnntPruning::Sherpa,
        },
        deduplicate_contexts: config.network_batching == ReazonNetworkBatching::CachedUnique,
        search_length_normalization: StatelessRnntLengthNormalization::Raw,
        final_length_normalization: StatelessRnntLengthNormalization::PerToken,
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the diagnostic profile sink is optional while model sessions and dimensions stay explicit"
)]
fn decode_modified_beam(
    decoder: &mut Session,
    joiner: &mut Session,
    predictor_cache: &mut PredictorCache,
    joiner_output_name: &'static str,
    joiner_run_options: &RunOptions<HasSelectedOutputs>,
    compact_joiner_run_options: Option<&RunOptions<HasSelectedOutputs>>,
    encoded: &[f32],
    frames: usize,
    config: StatelessRnntBeamConfig,
    batching: ReazonNetworkBatching,
    score_normalization: StatelessRnntScoreNormalization,
    profile: Option<&mut StatelessRnntSearchProfile>,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let search_results = search_modified_beam(
        decoder,
        joiner,
        predictor_cache,
        joiner_output_name,
        joiner_run_options,
        compact_joiner_run_options,
        encoded,
        frames,
        config,
        batching,
        score_normalization,
        profile,
        None,
    )?;
    let best = search_results
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("ReazonSpeech beam search returned no hypotheses"))?;
    Ok((best.token_ids, best.timestamps))
}

#[allow(
    clippy::too_many_arguments,
    reason = "the diagnostic N-best path shares the explicit model sessions and dimensions"
)]
fn search_modified_beam(
    decoder: &mut Session,
    joiner: &mut Session,
    predictor_cache: &mut PredictorCache,
    joiner_output_name: &'static str,
    joiner_run_options: &RunOptions<HasSelectedOutputs>,
    compact_joiner_run_options: Option<&RunOptions<HasSelectedOutputs>>,
    encoded: &[f32],
    frames: usize,
    config: StatelessRnntBeamConfig,
    batching: ReazonNetworkBatching,
    score_normalization: StatelessRnntScoreNormalization,
    profile: Option<&mut StatelessRnntSearchProfile>,
    hotwords: Option<&HotwordContextGraph>,
) -> Result<Vec<StatelessRnntHypothesis>> {
    let mut network = ReazonBeamNetwork {
        decoder,
        joiner,
        predictor_cache,
        batching,
        joiner_output_name,
        joiner_run_options,
        compact_joiner_run_options,
    };
    if let Some(profile) = profile {
        if hotwords.is_some() {
            bail!("profiled hotword N-best search is not supported");
        }
        modified_beam_search_stateless_profiled_with_search_options(
            &mut network,
            encoded,
            frames,
            ENCODER_DIM,
            VOCAB_SIZE,
            BLANK_ID,
            config,
            StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
            score_normalization,
            profile,
        )
    } else {
        match hotwords {
            Some(hotwords) => modified_beam_search_stateless_with_hotwords_and_search_options(
                &mut network,
                encoded,
                frames,
                ENCODER_DIM,
                VOCAB_SIZE,
                BLANK_ID,
                config,
                hotwords,
                StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
                score_normalization,
            ),
            None => modified_beam_search_stateless_with_config_and_search_options(
                &mut network,
                encoded,
                frames,
                ENCODER_DIM,
                VOCAB_SIZE,
                BLANK_ID,
                config,
                StatelessRnntTopTokenAlgorithm::CutoffBinarySearch,
                score_normalization,
            ),
        }
    }
}

struct GreedyJoiner<'a> {
    session: &'a mut Session,
    output_name: &'static str,
    run_options: &'a RunOptions<HasSelectedOutputs>,
    compact: bool,
}

fn decode_greedy(
    decoder: &mut Session,
    predictor_cache: &mut PredictorCache,
    joiner: &mut GreedyJoiner<'_>,
    encoded: &[f32],
    frames: usize,
) -> Result<(Vec<usize>, Vec<usize>)> {
    // The icefall ReazonSpeech ONNX reference initializes every predictor
    // context slot with blank. sherpa-onnx 1.13.3 instead uses [-1, 0];
    // keep the model-producing Python reference authoritative here.
    let mut context = [BLANK_ID as i64; 2];
    let mut decoder_output = predictor_cache
        .get_or_try_insert_with(context, |context| run_decoder(decoder, context))?
        .to_vec();
    let mut token_ids = Vec::new();
    let mut timestamps = Vec::new();
    for frame in 0..frames {
        let encoder_frame = &encoded[frame * ENCODER_DIM..(frame + 1) * ENCODER_DIM];
        let encoder_input =
            Tensor::from_array((vec![1_i64, ENCODER_DIM as i64], encoder_frame.to_vec()))?;
        let decoder_input =
            Tensor::from_array((vec![1_i64, ENCODER_DIM as i64], decoder_output.clone()))?;
        let joiner_outputs = if joiner.compact {
            let requested_input = Tensor::from_array((vec![1_i64, 1_i64], vec![BLANK_ID as i64]))?;
            let top_k_input = Tensor::from_array(([1_i64], vec![1_i64]))?;
            joiner.session.run_with_options(
                ort::inputs![
                    "encoder_out" => encoder_input,
                    "decoder_out" => decoder_input,
                    REQUESTED_TOKEN_IDS_INPUT => requested_input,
                    TOP_K_COUNT_INPUT => top_k_input,
                ],
                joiner.run_options,
            )?
        } else {
            joiner.session.run_with_options(
                ort::inputs![
                    "encoder_out" => encoder_input,
                    "decoder_out" => decoder_input,
                ],
                joiner.run_options,
            )?
        };
        let (_, logits) = joiner_outputs
            .get(joiner.output_name)
            .ok_or_else(|| anyhow!("ReazonSpeech joiner did not return {}", joiner.output_name))?
            .try_extract_tensor::<f32>()?;
        if logits.len() != VOCAB_SIZE {
            bail!("ReazonSpeech joiner vocabulary changed");
        }
        let token = logits
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 { (index, value) } else { best }
            })
            .0;
        drop(joiner_outputs);
        if token != BLANK_ID {
            token_ids.push(token);
            timestamps.push(frame);
            context = [context[1], token as i64];
            decoder_output = predictor_cache
                .get_or_try_insert_with(context, |context| run_decoder(decoder, context))?
                .to_vec();
        }
    }
    Ok((token_ids, timestamps))
}

struct ReazonBeamNetwork<'a> {
    decoder: &'a mut Session,
    joiner: &'a mut Session,
    predictor_cache: &'a mut PredictorCache,
    batching: ReazonNetworkBatching,
    joiner_output_name: &'static str,
    joiner_run_options: &'a RunOptions<HasSelectedOutputs>,
    compact_joiner_run_options: Option<&'a RunOptions<HasSelectedOutputs>>,
}

impl StatelessRnntNetwork for ReazonBeamNetwork<'_> {
    fn supports_compact_log_probabilities(&self) -> bool {
        self.compact_joiner_run_options.is_some()
    }

    fn logits(&mut self, encoder_frame: &[f32], contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
        let (unique_contexts, context_rows) = match self.batching {
            ReazonNetworkBatching::CachedUnique => unique_context_layout(contexts),
            ReazonNetworkBatching::Sherpa => {
                (contexts.to_vec(), (0..contexts.len()).collect::<Vec<_>>())
            }
        };
        let unique_batch = unique_contexts.len();
        let decoder_values = match self.batching {
            ReazonNetworkBatching::CachedUnique => {
                // Predictor output is a pure function of the two-token context. Run
                // every new context with the same batch-1 contract as direct greedy,
                // then cache it. Some INT8 kernels change rounding with batch shape;
                // canonical batch-1 values keep results independent of which other
                // hypotheses happened to be active at the same frame.
                let mut values = Vec::with_capacity(unique_batch * ENCODER_DIM);
                for context in unique_contexts.iter().copied() {
                    values.extend_from_slice(
                        self.predictor_cache
                            .get_or_try_insert_with(context, |context| {
                                run_decoder(self.decoder, context)
                            })?,
                    );
                }
                values
            }
            ReazonNetworkBatching::Sherpa => run_decoder_batch(self.decoder, &unique_contexts)?,
        };

        let encoder_values = (0..unique_batch)
            .flat_map(|_| encoder_frame.iter().copied())
            .collect::<Vec<_>>();
        let encoder_input = Tensor::from_array((
            vec![unique_batch as i64, ENCODER_DIM as i64],
            encoder_values,
        ))?;
        let decoder_input = Tensor::from_array((
            vec![unique_batch as i64, ENCODER_DIM as i64],
            decoder_values,
        ))?;
        let joiner_outputs = self.joiner.run_with_options(
            ort::inputs![
                "encoder_out" => encoder_input,
                "decoder_out" => decoder_input,
            ],
            self.joiner_run_options,
        )?;
        let (_, logits) = joiner_outputs
            .get(self.joiner_output_name)
            .ok_or_else(|| {
                anyhow!(
                    "ReazonSpeech joiner did not return {}",
                    self.joiner_output_name
                )
            })?
            .try_extract_tensor::<f32>()?;
        if logits.len() != unique_batch * VOCAB_SIZE {
            bail!("ReazonSpeech batched joiner vocabulary changed");
        }
        let mut expanded = Vec::with_capacity(contexts.len() * VOCAB_SIZE);
        for row in context_rows {
            expanded.extend_from_slice(&logits[row * VOCAB_SIZE..(row + 1) * VOCAB_SIZE]);
        }
        Ok(expanded)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the compact session call validates three coordinated dynamic outputs"
    )]
    fn compact_log_probabilities(
        &mut self,
        encoder_frame: &[f32],
        contexts: &[[i64; 2]],
        requested_token_ids: &[Vec<usize>],
        emitting_limit: usize,
        blank_id: usize,
        additional_non_emitting_id: Option<usize>,
    ) -> Result<Option<StatelessRnntCompactScores>> {
        let Some(run_options) = self.compact_joiner_run_options else {
            return Ok(None);
        };
        if contexts.len() != requested_token_ids.len() {
            bail!("ReazonSpeech compact request row count changed");
        }
        let batch = contexts.len();
        let decoder_values = match self.batching {
            ReazonNetworkBatching::CachedUnique => {
                let mut values = Vec::with_capacity(batch * ENCODER_DIM);
                for context in contexts.iter().copied() {
                    values.extend_from_slice(
                        self.predictor_cache
                            .get_or_try_insert_with(context, |context| {
                                run_decoder(self.decoder, context)
                            })?,
                    );
                }
                values
            }
            ReazonNetworkBatching::Sherpa => run_decoder_batch(self.decoder, contexts)?,
        };
        let encoder_values = (0..batch)
            .flat_map(|_| encoder_frame.iter().copied())
            .collect::<Vec<_>>();
        let requested_width = requested_token_ids.iter().map(Vec::len).max().unwrap_or(1);
        let raw_top_k = emitting_limit
            .saturating_add(1 + usize::from(additional_non_emitting_id.is_some()))
            .min(VOCAB_SIZE);
        let mut requested_values = Vec::with_capacity(batch * requested_width);
        for row in requested_token_ids {
            requested_values.extend(row.iter().map(|&token| token as i64));
            requested_values.resize(requested_values.len() + requested_width - row.len(), 0);
        }
        let encoder_input =
            Tensor::from_array((vec![batch as i64, ENCODER_DIM as i64], encoder_values))?;
        let decoder_input =
            Tensor::from_array((vec![batch as i64, ENCODER_DIM as i64], decoder_values))?;
        let requested_input =
            Tensor::from_array((vec![batch as i64, requested_width as i64], requested_values))?;
        let top_k_input = Tensor::from_array(([1_i64], vec![raw_top_k as i64]))?;
        let outputs = self.joiner.run_with_options(
            ort::inputs![
                "encoder_out" => encoder_input,
                "decoder_out" => decoder_input,
                REQUESTED_TOKEN_IDS_INPUT => requested_input,
                TOP_K_COUNT_INPUT => top_k_input,
            ],
            run_options,
        )?;
        let (_, top_values) = outputs
            .get(TOP_K_VALUES_OUTPUT)
            .ok_or_else(|| anyhow!("ReazonSpeech joiner did not return ORT TopK values"))?
            .try_extract_tensor::<f32>()?;
        let (_, top_indices) = outputs
            .get(TOP_K_INDICES_OUTPUT)
            .ok_or_else(|| anyhow!("ReazonSpeech joiner did not return ORT TopK indices"))?
            .try_extract_tensor::<i64>()?;
        let (_, requested_scores) = outputs
            .get(REQUESTED_SCORES_OUTPUT)
            .ok_or_else(|| anyhow!("ReazonSpeech joiner did not return gathered scores"))?
            .try_extract_tensor::<f32>()?;
        if top_values.len() != batch * raw_top_k
            || top_indices.len() != top_values.len()
            || requested_scores.len() != batch * requested_width
        {
            bail!("ReazonSpeech compact joiner output shape changed");
        }
        let mut top_tokens = Vec::with_capacity(batch);
        for row in 0..batch {
            let offset = row * raw_top_k;
            let mut emitting = Vec::with_capacity(emitting_limit);
            for column in 0..raw_top_k {
                let token_id = usize::try_from(top_indices[offset + column])
                    .map_err(|_| anyhow!("ReazonSpeech ORT TopK returned a negative token ID"))?;
                if token_id == blank_id || additional_non_emitting_id == Some(token_id) {
                    continue;
                }
                emitting.push((token_id, top_values[offset + column]));
                if emitting.len() == emitting_limit {
                    break;
                }
            }
            top_tokens.push(emitting);
        }
        let gathered = requested_token_ids
            .iter()
            .enumerate()
            .map(|(row, tokens)| {
                let offset = row * requested_width;
                requested_scores[offset..offset + tokens.len()].to_vec()
            })
            .collect::<Vec<_>>();
        Ok(Some(StatelessRnntCompactScores {
            top_tokens,
            requested_scores: gathered,
            network_output_values: top_values.len() + top_indices.len() + requested_scores.len(),
            network_output_bytes: std::mem::size_of_val(top_values)
                + std::mem::size_of_val(requested_scores)
                + std::mem::size_of_val(top_indices),
        }))
    }
}

struct ReazonSequenceScoreNetwork<'a> {
    decoder: &'a mut Session,
    joiner: &'a mut Session,
    predictor_cache: &'a mut PredictorCache,
    run_options: &'a RunOptions<HasSelectedOutputs>,
}

impl StatelessRnntSequenceScoreNetwork for ReazonSequenceScoreNetwork<'_> {
    fn requested_log_probabilities(
        &mut self,
        encoder_frame: &[f32],
        contexts: &[[i64; 2]],
        requested_token_ids: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        if contexts.is_empty() || contexts.len() != requested_token_ids.len() {
            bail!("ReazonSpeech fixed-sequence request row count changed");
        }
        if encoder_frame.len() != ENCODER_DIM {
            bail!("ReazonSpeech fixed-sequence encoder dimension changed");
        }
        if requested_token_ids
            .iter()
            .any(|tokens| tokens.is_empty() || tokens.iter().any(|&token| token >= VOCAB_SIZE))
        {
            bail!("ReazonSpeech fixed-sequence token request changed");
        }

        let batch = contexts.len();
        let mut decoder_values = Vec::with_capacity(batch * ENCODER_DIM);
        for context in contexts.iter().copied() {
            decoder_values.extend_from_slice(
                self.predictor_cache
                    .get_or_try_insert_with(context, |context| {
                        run_decoder(self.decoder, context)
                    })?,
            );
        }
        let encoder_values = (0..batch)
            .flat_map(|_| encoder_frame.iter().copied())
            .collect::<Vec<_>>();
        let requested_width = requested_token_ids
            .iter()
            .map(Vec::len)
            .max()
            .expect("requests are nonempty");
        let mut requested_values = Vec::with_capacity(batch * requested_width);
        for row in requested_token_ids {
            requested_values.extend(row.iter().map(|&token| token as i64));
            requested_values.resize(requested_values.len() + requested_width - row.len(), 0);
        }
        let encoder_input =
            Tensor::from_array((vec![batch as i64, ENCODER_DIM as i64], encoder_values))?;
        let decoder_input =
            Tensor::from_array((vec![batch as i64, ENCODER_DIM as i64], decoder_values))?;
        let requested_input =
            Tensor::from_array((vec![batch as i64, requested_width as i64], requested_values))?;
        // The compact graph declares TopK's count as an input even when the
        // requested-score-only output selector lets ORT prune that branch.
        let top_k_input = Tensor::from_array(([1_i64], vec![1_i64]))?;
        let outputs = self.joiner.run_with_options(
            ort::inputs![
                "encoder_out" => encoder_input,
                "decoder_out" => decoder_input,
                REQUESTED_TOKEN_IDS_INPUT => requested_input,
                TOP_K_COUNT_INPUT => top_k_input,
            ],
            self.run_options,
        )?;
        let (_, selected) = outputs
            .get(REQUESTED_SCORES_OUTPUT)
            .ok_or_else(|| anyhow!("ReazonSpeech joiner did not return requested scores"))?
            .try_extract_tensor::<f32>()?;
        if selected.len() != batch * requested_width {
            bail!("ReazonSpeech fixed-sequence selected-score shape changed");
        }
        Ok(requested_token_ids
            .iter()
            .enumerate()
            .map(|(row, tokens)| {
                let offset = row * requested_width;
                selected[offset..offset + tokens.len()].to_vec()
            })
            .collect())
    }
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

fn run_decoder(decoder: &mut Session, context: [i64; 2]) -> Result<Vec<f32>> {
    run_decoder_batch(decoder, &[context])
}

fn run_decoder_batch(decoder: &mut Session, contexts: &[[i64; 2]]) -> Result<Vec<f32>> {
    let batch = contexts.len();
    let values = contexts.iter().flatten().copied().collect::<Vec<_>>();
    let input = Tensor::from_array((vec![batch as i64, 2], values))?;
    let outputs = decoder.run(ort::inputs!["y" => input])?;
    let (_, values) = outputs
        .get("decoder_out")
        .ok_or_else(|| anyhow!("ReazonSpeech decoder did not return decoder_out"))?
        .try_extract_tensor::<f32>()?;
    if values.len() != batch * ENCODER_DIM {
        bail!("ReazonSpeech decoder dimension changed");
    }
    Ok(values.to_vec())
}

#[must_use]
pub fn required_model_file_names(precision: AsrPrecision) -> &'static [&'static str] {
    match precision {
        AsrPrecision::Int8 => &[
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ],
        AsrPrecision::Int8Float32 => &[
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ],
        AsrPrecision::Float32 => &[
            "encoder-epoch-99-avg-1.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.onnx",
            "tokens.txt",
        ],
    }
}

fn model_files(precision: AsrPrecision) -> (&'static str, &'static str, &'static str) {
    let files = required_model_file_names(precision);
    (files[0], files[1], files[2])
}

fn load_session(path: &Path, threads: usize, label: &str) -> Result<Session> {
    if !path.is_file() {
        bail!("{label} file not found: {}", path.display());
    }
    session_builder(threads, label)?
        .commit_from_file(path)
        .map_err(|error| anyhow!("failed to load {label} {}: {error}", path.display()))
}

fn session_builder(threads: usize, label: &str) -> Result<SessionBuilder> {
    Session::builder()
        .map_err(|error| anyhow!("failed to create {label} session builder: {error}"))?
        .with_intra_threads(threads)
        .map_err(|error| anyhow!("failed to configure {label} intra-op threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow!("failed to configure {label} inter-op threads: {error}"))
}

#[allow(
    clippy::too_many_lines,
    reason = "the in-memory ONNX extension declares and connects all score outputs together"
)]
fn load_joiner_with_log_softmax(
    path: &Path,
    threads: usize,
    label: &str,
    compact_output: bool,
) -> Result<Session> {
    if !path.is_file() {
        bail!("{label} file not found: {}", path.display());
    }
    let mut graph = Graph::new()
        .map_err(|error| anyhow!("failed to create {label} extension graph: {error}"))?;
    let float_tensor_type = |width: i64| ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape: Shape::new([-1, width]),
        dimension_symbols: SymbolicDimensions::empty(2),
    };
    let integer_tensor_type = |width: i64| ValueType::Tensor {
        ty: TensorElementType::Int64,
        shape: Shape::new([-1, width]),
        dimension_symbols: SymbolicDimensions::empty(2),
    };
    let mut inputs = vec![
        Outlet::new("encoder_out", float_tensor_type(ENCODER_DIM as i64)),
        Outlet::new("decoder_out", float_tensor_type(ENCODER_DIM as i64)),
    ];
    if compact_output {
        inputs.push(Outlet::new(
            REQUESTED_TOKEN_IDS_INPUT,
            integer_tensor_type(-1),
        ));
        inputs.push(Outlet::new(
            TOP_K_COUNT_INPUT,
            ValueType::Tensor {
                ty: TensorElementType::Int64,
                shape: Shape::new([1]),
                dimension_symbols: SymbolicDimensions::empty(1),
            },
        ));
    }
    graph
        .set_inputs(inputs)
        .map_err(|error| anyhow!("failed to set {label} extension input: {error}"))?;
    let mut outputs = vec![Outlet::new(
        LOG_PROBABILITY_JOINER_OUTPUT,
        float_tensor_type(VOCAB_SIZE as i64),
    )];
    if compact_output {
        outputs.extend([
            Outlet::new(TOP_K_VALUES_OUTPUT, float_tensor_type(-1)),
            Outlet::new(TOP_K_INDICES_OUTPUT, integer_tensor_type(-1)),
            Outlet::new(REQUESTED_SCORES_OUTPUT, float_tensor_type(-1)),
        ]);
    }
    graph
        .set_outputs(outputs)
        .map_err(|error| anyhow!("failed to set {label} extension output: {error}"))?;
    let axis = Attribute::new("axis", -1_i64)
        .map_err(|error| anyhow!("failed to configure {label} LogSoftmax: {error}"))?;
    let log_softmax = Node::new(
        "LogSoftmax",
        ONNX_DOMAIN,
        "parapper.log_softmax",
        [RAW_JOINER_OUTPUT],
        [LOG_PROBABILITY_JOINER_OUTPUT],
        [axis],
    )
    .map_err(|error| anyhow!("failed to create {label} LogSoftmax: {error}"))?;
    graph
        .add_node(log_softmax)
        .map_err(|error| anyhow!("failed to add {label} LogSoftmax: {error}"))?;
    if compact_output {
        let top_k_node = Node::new(
            "TopK",
            ONNX_DOMAIN,
            "parapper.top_k",
            [LOG_PROBABILITY_JOINER_OUTPUT, TOP_K_COUNT_INPUT],
            [TOP_K_VALUES_OUTPUT, TOP_K_INDICES_OUTPUT],
            [
                Attribute::new("axis", -1_i64)
                    .map_err(|error| anyhow!("failed to configure {label} TopK axis: {error}"))?,
                Attribute::new("largest", 1_i64).map_err(|error| {
                    anyhow!("failed to configure {label} TopK ordering: {error}")
                })?,
                Attribute::new("sorted", 1_i64).map_err(|error| {
                    anyhow!("failed to configure {label} TopK sorting: {error}")
                })?,
            ],
        )
        .map_err(|error| anyhow!("failed to create {label} TopK: {error}"))?;
        graph
            .add_node(top_k_node)
            .map_err(|error| anyhow!("failed to add {label} TopK: {error}"))?;
        let gather = Node::new(
            "GatherElements",
            ONNX_DOMAIN,
            "parapper.gather_requested_scores",
            [LOG_PROBABILITY_JOINER_OUTPUT, REQUESTED_TOKEN_IDS_INPUT],
            [REQUESTED_SCORES_OUTPUT],
            [Attribute::new("axis", 1_i64)
                .map_err(|error| anyhow!("failed to configure {label} GatherElements: {error}"))?],
        )
        .map_err(|error| anyhow!("failed to create {label} GatherElements: {error}"))?;
        graph
            .add_node(gather)
            .map_err(|error| anyhow!("failed to add {label} GatherElements: {error}"))?;
    }
    let mut extension = Model::new([Opset::new(ONNX_DOMAIN, 13)
        .map_err(|error| anyhow!("failed to declare {label} ONNX opset: {error}"))?])
    .map_err(|error| anyhow!("failed to create {label} extension model: {error}"))?;
    extension
        .add_graph(graph)
        .map_err(|error| anyhow!("failed to attach {label} extension graph: {error}"))?;

    let mut builder = session_builder(threads, label)?;
    let mut editable = builder
        .edit_from_file(path)
        .map_err(|error| anyhow!("failed to edit {label} {}: {error}", path.display()))?;
    editable
        .apply_model(&extension)
        .map_err(|error| anyhow!("failed to append {label} LogSoftmax: {error}"))?;
    editable
        .into_session()
        .map_err(|error| anyhow!("failed to finalize {label} {}: {error}", path.display()))
}

fn load_tokens(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read ReazonSpeech tokens: {}", path.display()))?;
    let mut tokens = vec![None; VOCAB_SIZE];
    for line in contents.lines() {
        let (token, raw_id) = line
            .rsplit_once('\t')
            .or_else(|| line.rsplit_once(' '))
            .ok_or_else(|| anyhow!("invalid ReazonSpeech token line"))?;
        let id = raw_id.parse::<usize>()?;
        let slot = tokens
            .get_mut(id)
            .ok_or_else(|| anyhow!("ReazonSpeech token id {id} exceeds vocabulary"))?;
        if slot.replace(token.to_string()).is_some() {
            bail!("duplicate ReazonSpeech token id {id}");
        }
    }
    if tokens.iter().any(Option::is_none) {
        bail!("ReazonSpeech token table is not contiguous");
    }
    let tokens = tokens.into_iter().flatten().collect::<Vec<_>>();
    if tokens[BLANK_ID] != "<blk>"
        || tokens[UNKNOWN_ID] != "<unk>"
        || tokens[SOS_EOS_ID] != "<sos/eos>"
    {
        bail!("ReazonSpeech special-token contract changed");
    }
    Ok(tokens)
}

fn validate_contract(
    encoder: &Session,
    decoder: &Session,
    joiner: &Session,
    joiner_output_name: &str,
) -> Result<()> {
    let encoder_inputs = encoder
        .inputs()
        .iter()
        .map(ort::value::Outlet::name)
        .collect::<Vec<_>>();
    let encoder_outputs = encoder
        .outputs()
        .iter()
        .map(ort::value::Outlet::name)
        .collect::<Vec<_>>();
    let joiner_inputs = joiner
        .inputs()
        .iter()
        .map(ort::value::Outlet::name)
        .collect::<Vec<_>>();
    let joiner_outputs = joiner
        .outputs()
        .iter()
        .map(ort::value::Outlet::name)
        .collect::<Vec<_>>();
    if encoder_inputs != ["x", "x_lens"]
        || encoder_outputs != ["encoder_out", "encoder_out_lens"]
        || decoder.inputs()[0].name() != "y"
        || decoder.outputs()[0].name() != "decoder_out"
        || joiner_inputs != ["encoder_out", "decoder_out"]
        || joiner_outputs != [joiner_output_name]
    {
        bail!(
            "ReazonSpeech ONNX I/O contract changed: encoder={encoder_inputs:?}->{encoder_outputs:?}, decoder={:?}->{:?}, joiner={joiner_inputs:?}->{joiner_outputs:?}",
            decoder
                .inputs()
                .iter()
                .map(Outlet::name)
                .collect::<Vec<_>>(),
            decoder
                .outputs()
                .iter()
                .map(Outlet::name)
                .collect::<Vec<_>>()
        );
    }
    if decoder.metadata()?.custom("context_size").as_deref() != Some("2")
        || decoder.metadata()?.custom("vocab_size").as_deref() != Some("5224")
    {
        bail!("ReazonSpeech decoder metadata changed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::HashMap};

    use super::{
        PredictorCache, tokenize_reazon_hotword_entries, transcript_from_tokens_with_hotwords,
        unique_context_layout,
    };
    use crate::decoder::hotword::{
        HotwordContextGraph, HotwordEntry, HotwordPathKind, HotwordTokenPath,
    };

    #[test]
    fn duplicate_predictor_pairs_share_one_joiner_row_without_dropping_histories() {
        let contexts = [[7, 8], [2, 3], [7, 8], [4, 5], [2, 3]];

        let (unique, rows) = unique_context_layout(&contexts);

        assert_eq!(unique, vec![[7, 8], [2, 3], [4, 5]]);
        assert_eq!(rows, vec![0, 1, 0, 2, 1]);
        assert_eq!(
            rows.iter().map(|&row| unique[row]).collect::<Vec<_>>(),
            contexts
        );
    }

    #[test]
    fn predictor_output_is_computed_once_for_a_context_reused_by_sequential_decodes() {
        let mut cache = PredictorCache::default();
        let calls = Cell::new(0);
        let mut compute = |context: [i64; 2]| {
            calls.set(calls.get() + 1);
            Ok(vec![context[0] as f32, context[1] as f32])
        };

        let greedy_value = cache
            .get_or_try_insert_with([0, 7], &mut compute)
            .expect("the first decode computes the predictor output")
            .to_vec();
        let beam_value = cache
            .get_or_try_insert_with([0, 7], &mut compute)
            .expect("the later beam decode reuses the engine cache")
            .to_vec();

        assert_eq!(greedy_value, vec![0.0, 7.0]);
        assert_eq!(beam_value, greedy_value);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn hotword_surface_rewrite_without_seed_timestamps_keeps_timing_optional() {
        let graph = HotwordContextGraph::from_token_paths(
            vec![HotwordTokenPath {
                tokens: vec![1, 2],
                entry_id: 0,
                surface: "斎藤".to_string(),
                kind: HotwordPathKind::Reading,
                phrase_score: None,
            }],
            0.7,
        )
        .unwrap();
        let transcript = transcript_from_tokens_with_hotwords(
            &["さ".to_string(), "い".to_string()],
            &[1, 2],
            None,
            &graph,
        );
        assert_eq!(transcript.text, "斎藤");
        assert_eq!(transcript.tokens[0].start_sec, None);
        assert_eq!(transcript.tokens[0].duration_sec, None);
    }

    #[test]
    fn oov_display_surface_is_allowed_only_when_a_spoken_reading_is_tokenizable() {
        let token_ids = HashMap::from([
            ("す", 1),
            ("ろ", 2),
            ("べ", 3),
            ("に", 4),
            ("あ", 5),
            ("ス", 6),
            ("ロ", 7),
            ("ベ", 8),
            ("ニ", 9),
            ("ア", 10),
        ]);
        let entry = HotwordEntry {
            surface: "Tony Hawk's".to_owned(),
            readings: vec!["すろべにあ".to_owned()],
            phrase_score: None,
        };

        let paths =
            tokenize_reazon_hotword_entries(std::slice::from_ref(&entry), &token_ids).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(
            paths
                .iter()
                .all(|path| path.kind == HotwordPathKind::Reading)
        );
        assert!(
            tokenize_reazon_hotword_entries(
                &[HotwordEntry {
                    readings: Vec::new(),
                    ..entry
                }],
                &token_ids,
            )
            .is_err()
        );
    }
}
