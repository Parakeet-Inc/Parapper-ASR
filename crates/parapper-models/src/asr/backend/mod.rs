pub mod direct_ort;
pub mod direct_tdt;
pub mod nemotron_ort;
pub mod parakeet_ja;
pub mod reazon_ort;
mod reazon_rerank;

use std::path::Path;

use anyhow::Result;

use crate::{AsrEngine, AsrModel, AsrPrecision};

pub use crate::decoder::hotword::{
    HotwordEntry, HotwordMatch, HotwordPathKind, HotwordTokenPath, normalize_reading,
};

/// Beam width exposed by the desktop application's `ReazonSpeech` accuracy mode.
pub const REAZON_PRODUCTION_BEAM_SIZE: usize = 4;
/// Number of lattice candidates scored by the static language model in the
/// desktop application's balanced accuracy mode.
pub const REAZON_PRODUCTION_RETAINED_CANDIDATES: usize = 2;
pub const REAZON_STATIC_EMBEDDING_DIR_NAME: &str = reazon_rerank::STATIC_EMBEDDING_DIR_NAME;
pub const REAZON_STATIC_EMBEDDING_REQUIRED_FILES: &[&str] =
    reazon_rerank::STATIC_EMBEDDING_REQUIRED_FILES;

/// Beam width used by the desktop application's Japanese Parakeet accuracy mode.
pub const PARAKEET_JA_PRODUCTION_BEAM_SIZE: usize = 2;
/// CTC local-alignment margin used to admit registered hotwords in that mode.
pub const PARAKEET_JA_PRODUCTION_CTC_GATE_THRESHOLD: f32 = -5.0;
/// The acoustic top-k retained before direct hotword continuation injection.
pub const PARAKEET_JA_HOTWORD_ACOUSTIC_TOP_K: usize = 8;
/// Largest allowed CTC frame gap between successive hotword tokens.
pub const PARAKEET_JA_HOTWORD_MAX_GAP_FRAMES: usize = 8;
/// One complete hotword phrase's likelihood multiplier.
pub const PARAKEET_JA_HOTWORD_PHRASE_MULTIPLIER: f32 = 100.0;

/// Tuning controls exposed by the standalone ASR server for Japanese
/// Parakeet's variable-width TDT DAG decoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParakeetJaTdtDagConfig {
    pub beam_size: usize,
    /// `None` bypasses the CTC head completely. A threshold is applied only
    /// when at least one hotword is registered.
    pub ctc_gate_threshold: Option<f32>,
}

impl ParakeetJaTdtDagConfig {
    #[must_use]
    pub const fn production() -> Self {
        Self {
            beam_size: PARAKEET_JA_PRODUCTION_BEAM_SIZE,
            ctc_gate_threshold: Some(PARAKEET_JA_PRODUCTION_CTC_GATE_THRESHOLD),
        }
    }

    fn validate(self) -> Result<()> {
        if self.beam_size == 0 {
            anyhow::bail!("Parakeet TDT DAG beam size must be positive");
        }
        if self
            .ctc_gate_threshold
            .is_some_and(|threshold| !threshold.is_finite() || threshold > 0.0)
        {
            anyhow::bail!("Parakeet CTC hotword gate threshold must be finite and non-positive");
        }
        Ok(())
    }
}

/// Reusable Japanese static-embedding coherence scorer.
///
/// The artifact is shared with `ReazonSpeech`, but the scorer itself is
/// independent of the acoustic model and can rerank any Japanese N-best list.
pub struct JapaneseStaticEmbeddingModel {
    inner: reazon_rerank::StaticEmbeddingModel,
}

impl JapaneseStaticEmbeddingModel {
    /// Loads a pinned `static-embedding-japanese` snapshot directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the tokenizer or embedding tensor is missing or
    /// violates the pinned artifact contract.
    pub fn load(snapshot: &Path) -> Result<Self> {
        Ok(Self {
            inner: reazon_rerank::StaticEmbeddingModel::load(snapshot)?,
        })
    }

    /// Computes the normalized piece-mean coherence used by the ASR reranker.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization produces an invalid embedding row.
    pub fn piece_mean(&mut self, text: &str) -> Result<f64> {
        self.inner.piece_mean(text)
    }

    /// Returns the L2-normalized mean embedding of a sentence.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization produces an invalid embedding row.
    pub fn sentence_embedding(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.sentence_embedding(text)
    }

    /// Computes cosine similarity between two sentence embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, mismatched, or non-finite vectors.
    pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f64> {
        reazon_rerank::cosine_similarity(left, right)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReazonInitialContext {
    #[default]
    ModelReference,
    Sherpa,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReazonBeamPruning {
    #[default]
    StateDominance,
    FullPrefix,
    Sherpa,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReazonNetworkBatching {
    #[default]
    CachedUnique,
    Sherpa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReazonBeamAblation {
    pub beam_size: usize,
    pub initial_context: ReazonInitialContext,
    pub unknown_is_non_emitting: bool,
    pub pruning: ReazonBeamPruning,
    pub network_batching: ReazonNetworkBatching,
}

impl ReazonBeamAblation {
    #[must_use]
    pub const fn production(beam_size: usize) -> Self {
        Self {
            beam_size,
            initial_context: ReazonInitialContext::ModelReference,
            unknown_is_non_emitting: false,
            pruning: ReazonBeamPruning::StateDominance,
            network_batching: ReazonNetworkBatching::CachedUnique,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum AsrDecodingStrategy {
    #[default]
    Greedy,
    ReazonModifiedBeam {
        beam_size: usize,
    },
    ReazonOneSpliceRerank {
        beam_size: usize,
        retained_candidates: usize,
    },
    ReazonModifiedBeamAblation(ReazonBeamAblation),
    /// Legacy Japanese Parakeet CTC export (`model.int8.onnx`).
    ParakeetJaCtcGreedy,
    /// Shared-encoder Japanese Parakeet hybrid export, decoded with the TDT
    /// variable-width DAG. CTC gating runs only for non-empty hotwords.
    ParakeetJaTdtVariableDag(ParakeetJaTdtDagConfig),
}

impl AsrDecodingStrategy {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::ReazonModifiedBeam { .. } | Self::ReazonModifiedBeamAblation(_) => {
                "modified_beam"
            }
            Self::ReazonOneSpliceRerank { .. } => "one_splice_static_rerank",
            Self::ParakeetJaCtcGreedy => "parakeet_ctc_greedy",
            Self::ParakeetJaTdtVariableDag(_) => "parakeet_tdt_variable_dag",
        }
    }

    #[must_use]
    pub const fn beam_size(self) -> Option<usize> {
        match self {
            Self::Greedy | Self::ParakeetJaCtcGreedy => None,
            Self::ReazonModifiedBeam { beam_size }
            | Self::ReazonOneSpliceRerank { beam_size, .. } => Some(beam_size),
            Self::ReazonModifiedBeamAblation(config) => Some(config.beam_size),
            Self::ParakeetJaTdtVariableDag(config) => Some(config.beam_size),
        }
    }

    #[must_use]
    pub const fn reazon_ablation(self) -> Option<ReazonBeamAblation> {
        match self {
            Self::ReazonModifiedBeamAblation(config) => Some(config),
            Self::ReazonModifiedBeam { beam_size } => {
                Some(ReazonBeamAblation::production(beam_size))
            }
            Self::Greedy
            | Self::ReazonOneSpliceRerank { .. }
            | Self::ParakeetJaCtcGreedy
            | Self::ParakeetJaTdtVariableDag(_) => None,
        }
    }
}

/// Builds the concrete direct-ORT engine used by every host.
///
/// Hosts remain responsible for resolving and validating the model directory,
/// while desktop recognition, offline evaluation, and future adapters share
/// this single model-family dispatch.
///
/// # Errors
///
/// Returns an error when the selected model artifacts are missing or an ONNX
/// Runtime Session cannot be constructed.
pub fn build_engine(
    model_dir: &Path,
    model: AsrModel,
    precision: AsrPrecision,
    num_threads: i32,
) -> Result<Box<dyn AsrEngine>> {
    build_engine_with_decoding(
        model_dir,
        model,
        precision,
        num_threads,
        AsrDecodingStrategy::Greedy,
    )
}

/// Builds a direct-ORT engine with an explicit decoding strategy.
///
/// # Errors
///
/// Returns an error when the selected strategy is unsupported by the model,
/// model artifacts are invalid, or an ONNX Runtime Session cannot be created.
pub fn build_engine_with_decoding(
    model_dir: &Path,
    model: AsrModel,
    precision: AsrPrecision,
    num_threads: i32,
    decoding: AsrDecodingStrategy,
) -> Result<Box<dyn AsrEngine>> {
    build_engine_with_decoding_and_hotwords(model_dir, model, precision, num_threads, decoding, &[])
}

/// Builds a direct-ORT engine with explicit decoding and context-bias entries.
///
/// Hotwords are supported by beam-decoded `ReazonSpeech` and the Japanese
/// Parakeet TDT-DAG strategy. Hosts still use this shared factory instead of
/// constructing a backend concrete type themselves.
///
/// # Errors
///
/// Returns an error when hotwords are requested for greedy decoding or an
/// unsupported model family, in addition to the ordinary model-load errors.
#[allow(
    clippy::too_many_lines,
    reason = "the model-family factory keeps decoder/hotword compatibility checks adjacent to dispatch"
)]
pub fn build_engine_with_decoding_and_hotwords(
    model_dir: &Path,
    model: AsrModel,
    precision: AsrPrecision,
    num_threads: i32,
    decoding: AsrDecodingStrategy,
    hotwords: &[HotwordEntry],
) -> Result<Box<dyn AsrEngine>> {
    if matches!(
        decoding,
        AsrDecodingStrategy::ReazonModifiedBeam { .. }
            | AsrDecodingStrategy::ReazonOneSpliceRerank { .. }
            | AsrDecodingStrategy::ReazonModifiedBeamAblation(_)
    ) && model != AsrModel::ReazonSpeechK2V2
    {
        anyhow::bail!("Reazon modified beam search cannot decode {model:?}");
    }
    if matches!(
        decoding,
        AsrDecodingStrategy::ParakeetJaCtcGreedy | AsrDecodingStrategy::ParakeetJaTdtVariableDag(_)
    ) && model != AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8
    {
        anyhow::bail!("Japanese Parakeet decoding cannot decode {model:?}");
    }
    if let AsrDecodingStrategy::ParakeetJaTdtVariableDag(config) = decoding {
        config.validate()?;
    }
    if !hotwords.is_empty() {
        match (model, decoding) {
            (AsrModel::ReazonSpeechK2V2, AsrDecodingStrategy::Greedy) => {
                anyhow::bail!("Reazon hotwords require beam decoding");
            }
            (AsrModel::ReazonSpeechK2V2, _)
            | (
                AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
                AsrDecodingStrategy::ParakeetJaTdtVariableDag(_),
            ) => {}
            (AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8, _) => {
                anyhow::bail!("Japanese Parakeet hotwords require TDT DAG decoding");
            }
            _ => anyhow::bail!("hotwords cannot decode {model:?}"),
        }
    }
    let engine: Box<dyn AsrEngine> = match model {
        AsrModel::ReazonSpeechK2V2 => {
            let reazon_decoding = match decoding {
                AsrDecodingStrategy::Greedy => reazon_ort::ReazonDecodingStrategy::Greedy,
                AsrDecodingStrategy::ReazonModifiedBeam { beam_size } => {
                    reazon_ort::ReazonDecodingStrategy::ModifiedBeam { beam_size }
                }
                AsrDecodingStrategy::ReazonOneSpliceRerank {
                    beam_size,
                    retained_candidates,
                } => reazon_ort::ReazonDecodingStrategy::OneSpliceRerank {
                    beam_size,
                    retained_candidates,
                },
                AsrDecodingStrategy::ReazonModifiedBeamAblation(config) => {
                    reazon_ort::ReazonDecodingStrategy::ModifiedBeamAblation { config }
                }
                AsrDecodingStrategy::ParakeetJaCtcGreedy
                | AsrDecodingStrategy::ParakeetJaTdtVariableDag(_) => {
                    unreachable!("Japanese Parakeet strategies were rejected for ReazonSpeech")
                }
            };
            Box::new(
                reazon_ort::ReazonSpeechOrtAsrEngine::new_with_decoding_and_hotwords(
                    model_dir,
                    model,
                    precision,
                    num_threads,
                    reazon_decoding,
                    hotwords,
                )?,
            )
        }
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 => match decoding {
            AsrDecodingStrategy::Greedy | AsrDecodingStrategy::ParakeetJaCtcGreedy => {
                if precision != AsrPrecision::Int8 {
                    anyhow::bail!("Japanese Parakeet shared CTC backend only supports int8");
                }
                Box::new(parakeet_ja::SharedEncoderCtcJaOrtEngine::new(
                    model_dir,
                    num_threads,
                )?)
            }
            AsrDecodingStrategy::ParakeetJaTdtVariableDag(config) => {
                if precision != AsrPrecision::Int8 {
                    anyhow::bail!("Japanese Parakeet hybrid backend only supports the int8 export");
                }
                Box::new(parakeet_ja::ParakeetJaTdtDagOrtAsrEngine::new(
                    model_dir,
                    num_threads,
                    1,
                    parakeet_ja::FusedTdtDagConfig {
                        beam_size: config.beam_size,
                        max_symbols_per_step:
                            crate::decoder::tdt::NVIDIA_GREEDY_MAX_SYMBOLS_PER_STEP,
                        duration_expansion: parakeet_ja::FusedTdtDurationExpansion::Argmax,
                    },
                    config.ctc_gate_threshold,
                    hotwords,
                )?)
            }
            AsrDecodingStrategy::ReazonModifiedBeam { .. }
            | AsrDecodingStrategy::ReazonOneSpliceRerank { .. }
            | AsrDecodingStrategy::ReazonModifiedBeamAblation(_) => {
                unreachable!("Reazon strategies were rejected for Japanese Parakeet")
            }
        },
        AsrModel::NemoParakeetTdt0_6BV2Int8 | AsrModel::NemoParakeetTdt0_6BV3Int8 => {
            if decoding != AsrDecodingStrategy::Greedy {
                anyhow::bail!("default decoding is the only supported strategy for {model:?}");
            }
            Box::new(direct_tdt::NvidiaTdtOrtAsrEngine::new(
                model_dir,
                model,
                precision,
                num_threads,
            )?)
        }
        AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8 => {
            if decoding != AsrDecodingStrategy::Greedy {
                anyhow::bail!("default decoding is the only supported strategy for {model:?}");
            }
            Box::new(nemotron_ort::NemotronOrtAsrEngine::new(
                model_dir,
                model,
                precision,
                num_threads,
            )?)
        }
    };
    Ok(engine)
}

/// Returns the model artifact files required by the direct ORT backend.
#[must_use]
pub fn required_model_file_names(
    model: AsrModel,
    precision: AsrPrecision,
) -> &'static [&'static str] {
    match model {
        AsrModel::ReazonSpeechK2V2 => reazon_ort::required_model_file_names(precision),
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 => parakeet_ja::SHARED_CTC_REQUIRED_FILES,
        AsrModel::NemoParakeetTdt0_6BV2Int8
        | AsrModel::NemoParakeetTdt0_6BV3Int8
        | AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8 => &[
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ],
    }
}

#[cfg(test)]
mod tests {
    use crate::{AsrModel, AsrPrecision};

    use super::{
        AsrDecodingStrategy, HotwordEntry, ParakeetJaTdtDagConfig, build_engine_with_decoding,
        build_engine_with_decoding_and_hotwords, required_model_file_names,
    };

    #[test]
    fn every_direct_backend_exposes_the_artifacts_the_host_must_download() {
        for model in [
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            AsrModel::NemoParakeetTdt0_6BV3Int8,
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
        ] {
            let names = required_model_file_names(model, AsrPrecision::Int8);
            assert!(!names.is_empty(), "{model:?} must declare its artifacts");
            assert!(
                names.iter().all(|name| !name.is_empty()),
                "{model:?} contains an empty artifact name"
            );
            assert!(
                matches!(names.last(), Some(&"tokens.txt" | &"vocab.txt")),
                "{model:?} must expose its token table"
            );
        }
    }

    #[test]
    fn reazon_modified_beam_is_rejected_for_other_model_families_before_loading_files() {
        let error = build_engine_with_decoding(
            std::path::Path::new("missing"),
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            AsrPrecision::Int8,
            1,
            AsrDecodingStrategy::ReazonModifiedBeam { beam_size: 4 },
        )
        .err()
        .expect("the model/decoder mismatch must fail");

        assert_eq!(
            error.to_string(),
            "Reazon modified beam search cannot decode NemoParakeetTdtCtc0_6BJa35000Int8"
        );
    }

    #[test]
    fn hotwords_are_rejected_before_loading_for_greedy_and_other_model_families() {
        let hotwords = [HotwordEntry {
            surface: "斎藤".to_string(),
            readings: vec!["さいとう".to_string()],
            phrase_score: None,
        }];
        let greedy_error = build_engine_with_decoding_and_hotwords(
            std::path::Path::new("missing"),
            AsrModel::ReazonSpeechK2V2,
            AsrPrecision::Float32,
            1,
            AsrDecodingStrategy::Greedy,
            &hotwords,
        )
        .err()
        .expect("hotword greedy must fail before model loading");
        assert_eq!(
            greedy_error.to_string(),
            "Reazon hotwords require beam decoding"
        );

        let family_error = build_engine_with_decoding_and_hotwords(
            std::path::Path::new("missing"),
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            AsrPrecision::Int8,
            1,
            AsrDecodingStrategy::Greedy,
            &hotwords,
        )
        .err()
        .expect("another model family must reject Reazon hotwords before loading");
        assert_eq!(
            family_error.to_string(),
            "Japanese Parakeet hotwords require TDT DAG decoding"
        );
    }

    #[test]
    fn normal_reazon_beam_uses_token_count_and_last_two_token_state_with_cached_contexts() {
        let effective = AsrDecodingStrategy::ReazonModifiedBeam { beam_size: 8 }
            .reazon_ablation()
            .expect("Reazon beam exposes its effective search contract");

        assert_eq!(effective.beam_size, 8);
        assert_eq!(effective.pruning, super::ReazonBeamPruning::StateDominance);
        assert_eq!(
            effective.network_batching,
            super::ReazonNetworkBatching::CachedUnique
        );
        assert_eq!(
            effective.initial_context,
            super::ReazonInitialContext::ModelReference
        );
        assert!(!effective.unknown_is_non_emitting);
    }

    #[test]
    fn parakeet_strategy_rejects_another_model_before_artifact_loading() {
        let error = build_engine_with_decoding(
            std::path::Path::new("missing"),
            AsrModel::ReazonSpeechK2V2,
            AsrPrecision::Int8,
            1,
            AsrDecodingStrategy::ParakeetJaTdtVariableDag(ParakeetJaTdtDagConfig::production()),
        )
        .err()
        .expect("the model/decoder mismatch must fail");

        assert_eq!(
            error.to_string(),
            "Japanese Parakeet decoding cannot decode ReazonSpeechK2V2"
        );
    }

    #[test]
    fn parakeet_tdt_config_rejects_invalid_beam_and_gate_before_artifact_loading() {
        for config in [
            ParakeetJaTdtDagConfig {
                beam_size: 0,
                ctc_gate_threshold: Some(-5.0),
            },
            ParakeetJaTdtDagConfig {
                beam_size: 2,
                ctc_gate_threshold: Some(0.1),
            },
        ] {
            let error = build_engine_with_decoding(
                std::path::Path::new("missing"),
                AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
                AsrPrecision::Int8,
                1,
                AsrDecodingStrategy::ParakeetJaTdtVariableDag(config),
            )
            .err()
            .expect("the invalid Parakeet config must fail");
            assert!(error.to_string().contains("Parakeet"));
        }
    }

    #[test]
    fn parakeet_fast_strategy_requires_the_shared_encoder_and_ctc_head_contract() {
        assert_eq!(
            required_model_file_names(
                AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
                AsrPrecision::Int8,
            ),
            [
                "encoder-model.int8.onnx",
                "encoder-model.int8.onnx.data",
                "ctc-head-model.onnx",
                "ctc-head-model.onnx_data",
                "vocab.txt",
            ]
        );
    }

    #[test]
    fn parakeet_fast_build_fails_on_the_split_encoder_instead_of_the_legacy_ctc_graph() {
        let error = build_engine_with_decoding(
            std::path::Path::new("missing"),
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            AsrPrecision::Int8,
            1,
            AsrDecodingStrategy::ParakeetJaCtcGreedy,
        )
        .err()
        .expect("the missing split encoder must fail before ORT initialization");

        assert!(error.to_string().contains("shared CTC required artifact"));
        assert!(error.to_string().contains("encoder-model.int8.onnx"));
    }

    #[test]
    fn parakeet_hotwords_are_rejected_for_ctc_greedy_before_artifact_loading() {
        let error = build_engine_with_decoding_and_hotwords(
            std::path::Path::new("missing"),
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            AsrPrecision::Int8,
            1,
            AsrDecodingStrategy::ParakeetJaCtcGreedy,
            &[HotwordEntry::new("斎藤").unwrap()],
        )
        .err()
        .expect("the greedy Parakeet hotword request must fail");
        assert_eq!(
            error.to_string(),
            "Japanese Parakeet hotwords require TDT DAG decoding"
        );
    }
}
