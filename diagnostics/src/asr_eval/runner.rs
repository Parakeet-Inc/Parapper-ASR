use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use parapper_models::asr::{
    AsrModel, AsrTranscript,
    backend::reazon_ort::{
        ReazonAlignedNBestCandidate, ReazonApproximateLatticeResult, ReazonDecodingStrategy,
        ReazonEncodedAudio, ReazonNBestCandidate, ReazonRescoredNBestCandidate,
        ReazonSpeechOrtAsrEngine,
    },
    decoder::rnnt::{StatelessRnntLatticeArcMerge, StatelessRnntLengthNormalization},
};
use parapper_stt_engine::{
    OfflineTranscriptionError, OfflineTranscriptionRequest, OfflineTranscriptionResult,
    OfflineTranscriptionService, SAMPLE_RATE_HZ, StreamingFileTranscriptionService,
    prepare_offline_model_input_audio,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    HybridParakeetJaOrtEngine, ParakeetJaDecodingStrategy, ParakeetJaEncodedAudio,
    RunnerManifestV1, RunnerSampleV1, decode_canonical_pcm16_wav,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EvalRecordV1 {
    Completed {
        schema_version: u32,
        utterance_id: String,
        reference: String,
        hypothesis: String,
        duration_samples: u64,
        inference_elapsed_ms: f64,
    },
    Failed {
        schema_version: u32,
        utterance_id: String,
        stage: String,
        message: String,
    },
}

impl EvalRecordV1 {
    #[must_use]
    pub fn completed(
        utterance_id: impl Into<String>,
        reference: impl Into<String>,
        hypothesis: impl Into<String>,
        duration_samples: u64,
        inference_elapsed_ms: f64,
    ) -> Self {
        Self::Completed {
            schema_version: 1,
            utterance_id: utterance_id.into(),
            reference: reference.into(),
            hypothesis: hypothesis.into(),
            duration_samples,
            inference_elapsed_ms,
        }
    }

    #[must_use]
    pub fn failed(
        utterance_id: impl Into<String>,
        stage: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Failed {
            schema_version: 1,
            utterance_id: utterance_id.into(),
            stage: stage.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReazonNBestSearchNormalization {
    Raw,
    PerToken,
}

impl ReazonNBestSearchNormalization {
    fn model_value(self) -> StatelessRnntLengthNormalization {
        match self {
            Self::Raw => StatelessRnntLengthNormalization::Raw,
            Self::PerToken => StatelessRnntLengthNormalization::PerToken,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReazonNBestCondition {
    pub beam_size: usize,
    pub search_normalization: ReazonNBestSearchNormalization,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReazonNBestCandidateV1 {
    pub rank: usize,
    pub hypothesis: String,
    pub raw_score: f32,
    pub token_ids: Vec<usize>,
    pub timestamps: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReazonNBestRecordV1 {
    Completed {
        schema_version: u32,
        utterance_id: String,
        reference: String,
        duration_samples: u64,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
        candidates: Vec<ReazonNBestCandidateV1>,
    },
    Failed {
        schema_version: u32,
        utterance_id: String,
        stage: String,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReazonRescoredNBestCandidateV1 {
    pub rank: usize,
    pub hypothesis: String,
    pub search_raw_score: f32,
    pub viterbi_score: f32,
    pub forward_score: f32,
    pub token_ids: Vec<usize>,
    pub timestamps: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReazonRescoredNBestRecordV1 {
    Completed {
        schema_version: u32,
        utterance_id: String,
        reference: String,
        duration_samples: u64,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
        search_and_rescore_elapsed_ms: f64,
        candidates: Vec<ReazonRescoredNBestCandidateV1>,
    },
    Failed {
        schema_version: u32,
        utterance_id: String,
        stage: String,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReazonTokenAlignmentV1 {
    pub viterbi_timestamp: f32,
    pub posterior_mode_timestamp: f32,
    pub expected_timestamp: f32,
    pub posterior_lower_timestamp: f32,
    pub posterior_upper_timestamp: f32,
    pub entropy: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReazonAlignedNBestCandidateV1 {
    pub rank: usize,
    pub hypothesis: String,
    pub search_raw_score: f32,
    pub viterbi_score: f32,
    pub forward_score: f32,
    pub token_ids: Vec<usize>,
    pub search_timestamps: Vec<f32>,
    pub token_alignments: Vec<ReazonTokenAlignmentV1>,
    pub frame_emission_probabilities: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReazonAlignedNBestRecordV1 {
    Completed {
        schema_version: u32,
        utterance_id: String,
        reference: String,
        duration_samples: u64,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
        search_and_alignment_elapsed_ms: f64,
        candidates: Vec<ReazonAlignedNBestCandidateV1>,
    },
    Failed {
        schema_version: u32,
        utterance_id: String,
        stage: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReazonLatticeArcMerge {
    Maximum,
    LogMeanExp,
}

impl ReazonLatticeArcMerge {
    fn model_value(self) -> StatelessRnntLatticeArcMerge {
        match self {
            Self::Maximum => StatelessRnntLatticeArcMerge::Maximum,
            Self::LogMeanExp => StatelessRnntLatticeArcMerge::LogMeanExp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReazonApproximateLatticeCandidateV1 {
    pub rank: usize,
    pub hypothesis: String,
    pub lattice_score: f32,
    pub token_ids: Vec<usize>,
    pub is_seed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReazonApproximateLatticeRecordV1 {
    Completed {
        schema_version: u32,
        utterance_id: String,
        reference: String,
        duration_samples: u64,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
        arc_merge: ReazonLatticeArcMerge,
        search_and_lattice_elapsed_ms: f64,
        seeds: Vec<ReazonNBestCandidateV1>,
        candidates: Vec<ReazonApproximateLatticeCandidateV1>,
    },
    Failed {
        schema_version: u32,
        utterance_id: String,
        stage: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalRunSummary {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
}

pub trait EvalTranscriber {
    /// Transcribes one already-preflighted canonical PCM request.
    ///
    /// # Errors
    ///
    /// Returns a typed model availability, input, streaming lifecycle, or
    /// inference error that becomes a terminal `failed` JSONL record.
    fn transcribe(
        &mut self,
        request: OfflineTranscriptionRequest,
    ) -> Result<OfflineTranscriptionResult, OfflineTranscriptionError>;
}

/// `ReazonSpeech` inference split at the encoder/decoder boundary for
/// accuracy-only search sweeps.
pub trait CachedReazonTranscriber {
    type Encoded;

    /// Runs the shared frontend and encoder work once for one utterance.
    ///
    /// # Errors
    ///
    /// Returns an error when feature extraction or encoder inference fails.
    fn encode(&mut self, samples: &[f32]) -> Result<Self::Encoded>;

    /// Decodes one cached encoder output with a requested search strategy.
    ///
    /// # Errors
    ///
    /// Returns an error when decoder or joiner inference fails.
    fn decode_encoded(
        &mut self,
        encoded: &Self::Encoded,
        strategy: ReazonDecodingStrategy,
    ) -> Result<AsrTranscript>;
}

/// Japanese hybrid Parakeet inference split at its one shared encoder and two heads.
pub trait CachedParakeetJaTranscriber {
    type Encoded;

    /// Runs the frontend and shared encoder once for one utterance.
    ///
    /// # Errors
    ///
    /// Returns an error when feature extraction or encoder inference fails.
    fn encode_parakeet_ja(&mut self, samples: &[f32]) -> Result<Self::Encoded>;

    /// Decodes one shared encoder output with the selected CTC or TDT head.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected decoder head fails.
    fn decode_parakeet_ja(
        &mut self,
        encoded: &Self::Encoded,
        strategy: ParakeetJaDecodingStrategy,
    ) -> Result<AsrTranscript>;
}

/// `ReazonSpeech` inference split at the encoder/diagnostic-N-best boundary.
pub trait CachedReazonNBestTranscriber {
    type Encoded;

    /// Runs the frontend and encoder once for one utterance.
    ///
    /// # Errors
    ///
    /// Returns an error when frontend or encoder inference fails.
    fn encode_nbest(&mut self, samples: &[f32]) -> Result<Self::Encoded>;

    /// Returns all surviving full-prefix candidates for one search policy.
    ///
    /// # Errors
    ///
    /// Returns an error when the beam configuration or model invocation fails.
    fn decode_encoded_nbest(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
    ) -> Result<Vec<ReazonNBestCandidate>>;

    /// Generates N-best candidates and recomputes their Viterbi and forward
    /// alignment scores without beam pruning.
    ///
    /// # Errors
    ///
    /// Returns an error when search or fixed-sequence model inference fails.
    fn decode_encoded_nbest_rescored(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
    ) -> Result<Vec<ReazonRescoredNBestCandidate>>;

    /// Generates N-best candidates and computes exact fixed-transcript
    /// monotonic timing posteriors.
    ///
    /// # Errors
    ///
    /// Returns an error when search or forward-backward model inference fails.
    fn decode_encoded_nbest_aligned(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
    ) -> Result<Vec<ReazonAlignedNBestCandidate>>;

    /// Projects aligned N-best candidates onto a time-free lattice.
    ///
    /// # Errors
    ///
    /// Returns an error when search, projection, or selected-score inference
    /// fails.
    fn decode_encoded_approximate_lattice(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
        merge: ReazonLatticeArcMerge,
    ) -> Result<ReazonApproximateLatticeResult>;
}

impl CachedReazonTranscriber for ReazonSpeechOrtAsrEngine {
    type Encoded = ReazonEncodedAudio;

    fn encode(&mut self, samples: &[f32]) -> Result<Self::Encoded> {
        ReazonSpeechOrtAsrEngine::encode(self, samples)
    }

    fn decode_encoded(
        &mut self,
        encoded: &Self::Encoded,
        strategy: ReazonDecodingStrategy,
    ) -> Result<AsrTranscript> {
        ReazonSpeechOrtAsrEngine::decode_encoded(self, encoded, strategy)
    }
}

impl CachedParakeetJaTranscriber for HybridParakeetJaOrtEngine {
    type Encoded = ParakeetJaEncodedAudio;

    fn encode_parakeet_ja(&mut self, samples: &[f32]) -> Result<Self::Encoded> {
        HybridParakeetJaOrtEngine::encode(self, samples)
    }

    fn decode_parakeet_ja(
        &mut self,
        encoded: &Self::Encoded,
        strategy: ParakeetJaDecodingStrategy,
    ) -> Result<AsrTranscript> {
        HybridParakeetJaOrtEngine::decode_encoded(self, encoded, strategy)
    }
}

impl CachedReazonNBestTranscriber for ReazonSpeechOrtAsrEngine {
    type Encoded = ReazonEncodedAudio;

    fn encode_nbest(&mut self, samples: &[f32]) -> Result<Self::Encoded> {
        ReazonSpeechOrtAsrEngine::encode(self, samples)
    }

    fn decode_encoded_nbest(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
    ) -> Result<Vec<ReazonNBestCandidate>> {
        ReazonSpeechOrtAsrEngine::decode_encoded_nbest(
            self,
            encoded,
            beam_size,
            search_normalization.model_value(),
        )
    }

    fn decode_encoded_nbest_rescored(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
    ) -> Result<Vec<ReazonRescoredNBestCandidate>> {
        ReazonSpeechOrtAsrEngine::decode_encoded_nbest_rescored(
            self,
            encoded,
            beam_size,
            search_normalization.model_value(),
        )
    }

    fn decode_encoded_nbest_aligned(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
    ) -> Result<Vec<ReazonAlignedNBestCandidate>> {
        ReazonSpeechOrtAsrEngine::decode_encoded_nbest_aligned(
            self,
            encoded,
            beam_size,
            search_normalization.model_value(),
        )
    }

    fn decode_encoded_approximate_lattice(
        &mut self,
        encoded: &Self::Encoded,
        beam_size: usize,
        search_normalization: ReazonNBestSearchNormalization,
        merge: ReazonLatticeArcMerge,
    ) -> Result<ReazonApproximateLatticeResult> {
        ReazonSpeechOrtAsrEngine::decode_encoded_approximate_lattice(
            self,
            encoded,
            beam_size,
            search_normalization.model_value(),
            merge.model_value(),
        )
    }
}

impl EvalTranscriber for OfflineTranscriptionService {
    fn transcribe(
        &mut self,
        request: OfflineTranscriptionRequest,
    ) -> Result<OfflineTranscriptionResult, OfflineTranscriptionError> {
        Self::transcribe(self, request)
    }
}

impl EvalTranscriber for StreamingFileTranscriptionService {
    fn transcribe(
        &mut self,
        request: OfflineTranscriptionRequest,
    ) -> Result<OfflineTranscriptionResult, OfflineTranscriptionError> {
        Self::transcribe(self, request)
    }
}

/// Runs a preflighted evaluation manifest one file at a time.
///
/// The model session is owned by `service` and reused across every utterance.
/// Audio buffers are read, verified, inferred, serialized, and dropped in
/// manifest order, so a 1,000-item subset is never cached in memory. Each JSONL
/// record is flushed before the next item and failures are recorded, not skipped.
///
/// # Errors
///
/// Returns an error only when the JSONL output cannot be written. Per-sample
/// audio and inference errors become `failed` records and remain in the summary.
pub fn run_manifest<W: Write, T: EvalTranscriber + ?Sized>(
    manifest_path: &Path,
    manifest: &RunnerManifestV1,
    model: AsrModel,
    service: &mut T,
    writer: &mut W,
) -> Result<EvalRunSummary> {
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut summary = EvalRunSummary {
        total: manifest.samples.len(),
        completed: 0,
        failed: 0,
    };

    for sample in &manifest.samples {
        let record = process_sample(manifest_root, sample, model, service);
        write_record(writer, &record)?;
        match record {
            EvalRecordV1::Completed { .. } => summary.completed += 1,
            EvalRecordV1::Failed { .. } => summary.failed += 1,
        }
    }

    Ok(summary)
}

/// Runs several `ReazonSpeech` search strategies from one encoder output per
/// utterance.
///
/// This path is deliberately accuracy-only. Every completed JSONL record uses
/// `inference_elapsed_ms = 0.0` because encoder work and predictor cache entries
/// are shared across conditions and cannot produce comparable latency values.
///
/// # Errors
///
/// Returns an error for a strategy/output cardinality mismatch or when any
/// output JSONL cannot be written. Audio, encoder, and decoder failures remain
/// terminal records in every affected condition.
pub fn run_reazon_accuracy_sweep<T: CachedReazonTranscriber + ?Sized>(
    manifest_path: &Path,
    manifest: &RunnerManifestV1,
    transcriber: &mut T,
    strategies: &[ReazonDecodingStrategy],
    outputs: &mut [&mut dyn Write],
) -> Result<Vec<EvalRunSummary>> {
    if strategies.is_empty() || strategies.len() != outputs.len() {
        bail!(
            "Reazon accuracy sweep requires one output per strategy (strategies={}, outputs={})",
            strategies.len(),
            outputs.len()
        );
    }
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut summaries = vec![
        EvalRunSummary {
            total: manifest.samples.len(),
            completed: 0,
            failed: 0,
        };
        strategies.len()
    ];

    for sample in &manifest.samples {
        let wav = match load_preflighted_audio(manifest_root, sample) {
            Ok(wav) => wav,
            Err(record) => {
                for (output, summary) in outputs.iter_mut().zip(&mut summaries) {
                    write_record(*output, &record)?;
                    summary.failed += 1;
                }
                continue;
            }
        };
        let model_input =
            prepare_offline_model_input_audio(AsrModel::ReazonSpeechK2V2, &wav.samples);
        let encoded = match transcriber.encode(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                let record =
                    EvalRecordV1::failed(&sample.utterance_id, "encoder", error.to_string());
                for (output, summary) in outputs.iter_mut().zip(&mut summaries) {
                    write_record(*output, &record)?;
                    summary.failed += 1;
                }
                continue;
            }
        };

        for ((strategy, output), summary) in strategies
            .iter()
            .zip(outputs.iter_mut())
            .zip(&mut summaries)
        {
            let record = match transcriber.decode_encoded(&encoded, *strategy) {
                Ok(transcript) => EvalRecordV1::completed(
                    &sample.utterance_id,
                    &sample.reference.normalized,
                    transcript.text,
                    wav.samples.len() as u64,
                    0.0,
                ),
                Err(error) => {
                    EvalRecordV1::failed(&sample.utterance_id, "decoder", error.to_string())
                }
            };
            write_record(*output, &record)?;
            match record {
                EvalRecordV1::Completed { .. } => summary.completed += 1,
                EvalRecordV1::Failed { .. } => summary.failed += 1,
            }
        }
    }

    Ok(summaries)
}

/// Runs the CTC and TDT heads from one shared Japanese Parakeet encoder output.
///
/// This path is accuracy-only: its completed records use zero elapsed time
/// because the encoder is intentionally shared. End-to-end latency must be
/// measured by running each configured branch through [`run_manifest`].
///
/// # Errors
///
/// Returns an error for a strategy/output cardinality mismatch or an output
/// write failure. Per-sample audio and inference errors remain terminal JSONL
/// records for every affected branch.
pub fn run_parakeet_ja_accuracy_sweep<T: CachedParakeetJaTranscriber + ?Sized>(
    manifest_path: &Path,
    manifest: &RunnerManifestV1,
    transcriber: &mut T,
    strategies: &[ParakeetJaDecodingStrategy],
    outputs: &mut [&mut dyn Write],
) -> Result<Vec<EvalRunSummary>> {
    if strategies.is_empty() || strategies.len() != outputs.len() {
        bail!(
            "Parakeet ja accuracy sweep requires one output per strategy (strategies={}, outputs={})",
            strategies.len(),
            outputs.len()
        );
    }
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut summaries = vec![
        EvalRunSummary {
            total: manifest.samples.len(),
            completed: 0,
            failed: 0,
        };
        strategies.len()
    ];
    let model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;

    for sample in &manifest.samples {
        let wav = match load_preflighted_audio(manifest_root, sample) {
            Ok(wav) => wav,
            Err(record) => {
                for (output, summary) in outputs.iter_mut().zip(&mut summaries) {
                    write_record(*output, &record)?;
                    summary.failed += 1;
                }
                continue;
            }
        };
        let model_input = prepare_offline_model_input_audio(model, &wav.samples);
        let encoded = match transcriber.encode_parakeet_ja(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                let record =
                    EvalRecordV1::failed(&sample.utterance_id, "encoder", error.to_string());
                for (output, summary) in outputs.iter_mut().zip(&mut summaries) {
                    write_record(*output, &record)?;
                    summary.failed += 1;
                }
                continue;
            }
        };

        for ((strategy, output), summary) in strategies
            .iter()
            .zip(outputs.iter_mut())
            .zip(&mut summaries)
        {
            let record = match transcriber.decode_parakeet_ja(&encoded, *strategy) {
                Ok(transcript) => EvalRecordV1::completed(
                    &sample.utterance_id,
                    &sample.reference.normalized,
                    transcript.text,
                    wav.samples.len() as u64,
                    0.0,
                ),
                Err(error) => {
                    EvalRecordV1::failed(&sample.utterance_id, "decoder", error.to_string())
                }
            };
            write_record(*output, &record)?;
            match record {
                EvalRecordV1::Completed { .. } => summary.completed += 1,
                EvalRecordV1::Failed { .. } => summary.failed += 1,
            }
        }
    }

    Ok(summaries)
}

/// Runs full-prefix `ReazonSpeech` N-best searches from one encoder output per
/// utterance.
///
/// Each condition owns one JSONL stream. Candidate ranks are the search
/// policy's native final order, while raw acoustic scores and token counts are
/// retained so quarter-step final length penalties can be evaluated without
/// another decoder/joiner invocation.
///
/// # Errors
///
/// Returns an error for a condition/output cardinality mismatch or an output
/// write failure. Audio, encoder, and decoder failures become terminal records
/// in every affected condition.
pub fn run_reazon_nbest_sweep<T: CachedReazonNBestTranscriber + ?Sized>(
    manifest_path: &Path,
    manifest: &RunnerManifestV1,
    transcriber: &mut T,
    conditions: &[ReazonNBestCondition],
    outputs: &mut [&mut dyn Write],
) -> Result<Vec<EvalRunSummary>> {
    if conditions.is_empty() || conditions.len() != outputs.len() {
        bail!(
            "Reazon N-best sweep requires one output per condition (conditions={}, outputs={})",
            conditions.len(),
            outputs.len()
        );
    }
    if conditions.iter().any(|condition| condition.beam_size == 0) {
        bail!("Reazon N-best sweep requires positive beam sizes");
    }
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut summaries = vec![
        EvalRunSummary {
            total: manifest.samples.len(),
            completed: 0,
            failed: 0,
        };
        conditions.len()
    ];

    for sample in &manifest.samples {
        let wav = match load_preflighted_audio(manifest_root, sample) {
            Ok(wav) => wav,
            Err(record) => {
                let failed = nbest_failure_from_eval_record(record);
                for (output, summary) in outputs.iter_mut().zip(&mut summaries) {
                    write_nbest_record(*output, &failed)?;
                    summary.failed += 1;
                }
                continue;
            }
        };
        let model_input =
            prepare_offline_model_input_audio(AsrModel::ReazonSpeechK2V2, &wav.samples);
        let encoded = match transcriber.encode_nbest(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                let failed = ReazonNBestRecordV1::Failed {
                    schema_version: 1,
                    utterance_id: sample.utterance_id.clone(),
                    stage: "encoder".to_owned(),
                    message: error.to_string(),
                };
                for (output, summary) in outputs.iter_mut().zip(&mut summaries) {
                    write_nbest_record(*output, &failed)?;
                    summary.failed += 1;
                }
                continue;
            }
        };

        for ((condition, output), summary) in conditions
            .iter()
            .zip(outputs.iter_mut())
            .zip(&mut summaries)
        {
            let record = match transcriber.decode_encoded_nbest(
                &encoded,
                condition.beam_size,
                condition.search_normalization,
            ) {
                Ok(candidates) => ReazonNBestRecordV1::Completed {
                    schema_version: 1,
                    utterance_id: sample.utterance_id.clone(),
                    reference: sample.reference.normalized.clone(),
                    duration_samples: wav.samples.len() as u64,
                    beam_size: condition.beam_size,
                    search_normalization: condition.search_normalization,
                    candidates: candidates
                        .into_iter()
                        .enumerate()
                        .map(|(rank, candidate)| ReazonNBestCandidateV1 {
                            rank: rank + 1,
                            hypothesis: candidate.text,
                            raw_score: candidate.raw_score,
                            token_ids: candidate.token_ids,
                            timestamps: candidate.timestamps,
                        })
                        .collect(),
                },
                Err(error) => ReazonNBestRecordV1::Failed {
                    schema_version: 1,
                    utterance_id: sample.utterance_id.clone(),
                    stage: "decoder".to_owned(),
                    message: error.to_string(),
                },
            };
            write_nbest_record(*output, &record)?;
            match record {
                ReazonNBestRecordV1::Completed { .. } => summary.completed += 1,
                ReazonNBestRecordV1::Failed { .. } => summary.failed += 1,
            }
        }
    }

    Ok(summaries)
}

/// Generates one full-prefix N-best set per utterance and recomputes exact
/// fixed-transcript scores over the frame-synchronous alignment graph.
///
/// # Errors
///
/// Returns an error for a zero beam or output write failure. Per-utterance
/// audio, encoder, search, and rescore failures are written as terminal JSONL
/// records.
pub fn run_reazon_nbest_rescore<T: CachedReazonNBestTranscriber + ?Sized>(
    manifest_path: &Path,
    manifest: &RunnerManifestV1,
    transcriber: &mut T,
    condition: ReazonNBestCondition,
    output: &mut dyn Write,
) -> Result<EvalRunSummary> {
    if condition.beam_size == 0 {
        bail!("Reazon N-best rescore requires a positive beam size");
    }
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut summary = EvalRunSummary {
        total: manifest.samples.len(),
        completed: 0,
        failed: 0,
    };
    for sample in &manifest.samples {
        let wav = match load_preflighted_audio(manifest_root, sample) {
            Ok(wav) => wav,
            Err(record) => {
                let failed = rescored_nbest_failure_from_eval_record(record);
                write_rescored_nbest_record(output, &failed)?;
                summary.failed += 1;
                continue;
            }
        };
        let model_input =
            prepare_offline_model_input_audio(AsrModel::ReazonSpeechK2V2, &wav.samples);
        let encoded = match transcriber.encode_nbest(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                let failed = ReazonRescoredNBestRecordV1::Failed {
                    schema_version: 1,
                    utterance_id: sample.utterance_id.clone(),
                    stage: "encoder".to_owned(),
                    message: error.to_string(),
                };
                write_rescored_nbest_record(output, &failed)?;
                summary.failed += 1;
                continue;
            }
        };
        let started = Instant::now();
        let record = match transcriber.decode_encoded_nbest_rescored(
            &encoded,
            condition.beam_size,
            condition.search_normalization,
        ) {
            Ok(candidates) => ReazonRescoredNBestRecordV1::Completed {
                schema_version: 1,
                utterance_id: sample.utterance_id.clone(),
                reference: sample.reference.normalized.clone(),
                duration_samples: wav.samples.len() as u64,
                beam_size: condition.beam_size,
                search_normalization: condition.search_normalization,
                search_and_rescore_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                candidates: candidates
                    .into_iter()
                    .enumerate()
                    .map(|(rank, rescored)| ReazonRescoredNBestCandidateV1 {
                        rank: rank + 1,
                        hypothesis: rescored.candidate.text,
                        search_raw_score: rescored.candidate.raw_score,
                        viterbi_score: rescored.viterbi_score,
                        forward_score: rescored.forward_score,
                        token_ids: rescored.candidate.token_ids,
                        timestamps: rescored.candidate.timestamps,
                    })
                    .collect(),
            },
            Err(error) => ReazonRescoredNBestRecordV1::Failed {
                schema_version: 1,
                utterance_id: sample.utterance_id.clone(),
                stage: "rescore".to_owned(),
                message: error.to_string(),
            },
        };
        write_rescored_nbest_record(output, &record)?;
        match record {
            ReazonRescoredNBestRecordV1::Completed { .. } => summary.completed += 1,
            ReazonRescoredNBestRecordV1::Failed { .. } => summary.failed += 1,
        }
    }
    Ok(summary)
}

/// Generates width-N full-prefix candidates and writes exact monotonic timing
/// posteriors for every fixed transcript.
///
/// # Errors
///
/// Returns an error for a zero beam or output write failure. Per-utterance
/// audio, encoder, search, and alignment failures become terminal JSONL
/// records so dataset accounting remains explicit.
pub fn run_reazon_monotonic_alignment<T: CachedReazonNBestTranscriber + ?Sized>(
    manifest_path: &Path,
    manifest: &RunnerManifestV1,
    transcriber: &mut T,
    condition: ReazonNBestCondition,
    output: &mut dyn Write,
) -> Result<EvalRunSummary> {
    if condition.beam_size == 0 {
        bail!("Reazon monotonic alignment requires a positive beam size");
    }
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut summary = EvalRunSummary {
        total: manifest.samples.len(),
        completed: 0,
        failed: 0,
    };
    for sample in &manifest.samples {
        let wav = match load_preflighted_audio(manifest_root, sample) {
            Ok(wav) => wav,
            Err(record) => {
                let failed = aligned_nbest_failure_from_eval_record(record);
                write_aligned_nbest_record(output, &failed)?;
                summary.failed += 1;
                continue;
            }
        };
        let model_input =
            prepare_offline_model_input_audio(AsrModel::ReazonSpeechK2V2, &wav.samples);
        let encoded = match transcriber.encode_nbest(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                let failed = ReazonAlignedNBestRecordV1::Failed {
                    schema_version: 1,
                    utterance_id: sample.utterance_id.clone(),
                    stage: "encoder".to_owned(),
                    message: error.to_string(),
                };
                write_aligned_nbest_record(output, &failed)?;
                summary.failed += 1;
                continue;
            }
        };
        let started = Instant::now();
        let record = match transcriber.decode_encoded_nbest_aligned(
            &encoded,
            condition.beam_size,
            condition.search_normalization,
        ) {
            Ok(candidates) => ReazonAlignedNBestRecordV1::Completed {
                schema_version: 1,
                utterance_id: sample.utterance_id.clone(),
                reference: sample.reference.normalized.clone(),
                duration_samples: wav.samples.len() as u64,
                beam_size: condition.beam_size,
                search_normalization: condition.search_normalization,
                search_and_alignment_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                candidates: candidates
                    .into_iter()
                    .enumerate()
                    .map(|(rank, aligned)| ReazonAlignedNBestCandidateV1 {
                        rank: rank + 1,
                        hypothesis: aligned.candidate.text,
                        search_raw_score: aligned.candidate.raw_score,
                        viterbi_score: aligned.viterbi_score,
                        forward_score: aligned.forward_score,
                        token_ids: aligned.candidate.token_ids,
                        search_timestamps: aligned.candidate.timestamps,
                        token_alignments: aligned
                            .tokens
                            .into_iter()
                            .map(|token| ReazonTokenAlignmentV1 {
                                viterbi_timestamp: token.viterbi_timestamp,
                                posterior_mode_timestamp: token.posterior_mode_timestamp,
                                expected_timestamp: token.expected_timestamp,
                                posterior_lower_timestamp: token.posterior_lower_timestamp,
                                posterior_upper_timestamp: token.posterior_upper_timestamp,
                                entropy: token.entropy,
                            })
                            .collect(),
                        frame_emission_probabilities: aligned.frame_emission_probabilities,
                    })
                    .collect(),
            },
            Err(error) => ReazonAlignedNBestRecordV1::Failed {
                schema_version: 1,
                utterance_id: sample.utterance_id.clone(),
                stage: "alignment".to_owned(),
                message: error.to_string(),
            },
        };
        write_aligned_nbest_record(output, &record)?;
        match record {
            ReazonAlignedNBestRecordV1::Completed { .. } => summary.completed += 1,
            ReazonAlignedNBestRecordV1::Failed { .. } => summary.failed += 1,
        }
    }
    Ok(summary)
}

/// Generates width-N full-prefix seeds, projects their representative
/// alignments onto a time-free context lattice, and writes every recovered
/// terminal path.
///
/// # Errors
///
/// Returns an error for a zero beam or output write failure. Per-utterance
/// inference failures are written as terminal JSONL records.
pub fn run_reazon_approximate_lattice<T: CachedReazonNBestTranscriber + ?Sized>(
    manifest_path: &Path,
    manifest: &RunnerManifestV1,
    transcriber: &mut T,
    condition: ReazonNBestCondition,
    merge: ReazonLatticeArcMerge,
    output: &mut dyn Write,
) -> Result<EvalRunSummary> {
    if condition.beam_size == 0 {
        bail!("Reazon approximate lattice requires a positive beam size");
    }
    let manifest_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut summary = EvalRunSummary {
        total: manifest.samples.len(),
        completed: 0,
        failed: 0,
    };
    for sample in &manifest.samples {
        let wav = match load_preflighted_audio(manifest_root, sample) {
            Ok(wav) => wav,
            Err(record) => {
                let failed = lattice_failure_from_eval_record(record);
                write_lattice_record(output, &failed)?;
                summary.failed += 1;
                continue;
            }
        };
        let model_input =
            prepare_offline_model_input_audio(AsrModel::ReazonSpeechK2V2, &wav.samples);
        let encoded = match transcriber.encode_nbest(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                let failed = ReazonApproximateLatticeRecordV1::Failed {
                    schema_version: 1,
                    utterance_id: sample.utterance_id.clone(),
                    stage: "encoder".to_owned(),
                    message: error.to_string(),
                };
                write_lattice_record(output, &failed)?;
                summary.failed += 1;
                continue;
            }
        };
        let started = Instant::now();
        let record = match transcriber.decode_encoded_approximate_lattice(
            &encoded,
            condition.beam_size,
            condition.search_normalization,
            merge,
        ) {
            Ok(result) => ReazonApproximateLatticeRecordV1::Completed {
                schema_version: 1,
                utterance_id: sample.utterance_id.clone(),
                reference: sample.reference.normalized.clone(),
                duration_samples: wav.samples.len() as u64,
                beam_size: condition.beam_size,
                search_normalization: condition.search_normalization,
                arc_merge: merge,
                search_and_lattice_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                seeds: result
                    .seeds
                    .into_iter()
                    .enumerate()
                    .map(|(rank, candidate)| ReazonNBestCandidateV1 {
                        rank: rank + 1,
                        hypothesis: candidate.text,
                        raw_score: candidate.raw_score,
                        token_ids: candidate.token_ids,
                        timestamps: candidate.timestamps,
                    })
                    .collect(),
                candidates: result
                    .candidates
                    .into_iter()
                    .enumerate()
                    .map(|(rank, candidate)| ReazonApproximateLatticeCandidateV1 {
                        rank: rank + 1,
                        hypothesis: candidate.text,
                        lattice_score: candidate.score,
                        token_ids: candidate.token_ids,
                        is_seed: candidate.is_seed,
                    })
                    .collect(),
            },
            Err(error) => ReazonApproximateLatticeRecordV1::Failed {
                schema_version: 1,
                utterance_id: sample.utterance_id.clone(),
                stage: "lattice".to_owned(),
                message: error.to_string(),
            },
        };
        write_lattice_record(output, &record)?;
        match record {
            ReazonApproximateLatticeRecordV1::Completed { .. } => summary.completed += 1,
            ReazonApproximateLatticeRecordV1::Failed { .. } => summary.failed += 1,
        }
    }
    Ok(summary)
}

fn nbest_failure_from_eval_record(record: EvalRecordV1) -> ReazonNBestRecordV1 {
    match record {
        EvalRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        } => ReazonNBestRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        },
        EvalRecordV1::Completed { .. } => {
            unreachable!("audio preflight failures never produce completed records")
        }
    }
}

fn rescored_nbest_failure_from_eval_record(record: EvalRecordV1) -> ReazonRescoredNBestRecordV1 {
    match record {
        EvalRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        } => ReazonRescoredNBestRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        },
        EvalRecordV1::Completed { .. } => {
            unreachable!("audio preflight only returns failed records")
        }
    }
}

fn aligned_nbest_failure_from_eval_record(record: EvalRecordV1) -> ReazonAlignedNBestRecordV1 {
    match record {
        EvalRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        } => ReazonAlignedNBestRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        },
        EvalRecordV1::Completed { .. } => {
            unreachable!("audio preflight only returns failed records")
        }
    }
}

fn lattice_failure_from_eval_record(record: EvalRecordV1) -> ReazonApproximateLatticeRecordV1 {
    match record {
        EvalRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        } => ReazonApproximateLatticeRecordV1::Failed {
            schema_version,
            utterance_id,
            stage,
            message,
        },
        EvalRecordV1::Completed { .. } => {
            unreachable!("audio preflight only returns failed records")
        }
    }
}

fn process_sample<T: EvalTranscriber + ?Sized>(
    manifest_root: &Path,
    sample: &RunnerSampleV1,
    model: AsrModel,
    service: &mut T,
) -> EvalRecordV1 {
    let wav = match load_preflighted_audio(manifest_root, sample) {
        Ok(wav) => wav,
        Err(record) => return record,
    };
    let started = Instant::now();
    let result = service.transcribe(OfflineTranscriptionRequest {
        job_id: sample.utterance_id.clone(),
        model,
        sample_rate_hz: SAMPLE_RATE_HZ,
        samples: wav.samples,
    });
    let inference_elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    match result {
        Ok(result) => EvalRecordV1::completed(
            &sample.utterance_id,
            &sample.reference.normalized,
            result.transcript.text,
            result.source_duration_samples,
            inference_elapsed_ms,
        ),
        Err(error) => EvalRecordV1::failed(&sample.utterance_id, "inference", error.to_string()),
    }
}

fn load_preflighted_audio(
    manifest_root: &Path,
    sample: &RunnerSampleV1,
) -> std::result::Result<super::CanonicalWav, EvalRecordV1> {
    let audio_path = manifest_root.join(&sample.audio.relative_path);
    let bytes = match fs::read(&audio_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(EvalRecordV1::failed(
                &sample.utterance_id,
                "audio_read",
                format!("{}: {error}", audio_path.display()),
            ));
        }
    };
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != sample.audio.sha256 {
        return Err(EvalRecordV1::failed(
            &sample.utterance_id,
            "audio_preflight",
            format!(
                "SHA-256 mismatch: expected {}, got {actual_sha256}",
                sample.audio.sha256
            ),
        ));
    }
    let wav = match decode_canonical_pcm16_wav(&bytes) {
        Ok(wav) => wav,
        Err(error) => {
            return Err(EvalRecordV1::failed(
                &sample.utterance_id,
                "audio_preflight",
                error.to_string(),
            ));
        }
    };
    if wav.samples.len() as u64 != sample.audio.duration_samples {
        return Err(EvalRecordV1::failed(
            &sample.utterance_id,
            "audio_preflight",
            format!(
                "duration_samples mismatch: expected {}, got {}",
                sample.audio.duration_samples,
                wav.samples.len()
            ),
        ));
    }
    Ok(wav)
}

fn write_record<W: Write + ?Sized>(writer: &mut W, record: &EvalRecordV1) -> Result<()> {
    serde_json::to_writer(&mut *writer, record).context("failed to serialize JSONL record")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate JSONL record")?;
    writer.flush().context("failed to flush JSONL record")
}

fn write_nbest_record<W: Write + ?Sized>(
    writer: &mut W,
    record: &ReazonNBestRecordV1,
) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)
        .context("failed to serialize N-best JSONL record")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate N-best JSONL record")?;
    writer
        .flush()
        .context("failed to flush N-best JSONL record")
}

fn write_rescored_nbest_record<W: Write + ?Sized>(
    writer: &mut W,
    record: &ReazonRescoredNBestRecordV1,
) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)
        .context("failed to serialize rescored N-best JSONL record")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate rescored N-best JSONL record")?;
    writer
        .flush()
        .context("failed to flush rescored N-best JSONL record")
}

fn write_aligned_nbest_record<W: Write + ?Sized>(
    writer: &mut W,
    record: &ReazonAlignedNBestRecordV1,
) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)
        .context("failed to serialize aligned N-best JSONL record")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate aligned N-best JSONL record")?;
    writer
        .flush()
        .context("failed to flush aligned N-best JSONL record")
}

fn write_lattice_record<W: Write + ?Sized>(
    writer: &mut W,
    record: &ReazonApproximateLatticeRecordV1,
) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)
        .context("failed to serialize approximate lattice JSONL record")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate approximate lattice JSONL record")?;
    writer
        .flush()
        .context("failed to flush approximate lattice JSONL record")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use anyhow::Result;
    use parapper_models::asr::{
        AsrEngine, AsrModel, AsrTranscript,
        backend::reazon_ort::{
            ReazonAlignedNBestCandidate, ReazonApproximateLatticeCandidate,
            ReazonApproximateLatticeResult, ReazonDecodingStrategy, ReazonNBestCandidate,
            ReazonRescoredNBestCandidate, ReazonTokenAlignment,
        },
    };
    use parapper_stt_engine::{AsrModelRegistry, OfflineTranscriptionService};
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::{
        CachedParakeetJaTranscriber, CachedReazonNBestTranscriber, CachedReazonTranscriber,
        EvalRecordV1, EvalRunSummary, ReazonAlignedNBestCandidateV1, ReazonAlignedNBestRecordV1,
        ReazonApproximateLatticeCandidateV1, ReazonApproximateLatticeRecordV1,
        ReazonLatticeArcMerge, ReazonNBestCondition, ReazonNBestRecordV1,
        ReazonNBestSearchNormalization, ReazonRescoredNBestCandidateV1,
        ReazonRescoredNBestRecordV1, ReazonTokenAlignmentV1, run_manifest,
        run_parakeet_ja_accuracy_sweep, run_reazon_accuracy_sweep, run_reazon_approximate_lattice,
        run_reazon_monotonic_alignment, run_reazon_nbest_rescore, run_reazon_nbest_sweep,
    };
    use crate::asr_eval::{
        AudioFormatContract, DatasetIdentity, DerivedAudio, FusedTdtDagConfig,
        FusedTdtDurationExpansion, NormalizationIdentity, ParakeetJaDecodingStrategy,
        ReferenceText, RunnerManifestV1, RunnerSampleV1,
    };

    struct FixedEngine {
        calls: Arc<Mutex<Vec<usize>>>,
    }

    struct RecordingCachedReazon {
        events: Vec<String>,
    }

    struct RecordingCachedParakeetJa {
        events: Vec<String>,
    }

    struct RecordingNBestReazon {
        events: Vec<String>,
    }

    impl CachedReazonTranscriber for RecordingCachedReazon {
        type Encoded = usize;

        fn encode(&mut self, samples: &[f32]) -> Result<Self::Encoded> {
            self.events.push(format!("encode:{}", samples.len()));
            Ok(samples.len())
        }

        fn decode_encoded(
            &mut self,
            encoded: &Self::Encoded,
            strategy: ReazonDecodingStrategy,
        ) -> Result<AsrTranscript> {
            let label = match strategy {
                ReazonDecodingStrategy::Greedy => "greedy".to_owned(),
                ReazonDecodingStrategy::ModifiedBeam { beam_size } => {
                    format!("state-{beam_size}")
                }
                ReazonDecodingStrategy::OneSpliceRerank {
                    beam_size,
                    retained_candidates,
                } => format!("one-splice-{beam_size}-retain-{retained_candidates}"),
                ReazonDecodingStrategy::ModifiedBeamAblation { .. } => {
                    unreachable!("the test uses production strategies")
                }
            };
            self.events.push(format!("decode:{label}:{encoded}"));
            Ok(AsrTranscript::from_text(label))
        }
    }

    impl CachedParakeetJaTranscriber for RecordingCachedParakeetJa {
        type Encoded = usize;

        fn encode_parakeet_ja(&mut self, samples: &[f32]) -> Result<Self::Encoded> {
            self.events.push(format!("encode:{}", samples.len()));
            Ok(samples.len())
        }

        fn decode_parakeet_ja(
            &mut self,
            encoded: &Self::Encoded,
            strategy: ParakeetJaDecodingStrategy,
        ) -> Result<AsrTranscript> {
            let label = match strategy {
                ParakeetJaDecodingStrategy::CtcGreedy => "ctc".to_owned(),
                ParakeetJaDecodingStrategy::TdtGreedy => "tdt".to_owned(),
                ParakeetJaDecodingStrategy::TdtVariableDag(config) => {
                    format!("tdt-dag-{}", config.beam_size)
                }
                ParakeetJaDecodingStrategy::TdtVariableDagStaticEmbedding(config) => {
                    format!("tdt-dag-{}-static", config.beam_size)
                }
            };
            self.events.push(format!("decode:{label}:{encoded}"));
            Ok(AsrTranscript::from_text(label))
        }
    }

    impl CachedReazonNBestTranscriber for RecordingNBestReazon {
        type Encoded = usize;

        fn encode_nbest(&mut self, samples: &[f32]) -> Result<Self::Encoded> {
            self.events.push(format!("encode:{}", samples.len()));
            Ok(samples.len())
        }

        fn decode_encoded_nbest(
            &mut self,
            encoded: &Self::Encoded,
            beam_size: usize,
            search_normalization: ReazonNBestSearchNormalization,
        ) -> Result<Vec<ReazonNBestCandidate>> {
            self.events.push(format!(
                "decode:{beam_size}:{search_normalization:?}:{encoded}"
            ));
            Ok(vec![
                ReazonNBestCandidate {
                    raw_score: -1.0,
                    token_ids: vec![1, 2],
                    timestamps: vec![0.0, 0.04],
                    text: "第一候補".to_owned(),
                },
                ReazonNBestCandidate {
                    raw_score: -1.1,
                    token_ids: vec![3],
                    timestamps: vec![0.08],
                    text: "別候補".to_owned(),
                },
            ])
        }

        fn decode_encoded_nbest_rescored(
            &mut self,
            encoded: &Self::Encoded,
            beam_size: usize,
            search_normalization: ReazonNBestSearchNormalization,
        ) -> Result<Vec<ReazonRescoredNBestCandidate>> {
            self.events.push(format!(
                "rescore:{beam_size}:{search_normalization:?}:{encoded}"
            ));
            Ok(vec![
                ReazonRescoredNBestCandidate {
                    candidate: ReazonNBestCandidate {
                        raw_score: -1.0,
                        token_ids: vec![1, 2],
                        timestamps: vec![0.0, 0.04],
                        text: "第一候補".to_owned(),
                    },
                    viterbi_score: -1.5,
                    forward_score: -0.9,
                },
                ReazonRescoredNBestCandidate {
                    candidate: ReazonNBestCandidate {
                        raw_score: -1.1,
                        token_ids: vec![3],
                        timestamps: vec![0.08],
                        text: "別候補".to_owned(),
                    },
                    viterbi_score: -1.4,
                    forward_score: -0.8,
                },
            ])
        }

        fn decode_encoded_nbest_aligned(
            &mut self,
            encoded: &Self::Encoded,
            beam_size: usize,
            search_normalization: ReazonNBestSearchNormalization,
        ) -> Result<Vec<ReazonAlignedNBestCandidate>> {
            self.events.push(format!(
                "align:{beam_size}:{search_normalization:?}:{encoded}"
            ));
            Ok(vec![ReazonAlignedNBestCandidate {
                candidate: ReazonNBestCandidate {
                    raw_score: -1.0,
                    token_ids: vec![1],
                    timestamps: vec![0.04],
                    text: "整列候補".to_owned(),
                },
                viterbi_score: -1.2,
                forward_score: -0.8,
                tokens: vec![ReazonTokenAlignment {
                    viterbi_timestamp: 0.08,
                    posterior_mode_timestamp: 0.08,
                    expected_timestamp: 0.09,
                    posterior_lower_timestamp: 0.04,
                    posterior_upper_timestamp: 0.16,
                    entropy: 0.5,
                }],
                frame_emission_probabilities: vec![0.1, 0.7, 0.2],
            }])
        }

        fn decode_encoded_approximate_lattice(
            &mut self,
            encoded: &Self::Encoded,
            beam_size: usize,
            search_normalization: ReazonNBestSearchNormalization,
            merge: ReazonLatticeArcMerge,
        ) -> Result<ReazonApproximateLatticeResult> {
            self.events.push(format!(
                "lattice:{beam_size}:{search_normalization:?}:{merge:?}:{encoded}"
            ));
            Ok(ReazonApproximateLatticeResult {
                seeds: vec![ReazonNBestCandidate {
                    raw_score: -1.0,
                    token_ids: vec![1, 2],
                    timestamps: vec![0.0, 0.04],
                    text: "第一候補".to_owned(),
                }],
                candidates: vec![ReazonApproximateLatticeCandidate {
                    score: -0.7,
                    token_ids: vec![1, 3],
                    text: "再結合候補".to_owned(),
                    is_seed: false,
                }],
            })
        }
    }

    impl AsrEngine for FixedEngine {
        fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
            self.calls.lock().unwrap().push(samples.len());
            Ok(AsrTranscript::from_text("仮説"))
        }
    }

    fn pcm16_wav(samples: &[i16]) -> Vec<u8> {
        let data_len = u32::try_from(samples.len() * 2).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&16_000_u32.to_le_bytes());
        bytes.extend_from_slice(&32_000_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn jsonl_records_keep_success_and_failure_as_a_stable_discriminated_contract() {
        let success = EvalRecordV1::completed("sample-1", "reference", "hypothesis", 16_000, 12.5);
        let failure = EvalRecordV1::failed("sample-2", "audio_preflight", "hash mismatch");

        assert_eq!(
            serde_json::to_value(success).unwrap(),
            json!({
                "schema_version": 1,
                "status": "completed",
                "utterance_id": "sample-1",
                "reference": "reference",
                "hypothesis": "hypothesis",
                "duration_samples": 16000,
                "inference_elapsed_ms": 12.5
            })
        );
        assert_eq!(
            serde_json::to_value(failure).unwrap(),
            json!({
                "schema_version": 1,
                "status": "failed",
                "utterance_id": "sample-2",
                "stage": "audio_preflight",
                "message": "hash mismatch"
            })
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn manifest_audio_preflight_and_offline_service_emit_one_terminal_record_per_sample() {
        let temp = std::env::temp_dir().join(format!(
            "parapper-asr-eval-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("wav")).unwrap();
        let valid_bytes = pcm16_wav(&[0, 1, -1]);
        fs::write(temp.join("wav/valid.wav"), &valid_bytes).unwrap();
        fs::write(temp.join("wav/bad-hash.wav"), pcm16_wav(&[2])).unwrap();
        let valid_sha = format!("{:x}", Sha256::digest(&valid_bytes));
        let manifest = RunnerManifestV1 {
            schema_version: 1,
            split_id: "test-split".to_owned(),
            dataset: DatasetIdentity {
                id: "test".to_owned(),
                release: "1".to_owned(),
                source_split: "dev".to_owned(),
                language: "ja".to_owned(),
            },
            normalization: NormalizationIdentity {
                id: "test-normalization".to_owned(),
                version: "1".to_owned(),
            },
            audio_format: AudioFormatContract {
                encoding: "pcm_s16le".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![
                RunnerSampleV1 {
                    utterance_id: "valid".to_owned(),
                    audio: DerivedAudio {
                        relative_path: "wav/valid.wav".to_owned(),
                        sha256: valid_sha,
                        duration_samples: 3,
                    },
                    reference: ReferenceText {
                        raw: "参照。".to_owned(),
                        normalized: "参照".to_owned(),
                    },
                },
                RunnerSampleV1 {
                    utterance_id: "bad-hash".to_owned(),
                    audio: DerivedAudio {
                        relative_path: "wav/bad-hash.wav".to_owned(),
                        sha256: "a".repeat(64),
                        duration_samples: 1,
                    },
                    reference: ReferenceText {
                        raw: "失敗".to_owned(),
                        normalized: "失敗".to_owned(),
                    },
                },
            ],
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut models = AsrModelRegistry::default();
        models
            .insert(
                AsrModel::ReazonSpeechK2V2,
                Box::new(FixedEngine {
                    calls: Arc::clone(&calls),
                }),
            )
            .unwrap();
        let mut service = OfflineTranscriptionService::new(models);
        let mut output = Vec::new();

        let summary = run_manifest(
            &temp.join("manifest.json"),
            &manifest,
            AsrModel::ReazonSpeechK2V2,
            &mut service,
            &mut output,
        )
        .unwrap();

        assert_eq!(
            summary,
            EvalRunSummary {
                total: 2,
                completed: 1,
                failed: 1,
            }
        );
        assert_eq!(calls.lock().unwrap().as_slice(), &[10_243]);
        let records = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<EvalRecordV1>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert!(matches!(
            &records[0],
            EvalRecordV1::Completed {
                utterance_id,
                reference,
                hypothesis,
                duration_samples: 3,
                ..
            } if utterance_id == "valid" && reference == "参照" && hypothesis == "仮説"
        ));
        assert!(matches!(
            &records[1],
            EvalRecordV1::Failed {
                utterance_id,
                stage,
                message,
                ..
            } if utterance_id == "bad-hash"
                && stage == "audio_preflight"
                && message.contains("SHA-256 mismatch")
        ));

        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn accuracy_sweep_encodes_each_audio_once_and_decodes_all_widths_from_the_same_tensor() {
        let temp = std::env::temp_dir().join(format!(
            "parapper-reazon-sweep-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("wav")).unwrap();
        let wav_bytes = pcm16_wav(&[0, 1, -1]);
        fs::write(temp.join("wav/valid.wav"), &wav_bytes).unwrap();
        let manifest = RunnerManifestV1 {
            schema_version: 1,
            split_id: "test-split".to_owned(),
            dataset: DatasetIdentity {
                id: "test".to_owned(),
                release: "1".to_owned(),
                source_split: "dev".to_owned(),
                language: "ja".to_owned(),
            },
            normalization: NormalizationIdentity {
                id: "identity".to_owned(),
                version: "1".to_owned(),
            },
            audio_format: AudioFormatContract {
                encoding: "pcm_s16le".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![RunnerSampleV1 {
                utterance_id: "valid".to_owned(),
                audio: DerivedAudio {
                    relative_path: "wav/valid.wav".to_owned(),
                    sha256: format!("{:x}", Sha256::digest(&wav_bytes)),
                    duration_samples: 3,
                },
                reference: ReferenceText {
                    raw: "参照".to_owned(),
                    normalized: "参照".to_owned(),
                },
            }],
        };
        let strategies = [
            ReazonDecodingStrategy::Greedy,
            ReazonDecodingStrategy::ModifiedBeam { beam_size: 2 },
            ReazonDecodingStrategy::ModifiedBeam { beam_size: 4 },
            ReazonDecodingStrategy::ModifiedBeam { beam_size: 8 },
        ];
        let mut transcriber = RecordingCachedReazon { events: Vec::new() };
        let mut greedy = Vec::new();
        let mut beam2 = Vec::new();
        let mut beam4 = Vec::new();
        let mut beam8 = Vec::new();
        let mut outputs: [&mut dyn Write; 4] = [&mut greedy, &mut beam2, &mut beam4, &mut beam8];

        let summaries = run_reazon_accuracy_sweep(
            &temp.join("manifest.json"),
            &manifest,
            &mut transcriber,
            &strategies,
            &mut outputs,
        )
        .unwrap();

        assert_eq!(
            transcriber.events,
            vec![
                "encode:10243",
                "decode:greedy:10243",
                "decode:state-2:10243",
                "decode:state-4:10243",
                "decode:state-8:10243",
            ]
        );
        assert_eq!(
            summaries,
            vec![
                EvalRunSummary {
                    total: 1,
                    completed: 1,
                    failed: 0,
                };
                4
            ]
        );
        for (bytes, hypothesis) in [
            (greedy, "greedy"),
            (beam2, "state-2"),
            (beam4, "state-4"),
            (beam8, "state-8"),
        ] {
            assert_eq!(
                String::from_utf8(bytes).unwrap(),
                format!(
                    "{{\"status\":\"completed\",\"schema_version\":1,\"utterance_id\":\"valid\",\"reference\":\"参照\",\"hypothesis\":\"{hypothesis}\",\"duration_samples\":3,\"inference_elapsed_ms\":0.0}}\n"
                )
            );
        }

        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn parakeet_hybrid_sweep_encodes_once_then_decodes_every_requested_branch() {
        let temp = std::env::temp_dir().join(format!(
            "parapper-parakeet-hybrid-sweep-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("wav")).unwrap();
        let wav_bytes = pcm16_wav(&[0, 1, -1]);
        fs::write(temp.join("wav/valid.wav"), &wav_bytes).unwrap();
        let manifest = RunnerManifestV1 {
            schema_version: 1,
            split_id: "test-split".to_owned(),
            dataset: DatasetIdentity {
                id: "test".to_owned(),
                release: "1".to_owned(),
                source_split: "train".to_owned(),
                language: "ja".to_owned(),
            },
            normalization: NormalizationIdentity {
                id: "identity".to_owned(),
                version: "1".to_owned(),
            },
            audio_format: AudioFormatContract {
                encoding: "pcm_s16le".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![RunnerSampleV1 {
                utterance_id: "valid".to_owned(),
                audio: DerivedAudio {
                    relative_path: "wav/valid.wav".to_owned(),
                    sha256: format!("{:x}", Sha256::digest(&wav_bytes)),
                    duration_samples: 3,
                },
                reference: ReferenceText {
                    raw: "参照".to_owned(),
                    normalized: "参照".to_owned(),
                },
            }],
        };
        let strategies = [
            ParakeetJaDecodingStrategy::CtcGreedy,
            ParakeetJaDecodingStrategy::TdtGreedy,
            ParakeetJaDecodingStrategy::TdtVariableDag(FusedTdtDagConfig {
                beam_size: 2,
                max_symbols_per_step: 10,
                duration_expansion: FusedTdtDurationExpansion::All,
            }),
        ];
        let mut transcriber = RecordingCachedParakeetJa { events: Vec::new() };
        let mut ctc = Vec::new();
        let mut tdt = Vec::new();
        let mut dag2 = Vec::new();
        let mut outputs: [&mut dyn Write; 3] = [&mut ctc, &mut tdt, &mut dag2];

        let summaries = run_parakeet_ja_accuracy_sweep(
            &temp.join("manifest.json"),
            &manifest,
            &mut transcriber,
            &strategies,
            &mut outputs,
        )
        .unwrap();

        assert_eq!(
            transcriber.events,
            vec![
                "encode:3",
                "decode:ctc:3",
                "decode:tdt:3",
                "decode:tdt-dag-2:3"
            ]
        );
        assert_eq!(
            summaries,
            vec![
                EvalRunSummary {
                    total: 1,
                    completed: 1,
                    failed: 0,
                };
                3
            ]
        );
        for (bytes, hypothesis) in [(ctc, "ctc"), (tdt, "tdt"), (dag2, "tdt-dag-2")] {
            assert_eq!(
                String::from_utf8(bytes).unwrap(),
                format!(
                    "{{\"status\":\"completed\",\"schema_version\":1,\"utterance_id\":\"valid\",\"reference\":\"参照\",\"hypothesis\":\"{hypothesis}\",\"duration_samples\":3,\"inference_elapsed_ms\":0.0}}\n"
                )
            );
        }

        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn nbest_sweep_reuses_one_encoder_output_and_preserves_every_candidate_score_and_history() {
        let temp = std::env::temp_dir().join(format!(
            "parapper-reazon-nbest-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(temp.join("wav")).unwrap();
        let wav_bytes = pcm16_wav(&[0, 1, -1]);
        fs::write(temp.join("wav/valid.wav"), &wav_bytes).unwrap();
        let manifest = RunnerManifestV1 {
            schema_version: 1,
            split_id: "test-split".to_owned(),
            dataset: DatasetIdentity {
                id: "test".to_owned(),
                release: "1".to_owned(),
                source_split: "dev".to_owned(),
                language: "ja".to_owned(),
            },
            normalization: NormalizationIdentity {
                id: "identity".to_owned(),
                version: "1".to_owned(),
            },
            audio_format: AudioFormatContract {
                encoding: "pcm_s16le".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![RunnerSampleV1 {
                utterance_id: "valid".to_owned(),
                audio: DerivedAudio {
                    relative_path: "wav/valid.wav".to_owned(),
                    sha256: format!("{:x}", Sha256::digest(&wav_bytes)),
                    duration_samples: 3,
                },
                reference: ReferenceText {
                    raw: "参照".to_owned(),
                    normalized: "参照".to_owned(),
                },
            }],
        };
        let conditions = [
            ReazonNBestCondition {
                beam_size: 4,
                search_normalization: ReazonNBestSearchNormalization::Raw,
            },
            ReazonNBestCondition {
                beam_size: 8,
                search_normalization: ReazonNBestSearchNormalization::PerToken,
            },
        ];
        let mut transcriber = RecordingNBestReazon { events: Vec::new() };
        let mut raw = Vec::new();
        let mut normalized = Vec::new();
        let mut outputs: [&mut dyn Write; 2] = [&mut raw, &mut normalized];

        let summaries = run_reazon_nbest_sweep(
            &temp.join("manifest.json"),
            &manifest,
            &mut transcriber,
            &conditions,
            &mut outputs,
        )
        .unwrap();

        assert_eq!(
            transcriber.events,
            vec![
                "encode:10243",
                "decode:4:Raw:10243",
                "decode:8:PerToken:10243",
            ]
        );
        assert_eq!(
            summaries,
            vec![
                EvalRunSummary {
                    total: 1,
                    completed: 1,
                    failed: 0,
                };
                2
            ]
        );
        let raw_record = serde_json::from_slice::<ReazonNBestRecordV1>(&raw).unwrap();
        assert_eq!(
            raw_record,
            ReazonNBestRecordV1::Completed {
                schema_version: 1,
                utterance_id: "valid".to_owned(),
                reference: "参照".to_owned(),
                duration_samples: 3,
                beam_size: 4,
                search_normalization: ReazonNBestSearchNormalization::Raw,
                candidates: vec![
                    super::ReazonNBestCandidateV1 {
                        rank: 1,
                        hypothesis: "第一候補".to_owned(),
                        raw_score: -1.0,
                        token_ids: vec![1, 2],
                        timestamps: vec![0.0, 0.04],
                    },
                    super::ReazonNBestCandidateV1 {
                        rank: 2,
                        hypothesis: "別候補".to_owned(),
                        raw_score: -1.1,
                        token_ids: vec![3],
                        timestamps: vec![0.08],
                    },
                ],
            }
        );

        let mut rescored = Vec::new();
        let rescore_summary = run_reazon_nbest_rescore(
            &temp.join("manifest.json"),
            &manifest,
            &mut transcriber,
            ReazonNBestCondition {
                beam_size: 8,
                search_normalization: ReazonNBestSearchNormalization::Raw,
            },
            &mut rescored,
        )
        .unwrap();
        assert_eq!(
            transcriber.events,
            vec![
                "encode:10243",
                "decode:4:Raw:10243",
                "decode:8:PerToken:10243",
                "encode:10243",
                "rescore:8:Raw:10243",
            ]
        );
        assert_eq!(
            rescore_summary,
            EvalRunSummary {
                total: 1,
                completed: 1,
                failed: 0,
            }
        );
        let rescored_record =
            serde_json::from_slice::<ReazonRescoredNBestRecordV1>(&rescored).unwrap();
        let ReazonRescoredNBestRecordV1::Completed {
            schema_version,
            utterance_id,
            reference,
            duration_samples,
            beam_size,
            search_normalization,
            search_and_rescore_elapsed_ms,
            candidates,
        } = rescored_record
        else {
            panic!("expected completed rescore record");
        };
        assert_eq!(schema_version, 1);
        assert_eq!(utterance_id, "valid");
        assert_eq!(reference, "参照");
        assert_eq!(duration_samples, 3);
        assert_eq!(beam_size, 8);
        assert_eq!(search_normalization, ReazonNBestSearchNormalization::Raw);
        assert!(search_and_rescore_elapsed_ms >= 0.0);
        assert_eq!(
            candidates,
            vec![
                ReazonRescoredNBestCandidateV1 {
                    rank: 1,
                    hypothesis: "第一候補".to_owned(),
                    search_raw_score: -1.0,
                    viterbi_score: -1.5,
                    forward_score: -0.9,
                    token_ids: vec![1, 2],
                    timestamps: vec![0.0, 0.04],
                },
                ReazonRescoredNBestCandidateV1 {
                    rank: 2,
                    hypothesis: "別候補".to_owned(),
                    search_raw_score: -1.1,
                    viterbi_score: -1.4,
                    forward_score: -0.8,
                    token_ids: vec![3],
                    timestamps: vec![0.08],
                },
            ]
        );

        let mut aligned = Vec::new();
        let alignment_summary = run_reazon_monotonic_alignment(
            &temp.join("manifest.json"),
            &manifest,
            &mut transcriber,
            ReazonNBestCondition {
                beam_size: 8,
                search_normalization: ReazonNBestSearchNormalization::Raw,
            },
            &mut aligned,
        )
        .unwrap();
        assert_eq!(
            alignment_summary,
            EvalRunSummary {
                total: 1,
                completed: 1,
                failed: 0,
            }
        );
        let aligned_record =
            serde_json::from_slice::<ReazonAlignedNBestRecordV1>(&aligned).unwrap();
        let ReazonAlignedNBestRecordV1::Completed {
            schema_version,
            utterance_id,
            reference,
            duration_samples,
            beam_size,
            search_normalization,
            search_and_alignment_elapsed_ms,
            candidates,
        } = aligned_record
        else {
            panic!("expected completed alignment record");
        };
        assert_eq!(schema_version, 1);
        assert_eq!(utterance_id, "valid");
        assert_eq!(reference, "参照");
        assert_eq!(duration_samples, 3);
        assert_eq!(beam_size, 8);
        assert_eq!(search_normalization, ReazonNBestSearchNormalization::Raw);
        assert!(search_and_alignment_elapsed_ms >= 0.0);
        assert_eq!(
            candidates,
            vec![ReazonAlignedNBestCandidateV1 {
                rank: 1,
                hypothesis: "整列候補".to_owned(),
                search_raw_score: -1.0,
                viterbi_score: -1.2,
                forward_score: -0.8,
                token_ids: vec![1],
                search_timestamps: vec![0.04],
                token_alignments: vec![ReazonTokenAlignmentV1 {
                    viterbi_timestamp: 0.08,
                    posterior_mode_timestamp: 0.08,
                    expected_timestamp: 0.09,
                    posterior_lower_timestamp: 0.04,
                    posterior_upper_timestamp: 0.16,
                    entropy: 0.5,
                }],
                frame_emission_probabilities: vec![0.1, 0.7, 0.2],
            }]
        );
        assert_eq!(
            transcriber.events.last().map(String::as_str),
            Some("align:8:Raw:10243")
        );

        let mut lattice = Vec::new();
        let lattice_summary = run_reazon_approximate_lattice(
            &temp.join("manifest.json"),
            &manifest,
            &mut transcriber,
            ReazonNBestCondition {
                beam_size: 8,
                search_normalization: ReazonNBestSearchNormalization::Raw,
            },
            ReazonLatticeArcMerge::Maximum,
            &mut lattice,
        )
        .unwrap();
        assert_eq!(
            lattice_summary,
            EvalRunSummary {
                total: 1,
                completed: 1,
                failed: 0,
            }
        );
        assert_eq!(
            transcriber.events.last().map(String::as_str),
            Some("lattice:8:Raw:Maximum:10243")
        );
        let lattice_record =
            serde_json::from_slice::<ReazonApproximateLatticeRecordV1>(&lattice).unwrap();
        let ReazonApproximateLatticeRecordV1::Completed {
            schema_version,
            utterance_id,
            reference,
            duration_samples,
            beam_size,
            search_normalization,
            arc_merge,
            search_and_lattice_elapsed_ms,
            seeds,
            candidates,
        } = lattice_record
        else {
            panic!("expected completed lattice record");
        };
        assert_eq!(
            (
                schema_version,
                utterance_id,
                reference,
                duration_samples,
                beam_size,
                search_normalization,
                arc_merge,
            ),
            (
                1,
                "valid".to_owned(),
                "参照".to_owned(),
                3,
                8,
                ReazonNBestSearchNormalization::Raw,
                ReazonLatticeArcMerge::Maximum,
            )
        );
        assert!(search_and_lattice_elapsed_ms >= 0.0);
        assert_eq!(seeds.len(), 1);
        assert_eq!(
            candidates,
            vec![ReazonApproximateLatticeCandidateV1 {
                rank: 1,
                hypothesis: "再結合候補".to_owned(),
                lattice_score: -0.7,
                token_ids: vec![1, 3],
                is_seed: false,
            }]
        );

        fs::remove_dir_all(temp).unwrap();
    }
}
