mod audio;
mod manifest;
mod runner;

pub use audio::{CanonicalWav, CanonicalWavError, decode_canonical_pcm16_wav};
pub use manifest::{
    AudioFormatContract, DatasetIdentity, DerivedAudio, ManifestValidationError,
    NormalizationIdentity, ReferenceText, RunnerManifestV1, RunnerSampleV1,
};
pub use parapper_models::asr::backend::parakeet_ja::{
    FusedTdtDagConfig, FusedTdtDagMergeAblation, FusedTdtDagMergeOrder, FusedTdtDagResult,
    FusedTdtDagStats, FusedTdtDurationExpansion, FusedTdtHotwordCandidatePolicy,
    FusedTdtHypothesis, FusedTdtJaOrtEngine, FusedTdtStep, HybridParakeetJaOrtEngine,
    OnnxAsrCtcJaOrtEngine, ParakeetJaCtcOutput, ParakeetJaDecodingStrategy, ParakeetJaEncodedAudio,
    ParakeetJaTdtCandidate, ctc_gate_hotword_paths, ctc_local_keyword_score, greedy_fused_tdt,
    select_ctc_anchor_candidate, variable_width_dag_fused_tdt,
    variable_width_dag_fused_tdt_with_hotword_policy, variable_width_dag_fused_tdt_with_hotwords,
    variable_width_dag_fused_tdt_with_merge_ablation,
};
pub use runner::{
    CachedParakeetJaTranscriber, CachedReazonNBestTranscriber, CachedReazonTranscriber,
    EvalRecordV1, EvalRunSummary, EvalTranscriber, ReazonAlignedNBestCandidateV1,
    ReazonAlignedNBestRecordV1, ReazonApproximateLatticeCandidateV1,
    ReazonApproximateLatticeRecordV1, ReazonLatticeArcMerge, ReazonNBestCandidateV1,
    ReazonNBestCondition, ReazonNBestRecordV1, ReazonNBestSearchNormalization,
    ReazonRescoredNBestCandidateV1, ReazonRescoredNBestRecordV1, ReazonTokenAlignmentV1,
    run_manifest, run_parakeet_ja_accuracy_sweep, run_reazon_accuracy_sweep,
    run_reazon_approximate_lattice, run_reazon_monotonic_alignment, run_reazon_nbest_rescore,
    run_reazon_nbest_sweep,
};
