// Category boundaries:
// - request_dispatch: ASR request construction/submission, target/range/audio, queue merge/drop, and promotion to a request.
// - route_selection: route choice, SLI gating, detected language, route refresh, and route cache/config lifecycle.
// - asr_result: completed ASR result matching, stale/failure/empty handling, transcript application, and result metadata propagation.
// - asr_runner: direct ASR runner behavior before RecognitionSession consumes the result.
// - finalization: direct final/keep-open/timeout output behavior and finalization ordering.
// - turn_split: internal boundary candidates that must keep a turn whole until a terminal candidate or timeout.
// - grammar_boundary_decision: grammar boundary class policy before it becomes a concrete continue/final decision.
#[path = "unit_test/asr_result.rs"]
mod asr_result;
#[path = "unit_test/asr_runner.rs"]
mod asr_runner;
#[path = "unit_test/finalization.rs"]
mod finalization;
#[path = "unit_test/grammar_boundary_decision.rs"]
mod grammar_boundary_decision;
#[path = "unit_test/request_dispatch.rs"]
mod request_dispatch;
#[path = "unit_test/route_selection.rs"]
mod route_selection;
#[path = "unit_test/turn_split.rs"]
mod turn_split;
