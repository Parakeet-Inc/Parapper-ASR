// Engine-owned behavior tests. Host/model adapter tests stay in `src-tauri`.
#[path = "unit/asr_result.rs"]
mod asr_result;
#[path = "unit/finalization.rs"]
mod finalization;
#[path = "unit/grammar_boundary_decision.rs"]
mod grammar_boundary_decision;
#[path = "unit/request_dispatch.rs"]
mod request_dispatch;
#[path = "unit/route_selection.rs"]
mod route_selection;
#[path = "unit/turn_split.rs"]
mod turn_split;
