#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::recognition) enum Action {
    Complete,
    Continue { emit_interim: bool },
}

pub(in crate::recognition) fn action(is_final: bool) -> Action {
    if is_final {
        Action::Complete
    } else {
        Action::Continue { emit_interim: true }
    }
}
