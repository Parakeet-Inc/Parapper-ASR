#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Complete,
    Continue { emit_interim: bool },
}

#[must_use]
pub const fn action(is_final: bool) -> Action {
    if is_final {
        Action::Complete
    } else {
        Action::Continue { emit_interim: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namo_continue_keeps_the_turn_open_and_requests_interim_output() {
        assert_eq!(action(false), Action::Continue { emit_interim: true });
        assert_eq!(action(true), Action::Complete);
    }
}
