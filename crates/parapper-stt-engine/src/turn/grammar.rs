use super::GrammarBoundaryClass;

pub struct Candidate {
    pub class: GrammarBoundaryClass,
    pub is_at_text_end: bool,
    pub normal_end_is_confirmed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    CompleteTurn,
    ContinueOpen { emit_interim: bool },
    DecideWithNamo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoCandidateAction {
    DecideWithNamo,
    ContinueOpen,
}

#[must_use]
pub fn action_after_rerecognition(
    candidates: Vec<Candidate>,
    no_candidate_action: NoCandidateAction,
) -> Action {
    if candidates.is_empty() {
        return match no_candidate_action {
            NoCandidateAction::DecideWithNamo => Action::DecideWithNamo,
            NoCandidateAction::ContinueOpen => Action::ContinueOpen { emit_interim: true },
        };
    }

    let Some(evaluated) = candidates
        .into_iter()
        .rev()
        .find(|candidate| candidate.is_at_text_end)
    else {
        return Action::ContinueOpen { emit_interim: true };
    };

    if candidate_is_confirmed(&evaluated) {
        return Action::CompleteTurn;
    }

    Action::ContinueOpen { emit_interim: true }
}

fn candidate_is_confirmed(candidate: &Candidate) -> bool {
    match candidate.class {
        GrammarBoundaryClass::StrongEnd | GrammarBoundaryClass::PredicateEnd => true,
        GrammarBoundaryClass::NormalEnd => candidate.normal_end_is_confirmed,
        GrammarBoundaryClass::Reject | GrammarBoundaryClass::ClauseWeak => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_confirmed_boundary_completes_the_turn() {
        assert_eq!(
            action_after_rerecognition(
                vec![candidate(GrammarBoundaryClass::StrongEnd)],
                NoCandidateAction::DecideWithNamo,
            ),
            Action::CompleteTurn
        );

        assert_eq!(
            action_after_rerecognition(
                vec![
                    nonterminal_candidate(GrammarBoundaryClass::StrongEnd),
                    nonterminal_candidate(GrammarBoundaryClass::ClauseWeak),
                    candidate(GrammarBoundaryClass::PredicateEnd),
                ],
                NoCandidateAction::DecideWithNamo,
            ),
            Action::CompleteTurn,
            "only the candidate at the completion ASR text end should finalize the turn"
        );
    }

    #[test]
    fn internal_or_unconfirmed_boundary_keeps_the_turn_open() {
        for candidates in [
            vec![nonterminal_candidate(GrammarBoundaryClass::PredicateEnd)],
            vec![candidate(GrammarBoundaryClass::ClauseWeak)],
            vec![candidate(GrammarBoundaryClass::Reject)],
        ] {
            assert_eq!(
                action_after_rerecognition(candidates, NoCandidateAction::DecideWithNamo),
                Action::ContinueOpen { emit_interim: true }
            );
        }
    }

    #[test]
    fn no_candidate_uses_the_configured_fallback_policy() {
        assert_eq!(
            action_after_rerecognition(Vec::new(), NoCandidateAction::DecideWithNamo),
            Action::DecideWithNamo
        );
        assert_eq!(
            action_after_rerecognition(Vec::new(), NoCandidateAction::ContinueOpen),
            Action::ContinueOpen { emit_interim: true }
        );
    }

    fn candidate(class: GrammarBoundaryClass) -> Candidate {
        Candidate {
            class,
            is_at_text_end: true,
            normal_end_is_confirmed: false,
        }
    }

    fn nonterminal_candidate(class: GrammarBoundaryClass) -> Candidate {
        Candidate {
            is_at_text_end: false,
            ..candidate(class)
        }
    }
}
