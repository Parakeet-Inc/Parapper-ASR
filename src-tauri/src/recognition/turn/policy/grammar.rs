use crate::recognition::turn::GrammarBoundaryClass;

pub(in crate::recognition) struct Candidate {
    pub(in crate::recognition) class: GrammarBoundaryClass,
    pub(in crate::recognition) is_at_text_end: bool,
    pub(in crate::recognition) normal_end_is_confirmed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::recognition) enum Action {
    CompleteTurn,
    ContinueOpen { emit_interim: bool },
    DecideWithNamo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::recognition) enum NoCandidateAction {
    DecideWithNamo,
    ContinueOpen,
}

pub(in crate::recognition) fn action_after_rerecognition(
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
    fn action_completes_terminal_confirmed_boundary() {
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
                    nonterminal_candidate_at(1, GrammarBoundaryClass::StrongEnd),
                    nonterminal_candidate_at(2, GrammarBoundaryClass::ClauseWeak),
                    candidate(GrammarBoundaryClass::PredicateEnd),
                ],
                NoCandidateAction::DecideWithNamo,
            ),
            Action::CompleteTurn,
            "only the candidate at the completion ASR text end should finalize the turn"
        );

        assert_eq!(
            action_after_rerecognition(
                vec![
                    candidate_at(1, GrammarBoundaryClass::StrongEnd),
                    candidate_at(2, GrammarBoundaryClass::StrongEnd),
                    candidate_at(3, GrammarBoundaryClass::ClauseWeak),
                    candidate(GrammarBoundaryClass::StrongEnd),
                ],
                NoCandidateAction::DecideWithNamo,
            ),
            Action::CompleteTurn,
            "when the final candidate is confirmed, the whole turn should be finalized together"
        );
    }

    #[test]
    fn action_keeps_open_when_confirmed_boundary_is_not_at_completion_text_end() {
        assert_eq!(
            action_after_rerecognition(
                vec![nonterminal_candidate_at(
                    3,
                    GrammarBoundaryClass::PredicateEnd
                )],
                NoCandidateAction::DecideWithNamo,
            ),
            Action::ContinueOpen { emit_interim: true },
            "an internal predicate end must keep the turn open when suffix text remains"
        );

        assert_eq!(
            action_after_rerecognition(
                vec![
                    nonterminal_candidate_at(3, GrammarBoundaryClass::PredicateEnd),
                    candidate_at(7, GrammarBoundaryClass::Reject),
                ],
                NoCandidateAction::DecideWithNamo,
            ),
            Action::ContinueOpen { emit_interim: true },
            "an internal predicate end must keep the turn open before a trailing connective candidate"
        );
    }

    #[test]
    fn action_continues_or_delegates_when_no_completable_boundary_is_confirmed() {
        assert_eq!(
            action_after_rerecognition(
                vec![candidate(GrammarBoundaryClass::ClauseWeak)],
                NoCandidateAction::DecideWithNamo,
            ),
            Action::ContinueOpen { emit_interim: true }
        );

        assert_eq!(
            action_after_rerecognition(Vec::new(), NoCandidateAction::DecideWithNamo),
            Action::DecideWithNamo
        );

        assert_eq!(
            action_after_rerecognition(Vec::new(), NoCandidateAction::ContinueOpen),
            Action::ContinueOpen { emit_interim: true },
            "Morph mode should keep the turn open when grammar has no boundary candidate"
        );
    }

    fn candidate(class: GrammarBoundaryClass) -> Candidate {
        Candidate {
            class,
            is_at_text_end: true,
            normal_end_is_confirmed: false,
        }
    }

    fn candidate_at(_char_end: usize, class: GrammarBoundaryClass) -> Candidate {
        Candidate {
            is_at_text_end: true,
            ..candidate(class)
        }
    }

    fn nonterminal_candidate_at(char_end: usize, class: GrammarBoundaryClass) -> Candidate {
        Candidate {
            is_at_text_end: false,
            ..candidate_at(char_end, class)
        }
    }
}
