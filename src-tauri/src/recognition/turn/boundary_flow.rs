use crate::recognition::{
    control::RecognitionSession,
    turn::{GrammarBoundaryClass, TurnBoundaryCandidate, boundary::slice_chars, policy::grammar},
};

impl RecognitionSession {
    pub(in crate::recognition) fn process_grammar_boundaries_after_rerecognition(
        &mut self,
        turn_id: u64,
    ) {
        let Some(candidates) = self
            .turn_store
            .turns
            .get(&turn_id)
            .map(|turn| turn.draft().boundary_candidates.clone())
        else {
            return;
        };
        match self.grammar_boundary_action(turn_id, candidates) {
            grammar::Action::CompleteTurn => {
                self.complete_whole_turn_after_grammar_boundary(turn_id);
            }
            grammar::Action::ContinueOpen { emit_interim } => {
                self.keep_turn_open(turn_id, emit_interim);
            }
            grammar::Action::DecideWithNamo => {
                self.complete_or_continue_turn_with_namo(turn_id);
            }
        }
    }

    fn grammar_boundary_action(
        &mut self,
        turn_id: u64,
        candidates: Vec<TurnBoundaryCandidate>,
    ) -> grammar::Action {
        let text_len = self
            .turn_store
            .turns
            .get(&turn_id)
            .map(|turn| turn.draft().combined_text.chars().count())
            .unwrap_or_default();
        let no_candidate_action = if self.config.uses_namo_turn_detector() {
            grammar::NoCandidateAction::DecideWithNamo
        } else {
            grammar::NoCandidateAction::ContinueOpen
        };
        let confirm_normal_end_with_namo = self.config.confirms_normal_end_with_namo();
        let mut combined_text_for_namo = None;
        let mut evaluated = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let is_at_text_end = candidate.char_end >= text_len;
            let normal_end_is_confirmed =
                if is_at_text_end && matches!(candidate.class, GrammarBoundaryClass::NormalEnd) {
                    if confirm_normal_end_with_namo {
                        let combined_text = combined_text_for_namo.get_or_insert_with(|| {
                            self.turn_store
                                .turns
                                .get(&turn_id)
                                .map(|turn| turn.draft().combined_text.clone())
                                .unwrap_or_default()
                        });
                        let text = slice_chars(combined_text, 0..candidate.char_end);
                        self.namo_final_decision_for_text(turn_id, &text)
                    } else {
                        true
                    }
                } else {
                    false
                };
            evaluated.push(grammar::Candidate {
                class: candidate.class,
                is_at_text_end,
                normal_end_is_confirmed,
            });
        }
        grammar::action_after_rerecognition(evaluated, no_candidate_action)
    }

    fn complete_whole_turn_after_grammar_boundary(&mut self, turn_id: u64) {
        self.emit_stale_turn_finals(turn_id);
        self.emit_turn_output(turn_id, true);
        self.clear_open_turn();
    }
}
