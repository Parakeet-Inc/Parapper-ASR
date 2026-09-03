use super::{RerecognitionPurpose, TurnDetector};

#[must_use]
pub const fn rerecognition_purpose(
    turn_detector: TurnDetector,
    rerecognize_full_on_complete: bool,
) -> Option<RerecognitionPurpose> {
    match turn_detector {
        TurnDetector::Namo | TurnDetector::Morph => {
            return Some(RerecognitionPurpose::GrammarAfterCompletion);
        }
        TurnDetector::Simple => {}
    }
    if rerecognize_full_on_complete {
        Some(RerecognitionPurpose::SimpleTurnCheckFinal)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_mode_controls_completion_rerecognition_without_application_config() {
        assert_eq!(
            rerecognition_purpose(TurnDetector::Namo, false),
            Some(RerecognitionPurpose::GrammarAfterCompletion)
        );
        assert_eq!(
            rerecognition_purpose(TurnDetector::Morph, false),
            Some(RerecognitionPurpose::GrammarAfterCompletion)
        );
        assert_eq!(
            rerecognition_purpose(TurnDetector::Simple, true),
            Some(RerecognitionPurpose::SimpleTurnCheckFinal)
        );
        assert_eq!(rerecognition_purpose(TurnDetector::Simple, false), None);
    }
}
