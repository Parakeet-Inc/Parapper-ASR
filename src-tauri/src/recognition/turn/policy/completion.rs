use crate::{config::ParapperConfig, recognition::control::RerecognitionPurpose};

pub(in crate::recognition) fn rerecognition_purpose(
    config: &ParapperConfig,
) -> Option<RerecognitionPurpose> {
    match config.turn.detector {
        crate::config::TurnDetector::Namo | crate::config::TurnDetector::Morph => {
            return Some(RerecognitionPurpose::GrammarAfterCompletion);
        }
        crate::config::TurnDetector::Simple => {}
    }
    config
        .turn
        .rerecognize_full_on_complete
        .then_some(RerecognitionPurpose::SimpleTurnCheckFinal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurnDetector;

    #[test]
    fn rerecognition_purpose_follows_turn_detector_and_simple_config() {
        assert_eq!(
            rerecognition_purpose(&parapper_config! {
                turn_detector: TurnDetector::Namo,
                ..ParapperConfig::default()
            }),
            Some(RerecognitionPurpose::GrammarAfterCompletion)
        );
        assert_eq!(
            rerecognition_purpose(&parapper_config! {
                turn_detector: TurnDetector::Morph,
                ..ParapperConfig::default()
            }),
            Some(RerecognitionPurpose::GrammarAfterCompletion)
        );
        assert_eq!(
            rerecognition_purpose(&parapper_config! {
                turn_detector: TurnDetector::Simple,
                turn_rerecognize_full_on_complete: true,
                ..ParapperConfig::default()
            }),
            Some(RerecognitionPurpose::SimpleTurnCheckFinal)
        );
        assert_eq!(
            rerecognition_purpose(&parapper_config! {
                turn_detector: TurnDetector::Simple,
                turn_rerecognize_full_on_complete: false,
                ..ParapperConfig::default()
            }),
            None
        );
    }
}
