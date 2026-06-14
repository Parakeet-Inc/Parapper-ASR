use crate::{
    config::ParapperConfig,
    recognition::transcription::route::{
        RecognitionRoute, RecognitionRouteSelection,
        language_id::{
            LanguageDetectionWarningSink, LanguageDetector, SliContext, detect_recognition_route,
        },
    },
};

#[derive(Clone, Copy)]
pub(crate) struct TurnInput<'a> {
    pub(crate) config: &'a ParapperConfig,
    pub(crate) warning_sink: Option<&'a dyn LanguageDetectionWarningSink>,
    pub(crate) current_route: Option<RecognitionRoute>,
    pub(crate) full_audio: &'a [f32],
}

pub(crate) fn refresh_turn<'a>(
    input: TurnInput<'a>,
    language_id: Option<&'a mut (dyn LanguageDetector + 'a)>,
) -> Option<RecognitionRouteSelection> {
    if input.full_audio.is_empty() {
        return None;
    }
    input.warning_sink?;
    Some(detect_recognition_route(
        &mut SliContext {
            config: input.config,
            warning_sink: input.warning_sink,
            language_id,
        },
        input.current_route,
        input.full_audio,
    ))
}
