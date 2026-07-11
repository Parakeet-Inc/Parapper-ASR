use std::borrow::Cow;

use crate::{
    config::ParapperConfig,
    recognition::{
        segmentation::segment::builder::SegmentCloseReason,
        transcription::{
            asr::task::AsrTaskKind,
            route::{
                RecognitionRoute, RecognitionRouteSelection,
                language_id::{
                    LanguageDetectionWarningSink, LanguageDetector, SliContext,
                    detect_recognition_route,
                },
            },
        },
    },
};

#[derive(Clone, Copy)]
pub(crate) struct AsrInput<'a> {
    pub(crate) config: &'a ParapperConfig,
    pub(crate) warning_sink: Option<&'a dyn LanguageDetectionWarningSink>,
    pub(crate) kind: AsrTaskKind,
    pub(crate) close_reason: SegmentCloseReason,
    pub(crate) current_route: Option<RecognitionRoute>,
    pub(crate) fallback_route: RecognitionRoute,
    pub(crate) draft_audio: Option<&'a [f32]>,
    pub(crate) request_audio: &'a [f32],
}

pub(crate) fn select_asr<'a>(
    input: AsrInput<'a>,
    language_id: Option<&'a mut (dyn LanguageDetector + 'a)>,
) -> RecognitionRouteSelection {
    let split_route = configured_split_route(input.config, input.kind);
    let current_route = usable_current_route(input.current_route, input.kind);
    let default_selection = || RecognitionRouteSelection {
        route: split_route
            .or(current_route)
            .unwrap_or(input.fallback_route),
        detected_language: None,
    };
    if split_route.is_some() {
        return default_selection();
    }
    if !should_run_sli(input.kind, input.close_reason) {
        return default_selection();
    }

    if input.warning_sink.is_none() {
        return default_selection();
    }
    let detection_audio = full_audio(input.draft_audio, input.request_audio);
    detect_recognition_route(
        &mut SliContext {
            config: input.config,
            warning_sink: input.warning_sink,
            language_id,
        },
        current_route,
        detection_audio.as_ref(),
    )
}

pub(crate) fn configured_split_route(
    config: &ParapperConfig,
    kind: AsrTaskKind,
) -> Option<RecognitionRoute> {
    match kind {
        AsrTaskKind::InterimDisplay => config.asr.interim_model.map(RecognitionRoute::from_model),
        AsrTaskKind::CompletionCheck | AsrTaskKind::Rerecognition => None,
    }
}

fn should_run_sli(kind: AsrTaskKind, close_reason: SegmentCloseReason) -> bool {
    kind == AsrTaskKind::CompletionCheck && close_reason == SegmentCloseReason::EndSilenceReached
}

fn usable_current_route(
    current_route: Option<RecognitionRoute>,
    kind: AsrTaskKind,
) -> Option<RecognitionRoute> {
    let current_route = current_route?;
    if kind != AsrTaskKind::InterimDisplay && current_route.model.is_interim_only() {
        return None;
    }
    Some(current_route)
}

fn full_audio<'a>(draft_audio: Option<&[f32]>, request_audio: &'a [f32]) -> Cow<'a, [f32]> {
    let Some(draft_audio) = draft_audio else {
        return Cow::Borrowed(request_audio);
    };
    if draft_audio.is_empty() {
        return Cow::Borrowed(request_audio);
    }

    let mut full_audio = Vec::with_capacity(draft_audio.len() + request_audio.len());
    full_audio.extend_from_slice(draft_audio);
    full_audio.extend_from_slice(request_audio);
    Cow::Owned(full_audio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AsrModel;

    #[test]
    fn sli_gate_allows_only_end_silence_completion_checks() {
        assert!(
            should_run_sli(
                AsrTaskKind::CompletionCheck,
                SegmentCloseReason::EndSilenceReached
            ),
            "SLI should run for completion checks caused by turn-check silence"
        );
        assert!(
            !should_run_sli(
                AsrTaskKind::InterimDisplay,
                SegmentCloseReason::InterimResultSilenceReached
            ),
            "SLI must not run for interim display ASR"
        );
        assert!(
            !should_run_sli(
                AsrTaskKind::CompletionCheck,
                SegmentCloseReason::SegmentMaxChunksReached
            ),
            "non-silence completion must reuse the current route without SLI"
        );
    }

    #[test]
    fn configured_split_route_uses_nemotron_only_for_interim_display() {
        let config = parapper_config! {
            asr_model: AsrModel::ReazonSpeechK2V2,
            interim_asr_model: Some(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8),
            ..ParapperConfig::default()
        };

        assert_eq!(
            configured_split_route(&config, AsrTaskKind::InterimDisplay),
            Some(RecognitionRoute::from_model(
                AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
            ))
        );
        assert_eq!(
            configured_split_route(&config, AsrTaskKind::CompletionCheck),
            None,
            "completion ASR keeps the normal route selection path"
        );
        assert_eq!(
            configured_split_route(&config, AsrTaskKind::Rerecognition),
            None,
            "rerecognition keeps the normal route selection path"
        );
    }
}
