use std::borrow::Cow;

use crate::recognition::{
    route::{RecognitionRoute, RecognitionRouteSelection},
    segment_builder::SegmentCloseReason,
    sli::{SliContext, detect_recognition_route, route_without_language_detection},
};

use super::{AsrJobContext, rerecognize::rerecognize_full_turn_with_route};

pub(super) fn select_segment_route(
    context: &mut AsrJobContext<'_>,
    turn_id: u64,
    current_route: Option<RecognitionRoute>,
    audio: &[f32],
    reason: SegmentCloseReason,
) -> RecognitionRouteSelection {
    if reason != SegmentCloseReason::EndSilenceReached {
        return RecognitionRouteSelection {
            route: route_without_language_detection(context.config, current_route),
            detected_language: None,
        };
    }

    let route_audio = full_turn_audio_with_segment(context, turn_id, audio);
    detect_recognition_route(
        &mut SliContext {
            handle: context.handle,
            config: context.config,
            language_id: context
                .language_id
                .as_deref_mut()
                .map(|engine| engine as &mut dyn crate::recognition::sli::LanguageDetector),
        },
        current_route,
        route_audio.as_ref(),
    )
}

fn full_turn_audio_with_segment<'a>(
    context: &AsrJobContext<'_>,
    turn_id: u64,
    segment_audio: &'a [f32],
) -> Cow<'a, [f32]> {
    let Some(turn) = context.turns.get(&turn_id) else {
        return Cow::Borrowed(segment_audio);
    };
    let draft_audio = &turn.draft().full_audio;
    if draft_audio.is_empty() {
        return Cow::Borrowed(segment_audio);
    }

    let mut full_audio = Vec::with_capacity(draft_audio.len() + segment_audio.len());
    full_audio.extend_from_slice(draft_audio);
    full_audio.extend_from_slice(segment_audio);
    Cow::Owned(full_audio)
}

pub(super) fn refresh_turn_route_before_turn_decision(
    context: &mut AsrJobContext<'_>,
    turn_id: u64,
) -> bool {
    let Some((current_route, full_audio)) = context.turns.get(&turn_id).map(|turn| {
        (
            turn.draft().route.or(*context.last_spoken_route),
            turn.draft().full_audio.clone(),
        )
    }) else {
        return false;
    };
    if full_audio.is_empty() {
        return false;
    }

    let selection = detect_recognition_route(
        &mut SliContext {
            handle: context.handle,
            config: context.config,
            language_id: context
                .language_id
                .as_deref_mut()
                .map(|engine| engine as &mut dyn crate::recognition::sli::LanguageDetector),
        },
        current_route,
        &full_audio,
    );
    if let Some(turn) = context.turns.get_mut(&turn_id) {
        turn.draft_mut()
            .set_detected_language(selection.detected_language);
    }

    if current_route == Some(selection.route) {
        return false;
    }

    rerecognize_full_turn_with_route(context, turn_id, selection.route)
}
