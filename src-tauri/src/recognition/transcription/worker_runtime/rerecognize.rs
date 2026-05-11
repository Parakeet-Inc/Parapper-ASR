use std::time::Instant;

use crate::recognition::route::RecognitionRoute;

use super::{AsrJobContext, emit_asr_warning, transcribe_asr};

pub(super) fn rerecognize_full_turn_if_needed(context: &mut AsrJobContext<'_>, turn_id: u64) {
    if !context.config.turn_rerecognize_full_on_complete {
        return;
    }

    let Some((route, full_audio)) = context.turns.get(&turn_id).and_then(|turn| {
        turn.draft()
            .route
            .map(|route| (route, turn.draft().full_audio.clone()))
    }) else {
        return;
    };
    let _ = rerecognize_full_turn_with_audio(context, turn_id, route, &full_audio);
}

pub(super) fn rerecognize_full_turn_with_route(
    context: &mut AsrJobContext<'_>,
    turn_id: u64,
    route: RecognitionRoute,
) -> bool {
    let Some(full_audio) = context
        .turns
        .get(&turn_id)
        .map(|turn| turn.draft().full_audio.clone())
    else {
        return false;
    };
    rerecognize_full_turn_with_audio(context, turn_id, route, &full_audio)
}

fn rerecognize_full_turn_with_audio(
    context: &mut AsrJobContext<'_>,
    turn_id: u64,
    route: RecognitionRoute,
    full_audio: &[f32],
) -> bool {
    if full_audio.is_empty() {
        return false;
    }

    let started_at = Instant::now();
    let text = match transcribe_asr(context, route, full_audio) {
        Ok(text) if !text.is_empty() => text,
        Ok(_) => return false,
        Err(err) => {
            emit_asr_warning(context.handle, &err);
            return false;
        }
    };
    let elapsed_millis = started_at.elapsed().as_millis();
    if let Some(turn) = context.turns.get_mut(&turn_id) {
        turn.draft_mut()
            .replace_with_full_turn_transcription(route, text, elapsed_millis);
        return true;
    }
    false
}
