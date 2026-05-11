use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::Receiver,
    },
    time::{Duration, Instant},
};

use tauri::{AppHandle, Emitter};

use super::super::job::AsrJob;
use crate::{
    audio::ASR_SAMPLE_RATE,
    config::ParapperConfig,
    delivery::{RecognizedTextOutput, dispatch_recognized_text, spawn_mute_check_if_needed},
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    recognition::{
        engine_cache::{AsrEngineCache, NamoTurnDetectorCache, build_language_id_engine},
        engines::SpokenLanguageIdentificationEngine,
        events::AsrMissingEvent,
        route::RecognitionRoute,
        segment_builder::SegmentCloseReason,
        turn::{Turn, turn_event_id},
    },
};

use super::decision::namo_final_decision;
use super::decision::{
    PostSegmentAction, TurnProgressDecision, post_segment_action,
    refresh_open_turn_timeout_origin_if_activity_changed,
};
use super::output::{emit_stale_turn_finals, emit_turn_output};
use super::rerecognize::{rerecognize_full_turn_if_needed, rerecognize_full_turn_with_route};
use super::route::{refresh_turn_route_before_turn_decision, select_segment_route};

static NEXT_TURN_SESSION_ID: AtomicU64 = AtomicU64::new(1);
pub(crate) const MIN_LANGUAGE_ID_SAMPLES: usize = ASR_SAMPLE_RATE as usize;
const NORMALIZED_ASR_INPUT_PEAK: f32 = 0.95;

pub(super) struct AsrJobContext<'a> {
    pub(super) handle: &'a AppHandle,
    pub(super) config: &'a ParapperConfig,
    pub(super) asr: &'a mut AsrEngineCache,
    pub(super) language_id: Option<&'a mut SpokenLanguageIdentificationEngine>,
    pub(super) turn_detectors: &'a mut NamoTurnDetectorCache,
    pub(super) turns: &'a mut HashMap<u64, Turn>,
    pub(super) turn_revisions: &'a mut HashMap<u64, u64>,
    pub(super) open_turn_id: &'a mut Option<u64>,
    pub(super) open_turn_since: &'a mut Option<Instant>,
    pub(super) open_turn_activity_epoch: &'a mut u64,
    pub(super) last_spoken_route: &'a mut Option<RecognitionRoute>,
    pub(super) segment_activity_epoch: &'a AtomicU64,
    pub(super) turn_session_id: u64,
    pub(super) next_output_sequence: &'a mut u64,
}

impl AsrJobContext<'_> {
    pub(super) fn emit(&mut self, output: &RecognizedTextOutput) {
        let mute_check = spawn_mute_check_if_needed(self.handle, self.config);
        dispatch_recognized_text(self.handle, self.config, mute_check, output);
    }
}

pub(in crate::recognition::transcription) fn run_asr_worker(
    handle: &AppHandle,
    config: &ParapperConfig,
    runtime_config: &Arc<RwLock<ParapperConfig>>,
    receiver: &Receiver<AsrJob>,
    segment_activity_epoch: &AtomicU64,
    stop_requested: &AtomicBool,
) {
    let mut asr = AsrEngineCache::default();
    for reason in asr.preload_required(handle, config) {
        log::warn!("{reason}");
        emit_asr_missing_event(handle, reason);
    }
    let mut language_id = match build_language_id_engine(handle, config) {
        Ok(language_id) => language_id,
        Err(err) => {
            let reason = format!("Failed to initialize language identification: {err}");
            log::warn!("{reason}");
            emit_asr_missing_event(handle, reason);
            None
        }
    };
    let mut turn_detectors = NamoTurnDetectorCache::default();
    for reason in turn_detectors.preload_required(handle, config) {
        log::warn!("{reason}");
        emit_asr_missing_event(handle, reason);
    }

    let mut turns = HashMap::<u64, Turn>::new();
    let mut turn_revisions = HashMap::<u64, u64>::new();
    let mut open_turn_id = None;
    let mut open_turn_since = None;
    let mut open_turn_activity_epoch = 0;
    let mut last_spoken_route = None;
    let turn_session_id = NEXT_TURN_SESSION_ID.fetch_add(1, Ordering::Relaxed);
    let mut next_output_sequence = 1;

    while !stop_requested.load(Ordering::Acquire) {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(job) => {
                let current_config = runtime_config
                    .read()
                    .map_or_else(|_| config.clone(), |config| config.clone());
                handle_asr_job(
                    AsrJobContext {
                        handle,
                        config: &current_config,
                        asr: &mut asr,
                        language_id: language_id.as_mut(),
                        turn_detectors: &mut turn_detectors,
                        turns: &mut turns,
                        turn_revisions: &mut turn_revisions,
                        open_turn_id: &mut open_turn_id,
                        open_turn_since: &mut open_turn_since,
                        open_turn_activity_epoch: &mut open_turn_activity_epoch,
                        last_spoken_route: &mut last_spoken_route,
                        segment_activity_epoch,
                        turn_session_id,
                        next_output_sequence: &mut next_output_sequence,
                    },
                    job,
                );
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let current_config = runtime_config
                    .read()
                    .map_or_else(|_| config.clone(), |config| config.clone());
                handle_tick(&mut AsrJobContext {
                    handle,
                    config: &current_config,
                    asr: &mut asr,
                    language_id: language_id.as_mut(),
                    turn_detectors: &mut turn_detectors,
                    turns: &mut turns,
                    turn_revisions: &mut turn_revisions,
                    open_turn_id: &mut open_turn_id,
                    open_turn_since: &mut open_turn_since,
                    open_turn_activity_epoch: &mut open_turn_activity_epoch,
                    last_spoken_route: &mut last_spoken_route,
                    segment_activity_epoch,
                    turn_session_id,
                    next_output_sequence: &mut next_output_sequence,
                });
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

pub(in crate::recognition::transcription::worker_runtime) fn handle_asr_job(
    mut context: AsrJobContext<'_>,
    job: AsrJob,
) {
    match job {
        AsrJob::SegmentClosed {
            segment_id,
            previous_segment_id,
            full_audio,
            reason,
        } => {
            handle_segment_closed(
                &mut context,
                segment_id,
                previous_segment_id,
                &full_audio,
                reason,
            );
        }
        AsrJob::TurnCheckSilenceReached {
            previous_segment_id,
        } => handle_turn_check_silence_reached(&mut context, previous_segment_id),
    }
}

fn handle_segment_closed(
    context: &mut AsrJobContext<'_>,
    segment_id: u64,
    previous_segment_id: Option<u64>,
    audio: &[f32],
    reason: SegmentCloseReason,
) {
    if audio.is_empty() {
        return;
    }

    let turn_id = context.open_turn_id.unwrap_or(segment_id);
    let current_route = context
        .turns
        .get(&turn_id)
        .and_then(|turn| turn.draft().route)
        .or(*context.last_spoken_route);
    let selection = select_segment_route(context, turn_id, current_route, audio, reason);
    let route = selection.route;
    let started_at = Instant::now();
    let text = match transcribe_asr(context, route, audio) {
        Ok(text) if !text.is_empty() => text,
        Ok(_) => return,
        Err(err) => {
            emit_asr_warning(context.handle, &err);
            return;
        }
    };
    let elapsed_millis = started_at.elapsed().as_millis();
    let revision = *context.turn_revisions.entry(turn_id).or_insert(0);
    let previous_draft_route = context
        .turns
        .get(&turn_id)
        .and_then(|turn| turn.draft().route);
    let turn = context.turns.entry(turn_id).or_insert_with(|| {
        Turn::new(
            turn_event_id(context.turn_session_id, turn_id, revision),
            revision,
        )
    });
    let draft = turn.draft_mut();
    draft.set_detected_language(selection.detected_language);
    draft.append_recognized_segment(
        segment_id,
        previous_segment_id,
        audio,
        route,
        text,
        elapsed_millis,
    );
    if reason == SegmentCloseReason::EndSilenceReached
        && previous_draft_route.is_some_and(|previous_route| previous_route != route)
    {
        rerecognize_full_turn_with_route(context, turn_id, route);
    }

    let action = if reason == SegmentCloseReason::EndSilenceReached
        && context.config.uses_namo_turn_detector()
        && !namo_final_decision(context, turn_id)
    {
        post_segment_action(reason, TurnProgressDecision::Continue)
    } else {
        post_segment_action(reason, TurnProgressDecision::Complete)
    };
    if action == PostSegmentAction::KeepOpenInterim {
        *context.open_turn_id = Some(turn_id);
        *context.open_turn_since = Some(Instant::now());
        *context.open_turn_activity_epoch = context.segment_activity_epoch.load(Ordering::Acquire);
        emit_turn_output(context, turn_id, false);
        return;
    }

    emit_stale_turn_finals(context, turn_id);
    rerecognize_full_turn_if_needed(context, turn_id);
    emit_turn_output(context, turn_id, true);
    *context.open_turn_id = None;
    *context.open_turn_since = None;
    *context.open_turn_activity_epoch = context.segment_activity_epoch.load(Ordering::Acquire);
}

fn handle_turn_check_silence_reached(context: &mut AsrJobContext<'_>, previous_segment_id: u64) {
    let Some(turn_id) = *context.open_turn_id else {
        return;
    };
    let Some(turn) = context.turns.get(&turn_id) else {
        return;
    };
    if turn.draft().latest_segment_id != Some(previous_segment_id) {
        return;
    }

    let turn_text_changed = if context.config.uses_namo_turn_detector() {
        refresh_turn_route_before_turn_decision(context, turn_id)
    } else {
        false
    };
    let action =
        if context.config.uses_namo_turn_detector() && !namo_final_decision(context, turn_id) {
            post_segment_action(
                SegmentCloseReason::EndSilenceReached,
                TurnProgressDecision::Continue,
            )
        } else {
            post_segment_action(
                SegmentCloseReason::EndSilenceReached,
                TurnProgressDecision::Complete,
            )
        };
    if action == PostSegmentAction::KeepOpenInterim {
        // TurnCheckSilenceReached has no new ASR text. Namo Continue only keeps the existing
        // TurnDraft open and resets the fallback timeout origin.
        *context.open_turn_since = Some(Instant::now());
        *context.open_turn_activity_epoch = context.segment_activity_epoch.load(Ordering::Acquire);
        if turn_text_changed {
            emit_turn_output(context, turn_id, false);
        }
        return;
    }

    emit_stale_turn_finals(context, turn_id);
    rerecognize_full_turn_if_needed(context, turn_id);
    emit_turn_output(context, turn_id, true);
    *context.open_turn_id = None;
    *context.open_turn_since = None;
    *context.open_turn_activity_epoch = context.segment_activity_epoch.load(Ordering::Acquire);
}

fn handle_tick(context: &mut AsrJobContext<'_>) {
    handle_tick_at(context, Instant::now());
}

pub(in crate::recognition::transcription::worker_runtime) fn handle_tick_at(
    context: &mut AsrJobContext<'_>,
    now: Instant,
) {
    let Some(turn_id) = *context.open_turn_id else {
        return;
    };
    let current_activity_epoch = context.segment_activity_epoch.load(Ordering::Acquire);
    if refresh_open_turn_timeout_origin_if_activity_changed(
        Some(turn_id),
        context.open_turn_since,
        context.open_turn_activity_epoch,
        current_activity_epoch,
        now,
    ) {
        return;
    }
    let Some(open_since) = *context.open_turn_since else {
        return;
    };
    let timeout = Duration::from_millis(u64::from(context.config.turn_check_silence_ms) * 2);
    if now.saturating_duration_since(open_since) < timeout {
        return;
    }

    emit_stale_turn_finals(context, turn_id);
    rerecognize_full_turn_if_needed(context, turn_id);
    emit_turn_output(context, turn_id, true);
    *context.open_turn_id = None;
    *context.open_turn_since = None;
    *context.open_turn_activity_epoch = current_activity_epoch;
}

fn emit_asr_missing_event(handle: &AppHandle, reason: String) {
    let _ = handle.emit("parapper://asr-missing", AsrMissingEvent { reason });
}

pub(in crate::recognition::transcription::worker_runtime) fn transcribe_asr(
    context: &mut AsrJobContext<'_>,
    route: RecognitionRoute,
    audio: &[f32],
) -> anyhow::Result<String> {
    let audio = normalize_asr_input_audio(context.config, audio);
    context.asr.transcribe(route, audio.as_ref())
}

pub(crate) fn normalize_asr_input_audio<'a>(
    config: &ParapperConfig,
    audio: &'a [f32],
) -> Cow<'a, [f32]> {
    if !config.asr_normalize_input_audio {
        return Cow::Borrowed(audio);
    }

    let peak = audio
        .iter()
        .copied()
        .filter(|sample| sample.is_finite())
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    if peak <= f32::EPSILON {
        return Cow::Borrowed(audio);
    }

    let gain = NORMALIZED_ASR_INPUT_PEAK / peak;
    if (gain - 1.0).abs() <= f32::EPSILON {
        return Cow::Borrowed(audio);
    }

    Cow::Owned(
        audio
            .iter()
            .copied()
            .map(|sample| {
                if sample.is_finite() {
                    sample * gain
                } else {
                    0.0
                }
            })
            .collect(),
    )
}

pub(crate) fn emit_asr_warning(handle: &AppHandle, err: &anyhow::Error) {
    emit_parapper_error(
        handle,
        ParapperErrorType::Asr,
        ErrorSeverity::Warning,
        Some(err.to_string()),
    );
}
