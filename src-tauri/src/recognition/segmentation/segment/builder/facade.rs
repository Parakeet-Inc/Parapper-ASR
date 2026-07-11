use std::collections::VecDeque;

use crate::{config::ParapperConfig, recognition::segmentation::vad::engine::VadResult};

use super::{
    buffer::{PreSegmentBuffer, SegmentChunk},
    config::SegmentBuilderConfig,
};

// SegmentBuilder は VAD の chunk 列から ASR 用 Segment を組み立てる。
//
// 全体の流れ:
// 1. Idle では、発話前の短い無音を pre_speech に残しながら speech chunk を待つ。
// 2. speech chunk が segment_start_speech_ms まで連続したら SegmentStarted を出し、Active に入る。
// 3. Active では VAD 結果にかかわらず chunk を蓄積し、SegmentExtended を出し続ける。
// 4. interim_result_enabled かつ短い無音が interim_result_silence_ms 以上続いたあと、
//    turn_check_silence_ms に届く前に speech が戻ったら InterimResultSilenceReached で SegmentClosed を出し、
//    そこまでの音声を ASR に渡して途中経過を表示する。Turn は未完了なので、
//    次 segment は同じ Turn の続きとして扱う。
// 5. 無音が turn_check_silence_ms まで続いたら EndSilenceReached で SegmentClosed を出し、
//    VAD 上の発話候補を閉じる。
// 6. 長すぎる場合は SegmentMaxChunksReached で SegmentClosed を出すが、
//    発話候補は未完了なので次 segment を子としてつなぐ。
//
// 連続 segment の考え方:
// - previous_segment_id == None の segment は、新しい発話候補の先頭。
// - previous_segment_id == Some(id) の segment は、同じ発話候補内で直前 segment の続き。
// - EndSilenceReached は VAD 上の発話完了なので、次に始まる segment は必ず新しい発話候補になる。
//
// chunk メタの考え方:
// - SegmentStarted/SegmentClosed の vad_results は、audio に連結された chunk と同じ順番で並ぶ。
// - EndSilenceReached で閉じた末尾無音は、次の root segment の pre_speech にも入れる。
//   これにより xxoooxxxooo のような入力を xxoooxxx / xxxooo... と重ねて切り、
//   ASR が各 segment の前後に十分な無音余白を持てるようにする。
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SegmentBuilderEvent {
    SegmentStarted {
        segment_id: u64,
        previous_segment_id: Option<u64>,
        audio_so_far: Vec<f32>,
        vad_results: Vec<VadResult>,
    },
    SegmentExtended {
        segment_id: u64,
        previous_segment_id: Option<u64>,
        new_audio: Vec<f32>,
        vad_result: VadResult,
    },
    SegmentClosed {
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: Vec<f32>,
        vad_results: Vec<VadResult>,
        source_audio: Vec<f32>,
        source_vad_results: Vec<VadResult>,
        reason: SegmentCloseReason,
    },
    TurnCheckSilenceReached {
        previous_segment_id: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::enum_variant_names,
    reason = "segment_builder README とテストで close threshold 到達理由として明示している"
)]
pub(crate) enum SegmentCloseReason {
    InterimChunkReached,
    InterimResultSilenceReached,
    EndSilenceReached,
    SegmentMaxChunksReached,
}

pub(crate) struct SegmentBuilder {
    config: SegmentBuilderConfig,
    state: SegmentBuilderState,
    next_segment_id: u64,
    // SegmentMaxChunksReached で閉じた直後だけ、次の SegmentStarted を直前 segment の続きにする。
    pending_previous_segment_id: Option<u64>,
}

enum SegmentBuilderState {
    Idle {
        // 発話開始直前の無音を少し残しておき、ASR に語頭を欠けさせないためのバッファ。
        pre_speech: PreSegmentBuffer,
        // VAD の単発誤検出を避けるため、segment_start_speech_ms に届くまでは発話開始を保留する。
        pending_speech_chunks: VecDeque<SegmentChunk>,
    },
    AfterInterimSilence {
        pre_speech: PreSegmentBuffer,
        pending_speech_chunks: VecDeque<SegmentChunk>,
        previous_segment_id: u64,
        silence_chunks: u32,
    },
    Active(SegmentInProgress),
}

#[derive(Debug, Clone)]
struct SegmentInProgress {
    id: u64,
    previous_segment_id: Option<u64>,
    audio: Vec<f32>,
    chunks: Vec<SegmentChunk>,
    audio_chunks: u32,
    silence_chunks: u32,
}

impl SegmentBuilder {
    pub(crate) fn new(config: &ParapperConfig) -> Self {
        let config = SegmentBuilderConfig::from_config(config);
        Self {
            state: SegmentBuilderState::new(&config),
            config,
            next_segment_id: 1,
            pending_previous_segment_id: None,
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "VAD chunk から Segment event を出す状態遷移を 1 箇所で追えるように保持している"
    )]
    pub(crate) fn push(
        &mut self,
        samples: &[f32],
        vad_result: VadResult,
    ) -> Vec<SegmentBuilderEvent> {
        let mut events = Vec::new();
        // state を一度取り出してから次 state を返す形にして、Idle/Active の遷移を 1 箇所で完結させる。
        let state = std::mem::replace(&mut self.state, SegmentBuilderState::new(&self.config));
        self.state = match state {
            SegmentBuilderState::Idle {
                mut pre_speech,
                mut pending_speech_chunks,
            } => {
                if vad_result.is_speech {
                    // speech が閾値まで続いたら segment 開始。
                    // pending_previous_segment_id があれば前 segment の続き、なければ新しい発話候補として始める。
                    pending_speech_chunks
                        .push_back(SegmentChunk::new(samples.to_vec(), vad_result));
                    if pending_speech_chunks.len() >= self.config.segment_start_threshold as usize {
                        let segment_id = self.take_next_segment_id();
                        let previous_segment_id = self.pending_previous_segment_id.take();
                        let mut audio_so_far = Vec::new();
                        let mut chunks = pre_speech.drain_into(&mut audio_so_far);
                        let mut vad_results =
                            chunks.iter().map(|chunk| chunk.vad).collect::<Vec<_>>();
                        for chunk in pending_speech_chunks.drain(..) {
                            audio_so_far.extend_from_slice(&chunk.audio);
                            vad_results.push(chunk.vad);
                            chunks.push(chunk);
                        }
                        events.push(SegmentBuilderEvent::SegmentStarted {
                            segment_id,
                            previous_segment_id,
                            audio_so_far: audio_so_far.clone(),
                            vad_results,
                        });
                        SegmentBuilderState::Active(SegmentInProgress {
                            id: segment_id,
                            previous_segment_id,
                            audio: audio_so_far,
                            chunks,
                            audio_chunks: self.config.segment_start_threshold,
                            silence_chunks: 0,
                        })
                    } else {
                        SegmentBuilderState::Idle {
                            pre_speech,
                            pending_speech_chunks,
                        }
                    }
                } else {
                    // speech 開始前に無音を挟んだら、前 segment の続きではない。
                    // そのため pending_previous_segment_id を捨て、次の SegmentStarted は新しい発話候補にする。
                    // 閾値未満の speech 候補は発話冒頭の VAD 揺れかもしれないため、pre_speech に戻す。
                    self.pending_previous_segment_id = None;
                    move_pending_speech_to_pre_speech(&mut pending_speech_chunks, &mut pre_speech);
                    pre_speech.push(samples, vad_result);
                    SegmentBuilderState::Idle {
                        pre_speech,
                        pending_speech_chunks,
                    }
                }
            }
            SegmentBuilderState::AfterInterimSilence {
                mut pre_speech,
                mut pending_speech_chunks,
                previous_segment_id,
                mut silence_chunks,
            } => {
                if vad_result.is_speech {
                    pending_speech_chunks
                        .push_back(SegmentChunk::new(samples.to_vec(), vad_result));
                    if pending_speech_chunks.len() >= self.config.segment_start_threshold as usize {
                        let segment_id = self.take_next_segment_id();
                        let mut audio_so_far = Vec::new();
                        let mut chunks = pre_speech.drain_into(&mut audio_so_far);
                        let mut vad_results =
                            chunks.iter().map(|chunk| chunk.vad).collect::<Vec<_>>();
                        for chunk in pending_speech_chunks.drain(..) {
                            audio_so_far.extend_from_slice(&chunk.audio);
                            vad_results.push(chunk.vad);
                            chunks.push(chunk);
                        }
                        events.push(SegmentBuilderEvent::SegmentStarted {
                            segment_id,
                            previous_segment_id: Some(previous_segment_id),
                            audio_so_far: audio_so_far.clone(),
                            vad_results,
                        });
                        SegmentBuilderState::Active(SegmentInProgress {
                            id: segment_id,
                            previous_segment_id: Some(previous_segment_id),
                            audio: audio_so_far,
                            chunks,
                            audio_chunks: self.config.segment_start_threshold,
                            silence_chunks: 0,
                        })
                    } else {
                        SegmentBuilderState::AfterInterimSilence {
                            pre_speech,
                            pending_speech_chunks,
                            previous_segment_id,
                            silence_chunks,
                        }
                    }
                } else {
                    move_pending_speech_to_pre_speech(&mut pending_speech_chunks, &mut pre_speech);
                    // This silence still belongs to the continuing open turn until turn-check
                    // silence is reached, so keep it as source audio instead of ASR-only padding.
                    pre_speech.push(samples, vad_result);
                    silence_chunks = silence_chunks.saturating_add(1);
                    if silence_chunks >= self.config.turn_check_threshold {
                        events.push(SegmentBuilderEvent::TurnCheckSilenceReached {
                            previous_segment_id,
                        });
                        SegmentBuilderState::Idle {
                            pre_speech,
                            pending_speech_chunks,
                        }
                    } else {
                        SegmentBuilderState::AfterInterimSilence {
                            pre_speech,
                            pending_speech_chunks,
                            previous_segment_id,
                            silence_chunks,
                        }
                    }
                }
            }
            SegmentBuilderState::Active(mut active) => {
                if vad_result.is_speech
                    && self
                        .config
                        .interim_result_threshold
                        .is_some_and(|threshold| {
                            active.silence_chunks >= threshold
                                && active.silence_chunks < self.config.turn_check_threshold
                        })
                {
                    let previous_segment_id = active.id;
                    let trailing_silence = trailing_silence_chunks(&active);
                    let silence_chunks = trailing_silence.len().try_into().unwrap_or(u32::MAX);
                    let vad_results = active.chunks.iter().map(|chunk| chunk.vad).collect();
                    let source_audio = source_audio_from_chunks(&active.chunks);
                    let source_vad_results = source_vad_results_from_chunks(&active.chunks);
                    events.push(SegmentBuilderEvent::SegmentClosed {
                        segment_id: active.id,
                        previous_segment_id: active.previous_segment_id,
                        full_audio: active.audio,
                        vad_results,
                        source_audio,
                        source_vad_results,
                        reason: SegmentCloseReason::InterimResultSilenceReached,
                    });

                    let mut pre_speech = PreSegmentBuffer::new(self.config.pre_speech_max_chunks);
                    for chunk in trailing_silence {
                        pre_speech.push_chunk(chunk.into_asr_padding());
                    }
                    let mut pending_speech_chunks = VecDeque::new();
                    pending_speech_chunks
                        .push_back(SegmentChunk::new(samples.to_vec(), vad_result));
                    if pending_speech_chunks.len() >= self.config.segment_start_threshold as usize {
                        let segment_id = self.take_next_segment_id();
                        let mut audio_so_far = Vec::new();
                        let mut chunks = pre_speech.drain_into(&mut audio_so_far);
                        let mut vad_results =
                            chunks.iter().map(|chunk| chunk.vad).collect::<Vec<_>>();
                        for chunk in pending_speech_chunks.drain(..) {
                            audio_so_far.extend_from_slice(&chunk.audio);
                            vad_results.push(chunk.vad);
                            chunks.push(chunk);
                        }
                        events.push(SegmentBuilderEvent::SegmentStarted {
                            segment_id,
                            previous_segment_id: Some(previous_segment_id),
                            audio_so_far: audio_so_far.clone(),
                            vad_results,
                        });
                        self.state = SegmentBuilderState::Active(SegmentInProgress {
                            id: segment_id,
                            previous_segment_id: Some(previous_segment_id),
                            audio: audio_so_far,
                            chunks,
                            audio_chunks: self.config.segment_start_threshold,
                            silence_chunks: 0,
                        });
                    } else {
                        self.state = SegmentBuilderState::AfterInterimSilence {
                            pre_speech,
                            pending_speech_chunks,
                            previous_segment_id,
                            silence_chunks,
                        };
                    }
                    return events;
                }

                // Active 中の chunk は VAD 結果にかかわらず同じ segment の音声として蓄積する。
                // 無音 chunk も残すことで、発話末尾の音が不自然に切れないようにする。
                active.audio_chunks = active.audio_chunks.saturating_add(1);
                active.audio.extend_from_slice(samples);
                active
                    .chunks
                    .push(SegmentChunk::new(samples.to_vec(), vad_result));
                events.push(SegmentBuilderEvent::SegmentExtended {
                    segment_id: active.id,
                    previous_segment_id: active.previous_segment_id,
                    new_audio: samples.to_vec(),
                    vad_result,
                });

                if vad_result.is_speech {
                    active.silence_chunks = 0;
                } else {
                    active.silence_chunks = active.silence_chunks.saturating_add(1);
                }

                // 終了条件は 2 種類。
                // - 長すぎる発話は max_chunks で ASR 入力単位を分け、次 segment に previous を渡す。
                // - 無音が turn_check_silence_ms まで続いた場合は発話完了として閉じる。
                //   interim_result_silence_ms 以上で止まった短い無音は、次の speech が来た時点で閉じる。
                let close_reason = if active.audio_chunks >= self.config.max_chunks {
                    Some(SegmentCloseReason::SegmentMaxChunksReached)
                } else if active.silence_chunks >= self.config.turn_check_threshold {
                    Some(SegmentCloseReason::EndSilenceReached)
                } else {
                    None
                };

                if let Some(reason) = close_reason {
                    let next_state = state_after_close(&active, reason, &self.config);
                    // - InterimResultSilenceReached / SegmentMaxChunksReached は未完了 Turn の ASR 入力単位なので、
                    //   次 segment は active.id の続きにする。
                    // - EndSilenceReached は発話候補の完了なので、次 segment は previous なしの新規開始にする。
                    self.pending_previous_segment_id =
                        if reason == SegmentCloseReason::SegmentMaxChunksReached {
                            Some(active.id)
                        } else {
                            None
                        };
                    let vad_results = active.chunks.iter().map(|chunk| chunk.vad).collect();
                    let source_audio = source_audio_from_chunks(&active.chunks);
                    let source_vad_results = source_vad_results_from_chunks(&active.chunks);
                    events.push(SegmentBuilderEvent::SegmentClosed {
                        segment_id: active.id,
                        previous_segment_id: active.previous_segment_id,
                        full_audio: active.audio,
                        vad_results,
                        source_audio,
                        source_vad_results,
                        reason,
                    });
                    next_state
                } else {
                    SegmentBuilderState::Active(active)
                }
            }
        };
        events
    }

    pub(crate) fn update_config(&mut self, config: &ParapperConfig) {
        let next_config = SegmentBuilderConfig::from_config(config);
        self.state.update_config(&next_config);
        self.config = next_config;
    }

    pub(crate) fn flush(&mut self) -> Vec<SegmentBuilderEvent> {
        let state = std::mem::replace(&mut self.state, SegmentBuilderState::new(&self.config));
        match state {
            SegmentBuilderState::Active(active) => {
                let next_state =
                    state_after_close(&active, SegmentCloseReason::EndSilenceReached, &self.config);
                self.pending_previous_segment_id = None;
                let vad_results = active.chunks.iter().map(|chunk| chunk.vad).collect();
                let source_audio = source_audio_from_chunks(&active.chunks);
                let source_vad_results = source_vad_results_from_chunks(&active.chunks);
                self.state = next_state;
                vec![SegmentBuilderEvent::SegmentClosed {
                    segment_id: active.id,
                    previous_segment_id: active.previous_segment_id,
                    full_audio: active.audio,
                    vad_results,
                    source_audio,
                    source_vad_results,
                    reason: SegmentCloseReason::EndSilenceReached,
                }]
            }
            SegmentBuilderState::AfterInterimSilence {
                previous_segment_id,
                ..
            } => {
                self.pending_previous_segment_id = None;
                self.state = SegmentBuilderState::new(&self.config);
                vec![SegmentBuilderEvent::TurnCheckSilenceReached {
                    previous_segment_id,
                }]
            }
            idle @ SegmentBuilderState::Idle { .. } => {
                self.state = idle;
                Vec::new()
            }
        }
    }

    fn take_next_segment_id(&mut self) -> u64 {
        let segment_id = self.next_segment_id;
        self.next_segment_id = self.next_segment_id.saturating_add(1);
        segment_id
    }
}

fn state_after_close(
    active: &SegmentInProgress,
    reason: SegmentCloseReason,
    config: &SegmentBuilderConfig,
) -> SegmentBuilderState {
    match reason {
        SegmentCloseReason::InterimChunkReached => {
            unreachable!("streaming interim chunk is not a segment-builder close reason")
        }
        SegmentCloseReason::InterimResultSilenceReached => {
            let trailing_silence = trailing_silence_chunks(active);
            let silence_chunks = trailing_silence.len().try_into().unwrap_or(u32::MAX);
            let mut pre_speech = PreSegmentBuffer::new(config.pre_speech_max_chunks);
            for chunk in trailing_silence {
                pre_speech.push_chunk(chunk.into_asr_padding());
            }
            SegmentBuilderState::AfterInterimSilence {
                pre_speech,
                pending_speech_chunks: VecDeque::new(),
                previous_segment_id: active.id,
                silence_chunks,
            }
        }
        SegmentCloseReason::EndSilenceReached => SegmentBuilderState::new_with_pre_speech_chunks(
            config,
            trailing_silence_chunks(active)
                .into_iter()
                .map(SegmentChunk::into_asr_padding)
                .collect(),
        ),
        SegmentCloseReason::SegmentMaxChunksReached => SegmentBuilderState::new(config),
    }
}

fn trailing_silence_chunks(active: &SegmentInProgress) -> Vec<SegmentChunk> {
    let mut trailing_silence = active
        .chunks
        .iter()
        .rev()
        .take_while(|chunk| !chunk.vad.is_speech)
        .cloned()
        .collect::<Vec<_>>();
    trailing_silence.reverse();
    trailing_silence
}

fn move_pending_speech_to_pre_speech(
    pending_speech_chunks: &mut VecDeque<SegmentChunk>,
    pre_speech: &mut PreSegmentBuffer,
) {
    for chunk in pending_speech_chunks.drain(..) {
        pre_speech.push_chunk(chunk);
    }
}

fn source_audio_from_chunks(chunks: &[SegmentChunk]) -> Vec<f32> {
    let mut audio = Vec::new();
    for chunk in chunks.iter().filter(|chunk| chunk.include_in_turn_audio) {
        audio.extend_from_slice(&chunk.audio);
    }
    audio
}

fn source_vad_results_from_chunks(chunks: &[SegmentChunk]) -> Vec<VadResult> {
    chunks
        .iter()
        .filter(|chunk| chunk.include_in_turn_audio)
        .map(|chunk| chunk.vad)
        .collect()
}

impl SegmentBuilderState {
    fn new(config: &SegmentBuilderConfig) -> Self {
        Self::Idle {
            pre_speech: PreSegmentBuffer::new(config.pre_speech_max_chunks),
            pending_speech_chunks: VecDeque::new(),
        }
    }

    fn new_with_pre_speech_chunks(
        config: &SegmentBuilderConfig,
        chunks: Vec<SegmentChunk>,
    ) -> Self {
        let mut pre_speech = PreSegmentBuffer::new(config.pre_speech_max_chunks);
        for chunk in chunks {
            pre_speech.push_chunk(chunk);
        }
        Self::Idle {
            pre_speech,
            pending_speech_chunks: VecDeque::new(),
        }
    }

    fn update_config(&mut self, config: &SegmentBuilderConfig) {
        match self {
            Self::Idle { pre_speech, .. } | Self::AfterInterimSilence { pre_speech, .. } => {
                pre_speech.update_max_chunks(config.pre_speech_max_chunks);
            }
            // Active 中の終了閾値は SegmentBuilder.config を push ごとに読む。
            // pre_speech は Idle 専用なので、サイズ変更は次に Idle へ戻ったときに反映される。
            Self::Active(_) => {}
        }
    }
}
