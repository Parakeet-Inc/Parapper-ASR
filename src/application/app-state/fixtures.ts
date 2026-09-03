import type { RecognizedTextEvent } from "../../lib/types";
import type { FrontendEvent } from "../frontend-services";

const recognized = (
  id: string,
  text: string,
  outputSequence: number,
  isFinal: boolean,
): RecognizedTextEvent => ({
  id,
  source: {
    identity: {
      source_id: "legacy-single-source",
      speaker_label: "Legacy single input",
      capture_endpoint_id: "legacy-single-capture",
      channel_index: null,
    },
    turn_session_id: 7,
    turn_id: 1,
    turn_revision: 0,
    output_sequence: outputSequence,
    segment_id: 1,
    previous_segment_id: null,
  },
  is_final: isFinal,
  update_mode: "replace",
  text,
  source_asr_model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
  source_language: "japanese",
  detected_language: "ja",
  recognized_at_millis: 1_700_000_000_000 + outputSequence,
  audio_seconds: 1.25,
  elapsed_millis: 90 + outputSequence,
  audio_frames: 20_000,
  debug_asr_audio_sample_rate: null,
  debug_asr_audio_samples: null,
});

export const frontendPreviewEvents: readonly FrontendEvent[] = [
  { type: "recognitionStatusChanged", payload: "listening" },
  {
    type: "recognizedTextReceived",
    payload: recognized("turn-1-interim-1", "音声", 1, false),
  },
  {
    type: "recognizedTextReceived",
    payload: recognized("turn-1-interim-2", "音声認識", 2, false),
  },
  {
    type: "recognizedTextReceived",
    payload: recognized("turn-1-final", "音声認識です。", 3, true),
  },
  {
    type: "recognizedTextReceived",
    payload: recognized("turn-1-late-interim", "古い途中結果", 2, false),
  },
  {
    type: "applicationError",
    payload: {
      errorType: "ASR",
      severity: "warning",
      detail: "preview fixture error",
    },
  },
] as const;
