export type AsrPrecision = "int8" | "int8_float32" | "float32";
export type AsrLanguage = "japanese" | "english" | "european_multilingual";
export type AsrModel =
  | "reazonspeech_k2_v2"
  | "nemo_parakeet_tdt_0_6b_v2_int8"
  | "nemo_parakeet_tdt_0_6b_v3_int8";

export type RecognitionStatus = "idle" | "listening" | "stopped" | "error";

export type ParapperConfig = {
  neo_http_enabled: boolean;
  neo_http_port: number;
  input_device_id: string | null;
  input_device_host: string | null;
  input_device_name: string | null;
  asr_language: AsrLanguage;
  asr_model: AsrModel;
  asr_precision: AsrPrecision;
  asr_num_threads: number;
  model_dir: string | null;
  vad_threshold: number;
  vad_interval_ms: number;
  pause_threshold: number;
  phrase_threshold: number;
  vrc_osc_micmute: boolean;
  debug_asr_audio_playback: boolean;
  recognition_log_limit: number | null;
  debug_audio_log_limit: number | null;
};

export type AudioDeviceInfo = {
  id: string;
  host: string;
  display_name: string;
  channels: number;
  sample_rate: number;
};

export type VadStateEvent = {
  state: "speech" | "silence";
  probability: number;
};

export type RecognizedTextEvent = {
  text: string;
  recognized_at_millis: number;
  audio_seconds: number;
  elapsed_millis: number;
  audio_frames: number;
  debug_asr_audio_sample_rate: number | null;
  debug_asr_audio_samples: number[] | null;
};

export type AsrMissingEvent = {
  reason: string;
};

export type OscMuteStateEvent = {
  muted: boolean | null;
};

export type ConnectionStateEvent = {
  target: "neo" | "vrchat";
  found: boolean;
  detail: string | null;
};

export type ModelAssetStatus = {
  installed: boolean;
  path: string;
};

export type ModelStatus = {
  root_dir: string;
  vad: ModelAssetStatus;
  asr: ModelAssetStatus;
};

export type ModelDownloadProgress = {
  file_name: string;
  file_index: number;
  total_files: number;
  downloaded_bytes: number;
  total_bytes: number | null;
  progress: number;
  finished: boolean;
};

export type ErrorSeverity = "warning" | "fatal";

export type ParapperErrorType =
  | "AUDIO_INPUT"
  | "RESAMPLER"
  | "VAD"
  | "ASR"
  | "MODEL_DOWNLOAD"
  | "NEO_HTTP"
  | "OSC_QUERY"
  | "FILE_SAVE"
  | "CONFIG"
  | "UNKNOWN";

export type ParapperErrorPayload = {
  errorType: ParapperErrorType;
  severity: ErrorSeverity;
  detail: string | null;
};
