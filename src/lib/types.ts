export type AsrPrecision = "int8" | "int8_float32" | "float32";
export type AsrLanguage =
  | "japanese"
  | "english"
  | "european_multilingual"
  | "multilingual";
export type TurnDetector = "simple" | "morph" | "namo";
export type NoiseCancellationModel = "ul_unas";
export type NeoSendTiming = "interim" | "final";
export type DeveloperConnectionMode = "http" | "web_socket";
export type SpeechSourceKind = "recognition" | "translation";
export type SpeechBackend = "ync" | "local_tts";
export type TranslationBackend = "ync" | "local";
export type LocalTranslationModel = "lfm2_q4";
export type TranslationLanguage = "en" | "ja";
export type LocalTtsVoice =
  | "vits_piper_en_US_kristin_medium"
  | "vits_piper_en_US_john_medium"
  | "vits_piper_en_US_norman_medium"
  | "supertonic_2_onnx"
  | "supertonic_3_onnx";
export type AsrModel =
  | "reazonspeech_k2_v2"
  | "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8"
  | "nemo_parakeet_tdt_0_6b_v2_int8"
  | "nemo_parakeet_tdt_0_6b_v3_int8"
  | "nemotron_speech_streaming_en_0_6b_160ms_int8"
  | "nemotron_speech_streaming_en_0_6b_560ms_int8"
  | "nemotron_3_5_asr_streaming_0_6b_160ms_int8"
  | "nemotron_3_5_asr_streaming_0_6b_560ms_int8";

export type RecognitionStatus =
  | "idle"
  | "waiting_for_client"
  | "listening"
  | "draining"
  | "stopped"
  | "error";
export type InputSourceKind = "desktop_audio" | "web_socket";
export type StreamingRecognitionOutputMode =
  | "web_socket_only"
  | "web_socket_and_desktop";

export type ParapperConfig = {
  neo_http_enabled: boolean;
  neo_http_port: number;
  input_source_kind: InputSourceKind;
  input_device_id: string | null;
  input_device_host: string | null;
  input_device_name: string | null;
  input_volume_db: number;
  asr_language: AsrLanguage;
  asr_model: AsrModel;
  interim_asr_model: AsrModel | null;
  asr_precision: AsrPrecision;
  asr_num_threads: number;
  asr_normalize_input_audio: boolean;
  multilingual_asr_enabled: boolean;
  enabled_asr_models: AsrModel[];
  translation_enabled: boolean;
  ync_plugin_port: number;
  translation_local_server_port: number;
  translation_local_server_model: LocalTranslationModel;
  translation_send_timing: NeoSendTiming;
  translation_mappings: TranslationMapping[];
  speech_mappings: SpeechMapping[];
  model_dir: string | null;
  vad_threshold: number;
  vad_interval_ms: number;
  segment_start_speech_ms: number;
  turn_detector: TurnDetector;
  interim_result_enabled: boolean;
  interim_result_silence_ms: number;
  turn_check_silence_ms: number;
  namo_turn_confidence_threshold: number;
  namo_context_max_tokens: number;
  turn_rerecognize_full_on_complete: boolean;
  noise_cancellation_enabled: boolean;
  noise_cancellation_model: NoiseCancellationModel;
  vrc_osc_micmute: boolean;
  streaming_recognition_enabled: boolean;
  developer_connection_mode: DeveloperConnectionMode;
  developer_http_url: string;
  streaming_recognition_bind_address: string;
  streaming_recognition_port: number;
  streaming_recognition_api_key: string | null;
  streaming_recognition_output_mode: StreamingRecognitionOutputMode;
  debug_asr_audio_playback: boolean;
  recognition_log_limit: number | null;
  debug_audio_log_limit: number | null;
};

export type TranslationHttpListenerStatus = {
  state: "stopped" | "starting" | "running" | "stopping" | "error";
  port: number | null;
  error: string | null;
};

export type ConfigPreset = {
  name: string;
  built_in: boolean;
  config: ParapperConfig;
};

export type TranslationMapping = {
  id: string;
  source_asr_model: AsrModel | null;
  backend: TranslationBackend;
  local_model: LocalTranslationModel;
  source_lang: TranslationLanguage;
  target_lang: TranslationLanguage;
};

export type SpeechMapping = {
  id: string;
  source_kind: SpeechSourceKind;
  source_asr_model: AsrModel | null;
  target_lang: string | null;
  backend: SpeechBackend;
  talker: string;
  local_tts_voice: LocalTtsVoice | null;
  local_tts_language: string | null;
  local_tts_speaker_id: number | null;
  output_device_id: string | null;
  output_device_host: string | null;
  output_device_name: string | null;
  muted: boolean;
  volume: number;
};

export type AudioDeviceInfo = {
  id: string;
  host: string;
  display_name: string;
  channels: number;
  sample_rate: number;
};

export type InputLevelEvent = {
  pre_gain_level: number;
  post_gain_level: number;
};

export type VadStateEvent = {
  state: "speech" | "silence";
  probability: number;
};

export type RecognizedTextEvent = {
  id: string;
  source: RecognitionSourceMeta;
  is_final: boolean;
  update_mode: "append" | "replace";
  text: string;
  source_asr_model: AsrModel;
  source_language: AsrLanguage;
  detected_language: string | null;
  recognized_at_millis: number;
  audio_seconds: number;
  elapsed_millis: number;
  audio_frames: number;
  debug_asr_audio_sample_rate: number | null;
  debug_asr_audio_samples: number[] | null;
};

export type RecognitionSourceMeta = {
  turn_session_id: number;
  turn_id: number;
  turn_revision: number;
  output_sequence: number;
  segment_id: number;
  previous_segment_id: number | null;
};

export type TranslationTextEvent = {
  id: string;
  source_recognition_id: string;
  source: RecognitionSourceMeta;
  source_asr_model: AsrModel;
  source_text: string;
  source_detected_language: string | null;
  target_lang: string;
  translated_text: string;
  is_final: boolean;
  update_mode: "append" | "replace";
  translated_at_millis: number;
  elapsed_millis: number;
  status: "success" | "failure";
  error: string | null;
};

export type SpeechRequestEvent = {
  id: string;
  source_event_id: string;
  source_kind: SpeechSourceKind;
  target_lang: string | null;
  elapsed_millis: number;
  status: "accepted" | "failure";
  error: string | null;
};

export type AsrMissingEvent = {
  kind: "asr" | "language_id" | "turn_detector";
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
  preparing: boolean;
  path: string;
};

export type ModelStatus = {
  root_dir: string;
  vad: ModelAssetStatus;
  asr: ModelAssetStatus;
  japanese_morph: ModelAssetStatus | null;
  language_id: ModelAssetStatus | null;
  turn_detectors: ModelAssetStatus[];
  tts: ModelAssetStatus[];
  local_translation: ModelAssetStatus | null;
  noise_cancellation: ModelAssetStatus | null;
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
  | "RECOGNITION_BUSY"
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
