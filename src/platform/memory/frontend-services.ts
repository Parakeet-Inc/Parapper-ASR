import { frontendPreviewEvents } from "../../application/app-state/fixtures";
import type {
  FrontendCapabilities,
  FrontendEvent,
  FrontendServices,
} from "../../application/frontend-services";
import type {
  ConfigPreset,
  ModelStatus,
  ParapperConfig,
  RecognitionStatus,
} from "../../lib/types";

const unsupported = (capability: string): never => {
  throw new Error(`${capability} is not available in the web preview`);
};

const previewConfig: ParapperConfig = {
  neo_http_enabled: false,
  neo_http_port: 15520,
  input_source_kind: "web_socket",
  input_device_id: null,
  input_device_host: null,
  input_device_name: null,
  input_volume_db: 0,
  asr_language: "japanese",
  asr_model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
  interim_asr_model: null,
  asr_precision: "int8",
  asr_num_threads: 4,
  asr_mode: "fast",
  asr_hotwords_enabled: false,
  asr_hotwords: [],
  asr_normalize_input_audio: true,
  multilingual_asr_enabled: false,
  enabled_asr_models: ["nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8"],
  translation_enabled: false,
  ync_plugin_port: 15520,
  translation_local_server_port: 18081,
  translation_local_server_model: "lfm2_q4",
  translation_send_timing: "final",
  translation_mappings: [],
  speech_mappings: [],
  model_dir: null,
  vad_threshold: 0.5,
  vad_interval_ms: 100,
  segment_start_speech_ms: 200,
  turn_detector: "simple",
  interim_result_enabled: true,
  interim_result_silence_ms: 300,
  turn_check_silence_ms: 700,
  namo_turn_confidence_threshold: 0.5,
  namo_context_max_tokens: 256,
  turn_rerecognize_full_on_complete: false,
  noise_cancellation_enabled: false,
  noise_cancellation_model: "ul_unas",
  noise_cancellation_target: "vad_only",
  vrc_osc_micmute: false,
  streaming_recognition_enabled: true,
  developer_connection_mode: "web_socket",
  developer_http_url: "http://127.0.0.1:18080",
  streaming_recognition_bind_address: "127.0.0.1",
  streaming_recognition_port: 18082,
  streaming_recognition_api_key: null,
  streaming_recognition_output_mode: "web_socket_only",
  debug_asr_audio_playback: false,
  recognition_log_limit: 100,
  debug_audio_log_limit: 10,
};

const installedAsset = (path: string) => ({
  installed: true,
  preparing: false,
  path,
});

const previewModelStatus: ModelStatus = {
  root_dir: "memory://models",
  vad: installedAsset("memory://models/vad"),
  asr: installedAsset("memory://models/asr"),
  japanese_morph: null,
  language_id: null,
  turn_detectors: [],
  tts: [],
  local_translation: null,
  noise_cancellation: null,
};

export const memoryCapabilities: FrontendCapabilities = {
  localAudioDevices: false,
  outputAudioDevices: false,
  modelManagement: false,
  systemAudioPermission: false,
  externalConnectionProbe: false,
  fileExport: false,
  recognitionControl: true,
  speechControl: false,
  localTranslationServer: false,
};

export const createMemoryFrontendServices = (): FrontendServices => {
  let config = structuredClone(previewConfig);
  let status: RecognitionStatus = "idle";
  let presets: ConfigPreset[] = [];
  const listeners = new Set<(event: FrontendEvent) => void>();
  let audioContext: AudioContext | null = null;
  let audioSource: AudioBufferSourceNode | null = null;
  const emit = (event: FrontendEvent) =>
    listeners.forEach((listener) => listener(event));

  return {
    config: {
      load: async () => structuredClone(config),
      save: async (nextConfig) => {
        config = structuredClone(nextConfig);
        return structuredClone(config);
      },
      reset: async () => {
        config = structuredClone(previewConfig);
        return structuredClone(config);
      },
    },
    recognition: {
      status: async () => status,
      start: async () => {
        status = "listening";
        emit({ type: "recognitionStatusChanged", payload: status });
        frontendPreviewEvents
          .filter((event) => event.type !== "recognitionStatusChanged")
          .forEach(emit);
        return status;
      },
      stop: async () => {
        status = "stopped";
        emit({ type: "recognitionStatusChanged", payload: status });
        return status;
      },
    },
    events: {
      subscribe: async (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    models: {
      status: async () => structuredClone(previewModelStatus),
      hasAnyInstalled: async () => true,
      download: async () => unsupported("model management"),
      downloadLocalTranslation: async () => unsupported("model management"),
      isLocalTranslationInstalled: async () => false,
    },
    audioDevices: {
      inputDevices: async () => [],
      outputDevices: async () => [],
      requestLoopbackPermission: async () =>
        unsupported("system audio permission"),
      openLoopbackPermissionSettings: async () =>
        unsupported("system audio permission"),
    },
    presets: {
      list: async () => structuredClone(presets),
      save: async (name, presetConfig) => {
        presets = [
          ...presets.filter((preset) => preset.name !== name),
          {
            name,
            built_in: false,
            config: structuredClone(presetConfig),
          },
        ];
        return structuredClone(presets);
      },
      delete: async (name) => {
        presets = presets.filter((preset) => preset.name !== name);
        return structuredClone(presets);
      },
    },
    connections: {
      checkNeo: async () => unsupported("external connection probe"),
      checkVrchat: async () => unsupported("external connection probe"),
      findNeoPort: async () => unsupported("external connection probe"),
      findYncPluginPort: async () => unsupported("external connection probe"),
      fetchNeoVoices: async () => unsupported("external connection probe"),
    },
    speech: {
      stop: async () => unsupported("speech control"),
    },
    hotwordReadings: {
      suggest: async () => unsupported("hotword reading autofill"),
    },
    translationServer: {
      status: async () => ({ state: "stopped", port: null, error: null }),
      start: async () => unsupported("local translation server"),
      stop: async () => unsupported("local translation server"),
    },
    system: {
      saveRecognitionCsv: async () => unsupported("file export"),
      saveAsrInputWav: async () => unsupported("file export"),
      openExternalUrl: async (url) => {
        window.open(url, "_blank", "noopener,noreferrer");
      },
      playAudio: async (samples, sampleRate) => {
        audioContext ??= new AudioContext();
        if (audioContext.state === "suspended") {
          await audioContext.resume();
        }
        audioSource?.stop();
        const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
        buffer.copyToChannel(Float32Array.from(samples), 0);
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);
        source.start();
        audioSource = source;
      },
      loadRustLicenses: async () => {
        const response = await fetch("/licenses/rust.json");
        if (!response.ok) {
          throw new Error(`Failed to load Rust licenses: ${response.status}`);
        }
        return response.json();
      },
    },
  };
};
