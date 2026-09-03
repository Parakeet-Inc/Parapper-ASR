import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import type {
  FrontendCapabilities,
  FrontendEvent,
  FrontendServices,
  Unsubscribe,
} from "../../application/frontend-services";
import type {
  AsrMissingEvent,
  ConnectionStateEvent,
  InputLevelEvent,
  ModelDownloadProgress,
  OscMuteStateEvent,
  ParapperErrorPayload,
  RecognizedTextEvent,
  RecognitionStatus,
  SpeechRequestEvent,
  TranslationTextEvent,
  VadStateEvent,
} from "../../lib/types";

type InvokeFn = <T>(
  command: string,
  args?: Record<string, unknown>,
) => Promise<T>;
type ListenFn = <T>(
  channel: string,
  handler: (event: { payload: T }) => void,
) => Promise<Unsubscribe>;

type TauriDependencies = {
  invoke: InvokeFn;
  listen: ListenFn;
};

const defaultDependencies: TauriDependencies = {
  invoke: <T>(command: string, args?: Record<string, unknown>) =>
    invoke<T>(command, args),
  listen: <T>(channel: string, handler: (event: { payload: T }) => void) =>
    listen<T>(channel, handler),
};

const subscribeToFrontendEvents = async (
  dependencies: TauriDependencies,
  listener: (event: FrontendEvent) => void,
): Promise<Unsubscribe> => {
  const unsubscribes: Unsubscribe[] = [];
  const register = async <T>(channel: string, type: FrontendEvent["type"]) => {
    unsubscribes.push(
      await dependencies.listen<T>(channel, ({ payload }) =>
        listener({ type, payload } as FrontendEvent),
      ),
    );
  };

  try {
    await register<RecognitionStatus>(
      "parapper://status",
      "recognitionStatusChanged",
    );
    await register<InputLevelEvent | number>(
      "parapper://input-level",
      "inputLevelChanged",
    );
    await register<VadStateEvent>("parapper://vad-state", "vadStateChanged");
    await register<RecognizedTextEvent>(
      "parapper://recognized-text",
      "recognizedTextReceived",
    );
    await register<TranslationTextEvent>(
      "parapper://translated-text",
      "translationTextReceived",
    );
    await register<SpeechRequestEvent>(
      "parapper://speech-request",
      "speechRequestReceived",
    );
    await register<AsrMissingEvent>("parapper://asr-missing", "asrMissing");
    await register<OscMuteStateEvent>(
      "parapper://osc-mute-state",
      "oscMuteStateChanged",
    );
    await register<ConnectionStateEvent>(
      "parapper://connection-state",
      "connectionStateChanged",
    );
    await register<ModelDownloadProgress>(
      "parapper://model-download-progress",
      "modelDownloadProgressed",
    );
    await register<ParapperErrorPayload>(
      "parapper://error",
      "applicationError",
    );
  } catch (error) {
    unsubscribes.forEach((unsubscribe) => unsubscribe());
    throw error;
  }

  return () => unsubscribes.forEach((unsubscribe) => unsubscribe());
};

const createAudioPlayer = () => {
  let audioContext: AudioContext | null = null;
  let audioSource: AudioBufferSourceNode | null = null;

  return async (samples: number[], sampleRate: number) => {
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
  };
};

export const createTauriFrontendServices = (
  dependencies: TauriDependencies = defaultDependencies,
): FrontendServices => {
  const playAudio = createAudioPlayer();

  return {
    config: {
      load: () => dependencies.invoke("get_config"),
      save: (config) => dependencies.invoke("save_config", { config }),
      reset: () => dependencies.invoke("reset_config"),
    },
    recognition: {
      status: () => dependencies.invoke("get_recognition_status"),
      start: () => dependencies.invoke("start_recognition"),
      stop: () => dependencies.invoke("stop_recognition"),
    },
    events: {
      subscribe: (listener) =>
        subscribeToFrontendEvents(dependencies, listener),
    },
    models: {
      status: () => dependencies.invoke("get_model_status"),
      hasAnyInstalled: () => dependencies.invoke("has_any_model_installed"),
      download: (config) => dependencies.invoke("download_models", { config }),
      downloadLocalTranslation: (model) =>
        dependencies.invoke("download_local_translation_model", { model }),
      isLocalTranslationInstalled: (model) =>
        dependencies.invoke("get_local_translation_model_installed", { model }),
    },
    audioDevices: {
      inputDevices: () => dependencies.invoke("get_audio_devices"),
      outputDevices: () => dependencies.invoke("get_output_audio_devices"),
      requestLoopbackPermission: () =>
        dependencies.invoke("request_loopback_audio_permission"),
      openLoopbackPermissionSettings: () =>
        dependencies.invoke("open_system_audio_permission_settings"),
    },
    presets: {
      list: () => dependencies.invoke("get_config_presets"),
      save: (name, config) =>
        dependencies.invoke("save_config_preset", { name, config }),
      delete: (name) => dependencies.invoke("delete_config_preset", { name }),
    },
    connections: {
      checkNeo: (neoHttpEnabled, neoHttpPort) =>
        dependencies.invoke("check_neo_http_available", {
          neoHttpEnabled,
          neoHttpPort,
        }),
      checkVrchat: (vrcOscMicmute) =>
        dependencies.invoke("check_vrchat_oscquery_available", {
          vrcOscMicmute,
        }),
      findNeoPort: () => dependencies.invoke("find_neo_http_port"),
      findYncPluginPort: () => dependencies.invoke("find_ync_plugin_http_port"),
      fetchNeoVoices: (port) =>
        dependencies.invoke("fetch_neo_voice_list", { port }),
    },
    speech: {
      stop: (port) => dependencies.invoke("neo_speech_stop", { port }),
    },
    hotwordReadings: {
      suggest: (surface) =>
        dependencies.invoke("suggest_hotword_readings", { surface }),
    },
    translationServer: {
      status: () => dependencies.invoke("get_translation_http_listener_status"),
      start: (port, localModel) =>
        dependencies.invoke("start_translation_http_listener", {
          port,
          localModel,
        }),
      stop: () => dependencies.invoke("stop_translation_http_listener"),
    },
    system: {
      saveRecognitionCsv: ({ defaultFileName, content }) =>
        dependencies.invoke("save_recognition_csv", {
          defaultFileName,
          content,
        }),
      saveAsrInputWav: ({ defaultFileName, content }) =>
        dependencies.invoke("save_asr_input_wav", {
          defaultFileName,
          content,
        }),
      openExternalUrl: async (url) => {
        try {
          await dependencies.invoke("open_external_url", { url });
        } catch (error) {
          console.warn("Failed to open external URL through Tauri", error);
          window.open(url, "_blank", "noopener,noreferrer");
        }
      },
      playAudio,
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

const nativeConnectionsAvailable = () => {
  const platform = navigator.platform.toLowerCase();
  const userAgent = navigator.userAgent.toLowerCase();
  return !platform.includes("mac") && !userAgent.includes("mac os x");
};

export const desktopCapabilities = (): FrontendCapabilities => ({
  localAudioDevices: true,
  outputAudioDevices: true,
  modelManagement: true,
  systemAudioPermission: true,
  externalConnectionProbe: nativeConnectionsAvailable(),
  fileExport: true,
  recognitionControl: true,
  speechControl: nativeConnectionsAvailable(),
  localTranslationServer: true,
});
