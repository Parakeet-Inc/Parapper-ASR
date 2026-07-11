import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { TFunction } from "i18next";
import type { Dispatch, SetStateAction } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  configuredLimit,
  trimRecognizedTextLog,
} from "../components/ui/display";
import {
  DEFAULT_DEBUG_AUDIO_LOG_LIMIT,
  DEFAULT_RECOGNITION_LOG_LIMIT,
} from "../lib/constants";
import {
  normalizeParapperErrorPayload,
  notifyParapperIssue,
} from "../lib/error";
import { isMacOs } from "../lib/platform";
import { notificationColor } from "../lib/theme";
import type {
  AsrMissingEvent,
  AudioDeviceInfo,
  ConnectionStateEvent,
  InputLevelEvent,
  ModelDownloadProgress,
  ModelStatus,
  OscMuteStateEvent,
  ParapperConfig,
  ParapperErrorPayload,
  RecognizedTextEvent,
  RecognitionStatus,
  SpeechRequestEvent,
  TranslationTextEvent,
  VadStateEvent,
} from "../lib/types";

export type RuntimeState = {
  status: RecognitionStatus;
  running: boolean;
  starting: boolean;
  inputLevel: number;
  inputLevelBeforeGain: number;
  vadState: VadStateEvent | null;
  asrWarning: string | null;
  lastError: ParapperErrorPayload | null;
  oscMuted: boolean | null;
  neoNotFound: boolean;
  vrcNotFound: boolean;
  translationSpeechDelaySuspected: boolean;
};

export type ModelState = {
  status: ModelStatus | null;
  downloading: boolean;
  progress: ModelDownloadProgress | null;
};

export type UiState = {
  settingsOpen: boolean;
  settingsTab: string | null;
};

export type OnboardingState = {
  open: boolean;
  step: number;
};

const initialRuntimeState: RuntimeState = {
  status: "idle",
  running: false,
  starting: false,
  inputLevel: 0,
  inputLevelBeforeGain: 0,
  vadState: null,
  asrWarning: null,
  lastError: null,
  oscMuted: null,
  neoNotFound: false,
  vrcNotFound: false,
  translationSpeechDelaySuspected: false,
};

const recognitionIsRunning = (status: RecognitionStatus) =>
  status === "waiting_for_client" ||
  status === "listening" ||
  status === "draining";

const TRANSLATION_SPEECH_DELAY_WARNING_MS = 3000;

const initialModelState: ModelState = {
  status: null,
  downloading: false,
  progress: null,
};

const initialUiState: UiState = {
  settingsOpen: false,
  settingsTab: "connection",
};

const initialOnboardingState: OnboardingState = {
  open: false,
  step: 0,
};

type UseAppStateParams = {
  config: ParapperConfig | null;
  configRef: { current: ParapperConfig | null };
  setConfig: Dispatch<SetStateAction<ParapperConfig | null>>;
  setAppliedConfig: Dispatch<SetStateAction<ParapperConfig | null>>;
  t: TFunction;
};

export const useAppState = ({
  config,
  configRef,
  setConfig,
  setAppliedConfig,
  t,
}: UseAppStateParams) => {
  const [runtime, setRuntime] = useState<RuntimeState>(initialRuntimeState);
  const [model, setModel] = useState<ModelState>(initialModelState);
  const [ui, setUi] = useState<UiState>(initialUiState);
  const [onboarding, setOnboarding] = useState<OnboardingState>(
    initialOnboardingState,
  );
  const [inputAudioDevices, setInputAudioDevices] = useState<AudioDeviceInfo[]>(
    [],
  );
  const [outputAudioDevices, setOutputAudioDevices] = useState<
    AudioDeviceInfo[]
  >([]);
  const [refreshingAudioDevices, setRefreshingAudioDevices] = useState(false);
  const [recognizedTexts, setRecognizedTexts] = useState<RecognizedTextEvent[]>(
    [],
  );
  const [translatedTexts, setTranslatedTexts] = useState<
    TranslationTextEvent[]
  >([]);
  const nativeConnectionsDisabled = isMacOs();
  const notifiedMissingTargetsRef = useRef<Set<ConnectionStateEvent["target"]>>(
    new Set(),
  );

  const applyConnectionState = (
    target: ConnectionStateEvent["target"],
    found: boolean,
    detail?: string | null,
    clearOnFound = true,
  ) => {
    if (found && !clearOnFound) {
      return;
    }

    setRuntime((current) => ({
      ...current,
      neoNotFound: target === "neo" ? !found : current.neoNotFound,
      vrcNotFound: target === "vrchat" ? !found : current.vrcNotFound,
    }));

    if (found) {
      notifiedMissingTargetsRef.current.delete(target);
      return;
    }

    if (notifiedMissingTargetsRef.current.has(target)) {
      return;
    }
    notifiedMissingTargetsRef.current.add(target);
    notifications.show({
      title: t(`notifications.connectionNotFound.${target}.title`),
      message:
        detail ?? t(`notifications.connectionNotFound.${target}.message`),
      color: notificationColor.warn,
    });
    if (detail) console.warn(detail);
  };

  const loadAudioDevices = useCallback(async () => {
    setRefreshingAudioDevices(true);
    try {
      const [loadedInputAudioDevices, loadedOutputAudioDevices] =
        await Promise.all([
          invoke<AudioDeviceInfo[]>("get_audio_devices"),
          invoke<AudioDeviceInfo[]>("get_output_audio_devices"),
        ]);
      setInputAudioDevices(loadedInputAudioDevices);
      setOutputAudioDevices(loadedOutputAudioDevices);
    } finally {
      setRefreshingAudioDevices(false);
    }
  }, []);

  const refreshAudioDevices = useCallback(async () => {
    try {
      await loadAudioDevices();
    } catch (error) {
      notifications.show({
        title: t("notifications.audioDeviceRefreshFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    }
  }, [loadAudioDevices, t]);

  useEffect(() => {
    void (async () => {
      try {
        const loadedConfig = await invoke<ParapperConfig>("get_config");
        setConfig(loadedConfig);
        setAppliedConfig(loadedConfig);

        await loadAudioDevices().catch((error) => {
          notifications.show({
            title: t("notifications.audioDeviceRefreshFailed.title"),
            message: String(error),
            color: notificationColor.error,
          });
        });

        const loadedStatus = await invoke<RecognitionStatus>(
          "get_recognition_status",
        );
        setRuntime((current) => ({
          ...current,
          status: loadedStatus,
          running: recognitionIsRunning(loadedStatus),
          starting: false,
        }));

        const loadedModelStatus = await invoke<ModelStatus>("get_model_status");
        setModel((current) => ({ ...current, status: loadedModelStatus }));

        const hasAnyModelInstalled = await invoke<boolean>(
          "has_any_model_installed",
        );
        setOnboarding((current) => ({
          ...current,
          open: !hasAnyModelInstalled,
        }));
      } catch (error) {
        const payload = normalizeParapperErrorPayload(error);
        setRuntime((current) => ({ ...current, lastError: payload }));
        notifyParapperIssue(payload);
      }
    })();
  }, [loadAudioDevices, setAppliedConfig, setConfig, t]);

  useEffect(() => {
    configRef.current = config;
  }, [config, configRef]);

  useEffect(() => {
    if (!config) return;
    void invoke<ModelStatus>("get_model_status").then((status) =>
      setModel((current) => ({ ...current, status })),
    );
  }, [config]);

  useEffect(() => {
    if (!config) return;
    setRecognizedTexts((texts) =>
      trimRecognizedTextLog(
        texts,
        config.recognition_log_limit,
        config.debug_audio_log_limit,
      ),
    );
  }, [config]);

  useEffect(() => {
    if (!config) return;

    if (!canNeoMainTextDelayTranslationSpeech(config)) {
      setRuntime((current) => ({
        ...current,
        translationSpeechDelaySuspected: false,
      }));
    }

    if (nativeConnectionsDisabled) {
      applyConnectionState("neo", true);
      applyConnectionState("vrchat", true);
      return;
    }

    if (config.neo_http_enabled) {
      void invoke<boolean>("check_neo_http_available", {
        neoHttpEnabled: config.neo_http_enabled,
        neoHttpPort: config.neo_http_port,
      }).then(async (found) => {
        const detectedPort = found
          ? null
          : await invoke<number | null>("find_neo_http_port").catch(() => null);
        const detail =
          detectedPort && detectedPort !== config.neo_http_port
            ? t("notifications.connectionNotFound.neo.detectedPortMessage", {
                configuredPort: config.neo_http_port,
                detectedPort,
              })
            : null;
        applyConnectionState("neo", found, detail, false);
      });
    } else {
      applyConnectionState("neo", true);
    }

    if (config.vrc_osc_micmute) {
      void invoke<boolean>("check_vrchat_oscquery_available", {
        vrcOscMicmute: config.vrc_osc_micmute,
      }).then((found) => applyConnectionState("vrchat", found, null, false));
    } else {
      applyConnectionState("vrchat", true);
    }
  }, [
    config?.neo_http_enabled,
    config?.neo_http_port,
    config?.vrc_osc_micmute,
    nativeConnectionsDisabled,
  ]);

  useEffect(() => {
    const unlistenCallbacks = [
      listen<RecognitionStatus>("parapper://status", (event) => {
        setRuntime((current) => ({
          ...current,
          status: event.payload,
          running: recognitionIsRunning(event.payload),
          starting: false,
        }));
      }),
      listen<InputLevelEvent | number>("parapper://input-level", (event) => {
        const level = parseInputLevelEvent(event.payload);
        setRuntime((current) => ({
          ...current,
          inputLevel: Math.max(0, level.postGain),
          inputLevelBeforeGain: Math.max(0, level.preGain),
        }));
      }),
      listen<VadStateEvent>("parapper://vad-state", (event) => {
        setRuntime((current) => ({
          ...current,
          vadState: event.payload,
        }));
      }),
      listen<RecognizedTextEvent>("parapper://recognized-text", (event) => {
        const eventConfig = configRef.current;
        setRecognizedTexts((texts) =>
          trimRecognizedTextLog(
            upsertRecognizedText(texts, event.payload),
            configuredLimit(
              eventConfig?.recognition_log_limit,
              DEFAULT_RECOGNITION_LOG_LIMIT,
            ),
            configuredLimit(
              eventConfig?.debug_audio_log_limit,
              DEFAULT_DEBUG_AUDIO_LOG_LIMIT,
            ),
          ),
        );
      }),
      listen<TranslationTextEvent>("parapper://translated-text", (event) => {
        setTranslatedTexts((texts) =>
          upsertTranslatedText(texts, event.payload),
        );
      }),
      listen<SpeechRequestEvent>("parapper://speech-request", (event) => {
        const eventConfig = configRef.current;
        if (
          event.payload.source_kind !== "translation" ||
          event.payload.status !== "accepted" ||
          event.payload.elapsed_millis < TRANSLATION_SPEECH_DELAY_WARNING_MS ||
          !eventConfig ||
          !canNeoMainTextDelayTranslationSpeech(eventConfig)
        ) {
          return;
        }
        setRuntime((current) => ({
          ...current,
          translationSpeechDelaySuspected: true,
        }));
      }),
      listen<AsrMissingEvent>("parapper://asr-missing", (event) => {
        setRuntime((current) => ({
          ...current,
          asrWarning: event.payload.reason,
        }));
      }),
      listen<OscMuteStateEvent>("parapper://osc-mute-state", (event) => {
        setRuntime((current) => ({
          ...current,
          oscMuted: event.payload.muted,
        }));
      }),
      listen<ConnectionStateEvent>("parapper://connection-state", (event) => {
        const { target, found, detail } = event.payload;
        applyConnectionState(target, found, detail);
      }),
      listen<ModelDownloadProgress>(
        "parapper://model-download-progress",
        (event) => {
          setModel((current) => ({
            ...current,
            progress: event.payload,
          }));
        },
      ),
      listen<ParapperErrorPayload>("parapper://error", (event) => {
        const payload = normalizeParapperErrorPayload(event.payload);
        setRuntime((current) => ({ ...current, lastError: payload }));
        notifyParapperIssue(payload);
      }),
    ];

    return () => {
      void Promise.all(unlistenCallbacks).then((callbacks) => {
        callbacks.forEach((unlisten) => unlisten());
      });
    };
  }, [configRef, t]);

  const downloadSelectedModels = async (downloadConfig = config) => {
    if (!downloadConfig) return null;

    setModel((current) => ({
      ...current,
      downloading: true,
      progress: null,
    }));
    try {
      const downloaded = await invoke<ModelStatus>("download_models", {
        config: downloadConfig,
      });
      setModel((current) => ({ ...current, status: downloaded }));
      setRuntime((current) => ({ ...current, asrWarning: null }));
      notifications.show({
        title: t("notifications.modelsPrepared.title"),
        message: downloaded.root_dir,
      });
      return downloaded;
    } finally {
      setModel((current) => ({ ...current, downloading: false }));
    }
  };

  return {
    runtime,
    setRuntime,
    model,
    ui,
    setUi,
    onboarding,
    setOnboarding,
    inputAudioDevices,
    outputAudioDevices,
    refreshingAudioDevices,
    recognizedTexts,
    setRecognizedTexts,
    translatedTexts,
    setTranslatedTexts,
    refreshAudioDevices,
    downloadSelectedModels,
  };
};

const canNeoMainTextDelayTranslationSpeech = (config: ParapperConfig) =>
  config.neo_http_enabled &&
  config.translation_enabled &&
  config.translation_mappings.length > 0 &&
  config.speech_mappings.some(
    (mapping) =>
      !mapping.muted &&
      mapping.source_kind === "translation" &&
      mapping.backend === "ync" &&
      mapping.talker.trim() !== "",
  );

const parseInputLevelEvent = (
  payload: InputLevelEvent | number,
): { preGain: number; postGain: number } => {
  if (typeof payload === "number") {
    return {
      preGain: payload,
      postGain: payload,
    };
  }
  return {
    preGain: payload.pre_gain_level,
    postGain: payload.post_gain_level,
  };
};

const upsertRecognizedText = (
  texts: RecognizedTextEvent[],
  event: RecognizedTextEvent,
) => {
  if (event.update_mode !== "replace") {
    return [...texts, event];
  }

  const index = texts.findIndex((text) =>
    sameRecognitionSource(text.source, event.source),
  );
  if (index < 0) {
    return [...texts, event];
  }

  const current = texts[index];
  if (!shouldReplaceRecognitionEvent(current, event)) {
    return texts;
  }

  return texts.map((text, currentIndex) =>
    currentIndex === index ? event : text,
  );
};

const upsertTranslatedText = (
  texts: TranslationTextEvent[],
  event: TranslationTextEvent,
) => {
  if (event.update_mode !== "replace") {
    return [...texts, event];
  }

  const index = texts.findIndex(
    (text) =>
      sameRecognitionSource(text.source, event.source) &&
      text.target_lang === event.target_lang,
  );
  if (index < 0) {
    return [...texts, event];
  }

  const current = texts[index];
  if (!shouldReplaceRecognitionEvent(current, event)) {
    return texts;
  }

  return texts.map((text, currentIndex) =>
    currentIndex === index ? event : text,
  );
};

const sameRecognitionSource = (
  left: { turn_session_id: number; turn_id: number },
  right: { turn_session_id: number; turn_id: number },
) =>
  left.turn_session_id === right.turn_session_id &&
  left.turn_id === right.turn_id;

const shouldReplaceRecognitionEvent = (
  current: {
    source: {
      turn_revision: number;
      output_sequence: number;
    };
    is_final: boolean;
  },
  incoming: {
    source: {
      turn_revision: number;
      output_sequence: number;
    };
    is_final: boolean;
  },
) => {
  if (current.is_final && !incoming.is_final) {
    return false;
  }
  if (incoming.source.turn_revision !== current.source.turn_revision) {
    return incoming.source.turn_revision > current.source.turn_revision;
  }
  if (incoming.source.output_sequence !== current.source.output_sequence) {
    return incoming.source.output_sequence > current.source.output_sequence;
  }
  return incoming.is_final || !current.is_final;
};
