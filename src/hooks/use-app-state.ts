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
  ModelDownloadProgress,
  ModelStatus,
  OscMuteStateEvent,
  ParapperConfig,
  ParapperErrorPayload,
  RecognizedTextEvent,
  RecognitionStatus,
  VadStateEvent,
} from "../lib/types";

export type RuntimeState = {
  status: RecognitionStatus;
  running: boolean;
  inputLevel: number;
  vadState: VadStateEvent | null;
  asrWarning: string | null;
  lastError: ParapperErrorPayload | null;
  oscMuted: boolean | null;
  neoNotFound: boolean;
  vrcNotFound: boolean;
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
  inputLevel: 0,
  vadState: null,
  asrWarning: null,
  lastError: null,
  oscMuted: null,
  neoNotFound: false,
  vrcNotFound: false,
};

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
  const [audioDevices, setAudioDevices] = useState<AudioDeviceInfo[]>([]);
  const [refreshingAudioDevices, setRefreshingAudioDevices] = useState(false);
  const [recognizedTexts, setRecognizedTexts] = useState<RecognizedTextEvent[]>(
    [],
  );
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
      message: t(`notifications.connectionNotFound.${target}.message`),
      color: notificationColor.warn,
    });
    if (detail) console.warn(detail);
  };

  const loadAudioDevices = useCallback(async () => {
    setRefreshingAudioDevices(true);
    try {
      const loadedAudioDevices =
        await invoke<AudioDeviceInfo[]>("get_audio_devices");
      setAudioDevices(loadedAudioDevices);
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
      const loadedConfig = await invoke<ParapperConfig>("get_config");
      await loadAudioDevices();
      const loadedStatus = await invoke<RecognitionStatus>(
        "get_recognition_status",
      );
      const loadedModelStatus = await invoke<ModelStatus>("get_model_status");
      const hasAnyModelInstalled = await invoke<boolean>(
        "has_any_model_installed",
      );
      setConfig(loadedConfig);
      setAppliedConfig(loadedConfig);
      setRuntime((current) => ({
        ...current,
        status: loadedStatus,
        running: loadedStatus === "listening",
      }));
      setModel((current) => ({ ...current, status: loadedModelStatus }));
      setOnboarding((current) => ({
        ...current,
        open: !hasAnyModelInstalled,
      }));
    })();
  }, [loadAudioDevices, setAppliedConfig, setConfig]);

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

    if (nativeConnectionsDisabled) {
      applyConnectionState("neo", true);
      applyConnectionState("vrchat", true);
      return;
    }

    if (config.neo_http_enabled) {
      void invoke<boolean>("check_neo_http_available", {
        neoHttpEnabled: config.neo_http_enabled,
        neoHttpPort: config.neo_http_port,
      }).then((found) => applyConnectionState("neo", found, null, false));
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
      listen<number>("parapper://input-level", (event) => {
        setRuntime((current) => ({
          ...current,
          inputLevel: Math.min(1, Math.max(0, event.payload)),
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
            [...texts, event.payload],
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

  const downloadSelectedModels = async () => {
    if (!config) return null;

    setModel((current) => ({
      ...current,
      downloading: true,
      progress: null,
    }));
    try {
      const downloaded = await invoke<ModelStatus>("download_models", {
        config,
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
    setModel,
    ui,
    setUi,
    onboarding,
    setOnboarding,
    audioDevices,
    refreshingAudioDevices,
    recognizedTexts,
    setRecognizedTexts,
    refreshAudioDevices,
    downloadSelectedModels,
  };
};
