import { notifications } from "@mantine/notifications";
import type { TFunction } from "i18next";
import type { MutableRefObject } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import { upsertRecognizedText } from "../application/app-state/recognized-text";
import {
  applyConnectionAvailability,
  applyInputLevel,
  applyModelDownloadProgress,
  applyModelStatus,
  applyRecognitionStatus,
  initialModelState,
  initialRuntimeState,
  setModelDownloading,
} from "../application/app-state/reducer";
import type {
  ModelState,
  RuntimeState,
} from "../application/app-state/reducer";
import { upsertTranslatedText } from "../application/app-state/translated-text";
import type {
  FrontendCapabilities,
  FrontendEvent,
  FrontendServices,
} from "../application/frontend-services";
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
import { notificationColor } from "../lib/theme";
import type {
  AudioDeviceInfo,
  ConnectionStateEvent,
  InputLevelEvent,
  LocalTranslationModel,
  ParapperConfig,
  RecognizedTextEvent,
  TranslationTextEvent,
} from "../lib/types";

export type { RuntimeState } from "../application/app-state/reducer";

export type UiState = {
  settingsOpen: boolean;
  settingsTab: string | null;
};

export type OnboardingState = {
  open: boolean;
  step: number;
};

const TRANSLATION_SPEECH_DELAY_WARNING_MS = 3000;

const initialUiState: UiState = {
  settingsOpen: false,
  settingsTab: "connection",
};

const initialOnboardingState: OnboardingState = {
  open: false,
  step: 0,
};

type UseAppStateParams = {
  services: FrontendServices;
  capabilities: FrontendCapabilities;
  config: ParapperConfig | null;
  configRef: MutableRefObject<ParapperConfig | null>;
  setConfig: (config: ParapperConfig | null) => void;
  setAppliedConfig: (config: ParapperConfig | null) => void;
  t: TFunction;
};

export const useAppState = ({
  services,
  capabilities,
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
  const nativeConnectionsDisabled = !capabilities.externalConnectionProbe;
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

    setRuntime((current) =>
      applyConnectionAvailability(current, target, found),
    );

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
    if (!capabilities.localAudioDevices && !capabilities.outputAudioDevices) {
      setInputAudioDevices([]);
      setOutputAudioDevices([]);
      return;
    }
    setRefreshingAudioDevices(true);
    try {
      const [loadedInputAudioDevices, loadedOutputAudioDevices] =
        await Promise.all([
          capabilities.localAudioDevices
            ? services.audioDevices.inputDevices()
            : Promise.resolve([]),
          capabilities.outputAudioDevices
            ? services.audioDevices.outputDevices()
            : Promise.resolve([]),
        ]);
      setInputAudioDevices(loadedInputAudioDevices);
      setOutputAudioDevices(loadedOutputAudioDevices);
    } finally {
      setRefreshingAudioDevices(false);
    }
  }, [capabilities, services.audioDevices]);

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
        const loadedConfig = await services.config.load();
        setConfig(loadedConfig);
        setAppliedConfig(loadedConfig);

        await loadAudioDevices().catch((error) => {
          notifications.show({
            title: t("notifications.audioDeviceRefreshFailed.title"),
            message: String(error),
            color: notificationColor.error,
          });
        });

        const loadedStatus = await services.recognition.status();
        setRuntime((current) => applyRecognitionStatus(current, loadedStatus));

        if (capabilities.modelManagement) {
          const loadedModelStatus = await services.models.status();
          setModel((current) => applyModelStatus(current, loadedModelStatus));
        }

        const hasAnyModelInstalled = capabilities.modelManagement
          ? await services.models.hasAnyInstalled()
          : true;
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
  }, [
    capabilities.modelManagement,
    loadAudioDevices,
    services.config,
    services.models,
    services.recognition,
    setAppliedConfig,
    setConfig,
    t,
  ]);

  useEffect(() => {
    configRef.current = config;
  }, [config, configRef]);

  useEffect(() => {
    if (!config || !capabilities.modelManagement) return;
    void services.models
      .status()
      .then((status) =>
        setModel((current) => applyModelStatus(current, status)),
      );
  }, [capabilities.modelManagement, config, services.models]);

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
      void services.connections
        .checkNeo(config.neo_http_enabled, config.neo_http_port)
        .then(async (found) => {
          const detectedPort = found
            ? null
            : await services.connections.findNeoPort().catch(() => null);
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
      void services.connections
        .checkVrchat(config.vrc_osc_micmute)
        .then((found) => applyConnectionState("vrchat", found, null, false));
    } else {
      applyConnectionState("vrchat", true);
    }
  }, [
    config?.neo_http_enabled,
    config?.neo_http_port,
    config?.vrc_osc_micmute,
    nativeConnectionsDisabled,
    services.connections,
  ]);

  useEffect(() => {
    const applyEvent = (event: FrontendEvent) => {
      switch (event.type) {
        case "recognitionStatusChanged":
          setRuntime((current) =>
            applyRecognitionStatus(current, event.payload),
          );
          break;
        case "inputLevelChanged": {
          const level = parseInputLevelEvent(event.payload);
          setRuntime((current) =>
            applyInputLevel(
              current,
              level.preGain,
              level.postGain,
              level.sourceId,
            ),
          );
          break;
        }
        case "vadStateChanged":
          setRuntime((current) => ({ ...current, vadState: event.payload }));
          break;
        case "recognizedTextReceived": {
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
          break;
        }
        case "translationTextReceived":
          setTranslatedTexts((texts) =>
            upsertTranslatedText(texts, event.payload),
          );
          break;
        case "speechRequestReceived": {
          const eventConfig = configRef.current;
          if (
            event.payload.source_kind === "translation" &&
            event.payload.status === "accepted" &&
            event.payload.elapsed_millis >=
              TRANSLATION_SPEECH_DELAY_WARNING_MS &&
            eventConfig &&
            canNeoMainTextDelayTranslationSpeech(eventConfig)
          ) {
            setRuntime((current) => ({
              ...current,
              translationSpeechDelaySuspected: true,
            }));
          }
          break;
        }
        case "asrMissing":
          setRuntime((current) => ({
            ...current,
            asrWarning: event.payload.reason,
          }));
          break;
        case "oscMuteStateChanged":
          setRuntime((current) => ({
            ...current,
            oscMuted: event.payload.muted,
          }));
          break;
        case "connectionStateChanged": {
          const { target, found, detail } = event.payload;
          applyConnectionState(target, found, detail);
          break;
        }
        case "modelDownloadProgressed":
          setModel((current) =>
            applyModelDownloadProgress(current, event.payload),
          );
          break;
        case "applicationError": {
          const payload = normalizeParapperErrorPayload(event.payload);
          setRuntime((current) => ({ ...current, lastError: payload }));
          notifyParapperIssue(payload);
          break;
        }
      }
    };

    const unsubscribe = services.events.subscribe(applyEvent);

    return () => {
      void unsubscribe.then((stop) => stop());
    };
  }, [configRef, services.events, t]);

  const downloadSelectedModels = async (downloadConfig = config) => {
    if (!downloadConfig || !capabilities.modelManagement) return null;

    setModel((current) => setModelDownloading(current, true));
    try {
      const downloaded = await services.models.download(downloadConfig);
      setModel((current) => applyModelStatus(current, downloaded));
      setRuntime((current) => ({ ...current, asrWarning: null }));
      notifications.show({
        title: t("notifications.modelsPrepared.title"),
        message: downloaded.root_dir,
      });
      return downloaded;
    } finally {
      setModel((current) => setModelDownloading(current, false));
    }
  };

  const downloadLocalTranslationModel = async (
    localTranslationModel: LocalTranslationModel,
  ) => {
    if (!capabilities.modelManagement) return false;
    setModel((current) => setModelDownloading(current, true));
    try {
      const installed = await services.models.downloadLocalTranslation(
        localTranslationModel,
      );
      const status = await services.models.status();
      setModel((current) => applyModelStatus(current, status));
      return installed;
    } catch (error) {
      notifyParapperIssue(normalizeParapperErrorPayload(error));
      return false;
    } finally {
      setModel((current) => setModelDownloading(current, false));
    }
  };

  const refreshRecognitionStatus = async () => {
    const status = await services.recognition.status();
    setRuntime((current) => applyRecognitionStatus(current, status));
    return status;
  };

  const stopRecognition = async () => {
    try {
      const status = await services.recognition.stop();
      setRuntime((current) => applyRecognitionStatus(current, status));
    } catch (error) {
      const payload = normalizeParapperErrorPayload(error);
      setRuntime((current) => ({ ...current, lastError: payload }));
      notifyParapperIssue(payload);
      await refreshRecognitionStatus();
    }
  };

  const startRecognition = async () => {
    if (!capabilities.recognitionControl) return;
    setRuntime((current) => ({ ...current, starting: true }));
    try {
      const status = await services.recognition.start();
      setRuntime((current) => applyRecognitionStatus(current, status));
    } catch (error) {
      const payload = normalizeParapperErrorPayload(error);
      setRuntime((current) => ({
        ...current,
        running: false,
        starting: false,
        lastError: payload,
      }));
      notifyParapperIssue(payload);
      try {
        await services.recognition.stop();
      } finally {
        await refreshRecognitionStatus();
      }
    }
  };

  const stopSpeech = async (port: number) => {
    if (!capabilities.speechControl) return;
    try {
      await services.speech.stop(port);
    } catch (error) {
      notifications.show({
        title: t("notifications.speechStopFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    }
  };

  const ensureLoopbackPermission = async () => {
    if (!capabilities.systemAudioPermission) return;
    try {
      const granted = await services.audioDevices.requestLoopbackPermission();
      if (!granted) {
        notifications.show({
          title: t("notifications.loopbackPermissionRequired.title"),
          message: t("notifications.loopbackPermissionRequired.message"),
          color: notificationColor.warn,
        });
        await services.audioDevices.openLoopbackPermissionSettings();
      }
    } catch (error) {
      notifications.show({
        title: t("notifications.loopbackPermissionRequired.title"),
        message: String(error),
        color: notificationColor.error,
      });
    }
  };

  return {
    runtime,
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
    downloadLocalTranslationModel,
    startRecognition,
    stopRecognition,
    stopSpeech,
    ensureLoopbackPermission,
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
): { preGain: number; postGain: number; sourceId: string | null } => {
  if (typeof payload === "number") {
    return {
      preGain: payload,
      postGain: payload,
      sourceId: null,
    };
  }
  return {
    preGain: payload.pre_gain_level,
    postGain: payload.post_gain_level,
    sourceId: payload.source_id ?? null,
  };
};
