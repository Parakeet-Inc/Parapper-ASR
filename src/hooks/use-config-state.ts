import { notifications } from "@mantine/notifications";
import { useCallback, useReducer, useRef } from "react";

import {
  configStateReducer,
  initialConfigState,
} from "../application/config-state/reducer";
import type { ConfigService } from "../application/frontend-services";
import { configWithAsrModel } from "../lib/asr-mode";
import { notificationColor } from "../lib/theme";
import type { AsrModel, ParapperConfig } from "../lib/types";

export const useConfigState = (
  service: ConfigService,
  t: (key: string) => string,
) => {
  const [state, dispatch] = useReducer(configStateReducer, initialConfigState);
  const config = state.current;
  const configRef = useRef<ParapperConfig | null>(null);
  const saveQueueRef = useRef<Promise<void>>(Promise.resolve());
  const saveRevisionRef = useRef(0);

  const setConfig = useCallback((nextConfig: ParapperConfig | null) => {
    dispatch({ type: "currentReplaced", config: nextConfig });
  }, []);

  const setAppliedConfig = useCallback((nextConfig: ParapperConfig | null) => {
    dispatch({ type: "appliedReplaced", config: nextConfig });
  }, []);

  const saveAppliedConfig = async (
    nextConfig: ParapperConfig,
    revision: number,
  ) => {
    const saveTask = saveQueueRef.current.then(() => service.save(nextConfig));
    // Keep later saves queued even when this save fails; callers still await saveTask for errors.
    saveQueueRef.current = saveTask.then(
      () => undefined,
      () => undefined,
    );
    const saved = await saveTask;
    // Every successful queued save becomes the backend's latest persisted
    // state, even while a newer optimistic revision remains visible.
    dispatch({ type: "saveCompleted", config: saved, revision });
    return saved;
  };

  const applyConfig = (nextConfig: ParapperConfig) => {
    saveRevisionRef.current += 1;
    const revision = saveRevisionRef.current;
    dispatch({ type: "optimisticUpdate", config: nextConfig, revision });
    void saveAppliedConfig(nextConfig, revision).catch((error) => {
      dispatch({ type: "saveFailed", revision });
      notifications.show({
        title: t("notifications.configSaveFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    });
  };

  const updateConfig = <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => {
    if (!config) return;
    const nextConfig = { ...config, [key]: value };
    applyConfig(nextConfig);
  };

  const replaceConfig = (nextConfig: ParapperConfig) => {
    applyConfig(nextConfig);
  };

  const resetConfig = async () => {
    saveRevisionRef.current += 1;
    const revision = saveRevisionRef.current;
    const resetTask = saveQueueRef.current.then(() => service.reset());
    // Keep later saves queued even when this reset fails; callers still await resetTask for errors.
    saveQueueRef.current = resetTask.then(
      () => undefined,
      () => undefined,
    );
    const reset = await resetTask;
    if (revision === saveRevisionRef.current) {
      dispatch({ type: "loaded", config: reset });
    }
    notifications.show({
      title: t("notifications.configReset.title"),
      message: t("notifications.configReset.message"),
    });
    return reset;
  };

  const applyAsrModel = (asrModel: AsrModel) => {
    if (!config) return;
    applyConfig(configWithAsrModel(config, asrModel));
  };

  return {
    config,
    setConfig,
    setAppliedConfig,
    configRef,
    updateConfig,
    replaceConfig,
    resetConfig,
    applyAsrModel,
  };
};
