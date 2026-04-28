import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useRef, useState } from "react";

import { asrModelOption } from "../lib/constants";
import { notificationColor } from "../lib/theme";
import type { AsrModel, ParapperConfig } from "../lib/types";

export const useConfigState = (t: (key: string) => string) => {
  const [config, setConfig] = useState<ParapperConfig | null>(null);
  const [appliedConfig, setAppliedConfig] = useState<ParapperConfig | null>(
    null,
  );
  const configRef = useRef<ParapperConfig | null>(null);
  const saveQueueRef = useRef<Promise<void>>(Promise.resolve());
  const saveRevisionRef = useRef(0);

  const saveAppliedConfig = async (
    nextConfig: ParapperConfig,
    revision: number,
  ) => {
    const saveTask = saveQueueRef.current.then(() =>
      invoke<ParapperConfig>("save_config", {
        config: nextConfig,
      }),
    );
    // Keep later saves queued even when this save fails; callers still await saveTask for errors.
    saveQueueRef.current = saveTask.then(
      () => undefined,
      () => undefined,
    );
    const saved = await saveTask;
    if (revision === saveRevisionRef.current) {
      setConfig(saved);
      setAppliedConfig(saved);
    }
    return saved;
  };

  const applyConfig = (nextConfig: ParapperConfig) => {
    saveRevisionRef.current += 1;
    const revision = saveRevisionRef.current;
    setConfig(nextConfig);
    void saveAppliedConfig(nextConfig, revision).catch((error) => {
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
    applyConfig({ ...config, [key]: value });
  };

  const resetConfig = async () => {
    saveRevisionRef.current += 1;
    const revision = saveRevisionRef.current;
    const resetTask = saveQueueRef.current.then(() =>
      invoke<ParapperConfig>("reset_config"),
    );
    // Keep later saves queued even when this reset fails; callers still await resetTask for errors.
    saveQueueRef.current = resetTask.then(
      () => undefined,
      () => undefined,
    );
    const reset = await resetTask;
    if (revision === saveRevisionRef.current) {
      setConfig(reset);
      setAppliedConfig(reset);
    }
    notifications.show({
      title: t("notifications.configReset.title"),
      message: t("notifications.configReset.message"),
    });
    return reset;
  };

  const applyAsrModel = (asrModel: AsrModel) => {
    if (!config) return;
    const modelOption = asrModelOption(asrModel);
    applyConfig({
      ...config,
      asr_language: modelOption.language,
      asr_model: asrModel,
      asr_precision: modelOption.defaultPrecision,
    });
  };

  return {
    config,
    setConfig,
    appliedConfig,
    setAppliedConfig,
    configRef,
    updateConfig,
    applyConfig,
    resetConfig,
    applyAsrModel,
  };
};
