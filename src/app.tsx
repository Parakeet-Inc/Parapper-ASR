import { Group, Select, Stack, Text, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { OnboardingModal } from "./components/onboarding-modal";
import { RuntimePanel } from "./components/runtime-panel";
import { SettingsPanel } from "./components/settings-panel";
import { StatusBadges } from "./components/status-badges";
import { useAppState } from "./hooks/use-app-state";
import { useConfigState } from "./hooks/use-config-state";
import { availableLanguages, normalizeLanguage } from "./i18n";
import {
  asrModelOption,
  asrModelOptions,
  asrPrecisionOptions,
} from "./lib/constants";
import { notificationColor } from "./lib/theme";
import type { ParapperConfig } from "./lib/types";

export const App: React.FC = () => {
  const { i18n, t } = useTranslation();
  const {
    config,
    setConfig,
    setAppliedConfig,
    configRef,
    updateConfig,
    resetConfig: resetConfigInner,
    applyAsrModel,
  } = useConfigState(t);
  const {
    runtime,
    setRuntime,
    model,
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
  } = useAppState({
    config,
    configRef,
    setConfig,
    setAppliedConfig,
    t,
  });

  const languageOptions = useMemo(
    () =>
      availableLanguages.map((language) => ({
        label: t(language.labelKey),
        value: language.code,
      })),
    [t],
  );

  const currentLanguage = normalizeLanguage(
    i18n.resolvedLanguage ?? i18n.language,
  );
  const dateTimeLocale = currentLanguage === "en" ? "en-US" : "ja-JP";

  if (!config) {
    return (
      <Stack w="100vw" h="100vh" align="center" justify="center">
        <Text>{t("app.loading")}</Text>
      </Stack>
    );
  }

  const applyAudioDeviceConfig = async (nextConfig: ParapperConfig) => {
    setConfig(nextConfig);
    try {
      const saved = await invoke<ParapperConfig>("save_config", {
        config: nextConfig,
      });
      setConfig(saved);
      setAppliedConfig(saved);
    } catch (error) {
      notifications.show({
        title: t("notifications.audioDeviceSaveFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    }
  };

  const selectedAsrModelOption = asrModelOption(config.asr_model);
  const asrModelSelectOptions = asrModelOptions.map(({ labelKey, value }) => ({
    label: t(labelKey),
    value,
  }));
  const selectedAsrPrecisionOptions = asrPrecisionOptions.filter((option) =>
    selectedAsrModelOption.supportedPrecisions.includes(option.value),
  );
  const modelsMissing =
    model.status?.vad.installed === false ||
    model.status?.asr.installed === false;
  const canStartRecognition = !modelsMissing;

  return (
    <Stack w="100vw" h="100vh" p="lg" gap="md">
      <Group
        align="center"
        wrap="nowrap"
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          alignItems: "center",
        }}
      >
        <Stack gap={2}>
          <Title order={2}>Parapper</Title>
        </Stack>
        <Group gap="xs" wrap="nowrap">
          <Text size="sm" fw={500}>
            language
          </Text>
          <Select
            aria-label="language"
            data={languageOptions}
            value={currentLanguage}
            allowDeselect={false}
            size="xs"
            w={120}
            onChange={(value) => {
              if (value) {
                void i18n.changeLanguage(normalizeLanguage(value));
              }
            }}
          />
        </Group>
        <StatusBadges runtime={runtime} />
      </Group>

      <OnboardingModal
        onboarding={onboarding}
        config={config}
        languageOptions={languageOptions}
        currentLanguage={currentLanguage}
        asrModelSelectOptions={asrModelSelectOptions}
        downloadingModels={model.downloading}
        modelDownloadProgress={model.progress}
        onClose={() =>
          setOnboarding((current) => ({ ...current, open: false }))
        }
        onBack={() => setOnboarding((current) => ({ ...current, step: 0 }))}
        onNext={() => setOnboarding((current) => ({ ...current, step: 1 }))}
        onLanguageChange={(language) =>
          void i18n.changeLanguage(normalizeLanguage(language))
        }
        onApplyAsrModel={applyAsrModel}
        onDownloadSelectedModels={downloadSelectedModels}
      />

      <Group
        align="stretch"
        wrap="nowrap"
        gap="md"
        style={{ flex: 1, minHeight: 0, overflow: "hidden" }}
      >
        <SettingsPanel
          config={config}
          settingsOpen={ui.settingsOpen}
          settingsTab={ui.settingsTab}
          running={runtime.running}
          selectedAsrPrecisionOptions={selectedAsrPrecisionOptions}
          asrModelSelectOptions={asrModelSelectOptions}
          modelStatus={model.status}
          downloadingModels={model.downloading}
          modelDownloadProgress={model.progress}
          onSettingsOpenChange={(settingsOpen) =>
            setUi((current) => ({ ...current, settingsOpen }))
          }
          onSettingsTabChange={(settingsTab) =>
            setUi((current) => ({ ...current, settingsTab }))
          }
          onUpdateConfig={updateConfig}
          onApplyAsrModel={applyAsrModel}
          onDownloadSelectedModels={() => void downloadSelectedModels()}
          onResetConfig={resetConfigInner}
        />

        <RuntimePanel
          config={config}
          audioDevices={audioDevices}
          recognizedTexts={recognizedTexts}
          runtime={runtime}
          setRuntime={setRuntime}
          refreshingAudioDevices={refreshingAudioDevices}
          dateTimeLocale={dateTimeLocale}
          canStartRecognition={canStartRecognition}
          onClearRecognizedTexts={() => setRecognizedTexts([])}
          onRefreshAudioDevices={() => void refreshAudioDevices()}
          onApplyAudioDeviceConfig={(nextConfig) =>
            void applyAudioDeviceConfig(nextConfig)
          }
        />
      </Group>
    </Stack>
  );
};
