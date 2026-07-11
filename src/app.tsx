import { Group, Select, Stack, Text, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { OnboardingModal } from "./components/onboarding-modal";
import { RuntimePanel } from "./components/runtime-panel";
import { SettingsPanel } from "./components/settings-panel";
import { StatusBadges } from "./components/status-badges";
import { TranslationSidePanel } from "./components/translation-side-panel";
import { useAppState } from "./hooks/use-app-state";
import { useConfigState } from "./hooks/use-config-state";
import { availableLanguages, normalizeLanguage } from "./i18n";
import { zeroMinHeight } from "./lib/layout-styles";
import { notificationColor } from "./lib/theme";
import type { ConfigPreset, ParapperConfig } from "./lib/types";

export const App: React.FC = () => {
  const { i18n, t } = useTranslation();
  const {
    config,
    setConfig,
    setAppliedConfig,
    configRef,
    updateConfig,
    replaceConfig,
    resetConfig: resetConfigInner,
    applyAsrModel,
  } = useConfigState(t);
  const [configPresets, setConfigPresets] = useState<ConfigPreset[]>([]);
  const {
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
  useEffect(() => {
    void invoke<ConfigPreset[]>("get_config_presets")
      .then(setConfigPresets)
      .catch((error) => {
        notifications.show({
          title: t("notifications.configPresetsLoadFailed.title"),
          message: String(error),
          color: notificationColor.error,
        });
      });
  }, [t]);

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

  const saveConfigPreset = async (name: string) => {
    const presets = await invoke<ConfigPreset[]>("save_config_preset", {
      name,
      config,
    });
    setConfigPresets(presets);
    return presets;
  };

  const deleteConfigPreset = async (name: string) => {
    const presets = await invoke<ConfigPreset[]>("delete_config_preset", {
      name,
    });
    setConfigPresets(presets);
    return presets;
  };

  const applyPresetAndDownloadModels = async (presetConfig: ParapperConfig) => {
    replaceConfig(presetConfig);
    return downloadSelectedModels(presetConfig);
  };

  const modelsMissing =
    model.status?.vad.installed === false ||
    model.status?.asr.installed === false ||
    model.status?.japanese_morph?.installed === false ||
    model.status?.language_id?.installed === false ||
    model.status?.turn_detectors.some((status) => !status.installed) === true ||
    model.status?.tts.some((status) => !status.installed) === true ||
    model.status?.local_translation?.installed === false ||
    model.status?.noise_cancellation?.installed === false;
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
        languageOptions={languageOptions}
        currentLanguage={currentLanguage}
        configPresets={configPresets}
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
        onApplyPresetAndDownload={applyPresetAndDownloadModels}
      />

      <Group
        align="stretch"
        wrap="nowrap"
        gap="md"
        style={{ flex: 1, ...zeroMinHeight, overflow: "hidden" }}
      >
        <SettingsPanel
          config={config}
          outputAudioDevices={outputAudioDevices}
          settingsOpen={ui.settingsOpen}
          settingsTab={ui.settingsTab}
          running={runtime.running || runtime.starting}
          translationSpeechDelaySuspected={
            runtime.translationSpeechDelaySuspected
          }
          modelStatus={model.status}
          downloadingModels={model.downloading}
          modelDownloadProgress={model.progress}
          configPresets={configPresets}
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
          onSaveConfigPreset={saveConfigPreset}
          onDeleteConfigPreset={deleteConfigPreset}
          onApplyConfigPreset={replaceConfig}
        />

        <RuntimePanel
          config={config}
          inputAudioDevices={inputAudioDevices}
          recognizedTexts={recognizedTexts}
          runtime={runtime}
          setRuntime={setRuntime}
          refreshingAudioDevices={refreshingAudioDevices}
          translationPanel={
            config.translation_enabled ? (
              <TranslationSidePanel
                config={config}
                recognizedTexts={recognizedTexts}
                translatedTexts={translatedTexts}
              />
            ) : null
          }
          dateTimeLocale={dateTimeLocale}
          canStartRecognition={canStartRecognition}
          downloadingModels={model.downloading}
          canClearLogs={
            recognizedTexts.length > 0 || translatedTexts.length > 0
          }
          onClearRecognizedTexts={() => {
            setRecognizedTexts([]);
            setTranslatedTexts([]);
          }}
          onRefreshAudioDevices={() => void refreshAudioDevices()}
          onApplyAudioDeviceConfig={(nextConfig) =>
            void applyAudioDeviceConfig(nextConfig)
          }
          onUpdateConfig={updateConfig}
          onOpenModelDownload={() => {
            setUi((current) => ({
              ...current,
              settingsOpen: true,
              settingsTab: "connection",
            }));
            if (!model.downloading) {
              void downloadSelectedModels();
            }
          }}
        />
      </Group>
    </Stack>
  );
};
