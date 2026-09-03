import { Group, Select, Stack, Text, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import type {
  FrontendCapabilities,
  FrontendServices,
} from "./application/frontend-services";
import { OnboardingModal } from "./components/onboarding-modal";
import { RuntimePanel } from "./components/runtime-panel";
import { SettingsPanel } from "./components/settings-panel";
import { StatusBadges } from "./components/status-badges";
import { TranslationSidePanel } from "./components/translation-side-panel";
import { useAppState } from "./hooks/use-app-state";
import { useConfigState } from "./hooks/use-config-state";
import { useSttProfileSelection } from "./hooks/use-stt-profile-selection";
import { availableLanguages, normalizeLanguage } from "./i18n";
import { configWithDeveloperConnectionEnabled } from "./lib/developer-connection";
import { zeroMinHeight } from "./lib/layout-styles";
import {
  addSttProfile,
  deleteSttProfile,
  effectiveSttProfiles,
  setSttProfileEnabled,
  updateSttProfile,
} from "./lib/stt-profiles";
import { notificationColor } from "./lib/theme";
import type { ConfigPreset, ParapperConfig } from "./lib/types";

export const App: React.FC<{
  services: FrontendServices;
  capabilities: FrontendCapabilities;
}> = ({ services, capabilities }) => {
  const { i18n, t } = useTranslation();
  const {
    config,
    setConfig,
    setAppliedConfig,
    configRef,
    updateConfig,
    replaceConfig,
    resetConfig: resetConfigInner,
  } = useConfigState(services.config, t);
  const [configPresets, setConfigPresets] = useState<ConfigPreset[]>([]);
  const {
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
  } = useAppState({
    services,
    capabilities,
    config,
    configRef,
    setConfig,
    setAppliedConfig,
    t,
  });
  const sttProfiles = config ? effectiveSttProfiles(config) : [];
  const { selectedProfileId, selectProfile } =
    useSttProfileSelection(sttProfiles);

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
    void services.presets
      .list()
      .then(setConfigPresets)
      .catch((error) => {
        notifications.show({
          title: t("notifications.configPresetsLoadFailed.title"),
          message: String(error),
          color: notificationColor.error,
        });
      });
  }, [services.presets, t]);

  if (!config) {
    return (
      <Stack w="100vw" h="100vh" align="center" justify="center">
        <Text>{t("app.loading")}</Text>
      </Stack>
    );
  }

  const saveConfigPreset = async (name: string) => {
    const presets = await services.presets.save(name, config);
    setConfigPresets(presets);
    return presets;
  };

  const deleteConfigPreset = async (name: string) => {
    const presets = await services.presets.delete(name);
    setConfigPresets(presets);
    return presets;
  };

  const applyPresetAndDownloadModels = async (presetConfig: ParapperConfig) => {
    replaceConfig(presetConfig);
    return downloadSelectedModels(presetConfig);
  };

  const modelsMissing =
    capabilities.modelManagement &&
    (model.status?.vad.installed === false ||
      model.status?.asr.installed === false ||
      model.status?.japanese_morph?.installed === false ||
      model.status?.language_id?.installed === false ||
      model.status?.turn_detectors.some((status) => !status.installed) ===
        true ||
      model.status?.tts.some((status) => !status.installed) === true ||
      model.status?.local_translation?.installed === false ||
      model.status?.noise_cancellation?.installed === false);
  const canStartRecognition = !modelsMissing;
  const runtimeLocked = runtime.running || runtime.starting;
  const updateProfile = (
    profileId: string,
    update: (
      profile: (typeof sttProfiles)[number],
    ) => (typeof sttProfiles)[number],
  ) => replaceConfig(updateSttProfile(config, profileId, update));
  const addProfile = () => {
    if (runtimeLocked || !selectedProfileId) return;
    const nextConfig = addSttProfile(
      config,
      selectedProfileId,
      inputAudioDevices,
    );
    if (!nextConfig) {
      notifications.show({
        title: t("notifications.sttProfileAddFailed.title"),
        message: t("notifications.sttProfileAddFailed.message"),
        color: notificationColor.warn,
      });
      return;
    }
    const added = effectiveSttProfiles(nextConfig).at(-1);
    replaceConfig(nextConfig);
    if (added) selectProfile(added.id);
  };
  const removeProfile = (profileId: string) => {
    if (runtimeLocked) return;
    const removed = deleteSttProfile(config, profileId);
    if (!removed) return;
    replaceConfig(removed.config);
    selectProfile(removed.selectedProfileId);
  };
  const setProfileEnabled = (profileId: string, enabled: boolean) => {
    if (runtimeLocked) return;
    replaceConfig(setSttProfileEnabled(config, profileId, enabled));
  };

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
        <StatusBadges
          runtime={runtime}
          nativeConnectionsAvailable={capabilities.externalConnectionProbe}
        />
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
          capabilities={capabilities}
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
          sttProfiles={sttProfiles}
          selectedSttProfileId={selectedProfileId ?? ""}
          inputAudioDevices={inputAudioDevices}
          refreshingAudioDevices={refreshingAudioDevices}
          onSettingsOpenChange={(settingsOpen) =>
            setUi((current) => ({ ...current, settingsOpen }))
          }
          onSettingsTabChange={(settingsTab) =>
            setUi((current) => ({ ...current, settingsTab }))
          }
          onUpdateConfig={updateConfig}
          onSuggestHotwordReadings={services.hotwordReadings.suggest}
          onDownloadSelectedModels={() => void downloadSelectedModels()}
          onDownloadLocalTranslationModel={downloadLocalTranslationModel}
          onResetConfig={resetConfigInner}
          onSaveConfigPreset={saveConfigPreset}
          onDeleteConfigPreset={deleteConfigPreset}
          onApplyConfigPreset={replaceConfig}
          onSelectSttProfile={selectProfile}
          onAddSttProfile={addProfile}
          onDeleteSttProfile={removeProfile}
          onUpdateSttProfile={updateProfile}
          onSetSttProfileEnabled={setProfileEnabled}
          onSetDeveloperConnectionEnabled={(enabled) =>
            replaceConfig(configWithDeveloperConnectionEnabled(config, enabled))
          }
          onRefreshAudioDevices={() => void refreshAudioDevices()}
          onRequestLoopbackPermission={ensureLoopbackPermission}
          onFindNeoPort={services.connections.findNeoPort}
          onFindYncPluginPort={services.connections.findYncPluginPort}
          onFetchNeoVoices={services.connections.fetchNeoVoices}
          onGetTranslationServerStatus={services.translationServer.status}
          onGetLocalTranslationInstalled={
            services.models.isLocalTranslationInstalled
          }
          onStartTranslationServer={services.translationServer.start}
          onStopTranslationServer={services.translationServer.stop}
          onOpenExternalUrl={services.system.openExternalUrl}
          onLoadRustLicenses={services.system.loadRustLicenses}
        />

        <RuntimePanel
          config={config}
          profiles={sttProfiles}
          recognizedTexts={recognizedTexts}
          runtime={runtime}
          translationPanel={
            config.translation_enabled ? (
              <TranslationSidePanel
                config={config}
                recognizedTexts={recognizedTexts}
                translatedTexts={translatedTexts}
                profiles={sttProfiles}
              />
            ) : null
          }
          dateTimeLocale={dateTimeLocale}
          canStartRecognition={canStartRecognition}
          downloadingModels={model.downloading}
          fileExportAvailable={capabilities.fileExport}
          recognitionControlAvailable={capabilities.recognitionControl}
          speechControlAvailable={capabilities.speechControl}
          canClearLogs={
            recognizedTexts.length > 0 || translatedTexts.length > 0
          }
          onClearRecognizedTexts={() => {
            setRecognizedTexts([]);
            setTranslatedTexts([]);
          }}
          onSetProfileVolume={(profileId, volumePercent) =>
            updateProfile(profileId, (profile) => ({
              ...profile,
              input: { ...profile.input, volume_percent: volumePercent },
            }))
          }
          onToggleProfileMute={(profileId) => {
            const profile = sttProfiles.find(
              (candidate) => candidate.id === profileId,
            );
            if (profile) {
              updateProfile(profileId, (current) => ({
                ...current,
                input: { ...current.input, muted: !current.input.muted },
              }));
            }
          }}
          onOpenModelDownload={() => {
            setUi((current) => ({
              ...current,
              settingsOpen: true,
              settingsTab: "downloads",
            }));
            if (!model.downloading) {
              void downloadSelectedModels();
            }
          }}
          onStartRecognition={startRecognition}
          onStopRecognition={stopRecognition}
          onStopSpeech={() => stopSpeech(config.ync_plugin_port)}
          onSaveRecognitionCsv={(defaultFileName, content) =>
            services.system.saveRecognitionCsv({ defaultFileName, content })
          }
          onSaveAsrInputWav={(defaultFileName, content) =>
            services.system.saveAsrInputWav({ defaultFileName, content })
          }
          onPlayAudio={(samples, sampleRate) =>
            services.system.playAudio(samples, sampleRate)
          }
        />
      </Group>
    </Stack>
  );
};
