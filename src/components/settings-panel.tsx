import { Button, Group, Paper, Stack, Tabs, Title } from "@mantine/core";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { zeroMinHeight, zeroMinSize } from "../lib/layout-styles";
import type {
  AudioDeviceInfo,
  AsrModel,
  ConfigPreset,
  ModelDownloadProgress,
  ModelStatus,
  ParapperConfig,
} from "../lib/types";
import { AsrSettings } from "./settings/asr-settings";
import { ConnectionSettings } from "./settings/connection-settings";
import { LicenseSettings } from "./settings/license-settings";
import { ModelAssetsSettings } from "./settings/model-assets-settings";
import { NoiseCancellationSettings } from "./settings/noise-cancellation-settings";
import { OtherSettings } from "./settings/other-settings";
import { SpeechSettings } from "./settings/speech-settings";
import { TranslationSettings } from "./settings/translation-settings";
import { VadSettings } from "./settings/vad-settings";

import IconContentCut from "~icons/material-symbols/content-cut";
import IconDescription from "~icons/material-symbols/description";
import IconMicNoiseCancelHigh from "~icons/material-symbols/mic-noise-cancel-high";
import IconSettingsEthernet from "~icons/material-symbols/settings-ethernet";
import IconSpeechToText from "~icons/material-symbols/speech-to-text";
import IconTranslate from "~icons/material-symbols/translate";
import IconTune from "~icons/material-symbols/tune";
import IconVoiceSelection from "~icons/material-symbols/voice-selection";

export type SettingsPanelProps = {
  config: ParapperConfig;
  outputAudioDevices: AudioDeviceInfo[];
  settingsOpen: boolean;
  settingsTab: string | null;
  running: boolean;
  translationSpeechDelaySuspected: boolean;
  modelStatus: ModelStatus | null;
  downloadingModels: boolean;
  modelDownloadProgress: ModelDownloadProgress | null;
  configPresets: ConfigPreset[];
  onSettingsOpenChange: (open: boolean) => void;
  onSettingsTabChange: (tab: string | null) => void;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onApplyAsrModel: (model: AsrModel) => void;
  onDownloadSelectedModels: () => void;
  onResetConfig: () => Promise<unknown>;
  onSaveConfigPreset: (name: string) => Promise<ConfigPreset[]>;
  onDeleteConfigPreset: (name: string) => Promise<ConfigPreset[]>;
  onApplyConfigPreset: (config: ParapperConfig) => void;
};

const panelStyle = {
  flex: 1,
  ...zeroMinSize,
  overflowY: "auto",
  paddingRight: 8,
} as const;

const tabIconStyle = { width: 18, height: 18 } as const;

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  config,
  outputAudioDevices,
  settingsOpen,
  settingsTab,
  running,
  translationSpeechDelaySuspected,
  modelStatus,
  downloadingModels,
  modelDownloadProgress,
  configPresets,
  onSettingsOpenChange,
  onSettingsTabChange,
  onUpdateConfig,
  onApplyAsrModel,
  onDownloadSelectedModels,
  onResetConfig,
  onSaveConfigPreset,
  onDeleteConfigPreset,
  onApplyConfigPreset,
}) => {
  const { t } = useTranslation();
  const [saving, setSaving] = useState(false);

  const resetConfig = async () => {
    setSaving(true);
    try {
      await onResetConfig();
    } finally {
      setSaving(false);
    }
  };

  return (
    <Paper
      withBorder
      radius="sm"
      p="md"
      style={{
        flex: settingsOpen ? "0 0 620px" : "0 0 152px",
        minWidth: settingsOpen ? 620 : 152,
        overflow: "hidden",
        transition: "flex-basis 160ms ease, min-width 160ms ease",
      }}
    >
      <Stack h="100%" gap="md" style={{ overflow: "hidden" }}>
        <Group justify="space-between">
          <Title order={4}>{t("tabs.settings")}</Title>
          <Button
            variant={settingsOpen ? "subtle" : "light"}
            px="xs"
            aria-label={
              settingsOpen
                ? t("tabs.collapseSettings")
                : t("tabs.expandSettings")
            }
            onClick={() => onSettingsOpenChange(!settingsOpen)}
          >
            {settingsOpen ? "<" : ">"}
          </Button>
        </Group>

        <Tabs
          value={settingsTab}
          onChange={(value) => {
            onSettingsTabChange(value);
            if (value) {
              onSettingsOpenChange(true);
            }
          }}
          orientation="vertical"
          keepMounted={false}
          style={{ flex: 1, gap: 16, ...zeroMinHeight, overflow: "hidden" }}
        >
          <Tabs.List style={{ flex: "0 0 120px" }}>
            <Tabs.Tab
              value="connection"
              leftSection={<IconSettingsEthernet style={tabIconStyle} />}
            >
              {t("tabs.connection")}
            </Tabs.Tab>
            <Tabs.Tab
              value="noise-cancellation"
              leftSection={<IconMicNoiseCancelHigh style={tabIconStyle} />}
            >
              {t("tabs.noiseCancellation")}
            </Tabs.Tab>
            <Tabs.Tab
              value="vad"
              leftSection={<IconContentCut style={tabIconStyle} />}
            >
              {t("tabs.vad")}
            </Tabs.Tab>
            <Tabs.Tab
              value="asr"
              leftSection={<IconSpeechToText style={tabIconStyle} />}
            >
              {t("tabs.asr")}
            </Tabs.Tab>
            <Tabs.Tab
              value="translation"
              leftSection={<IconTranslate style={tabIconStyle} />}
            >
              {t("tabs.translation")}
            </Tabs.Tab>
            <Tabs.Tab
              value="speech"
              leftSection={<IconVoiceSelection style={tabIconStyle} />}
            >
              {t("tabs.speech")}
            </Tabs.Tab>
            <Tabs.Tab
              value="other"
              leftSection={<IconTune style={tabIconStyle} />}
            >
              {t("tabs.other")}
            </Tabs.Tab>
            <Tabs.Tab
              value="licenses"
              leftSection={<IconDescription style={tabIconStyle} />}
            >
              {t("tabs.licenses")}
            </Tabs.Tab>
          </Tabs.List>

          {settingsOpen ? (
            <>
              <Tabs.Panel value="connection" pt="md" pl="md" style={panelStyle}>
                <Stack gap="md">
                  <ModelAssetsSettings
                    modelStatus={modelStatus}
                    downloading={downloadingModels}
                    progress={modelDownloadProgress}
                    runtimeLocked={running}
                    onDownload={onDownloadSelectedModels}
                  />
                  <ConnectionSettings
                    config={config}
                    runtimeLocked={running}
                    onUpdateConfig={onUpdateConfig}
                  />
                </Stack>
              </Tabs.Panel>

              <Tabs.Panel
                value="noise-cancellation"
                pt="md"
                pl="md"
                style={panelStyle}
              >
                <NoiseCancellationSettings
                  config={config}
                  runtimeLocked={running}
                  onUpdateConfig={onUpdateConfig}
                />
              </Tabs.Panel>

              <Tabs.Panel value="vad" pt="md" pl="md" style={panelStyle}>
                <VadSettings
                  config={config}
                  runtimeLocked={running}
                  onUpdateConfig={onUpdateConfig}
                />
              </Tabs.Panel>

              <Tabs.Panel value="asr" pt="md" pl="md" style={panelStyle}>
                <AsrSettings
                  config={config}
                  runtimeLocked={running}
                  onUpdateConfig={onUpdateConfig}
                  onApplyAsrModel={onApplyAsrModel}
                />
              </Tabs.Panel>

              <Tabs.Panel
                value="translation"
                pt="md"
                pl="md"
                style={panelStyle}
              >
                <TranslationSettings
                  config={config}
                  runtimeLocked={running}
                  onUpdateConfig={onUpdateConfig}
                />
              </Tabs.Panel>

              <Tabs.Panel value="speech" pt="md" pl="md" style={panelStyle}>
                <SpeechSettings
                  config={config}
                  outputAudioDevices={outputAudioDevices}
                  runtimeLocked={running}
                  neoReadAloudDelaySuspected={translationSpeechDelaySuspected}
                  onUpdateConfig={onUpdateConfig}
                />
              </Tabs.Panel>

              <Tabs.Panel value="other" pt="md" pl="md" style={panelStyle}>
                <OtherSettings
                  config={config}
                  saving={saving}
                  running={running}
                  presets={configPresets}
                  onUpdateConfig={onUpdateConfig}
                  onResetConfig={() => void resetConfig()}
                  onSavePreset={onSaveConfigPreset}
                  onDeletePreset={onDeleteConfigPreset}
                  onApplyPreset={onApplyConfigPreset}
                />
              </Tabs.Panel>

              <Tabs.Panel value="licenses" pt="md" pl="md" style={panelStyle}>
                <LicenseSettings />
              </Tabs.Panel>
            </>
          ) : null}
        </Tabs>
      </Stack>
    </Paper>
  );
};
