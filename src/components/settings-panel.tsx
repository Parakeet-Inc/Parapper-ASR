import {
  ActionIcon,
  Button,
  Checkbox,
  ColorSwatch,
  Group,
  Modal,
  Paper,
  Select,
  Stack,
  Tabs,
  Text,
  Title,
} from "@mantine/core";
import { type CSSProperties, type ReactNode, useState } from "react";
import { useTranslation } from "react-i18next";

import type {
  FrontendCapabilities,
  RustLicensesDocument,
} from "../application/frontend-services";
import { zeroMinHeight, zeroMinSize } from "../lib/layout-styles";
import {
  settingsNavigationForGroup,
  type SettingsNavigationTab,
} from "../lib/settings-navigation";
import { STT_PROFILE_DISPLAY_COLOR_CSS } from "../lib/stt-profile-colors";
import {
  sttProfileEditorConfig,
  sttProfileDisplayName,
  sttProfileListPresentation,
  sttProfileWithAsrModel,
  updateSttProfileFromEditorField,
} from "../lib/stt-profiles";
import type {
  AudioDeviceInfo,
  ConfigPreset,
  LocalTranslationModel,
  ModelDownloadProgress,
  ModelStatus,
  ParapperConfig,
  SttProfileConfig,
  TranslationHttpListenerStatus,
} from "../lib/types";
import { AsrSettings } from "./settings/asr-settings";
import { ConnectionSettings } from "./settings/connection-settings";
import { ExternalAppSettings } from "./settings/external-app-settings";
import { LicenseSettings } from "./settings/license-settings";
import { ModelAssetsSettings } from "./settings/model-assets-settings";
import { NoiseCancellationSettings } from "./settings/noise-cancellation-settings";
import { OtherSettings } from "./settings/other-settings";
import { SpeechSettings } from "./settings/speech-settings";
import { TranslationSettings } from "./settings/translation-settings";
import { VadSettings } from "./settings/vad-settings";

import IconAdd from "~icons/material-symbols/add";
import IconContentCut from "~icons/material-symbols/content-cut";
import IconDelete from "~icons/material-symbols/delete";
import IconDescription from "~icons/material-symbols/description";
import IconDownload from "~icons/material-symbols/download";
import IconMic from "~icons/material-symbols/mic";
import IconMicNoiseCancelHigh from "~icons/material-symbols/mic-noise-cancel-high";
import IconSettingsEthernet from "~icons/material-symbols/settings-ethernet";
import IconSpeechToText from "~icons/material-symbols/speech-to-text";
import IconTranslate from "~icons/material-symbols/translate";
import IconTune from "~icons/material-symbols/tune";
import IconVoiceSelection from "~icons/material-symbols/voice-selection";

export type SettingsPanelProps = {
  config: ParapperConfig;
  capabilities: FrontendCapabilities;
  outputAudioDevices: AudioDeviceInfo[];
  settingsOpen: boolean;
  settingsTab: string | null;
  running: boolean;
  translationSpeechDelaySuspected: boolean;
  modelStatus: ModelStatus | null;
  downloadingModels: boolean;
  modelDownloadProgress: ModelDownloadProgress | null;
  configPresets: ConfigPreset[];
  sttProfiles: SttProfileConfig[];
  selectedSttProfileId: string;
  inputAudioDevices: AudioDeviceInfo[];
  refreshingAudioDevices: boolean;
  onSettingsOpenChange: (open: boolean) => void;
  onSettingsTabChange: (tab: string | null) => void;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onSuggestHotwordReadings: (surface: string) => Promise<string[]>;
  onDownloadSelectedModels: () => void;
  onDownloadLocalTranslationModel: (
    model: LocalTranslationModel,
  ) => Promise<boolean>;
  onResetConfig: () => Promise<unknown>;
  onSaveConfigPreset: (name: string) => Promise<ConfigPreset[]>;
  onDeleteConfigPreset: (name: string) => Promise<ConfigPreset[]>;
  onApplyConfigPreset: (config: ParapperConfig) => void;
  onSelectSttProfile: (profileId: string) => void;
  onAddSttProfile: () => void;
  onDeleteSttProfile: (profileId: string) => void;
  onUpdateSttProfile: (
    profileId: string,
    update: (profile: SttProfileConfig) => SttProfileConfig,
  ) => void;
  onSetSttProfileEnabled: (profileId: string, enabled: boolean) => void;
  onSetDeveloperConnectionEnabled: (enabled: boolean) => void;
  onRefreshAudioDevices: () => void;
  onRequestLoopbackPermission: () => Promise<void>;
  onFindNeoPort: () => Promise<number | null>;
  onFindYncPluginPort: () => Promise<number | null>;
  onFetchNeoVoices: (port: number) => Promise<string[]>;
  onGetTranslationServerStatus: () => Promise<TranslationHttpListenerStatus>;
  onGetLocalTranslationInstalled: (
    model: LocalTranslationModel,
  ) => Promise<boolean>;
  onStartTranslationServer: (
    port: number,
    model: LocalTranslationModel,
  ) => Promise<TranslationHttpListenerStatus>;
  onStopTranslationServer: () => Promise<TranslationHttpListenerStatus>;
  onOpenExternalUrl: (url: string) => Promise<void>;
  onLoadRustLicenses: () => Promise<RustLicensesDocument>;
};

const panelStyle = {
  flex: 1,
  ...zeroMinSize,
  overflowY: "auto",
  paddingRight: 8,
} as const;

const tabIconStyle = { width: 18, height: 18 } as const;

const tabIconByTab: Record<SettingsNavigationTab, ReactNode> = {
  connection: <IconMic style={tabIconStyle} />,
  "noise-cancellation": <IconMicNoiseCancelHigh style={tabIconStyle} />,
  vad: <IconContentCut style={tabIconStyle} />,
  asr: <IconSpeechToText style={tabIconStyle} />,
  "external-apps": <IconSettingsEthernet style={tabIconStyle} />,
  translation: <IconTranslate style={tabIconStyle} />,
  speech: <IconVoiceSelection style={tabIconStyle} />,
  other: <IconTune style={tabIconStyle} />,
  downloads: <IconDownload style={tabIconStyle} />,
  licenses: <IconDescription style={tabIconStyle} />,
};

type SttProfileSelectorProps = {
  profiles: ReturnType<typeof sttProfileListPresentation>;
  value: string;
  disabled: boolean;
  onChange: (profileId: string) => void;
};

const SttProfileSelector: React.FC<SttProfileSelectorProps> = ({
  profiles,
  value,
  disabled,
  onChange,
}) => {
  const { t } = useTranslation();
  const selectedProfile = profiles.find((profile) => profile.id === value);
  const selectedProfileColor = selectedProfile
    ? STT_PROFILE_DISPLAY_COLOR_CSS[selectedProfile.color]
    : null;
  return (
    <Select
      label={t("sttProfiles.selectorLabel")}
      data={profiles.map(({ id, label }) => ({ value: id, label }))}
      value={value}
      leftSection={
        selectedProfile ? (
          <ColorSwatch
            color={selectedProfileColor ?? "transparent"}
            size={12}
            style={{ border: "1px solid var(--mantine-color-default-border)" }}
          />
        ) : undefined
      }
      renderOption={({ option }) => {
        const profile = profiles.find(({ id }) => id === option.value);
        return (
          <Group gap="xs" wrap="nowrap">
            {profile ? (
              <ColorSwatch
                color={STT_PROFILE_DISPLAY_COLOR_CSS[profile.color]}
                size={12}
                style={{
                  border: "1px solid var(--mantine-color-default-border)",
                }}
              />
            ) : null}
            <Text size="sm">{option.label}</Text>
          </Group>
        );
      }}
      disabled={disabled}
      allowDeselect={false}
      onChange={(next) => {
        if (next) onChange(next);
      }}
    />
  );
};

type SttProfileListProps = {
  profiles: Array<
    ReturnType<typeof sttProfileListPresentation>[number] & {
      enabled: boolean;
    }
  >;
  disabled: boolean;
  onChange: (profileId: string) => void;
  onEnabledChange: (profileId: string, enabled: boolean) => void;
  onDelete: (profileId: string) => void;
  onAdd: () => void;
};

export const SttProfileList: React.FC<SttProfileListProps> = ({
  profiles,
  disabled,
  onChange,
  onEnabledChange,
  onDelete,
  onAdd,
}) => {
  const { t } = useTranslation();
  const enabledProfileCount = profiles.filter(
    (profile) => profile.enabled,
  ).length;
  return (
    <Stack gap={4}>
      <Text size="sm" fw={500}>
        {t("sttProfiles.selectorLabel")}
      </Text>
      <Stack gap={4} role="listbox" aria-label={t("sttProfiles.selectorLabel")}>
        {profiles.map(({ id, label, color, selected, enabled }) => {
          const isLastEnabledProfile = enabled && enabledProfileCount === 1;
          const profileColor = STT_PROFILE_DISPLAY_COLOR_CSS[color];
          const profileButtonColor =
            profileColor === "transparent" ? "gray" : color;
          const selectedBorderColor =
            profileColor === "transparent"
              ? "var(--mantine-color-gray-6)"
              : profileColor;
          return (
            <Group
              key={id}
              gap="xs"
              wrap="nowrap"
              p={4}
              styles={{
                root: {
                  borderRadius: "var(--mantine-radius-sm)",
                  border: `1px solid ${
                    selected
                      ? selectedBorderColor
                      : "var(--mantine-color-default-border)"
                  }`,
                },
              }}
            >
              <Button
                role="option"
                aria-selected={selected}
                variant={selected ? "light" : "subtle"}
                color={selected ? profileButtonColor : "gray"}
                disabled={disabled}
                style={{ flex: 1 }}
                justify="flex-start"
                leftSection={
                  <ColorSwatch
                    color={profileColor}
                    size={14}
                    style={{
                      border: "1px solid var(--mantine-color-default-border)",
                    }}
                  />
                }
                onClick={() => onChange(id)}
              >
                {label}
              </Button>
              <Checkbox
                aria-label={t("sttProfiles.enabledFor", { name: label })}
                checked={enabled}
                disabled={disabled || isLastEnabledProfile}
                style={
                  isLastEnabledProfile
                    ? ({
                        opacity: 1,
                        "--checkbox-color": "var(--mantine-color-blue-filled)",
                      } as CSSProperties)
                    : undefined
                }
                styles={
                  isLastEnabledProfile
                    ? {
                        root: { opacity: 1 },
                        input: {
                          opacity: 1,
                          backgroundColor: "var(--mantine-color-blue-filled)",
                          borderColor: "var(--mantine-color-blue-filled)",
                        },
                        icon: {
                          opacity: 1,
                          color: "var(--mantine-color-white)",
                        },
                      }
                    : undefined
                }
                onChange={(event) =>
                  onEnabledChange(id, event.currentTarget.checked)
                }
              />
              <ActionIcon
                variant="subtle"
                color="red"
                aria-label={t("sttProfiles.deleteFor", { name: label })}
                disabled={disabled || profiles.length <= 1}
                onClick={() => onDelete(id)}
              >
                <IconDelete />
              </ActionIcon>
            </Group>
          );
        })}
      </Stack>
      <Button
        variant="light"
        leftSection={<IconAdd />}
        disabled={disabled}
        onClick={onAdd}
      >
        {t("sttProfiles.add")}
      </Button>
    </Stack>
  );
};

type SttProfileDeleteConfirmationModalProps = {
  profile: SttProfileConfig | null;
  disabled: boolean;
  withinPortal?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

export const SttProfileDeleteConfirmationModal: React.FC<
  SttProfileDeleteConfirmationModalProps
> = ({ profile, disabled, withinPortal, onCancel, onConfirm }) => {
  const { t } = useTranslation();
  const profileName = profile
    ? sttProfileDisplayName(profile, (number) =>
        t("sttProfiles.defaultName", { number }),
      )
    : "";

  return (
    <Modal
      opened={profile !== null}
      onClose={onCancel}
      title={t("sttProfiles.deleteConfirmationTitle")}
      centered
      withinPortal={withinPortal}
    >
      <Stack gap="md">
        <Text size="sm">
          {t("sttProfiles.deleteConfirmationBody", { name: profileName })}
        </Text>
        <Group justify="flex-end">
          <Button variant="subtle" onClick={onCancel}>
            {t("common.cancel")}
          </Button>
          <Button color="red" disabled={disabled} onClick={onConfirm}>
            {t("sttProfiles.delete")}
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
};

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  config,
  capabilities,
  outputAudioDevices,
  settingsOpen,
  settingsTab,
  running,
  translationSpeechDelaySuspected,
  modelStatus,
  downloadingModels,
  modelDownloadProgress,
  configPresets,
  sttProfiles,
  selectedSttProfileId,
  inputAudioDevices,
  refreshingAudioDevices,
  onSettingsOpenChange,
  onSettingsTabChange,
  onUpdateConfig,
  onSuggestHotwordReadings,
  onDownloadSelectedModels,
  onDownloadLocalTranslationModel,
  onResetConfig,
  onSaveConfigPreset,
  onDeleteConfigPreset,
  onApplyConfigPreset,
  onSelectSttProfile,
  onAddSttProfile,
  onDeleteSttProfile,
  onUpdateSttProfile,
  onSetSttProfileEnabled,
  onSetDeveloperConnectionEnabled,
  onRefreshAudioDevices,
  onRequestLoopbackPermission,
  onFindNeoPort,
  onFindYncPluginPort,
  onFetchNeoVoices,
  onGetTranslationServerStatus,
  onGetLocalTranslationInstalled,
  onStartTranslationServer,
  onStopTranslationServer,
  onOpenExternalUrl,
  onLoadRustLicenses,
}) => {
  const { t } = useTranslation();
  const [saving, setSaving] = useState(false);
  const [profilePendingDeletion, setProfilePendingDeletion] =
    useState<SttProfileConfig | null>(null);
  const selectedSttProfile =
    sttProfiles.find((profile) => profile.id === selectedSttProfileId) ??
    sttProfiles[0];
  const effectiveSelectedSttProfileId = selectedSttProfile?.id ?? "";
  if (!selectedSttProfile) return null;
  const sttProfileList = sttProfileListPresentation(
    sttProfiles,
    effectiveSelectedSttProfileId,
    (number) => t("sttProfiles.defaultName", { number }),
  );
  const sttEditorConfig = selectedSttProfile
    ? sttProfileEditorConfig(config, selectedSttProfile)
    : config;
  const updateSttEditorConfig = <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => {
    if (!selectedSttProfile || running) return;
    onUpdateSttProfile(selectedSttProfile.id, (profile) =>
      updateSttProfileFromEditorField(profile, key, value),
    );
  };

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
          <Tabs.List
            style={{
              flex: settingsOpen ? "0 0 156px" : "0 0 120px",
              overflowY: "auto",
            }}
          >
            {settingsOpen ? (
              <Stack gap={4} px="xs" pb="xs">
                <Text size="xs" fw={700} c="dimmed">
                  {t("settingsGroups.stt")}
                </Text>
              </Stack>
            ) : (
              <Text size="xs" fw={700} c="dimmed" px="xs" pb="xs">
                {t("settingsGroups.stt")}
              </Text>
            )}
            {settingsNavigationForGroup("stt").map((item) => (
              <Tabs.Tab
                key={item.tab}
                value={item.tab}
                leftSection={tabIconByTab[item.tab]}
              >
                {t(item.labelKey)}
              </Tabs.Tab>
            ))}
            <Text size="xs" fw={700} c="dimmed" px="xs" pt="md" pb={4}>
              {t("settingsGroups.output")}
            </Text>
            {settingsNavigationForGroup("output").map((item) => (
              <Tabs.Tab
                key={item.tab}
                value={item.tab}
                leftSection={tabIconByTab[item.tab]}
              >
                {t(item.labelKey)}
              </Tabs.Tab>
            ))}
            <Text size="xs" fw={700} c="dimmed" px="xs" pt="md" pb={4}>
              {t("settingsGroups.other")}
            </Text>
            {settingsNavigationForGroup("other").map((item) => (
              <Tabs.Tab
                key={item.tab}
                value={item.tab}
                leftSection={tabIconByTab[item.tab]}
              >
                {t(item.labelKey)}
              </Tabs.Tab>
            ))}
          </Tabs.List>

          {settingsOpen ? (
            <>
              <Tabs.Panel value="connection" pt="md" pl="md" style={panelStyle}>
                <Stack gap="md">
                  <SttProfileList
                    profiles={sttProfileList}
                    disabled={running}
                    onChange={onSelectSttProfile}
                    onEnabledChange={(profileId, enabled) => {
                      if (running) return;
                      onSetSttProfileEnabled(profileId, enabled);
                    }}
                    onDelete={(profileId) => {
                      const profile = sttProfiles.find(
                        (candidate) => candidate.id === profileId,
                      );
                      if (profile) setProfilePendingDeletion(profile);
                    }}
                    onAdd={onAddSttProfile}
                  />
                  <ConnectionSettings
                    input={selectedSttProfile.input}
                    profiles={sttProfiles}
                    selectedProfileId={selectedSttProfile.id}
                    profile={selectedSttProfile}
                    onUpdateProfile={onUpdateSttProfile}
                    inputAudioDevices={inputAudioDevices}
                    runtimeLocked={running}
                    localAudioDevicesAvailable={capabilities.localAudioDevices}
                    refreshingAudioDevices={refreshingAudioDevices}
                    onRefreshAudioDevices={onRefreshAudioDevices}
                    onRequestLoopbackPermission={onRequestLoopbackPermission}
                    onChange={(input) => {
                      if (!selectedSttProfile || running) return;
                      onUpdateSttProfile(selectedSttProfile.id, (profile) => ({
                        ...profile,
                        input,
                      }));
                    }}
                  />
                </Stack>
                <SttProfileDeleteConfirmationModal
                  profile={profilePendingDeletion}
                  disabled={running}
                  onCancel={() => setProfilePendingDeletion(null)}
                  onConfirm={() => {
                    if (!profilePendingDeletion || running) return;
                    onDeleteSttProfile(profilePendingDeletion.id);
                    setProfilePendingDeletion(null);
                  }}
                />
              </Tabs.Panel>

              <Tabs.Panel
                value="noise-cancellation"
                pt="md"
                pl="md"
                style={panelStyle}
              >
                <Stack gap="md">
                  <SttProfileSelector
                    profiles={sttProfileList}
                    value={effectiveSelectedSttProfileId}
                    disabled={running}
                    onChange={onSelectSttProfile}
                  />
                  <NoiseCancellationSettings
                    config={sttEditorConfig}
                    runtimeLocked={running}
                    onUpdateConfig={updateSttEditorConfig}
                  />
                </Stack>
              </Tabs.Panel>

              <Tabs.Panel value="vad" pt="md" pl="md" style={panelStyle}>
                <Stack gap="md">
                  <SttProfileSelector
                    profiles={sttProfileList}
                    value={effectiveSelectedSttProfileId}
                    disabled={running}
                    onChange={onSelectSttProfile}
                  />
                  <VadSettings
                    config={sttEditorConfig}
                    runtimeLocked={running}
                    onUpdateConfig={updateSttEditorConfig}
                  />
                </Stack>
              </Tabs.Panel>

              <Tabs.Panel value="asr" pt="md" pl="md" style={panelStyle}>
                <Stack gap="md">
                  <SttProfileSelector
                    profiles={sttProfileList}
                    value={effectiveSelectedSttProfileId}
                    disabled={running}
                    onChange={onSelectSttProfile}
                  />
                  <AsrSettings
                    config={sttEditorConfig}
                    runtimeLocked={running}
                    onUpdateConfig={updateSttEditorConfig}
                    onApplyAsrModel={(model) => {
                      if (!selectedSttProfile || running) return;
                      onUpdateSttProfile(selectedSttProfile.id, (profile) =>
                        sttProfileWithAsrModel(profile, model),
                      );
                    }}
                    onSuggestHotwordReadings={onSuggestHotwordReadings}
                  />
                </Stack>
              </Tabs.Panel>

              <Tabs.Panel
                value="external-apps"
                pt="md"
                pl="md"
                style={panelStyle}
              >
                <ExternalAppSettings
                  config={config}
                  sttProfiles={sttProfiles}
                  runtimeLocked={running}
                  nativeConnectionsAvailable={
                    capabilities.externalConnectionProbe
                  }
                  onFindNeoPort={onFindNeoPort}
                  onFindYncPluginPort={onFindYncPluginPort}
                  onSetDeveloperConnectionEnabled={
                    onSetDeveloperConnectionEnabled
                  }
                  onSetSttProfileNeoHttpEnabled={(profileId, enabled) => {
                    if (running) return;
                    onUpdateSttProfile(profileId, (profile) => ({
                      ...profile,
                      neo_http_enabled: enabled,
                    }));
                  }}
                  onSetSttProfileDeveloperHttpEnabled={(profileId, enabled) => {
                    if (running) return;
                    onUpdateSttProfile(profileId, (profile) => ({
                      ...profile,
                      developer_http_enabled: enabled,
                    }));
                  }}
                  onUpdateConfig={onUpdateConfig}
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
                  yncPluginAvailable={capabilities.externalConnectionProbe}
                  localTranslationServerAvailable={
                    capabilities.localTranslationServer
                  }
                  onUpdateConfig={onUpdateConfig}
                  onDownloadServerModel={onDownloadLocalTranslationModel}
                  onGetTranslationServerStatus={onGetTranslationServerStatus}
                  onGetLocalTranslationInstalled={
                    onGetLocalTranslationInstalled
                  }
                  onStartTranslationServer={onStartTranslationServer}
                  onStopTranslationServer={onStopTranslationServer}
                />
              </Tabs.Panel>

              <Tabs.Panel value="speech" pt="md" pl="md" style={panelStyle}>
                <SpeechSettings
                  config={config}
                  outputAudioDevices={outputAudioDevices}
                  runtimeLocked={running}
                  neoReadAloudDelaySuspected={translationSpeechDelaySuspected}
                  yncPluginAvailable={capabilities.externalConnectionProbe}
                  onFetchNeoVoices={onFetchNeoVoices}
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

              <Tabs.Panel value="downloads" pt="md" pl="md" style={panelStyle}>
                {capabilities.modelManagement ? (
                  <ModelAssetsSettings
                    modelStatus={modelStatus}
                    downloading={downloadingModels}
                    progress={modelDownloadProgress}
                    runtimeLocked={running}
                    onDownload={onDownloadSelectedModels}
                  />
                ) : null}
              </Tabs.Panel>

              <Tabs.Panel value="licenses" pt="md" pl="md" style={panelStyle}>
                <LicenseSettings
                  onOpenExternalUrl={onOpenExternalUrl}
                  onLoadRustLicenses={onLoadRustLicenses}
                />
              </Tabs.Panel>
            </>
          ) : null}
        </Tabs>
      </Stack>
    </Paper>
  );
};
