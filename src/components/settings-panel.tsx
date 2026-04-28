import { Button, Group, Paper, Stack, Tabs, Title } from "@mantine/core";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { isMacOs } from "../lib/platform";
import type { SelectOption } from "../lib/settings-options";
import type {
  AsrModel,
  AsrPrecision,
  ModelDownloadProgress,
  ModelStatus,
  ParapperConfig,
} from "../lib/types";
import { AsrSettings } from "./settings/asr-settings";
import { ConnectionSettings } from "./settings/connection-settings";
import { LicenseSettings } from "./settings/license-settings";
import { OtherSettings } from "./settings/other-settings";
import { VadSettings } from "./settings/vad-settings";

export type SettingsPanelProps = {
  config: ParapperConfig;
  settingsOpen: boolean;
  settingsTab: string | null;
  running: boolean;
  selectedAsrPrecisionOptions: SelectOption<AsrPrecision>[];
  asrModelSelectOptions: SelectOption<AsrModel>[];
  modelStatus: ModelStatus | null;
  downloadingModels: boolean;
  modelDownloadProgress: ModelDownloadProgress | null;
  onSettingsOpenChange: (open: boolean) => void;
  onSettingsTabChange: (tab: string | null) => void;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onApplyAsrModel: (model: AsrModel) => void;
  onDownloadSelectedModels: () => void;
  onResetConfig: () => Promise<unknown>;
};

const panelStyle = {
  flex: 1,
  minHeight: 0,
  minWidth: 0,
  overflowY: "auto",
  paddingRight: 8,
} as const;

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  config,
  settingsOpen,
  settingsTab,
  running,
  selectedAsrPrecisionOptions,
  asrModelSelectOptions,
  modelStatus,
  downloadingModels,
  modelDownloadProgress,
  onSettingsOpenChange,
  onSettingsTabChange,
  onUpdateConfig,
  onApplyAsrModel,
  onDownloadSelectedModels,
  onResetConfig,
}) => {
  const { t } = useTranslation();
  const [saving, setSaving] = useState(false);
  const hideConnectionSettings = isMacOs();
  const activeSettingsTab =
    hideConnectionSettings && settingsTab === "connection"
      ? "vad"
      : settingsTab;

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
        flex: settingsOpen ? "0 0 520px" : "0 0 152px",
        minWidth: settingsOpen ? 520 : 152,
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
          value={activeSettingsTab}
          onChange={(value) => {
            onSettingsTabChange(value);
            if (value) {
              onSettingsOpenChange(true);
            }
          }}
          orientation="vertical"
          keepMounted={false}
          style={{ flex: 1, gap: 16, minHeight: 0, overflow: "hidden" }}
        >
          <Tabs.List style={{ flex: "0 0 120px" }}>
            {!hideConnectionSettings ? (
              <Tabs.Tab value="connection">{t("tabs.connection")}</Tabs.Tab>
            ) : null}
            <Tabs.Tab value="vad">{t("tabs.vad")}</Tabs.Tab>
            <Tabs.Tab value="asr">{t("tabs.asr")}</Tabs.Tab>
            <Tabs.Tab value="other">{t("tabs.other")}</Tabs.Tab>
            <Tabs.Tab value="licenses">{t("tabs.licenses")}</Tabs.Tab>
          </Tabs.List>

          {settingsOpen ? (
            <>
              {!hideConnectionSettings ? (
                <Tabs.Panel
                  value="connection"
                  pt="md"
                  pl="md"
                  style={panelStyle}
                >
                  <ConnectionSettings
                    config={config}
                    runtimeLocked={running}
                    onUpdateConfig={onUpdateConfig}
                  />
                </Tabs.Panel>
              ) : null}

              <Tabs.Panel value="vad" pt="md" pl="md" style={panelStyle}>
                <VadSettings
                  config={config}
                  onUpdateConfig={onUpdateConfig}
                />
              </Tabs.Panel>

              <Tabs.Panel value="asr" pt="md" pl="md" style={panelStyle}>
                <AsrSettings
                  config={config}
                  selectedAsrPrecisionOptions={selectedAsrPrecisionOptions}
                  asrModelSelectOptions={asrModelSelectOptions}
                  modelStatus={modelStatus}
                  downloadingModels={downloadingModels}
                  modelDownloadProgress={modelDownloadProgress}
                  disabled={running}
                  onUpdateConfig={onUpdateConfig}
                  onApplyAsrModel={onApplyAsrModel}
                  onDownloadSelectedModels={onDownloadSelectedModels}
                />
              </Tabs.Panel>

              <Tabs.Panel value="other" pt="md" pl="md" style={panelStyle}>
                <OtherSettings
                  config={config}
                  saving={saving}
                  running={running}
                  onUpdateConfig={onUpdateConfig}
                  onResetConfig={() => void resetConfig()}
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
