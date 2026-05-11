import {
  Button,
  Divider,
  Group,
  NumberInput,
  Select,
  Stack,
  Switch,
  Text,
  TextInput,
  Title,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  DEFAULT_DEBUG_AUDIO_LOG_LIMIT,
  DEFAULT_RECOGNITION_LOG_LIMIT,
} from "../../lib/constants";
import {
  buildRetentionModeOptions,
  type SelectOption,
} from "../../lib/settings-options";
import { notificationColor } from "../../lib/theme";
import type { ConfigPreset, ParapperConfig } from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type OtherSettingsProps = {
  config: ParapperConfig;
  saving: boolean;
  running: boolean;
  presets: ConfigPreset[];
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onResetConfig: () => void;
  onSavePreset: (name: string) => Promise<ConfigPreset[]>;
  onDeletePreset: (name: string) => Promise<ConfigPreset[]>;
  onApplyPreset: (config: ParapperConfig) => void;
};

export const OtherSettings: React.FC<OtherSettingsProps> = ({
  config,
  saving,
  running,
  presets,
  onUpdateConfig,
  onResetConfig,
  onSavePreset,
  onDeletePreset,
  onApplyPreset,
}) => {
  const { t } = useTranslation();
  const [presetName, setPresetName] = useState("");
  const [selectedPresetName, setSelectedPresetName] = useState<string | null>(
    null,
  );
  const [presetSaving, setPresetSaving] = useState(false);
  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.name === selectedPresetName) ?? null,
    [presets, selectedPresetName],
  );
  const presetOptions = useMemo(
    () =>
      presets.map((preset) => ({
        value: preset.name,
        label: preset.built_in
          ? t("settings.configPresets.builtInLabel", { name: preset.name })
          : preset.name,
      })),
    [presets, t],
  );
  const retentionModeOptions: SelectOption[] = buildRetentionModeOptions(t);

  const savePreset = async () => {
    const name = presetName.trim();
    if (!name) {
      notifications.show({
        title: t("notifications.configPresetSaveFailed.title"),
        message: t("settings.configPresets.emptyName"),
        color: notificationColor.error,
      });
      return;
    }

    setPresetSaving(true);
    try {
      await onSavePreset(name);
      setSelectedPresetName(name);
      notifications.show({
        title: t("notifications.configPresetSaved.title"),
        message: t("notifications.configPresetSaved.message", { name }),
      });
    } catch (error) {
      notifications.show({
        title: t("notifications.configPresetSaveFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    } finally {
      setPresetSaving(false);
    }
  };

  const deletePreset = async () => {
    if (!selectedPreset || selectedPreset.built_in) return;

    setPresetSaving(true);
    try {
      await onDeletePreset(selectedPreset.name);
      notifications.show({
        title: t("notifications.configPresetDeleted.title"),
        message: t("notifications.configPresetDeleted.message", {
          name: selectedPreset.name,
        }),
      });
      setSelectedPresetName(null);
    } catch (error) {
      notifications.show({
        title: t("notifications.configPresetDeleteFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    } finally {
      setPresetSaving(false);
    }
  };

  const applyPreset = () => {
    if (!selectedPreset) return;
    onApplyPreset(selectedPreset.config);
    notifications.show({
      title: t("notifications.configPresetApplied.title"),
      message: t("notifications.configPresetApplied.message", {
        name: selectedPreset.name,
      }),
    });
  };

  return (
    <Stack gap="xs">
      <Stack gap="xs">
        <Title order={5}>{t("settings.configPresets.title")}</Title>
        <Text size="sm" c="dimmed">
          {t("settings.configPresets.description")}
        </Text>
        <Stack gap={4}>
          <TextInput
            label={t("settings.configPresets.nameLabel")}
            placeholder={t("settings.configPresets.namePlaceholder")}
            value={presetName}
            onChange={(event) => setPresetName(event.currentTarget.value)}
          />
          <Button
            loading={presetSaving}
            fullWidth
            onClick={() => void savePreset()}
          >
            {t("settings.configPresets.saveButton")}
          </Button>
        </Stack>
        <Stack gap={4}>
          <Select
            label={t("settings.configPresets.loadLabel")}
            placeholder={t("settings.configPresets.loadPlaceholder")}
            data={presetOptions}
            value={selectedPresetName}
            searchable
            clearable
            maxDropdownHeight={260}
            onChange={setSelectedPresetName}
          />
          <Group grow align="center">
            <DisabledReasonTooltip
              disabled={running}
              label={t("tooltip.runtimeLocked")}
            >
              <Button
                variant="light"
                disabled={running || !selectedPreset}
                onClick={applyPreset}
              >
                {t("settings.configPresets.applyButton")}
              </Button>
            </DisabledReasonTooltip>
            <Tooltip
              label={t("settings.configPresets.deleteBuiltInTooltip")}
              disabled={!selectedPreset?.built_in}
              multiline
              w={260}
            >
              <Button
                variant="default"
                color={notificationColor.error}
                loading={presetSaving}
                disabled={!selectedPreset || selectedPreset.built_in}
                onClick={() => void deletePreset()}
              >
                {t("settings.configPresets.deleteButton")}
              </Button>
            </Tooltip>
          </Group>
        </Stack>
      </Stack>
      <Divider my="xs" />
      <Stack gap={4}>
        {settingLabel(
          t("settings.debugAudioPlayback.label"),
          t("settings.debugAudioPlayback.description"),
        )}
        <Switch
          aria-label={t("settings.debugAudioPlayback.label")}
          checked={config.debug_asr_audio_playback}
          onChange={(event) =>
            onUpdateConfig(
              "debug_asr_audio_playback",
              event.currentTarget.checked,
            )
          }
        />
      </Stack>
      <Stack gap={4}>
        <Select
          label={settingLabel(
            t("settings.recognitionLogRetention.label"),
            t("settings.recognitionLogRetention.description"),
          )}
          data={retentionModeOptions}
          value={
            config.recognition_log_limit === null ? "unlimited" : "limited"
          }
          allowDeselect={false}
          onChange={(value) =>
            onUpdateConfig(
              "recognition_log_limit",
              value === "unlimited"
                ? null
                : (config.recognition_log_limit ??
                    DEFAULT_RECOGNITION_LOG_LIMIT),
            )
          }
        />
        {config.recognition_log_limit !== null ? (
          <NumberInput
            value={config.recognition_log_limit}
            min={1}
            max={100000}
            step={100}
            onChange={(value) =>
              onUpdateConfig(
                "recognition_log_limit",
                typeof value === "number"
                  ? Math.max(1, Math.floor(value))
                  : DEFAULT_RECOGNITION_LOG_LIMIT,
              )
            }
          />
        ) : null}
      </Stack>
      <Stack gap={4}>
        <Select
          label={settingLabel(
            t("settings.debugAudioRetention.label"),
            t("settings.debugAudioRetention.description"),
          )}
          data={retentionModeOptions}
          value={
            config.debug_audio_log_limit === null ? "unlimited" : "limited"
          }
          allowDeselect={false}
          onChange={(value) =>
            onUpdateConfig(
              "debug_audio_log_limit",
              value === "unlimited"
                ? null
                : (config.debug_audio_log_limit ??
                    DEFAULT_DEBUG_AUDIO_LOG_LIMIT),
            )
          }
        />
        {config.debug_audio_log_limit !== null ? (
          <NumberInput
            value={config.debug_audio_log_limit}
            min={0}
            max={100000}
            step={10}
            onChange={(value) =>
              onUpdateConfig(
                "debug_audio_log_limit",
                typeof value === "number"
                  ? Math.max(0, Math.floor(value))
                  : DEFAULT_DEBUG_AUDIO_LOG_LIMIT,
              )
            }
          />
        ) : null}
      </Stack>
      <DisabledReasonTooltip
        disabled={running}
        label={t("tooltip.runtimeLocked")}
      >
        <Tooltip
          label={t("settings.resetConfig.tooltipReady")}
          disabled={running}
          multiline
          w={280}
        >
          <Button
            variant="default"
            color={notificationColor.error}
            loading={saving}
            disabled={running}
            onClick={onResetConfig}
          >
            {t("settings.resetConfig.button")}
          </Button>
        </Tooltip>
      </DisabledReasonTooltip>
    </Stack>
  );
};
