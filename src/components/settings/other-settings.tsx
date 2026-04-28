import {
  Button,
  NumberInput,
  Select,
  Stack,
  Switch,
  Tooltip,
} from "@mantine/core";
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
import type { ParapperConfig } from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type OtherSettingsProps = {
  config: ParapperConfig;
  saving: boolean;
  running: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onResetConfig: () => void;
};

export const OtherSettings: React.FC<OtherSettingsProps> = ({
  config,
  saving,
  running,
  onUpdateConfig,
  onResetConfig,
}) => {
  const { t } = useTranslation();
  const retentionModeOptions: SelectOption[] = buildRetentionModeOptions(t);

  return (
    <Stack gap="xs">
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
