import {
  Badge,
  Button,
  Group,
  Progress,
  Select,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { useTranslation } from "react-i18next";

import {
  buildAsrThreadOptions,
  type SelectOption,
} from "../../lib/settings-options";
import { notificationColor } from "../../lib/theme";
import type {
  AsrModel,
  AsrPrecision,
  ModelDownloadProgress,
  ModelStatus,
  ParapperConfig,
} from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type AsrSettingsProps = {
  config: ParapperConfig;
  selectedAsrPrecisionOptions: SelectOption<AsrPrecision>[];
  asrModelSelectOptions: SelectOption<AsrModel>[];
  modelStatus: ModelStatus | null;
  downloadingModels: boolean;
  modelDownloadProgress: ModelDownloadProgress | null;
  disabled: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onApplyAsrModel: (model: AsrModel) => void;
  onDownloadSelectedModels: () => void;
};

export const AsrSettings: React.FC<AsrSettingsProps> = ({
  config,
  selectedAsrPrecisionOptions,
  asrModelSelectOptions,
  modelStatus,
  downloadingModels,
  modelDownloadProgress,
  disabled,
  onUpdateConfig,
  onApplyAsrModel,
  onDownloadSelectedModels,
}) => {
  const { t } = useTranslation();
  const asrThreadOptions = buildAsrThreadOptions(t);
  const runtimeLockedTooltip = t("tooltip.runtimeLocked");
  const downloadProgressPercent = Math.round(
    (modelDownloadProgress?.progress ?? 0) * 100,
  );
  const downloadButtonLabel =
    downloadingModels || modelDownloadProgress
      ? `${t("settings.downloadModels.button")} ${downloadProgressPercent}%`
      : t("settings.downloadModels.button");

  return (
    <Stack gap="sm">
      <DisabledReasonTooltip disabled={disabled} label={runtimeLockedTooltip}>
        <Select
          label={settingLabel(
            t("settings.asrModel.label"),
            t("settings.asrModel.description"),
          )}
          data={asrModelSelectOptions}
          value={config.asr_model}
          allowDeselect={false}
          disabled={disabled}
          onChange={(value) => {
            if (value) {
              onApplyAsrModel(value as AsrModel);
            }
          }}
        />
      </DisabledReasonTooltip>
      <DisabledReasonTooltip disabled={disabled} label={runtimeLockedTooltip}>
        <Select
          label={settingLabel(
            t("settings.asrPrecision.label"),
            t("settings.asrPrecision.description"),
          )}
          data={selectedAsrPrecisionOptions}
          value={config.asr_precision}
          allowDeselect={false}
          disabled={disabled}
          onChange={(value) => {
            if (value) {
              onUpdateConfig("asr_precision", value as AsrPrecision);
            }
          }}
        />
      </DisabledReasonTooltip>
      <DisabledReasonTooltip disabled={disabled} label={runtimeLockedTooltip}>
        <Select
          label={settingLabel(
            t("settings.asrThreads.label"),
            t("settings.asrThreads.description"),
          )}
          data={asrThreadOptions}
          value={String(config.asr_num_threads)}
          allowDeselect={false}
          disabled={disabled}
          onChange={(value) =>
            onUpdateConfig("asr_num_threads", Number(value ?? 4))
          }
        />
      </DisabledReasonTooltip>
      <Stack gap={2}>
        {settingLabel(
          t("settings.modelDir.label"),
          t("settings.modelDir.description"),
        )}
        <Text size="xs" c="dimmed" style={{ wordBreak: "break-all" }}>
          {modelStatus?.asr.path ?? t("common.unset")}
        </Text>
      </Stack>
      <Stack gap="sm">
        <Group gap="xs">
          <Badge
            color={
              modelStatus?.vad.installed
                ? notificationColor.ok
                : notificationColor.warn
            }
            variant="light"
          >
            VAD{" "}
            {modelStatus?.vad.installed
              ? t("status.ready")
              : t("status.missing")}
          </Badge>
          <Badge
            color={
              modelStatus?.asr.installed
                ? notificationColor.ok
                : notificationColor.warn
            }
            variant="light"
          >
            ASR{" "}
            {modelStatus?.asr.installed
              ? t("status.ready")
              : t("status.missing")}
          </Badge>
        </Group>
        <Tooltip
          label={
            disabled
              ? runtimeLockedTooltip
              : t("settings.downloadModels.tooltipReady")
          }
          multiline
          w={280}
        >
          <span style={{ display: "block", width: "fit-content" }}>
            <Button
              variant="default"
              disabled={disabled || downloadingModels}
              onClick={onDownloadSelectedModels}
            >
              {downloadButtonLabel}
            </Button>
          </span>
        </Tooltip>
        {downloadingModels || modelDownloadProgress ? (
          <Stack gap={4}>
            <Progress
              value={(modelDownloadProgress?.progress ?? 0) * 100}
              color="primary"
            />
            {modelDownloadProgress ? (
              <Text size="xs" c="dimmed">
                {t("downloadProgress.label", {
                  file: modelDownloadProgress.file_name,
                  index: modelDownloadProgress.file_index,
                  total: modelDownloadProgress.total_files,
                })}
              </Text>
            ) : null}
          </Stack>
        ) : null}
      </Stack>
    </Stack>
  );
};
