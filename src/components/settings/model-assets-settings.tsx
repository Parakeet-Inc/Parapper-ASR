import {
  Badge,
  Button,
  Group,
  Progress,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { useTranslation } from "react-i18next";

import { notificationColor } from "../../lib/theme";
import type { ModelDownloadProgress, ModelStatus } from "../../lib/types";

type ModelAssetsSettingsProps = {
  modelStatus: ModelStatus | null;
  downloading: boolean;
  progress: ModelDownloadProgress | null;
  runtimeLocked: boolean;
  onDownload: () => void;
};

export const ModelAssetsSettings: React.FC<ModelAssetsSettingsProps> = ({
  modelStatus,
  downloading,
  progress,
  runtimeLocked,
  onDownload,
}) => {
  const { t } = useTranslation();
  const progressPercent = Math.round((progress?.progress ?? 0) * 100);
  const assets = [
    ["noise-cancellation", "NC", modelStatus?.noise_cancellation],
    ["vad", "VAD", modelStatus?.vad],
    ["asr", "ASR", modelStatus?.asr],
    ["japanese-morph", "Morph", modelStatus?.japanese_morph],
    ["language-id", "SLI", modelStatus?.language_id],
    ["local-translation", "MT", modelStatus?.local_translation],
  ] as const;
  const collections = [
    ["turn-detectors", "TD", modelStatus?.turn_detectors],
    ["tts", "TTS", modelStatus?.tts],
  ] as const;

  return (
    <Stack gap="sm">
      <Text fw={600}>{t("settings.downloadModels.button")}</Text>
      <Text size="xs" c="dimmed" style={{ wordBreak: "break-all" }}>
        {modelStatus?.root_dir ?? t("common.unset")}
      </Text>
      <Group gap="xs">
        {assets
          .filter(([, , status]) => Boolean(status))
          .map(([key, label, status]) => (
            <AssetBadge
              key={key}
              label={label}
              installed={status?.installed === true}
              preparing={status?.preparing === true}
            />
          ))}
        {collections
          .filter(([, , statuses]) => Boolean(statuses?.length))
          .map(([key, label, statuses]) => (
            <AssetBadge
              key={key}
              label={label}
              installed={statuses?.every((status) => status.installed) === true}
              preparing={statuses?.some((status) => status.preparing) === true}
            />
          ))}
      </Group>
      <Tooltip
        label={runtimeLocked ? t("tooltip.runtimeLocked") : ""}
        disabled={!runtimeLocked}
      >
        <span style={{ display: "block", width: "fit-content" }}>
          <Button
            variant="default"
            disabled={runtimeLocked || downloading}
            onClick={onDownload}
          >
            {downloading || progress
              ? `${t("settings.downloadModels.button")} ${progressPercent}%`
              : t("settings.downloadModels.button")}
          </Button>
        </span>
      </Tooltip>
      {downloading || progress ? (
        <Stack gap={4}>
          <Progress value={(progress?.progress ?? 0) * 100} color="primary" />
          {progress ? (
            <Text size="xs" c="dimmed">
              {t("downloadProgress.label", {
                file: progress.file_name,
                index: progress.file_index,
                total: progress.total_files,
              })}
            </Text>
          ) : null}
        </Stack>
      ) : null}
    </Stack>
  );
};

const AssetBadge: React.FC<{
  label: string;
  installed: boolean;
  preparing: boolean;
}> = ({ label, installed, preparing }) => {
  const { t } = useTranslation();
  return (
    <Badge
      color={
        installed
          ? notificationColor.ok
          : preparing
            ? "blue"
            : notificationColor.warn
      }
      variant="light"
    >
      {label}{" "}
      {installed
        ? t("status.ready")
        : preparing
          ? t("status.downloading")
          : t("status.missing")}
    </Badge>
  );
};
