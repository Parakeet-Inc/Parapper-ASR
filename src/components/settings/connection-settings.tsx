import {
  Badge,
  Button,
  Group,
  NumberInput,
  Progress,
  SegmentedControl,
  Stack,
  Switch,
  Text,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { isMacOs } from "../../lib/platform";
import { notificationColor } from "../../lib/theme";
import type {
  ModelDownloadProgress,
  ModelStatus,
  NeoSendTiming,
  ParapperConfig,
} from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type ConnectionSettingsProps = {
  config: ParapperConfig;
  modelStatus: ModelStatus | null;
  downloadingModels: boolean;
  modelDownloadProgress: ModelDownloadProgress | null;
  runtimeLocked: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onDownloadSelectedModels: () => void;
};

export const ConnectionSettings: React.FC<ConnectionSettingsProps> = ({
  config,
  modelStatus,
  downloadingModels,
  modelDownloadProgress,
  runtimeLocked,
  onUpdateConfig,
  onDownloadSelectedModels,
}) => {
  const { t } = useTranslation();
  const [detectingNeoPort, setDetectingNeoPort] = useState(false);
  const [detectingPluginPort, setDetectingPluginPort] = useState(false);
  const runtimeLockedTooltip = t("tooltip.runtimeLocked");
  const nativeConnectionsDisabled = isMacOs();
  const neoHttpEnabled = !nativeConnectionsDisabled && config.neo_http_enabled;
  const yncPluginAvailable = !nativeConnectionsDisabled;
  const downloadProgressPercent = Math.round(
    (modelDownloadProgress?.progress ?? 0) * 100,
  );
  const downloadButtonLabel =
    downloadingModels || modelDownloadProgress
      ? `${t("settings.downloadModels.button")} ${downloadProgressPercent}%`
      : t("settings.downloadModels.button");
  const modelAssetBadges = [
    {
      key: "noise-cancellation",
      label: "NC",
      installed: modelStatus?.noise_cancellation?.installed === true,
      visible: Boolean(modelStatus?.noise_cancellation),
    },
    {
      key: "vad",
      label: "VAD",
      installed: modelStatus?.vad.installed === true,
      visible: true,
    },
    {
      key: "asr",
      label: "ASR",
      installed: modelStatus?.asr.installed === true,
      visible: true,
    },
    {
      key: "language-id",
      label: "SLI",
      installed: modelStatus?.language_id?.installed === true,
      visible: Boolean(modelStatus?.language_id),
    },
    {
      key: "turn-detectors",
      label: "TD",
      installed:
        modelStatus?.turn_detectors?.every((status) => status.installed) ===
        true,
      visible: Boolean(modelStatus?.turn_detectors?.length),
    },
    {
      key: "tts",
      label: "TTS",
      installed: modelStatus?.tts?.every((status) => status.installed) === true,
      visible: Boolean(modelStatus?.tts?.length),
    },
  ];

  const findNeoHttpPort = async () => {
    setDetectingNeoPort(true);
    try {
      const port = await invoke<number | null>("find_neo_http_port");
      if (!port) {
        notifications.show({
          title: t("notifications.neoPortNotFound.title"),
          message: t("notifications.neoPortNotFound.message"),
          color: notificationColor.warn,
        });
        return;
      }
      onUpdateConfig("neo_http_port", port);
      notifications.show({
        title: t("notifications.neoPortDetected.title"),
        message: t("notifications.neoPortDetected.message", { port }),
      });
    } finally {
      setDetectingNeoPort(false);
    }
  };

  const findPluginHttpPort = async () => {
    setDetectingPluginPort(true);
    try {
      const port = await invoke<number | null>("find_ync_plugin_http_port");
      if (!port) {
        notifications.show({
          title: t("notifications.pluginPortNotFound.title"),
          message: t("notifications.pluginPortNotFound.message"),
          color: notificationColor.warn,
        });
        return;
      }
      onUpdateConfig("translation_plugin_http_port", port);
      notifications.show({
        title: t("notifications.pluginPortDetected.title"),
        message: t("notifications.pluginPortDetected.message", { port }),
      });
    } finally {
      setDetectingPluginPort(false);
    }
  };

  return (
    <Stack gap="sm">
      <Stack gap={4}>
        {settingLabel(
          t("settings.neoHttpEnabled.label"),
          t("settings.neoHttpEnabled.description"),
        )}
        <Switch
          aria-label={t("settings.neoHttpEnabled.label")}
          checked={!nativeConnectionsDisabled && config.neo_http_enabled}
          disabled={nativeConnectionsDisabled}
          onChange={(event) =>
            onUpdateConfig("neo_http_enabled", event.currentTarget.checked)
          }
        />
      </Stack>
      {neoHttpEnabled ? (
        <>
          <Group align="end" gap="xs" wrap="nowrap">
            <NumberInput
              label={settingLabel(
                t("settings.neoHttpPort.label"),
                t("settings.neoHttpPort.description"),
              )}
              value={config.neo_http_port}
              min={1}
              max={65535}
              style={{ flex: 1 }}
              onChange={(value) =>
                onUpdateConfig(
                  "neo_http_port",
                  typeof value === "number" ? value : 15520,
                )
              }
            />
            <Button
              variant="light"
              loading={detectingNeoPort}
              onClick={() => void findNeoHttpPort()}
            >
              {t("common.search")}
            </Button>
          </Group>
        </>
      ) : null}
      {yncPluginAvailable ? (
        <>
          <Group align="end" gap="xs" wrap="nowrap">
            <NumberInput
              label={settingLabel(
                t("settings.translationPluginHttpPort.label"),
                t("settings.translationPluginHttpPort.description"),
              )}
              value={config.translation_plugin_http_port}
              min={1}
              max={65535}
              disabled={runtimeLocked}
              style={{ flex: 1 }}
              onChange={(value) =>
                onUpdateConfig(
                  "translation_plugin_http_port",
                  typeof value === "number" ? value : 8080,
                )
              }
            />
            <Button
              variant="light"
              loading={detectingPluginPort}
              disabled={runtimeLocked}
              onClick={() => void findPluginHttpPort()}
            >
              {t("common.search")}
            </Button>
          </Group>
        </>
      ) : null}
      {neoHttpEnabled ? (
        <Stack gap={4}>
          {settingLabel(
            t("settings.neoSendTiming.label"),
            t("settings.neoSendTiming.description"),
          )}
          <SegmentedControl
            aria-label={t("settings.neoSendTiming.label")}
            value={config.neo_send_timing}
            data={[
              {
                value: "interim",
                label: t("options.neoSendTiming.interim"),
              },
              {
                value: "final",
                label: t("options.neoSendTiming.final"),
              },
            ]}
            onChange={(value) =>
              onUpdateConfig("neo_send_timing", value as NeoSendTiming)
            }
          />
        </Stack>
      ) : null}
      <Stack gap={4}>
        {settingLabel(
          t("settings.oscQuery.muteSyncLabel"),
          t("settings.oscQuery.muteSyncDescription"),
        )}
        <DisabledReasonTooltip
          disabled={runtimeLocked}
          label={runtimeLockedTooltip}
        >
          <Switch
            aria-label={t("settings.oscQuery.muteSyncLabel")}
            checked={!nativeConnectionsDisabled && config.vrc_osc_micmute}
            disabled={nativeConnectionsDisabled || runtimeLocked}
            onChange={(event) =>
              onUpdateConfig("vrc_osc_micmute", event.currentTarget.checked)
            }
          />
        </DisabledReasonTooltip>
      </Stack>
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
          {modelAssetBadges
            .filter((badge) => badge.visible)
            .map((badge) => (
              <Badge
                key={badge.key}
                color={
                  badge.installed
                    ? notificationColor.ok
                    : notificationColor.warn
                }
                variant="light"
              >
                {badge.label}{" "}
                {badge.installed ? t("status.ready") : t("status.missing")}
              </Badge>
            ))}
        </Group>
        <Tooltip
          label={
            runtimeLocked
              ? runtimeLockedTooltip
              : t("settings.downloadModels.tooltipReady")
          }
          multiline
          w={280}
        >
          <span style={{ display: "block", width: "fit-content" }}>
            <Button
              variant="default"
              disabled={runtimeLocked || downloadingModels}
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
