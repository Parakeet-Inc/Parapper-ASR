import {
  ActionIcon,
  Box,
  Button,
  Flex,
  Paper,
  Progress,
  Select,
  Slider,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import type { Dispatch, ReactNode, SetStateAction } from "react";
import { useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";

import { RecognitionLog } from "./recognition-log";
import type { RuntimeState } from "../hooks/use-app-state";
import { useSyncedLogRowHeights } from "../hooks/use-synced-log-row-heights";
import { buildAudioDeviceOptions } from "../lib/audio-devices";
import {
  normalizeParapperErrorPayload,
  notifyParapperIssue,
} from "../lib/error";
import { zeroMinSize, zeroMinWidth } from "../lib/layout-styles";
import { notificationColor } from "../lib/theme";
import type {
  AudioDeviceInfo,
  ParapperConfig,
  RecognizedTextEvent,
  RecognitionStatus,
} from "../lib/types";

import IconMic from "~icons/material-symbols/mic";
import IconRefresh from "~icons/material-symbols/refresh";

type RuntimePanelProps = {
  config: ParapperConfig;
  inputAudioDevices: AudioDeviceInfo[];
  recognizedTexts: RecognizedTextEvent[];
  runtime: RuntimeState;
  setRuntime: Dispatch<SetStateAction<RuntimeState>>;
  refreshingAudioDevices: boolean;
  translationPanel?: ReactNode;
  dateTimeLocale: string;
  canStartRecognition: boolean;
  downloadingModels: boolean;
  onClearRecognizedTexts: () => void;
  onRefreshAudioDevices: () => void;
  onApplyAudioDeviceConfig: (config: ParapperConfig) => void;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onOpenModelDownload: () => void;
};

export const RuntimePanel: React.FC<RuntimePanelProps> = ({
  config,
  inputAudioDevices,
  recognizedTexts,
  runtime,
  setRuntime,
  refreshingAudioDevices,
  translationPanel,
  dateTimeLocale,
  canStartRecognition,
  downloadingModels,
  onClearRecognizedTexts,
  onRefreshAudioDevices,
  onApplyAudioDeviceConfig,
  onUpdateConfig,
  onOpenModelDownload,
}) => {
  const { t } = useTranslation();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const inputAudioDeviceOptions = useMemo(
    () => buildAudioDeviceOptions(inputAudioDevices),
    [inputAudioDevices],
  );
  const selectedInputAudioDevice =
    config.input_device_host && config.input_device_id
      ? `${config.input_device_host}\u0000${config.input_device_id}`
      : null;
  const hasTranslationPanel = Boolean(translationPanel);
  const hasYncSpeech =
    config.neo_http_enabled &&
    config.speech_mappings.some(
      (mapping) =>
        !mapping.muted &&
        mapping.backend === "ync" &&
        mapping.talker.trim() !== "",
    );
  const inputLevelColor = getInputLevelColor(
    runtime.inputLevel,
    runtime.inputLevelBeforeGain,
  );
  useSyncedLogRowHeights(panelRef, hasTranslationPanel);

  const refreshRecognitionStatus = async () => {
    const nextStatus = await invoke<RecognitionStatus>(
      "get_recognition_status",
    );
    setRuntime((current) => ({
      ...current,
      status: nextStatus,
      running: nextStatus === "listening",
    }));
    return nextStatus;
  };

  const stopRecognition = async () => {
    try {
      const nextStatus = await invoke<RecognitionStatus>("stop_recognition");
      setRuntime((current) => ({
        ...current,
        status: nextStatus,
        running: false,
      }));
    } catch (error) {
      const payload = normalizeParapperErrorPayload(error);
      setRuntime((current) => ({ ...current, lastError: payload }));
      notifyParapperIssue(payload);
      await refreshRecognitionStatus();
    }
  };

  const startRecognition = async () => {
    try {
      const nextStatus = await invoke<RecognitionStatus>("start_recognition");
      setRuntime((current) => ({
        ...current,
        status: nextStatus,
        running: nextStatus === "listening",
      }));
    } catch (error) {
      const payload = normalizeParapperErrorPayload(error);
      setRuntime((current) => ({ ...current, lastError: payload }));
      notifyParapperIssue(payload);
      try {
        await invoke<RecognitionStatus>("stop_recognition");
      } finally {
        await refreshRecognitionStatus();
      }
    }
  };

  const stopSpeech = async () => {
    try {
      await invoke("neo_speech_stop", {
        port: config.translation_plugin_http_port,
      });
    } catch (error) {
      notifications.show({
        title: t("notifications.speechStopFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    }
  };

  return (
    <Box
      ref={panelRef}
      style={{
        flex: "1 1 0",
        ...zeroMinSize,
        display: "grid",
        gridTemplateColumns: hasTranslationPanel
          ? "minmax(0, 1fr) minmax(0, 1fr)"
          : "minmax(0, 1fr)",
        gridTemplateRows: "minmax(0, 1fr) auto",
        gap: "var(--mantine-spacing-md)",
      }}
    >
      <Box style={zeroMinSize}>
        <RecognitionLog
          asrWarning={runtime.asrWarning}
          recognizedTexts={recognizedTexts}
          dateTimeLocale={dateTimeLocale}
          onClear={onClearRecognizedTexts}
        />
      </Box>

      {hasTranslationPanel ? (
        <Box style={zeroMinSize}>{translationPanel}</Box>
      ) : null}

      <Paper
        withBorder
        radius="sm"
        p="md"
        style={{
          gridColumn: hasTranslationPanel ? "1 / span 2" : "1",
          ...zeroMinWidth,
        }}
      >
        <Stack gap="md">
          <Flex align="end" gap="xs" wrap="wrap">
            <Tooltip
              label={t("tooltip.runtimeLocked")}
              disabled={!runtime.running}
              multiline
              w={280}
            >
              <Box style={{ flex: "1 1 240px", ...zeroMinWidth }}>
                <Select
                  label={t("settings.inputAudioDevice.label")}
                  placeholder={t("settings.inputAudioDevice.placeholder")}
                  data={inputAudioDeviceOptions}
                  value={selectedInputAudioDevice}
                  clearable
                  searchable
                  maxDropdownHeight={180}
                  disabled={runtime.running}
                  onChange={(value) => {
                    if (!value) {
                      onApplyAudioDeviceConfig({
                        ...config,
                        input_device_id: null,
                        input_device_host: null,
                        input_device_name: null,
                      });
                      return;
                    }

                    const [host, id] = value.split("\u0000");
                    const device = inputAudioDevices.find(
                      (candidate) =>
                        candidate.host === host && candidate.id === id,
                    );
                    onApplyAudioDeviceConfig({
                      ...config,
                      input_device_id: id,
                      input_device_host: host,
                      input_device_name: device?.display_name ?? null,
                    });
                  }}
                />
              </Box>
            </Tooltip>
            <Tooltip
              label={
                runtime.running
                  ? t("tooltip.runtimeLocked")
                  : t("settings.audioDevice.refreshTooltip")
              }
              openDelay={300}
            >
              <span>
                <ActionIcon
                  aria-label={t("settings.audioDevice.refreshAriaLabel")}
                  variant="default"
                  size="lg"
                  disabled={runtime.running}
                  loading={refreshingAudioDevices}
                  onClick={onRefreshAudioDevices}
                >
                  <IconRefresh />
                </ActionIcon>
              </span>
            </Tooltip>
            <Button
              color="primary"
              variant={runtime.running ? "outline" : "filled"}
              miw={96}
              style={{ flexShrink: 0, whiteSpace: "nowrap" }}
              loading={
                !runtime.running && !canStartRecognition && downloadingModels
              }
              onClick={() => {
                if (runtime.running) {
                  void stopRecognition();
                  return;
                }
                if (!canStartRecognition) {
                  onOpenModelDownload();
                  return;
                }
                void startRecognition();
              }}
            >
              {runtime.running
                ? t("common.stop")
                : canStartRecognition
                  ? t("common.start")
                  : t("settings.downloadModels.openButton")}
            </Button>
            {hasYncSpeech ? (
              <Button
                variant="default"
                miw={128}
                style={{ flexShrink: 0, whiteSpace: "nowrap" }}
                onClick={() => void stopSpeech()}
              >
                {t("speechSettings.stopButton")}
              </Button>
            ) : null}
          </Flex>
          <Box>
            <Flex align="center" gap="xs">
              <Box w={SETTING_ICON_SIZE} />
              <Box style={{ flex: 1, minWidth: 0 }}>
                <Text
                  size="xs"
                  c="dimmed"
                  ta="right"
                  style={{ fontVariantNumeric: "tabular-nums" }}
                >
                  {formatInputGainPercent(config.input_volume_db)}
                </Text>
              </Box>
            </Flex>
            <Flex align="center" gap="xs" mt={VOLUME_PERCENT_METER_GAP}>
              <Tooltip
                label={t("settings.inputVolume.label")}
                withArrow
                openDelay={300}
              >
                <Box
                  style={{
                    width: SETTING_ICON_SIZE,
                    height: SETTING_ICON_SIZE,
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "help",
                    lineHeight: 1,
                    transform: `translateY(${VOLUME_ICON_VERTICAL_NUDGE})`,
                  }}
                >
                  <IconMic
                    width={SETTING_ICON_SIZE}
                    height={SETTING_ICON_SIZE}
                  />
                </Box>
              </Tooltip>
              <Box style={{ flex: 1, minWidth: 0 }}>
                <MeterWithThresholds
                  linear={runtime.inputLevel}
                  color={inputLevelColor}
                  thresholds={inputMeterThresholds}
                />
                <Box mt={VOLUME_METER_SLIDER_GAP}>
                  <Slider
                    value={config.input_volume_db}
                    min={INPUT_VOLUME_MIN_DB}
                    max={INPUT_VOLUME_MAX_DB}
                    step={1}
                    marks={inputVolumeControlMarks}
                    label={(sliderValue) =>
                      `${formatSignedDb(Math.round(sliderValue))} dB`
                    }
                    mb={SLIDER_MARKS_BOTTOM_MARGIN}
                    onChange={(value) =>
                      onUpdateConfig("input_volume_db", value)
                    }
                  />
                </Box>
              </Box>
            </Flex>
          </Box>
        </Stack>
      </Paper>
    </Box>
  );
};

const INPUT_VOLUME_MIN_DB = -30;
const INPUT_VOLUME_MAX_DB = 30;
const INPUT_PRE_GAIN_ERROR_DB = -1;
const INPUT_LEVEL_WARN_DB = 0;
const LEVEL_METER_MIN_DB = -50;
const METER_MAX_DB = 5;
const METER_THRESHOLD_COLOR = "#868e96";
const SETTING_ICON_SIZE = "1.1rem";
const VOLUME_PERCENT_METER_GAP = 6;
const VOLUME_METER_SLIDER_GAP = 6;
const VOLUME_ICON_VERTICAL_NUDGE = "-8px";
const SLIDER_MARKS_BOTTOM_MARGIN = 15;

const inputVolumeControlMarks = [-30, -15, 0, 15, 30].map((value) => ({
  value,
  label: (
    <Text size="xs" style={{ fontVariantNumeric: "tabular-nums" }}>
      {formatSignedDb(value)}
    </Text>
  ),
}));

const inputMeterThresholds = [
  { db: INPUT_LEVEL_WARN_DB, color: METER_THRESHOLD_COLOR },
];

function formatSignedDb(value: number) {
  return value > 0 ? `+${value}` : String(value);
}

const dbToLinear = (db: number) => 10 ** (db / 20);

const linearToDb = (linear: number, minDb = LEVEL_METER_MIN_DB) => {
  if (!Number.isFinite(linear) || linear <= 0) {
    return minDb;
  }
  return Math.max(20 * Math.log10(linear), minDb);
};

const formatInputGainPercent = (db: number) =>
  `${(Math.max(0, dbToLinear(db)) * 100).toFixed(0)}%`;

const meterDbToProgress = (db: number) => {
  const clampedDb = Math.max(LEVEL_METER_MIN_DB, Math.min(METER_MAX_DB, db));
  return (
    ((clampedDb - LEVEL_METER_MIN_DB) / (METER_MAX_DB - LEVEL_METER_MIN_DB)) *
    100
  );
};

const getInputLevelColor = (postGainLinear: number, preGainLinear: number) => {
  const preGainDb = linearToDb(preGainLinear, LEVEL_METER_MIN_DB);
  if (preGainDb > INPUT_PRE_GAIN_ERROR_DB) {
    return notificationColor.error;
  }
  const postGainDb = linearToDb(postGainLinear, LEVEL_METER_MIN_DB);
  if (postGainDb > INPUT_LEVEL_WARN_DB) {
    return notificationColor.warn;
  }
  return notificationColor.primary;
};

const MeterWithThresholds: React.FC<{
  linear: number;
  color: string;
  thresholds: Array<{ db: number; color: string }>;
}> = ({ linear, color, thresholds }) => (
  <Box pos="relative">
    <Progress
      value={meterDbToProgress(linearToDb(linear, LEVEL_METER_MIN_DB))}
      color={color}
      size="sm"
    />
    {thresholds.map((threshold, idx) => (
      <Box
        key={`${threshold.db}-${idx}`}
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `calc(${meterDbToProgress(threshold.db)}% - 1px)`,
          width: "2px",
          backgroundColor: threshold.color,
          opacity: 0.95,
          pointerEvents: "none",
        }}
      />
    ))}
  </Box>
);
