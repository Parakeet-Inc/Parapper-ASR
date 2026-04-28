import {
  ActionIcon,
  Box,
  Button,
  Flex,
  Paper,
  Progress,
  Select,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { invoke } from "@tauri-apps/api/core";
import type { Dispatch, SetStateAction } from "react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { RecognitionLog } from "./recognition-log";
import type { RuntimeState } from "../hooks/use-app-state";
import { buildAudioDeviceOptions } from "../lib/audio-devices";
import {
  normalizeParapperErrorPayload,
  notifyParapperIssue,
} from "../lib/error";
import type {
  AudioDeviceInfo,
  ParapperConfig,
  RecognizedTextEvent,
  RecognitionStatus,
} from "../lib/types";
import { levelToProgress } from "./ui/display";

import IconRefresh from "~icons/material-symbols/refresh";

type RuntimePanelProps = {
  config: ParapperConfig;
  audioDevices: AudioDeviceInfo[];
  recognizedTexts: RecognizedTextEvent[];
  runtime: RuntimeState;
  setRuntime: Dispatch<SetStateAction<RuntimeState>>;
  refreshingAudioDevices: boolean;
  dateTimeLocale: string;
  canStartRecognition: boolean;
  onClearRecognizedTexts: () => void;
  onRefreshAudioDevices: () => void;
  onApplyAudioDeviceConfig: (config: ParapperConfig) => void;
};

export const RuntimePanel: React.FC<RuntimePanelProps> = ({
  config,
  audioDevices,
  recognizedTexts,
  runtime,
  setRuntime,
  refreshingAudioDevices,
  dateTimeLocale,
  canStartRecognition,
  onClearRecognizedTexts,
  onRefreshAudioDevices,
  onApplyAudioDeviceConfig,
}) => {
  const { t } = useTranslation();
  const audioDeviceOptions = useMemo(
    () => buildAudioDeviceOptions(audioDevices),
    [audioDevices],
  );
  const selectedAudioDevice =
    config.input_device_host && config.input_device_id
      ? `${config.input_device_host}\u0000${config.input_device_id}`
      : null;

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

  return (
    <Stack style={{ flex: "1 1 0", minWidth: 0, minHeight: 0 }}>
      <RecognitionLog
        asrWarning={runtime.asrWarning}
        recognizedTexts={recognizedTexts}
        dateTimeLocale={dateTimeLocale}
        onClear={onClearRecognizedTexts}
      />

      <Paper withBorder radius="sm" p="md">
        <Stack gap="md">
          <Flex align="end" gap="xs">
            <Tooltip
              label={t("tooltip.runtimeLocked")}
              disabled={!runtime.running}
              multiline
              w={280}
            >
              <Box style={{ flex: 1, minWidth: 0 }}>
                <Select
                  label={t("settings.audioDevice.label")}
                  placeholder={t("settings.audioDevice.placeholder")}
                  data={audioDeviceOptions}
                  value={selectedAudioDevice}
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
                    const device = audioDevices.find(
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
              disabled={!runtime.running && !canStartRecognition}
              miw={96}
              style={{ flexShrink: 0, whiteSpace: "nowrap" }}
              onClick={() => {
                if (runtime.running) {
                  void stopRecognition();
                  return;
                }
                void startRecognition();
              }}
            >
              {runtime.running ? t("common.stop") : t("common.start")}
            </Button>
          </Flex>
          <Stack gap={4}>
            <Text size="sm" c="dimmed">
              {t("common.volume")}
            </Text>
            <Progress
              value={levelToProgress(runtime.inputLevel)}
              color="primary"
              size="lg"
            />
          </Stack>
        </Stack>
      </Paper>
    </Stack>
  );
};
