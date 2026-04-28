import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Group,
  Paper,
  Stack,
  Text,
  Title,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import {
  buildRecognitionCsvExport,
  float32SamplesToWavBytes,
  formatCsvFileTimestamp,
  formatLogTime,
} from "../lib/recognition-log-csv";
import { notificationColor } from "../lib/theme";
import type { RecognizedTextEvent } from "../lib/types";

import IconDownload from "~icons/material-symbols/download";
import IconPlayArrow from "~icons/material-symbols/play-arrow";

type RecognitionLogProps = {
  asrWarning: string | null;
  recognizedTexts: RecognizedTextEvent[];
  dateTimeLocale: string;
  onClear: () => void;
};

export const RecognitionLog: React.FC<RecognitionLogProps> = ({
  asrWarning,
  recognizedTexts,
  dateTimeLocale,
  onClear,
}) => {
  const { t } = useTranslation();
  const logRef = useRef<HTMLDivElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);

  useEffect(() => {
    const logElement = logRef.current;
    if (!logElement) return;
    logElement.scrollTop = logElement.scrollHeight;
  }, [recognizedTexts.length]);

  const exportRecognizedTextsAsCsv = async () => {
    const csvExport = buildRecognitionCsvExport(recognizedTexts, {
      text: t("recognitionLog.csvHeaderText"),
      time: t("recognitionLog.csvHeaderTime"),
      seconds: t("recognitionLog.csvHeaderSeconds"),
      elapsedMs: t("recognitionLog.csvHeaderElapsedMs"),
    });
    try {
      const path = await invoke<string | null>("save_recognition_csv", {
        defaultFileName: csvExport.defaultFileName,
        content: csvExport.content,
      });
      if (!path) return;

      notifications.show({
        title: t("notifications.csvSaved.title"),
        message: path,
      });
    } catch (error) {
      notifications.show({
        title: t("notifications.csvSaveFailed.title"),
        message: error instanceof Error ? error.message : String(error),
        color: notificationColor.error,
      });
    }
  };

  const playAsrInputAudio = async (entry: RecognizedTextEvent) => {
    const samples = entry.debug_asr_audio_samples;
    const sampleRate = entry.debug_asr_audio_sample_rate;
    if (!samples?.length || !sampleRate) {
      notifications.show({
        title: t("notifications.audioNotPlayable.title"),
        message: t("notifications.audioNotPlayable.message"),
        color: notificationColor.warn,
      });
      return;
    }

    const audioContext = audioContextRef.current ?? new AudioContext();
    audioContextRef.current = audioContext;
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    audioSourceRef.current?.stop();

    const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(Float32Array.from(samples), 0);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start();
    audioSourceRef.current = source;
  };

  const downloadAsrInputAudio = async (entry: RecognizedTextEvent) => {
    const samples = entry.debug_asr_audio_samples;
    const sampleRate = entry.debug_asr_audio_sample_rate;
    if (!samples?.length || !sampleRate) {
      notifications.show({
        title: t("notifications.audioNotSavable.title"),
        message: t("notifications.audioNotSavable.message"),
        color: notificationColor.warn,
      });
      return;
    }

    try {
      const wavBytes = float32SamplesToWavBytes(samples, sampleRate);
      const path = await invoke<string | null>("save_asr_input_wav", {
        defaultFileName: `parapper-asr-input-${formatCsvFileTimestamp()}.wav`,
        content: Array.from(wavBytes),
      });
      if (!path) return;

      notifications.show({
        title: t("notifications.audioSaved.title"),
        message: path,
      });
    } catch (error) {
      notifications.show({
        title: t("notifications.audioSaveFailed.title"),
        message: error instanceof Error ? error.message : String(error),
        color: notificationColor.error,
      });
    }
  };

  return (
    <Paper
      withBorder
      radius="sm"
      p="md"
      style={{ flex: 1, minHeight: 0, overflow: "hidden" }}
    >
      <Stack h="100%" gap="sm" style={{ minHeight: 0 }}>
        <Group justify="space-between" align="center">
          <Title order={4}>{t("recognitionLog.title")}</Title>
          <Group gap="xs">
            {asrWarning ? (
              <Badge color={notificationColor.warn} variant="light">
                {t("status.asrWarning")}
              </Badge>
            ) : null}
            <Button
              variant="default"
              size="xs"
              disabled={recognizedTexts.length === 0}
              onClick={onClear}
            >
              {t("common.resetLogs")}
            </Button>
            <Button
              variant="light"
              size="xs"
              disabled={recognizedTexts.length === 0}
              onClick={() => void exportRecognizedTextsAsCsv()}
            >
              {t("common.csvExport")}
            </Button>
          </Group>
        </Group>
        {asrWarning ? (
          <Text size="sm" c={notificationColor.warn}>
            {asrWarning}
          </Text>
        ) : null}
        <Box
          ref={logRef}
          style={{
            flex: 1,
            minHeight: 0,
            overflowY: "auto",
            paddingRight: 4,
          }}
        >
          <Stack gap="xs" justify="end" style={{ minHeight: "100%" }}>
            {recognizedTexts.length === 0 ? (
              <Text c="dimmed" size="sm">
                {t("recognitionLog.empty")}
              </Text>
            ) : (
              recognizedTexts.map((entry, index) => (
                <Paper
                  key={`${entry.elapsed_millis}-${index}`}
                  p="xs"
                  withBorder
                  radius="sm"
                >
                  <Group wrap="nowrap" gap="sm">
                    <Text lineClamp={1} style={{ flex: 1, minWidth: 0 }}>
                      {entry.text}
                    </Text>
                    <Text size="xs" c="dimmed" w={72} ta="right">
                      {formatLogTime(
                        entry.recognized_at_millis,
                        dateTimeLocale,
                      )}
                    </Text>
                    <Text size="xs" c="dimmed" w={72} ta="right">
                      {entry.elapsed_millis} ms
                    </Text>
                    <Tooltip
                      label={
                        entry.debug_asr_audio_samples?.length
                          ? t("recognitionLog.audioPlayTooltip")
                          : t("recognitionLog.audioUnavailableTooltip")
                      }
                    >
                      <span>
                        <ActionIcon
                          aria-label={t("recognitionLog.playAriaLabel")}
                          variant="outline"
                          radius="xl"
                          size="sm"
                          disabled={!entry.debug_asr_audio_samples?.length}
                          onClick={() => void playAsrInputAudio(entry)}
                        >
                          <IconPlayArrow />
                        </ActionIcon>
                      </span>
                    </Tooltip>
                    <Tooltip
                      label={
                        entry.debug_asr_audio_samples?.length
                          ? t("recognitionLog.audioSaveTooltip")
                          : t("recognitionLog.audioUnavailableTooltip")
                      }
                    >
                      <span>
                        <ActionIcon
                          aria-label={t("recognitionLog.downloadAriaLabel")}
                          variant="outline"
                          radius="xl"
                          size="sm"
                          disabled={!entry.debug_asr_audio_samples?.length}
                          onClick={() => void downloadAsrInputAudio(entry)}
                        >
                          <IconDownload />
                        </ActionIcon>
                      </span>
                    </Tooltip>
                  </Group>
                </Paper>
              ))
            )}
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );
};
