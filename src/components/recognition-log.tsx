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

import { zeroMinHeight, zeroMinWidth } from "../lib/layout-styles";
import {
  buildRecognitionCsvExport,
  float32SamplesToWavBytes,
  formatCsvFileTimestamp,
  formatLogTime,
} from "../lib/recognition-log-csv";
import { recognitionSourceRowId } from "../lib/recognition-source";
import { notificationColor } from "../lib/theme";
import type { RecognizedTextEvent } from "../lib/types";

import IconDownload from "~icons/material-symbols/download";
import IconPlayArrow from "~icons/material-symbols/play-arrow";

type RecognitionLogProps = {
  asrWarning: string | null;
  recognizedTexts: RecognizedTextEvent[];
  reserveLanguageBadge: boolean;
  dateTimeLocale: string;
  canClearLogs: boolean;
  onClear: () => void;
};

const formatDetectedLanguage = (language: string) => language.toUpperCase();
const LANGUAGE_BADGE_SLOT_WIDTH = 44;

export const RecognitionLog: React.FC<RecognitionLogProps> = ({
  asrWarning,
  recognizedTexts,
  reserveLanguageBadge,
  dateTimeLocale,
  canClearLogs,
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
      style={{ height: "100%", ...zeroMinHeight, overflow: "hidden" }}
    >
      <Stack h="100%" gap="sm" style={zeroMinHeight}>
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
              disabled={!canClearLogs}
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
          data-log-scroll
          style={{
            flex: 1,
            ...zeroMinHeight,
            overflowY: "auto",
            paddingRight: 4,
          }}
        >
          <Stack gap="xs">
            {recognizedTexts.length === 0 ? (
              <Text c="dimmed" size="sm">
                {t("recognitionLog.empty")}
              </Text>
            ) : (
              recognizedTexts.map((entry) => (
                <Paper
                  key={entry.id}
                  data-log-row-id={recognitionSourceRowId(entry.source)}
                  p="xs"
                  withBorder
                  radius="sm"
                >
                  <Group wrap="nowrap" gap="sm">
                    <Text
                      style={{
                        flex: 1,
                        ...zeroMinWidth,
                        whiteSpace: "pre-wrap",
                        overflowWrap: "anywhere",
                      }}
                    >
                      {entry.text}
                    </Text>
                    {reserveLanguageBadge || entry.detected_language ? (
                      <Box
                        w={LANGUAGE_BADGE_SLOT_WIDTH}
                        miw={LANGUAGE_BADGE_SLOT_WIDTH}
                        style={{ display: "flex", justifyContent: "center" }}
                      >
                        {entry.detected_language ? (
                          <Badge color="blue" variant="light" size="sm">
                            {formatDetectedLanguage(entry.detected_language)}
                          </Badge>
                        ) : null}
                      </Box>
                    ) : null}
                    <Box
                      w={72}
                      miw={72}
                      style={{ display: "flex", justifyContent: "flex-end" }}
                    >
                      {!entry.is_final ? (
                        <Badge color="cyan" variant="light" size="sm">
                          {t("recognitionLog.partial")}
                        </Badge>
                      ) : (
                        <Text size="xs" c="dimmed" ta="right">
                          {formatLogTime(
                            entry.recognized_at_millis,
                            dateTimeLocale,
                          )}
                        </Text>
                      )}
                    </Box>
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
