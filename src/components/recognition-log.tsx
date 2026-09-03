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
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { zeroMinHeight } from "../lib/layout-styles";
import {
  buildRecognitionCsvExport,
  float32SamplesToWavBytes,
  formatCsvFileTimestamp,
} from "../lib/recognition-log-csv";
import { resolveRecognitionLogSourceColor } from "../lib/recognition-log-source";
import { notificationColor } from "../lib/theme";
import type { RecognizedTextEvent, SttProfileConfig } from "../lib/types";
import { RecognitionTextList } from "../ui/recognition/recognition-text-list";

import IconDownload from "~icons/material-symbols/download";
import IconPlayArrow from "~icons/material-symbols/play-arrow";

type RecognitionLogProps = {
  asrWarning: string | null;
  recognizedTexts: RecognizedTextEvent[];
  profiles: SttProfileConfig[];
  reserveLanguageBadge: boolean;
  dateTimeLocale: string;
  canClearLogs: boolean;
  fileExportAvailable: boolean;
  onClear: () => void;
  onSaveRecognitionCsv: (
    defaultFileName: string,
    content: string,
  ) => Promise<string | null>;
  onSaveAsrInputWav: (
    defaultFileName: string,
    content: number[],
  ) => Promise<string | null>;
  onPlayAudio: (samples: number[], sampleRate: number) => Promise<void>;
};

export const RecognitionLog: React.FC<RecognitionLogProps> = ({
  asrWarning,
  recognizedTexts,
  profiles,
  reserveLanguageBadge,
  dateTimeLocale,
  canClearLogs,
  fileExportAvailable,
  onClear,
  onSaveRecognitionCsv,
  onSaveAsrInputWav,
  onPlayAudio,
}) => {
  const { t } = useTranslation();
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const logElement = logRef.current;
    if (!logElement) return;
    logElement.scrollTop = logElement.scrollHeight;
  }, [recognizedTexts]);

  const exportRecognizedTextsAsCsv = async () => {
    const csvExport = buildRecognitionCsvExport(recognizedTexts, {
      text: t("recognitionLog.csvHeaderText"),
      time: t("recognitionLog.csvHeaderTime"),
      seconds: t("recognitionLog.csvHeaderSeconds"),
      elapsedMs: t("recognitionLog.csvHeaderElapsedMs"),
    });
    try {
      const path = await onSaveRecognitionCsv(
        csvExport.defaultFileName,
        csvExport.content,
      );
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

    await onPlayAudio(samples, sampleRate);
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
      const path = await onSaveAsrInputWav(
        `parapper-asr-input-${formatCsvFileTimestamp()}.wav`,
        Array.from(wavBytes),
      );
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
              disabled={!fileExportAvailable || recognizedTexts.length === 0}
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
          <RecognitionTextList
            entries={recognizedTexts}
            reserveLanguageBadge={reserveLanguageBadge}
            dateTimeLocale={dateTimeLocale}
            emptyLabel={t("recognitionLog.empty")}
            partialLabel={t("recognitionLog.partial")}
            sourceColor={(entry) =>
              resolveRecognitionLogSourceColor(
                profiles,
                entry.source.identity.source_id,
              )
            }
            renderActions={(entry) => (
              <>
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
                      disabled={
                        !fileExportAvailable ||
                        !entry.debug_asr_audio_samples?.length
                      }
                      onClick={() => void downloadAsrInputAudio(entry)}
                    >
                      <IconDownload />
                    </ActionIcon>
                  </span>
                </Tooltip>
              </>
            )}
          />
        </Box>
      </Stack>
    </Paper>
  );
};
