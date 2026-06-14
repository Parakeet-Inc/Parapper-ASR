import { Badge, Box, Group, Paper, Stack, Text, Title } from "@mantine/core";
import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";

import {
  fullSizeZeroMin,
  zeroMinHeight,
  zeroMinWidth,
} from "../lib/layout-styles";
import { recognitionSourceRowId } from "../lib/recognition-source";
import { notificationColor } from "../lib/theme";
import type {
  AsrLanguage,
  AsrModel,
  ParapperConfig,
  RecognitionSourceMeta,
  RecognizedTextEvent,
  TranslationMapping,
  TranslationTextEvent,
} from "../lib/types";

type TranslationSidePanelProps = {
  config: ParapperConfig;
  recognizedTexts: RecognizedTextEvent[];
  translatedTexts: TranslationTextEvent[];
};

export const TranslationSidePanel: React.FC<TranslationSidePanelProps> = ({
  config,
  recognizedTexts,
  translatedTexts,
}) => {
  return (
    <Stack gap="md" style={fullSizeZeroMin}>
      <TranslationLogPanel
        config={config}
        recognizedTexts={recognizedTexts}
        translatedTexts={translatedTexts}
      />
    </Stack>
  );
};

type PendingTranslationEntry = {
  kind: "pending";
  id: string;
  source_recognition_id: string;
  source: RecognitionSourceMeta;
  target_lang: string;
};

type ReadyTranslationEntry = {
  kind: "ready";
  event: TranslationTextEvent;
};

type PlaceholderTranslationEntry = {
  kind: "placeholder";
  id: string;
};

type TranslationLogEntry =
  | PendingTranslationEntry
  | ReadyTranslationEntry
  | PlaceholderTranslationEntry;

type TranslationLogRow = {
  rowId: string;
  entries: TranslationLogEntry[];
};

const TranslationLogPanel: React.FC<{
  config: ParapperConfig;
  recognizedTexts: RecognizedTextEvent[];
  translatedTexts: TranslationTextEvent[];
}> = ({ config, recognizedTexts, translatedTexts }) => {
  const { t } = useTranslation();
  const logRef = useRef<HTMLDivElement | null>(null);
  const rows = useMemo(
    () => buildTranslationLogRows(config, recognizedTexts, translatedTexts),
    [config, recognizedTexts, translatedTexts],
  );

  useEffect(() => {
    const logElement = logRef.current;
    if (!logElement) return;
    logElement.scrollTop = logElement.scrollHeight;
  }, [rows]);

  return (
    <Paper
      withBorder
      radius="sm"
      p="md"
      style={{ height: "100%", ...zeroMinHeight, overflow: "hidden" }}
    >
      <Stack h="100%" gap="sm" style={zeroMinHeight}>
        <Group justify="space-between" align="center">
          <Title order={4}>{t("translationLog.title")}</Title>
        </Group>
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
            {rows.length === 0 ? (
              <Text c="dimmed" size="sm">
                {t("translationLog.empty")}
              </Text>
            ) : (
              rows.map((row) => (
                <Paper
                  key={row.rowId}
                  data-log-row-id={
                    row.entries.length === 1 ? row.rowId : undefined
                  }
                  p="xs"
                  withBorder
                  radius="sm"
                >
                  <Stack gap={4}>
                    {row.entries.map((entry) => (
                      <TranslationLogEntryRow
                        key={translationEntryKey(entry)}
                        entry={entry}
                      />
                    ))}
                  </Stack>
                </Paper>
              ))
            )}
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );
};

const TranslationLogEntryRow: React.FC<{ entry: TranslationLogEntry }> = ({
  entry,
}) => {
  const { t } = useTranslation();
  if (entry.kind === "placeholder") {
    return <Box style={{ minHeight: "1.5em" }} />;
  }

  const isPending = entry.kind === "pending";
  const targetLang = isPending ? entry.target_lang : entry.event.target_lang;
  const text = isPending
    ? ""
    : entry.event.translated_text || entry.event.error || "";
  const status = isPending ? "pending" : entry.event.status;

  return (
    <Group
      align="center"
      wrap="nowrap"
      gap="sm"
      style={{ opacity: isPending ? 0.55 : 1 }}
    >
      <Text
        style={{
          flex: 1,
          ...zeroMinWidth,
          minHeight: "1.5em",
          whiteSpace: "pre-wrap",
          overflowWrap: "anywhere",
        }}
      >
        {text}
      </Text>
      <Badge
        color={
          status === "failure" ? notificationColor.warn : notificationColor.info
        }
        variant="light"
        size="sm"
      >
        {targetLang}
      </Badge>
      {status === "failure" ? (
        <Badge color={notificationColor.warn} variant="light">
          {t("translationLog.errorBadge")}
        </Badge>
      ) : null}
      <Text size="xs" c="dimmed" w={72} ta="right">
        {isPending ? "" : `${entry.event.elapsed_millis} ms`}
      </Text>
    </Group>
  );
};

const buildTranslationLogRows = (
  config: ParapperConfig,
  recognizedTexts: RecognizedTextEvent[],
  translatedTexts: TranslationTextEvent[],
): TranslationLogRow[] => {
  const translatedBySourceTarget = new Map<string, TranslationTextEvent>();
  for (const translated of translatedTexts) {
    translatedBySourceTarget.set(translationMapKey(translated), translated);
  }

  const usedTranslationIds = new Set<string>();
  const rows: TranslationLogRow[] = recognizedTexts.flatMap((recognized) => {
    const targets = translationTargetsForRecognizedText(
      config.translation_mappings,
      recognized.source_asr_model,
      recognized.source_language,
    );
    if (targets.length === 0) {
      return [];
    }

    const rowId = recognitionSourceRowId(recognized.source);
    const showPending = shouldShowPendingTranslationForRecognizedText(
      config,
      recognized,
    );
    const entries = targets.flatMap((target_lang): TranslationLogEntry[] => {
      const translated = translatedBySourceTarget.get(
        translationSourceTargetKey(recognized.source, target_lang),
      );
      if (translated) {
        usedTranslationIds.add(translated.id);
        return [{ kind: "ready", event: translated }];
      }
      if (!showPending) {
        return [];
      }
      return [
        {
          kind: "pending",
          id: `${rowId}|${target_lang}`,
          source_recognition_id: recognized.id,
          source: recognized.source,
          target_lang,
        },
      ];
    });
    if (entries.length === 0) {
      if (shouldReservePlaceholderTranslationRow(config, recognized)) {
        return [
          {
            rowId,
            entries: [{ kind: "placeholder", id: `${rowId}|placeholder` }],
          },
        ];
      }
      return [];
    }
    return [{ rowId, entries }];
  });
  const rowsById = new Map(rows.map((row) => [row.rowId, row]));

  const orphanRows = translatedTexts
    .filter((translated) => !usedTranslationIds.has(translated.id))
    .reduce<Map<string, TranslationLogEntry[]>>((grouped, translated) => {
      const rowId = recognitionSourceRowId(translated.source);
      const entries = grouped.get(rowId) ?? [];
      entries.push({ kind: "ready", event: translated });
      grouped.set(rowId, entries);
      return grouped;
    }, new Map());

  for (const [rowId, entries] of orphanRows) {
    const existing = rowsById.get(rowId);
    if (existing) {
      existing.entries.push(...entries);
    } else {
      rows.push({ rowId, entries });
    }
  }

  return rows;
};

const shouldShowPendingTranslationForRecognizedText = (
  config: ParapperConfig,
  recognized: RecognizedTextEvent,
) =>
  config.translation_enabled &&
  (config.translation_send_timing === "interim" || recognized.is_final);

const shouldReservePlaceholderTranslationRow = (
  config: ParapperConfig,
  recognized: RecognizedTextEvent,
) =>
  config.translation_enabled &&
  config.translation_send_timing === "final" &&
  !recognized.is_final;

const translationTargetsForRecognizedText = (
  mappings: TranslationMapping[],
  sourceAsrModel: AsrModel,
  sourceLanguage: AsrLanguage,
) => {
  const seen = new Set<string>();
  return mappings
    .filter(
      (mapping) =>
        mapping.source_asr_model === null ||
        mapping.source_asr_model === sourceAsrModel,
    )
    .map((mapping) => mapping.target_lang.trim())
    .filter((target) => target.length > 0)
    .filter(
      (target) =>
        !translationTargetMatchesSourceLanguage(target, sourceLanguage),
    )
    .filter((target) => {
      if (seen.has(target)) {
        return false;
      }
      seen.add(target);
      return true;
    });
};

const translationTargetMatchesSourceLanguage = (
  target: string,
  sourceLanguage: AsrLanguage,
) => {
  const normalized = target.replace(/[A-Z]/g, (letter) => letter.toLowerCase());
  if (sourceLanguage === "japanese") {
    return normalized.startsWith("ja");
  }
  if (sourceLanguage === "english") {
    return normalized.startsWith("en");
  }
  return false;
};

const translationMapKey = (event: TranslationTextEvent) =>
  translationSourceTargetKey(event.source, event.target_lang);

const translationSourceTargetKey = (
  source: RecognitionSourceMeta,
  targetLang: string,
) => `${recognitionSourceRowId(source)}|${targetLang}`;

const translationEntryKey = (entry: TranslationLogEntry) =>
  entry.kind === "ready" ? entry.event.id : entry.id;
