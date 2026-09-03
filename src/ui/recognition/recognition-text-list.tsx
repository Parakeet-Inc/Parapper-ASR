import { Badge, Box, Group, Paper, Stack, Text } from "@mantine/core";
import type { ReactNode } from "react";

import { zeroMinWidth } from "../../lib/layout-styles";
import { formatLogTime } from "../../lib/recognition-log-csv";
import { recognitionSourceRowId } from "../../lib/recognition-source";
import { STT_PROFILE_DISPLAY_COLOR_CSS } from "../../lib/stt-profile-colors";
import type {
  RecognizedTextEvent,
  SttProfileDisplayColor,
} from "../../lib/types";

const LANGUAGE_BADGE_SLOT_WIDTH = 44;

const entryKey = (entry: RecognizedTextEvent) =>
  entry.update_mode === "replace"
    ? recognitionSourceRowId(entry.source)
    : `${entry.id}:${entry.source.output_sequence}`;

type RecognitionTextListProps = {
  entries: RecognizedTextEvent[];
  reserveLanguageBadge?: boolean;
  dateTimeLocale: string;
  emptyLabel: string;
  partialLabel: string;
  sourceColor?: (entry: RecognizedTextEvent) => SttProfileDisplayColor;
  renderActions?: (entry: RecognizedTextEvent) => ReactNode;
};

export const RecognitionTextList: React.FC<RecognitionTextListProps> = ({
  entries,
  reserveLanguageBadge = false,
  dateTimeLocale,
  emptyLabel,
  partialLabel,
  sourceColor,
  renderActions,
}) => (
  <Stack gap="xs">
    {entries.length === 0 ? (
      <Text c="dimmed" size="sm">
        {emptyLabel}
      </Text>
    ) : (
      entries.map((entry) => (
        <Paper
          key={entryKey(entry)}
          data-log-row-id={recognitionSourceRowId(entry.source)}
          p="xs"
          withBorder
          radius="sm"
          pos="relative"
          style={{ overflow: "hidden" }}
        >
          {sourceColor ? (
            <Box
              aria-hidden
              data-recognition-source-color
              pos="absolute"
              top={0}
              bottom={0}
              left={0}
              w={3}
              style={{
                backgroundColor:
                  STT_PROFILE_DISPLAY_COLOR_CSS[sourceColor(entry)],
              }}
            />
          ) : null}
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
                    {entry.detected_language.toUpperCase()}
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
                  {partialLabel}
                </Badge>
              ) : (
                <Text size="xs" c="dimmed" ta="right">
                  {formatLogTime(entry.recognized_at_millis, dateTimeLocale)}
                </Text>
              )}
            </Box>
            <Text size="xs" c="dimmed" w={72} ta="right">
              {entry.elapsed_millis} ms
            </Text>
            {renderActions?.(entry)}
          </Group>
        </Paper>
      ))
    )}
  </Stack>
);
