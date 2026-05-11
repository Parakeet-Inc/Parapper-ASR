import {
  Badge,
  Box,
  Button,
  Group,
  Paper,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import {
  fullSizeZeroMin,
  zeroMinHeight,
  zeroMinWidth,
} from "../lib/layout-styles";
import { notificationColor } from "../lib/theme";
import type { TranslationTextEvent } from "../lib/types";

type TranslationSidePanelProps = {
  translatedTexts: TranslationTextEvent[];
  onClearTranslatedTexts: () => void;
};

export const TranslationSidePanel: React.FC<TranslationSidePanelProps> = ({
  translatedTexts,
  onClearTranslatedTexts,
}) => {
  return (
    <Stack
      gap="md"
      style={fullSizeZeroMin}
    >
      <TranslationLogPanel
        translatedTexts={translatedTexts}
        onClear={onClearTranslatedTexts}
      />
    </Stack>
  );
};

const TranslationLogPanel: React.FC<{
  translatedTexts: TranslationTextEvent[];
  onClear: () => void;
}> = ({ translatedTexts, onClear }) => {
  const { t } = useTranslation();
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const logElement = logRef.current;
    if (!logElement) return;
    logElement.scrollTop = logElement.scrollHeight;
  }, [translatedTexts.length]);

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
          <Button
            variant="default"
            size="xs"
            disabled={translatedTexts.length === 0}
            onClick={onClear}
          >
            {t("common.resetLogs")}
          </Button>
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
            {translatedTexts.length === 0 ? (
              <Text c="dimmed" size="sm">
                {t("translationLog.empty")}
              </Text>
            ) : (
              translatedTexts.map((entry) => (
                <Paper
                  key={entry.id}
                  data-log-row-id={entry.source_recognition_id}
                  p="xs"
                  withBorder
                  radius="sm"
                >
                  <Group align="flex-start" wrap="nowrap" gap="sm">
                    <Text
                      style={{
                        flex: 1,
                        ...zeroMinWidth,
                        whiteSpace: "pre-wrap",
                        overflowWrap: "anywhere",
                      }}
                    >
                      {entry.translated_text || entry.error}
                    </Text>
                    <Badge
                      color={
                        entry.status === "success"
                          ? notificationColor.info
                          : notificationColor.warn
                      }
                      variant="light"
                      size="sm"
                    >
                      {entry.target_lang}
                    </Badge>
                    {entry.status === "failure" ? (
                      <Badge color={notificationColor.warn} variant="light">
                        {t("translationLog.errorBadge")}
                      </Badge>
                    ) : null}
                    <Text size="xs" c="dimmed" w={72} ta="right">
                      {entry.elapsed_millis} ms
                    </Text>
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
