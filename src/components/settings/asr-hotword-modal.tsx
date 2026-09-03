import {
  ActionIcon,
  Alert,
  Button,
  Divider,
  Group,
  Modal,
  NumberInput,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  hasInvalidHotwords,
  mergeSuggestedHotwordReadings,
  sanitizeHotwords,
  splitHotwordReadings,
  validateHotwords,
} from "../../lib/asr-hotwords";
import type { HotwordValidation } from "../../lib/asr-hotwords";
import type { AsrHotword } from "../../lib/types";

import IconAdd from "~icons/material-symbols/add";
import IconAutoFixHigh from "~icons/material-symbols/auto-fix-high";
import IconDelete from "~icons/material-symbols/delete";

type AsrHotwordModalProps = {
  opened: boolean;
  hotwords: AsrHotword[];
  disabled: boolean;
  onClose: () => void;
  onSave: (hotwords: AsrHotword[]) => void;
  onSuggestReadings: (surface: string) => Promise<string[]>;
};

const cloneHotwords = (hotwords: AsrHotword[]) =>
  hotwords.map((hotword) => ({
    ...hotword,
    readings: [...hotword.readings],
  }));

const emptyHotword = (): AsrHotword => ({
  surface: "",
  readings: [],
  score: null,
});

export const AsrHotwordModal: React.FC<AsrHotwordModalProps> = ({
  opened,
  hotwords,
  disabled,
  onClose,
  onSave,
  onSuggestReadings,
}) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<AsrHotword[]>([]);
  const [autofillRows, setAutofillRows] = useState<Set<number>>(new Set());
  const [autofillStatus, setAutofillStatus] = useState<
    Record<number, "empty" | "error" | undefined>
  >({});
  const [saveValidation, setSaveValidation] = useState<
    HotwordValidation[] | null
  >(null);

  useEffect(() => {
    if (opened) {
      setDraft(cloneHotwords(hotwords));
      setAutofillRows(new Set());
      setAutofillStatus({});
      setSaveValidation(null);
    }
  }, [opened, hotwords]);

  const editingValidation = validateHotwords(draft, {
    checkTerminalPrefixes: false,
  });
  const validation = saveValidation ?? editingValidation;
  const autofillBusy = autofillRows.size > 0;
  const updateHotword = (index: number, patch: Partial<AsrHotword>) => {
    setSaveValidation(null);
    setDraft((current) =>
      current.map((hotword, currentIndex) =>
        currentIndex === index ? { ...hotword, ...patch } : hotword,
      ),
    );
  };
  const deleteHotword = (index: number) => {
    setSaveValidation(null);
    setDraft((current) =>
      current.filter((_, currentIndex) => currentIndex !== index),
    );
  };
  const addHotword = () => {
    setSaveValidation(null);
    setDraft((current) => [...current, emptyHotword()]);
  };
  const autofillReadings = async (index: number) => {
    const surface = draft[index]?.surface.trim() ?? "";
    if (!surface || disabled) return;
    setSaveValidation(null);
    setAutofillRows((current) => new Set(current).add(index));
    setAutofillStatus((current) => ({ ...current, [index]: undefined }));
    try {
      const suggestions = await onSuggestReadings(surface);
      if (suggestions.length === 0) {
        setAutofillStatus((current) => ({ ...current, [index]: "empty" }));
        return;
      }
      setDraft((current) =>
        current.map((hotword, currentIndex) =>
          currentIndex === index && hotword.surface.trim() === surface
            ? {
                ...hotword,
                readings: mergeSuggestedHotwordReadings(
                  hotword.readings,
                  suggestions,
                ),
              }
            : hotword,
        ),
      );
    } catch {
      setAutofillStatus((current) => ({ ...current, [index]: "error" }));
    } finally {
      setAutofillRows((current) => {
        const next = new Set(current);
        next.delete(index);
        return next;
      });
    }
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={t("settings.hotwords.title")}
      centered
      size="xl"
      closeOnClickOutside={false}
    >
      <Stack gap="md">
        <Text size="sm" c="dimmed" style={{ whiteSpace: "pre-line" }}>
          {t("settings.hotwords.description")}
        </Text>
        <Table.ScrollContainer minWidth={720}>
          <Table withTableBorder withColumnBorders verticalSpacing="sm">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>{t("settings.hotwords.surfaceColumn")}</Table.Th>
                <Table.Th>{t("settings.hotwords.readingsColumn")}</Table.Th>
                <Table.Th w={120}>
                  {t("settings.hotwords.scoreColumn")}
                </Table.Th>
                <Table.Th w={88} />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {draft.length === 0 ? (
                <Table.Tr>
                  <Table.Td colSpan={4}>
                    <Text size="sm" c="dimmed" ta="center" py="md">
                      {t("settings.hotwords.empty")}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ) : null}
              {draft.map((hotword, index) => {
                const rowValidation = validation[index];
                const readingText = hotword.readings.join("\n");

                return (
                  <Table.Tr key={index} style={{ verticalAlign: "top" }}>
                    <Table.Td>
                      <TextInput
                        aria-label={t("settings.hotwords.surface")}
                        placeholder={t("settings.hotwords.surfacePlaceholder")}
                        value={hotword.surface}
                        disabled={disabled}
                        error={
                          rowValidation.emptySurface
                            ? t("settings.hotwords.surfaceRequired")
                            : rowValidation.duplicateSurface
                              ? t("settings.hotwords.duplicateSurface")
                              : rowValidation.pathCollision
                                ? t("settings.hotwords.pathCollision")
                                : rowValidation.terminalPrefix
                                  ? t("settings.hotwords.terminalPrefix")
                                  : undefined
                        }
                        onChange={(event) => {
                          setAutofillStatus((current) => ({
                            ...current,
                            [index]: undefined,
                          }));
                          updateHotword(index, {
                            surface: event.currentTarget.value,
                          });
                        }}
                      />
                    </Table.Td>
                    <Table.Td>
                      <Stack gap={4}>
                        <Group align="flex-start" wrap="nowrap" gap="xs">
                          <Textarea
                            aria-label={t("settings.hotwords.readings")}
                            placeholder={t(
                              "settings.hotwords.readingsPlaceholder",
                            )}
                            minRows={1}
                            maxRows={4}
                            autosize
                            value={readingText}
                            disabled={disabled}
                            style={{ flex: 1 }}
                            error={
                              rowValidation.blankReading
                                ? t("settings.hotwords.blankReading")
                                : rowValidation.duplicateReading
                                  ? t("settings.hotwords.duplicateReading")
                                  : undefined
                            }
                            onChange={(event) =>
                              updateHotword(index, {
                                readings: splitHotwordReadings(
                                  event.currentTarget.value,
                                ),
                              })
                            }
                          />
                          <Tooltip label={t("settings.hotwords.autofillHelp")}>
                            <Button
                              aria-label={t("settings.hotwords.autofill")}
                              variant="light"
                              size="xs"
                              px="xs"
                              leftSection={<IconAutoFixHigh />}
                              disabled={disabled || !hotword.surface.trim()}
                              loading={autofillRows.has(index)}
                              onClick={() => void autofillReadings(index)}
                            >
                              {t("settings.hotwords.autofill")}
                            </Button>
                          </Tooltip>
                        </Group>
                        {autofillStatus[index] ? (
                          <Text
                            size="xs"
                            c={
                              autofillStatus[index] === "error"
                                ? "red"
                                : "dimmed"
                            }
                          >
                            {t(
                              autofillStatus[index] === "error"
                                ? "settings.hotwords.autofillFailed"
                                : "settings.hotwords.autofillEmpty",
                            )}
                          </Text>
                        ) : null}
                      </Stack>
                    </Table.Td>
                    <Table.Td>
                      <NumberInput
                        aria-label={t("settings.hotwords.score")}
                        placeholder={t("settings.hotwords.scorePlaceholder")}
                        min={0.001}
                        step={0.1}
                        decimalScale={3}
                        allowDecimal
                        value={hotword.score ?? ""}
                        disabled={disabled}
                        error={
                          rowValidation.invalidScore
                            ? t("settings.hotwords.invalidScore")
                            : undefined
                        }
                        onChange={(value) =>
                          updateHotword(index, {
                            score:
                              typeof value === "number" &&
                              Number.isFinite(value)
                                ? value
                                : null,
                          })
                        }
                      />
                    </Table.Td>
                    <Table.Td>
                      <Group justify="flex-end">
                        <Tooltip label={t("settings.hotwords.delete")}>
                          <ActionIcon
                            aria-label={t("settings.hotwords.delete")}
                            variant="subtle"
                            color="red"
                            disabled={disabled || autofillBusy}
                            onClick={() => deleteHotword(index)}
                          >
                            <IconDelete />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
        {saveValidation?.some(({ terminalPrefix }) => terminalPrefix) ? (
          <Alert color="yellow" variant="light">
            {t("settings.hotwords.terminalPrefix")}
          </Alert>
        ) : null}
        <Button
          variant="light"
          leftSection={<IconAdd />}
          disabled={disabled || autofillBusy}
          onClick={addHotword}
        >
          {t("settings.hotwords.add")}
        </Button>
        <Divider />
        <Group justify="flex-end">
          <Button variant="subtle" onClick={onClose}>
            {t("settings.hotwords.cancel")}
          </Button>
          <Button
            onClick={() => {
              if (!disabled) {
                const sanitized = sanitizeHotwords(draft);
                const nextValidation = validateHotwords(sanitized);
                if (
                  nextValidation.some(({ terminalPrefix }) => terminalPrefix)
                ) {
                  setSaveValidation(nextValidation);
                  return;
                }
                onSave(sanitized);
              }
            }}
            disabled={
              disabled ||
              autofillBusy ||
              hasInvalidHotwords(draft, { checkTerminalPrefixes: false })
            }
          >
            {t("settings.hotwords.save")}
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
};
