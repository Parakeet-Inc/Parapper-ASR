import {
  ActionIcon,
  Button,
  Group,
  Paper,
  Stack,
  Tooltip,
} from "@mantine/core";
import type { ReactNode } from "react";

import type { MappingRow } from "../../lib/mapping-row-utils";
import { moveById, removeById, updateById } from "../../lib/mapping-row-utils";

import IconAdd from "~icons/material-symbols/add";
import IconDelete from "~icons/material-symbols/delete";
import IconKeyboardArrowDown from "~icons/material-symbols/keyboard-arrow-down";
import IconKeyboardArrowUp from "~icons/material-symbols/keyboard-arrow-up";

type MappingListProps<T extends MappingRow> = {
  rows: T[];
  addLabel: string;
  moveUpLabel: string;
  moveDownLabel: string;
  deleteLabel: string;
  mutationDisabled?: boolean;
  createRow: () => T;
  onChange: (rows: T[]) => void;
  renderHeader: (row: T, updateRow: (patch: Partial<T>) => void) => ReactNode;
  renderBody: (row: T, updateRow: (patch: Partial<T>) => void) => ReactNode;
  renderBeforeDeleteActions?: (
    row: T,
    updateRow: (patch: Partial<T>) => void,
  ) => ReactNode;
};

export function MappingList<T extends MappingRow>({
  rows,
  addLabel,
  moveUpLabel,
  moveDownLabel,
  deleteLabel,
  mutationDisabled = false,
  createRow,
  onChange,
  renderHeader,
  renderBody,
  renderBeforeDeleteActions,
}: MappingListProps<T>) {
  const updateRow = (row: T, patch: Partial<T>) =>
    onChange(updateById(rows, row.id, patch));

  return (
    <Stack gap="xs">
      {rows.map((row, index) => (
        <Paper key={row.id} withBorder radius="sm" p="xs">
          <Stack gap="xs">
            <Group align="center" wrap="nowrap">
              {renderHeader(row, (patch) => updateRow(row, patch))}
              <Tooltip label={moveUpLabel}>
                <ActionIcon
                  aria-label={moveUpLabel}
                  variant="subtle"
                  disabled={mutationDisabled || index === 0}
                  onClick={() => onChange(moveById(rows, row.id, -1))}
                >
                  <IconKeyboardArrowUp />
                </ActionIcon>
              </Tooltip>
              <Tooltip label={moveDownLabel}>
                <ActionIcon
                  aria-label={moveDownLabel}
                  variant="subtle"
                  disabled={mutationDisabled || index === rows.length - 1}
                  onClick={() => onChange(moveById(rows, row.id, 1))}
                >
                  <IconKeyboardArrowDown />
                </ActionIcon>
              </Tooltip>
              {renderBeforeDeleteActions?.(row, (patch) =>
                updateRow(row, patch),
              )}
              <Tooltip label={deleteLabel}>
                <ActionIcon
                  aria-label={deleteLabel}
                  variant="outline"
                  color="red"
                  style={{
                    borderColor: "var(--mantine-color-default-border)",
                  }}
                  disabled={mutationDisabled}
                  onClick={() => onChange(removeById(rows, row.id))}
                >
                  <IconDelete />
                </ActionIcon>
              </Tooltip>
            </Group>
            {renderBody(row, (patch) => updateRow(row, patch))}
          </Stack>
        </Paper>
      ))}
      <Button
        variant="light"
        leftSection={<IconAdd />}
        disabled={mutationDisabled}
        onClick={() => onChange([...rows, createRow()])}
      >
        {addLabel}
      </Button>
    </Stack>
  );
}
