import { Text, Tooltip } from "@mantine/core";
import type React from "react";
import type { ReactNode } from "react";

import type { ParapperConfig, RecognizedTextEvent } from "../../lib/types";

export const configuredLimit = (
  value: number | null | undefined,
  fallback: number,
) => (value === undefined ? fallback : value);

const LEVEL_METER_MIN_DB = -50;

const linearToDb = (linear: number, minDb = LEVEL_METER_MIN_DB) => {
  if (!Number.isFinite(linear) || linear <= 0) {
    return minDb;
  }
  return Math.max(20 * Math.log10(linear), minDb);
};

export const levelToProgress = (linear: number) => {
  const db = linearToDb(linear);
  return ((db - LEVEL_METER_MIN_DB) / -LEVEL_METER_MIN_DB) * 100;
};

export const trimRecognizedTextLog = (
  texts: RecognizedTextEvent[],
  recognitionLogLimit: number | null,
  debugAudioLogLimit: number | null,
) => {
  const trimmed =
    recognitionLogLimit === null
      ? texts
      : texts.slice(-Math.max(1, recognitionLogLimit));
  if (debugAudioLogLimit === null) return trimmed;

  const firstDebugAudioIndex = Math.max(
    0,
    trimmed.length - Math.max(0, debugAudioLogLimit),
  );
  return trimmed.map((entry, index) =>
    index < firstDebugAudioIndex && entry.debug_asr_audio_samples?.length
      ? {
          ...entry,
          debug_asr_audio_sample_rate: null,
          debug_asr_audio_samples: null,
        }
      : entry,
  );
};

export const settingLabel = (label: string, description: string) => (
  <Tooltip label={description} multiline w={280}>
    <Text span size="sm" fw={500} style={{ cursor: "help" }}>
      {label}
    </Text>
  </Tooltip>
);

export const DisabledReasonTooltip: React.FC<{
  disabled: boolean;
  label: string;
  children: ReactNode;
}> = ({ disabled, label, children }) => (
  <Tooltip label={label} disabled={!disabled} multiline w={280}>
    <span style={{ display: "block", width: "100%" }}>{children}</span>
  </Tooltip>
);

export const configsEqual = (
  left: ParapperConfig | null,
  right: ParapperConfig | null,
) => JSON.stringify(left) === JSON.stringify(right);
