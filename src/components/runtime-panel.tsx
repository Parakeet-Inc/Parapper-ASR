import {
  ActionIcon,
  Box,
  Button,
  Flex,
  Paper,
  Progress,
  Slider,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { RecognitionLog } from "./recognition-log";
import type { RuntimeState } from "../hooks/use-app-state";
import { useSyncedLogRowHeights } from "../hooks/use-synced-log-row-heights";
import { zeroMinSize, zeroMinWidth } from "../lib/layout-styles";
import { STT_PROFILE_DISPLAY_COLOR_CSS } from "../lib/stt-profile-colors";
import { sttProfileDisplayName } from "../lib/stt-profiles";
import { notificationColor } from "../lib/theme";
import type {
  ParapperConfig,
  RecognizedTextEvent,
  SttProfileConfig,
} from "../lib/types";

import IconVolumeOff from "~icons/material-symbols/volume-off";
import IconVolumeUp from "~icons/material-symbols/volume-up";

type RuntimePanelProps = {
  config: ParapperConfig;
  profiles: SttProfileConfig[];
  recognizedTexts: RecognizedTextEvent[];
  runtime: RuntimeState;
  translationPanel?: ReactNode;
  dateTimeLocale: string;
  canStartRecognition: boolean;
  canClearLogs: boolean;
  downloadingModels: boolean;
  fileExportAvailable: boolean;
  recognitionControlAvailable: boolean;
  speechControlAvailable: boolean;
  onClearRecognizedTexts: () => void;
  onSetProfileVolume: (id: string, value: number) => void;
  onToggleProfileMute: (id: string) => void;
  onOpenModelDownload: () => void;
  onStartRecognition: () => Promise<void>;
  onStopRecognition: () => Promise<void>;
  onStopSpeech: () => Promise<void>;
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

export const activeSttProfiles = (profiles: readonly SttProfileConfig[]) =>
  profiles.filter((profile) => profile.enabled);

export const RuntimePanel: React.FC<RuntimePanelProps> = ({
  config,
  profiles,
  recognizedTexts,
  runtime,
  translationPanel,
  dateTimeLocale,
  canStartRecognition,
  canClearLogs,
  downloadingModels,
  fileExportAvailable,
  recognitionControlAvailable,
  speechControlAvailable,
  onClearRecognizedTexts,
  onSetProfileVolume,
  onToggleProfileMute,
  onOpenModelDownload,
  onStartRecognition,
  onStopRecognition,
  onStopSpeech,
  onSaveRecognitionCsv,
  onSaveAsrInputWav,
  onPlayAudio,
}) => {
  const { t } = useTranslation();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const hasTranslationPanel = Boolean(translationPanel);
  const hasYncSpeech =
    config.neo_http_enabled &&
    config.speech_mappings.some(
      (mapping) =>
        !mapping.muted &&
        mapping.backend === "ync" &&
        mapping.talker.trim() !== "",
    );
  const persistedProfileMode = (config.stt_profiles?.length ?? 0) > 0;
  const activeProfiles = activeSttProfiles(profiles);
  useSyncedLogRowHeights(panelRef, hasTranslationPanel);

  return (
    <Box
      ref={panelRef}
      style={{
        flex: "1 1 0",
        ...zeroMinSize,
        display: "grid",
        gridTemplateColumns: hasTranslationPanel
          ? "minmax(0, 1fr) minmax(0, 1fr)"
          : "minmax(0, 1fr)",
        gridTemplateRows: "minmax(0, 1fr) auto",
        gap: "var(--mantine-spacing-md)",
      }}
    >
      <Box style={zeroMinSize}>
        <RecognitionLog
          asrWarning={runtime.asrWarning}
          recognizedTexts={recognizedTexts}
          profiles={profiles}
          reserveLanguageBadge={profiles.some(
            (profile) => profile.asr.multilingual_enabled,
          )}
          dateTimeLocale={dateTimeLocale}
          canClearLogs={canClearLogs}
          fileExportAvailable={fileExportAvailable}
          onClear={onClearRecognizedTexts}
          onSaveRecognitionCsv={onSaveRecognitionCsv}
          onSaveAsrInputWav={onSaveAsrInputWav}
          onPlayAudio={onPlayAudio}
        />
      </Box>

      {hasTranslationPanel ? (
        <Box style={zeroMinSize}>{translationPanel}</Box>
      ) : null}

      <Paper
        withBorder
        radius="sm"
        p="md"
        style={{
          gridColumn: hasTranslationPanel ? "1 / span 2" : "1",
          ...zeroMinWidth,
        }}
      >
        <Stack gap="md">
          <Flex align="end" gap="xs" wrap="wrap">
            <Button
              color="primary"
              variant={runtime.running ? "outline" : "filled"}
              miw={96}
              style={{ flexShrink: 0, whiteSpace: "nowrap" }}
              loading={
                runtime.starting ||
                (!runtime.running && !canStartRecognition && downloadingModels)
              }
              disabled={!recognitionControlAvailable}
              onClick={() => {
                if (runtime.starting) {
                  return;
                }
                if (runtime.running) {
                  void onStopRecognition();
                  return;
                }
                if (!canStartRecognition) {
                  onOpenModelDownload();
                  return;
                }
                void onStartRecognition();
              }}
            >
              {runtime.running
                ? t("common.stop")
                : canStartRecognition
                  ? t("common.start")
                  : t("settings.downloadModels.openButton")}
            </Button>
            {hasYncSpeech && speechControlAvailable ? (
              <Button
                variant="default"
                miw={128}
                style={{ flexShrink: 0, whiteSpace: "nowrap" }}
                onClick={() => void onStopSpeech()}
              >
                {t("speechSettings.stopButton")}
              </Button>
            ) : null}
          </Flex>
          <Stack gap="xs">
            {activeProfiles.map((profile) => (
              <SttProfileVolumeRow
                key={profile.id}
                profile={profile}
                level={
                  runtime.inputLevelsBySource[profile.id] ??
                  (!persistedProfileMode
                    ? {
                        inputLevel: runtime.inputLevel,
                        inputLevelBeforeGain: runtime.inputLevelBeforeGain,
                      }
                    : { inputLevel: 0, inputLevelBeforeGain: 0 })
                }
                displayName={sttProfileDisplayName(profile, (number) =>
                  t("sttProfiles.defaultName", { number }),
                )}
                onSetVolume={onSetProfileVolume}
                onToggleMute={onToggleProfileMute}
              />
            ))}
          </Stack>
        </Stack>
      </Paper>
    </Box>
  );
};

type SttProfileVolumeRowProps = {
  profile: SttProfileConfig;
  displayName: string;
  level: {
    inputLevel: number;
    inputLevelBeforeGain: number;
  };
  onSetVolume: (id: string, value: number) => void;
  onToggleMute: (id: string) => void;
};

export const SttProfileVolumeRow: React.FC<SttProfileVolumeRowProps> = ({
  profile,
  displayName,
  level,
  onSetVolume,
  onToggleMute,
}) => {
  const { t } = useTranslation();
  const [volumeDraft, setVolumeDraft] = useState(profile.input.volume_percent);
  useEffect(() => {
    setVolumeDraft(profile.input.volume_percent);
  }, [profile.id, profile.input.volume_percent]);

  const inputLevelColor = getInputLevelColor(
    level.inputLevel,
    level.inputLevelBeforeGain,
  );
  const muteLabel = profile.input.muted
    ? t("sttProfiles.unmute")
    : t("sttProfiles.mute");

  return (
    <Flex align="center" gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
      <Box
        aria-hidden
        data-stt-profile-color={profile.display_color}
        w={10}
        h={10}
        style={{
          flex: "0 0 auto",
          borderRadius: "50%",
          backgroundColor: STT_PROFILE_DISPLAY_COLOR_CSS[profile.display_color],
        }}
      />
      <Text
        size="sm"
        fw={500}
        title={displayName}
        style={{
          flex: "0 1 9rem",
          minWidth: 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {displayName}
      </Text>
      <Tooltip label={muteLabel} withArrow openDelay={300}>
        <ActionIcon
          aria-label={`${displayName}: ${muteLabel}`}
          variant={profile.input.muted ? "filled" : "default"}
          color={profile.input.muted ? "red" : "gray"}
          onClick={() => onToggleMute(profile.id)}
        >
          {profile.input.muted ? <IconVolumeOff /> : <IconVolumeUp />}
        </ActionIcon>
      </Tooltip>
      <Slider
        aria-label={`${displayName}: ${t("sttProfiles.volumeLabel")}`}
        value={volumeDraft}
        min={0}
        max={100}
        step={1}
        label={(value) => `${value}%`}
        style={{ flex: "1 1 10rem", minWidth: 96 }}
        onChange={setVolumeDraft}
        onChangeEnd={(value) => onSetVolume(profile.id, value)}
      />
      <Box
        role="img"
        aria-label={`${displayName}: ${t("sttProfiles.inputLevel")}`}
        style={{ flex: "1 1 9rem", minWidth: 64 }}
      >
        <MeterWithThresholds
          linear={level.inputLevel}
          color={inputLevelColor}
          thresholds={inputMeterThresholds}
        />
      </Box>
      <Text
        size="xs"
        c="dimmed"
        ta="right"
        style={{ width: 2.8 * 16, fontVariantNumeric: "tabular-nums" }}
      >
        {volumeDraft}%
      </Text>
    </Flex>
  );
};

const INPUT_PRE_GAIN_ERROR_DB = -1;
const INPUT_LEVEL_WARN_DB = 0;
const LEVEL_METER_MIN_DB = -50;
const METER_MAX_DB = 5;
const METER_THRESHOLD_COLOR = "#868e96";

const inputMeterThresholds = [
  { db: INPUT_LEVEL_WARN_DB, color: METER_THRESHOLD_COLOR },
];

const linearToDb = (linear: number, minDb = LEVEL_METER_MIN_DB) => {
  if (!Number.isFinite(linear) || linear <= 0) {
    return minDb;
  }
  return Math.max(20 * Math.log10(linear), minDb);
};

const meterDbToProgress = (db: number) => {
  const clampedDb = Math.max(LEVEL_METER_MIN_DB, Math.min(METER_MAX_DB, db));
  return (
    ((clampedDb - LEVEL_METER_MIN_DB) / (METER_MAX_DB - LEVEL_METER_MIN_DB)) *
    100
  );
};

const getInputLevelColor = (postGainLinear: number, preGainLinear: number) => {
  const preGainDb = linearToDb(preGainLinear, LEVEL_METER_MIN_DB);
  if (preGainDb > INPUT_PRE_GAIN_ERROR_DB) {
    return notificationColor.error;
  }
  const postGainDb = linearToDb(postGainLinear, LEVEL_METER_MIN_DB);
  if (postGainDb > INPUT_LEVEL_WARN_DB) {
    return notificationColor.warn;
  }
  return notificationColor.primary;
};

const MeterWithThresholds: React.FC<{
  linear: number;
  color: string;
  thresholds: Array<{ db: number; color: string }>;
}> = ({ linear, color, thresholds }) => (
  <Box pos="relative">
    <Progress
      value={meterDbToProgress(linearToDb(linear, LEVEL_METER_MIN_DB))}
      color={color}
      size="sm"
    />
    {thresholds.map((threshold, idx) => (
      <Box
        key={`${threshold.db}-${idx}`}
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `calc(${meterDbToProgress(threshold.db)}% - 1px)`,
          width: "2px",
          backgroundColor: threshold.color,
          opacity: 0.95,
          pointerEvents: "none",
        }}
      />
    ))}
  </Box>
);
