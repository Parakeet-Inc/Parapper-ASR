import {
  ActionIcon,
  Alert,
  Box,
  Button,
  Group,
  Select,
  SegmentedControl,
  Slider,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { MappingList } from "./mapping-list";
import { useAsrModelOptions } from "../../hooks/use-asr-model-options";
import { useVoiceList } from "../../hooks/use-voice-list";
import { buildAudioDeviceOptions } from "../../lib/audio-devices";
import {
  languageOptions,
  makeId,
  modelOptionsWithAny,
} from "../../lib/mapping-options";
import { isMacOs } from "../../lib/platform";
import type {
  AudioDeviceInfo,
  AsrModel,
  ParapperConfig,
  LocalTtsVoice,
  SpeechBackend,
  SpeechMapping,
  SpeechSourceKind,
} from "../../lib/types";

import IconVolumeOff from "~icons/material-symbols/volume-off";
import IconVolumeUp from "~icons/material-symbols/volume-up";

type SpeechSettingsProps = {
  config: ParapperConfig;
  outputAudioDevices: AudioDeviceInfo[];
  runtimeLocked: boolean;
  neoReadAloudDelaySuspected: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
};

export const SpeechSettings: React.FC<SpeechSettingsProps> = ({
  config,
  outputAudioDevices,
  runtimeLocked,
  neoReadAloudDelaySuspected,
  onUpdateConfig,
}) => {
  const { t } = useTranslation();
  const { voiceList, refreshingVoiceList, refreshVoiceList } = useVoiceList(
    config.translation_plugin_http_port,
  );
  const shouldShowNeoReadAloudWarning =
    neoReadAloudDelaySuspected &&
    config.neo_http_enabled &&
    config.translation_enabled &&
    config.translation_mappings.length > 0 &&
    config.speech_mappings.some(
      (mapping) =>
        !mapping.muted &&
        mapping.source_kind === "translation" &&
        mapping.backend === "ync" &&
        mapping.talker.trim() !== "",
    );

  return (
    <Stack gap="sm">
      {shouldShowNeoReadAloudWarning ? (
        <Alert color="yellow" variant="light">
          {t("speechSettings.neoReadAloudWarning")}
        </Alert>
      ) : null}
      <SpeechMappingRows
        mappings={config.speech_mappings}
        config={config}
        outputAudioDevices={outputAudioDevices}
        voiceList={voiceList}
        refreshingVoiceList={refreshingVoiceList}
        modelLocked={runtimeLocked}
        onRefreshVoiceList={() => void refreshVoiceList()}
        onChange={(speechMappings) =>
          onUpdateConfig("speech_mappings", speechMappings)
        }
      />
    </Stack>
  );
};

const buildVoiceOptions = (
  voiceList: string[],
  selectedVoice?: string | null,
) => {
  const voices = new Set(voiceList);
  if (selectedVoice) {
    voices.add(selectedVoice);
  }
  return [...voices].map((voice) => ({ value: voice, label: voice }));
};

const localTtsVoiceOptions: { value: LocalTtsVoice; label: string }[] = [
  {
    value: "vits_piper_en_US_kristin_medium",
    label: "piper-voices en_US Kristin medium",
  },
  {
    value: "vits_piper_en_US_john_medium",
    label: "piper-voices en_US John medium",
  },
  {
    value: "vits_piper_en_US_norman_medium",
    label: "piper-voices en_US Norman medium",
  },
  {
    value: "supertonic_2_onnx",
    label: "Supertonic 2 ONNX",
  },
  {
    value: "supertonic_3_onnx",
    label: "Supertonic 3 ONNX",
  },
];

const defaultLocalTtsVoice: LocalTtsVoice = "vits_piper_en_US_kristin_medium";
const supertonic2TtsVoice: LocalTtsVoice = "supertonic_2_onnx";
const supertonic3TtsVoice: LocalTtsVoice = "supertonic_3_onnx";
const supertonic2LanguageOptions = [
  { value: "en", label: "English" },
  { value: "ko", label: "Korean" },
  { value: "es", label: "Spanish" },
  { value: "pt", label: "Portuguese" },
  { value: "fr", label: "French" },
];
const supertonic3LanguageOptions = [
  { value: "en", label: "English" },
  { value: "ko", label: "Korean" },
  { value: "ja", label: "Japanese" },
  { value: "bg", label: "Bulgarian" },
  { value: "cs", label: "Czech" },
  { value: "da", label: "Danish" },
  { value: "el", label: "Greek" },
  { value: "es", label: "Spanish" },
  { value: "et", label: "Estonian" },
  { value: "fi", label: "Finnish" },
  { value: "hu", label: "Hungarian" },
  { value: "it", label: "Italian" },
  { value: "nl", label: "Dutch" },
  { value: "pl", label: "Polish" },
  { value: "pt", label: "Portuguese" },
  { value: "ro", label: "Romanian" },
  { value: "ar", label: "Arabic" },
  { value: "de", label: "German" },
  { value: "fr", label: "French" },
  { value: "hi", label: "Hindi" },
  { value: "id", label: "Indonesian" },
  { value: "ru", label: "Russian" },
  { value: "vi", label: "Vietnamese" },
];
const supertonicSpeakerOptions = Array.from({ length: 10 }, (_, speakerId) => ({
  value: String(speakerId),
  label: speakerId < 5 ? `F${speakerId + 1}` : `M${speakerId - 4}`,
}));

const ttsVoiceLanguageOptions = (voice: LocalTtsVoice) => {
  if (voice === supertonic2TtsVoice) {
    return supertonic2LanguageOptions;
  }
  if (voice === supertonic3TtsVoice) {
    return supertonic3LanguageOptions;
  }
  return null;
};

const ttsVoiceSpeakerOptions = (voice: LocalTtsVoice) => {
  if (voice === supertonic2TtsVoice || voice === supertonic3TtsVoice) {
    return supertonicSpeakerOptions;
  }
  return null;
};

const defaultTtsVoiceLanguage = (_voice: LocalTtsVoice) => "en";

const SpeechMappingRows: React.FC<{
  mappings: SpeechMapping[];
  config: ParapperConfig;
  outputAudioDevices: AudioDeviceInfo[];
  voiceList: string[];
  refreshingVoiceList: boolean;
  modelLocked: boolean;
  onRefreshVoiceList: () => void;
  onChange: (mappings: SpeechMapping[]) => void;
}> = ({
  mappings,
  config,
  outputAudioDevices,
  voiceList,
  refreshingVoiceList,
  modelLocked,
  onRefreshVoiceList,
  onChange,
}) => {
  const { t } = useTranslation();
  const outputAudioDeviceOptions = useMemo(
    () => buildAudioDeviceOptions(outputAudioDevices),
    [outputAudioDevices],
  );
  const { asrModelSelectOptions } = useAsrModelOptions(config.asr_model);
  const asrModelOptions = modelOptionsWithAny(
    t("speechSettings.mapping.anyModel"),
    asrModelSelectOptions,
  );
  const defaultTranslationTarget =
    config.translation_mappings[0]?.target_lang ?? "en_US";
  const yncPluginAvailable = !isMacOs();
  const backendOptions = [
    ...(yncPluginAvailable
      ? [
          {
            value: "ync",
            label: t("speechSettings.backend.ync"),
          },
        ]
      : []),
    {
      value: "local_tts",
      label: t("speechSettings.backend.localTts"),
    },
  ];
  const defaultBackend: SpeechBackend = yncPluginAvailable
    ? "ync"
    : "local_tts";

  return (
    <MappingList
      rows={mappings}
      addLabel={t("speechSettings.mapping.addRow")}
      moveUpLabel={t("speechSettings.mapping.moveUp")}
      moveDownLabel={t("speechSettings.mapping.moveDown")}
      deleteLabel={t("speechSettings.mapping.deleteRow")}
      mutationDisabled={modelLocked}
      renderBeforeDeleteActions={(mapping, updateMapping) => (
        <Tooltip
          label={
            mapping.muted
              ? t("speechSettings.mapping.unmute")
              : t("speechSettings.mapping.mute")
          }
        >
          <ActionIcon
            aria-label={
              mapping.muted
                ? t("speechSettings.mapping.unmute")
                : t("speechSettings.mapping.mute")
            }
            variant="outline"
            color="gray"
            style={{
              borderColor: "var(--mantine-color-default-border)",
            }}
            onClick={() => updateMapping({ muted: !mapping.muted })}
          >
            {mapping.muted ? <IconVolumeOff /> : <IconVolumeUp />}
          </ActionIcon>
        </Tooltip>
      )}
      createRow={() => ({
        id: makeId("speech"),
        source_kind: "recognition" as const,
        source_asr_model: null,
        target_lang: null,
        backend: defaultBackend,
        talker: "",
        local_tts_voice: defaultLocalTtsVoice,
        local_tts_language: null,
        local_tts_speaker_id: null,
        output_device_id: null,
        output_device_host: null,
        output_device_name: null,
        muted: false,
        volume: 0,
      })}
      onChange={onChange}
      renderHeader={(mapping, updateMapping) => (
        <SegmentedControl
          value={mapping.source_kind}
          disabled={modelLocked}
          data={[
            {
              value: "recognition",
              label: t("speechSettings.mapping.sourceKind.recognition"),
            },
            {
              value: "translation",
              label: t("speechSettings.mapping.sourceKind.translation"),
            },
          ]}
          onChange={(value) =>
            updateMapping({
              source_kind: value as SpeechSourceKind,
              target_lang:
                value === "translation"
                  ? (mapping.target_lang ?? defaultTranslationTarget)
                  : null,
              source_asr_model:
                value === "recognition" ? mapping.source_asr_model : null,
            })
          }
        />
      )}
      renderBody={(mapping, updateMapping) => {
        const backend =
          !yncPluginAvailable && mapping.backend === "ync"
            ? "local_tts"
            : mapping.backend;

        return (
          <>
            {mapping.source_kind === "recognition" ? (
              <Select
                label={t("speechSettings.mapping.sourceModel")}
                data={asrModelOptions}
                value={mapping.source_asr_model ?? "any"}
                allowDeselect={false}
                disabled={modelLocked}
                onChange={(value) =>
                  updateMapping({
                    source_asr_model:
                      value && value !== "any" ? (value as AsrModel) : null,
                  })
                }
              />
            ) : (
              <Select
                label={t("speechSettings.mapping.targetLang")}
                data={languageOptions}
                value={mapping.target_lang ?? defaultTranslationTarget}
                searchable
                allowDeselect={false}
                disabled={modelLocked}
                onChange={(value) =>
                  updateMapping({
                    target_lang: value ?? defaultTranslationTarget,
                  })
                }
              />
            )}
            <Select
              label={t("speechSettings.backend.label")}
              data={backendOptions}
              value={backend}
              allowDeselect={false}
              disabled={modelLocked}
              onChange={(value) => {
                const backend = (value ?? defaultBackend) as SpeechBackend;
                const selectedLocalTtsVoice =
                  backend === "local_tts"
                    ? (mapping.local_tts_voice ?? defaultLocalTtsVoice)
                    : mapping.local_tts_voice;
                const localTtsVoice =
                  backend === "local_tts"
                    ? (selectedLocalTtsVoice ?? defaultLocalTtsVoice)
                    : null;
                updateMapping({
                  backend,
                  local_tts_voice: localTtsVoice,
                  local_tts_language:
                    localTtsVoice && ttsVoiceLanguageOptions(localTtsVoice)
                      ? (mapping.local_tts_language ??
                        defaultTtsVoiceLanguage(localTtsVoice))
                      : null,
                  local_tts_speaker_id:
                    localTtsVoice && ttsVoiceSpeakerOptions(localTtsVoice)
                      ? (mapping.local_tts_speaker_id ?? 0)
                      : null,
                });
              }}
            />
            {backend === "local_tts" ? (
              <>
                <Select
                  label={t("speechSettings.localTtsVoice.label")}
                  data={localTtsVoiceOptions}
                  value={mapping.local_tts_voice ?? defaultLocalTtsVoice}
                  allowDeselect={false}
                  disabled={modelLocked}
                  onChange={(value) => {
                    const localTtsVoice =
                      (value as LocalTtsVoice | null) ?? defaultLocalTtsVoice;
                    updateMapping({
                      local_tts_voice: localTtsVoice,
                      local_tts_language: ttsVoiceLanguageOptions(localTtsVoice)
                        ? defaultTtsVoiceLanguage(localTtsVoice)
                        : null,
                      local_tts_speaker_id: ttsVoiceSpeakerOptions(
                        localTtsVoice,
                      )
                        ? 0
                        : null,
                    });
                  }}
                />
                {ttsVoiceLanguageOptions(
                  mapping.local_tts_voice ?? defaultLocalTtsVoice,
                ) ? (
                  <Group grow>
                    <Select
                      label={t("speechSettings.localTtsLanguage.label")}
                      data={
                        ttsVoiceLanguageOptions(
                          mapping.local_tts_voice ?? defaultLocalTtsVoice,
                        ) ?? []
                      }
                      value={
                        mapping.local_tts_language ??
                        defaultTtsVoiceLanguage(
                          mapping.local_tts_voice ?? defaultLocalTtsVoice,
                        )
                      }
                      allowDeselect={false}
                      onChange={(value) =>
                        updateMapping({
                          local_tts_language:
                            value ??
                            defaultTtsVoiceLanguage(
                              mapping.local_tts_voice ?? defaultLocalTtsVoice,
                            ),
                        })
                      }
                    />
                    <Select
                      label={t("speechSettings.localTtsSpeaker.label")}
                      data={
                        ttsVoiceSpeakerOptions(
                          mapping.local_tts_voice ?? defaultLocalTtsVoice,
                        ) ?? []
                      }
                      value={String(mapping.local_tts_speaker_id ?? 0)}
                      allowDeselect={false}
                      onChange={(value) =>
                        updateMapping({
                          local_tts_speaker_id: Number(value ?? 0),
                        })
                      }
                    />
                  </Group>
                ) : null}
                <Select
                  label={t("speechSettings.outputAudioDevice.label")}
                  placeholder={t(
                    "speechSettings.outputAudioDevice.placeholder",
                  )}
                  data={outputAudioDeviceOptions}
                  value={
                    mapping.output_device_host && mapping.output_device_id
                      ? `${mapping.output_device_host}\u0000${mapping.output_device_id}`
                      : null
                  }
                  clearable
                  searchable
                  maxDropdownHeight={180}
                  disabled={modelLocked}
                  onChange={(value) => {
                    if (!value) {
                      updateMapping({
                        output_device_id: null,
                        output_device_host: null,
                        output_device_name: null,
                      });
                      return;
                    }

                    const [host, id] = value.split("\u0000");
                    const device = outputAudioDevices.find(
                      (candidate) =>
                        candidate.host === host && candidate.id === id,
                    );
                    updateMapping({
                      output_device_id: id,
                      output_device_host: host,
                      output_device_name: device?.display_name ?? null,
                    });
                  }}
                />
              </>
            ) : (
              <Group align="end" gap="xs" wrap="nowrap">
                <Select
                  label={t("speechSettings.talker.label")}
                  data={buildVoiceOptions(voiceList, mapping.talker)}
                  value={mapping.talker || null}
                  clearable
                  searchable
                  nothingFoundMessage={t("speechSettings.talker.empty")}
                  style={{ flex: 1 }}
                  onChange={(value) => updateMapping({ talker: value ?? "" })}
                />
                <Button
                  variant="light"
                  loading={refreshingVoiceList}
                  onClick={onRefreshVoiceList}
                >
                  {t("speechSettings.talker.refresh")}
                </Button>
              </Group>
            )}
            <VolumeSlider
              label={t("speechSettings.volume.label")}
              value={mapping.volume}
              onChange={(value) => updateMapping({ volume: value })}
            />
          </>
        );
      }}
    />
  );
};

const VolumeSlider: React.FC<{
  label: string;
  value: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}> = ({ label, value, disabled = false, onChange }) => (
  <Stack gap={4}>
    <Group justify="space-between" wrap="nowrap">
      <Text size="sm" fw={500}>
        {label}
      </Text>
      <Text size="sm" c="dimmed" style={{ fontVariantNumeric: "tabular-nums" }}>
        {value.toFixed(1)} dB
      </Text>
    </Group>
    <Box px={20} pb={12}>
      <Slider
        value={value}
        min={-20}
        max={20}
        step={1}
        marks={[
          {
            value: -20,
            label: (
              <Text size="xs" ta="left" w={40}>
                -20
              </Text>
            ),
          },
          {
            value: 0,
            label: (
              <Text size="xs" ta="center" w={40}>
                0
              </Text>
            ),
          },
          {
            value: 20,
            label: (
              <Text size="xs" ta="right" w={40}>
                20
              </Text>
            ),
          },
        ]}
        disabled={disabled}
        label={(sliderValue) => `${sliderValue} dB`}
        onChange={onChange}
      />
    </Box>
  </Stack>
);
