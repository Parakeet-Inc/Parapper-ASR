import { configWithAsrModel } from "./asr-mode";
import { asrModelOption } from "./constants";
import type {
  AsrLanguage,
  AsrModel,
  AudioDeviceInfo,
  ParapperConfig,
  RecognitionSourceConfig,
  SttProfileConfig,
  SttProfileDisplayColor,
} from "./types";

const DEFAULT_PROFILE_ID = "stt-profile-1";
export const DEFAULT_STT_PROFILE_DISPLAY_COLOR: SttProfileDisplayColor =
  "green";
export const STT_PROFILE_DISPLAY_COLORS: readonly SttProfileDisplayColor[] = [
  "green",
  "blue",
  "violet",
  "red",
  "orange",
  "yellow",
  "white",
];

export const sttProfileDisplayName = (
  profile: Pick<SttProfileConfig, "id" | "name">,
  defaultName: (ordinal: number) => string,
) => {
  if (profile.name !== profile.id) return profile.name;
  const match = /^stt-profile-(\d+)$/.exec(profile.id);
  return match ? defaultName(Number(match[1])) : profile.name;
};

export const resolveSttProfileNameEdit = (
  profiles: readonly Pick<SttProfileConfig, "id" | "name">[],
  profile: Pick<SttProfileConfig, "id" | "name">,
  draft: string,
  defaultName: (ordinal: number) => string,
) => {
  const name = draft.trim();
  if (
    !name ||
    name === profile.name ||
    name === sttProfileDisplayName(profile, defaultName)
  ) {
    return null;
  }
  const duplicate = profiles.some(
    (candidate) =>
      candidate.id !== profile.id &&
      sttProfileDisplayName(candidate, defaultName) === name,
  );
  return duplicate ? null : name;
};

export const sttProfileListPresentation = (
  profiles: readonly SttProfileConfig[],
  selectedProfileId: string,
  defaultName: (ordinal: number) => string,
) =>
  profiles.map((profile) => ({
    id: profile.id,
    label: sttProfileDisplayName(profile, defaultName),
    color: profile.display_color,
    enabled: profile.enabled,
    selected: profile.id === selectedProfileId,
  }));

const sttProfileAsrFromLegacy = (
  config: ParapperConfig,
  model = config.asr_model,
  interimModel = config.interim_asr_model,
  language: AsrLanguage = config.asr_language,
): SttProfileConfig["asr"] => ({
  language,
  model,
  interim_model: interimModel,
  precision: config.asr_precision,
  num_threads: config.asr_num_threads,
  mode: config.asr_mode,
  hotwords_enabled: config.asr_hotwords_enabled,
  hotwords: config.asr_hotwords,
  normalize_input_audio: config.asr_normalize_input_audio,
  multilingual_enabled: config.multilingual_asr_enabled,
  enabled_models: config.enabled_asr_models,
  // Runtime routing is flattened into the profile's model fields. Nested
  // runtime profiles are not accepted by the profile schema.
  runtime_profiles: [],
});

const sttProfileBaseFromLegacy = (
  config: ParapperConfig,
): Pick<SttProfileConfig, "noise_cancellation" | "segmentation" | "turn"> => ({
  noise_cancellation: {
    enabled: config.noise_cancellation_enabled,
    model: config.noise_cancellation_model,
    target: config.noise_cancellation_target,
  },
  segmentation: {
    vad_threshold: config.vad_threshold,
    vad_interval_ms: config.vad_interval_ms,
    segment_start_speech_ms: config.segment_start_speech_ms,
  },
  turn: {
    detector: config.turn_detector,
    interim_result_enabled: config.interim_result_enabled,
    interim_result_silence_ms: config.interim_result_silence_ms,
    check_silence_ms: config.turn_check_silence_ms,
    namo_confidence_threshold: config.namo_turn_confidence_threshold,
    namo_context_max_tokens: config.namo_context_max_tokens,
    rerecognize_full_on_complete: config.turn_rerecognize_full_on_complete,
  },
});

const inputVolumePercentFromLegacy = (config: ParapperConfig) =>
  Math.round(
    Math.min(100, Math.max(0, 100 * 10 ** (config.input_volume_db / 20))),
  );

export const virtualSttProfileFromLegacy = (
  config: ParapperConfig,
): SttProfileConfig => ({
  id: DEFAULT_PROFILE_ID,
  name: DEFAULT_PROFILE_ID,
  enabled: true,
  neo_http_enabled: true,
  developer_http_enabled: true,
  display_color: DEFAULT_STT_PROFILE_DISPLAY_COLOR,
  input: {
    device_host: config.input_device_host,
    device_id: config.input_device_id,
    device_name: config.input_device_name,
    channel_index: 0,
    volume_percent: inputVolumePercentFromLegacy(config),
    muted: config.input_muted ?? false,
  },
  ...sttProfileBaseFromLegacy(config),
  asr: sttProfileAsrFromLegacy(config),
  delivery_profile_id: null,
});

const asrModelForRuntime = (
  config: ParapperConfig,
  runtimeId: string | null | undefined,
) =>
  runtimeId
    ? config.asr_runtime_profiles?.find((runtime) => runtime.id === runtimeId)
        ?.model
    : undefined;

const virtualSttProfileFromRecognitionSource = (
  config: ParapperConfig,
  source: RecognitionSourceConfig,
): SttProfileConfig => {
  const completionModel =
    asrModelForRuntime(
      config,
      source.asr_route_policy?.completion_runtime_id,
    ) ?? config.asr_model;
  const interimModel = source.asr_route_policy
    ? (asrModelForRuntime(config, source.asr_route_policy.interim_runtime_id) ??
      null)
    : config.interim_asr_model;

  return {
    id: source.source_id,
    name: source.speaker_label,
    enabled: true,
    neo_http_enabled: true,
    developer_http_enabled: true,
    display_color: DEFAULT_STT_PROFILE_DISPLAY_COLOR,
    input: {
      device_host: config.capture_endpoint?.device_host ?? null,
      device_id: config.capture_endpoint?.device_id ?? null,
      device_name: config.capture_endpoint?.device_name ?? null,
      channel_index: source.channel_index,
      volume_percent: inputVolumePercentFromLegacy(config),
      muted: config.input_muted ?? false,
    },
    ...sttProfileBaseFromLegacy(config),
    asr: sttProfileAsrFromLegacy(
      config,
      completionModel,
      interimModel,
      asrModelOption(completionModel).language,
    ),
    delivery_profile_id: source.delivery_profile_id ?? null,
  };
};

export const effectiveSttProfiles = (config: ParapperConfig) => {
  if (config.stt_profiles?.length) return config.stt_profiles;
  if (config.capture_endpoint && config.recognition_sources?.length) {
    return config.recognition_sources.map((source) =>
      virtualSttProfileFromRecognitionSource(config, source),
    );
  }
  return [virtualSttProfileFromLegacy(config)];
};

export const materializeSttProfiles = (
  config: ParapperConfig,
  profiles = effectiveSttProfiles(config),
): ParapperConfig => ({
  ...config,
  input_source_kind: "desktop_audio",
  capture_endpoint: null,
  recognition_sources: [],
  asr_runtime_profiles: [],
  stt_profiles: [...profiles],
});

export const updateSttProfile = (
  config: ParapperConfig,
  profileId: string,
  update: (profile: SttProfileConfig) => SttProfileConfig,
): ParapperConfig =>
  materializeSttProfiles(
    config,
    effectiveSttProfiles(config).map((profile) =>
      profile.id === profileId ? update(profile) : profile,
    ),
  );

export const nextProfileId = (profiles: readonly SttProfileConfig[]) => {
  let ordinal = profiles.length + 1;
  let id = `stt-profile-${ordinal}`;
  while (profiles.some((profile) => profile.id === id)) {
    ordinal += 1;
    id = `stt-profile-${ordinal}`;
  }
  return id;
};

export const nextAvailableInputChannel = (
  profiles: readonly SttProfileConfig[],
  inputAudioDevices: readonly AudioDeviceInfo[],
  preferred: SttProfileConfig["input"],
) => {
  const preferredDevice = inputAudioDevices.find(
    (device) =>
      device.host === preferred.device_host &&
      device.id === preferred.device_id,
  );
  const candidates = preferredDevice
    ? [preferredDevice]
    : inputAudioDevices.filter((device) => device.channels > 0);

  for (const device of candidates) {
    for (
      let channelIndex = 0;
      channelIndex < device.channels;
      channelIndex += 1
    ) {
      const used = profiles.some(
        (profile) =>
          profile.input.device_host === device.host &&
          profile.input.device_id === device.id &&
          profile.input.channel_index === channelIndex,
      );
      if (!used) {
        return {
          device_host: device.host,
          device_id: device.id,
          device_name: device.display_name,
          channel_index: channelIndex,
          volume_percent: 100,
          muted: false,
        };
      }
    }
  }
  return null;
};

export const occupiedInputChannelsForDevice = (
  profiles: readonly SttProfileConfig[],
  selectedProfileId: string,
  input: SttProfileConfig["input"],
) =>
  new Set(
    profiles
      .filter(
        (profile) =>
          profile.id !== selectedProfileId &&
          profile.input.device_host === input.device_host &&
          profile.input.device_id === input.device_id,
      )
      .map((profile) => profile.input.channel_index),
  );

export type InputChannelChoice = {
  channelIndex: number;
  occupied: boolean;
};

export const buildInputChannelRows = (
  profiles: readonly SttProfileConfig[],
  selectedProfileId: string,
  device: AudioDeviceInfo,
  channelsPerRow = 16,
): InputChannelChoice[][] => {
  const occupied = occupiedInputChannelsForDevice(profiles, selectedProfileId, {
    device_host: device.host,
    device_id: device.id,
    device_name: device.display_name,
    channel_index: 0,
    volume_percent: 100,
    muted: false,
  });
  const choices = Array.from(
    { length: device.channels },
    (_, channelIndex) => ({
      channelIndex,
      occupied: occupied.has(channelIndex),
    }),
  );

  return Array.from(
    { length: Math.ceil(choices.length / channelsPerRow) },
    (_, rowIndex) =>
      choices.slice(rowIndex * channelsPerRow, (rowIndex + 1) * channelsPerRow),
  );
};

export const availableInputChannelForProfile = (
  profiles: readonly SttProfileConfig[],
  selectedProfileId: string,
  device: AudioDeviceInfo,
) => {
  const selected = profiles.find((profile) => profile.id === selectedProfileId);
  if (
    selected?.input.device_host === device.host &&
    selected.input.device_id === device.id &&
    selected.input.channel_index < device.channels
  ) {
    return selected.input.channel_index;
  }

  const occupied = occupiedInputChannelsForDevice(profiles, selectedProfileId, {
    device_host: device.host,
    device_id: device.id,
    device_name: device.display_name,
    channel_index: 0,
    volume_percent: 100,
    muted: false,
  });
  return (
    Array.from({ length: device.channels }, (_, channelIndex) =>
      occupied.has(channelIndex) ? null : channelIndex,
    ).find((channelIndex) => channelIndex !== null) ?? null
  );
};

export const addSttProfile = (
  config: ParapperConfig,
  selectedProfileId: string,
  inputAudioDevices: readonly AudioDeviceInfo[],
): ParapperConfig | null => {
  const profiles = effectiveSttProfiles(config);
  const selected =
    profiles.find((profile) => profile.id === selectedProfileId) ?? profiles[0];
  const input = nextAvailableInputChannel(
    profiles,
    inputAudioDevices,
    selected.input,
  );
  if (!input) return null;

  const id = nextProfileId(profiles);
  return materializeSttProfiles(config, [
    ...profiles,
    {
      ...selected,
      id,
      name: id,
      enabled: true,
      neo_http_enabled: true,
      developer_http_enabled: true,
      display_color: DEFAULT_STT_PROFILE_DISPLAY_COLOR,
      input,
      noise_cancellation: { ...selected.noise_cancellation },
      segmentation: { ...selected.segmentation },
      turn: { ...selected.turn },
      asr: {
        ...selected.asr,
        hotwords: structuredClone(selected.asr.hotwords),
        enabled_models: [...selected.asr.enabled_models],
        runtime_profiles: [],
      },
    },
  ]);
};

export const deleteSttProfile = (
  config: ParapperConfig,
  profileId: string,
): { config: ParapperConfig; selectedProfileId: string } | null => {
  const profiles = effectiveSttProfiles(config);
  if (profiles.length <= 1) return null;
  const index = profiles.findIndex((profile) => profile.id === profileId);
  if (index < 0) return null;
  const remaining = profiles.filter((profile) => profile.id !== profileId);
  const selectedProfileId = remaining[Math.min(index, remaining.length - 1)].id;
  const profilesWithAnEnabledProfile = remaining.some(
    (profile) => profile.enabled,
  )
    ? remaining
    : remaining.map((profile) =>
        profile.id === selectedProfileId
          ? { ...profile, enabled: true }
          : profile,
      );
  return {
    config: materializeSttProfiles(config, profilesWithAnEnabledProfile),
    selectedProfileId,
  };
};

/**
 * Updates a profile's activation without permitting a persisted profile mode
 * that has no capture lane left to run.
 */
export const setSttProfileEnabled = (
  config: ParapperConfig,
  profileId: string,
  enabled: boolean,
): ParapperConfig => {
  const profiles = effectiveSttProfiles(config);
  const profile = profiles.find((candidate) => candidate.id === profileId);
  if (!profile || profile.enabled === enabled) return config;
  if (
    !enabled &&
    profiles.filter((candidate) => candidate.enabled).length === 1
  ) {
    return config;
  }
  return materializeSttProfiles(
    config,
    profiles.map((candidate) =>
      candidate.id === profileId ? { ...candidate, enabled } : candidate,
    ),
  );
};

/** Adapts one nested STT profile to legacy editor components during migration. */
export const sttProfileEditorConfig = (
  config: ParapperConfig,
  profile: SttProfileConfig,
): ParapperConfig => ({
  ...config,
  input_device_host: profile.input.device_host,
  input_device_id: profile.input.device_id,
  input_device_name: profile.input.device_name,
  input_muted: profile.input.muted,
  noise_cancellation_enabled: profile.noise_cancellation.enabled,
  noise_cancellation_model: profile.noise_cancellation.model,
  noise_cancellation_target: profile.noise_cancellation.target,
  vad_threshold: profile.segmentation.vad_threshold,
  vad_interval_ms: profile.segmentation.vad_interval_ms,
  segment_start_speech_ms: profile.segmentation.segment_start_speech_ms,
  turn_detector: profile.turn.detector,
  interim_result_enabled: profile.turn.interim_result_enabled,
  interim_result_silence_ms: profile.turn.interim_result_silence_ms,
  turn_check_silence_ms: profile.turn.check_silence_ms,
  namo_turn_confidence_threshold: profile.turn.namo_confidence_threshold,
  namo_context_max_tokens: profile.turn.namo_context_max_tokens,
  turn_rerecognize_full_on_complete: profile.turn.rerecognize_full_on_complete,
  asr_language: profile.asr.language,
  asr_model: profile.asr.model,
  interim_asr_model: profile.asr.interim_model,
  asr_precision: profile.asr.precision,
  asr_num_threads: profile.asr.num_threads,
  asr_mode: profile.asr.mode,
  asr_hotwords_enabled: profile.asr.hotwords_enabled,
  asr_hotwords: profile.asr.hotwords,
  asr_normalize_input_audio: profile.asr.normalize_input_audio,
  multilingual_asr_enabled: profile.asr.multilingual_enabled,
  enabled_asr_models: profile.asr.enabled_models,
  asr_runtime_profiles: profile.asr.runtime_profiles,
});

export const updateSttProfileFromEditorField = <K extends keyof ParapperConfig>(
  profile: SttProfileConfig,
  key: K,
  value: ParapperConfig[K],
): SttProfileConfig => {
  switch (key) {
    case "noise_cancellation_enabled":
      return {
        ...profile,
        noise_cancellation: {
          ...profile.noise_cancellation,
          enabled: value as boolean,
        },
      };
    case "noise_cancellation_model":
      return {
        ...profile,
        noise_cancellation: {
          ...profile.noise_cancellation,
          model: value as SttProfileConfig["noise_cancellation"]["model"],
        },
      };
    case "noise_cancellation_target":
      return {
        ...profile,
        noise_cancellation: {
          ...profile.noise_cancellation,
          target: value as SttProfileConfig["noise_cancellation"]["target"],
        },
      };
    case "vad_threshold":
    case "vad_interval_ms":
    case "segment_start_speech_ms":
      return {
        ...profile,
        segmentation: {
          ...profile.segmentation,
          [key]: value,
        },
      } as SttProfileConfig;
    case "turn_detector":
    case "interim_result_enabled":
    case "interim_result_silence_ms":
    case "turn_check_silence_ms":
    case "namo_turn_confidence_threshold":
    case "namo_context_max_tokens":
    case "turn_rerecognize_full_on_complete": {
      const turnKey =
        key === "turn_check_silence_ms"
          ? "check_silence_ms"
          : key === "namo_turn_confidence_threshold"
            ? "namo_confidence_threshold"
            : key === "namo_context_max_tokens"
              ? "namo_context_max_tokens"
              : key === "turn_rerecognize_full_on_complete"
                ? "rerecognize_full_on_complete"
                : key === "turn_detector"
                  ? "detector"
                  : key;
      return {
        ...profile,
        turn: { ...profile.turn, [turnKey]: value },
      } as SttProfileConfig;
    }
    case "asr_language":
    case "asr_model":
    case "interim_asr_model":
    case "asr_precision":
    case "asr_num_threads":
    case "asr_mode":
    case "asr_hotwords_enabled":
    case "asr_hotwords":
    case "asr_normalize_input_audio":
    case "multilingual_asr_enabled":
    case "enabled_asr_models":
    case "asr_runtime_profiles": {
      const asrKey =
        key === "asr_language"
          ? "language"
          : key === "asr_model"
            ? "model"
            : key === "interim_asr_model"
              ? "interim_model"
              : key === "asr_precision"
                ? "precision"
                : key === "asr_num_threads"
                  ? "num_threads"
                  : key === "asr_normalize_input_audio"
                    ? "normalize_input_audio"
                    : key === "multilingual_asr_enabled"
                      ? "multilingual_enabled"
                      : key === "enabled_asr_models"
                        ? "enabled_models"
                        : key === "asr_runtime_profiles"
                          ? "runtime_profiles"
                          : key.replace("asr_", "");
      return {
        ...profile,
        asr: { ...profile.asr, [asrKey]: value },
      } as SttProfileConfig;
    }
    default:
      return profile;
  }
};

export const sttProfileWithAsrModel = (
  profile: SttProfileConfig,
  model: AsrModel,
): SttProfileConfig => {
  const selected = configWithAsrModel(
    {
      asr_language: profile.asr.language,
      asr_model: profile.asr.model,
      asr_precision: profile.asr.precision,
      asr_mode: profile.asr.mode,
    },
    model,
  );
  return {
    ...profile,
    asr: {
      ...profile.asr,
      language: selected.asr_language,
      model: selected.asr_model,
      precision: selected.asr_precision,
      mode: selected.asr_mode,
    },
  };
};
