import { describe, expect, it } from "vitest";

import {
  addSttProfile,
  availableInputChannelForProfile,
  buildInputChannelRows,
  deleteSttProfile,
  effectiveSttProfiles,
  occupiedInputChannelsForDevice,
  resolveSttProfileNameEdit,
  STT_PROFILE_DISPLAY_COLORS,
  sttProfileDisplayName,
  sttProfileListPresentation,
  setSttProfileEnabled,
  updateSttProfile,
} from "./stt-profiles";
import type { ParapperConfig } from "./types";

const legacyConfig = (): ParapperConfig =>
  ({
    input_device_host: "WASAPI",
    input_device_id: "interface",
    input_device_name: "Interface",
    input_volume_db: 0,
    input_muted: false,
    noise_cancellation_enabled: false,
    noise_cancellation_model: "ul_unas",
    noise_cancellation_target: "vad_only",
    vad_threshold: 0.5,
    vad_interval_ms: 100,
    segment_start_speech_ms: 200,
    turn_detector: "simple",
    interim_result_enabled: true,
    interim_result_silence_ms: 300,
    turn_check_silence_ms: 700,
    namo_turn_confidence_threshold: 0.5,
    namo_context_max_tokens: 256,
    turn_rerecognize_full_on_complete: false,
    asr_language: "japanese",
    asr_model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
    interim_asr_model: null,
    asr_precision: "int8",
    asr_num_threads: 4,
    asr_mode: "fast",
    asr_hotwords_enabled: false,
    asr_hotwords: [],
    asr_normalize_input_audio: true,
    multilingual_asr_enabled: false,
    enabled_asr_models: ["nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8"],
    asr_runtime_profiles: [],
  }) as unknown as ParapperConfig;

const explicitCaptureConfig = (): ParapperConfig => ({
  ...legacyConfig(),
  translation_enabled: true,
  capture_endpoint: {
    id: "interface-input",
    device_host: "WASAPI",
    device_id: "interface",
    device_name: "Interface",
  },
  recognition_sources: [
    {
      source_id: "speaker-a",
      speaker_label: "Alice",
      capture_endpoint_id: "interface-input",
      channel_index: 0,
      asr_route_policy: {
        completion_runtime_id: "ja-completion",
        interim_runtime_id: "en-interim",
      },
      delivery_profile_id: "alice-output",
    },
    {
      source_id: "speaker-b",
      speaker_label: "Bob",
      capture_endpoint_id: "interface-input",
      channel_index: 1,
      asr_route_policy: {
        completion_runtime_id: "en-completion",
        interim_runtime_id: null,
      },
      delivery_profile_id: "bob-output",
    },
  ],
  asr_runtime_profiles: [
    {
      id: "ja-completion",
      model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
    },
    {
      id: "en-interim",
      model: "nemotron_speech_streaming_en_0_6b_160ms_int8",
    },
    {
      id: "en-completion",
      model: "nemo_parakeet_tdt_0_6b_v2_int8",
    },
  ],
});

const interfaceDevice = {
  id: "interface",
  host: "WASAPI",
  display_name: "Interface",
  channels: 2,
  sample_rate: 48_000,
};

describe("persisted STT profiles", () => {
  it("uses the fixed seven-color profile palette with green as the default", () => {
    expect(STT_PROFILE_DISPLAY_COLORS).toEqual([
      "green",
      "blue",
      "violet",
      "red",
      "orange",
      "yellow",
      "white",
    ]);
    expect(effectiveSttProfiles(legacyConfig())[0].display_color).toBe("green");
    expect(effectiveSttProfiles(legacyConfig())[0].enabled).toBe(true);
  });

  it("keeps the final enabled profile active when an activation toggle would disable it", () => {
    const first = effectiveSttProfiles(legacyConfig())[0];
    const config = { ...legacyConfig(), stt_profiles: [first] };

    expect(setSttProfileEnabled(config, first.id, false)).toBe(config);
  });

  it("enables the newly selected remaining profile when deleting the only enabled profile", () => {
    const first = effectiveSttProfiles(legacyConfig())[0];
    const second = {
      ...first,
      id: "stt-profile-2",
      name: "stt-profile-2",
      enabled: false,
    };
    const config = { ...legacyConfig(), stt_profiles: [first, second] };

    expect(deleteSttProfile(config, first.id)).toMatchObject({
      selectedProfileId: second.id,
      config: { stt_profiles: [{ id: second.id, enabled: true }] },
    });
  });

  it("uses the localized default profile name consistently until the user renames it", () => {
    const profile = effectiveSttProfiles(legacyConfig())[0];

    expect(
      sttProfileDisplayName(profile, (number) => `Profile ${number}`),
    ).toBe("Profile 1");
    expect(
      sttProfileDisplayName(
        { ...profile, name: "Guest microphone" },
        (number) => `Profile ${number}`,
      ),
    ).toBe("Guest microphone");
  });

  it("does not persist an untouched localized default name or a visible duplicate", () => {
    const first = effectiveSttProfiles(legacyConfig())[0];
    const second = { ...first, id: "stt-profile-2", name: "stt-profile-2" };
    const defaultName = (number: number) => `Profile ${number}`;

    expect(
      resolveSttProfileNameEdit(
        [first, second],
        first,
        "Profile 1",
        defaultName,
      ),
    ).toBeNull();
    expect(
      resolveSttProfileNameEdit(
        [first, second],
        first,
        "Profile 2",
        defaultName,
      ),
    ).toBeNull();
    expect(
      resolveSttProfileNameEdit([first, second], first, " Guest ", defaultName),
    ).toBe("Guest");
  });

  it("shows every connection profile's active state alongside its selection and display color", () => {
    const first = effectiveSttProfiles(legacyConfig())[0];
    const profiles = [
      first,
      {
        ...first,
        id: "stt-profile-2",
        name: "Guest microphone",
        display_color: "violet" as const,
        enabled: false,
      },
    ];

    expect(
      sttProfileListPresentation(
        profiles,
        "stt-profile-2",
        (number) => `Profile ${number}`,
      ),
    ).toEqual([
      {
        id: "stt-profile-1",
        label: "Profile 1",
        color: "green",
        enabled: true,
        selected: false,
      },
      {
        id: "stt-profile-2",
        label: "Guest microphone",
        color: "violet",
        enabled: false,
        selected: true,
      },
    ]);
  });

  it("converts each explicit recognition source to a profile without losing routing identity", () => {
    expect(effectiveSttProfiles(explicitCaptureConfig())).toEqual([
      {
        id: "speaker-a",
        name: "Alice",
        enabled: true,
        neo_http_enabled: true,
        developer_http_enabled: true,
        display_color: "green",
        input: {
          device_host: "WASAPI",
          device_id: "interface",
          device_name: "Interface",
          channel_index: 0,
          volume_percent: 100,
          muted: false,
        },
        noise_cancellation: {
          enabled: false,
          model: "ul_unas",
          target: "vad_only",
        },
        segmentation: {
          vad_threshold: 0.5,
          vad_interval_ms: 100,
          segment_start_speech_ms: 200,
        },
        turn: {
          detector: "simple",
          interim_result_enabled: true,
          interim_result_silence_ms: 300,
          check_silence_ms: 700,
          namo_confidence_threshold: 0.5,
          namo_context_max_tokens: 256,
          rerecognize_full_on_complete: false,
        },
        asr: {
          language: "japanese",
          model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
          interim_model: "nemotron_speech_streaming_en_0_6b_160ms_int8",
          precision: "int8",
          num_threads: 4,
          mode: "fast",
          hotwords_enabled: false,
          hotwords: [],
          normalize_input_audio: true,
          multilingual_enabled: false,
          enabled_models: ["nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8"],
          runtime_profiles: [],
        },
        delivery_profile_id: "alice-output",
      },
      {
        id: "speaker-b",
        name: "Bob",
        enabled: true,
        neo_http_enabled: true,
        developer_http_enabled: true,
        display_color: "green",
        input: {
          device_host: "WASAPI",
          device_id: "interface",
          device_name: "Interface",
          channel_index: 1,
          volume_percent: 100,
          muted: false,
        },
        noise_cancellation: {
          enabled: false,
          model: "ul_unas",
          target: "vad_only",
        },
        segmentation: {
          vad_threshold: 0.5,
          vad_interval_ms: 100,
          segment_start_speech_ms: 200,
        },
        turn: {
          detector: "simple",
          interim_result_enabled: true,
          interim_result_silence_ms: 300,
          check_silence_ms: 700,
          namo_confidence_threshold: 0.5,
          namo_context_max_tokens: 256,
          rerecognize_full_on_complete: false,
        },
        asr: {
          language: "english",
          model: "nemo_parakeet_tdt_0_6b_v2_int8",
          interim_model: null,
          precision: "int8",
          num_threads: 4,
          mode: "fast",
          hotwords_enabled: false,
          hotwords: [],
          normalize_input_audio: true,
          multilingual_enabled: false,
          enabled_models: ["nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8"],
          runtime_profiles: [],
        },
        delivery_profile_id: "bob-output",
      },
    ]);
  });

  it("first explicit-source edit removes mixed capture fields but preserves output settings", () => {
    const config = explicitCaptureConfig();
    const updated = updateSttProfile(config, "speaker-a", (profile) => ({
      ...profile,
      input: { ...profile.input, muted: true },
    }));

    expect(updated.capture_endpoint).toBeNull();
    expect(updated.recognition_sources).toEqual([]);
    expect(updated.asr_runtime_profiles).toEqual([]);
    expect(updated.translation_enabled).toBe(true);
    expect(updated.stt_profiles?.map((profile) => profile.id)).toEqual([
      "speaker-a",
      "speaker-b",
    ]);
    expect(
      updated.stt_profiles?.map((profile) => profile.delivery_profile_id),
    ).toEqual(["alice-output", "bob-output"]);
    expect(
      updated.stt_profiles?.map((profile) => profile.asr.runtime_profiles),
    ).toEqual([[], []]);
  });

  it("first profile edit leaves legacy WebSocket input and starts profile desktop audio", () => {
    const updated = updateSttProfile(
      { ...explicitCaptureConfig(), input_source_kind: "web_socket" },
      "speaker-a",
      (profile) => ({
        ...profile,
        segmentation: { ...profile.segmentation, vad_threshold: 0.75 },
      }),
    );

    expect({
      input_source_kind: updated.input_source_kind,
      capture_endpoint: updated.capture_endpoint,
      recognition_sources: updated.recognition_sources,
      profile_ids: updated.stt_profiles?.map((profile) => profile.id),
    }).toEqual({
      input_source_kind: "desktop_audio",
      capture_endpoint: null,
      recognition_sources: [],
      profile_ids: ["speaker-a", "speaker-b"],
    });
  });

  it("materializes one legacy profile on the first profile edit", () => {
    const legacy = legacyConfig();
    const updated = updateSttProfile(legacy, "stt-profile-1", (profile) => ({
      ...profile,
      input: { ...profile.input, muted: true },
    }));

    expect(legacy.stt_profiles).toBeUndefined();
    expect(updated.stt_profiles).toEqual([
      expect.objectContaining({
        id: "stt-profile-1",
        input: expect.objectContaining({ muted: true, channel_index: 0 }),
      }),
    ]);
  });

  it("edits profile A without changing profile B", () => {
    const initial = legacyConfig();
    const added = addSttProfile(initial, "stt-profile-1", [
      {
        id: "interface",
        host: "WASAPI",
        display_name: "Interface",
        channels: 2,
        sample_rate: 48_000,
      },
    ])!;
    const updated = updateSttProfile(added, "stt-profile-1", (profile) => ({
      ...profile,
      segmentation: { ...profile.segmentation, vad_threshold: 0.9 },
    }));

    expect(
      effectiveSttProfiles(updated).map(
        (profile) => profile.segmentation.vad_threshold,
      ),
    ).toEqual([0.9, 0.5]);
  });

  it("updates each profile's Developer HTTP and NEO destinations independently", () => {
    const added = addSttProfile(legacyConfig(), "stt-profile-1", [
      {
        id: "interface",
        host: "WASAPI",
        display_name: "Interface",
        channels: 2,
        sample_rate: 48_000,
      },
    ])!;
    const updated = updateSttProfile(added, "stt-profile-1", (profile) => ({
      ...profile,
      developer_http_enabled: false,
      neo_http_enabled: false,
    }));

    expect(
      effectiveSttProfiles(updated).map((profile) => ({
        developerHttp: profile.developer_http_enabled,
        neoHttp: profile.neo_http_enabled,
      })),
    ).toEqual([
      { developerHttp: false, neoHttp: false },
      { developerHttp: true, neoHttp: true },
    ]);
  });

  it("adds the selected profile configuration on the next unused device channel", () => {
    const added = addSttProfile(legacyConfig(), "stt-profile-1", [
      {
        id: "interface",
        host: "WASAPI",
        display_name: "Interface",
        channels: 2,
        sample_rate: 48_000,
      },
    ]);

    expect(added?.stt_profiles).toEqual([
      expect.objectContaining({
        input: expect.objectContaining({ channel_index: 0 }),
      }),
      expect.objectContaining({
        id: "stt-profile-2",
        neo_http_enabled: true,
        developer_http_enabled: true,
        display_color: "green",
        input: expect.objectContaining({
          channel_index: 1,
          volume_percent: 100,
          muted: false,
        }),
      }),
    ]);
  });

  it("keeps global Developer HTTP enabled for every profile when adding a second source", () => {
    const config = {
      ...legacyConfig(),
      neo_http_enabled: true,
      streaming_recognition_enabled: true,
      developer_connection_mode: "http" as const,
      developer_http_url: "http://127.0.0.1:15522/api/events",
      translation_mappings: [
        {
          id: "translation-default",
          source_asr_model: null,
          backend: "ync" as const,
          local_model: "cat_translate_0_8b_q4_k_quant" as const,
          source_lang: "ja" as const,
          target_lang: "en" as const,
        },
      ],
      speech_mappings: [],
    };

    const added = addSttProfile(config, "stt-profile-1", [
      {
        id: "interface",
        host: "WASAPI",
        display_name: "Interface",
        channels: 2,
        sample_rate: 48_000,
      },
    ]);

    expect({
      developer_http_enabled: added?.streaming_recognition_enabled,
      profile_delivery_ids: added?.stt_profiles?.map(
        (profile) => profile.delivery_profile_id,
      ),
      delivery_profiles: added?.delivery_profiles,
      http_delivery_profiles: added?.http_delivery_profiles,
    }).toEqual({
      developer_http_enabled: true,
      profile_delivery_ids: [null, null],
      delivery_profiles: undefined,
      http_delivery_profiles: undefined,
    });
  });

  it("keeps existing source-aware routes unchanged when global Developer HTTP is enabled", () => {
    const first = {
      ...effectiveSttProfiles(legacyConfig())[0],
      delivery_profile_id: "studio-output",
    };
    const config = {
      ...legacyConfig(),
      streaming_recognition_enabled: true,
      developer_connection_mode: "http" as const,
      developer_http_url: "http://127.0.0.1:15522/api/events",
      stt_profiles: [first],
      delivery_profiles: [
        {
          id: "studio-output",
          gui_enabled: false,
          translation_mapping_ids: [],
          speech_mapping_ids: [],
          http_profile_ids: ["audit-output"],
          neo_text_enabled: false,
        },
      ],
      http_delivery_profiles: [
        {
          id: "audit-output",
          url: "https://example.test/audit",
          payload_format: "text_event_v1" as const,
          artifact_kinds: ["recognition" as const],
          send_timing: "final" as const,
        },
      ],
    };

    const added = addSttProfile(config, first.id, [
      {
        id: "interface",
        host: "WASAPI",
        display_name: "Interface",
        channels: 2,
        sample_rate: 48_000,
      },
    ]);

    expect(
      added?.stt_profiles?.map((profile) => profile.delivery_profile_id),
    ).toEqual(["studio-output", "studio-output"]);
    expect(added?.delivery_profiles).toEqual([
      {
        id: "studio-output",
        gui_enabled: false,
        translation_mapping_ids: [],
        speech_mapping_ids: [],
        http_profile_ids: ["audit-output"],
        neo_text_enabled: false,
      },
    ]);
    expect(added?.http_delivery_profiles).toEqual(
      config.http_delivery_profiles,
    );
  });

  it("does not mark the selected profile channel occupied when its editor input is cloned", () => {
    const withTwo = addSttProfile(legacyConfig(), "stt-profile-1", [
      {
        id: "interface",
        host: "WASAPI",
        display_name: "Interface",
        channels: 2,
        sample_rate: 48_000,
      },
    ])!;
    const [first] = effectiveSttProfiles(withTwo);

    expect([
      ...occupiedInputChannelsForDevice(withTwo.stt_profiles!, first.id, {
        ...first.input,
      }),
    ]).toEqual([1]);
  });

  it("keeps the selected profile channel and rejects a different fully occupied device", () => {
    const profiles = effectiveSttProfiles(explicitCaptureConfig());

    expect(
      availableInputChannelForProfile(profiles, "speaker-a", interfaceDevice),
    ).toBe(0);

    const movedSelected = profiles.map((profile) =>
      profile.id === "speaker-a"
        ? {
            ...profile,
            input: {
              ...profile.input,
              device_host: "WASAPI",
              device_id: "other-interface",
            },
          }
        : profile,
    );
    const fullTarget = [
      ...movedSelected,
      {
        ...profiles[0],
        id: "speaker-c",
        input: { ...profiles[0].input, channel_index: 0 },
      },
    ];
    expect(
      availableInputChannelForProfile(fullTarget, "speaker-a", interfaceDevice),
    ).toBeNull();
  });

  it("lays out numbered input channels in rows of sixteen and marks channels used by another profile", () => {
    const profiles = effectiveSttProfiles(explicitCaptureConfig());
    const rows = buildInputChannelRows(
      profiles,
      "speaker-a",
      { ...interfaceDevice, channels: 17 },
      16,
    );

    expect(
      rows.map((row) =>
        row.map(({ channelIndex, occupied }) => [channelIndex, occupied]),
      ),
    ).toEqual([
      [
        [0, false],
        [1, true],
        [2, false],
        [3, false],
        [4, false],
        [5, false],
        [6, false],
        [7, false],
        [8, false],
        [9, false],
        [10, false],
        [11, false],
        [12, false],
        [13, false],
        [14, false],
        [15, false],
      ],
      [[16, false]],
    ]);
  });

  it("deletes a selected profile and selects the profile shifted into its slot", () => {
    const withTwo = addSttProfile(legacyConfig(), "stt-profile-1", [
      {
        id: "interface",
        host: "WASAPI",
        display_name: "Interface",
        channels: 2,
        sample_rate: 48_000,
      },
    ])!;

    expect(deleteSttProfile(withTwo, "stt-profile-1")).toEqual(
      expect.objectContaining({ selectedProfileId: "stt-profile-2" }),
    );
    expect(deleteSttProfile(legacyConfig(), "stt-profile-1")).toBeNull();
  });
});
