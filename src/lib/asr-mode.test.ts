import { describe, expect, it } from "vitest";

import {
  asrModeOptionsForModel,
  canToggleAsrHotwords,
  configWithAsrModel,
  effectiveAsrMode,
  isAccurateAsrModel,
} from "./asr-mode";

describe("ASR mode availability", () => {
  it("offers fast and accurate modes for the two tuned Japanese models", () => {
    expect(asrModeOptionsForModel("reazonspeech_k2_v2")).toEqual([
      "fast",
      "accurate",
    ]);
    expect(
      asrModeOptionsForModel("nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8"),
    ).toEqual(["fast", "accurate"]);
  });

  it("offers only fast mode for models without an accuracy decoder", () => {
    expect(asrModeOptionsForModel("nemo_parakeet_tdt_0_6b_v2_int8")).toEqual([
      "fast",
    ]);
    expect(
      isAccurateAsrModel("nemotron_3_5_asr_streaming_0_6b_160ms_int8"),
    ).toBe(false);
  });

  it("falls back to fast when a model cannot keep the requested mode", () => {
    expect(effectiveAsrMode("nemo_parakeet_tdt_0_6b_v3_int8", "accurate")).toBe(
      "fast",
    );
    expect(effectiveAsrMode("reazonspeech_k2_v2", "accurate")).toBe("accurate");
  });

  it("changes all model-derived settings together and preserves unrelated config", () => {
    const current = {
      asr_language: "japanese" as const,
      asr_model: "reazonspeech_k2_v2" as const,
      asr_precision: "float32" as const,
      asr_mode: "accurate" as const,
      asr_hotwords_enabled: true,
    };

    expect(
      configWithAsrModel(current, "nemo_parakeet_tdt_0_6b_v2_int8"),
    ).toEqual({
      asr_language: "english",
      asr_model: "nemo_parakeet_tdt_0_6b_v2_int8",
      asr_precision: "int8",
      asr_mode: "fast",
      asr_hotwords_enabled: true,
    });
    expect(
      configWithAsrModel(current, "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8"),
    ).toMatchObject({
      asr_language: "japanese",
      asr_precision: "int8",
      asr_mode: "accurate",
      asr_hotwords_enabled: true,
    });
  });

  it("keeps accuracy mode when switching from Parakeet Japanese to ReazonSpeech", () => {
    const current = {
      asr_language: "japanese" as const,
      asr_model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8" as const,
      asr_precision: "int8" as const,
      asr_mode: "accurate" as const,
    };

    expect(configWithAsrModel(current, "reazonspeech_k2_v2")).toMatchObject({
      asr_model: "reazonspeech_k2_v2",
      asr_language: "japanese",
      asr_precision: "int8_float32",
      asr_mode: "accurate",
    });
  });

  it("allows the hotword toggle only in unlocked accuracy mode", () => {
    expect(canToggleAsrHotwords("accurate", false)).toBe(true);
    expect(canToggleAsrHotwords("accurate", true)).toBe(false);
    expect(canToggleAsrHotwords("fast", false)).toBe(false);
    expect(canToggleAsrHotwords("fast", true)).toBe(false);
  });
});
