import { describe, expect, it } from "vitest";

import {
  nemotronFamilyForModel,
  nemotronLatenciesForFamily,
  nemotronLatencyForModel,
  nemotronModelFor,
  nemotronModelForFamily,
  type NemotronFamily,
  type NemotronLatencyMs,
} from "./nemotron";
import type { AsrModel } from "./types";
import { en } from "../i18n/locales/en";
import { ja } from "../i18n/locales/ja";

describe("Nemotron interim model selection", () => {
  it("round-trips every family and latency through the persisted model id", () => {
    const cases: Array<[NemotronFamily, NemotronLatencyMs, AsrModel]> = [
      ["english", "80", "nemotron_speech_streaming_en_0_6b_80ms_int8"],
      ["english", "160", "nemotron_speech_streaming_en_0_6b_160ms_int8"],
      ["english", "320", "nemotron_speech_streaming_en_0_6b_320ms_int8"],
      ["english", "560", "nemotron_speech_streaming_en_0_6b_560ms_int8"],
      ["english", "1120", "nemotron_speech_streaming_en_0_6b_1120ms_int8"],
      ["multilingual", "80", "nemotron_3_5_asr_streaming_0_6b_80ms_int8"],
      ["multilingual", "160", "nemotron_3_5_asr_streaming_0_6b_160ms_int8"],
      ["multilingual", "320", "nemotron_3_5_asr_streaming_0_6b_320ms_int8"],
      ["multilingual", "560", "nemotron_3_5_asr_streaming_0_6b_560ms_int8"],
      ["multilingual", "1120", "nemotron_3_5_asr_streaming_0_6b_1120ms_int8"],
    ];

    for (const [family, latency, model] of cases) {
      expect(nemotronModelFor(family, latency)).toBe(model);
      expect(nemotronFamilyForModel(model)).toBe(family);
      expect(nemotronLatencyForModel(model)).toBe(latency);
    }
  });

  it("keeps the selected latency when switching language families", () => {
    expect(
      nemotronModelForFamily(
        "multilingual",
        "nemotron_speech_streaming_en_0_6b_560ms_int8",
      ),
    ).toBe("nemotron_3_5_asr_streaming_0_6b_560ms_int8");
  });

  it("uses 320ms when Nemotron is selected for the first time", () => {
    expect(nemotronModelForFamily("english", null)).toBe(
      "nemotron_speech_streaming_en_0_6b_320ms_int8",
    );
  });

  it("shows every latency choice as only its duration in each locale", () => {
    for (const locale of [ja, en]) {
      expect([
        locale.settings.interimAsrModel.latency80,
        locale.settings.interimAsrModel.latency160,
        locale.settings.interimAsrModel.latency320,
        locale.settings.interimAsrModel.latency560,
        locale.settings.interimAsrModel.latency1120,
      ]).toEqual(["80 ms", "160 ms", "320 ms", "560 ms", "1120 ms"]);
    }
  });

  it("exposes only the latency choices published for each family", () => {
    expect(nemotronLatenciesForFamily("english")).toEqual([
      "80",
      "160",
      "320",
      "560",
      "1120",
    ]);
    expect(nemotronLatenciesForFamily("multilingual")).toEqual([
      "80",
      "160",
      "320",
      "560",
      "1120",
    ]);
  });
});
