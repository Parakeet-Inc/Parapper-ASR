import type { AsrModel } from "./types";

export type NemotronFamily = "english" | "multilingual";
export type NemotronLatencyMs = "80" | "160" | "320" | "560" | "1120";

const NEMOTRON_LATENCIES: readonly NemotronLatencyMs[] = [
  "80",
  "160",
  "320",
  "560",
  "1120",
];

const NEMOTRON_MODELS: Record<
  NemotronFamily,
  Record<NemotronLatencyMs, AsrModel>
> = {
  english: {
    "80": "nemotron_speech_streaming_en_0_6b_80ms_int8",
    "160": "nemotron_speech_streaming_en_0_6b_160ms_int8",
    "320": "nemotron_speech_streaming_en_0_6b_320ms_int8",
    "560": "nemotron_speech_streaming_en_0_6b_560ms_int8",
    "1120": "nemotron_speech_streaming_en_0_6b_1120ms_int8",
  },
  multilingual: {
    "80": "nemotron_3_5_asr_streaming_0_6b_80ms_int8",
    "160": "nemotron_3_5_asr_streaming_0_6b_160ms_int8",
    "320": "nemotron_3_5_asr_streaming_0_6b_320ms_int8",
    "560": "nemotron_3_5_asr_streaming_0_6b_560ms_int8",
    "1120": "nemotron_3_5_asr_streaming_0_6b_1120ms_int8",
  },
};

export const nemotronFamilyForModel = (
  model: AsrModel | null,
): NemotronFamily | null => {
  if (model?.startsWith("nemotron_speech_streaming_en_")) return "english";
  if (model?.startsWith("nemotron_3_5_asr_streaming_")) return "multilingual";
  return null;
};

export const nemotronLatencyForModel = (
  model: AsrModel | null,
): NemotronLatencyMs | null => {
  if (!nemotronFamilyForModel(model)) return null;
  return (
    NEMOTRON_LATENCIES.find((latency) => model?.includes(`_${latency}ms_`)) ??
    null
  );
};

export const nemotronLatenciesForFamily = (
  _family: NemotronFamily,
): readonly NemotronLatencyMs[] => NEMOTRON_LATENCIES;

export const nemotronModelFor = (
  family: NemotronFamily,
  latency: NemotronLatencyMs,
): AsrModel => NEMOTRON_MODELS[family][latency];

export const nemotronModelForFamily = (
  family: NemotronFamily,
  currentModel: AsrModel | null,
): AsrModel =>
  nemotronModelFor(family, nemotronLatencyForModel(currentModel) ?? "320");
