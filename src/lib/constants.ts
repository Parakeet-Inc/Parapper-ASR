import type { AsrLanguage, AsrModel, AsrPrecision } from "./types";

export const asrPrecisionOptions: { label: string; value: AsrPrecision }[] = [
  { label: "int8", value: "int8" },
  { label: "int8-fp32", value: "int8_float32" },
  { label: "float32", value: "float32" },
];

export const asrModelOptions: {
  labelKey: string;
  value: AsrModel;
  language: AsrLanguage;
  supportedPrecisions: AsrPrecision[];
  defaultPrecision: AsrPrecision;
}[] = [
  {
    labelKey: "options.asrModel.reazonspeechK2V2",
    value: "reazonspeech_k2_v2",
    language: "japanese",
    supportedPrecisions: ["int8", "int8_float32", "float32"],
    defaultPrecision: "int8_float32",
  },
  {
    labelKey: "options.asrModel.nemoParakeetTdtCtcJa35000Int8",
    value: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
    language: "japanese",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
  },
  {
    labelKey: "options.asrModel.nemoParakeetTdtV2Int8",
    value: "nemo_parakeet_tdt_0_6b_v2_int8",
    language: "english",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
  },
  {
    labelKey: "options.asrModel.nemoParakeetTdtV3Int8",
    value: "nemo_parakeet_tdt_0_6b_v3_int8",
    language: "european_multilingual",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
  },
];

export const asrModelOption = (model: AsrModel) =>
  asrModelOptions.find((option) => option.value === model) ??
  asrModelOptions[0];

export const DEFAULT_RECOGNITION_LOG_LIMIT = 500;
export const DEFAULT_DEBUG_AUDIO_LOG_LIMIT = 20;
