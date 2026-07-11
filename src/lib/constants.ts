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
  implementation:
    | "reazonspeech_k2"
    | "nemo_parakeet_tdt_ctc"
    | "nemo_parakeet_tdt"
    | "nemotron";
  capability: "completion_and_interim" | "interim_only";
}[] = [
  {
    labelKey: "options.asrModel.reazonspeechK2V2",
    value: "reazonspeech_k2_v2",
    language: "japanese",
    supportedPrecisions: ["int8", "int8_float32", "float32"],
    defaultPrecision: "int8_float32",
    implementation: "reazonspeech_k2",
    capability: "completion_and_interim",
  },
  {
    labelKey: "options.asrModel.nemoParakeetTdtCtcJa35000Int8",
    value: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
    language: "japanese",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
    implementation: "nemo_parakeet_tdt_ctc",
    capability: "completion_and_interim",
  },
  {
    labelKey: "options.asrModel.nemoParakeetTdtV2Int8",
    value: "nemo_parakeet_tdt_0_6b_v2_int8",
    language: "english",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
    implementation: "nemo_parakeet_tdt",
    capability: "completion_and_interim",
  },
  {
    labelKey: "options.asrModel.nemoParakeetTdtV3Int8",
    value: "nemo_parakeet_tdt_0_6b_v3_int8",
    language: "european_multilingual",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
    implementation: "nemo_parakeet_tdt",
    capability: "completion_and_interim",
  },
  {
    labelKey: "options.asrModel.nemotronSpeechStreamingEn160MsInt8",
    value: "nemotron_speech_streaming_en_0_6b_160ms_int8",
    language: "english",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
    implementation: "nemotron",
    capability: "interim_only",
  },
  {
    labelKey: "options.asrModel.nemotronSpeechStreamingEn560MsInt8",
    value: "nemotron_speech_streaming_en_0_6b_560ms_int8",
    language: "english",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
    implementation: "nemotron",
    capability: "interim_only",
  },
  {
    labelKey: "options.asrModel.nemotron35AsrStreaming160MsInt8",
    value: "nemotron_3_5_asr_streaming_0_6b_160ms_int8",
    language: "multilingual",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
    implementation: "nemotron",
    capability: "interim_only",
  },
  {
    labelKey: "options.asrModel.nemotron35AsrStreaming560MsInt8",
    value: "nemotron_3_5_asr_streaming_0_6b_560ms_int8",
    language: "multilingual",
    supportedPrecisions: ["int8"],
    defaultPrecision: "int8",
    implementation: "nemotron",
    capability: "interim_only",
  },
];

export const completionAsrModelOptions = asrModelOptions.filter(
  (option) => option.capability === "completion_and_interim",
);

export const interimOnlyAsrModelOptions = asrModelOptions.filter(
  (option) => option.capability === "interim_only",
);

export const asrModelOption = (model: AsrModel) =>
  asrModelOptions.find((option) => option.value === model) ??
  asrModelOptions[0];

export const DEFAULT_RECOGNITION_LOG_LIMIT = 500;
export const DEFAULT_DEBUG_AUDIO_LOG_LIMIT = 20;
