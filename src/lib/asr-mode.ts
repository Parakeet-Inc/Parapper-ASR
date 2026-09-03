import { asrModelOption } from "./constants";
import type { AsrLanguage, AsrMode, AsrModel, AsrPrecision } from "./types";

/** Models for which the product exposes the tuned high-accuracy decoder. */
export const accurateAsrModels: readonly AsrModel[] = [
  "reazonspeech_k2_v2",
  "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
];

export const isAccurateAsrModel = (model: AsrModel) =>
  accurateAsrModels.includes(model);

export const asrModeOptionsForModel = (model: AsrModel): AsrMode[] =>
  isAccurateAsrModel(model) ? ["fast", "accurate"] : ["fast"];

/** Keep a selected mode only when the new model supports it. */
export const effectiveAsrMode = (
  model: AsrModel,
  requestedMode: AsrMode,
): AsrMode =>
  asrModeOptionsForModel(model).includes(requestedMode)
    ? requestedMode
    : "fast";

type AsrModelConfigFields = {
  asr_language: AsrLanguage;
  asr_model: AsrModel;
  asr_precision: AsrPrecision;
  asr_mode: AsrMode;
};

type ConfigWithSelectedAsrModel<T> = Omit<T, keyof AsrModelConfigFields> &
  AsrModelConfigFields;

/** Apply all model-derived settings in one immutable config update. */
export const configWithAsrModel = <T extends AsrModelConfigFields>(
  config: T,
  model: AsrModel,
): ConfigWithSelectedAsrModel<T> => {
  const option = asrModelOption(model);
  return {
    ...config,
    asr_language: option.language,
    asr_model: model,
    asr_precision: option.defaultPrecision,
    asr_mode: effectiveAsrMode(model, config.asr_mode),
  };
};

export const canToggleAsrHotwords = (mode: AsrMode, runtimeLocked: boolean) =>
  mode === "accurate" && !runtimeLocked;
