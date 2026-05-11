import type { SelectOption } from "./settings-options";
import type { AsrModel } from "./types";

export const languageOptions = [
  { value: "ja_JP", label: "ja_JP" },
  { value: "en_US", label: "en_US" },
  { value: "ko_KR", label: "ko_KR" },
  { value: "bg_BG", label: "bg_BG" },
  { value: "cs_CZ", label: "cs_CZ" },
  { value: "da_DK", label: "da_DK" },
  { value: "el_GR", label: "el_GR" },
  { value: "es_ES", label: "es_ES" },
  { value: "et_EE", label: "et_EE" },
  { value: "fi_FI", label: "fi_FI" },
  { value: "hu_HU", label: "hu_HU" },
  { value: "it_IT", label: "it_IT" },
  { value: "nl_NL", label: "nl_NL" },
  { value: "pl_PL", label: "pl_PL" },
  { value: "pt_PT", label: "pt_PT" },
  { value: "ro_RO", label: "ro_RO" },
  { value: "ar_SA", label: "ar_SA" },
  { value: "de_DE", label: "de_DE" },
  { value: "fr_FR", label: "fr_FR" },
  { value: "hi_IN", label: "hi_IN" },
  { value: "id_ID", label: "id_ID" },
  { value: "ru_RU", label: "ru_RU" },
  { value: "vi_VN", label: "vi_VN" },
];

export const makeId = (prefix: string) =>
  `${prefix}-${globalThis.crypto?.randomUUID?.() ?? Date.now().toString(36)}`;

export const modelOptionsWithAny = (
  label: string,
  asrModelSelectOptions: SelectOption<AsrModel>[],
) => [{ value: "any", label }, ...asrModelSelectOptions];
