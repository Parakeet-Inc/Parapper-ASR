export type SelectOption<T extends string = string> = {
  label: string;
  value: T;
};

export const buildRetentionModeOptions = (
  t: (key: string) => string,
): SelectOption[] => [
  { label: t("options.retention.limited"), value: "limited" },
  { label: t("options.retention.unlimited"), value: "unlimited" },
];

export const buildAsrThreadOptions = (
  t: (key: string) => string,
): SelectOption[] => [
  { label: "1", value: "1" },
  { label: "4", value: "4" },
  { label: t("settings.asrThreads.max"), value: "0" },
];
