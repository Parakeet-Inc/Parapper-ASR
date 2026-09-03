export type SettingsNavigationGroup = "stt" | "output" | "other";

export type SettingsNavigationItem = {
  group: SettingsNavigationGroup;
  tab: SettingsNavigationTab;
  labelKey: string;
};

export type SettingsNavigationTab =
  | "connection"
  | "noise-cancellation"
  | "vad"
  | "asr"
  | "external-apps"
  | "translation"
  | "speech"
  | "other"
  | "downloads"
  | "licenses";

export const settingsNavigation: readonly SettingsNavigationItem[] = [
  { group: "stt", tab: "connection", labelKey: "tabs.connection" },
  {
    group: "stt",
    tab: "noise-cancellation",
    labelKey: "tabs.noiseCancellation",
  },
  { group: "stt", tab: "vad", labelKey: "tabs.vad" },
  { group: "stt", tab: "asr", labelKey: "tabs.asr" },
  { group: "output", tab: "external-apps", labelKey: "tabs.externalApps" },
  { group: "output", tab: "translation", labelKey: "tabs.translation" },
  { group: "output", tab: "speech", labelKey: "tabs.speech" },
  { group: "other", tab: "other", labelKey: "tabs.other" },
  { group: "other", tab: "downloads", labelKey: "tabs.downloads" },
  { group: "other", tab: "licenses", labelKey: "tabs.licenses" },
];

export const settingsNavigationForGroup = (group: SettingsNavigationGroup) =>
  settingsNavigation.filter((item) => item.group === group);
