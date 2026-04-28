import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import { en } from "./locales/en";
import { ja } from "./locales/ja";

const STORAGE_KEY = "parapper.language";

export const resources = {
  ja: { translation: ja },
  en: { translation: en },
} as const;

export type SupportedLanguage = keyof typeof resources;

export const availableLanguages: {
  code: SupportedLanguage;
  labelKey: string;
}[] = [
  { code: "ja", labelKey: "language.ja" },
  { code: "en", labelKey: "language.en" },
];

export const normalizeLanguage = (language: string | null | undefined) => {
  const normalized = language?.toLowerCase().split("-")[0];
  return normalized && normalized in resources
    ? (normalized as SupportedLanguage)
    : "ja";
};

const initialLanguage = normalizeLanguage(
  localStorage.getItem(STORAGE_KEY) ?? navigator.language,
);

void i18n.use(initReactI18next).init({
  resources,
  lng: initialLanguage,
  fallbackLng: "ja",
  interpolation: {
    escapeValue: false,
  },
  returnNull: false,
});

i18n.on("languageChanged", (language) => {
  localStorage.setItem(STORAGE_KEY, normalizeLanguage(language));
});

export default i18n;
