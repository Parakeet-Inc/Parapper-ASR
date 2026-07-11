import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import {
  asrModelOption,
  asrPrecisionOptions,
  completionAsrModelOptions,
  interimOnlyAsrModelOptions,
} from "../lib/constants";
import type { AsrModel } from "../lib/types";

export const useAsrModelOptions = (selectedModel: AsrModel | null) => {
  const { t } = useTranslation();

  const asrModelSelectOptions = useMemo(
    () =>
      completionAsrModelOptions.map(({ labelKey, value }) => ({
        label: t(labelKey),
        value,
      })),
    [t],
  );
  const interimOnlyAsrModelSelectOptions = useMemo(
    () =>
      interimOnlyAsrModelOptions.map(({ labelKey, value }) => ({
        label: t(labelKey),
        value,
      })),
    [t],
  );
  const selectedAsrPrecisionOptions = useMemo(() => {
    if (!selectedModel) {
      return [];
    }
    const selectedAsrModelOption = asrModelOption(selectedModel);
    return asrPrecisionOptions.filter((option) =>
      selectedAsrModelOption.supportedPrecisions.includes(option.value),
    );
  }, [selectedModel]);

  return {
    asrModelSelectOptions,
    interimOnlyAsrModelSelectOptions,
    selectedAsrPrecisionOptions,
  };
};
