import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import {
  asrModelOption,
  asrModelOptions,
  asrPrecisionOptions,
} from "../lib/constants";
import type { AsrModel } from "../lib/types";

export const useAsrModelOptions = (selectedModel: AsrModel | null) => {
  const { t } = useTranslation();

  const asrModelSelectOptions = useMemo(
    () =>
      asrModelOptions.map(({ labelKey, value }) => ({
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

  return { asrModelSelectOptions, selectedAsrPrecisionOptions };
};
