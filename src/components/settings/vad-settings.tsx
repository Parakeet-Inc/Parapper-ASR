import { NumberInput, Stack } from "@mantine/core";
import { useTranslation } from "react-i18next";

import type { ParapperConfig } from "../../lib/types";
import {
  settingLabel,
  thresholdChunksToMs,
  thresholdMsToChunks,
} from "../ui/display";

type VadSettingsProps = {
  config: ParapperConfig;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
};

export const VadSettings: React.FC<VadSettingsProps> = ({
  config,
  onUpdateConfig,
}) => {
  const { t } = useTranslation();

  return (
    <Stack gap="sm">
      <NumberInput
        label={settingLabel(
          t("settings.pauseThreshold.label"),
          t("settings.pauseThreshold.description"),
        )}
        value={thresholdChunksToMs(
          config.pause_threshold,
          config.vad_interval_ms,
        )}
        min={32}
        max={30000}
        step={32}
        onChange={(value) =>
          onUpdateConfig(
            "pause_threshold",
            thresholdMsToChunks(
              typeof value === "number" ? value : 300,
              config.vad_interval_ms,
            ),
          )
        }
      />
      <NumberInput
        label={settingLabel(
          t("settings.phraseThreshold.label"),
          t("settings.phraseThreshold.description"),
        )}
        value={thresholdChunksToMs(
          config.phrase_threshold,
          config.vad_interval_ms,
        )}
        min={32}
        max={30000}
        step={32}
        onChange={(value) =>
          onUpdateConfig(
            "phrase_threshold",
            thresholdMsToChunks(
              typeof value === "number" ? value : 320,
              config.vad_interval_ms,
            ),
          )
        }
      />
      <NumberInput
        label={settingLabel(
          t("settings.vadThreshold.label"),
          t("settings.vadThreshold.description"),
        )}
        value={config.vad_threshold}
        min={0}
        max={1}
        step={0.1}
        onChange={(value) =>
          onUpdateConfig(
            "vad_threshold",
            typeof value === "number" ? value : 0.5,
          )
        }
      />
    </Stack>
  );
};
