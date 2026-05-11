import { NumberInput, Select, Stack, Switch } from "@mantine/core";
import { useTranslation } from "react-i18next";

import { buildTurnDetectorOptions } from "../../lib/settings-options";
import type { ParapperConfig } from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type VadSettingsProps = {
  config: ParapperConfig;
  runtimeLocked: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
};

export const VadSettings: React.FC<VadSettingsProps> = ({
  config,
  runtimeLocked,
  onUpdateConfig,
}) => {
  const { t } = useTranslation();
  const runtimeLockedTooltip = t("tooltip.runtimeLocked");
  const turnDetectorOptions = buildTurnDetectorOptions(t);
  const namoEnabled = config.turn_detector === "namo";

  return (
    <Stack gap="sm">
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Select
          label={settingLabel(
            t("settings.turnDetector.label"),
            t("settings.turnDetector.description"),
          )}
          data={turnDetectorOptions}
          value={config.turn_detector}
          allowDeselect={false}
          disabled={runtimeLocked}
          onChange={(value) => {
            if (value === "simple" || value === "namo") {
              onUpdateConfig("turn_detector", value);
            }
          }}
        />
      </DisabledReasonTooltip>
      <Switch
        label={settingLabel(
          t("settings.interimResult.label"),
          t("settings.interimResult.description"),
        )}
        checked={config.interim_result_enabled}
        onChange={(event) =>
          onUpdateConfig(
            "interim_result_enabled",
            event.currentTarget.checked,
          )
        }
      />
      {config.interim_result_enabled ? (
        <NumberInput
          label={settingLabel(
            t("settings.interimResultSilence.label"),
            t("settings.interimResultSilence.description"),
          )}
          value={config.interim_result_silence_ms}
          min={32}
          max={30000}
          step={32}
          onChange={(value) =>
            onUpdateConfig(
              "interim_result_silence_ms",
              typeof value === "number" ? value : 320,
            )
          }
        />
      ) : null}
      <NumberInput
        label={settingLabel(
          t("settings.turnCheckSilence.label"),
          t("settings.turnCheckSilence.description"),
        )}
        value={config.turn_check_silence_ms}
        min={32}
        max={30000}
        step={32}
        onChange={(value) =>
          onUpdateConfig(
            "turn_check_silence_ms",
            typeof value === "number" ? value : 960,
          )
        }
      />
      <NumberInput
        label={settingLabel(
          t("settings.segmentStartSpeech.label"),
          t("settings.segmentStartSpeech.description"),
        )}
        value={config.segment_start_speech_ms}
        min={32}
        max={30000}
        step={32}
        onChange={(value) =>
          onUpdateConfig(
            "segment_start_speech_ms",
            typeof value === "number" ? value : 320,
          )
        }
      />
      {namoEnabled ? (
        <>
          <NumberInput
            label={settingLabel(
              t("settings.namoTurnConfidence.label"),
              t("settings.namoTurnConfidence.description"),
            )}
            value={config.namo_turn_confidence_threshold}
            min={0}
            max={1}
            step={0.05}
            onChange={(value) =>
              onUpdateConfig(
                "namo_turn_confidence_threshold",
                typeof value === "number" ? value : 0.8,
              )
            }
          />
          <NumberInput
            label={settingLabel(
              t("settings.namoContextMaxTokens.label"),
              t("settings.namoContextMaxTokens.description"),
            )}
            value={config.namo_context_max_tokens}
            min={0}
            max={512}
            step={16}
            onChange={(value) =>
              onUpdateConfig(
                "namo_context_max_tokens",
                typeof value === "number" ? value : 256,
              )
            }
          />
        </>
      ) : null}
      <Switch
        label={settingLabel(
          t("settings.turnRerecognizeFull.label"),
          t("settings.turnRerecognizeFull.description"),
        )}
        checked={config.turn_rerecognize_full_on_complete}
        onChange={(event) =>
          onUpdateConfig(
            "turn_rerecognize_full_on_complete",
            event.currentTarget.checked,
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
