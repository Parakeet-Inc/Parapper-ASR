import { Select, Stack, Switch } from "@mantine/core";
import { useTranslation } from "react-i18next";

import { buildNoiseCancellationModelOptions } from "../../lib/settings-options";
import type {
  NoiseCancellationModel,
  ParapperConfig,
} from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type NoiseCancellationSettingsProps = {
  config: ParapperConfig;
  runtimeLocked: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
};

export const NoiseCancellationSettings: React.FC<
  NoiseCancellationSettingsProps
> = ({ config, runtimeLocked, onUpdateConfig }) => {
  const { t } = useTranslation();
  const runtimeLockedTooltip = t("tooltip.runtimeLocked");
  const noiseCancellationModelOptions = buildNoiseCancellationModelOptions(t);

  return (
    <Stack gap="sm">
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Stack gap={4}>
          {settingLabel(
            t("noiseCancellationSettings.enable.label"),
            t("noiseCancellationSettings.enable.description"),
          )}
          <Switch
            aria-label={t("noiseCancellationSettings.enable.label")}
            checked={config.noise_cancellation_enabled}
            disabled={runtimeLocked}
            onChange={(event) =>
              onUpdateConfig(
                "noise_cancellation_enabled",
                event.currentTarget.checked,
              )
            }
          />
        </Stack>
      </DisabledReasonTooltip>
      <DisabledReasonTooltip
        disabled={runtimeLocked || !config.noise_cancellation_enabled}
        label={
          runtimeLocked
            ? runtimeLockedTooltip
            : t("noiseCancellationSettings.model.disabledTooltip")
        }
      >
        <Select
          label={settingLabel(
            t("noiseCancellationSettings.model.label"),
            t("noiseCancellationSettings.model.description"),
          )}
          data={noiseCancellationModelOptions}
          value={config.noise_cancellation_model}
          allowDeselect={false}
          disabled={runtimeLocked || !config.noise_cancellation_enabled}
          onChange={(value) => {
            if (value === "ul_unas") {
              onUpdateConfig(
                "noise_cancellation_model",
                value as NoiseCancellationModel,
              );
            }
          }}
        />
      </DisabledReasonTooltip>
    </Stack>
  );
};
