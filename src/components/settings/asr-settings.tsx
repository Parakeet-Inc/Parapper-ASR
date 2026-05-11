import { Checkbox, Select, Stack } from "@mantine/core";
import { useTranslation } from "react-i18next";

import { useAsrModelOptions } from "../../hooks/use-asr-model-options";
import { buildAsrThreadOptions } from "../../lib/settings-options";
import type { AsrModel, AsrPrecision, ParapperConfig } from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type AsrSettingsProps = {
  config: ParapperConfig;
  runtimeLocked: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onApplyAsrModel: (model: AsrModel) => void;
};

export const AsrSettings: React.FC<AsrSettingsProps> = ({
  config,
  runtimeLocked,
  onUpdateConfig,
  onApplyAsrModel,
}) => {
  const { t } = useTranslation();
  const { asrModelSelectOptions, selectedAsrPrecisionOptions } =
    useAsrModelOptions(config.asr_model);
  const asrThreadOptions = buildAsrThreadOptions(t);
  const runtimeLockedTooltip = t("tooltip.runtimeLocked");
  const selectedEnabledAsrModels = config.enabled_asr_models?.length
    ? config.enabled_asr_models
    : [config.asr_model];
  const updateEnabledAsrModels = (models: string[]) => {
    const selectedModels = models as AsrModel[];
    onUpdateConfig(
      "enabled_asr_models",
      selectedModels.length ? selectedModels : [config.asr_model],
    );
  };

  return (
    <Stack gap="sm">
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Select
          label={settingLabel(
            t("settings.asrModel.label"),
            t("settings.asrModel.description"),
          )}
          data={asrModelSelectOptions}
          value={config.asr_model}
          allowDeselect={false}
          disabled={runtimeLocked}
          onChange={(value) => {
            if (value) {
              onApplyAsrModel(value as AsrModel);
            }
          }}
        />
      </DisabledReasonTooltip>
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Select
          label={settingLabel(
            t("settings.asrPrecision.label"),
            t("settings.asrPrecision.description"),
          )}
          data={selectedAsrPrecisionOptions}
          value={config.asr_precision}
          allowDeselect={false}
          disabled={runtimeLocked}
          onChange={(value) => {
            if (value) {
              onUpdateConfig("asr_precision", value as AsrPrecision);
            }
          }}
        />
      </DisabledReasonTooltip>
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Select
          label={settingLabel(
            t("settings.asrThreads.label"),
            t("settings.asrThreads.description"),
          )}
          data={asrThreadOptions}
          value={String(config.asr_num_threads)}
          allowDeselect={false}
          disabled={runtimeLocked}
          onChange={(value) =>
            onUpdateConfig("asr_num_threads", Number(value ?? 4))
          }
        />
      </DisabledReasonTooltip>
      <Checkbox
        label={settingLabel(
          t("settings.asrNormalizeInput.label"),
          t("settings.asrNormalizeInput.description"),
        )}
        checked={config.asr_normalize_input_audio}
        onChange={(event) =>
          onUpdateConfig(
            "asr_normalize_input_audio",
            event.currentTarget.checked,
          )
        }
      />
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Checkbox
          label={settingLabel(
            t("settings.multilingualAsr.label"),
            t("settings.multilingualAsr.description"),
          )}
          checked={config.multilingual_asr_enabled}
          disabled={runtimeLocked}
          onChange={(event) =>
            onUpdateConfig(
              "multilingual_asr_enabled",
              event.currentTarget.checked,
            )
          }
        />
      </DisabledReasonTooltip>
      {config.multilingual_asr_enabled ? (
        <>
          <DisabledReasonTooltip
            disabled={runtimeLocked}
            label={runtimeLockedTooltip}
          >
            <Checkbox.Group
              label={settingLabel(
                t("settings.enabledAsrModels.label"),
                t("settings.enabledAsrModels.description"),
              )}
              value={selectedEnabledAsrModels}
              onChange={updateEnabledAsrModels}
            >
              <Stack gap={4} mt={4}>
                {asrModelSelectOptions.map((option) => (
                  <Checkbox
                    key={option.value}
                    value={option.value}
                    label={option.label}
                    disabled={runtimeLocked}
                  />
                ))}
              </Stack>
            </Checkbox.Group>
          </DisabledReasonTooltip>
        </>
      ) : null}
    </Stack>
  );
};
