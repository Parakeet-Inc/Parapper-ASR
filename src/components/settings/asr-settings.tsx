import { Button, Checkbox, Group, Select, Stack, Text } from "@mantine/core";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { AsrHotwordModal } from "./asr-hotword-modal";
import { useAsrModelOptions } from "../../hooks/use-asr-model-options";
import {
  asrModeOptionsForModel,
  canToggleAsrHotwords,
  effectiveAsrMode,
} from "../../lib/asr-mode";
import {
  nemotronFamilyForModel,
  nemotronLatenciesForFamily,
  nemotronLatencyForModel,
  nemotronModelFor,
  nemotronModelForFamily,
  type NemotronFamily,
  type NemotronLatencyMs,
} from "../../lib/nemotron";
import { buildAsrThreadOptions } from "../../lib/settings-options";
import type {
  AsrMode,
  AsrModel,
  AsrPrecision,
  ParapperConfig,
} from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type AsrSettingsProps = {
  config: ParapperConfig;
  runtimeLocked: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
  onApplyAsrModel: (model: AsrModel) => void;
  onSuggestHotwordReadings: (surface: string) => Promise<string[]>;
};

export const AsrSettings: React.FC<AsrSettingsProps> = ({
  config,
  runtimeLocked,
  onUpdateConfig,
  onApplyAsrModel,
  onSuggestHotwordReadings,
}) => {
  const { t } = useTranslation();
  const [hotwordsOpened, setHotwordsOpened] = useState(false);
  const { asrModelSelectOptions, selectedAsrPrecisionOptions } =
    useAsrModelOptions(config.asr_model);
  const asrThreadOptions = buildAsrThreadOptions(t);
  const asrModeOptions = asrModeOptionsForModel(config.asr_model).map(
    (mode) => ({
      value: mode,
      label: t(`settings.asrMode.options.${mode}`),
    }),
  );
  const selectedAsrMode = effectiveAsrMode(config.asr_model, config.asr_mode);
  const canToggleHotwords = canToggleAsrHotwords(
    selectedAsrMode,
    runtimeLocked,
  );
  const runtimeLockedTooltip = t("tooltip.runtimeLocked");
  const primaryAsrModelValue = "__primary_asr_model__";
  const splitAsrModelOptions = [
    {
      value: primaryAsrModelValue,
      label: t("settings.interimAsrModel.primary"),
    },
    {
      value: "english",
      label: t("settings.interimAsrModel.nemotronEnglish"),
    },
    {
      value: "multilingual",
      label: t("settings.interimAsrModel.nemotronMultilingual"),
    },
  ];
  const selectedNemotronFamily = nemotronFamilyForModel(
    config.interim_asr_model,
  );
  const selectedNemotronLatency = nemotronLatencyForModel(
    config.interim_asr_model,
  );
  const nemotronLatencyOptions = selectedNemotronFamily
    ? nemotronLatenciesForFamily(selectedNemotronFamily).map((latency) => ({
        value: latency,
        label: t(`settings.interimAsrModel.latency${latency}`),
      }))
    : [];
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
            t("settings.interimAsrModel.label"),
            t("settings.interimAsrModel.description"),
          )}
          data={splitAsrModelOptions}
          value={selectedNemotronFamily ?? primaryAsrModelValue}
          allowDeselect={false}
          disabled={runtimeLocked}
          onChange={(value) =>
            onUpdateConfig(
              "interim_asr_model",
              value && value !== primaryAsrModelValue
                ? nemotronModelForFamily(
                    value as NemotronFamily,
                    config.interim_asr_model,
                  )
                : null,
            )
          }
        />
      </DisabledReasonTooltip>
      {selectedNemotronFamily && selectedNemotronLatency ? (
        <DisabledReasonTooltip
          disabled={runtimeLocked}
          label={runtimeLockedTooltip}
        >
          <Select
            label={settingLabel(
              t("settings.interimAsrModel.latencyLabel"),
              t("settings.interimAsrModel.latencyDescription"),
            )}
            data={nemotronLatencyOptions}
            value={selectedNemotronLatency}
            allowDeselect={false}
            disabled={runtimeLocked}
            onChange={(value) => {
              if (value) {
                onUpdateConfig(
                  "interim_asr_model",
                  nemotronModelFor(
                    selectedNemotronFamily,
                    value as NemotronLatencyMs,
                  ),
                );
              }
            }}
          />
        </DisabledReasonTooltip>
      ) : null}
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Select
          label={settingLabel(
            t("settings.asrMode.label"),
            t("settings.asrMode.description"),
          )}
          data={asrModeOptions}
          value={selectedAsrMode}
          allowDeselect={false}
          disabled={runtimeLocked}
          onChange={(value) => {
            if (value) {
              onUpdateConfig("asr_mode", value as AsrMode);
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
      <DisabledReasonTooltip
        disabled={!canToggleHotwords}
        label={
          runtimeLocked
            ? runtimeLockedTooltip
            : t("settings.hotwords.accurateOnly")
        }
      >
        <Checkbox
          label={settingLabel(
            t("settings.hotwords.enable"),
            t("settings.hotwords.enableDescription"),
          )}
          checked={config.asr_hotwords_enabled}
          disabled={!canToggleHotwords}
          onChange={(event) =>
            onUpdateConfig("asr_hotwords_enabled", event.currentTarget.checked)
          }
        />
      </DisabledReasonTooltip>
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={runtimeLockedTooltip}
      >
        <Group justify="space-between" align="center" wrap="nowrap">
          <Stack gap={2}>
            <Text size="sm" fw={500}>
              {t("settings.hotwords.label")}
            </Text>
            <Text size="xs" c="dimmed">
              {t("settings.hotwords.count", {
                count: config.asr_hotwords.length,
              })}
            </Text>
          </Stack>
          <Button
            variant="light"
            disabled={runtimeLocked}
            onClick={() => setHotwordsOpened(true)}
          >
            {t("settings.hotwords.manage")}
          </Button>
        </Group>
      </DisabledReasonTooltip>
      <AsrHotwordModal
        opened={hotwordsOpened}
        hotwords={config.asr_hotwords}
        disabled={runtimeLocked}
        onSuggestReadings={onSuggestHotwordReadings}
        onClose={() => setHotwordsOpened(false)}
        onSave={(hotwords) => {
          if (runtimeLocked) return;
          onUpdateConfig("asr_hotwords", hotwords);
          setHotwordsOpened(false);
        }}
      />
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
