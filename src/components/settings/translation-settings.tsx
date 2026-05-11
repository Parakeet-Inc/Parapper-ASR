import { Select, SegmentedControl, Stack, Switch } from "@mantine/core";
import { useTranslation } from "react-i18next";

import { MappingList } from "./mapping-list";
import { useAsrModelOptions } from "../../hooks/use-asr-model-options";
import {
  languageOptions,
  makeId,
  modelOptionsWithAny,
} from "../../lib/mapping-options";
import { isMacOs } from "../../lib/platform";
import type {
  AsrModel,
  NeoSendTiming,
  ParapperConfig,
  TranslationMapping,
} from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

type TranslationSettingsProps = {
  config: ParapperConfig;
  runtimeLocked: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
};

const sendTimingOptions = (t: (key: string) => string) => [
  { value: "interim", label: t("options.neoSendTiming.interim") },
  { value: "final", label: t("options.neoSendTiming.final") },
];

export const TranslationSettings: React.FC<TranslationSettingsProps> = ({
  config,
  runtimeLocked,
  onUpdateConfig,
}) => {
  const { t } = useTranslation();
  const { asrModelSelectOptions } = useAsrModelOptions(config.asr_model);
  const yncPluginAvailable = !isMacOs();
  const disabledReason = !yncPluginAvailable
    ? t("tooltip.nativeConnectionsDisabled")
    : t("tooltip.runtimeLocked");
  const asrModelOptions = modelOptionsWithAny(
    t("translationSettings.mapping.anyModel"),
    asrModelSelectOptions,
  );

  return (
    <Stack gap="sm">
      <DisabledReasonTooltip
        disabled={runtimeLocked || !yncPluginAvailable}
        label={disabledReason}
      >
        <Switch
          label={settingLabel(
            t("translationSettings.enable.label"),
            t("translationSettings.enable.description"),
          )}
          checked={yncPluginAvailable && config.translation_enabled}
          disabled={runtimeLocked || !yncPluginAvailable}
          onChange={(event) =>
            onUpdateConfig("translation_enabled", event.currentTarget.checked)
          }
        />
      </DisabledReasonTooltip>
      <SegmentedControl
        aria-label={t("translationSettings.sendTiming.label")}
        value={config.translation_send_timing}
        disabled={runtimeLocked || !yncPluginAvailable}
        data={sendTimingOptions(t)}
        onChange={(value) =>
          onUpdateConfig("translation_send_timing", value as NeoSendTiming)
        }
      />
      <TranslationMappingRows
        mappings={config.translation_mappings}
        asrModelOptions={asrModelOptions}
        disabled={runtimeLocked || !yncPluginAvailable}
        onChange={(translationMappings) =>
          onUpdateConfig("translation_mappings", translationMappings)
        }
      />
    </Stack>
  );
};

const TranslationMappingRows: React.FC<{
  mappings: TranslationMapping[];
  asrModelOptions: { value: string; label: string }[];
  disabled: boolean;
  onChange: (mappings: TranslationMapping[]) => void;
}> = ({ mappings, asrModelOptions, disabled, onChange }) => {
  const { t } = useTranslation();

  return (
    <MappingList
      rows={mappings}
      addLabel={t("translationSettings.mapping.addRow")}
      moveUpLabel={t("translationSettings.mapping.moveUp")}
      moveDownLabel={t("translationSettings.mapping.moveDown")}
      deleteLabel={t("translationSettings.mapping.deleteRow")}
      mutationDisabled={disabled}
      createRow={() => ({
        id: makeId("translation"),
        source_asr_model: null,
        target_lang: "en_US",
      })}
      onChange={onChange}
      renderHeader={(mapping, updateMapping) => (
        <Select
          aria-label={t("translationSettings.mapping.sourceModel")}
          data={asrModelOptions}
          value={mapping.source_asr_model ?? "any"}
          allowDeselect={false}
          disabled={disabled}
          style={{ flex: 1 }}
          onChange={(value) =>
            updateMapping({
              source_asr_model:
                value && value !== "any" ? (value as AsrModel) : null,
            })
          }
        />
      )}
      renderBody={(mapping, updateMapping) => (
        <Select
          label={t("translationSettings.mapping.targetLang")}
          data={languageOptions}
          value={mapping.target_lang}
          searchable
          allowDeselect={false}
          disabled={disabled}
          onChange={(value) => updateMapping({ target_lang: value ?? "en_US" })}
        />
      )}
    />
  );
};
