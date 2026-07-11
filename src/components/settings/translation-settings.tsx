import {
  Accordion,
  Button,
  Group,
  List,
  NumberInput,
  Paper,
  Select,
  SegmentedControl,
  Stack,
  Switch,
  Text,
} from "@mantine/core";
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { MappingList } from "./mapping-list";
import { useAsrModelOptions } from "../../hooks/use-asr-model-options";
import {
  normalizeParapperErrorPayload,
  notifyParapperIssue,
} from "../../lib/error";
import {
  localTranslationModelOptions,
  makeId,
  modelOptionsWithAny,
  translationLanguageOptions,
} from "../../lib/mapping-options";
import { isMacOs } from "../../lib/platform";
import type {
  AsrModel,
  LocalTranslationModel,
  NeoSendTiming,
  ParapperConfig,
  TranslationBackend,
  TranslationHttpListenerStatus,
  TranslationLanguage,
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
const defaultLocalTranslationModel: LocalTranslationModel = "lfm2_q4";

export const TranslationSettings: React.FC<TranslationSettingsProps> = ({
  config,
  runtimeLocked,
  onUpdateConfig,
}) => {
  const { t } = useTranslation();
  const { asrModelSelectOptions } = useAsrModelOptions(config.asr_model);
  const yncPluginAvailable = !isMacOs();
  const [translationListenerStatus, setTranslationListenerStatus] =
    useState<TranslationHttpListenerStatus>({
      state: "stopped",
      port: null,
      error: null,
    });
  const [translationListenerPending, setTranslationListenerPending] =
    useState(false);
  const [serverModelInstalled, setServerModelInstalled] = useState<
    boolean | null
  >(null);
  const [downloadingServerModel, setDownloadingServerModel] = useState(false);
  const translationListenerRunning = [
    "starting",
    "running",
    "stopping",
  ].includes(translationListenerStatus.state);
  const serverModelMissing = serverModelInstalled === false;
  const asrModelOptions = modelOptionsWithAny(
    t("translationSettings.mapping.anyModel"),
    asrModelSelectOptions,
  );

  useEffect(() => {
    void invoke<TranslationHttpListenerStatus>(
      "get_translation_http_listener_status",
    ).then(setTranslationListenerStatus);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setServerModelInstalled(null);
    void invoke<boolean>("get_local_translation_model_installed", {
      model: config.translation_local_server_model,
    }).then((installed) => {
      if (!cancelled) {
        setServerModelInstalled(installed);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [config.translation_local_server_model]);

  const downloadServerModel = () => {
    setDownloadingServerModel(true);
    void invoke<boolean>("download_local_translation_model", {
      model: config.translation_local_server_model,
    })
      .then(setServerModelInstalled)
      .catch((error) =>
        notifyParapperIssue(normalizeParapperErrorPayload(error)),
      )
      .finally(() => setDownloadingServerModel(false));
  };

  return (
    <Stack gap="sm">
      <DisabledReasonTooltip
        disabled={runtimeLocked}
        label={t("tooltip.runtimeLocked")}
      >
        <Switch
          label={settingLabel(
            t("translationSettings.enable.label"),
            t("translationSettings.enable.description"),
          )}
          checked={config.translation_enabled}
          disabled={runtimeLocked}
          onChange={(event) =>
            onUpdateConfig("translation_enabled", event.currentTarget.checked)
          }
        />
      </DisabledReasonTooltip>
      <SegmentedControl
        aria-label={t("translationSettings.sendTiming.label")}
        value={config.translation_send_timing}
        disabled={runtimeLocked}
        data={sendTimingOptions(t)}
        onChange={(value) =>
          onUpdateConfig("translation_send_timing", value as NeoSendTiming)
        }
      />
      <TranslationMappingRows
        mappings={config.translation_mappings}
        asrModelOptions={asrModelOptions}
        yncPluginAvailable={yncPluginAvailable}
        disabled={runtimeLocked}
        onChange={(translationMappings) =>
          onUpdateConfig("translation_mappings", translationMappings)
        }
      />
      <Paper withBorder radius="md" p="md" style={{ order: -1 }}>
        <Stack gap="sm">
          <Text fw={600}>{t("translationSettings.localServer.title")}</Text>
          <Group grow align="end">
            <NumberInput
              label={t("translationSettings.localServer.port.label")}
              value={config.translation_local_server_port}
              min={1}
              max={65535}
              disabled={runtimeLocked || translationListenerRunning}
              onChange={(value) =>
                onUpdateConfig(
                  "translation_local_server_port",
                  typeof value === "number" ? value : 18081,
                )
              }
            />
            <Select
              label={t("translationSettings.localServer.model.label")}
              data={localTranslationModelOptions}
              value={config.translation_local_server_model}
              allowDeselect={false}
              disabled={
                runtimeLocked ||
                translationListenerRunning ||
                downloadingServerModel
              }
              onChange={(value) =>
                onUpdateConfig(
                  "translation_local_server_model",
                  (value ??
                    defaultLocalTranslationModel) as LocalTranslationModel,
                )
              }
            />
          </Group>
          <Group justify="space-between">
            <Text
              size="xs"
              c={translationListenerStatus.state === "error" ? "red" : "dimmed"}
            >
              {translationListenerStatus.error ??
                (serverModelMissing && !translationListenerRunning
                  ? t("translationSettings.localServer.status.modelMissing")
                  : t(
                      `translationSettings.localServer.status.${translationListenerStatus.state}`,
                      { port: translationListenerStatus.port },
                    ))}
            </Text>
            {serverModelMissing && !translationListenerRunning ? (
              <Button
                variant="filled"
                loading={downloadingServerModel}
                disabled={runtimeLocked}
                onClick={downloadServerModel}
              >
                {t("translationSettings.localServer.downloadModel")}
              </Button>
            ) : (
              <Button
                variant={translationListenerRunning ? "light" : "filled"}
                loading={translationListenerPending}
                disabled={
                  translationListenerStatus.state === "starting" ||
                  translationListenerStatus.state === "stopping"
                }
                onClick={() => {
                  setTranslationListenerPending(true);
                  const command = translationListenerRunning
                    ? invoke<TranslationHttpListenerStatus>(
                        "stop_translation_http_listener",
                      )
                    : invoke<TranslationHttpListenerStatus>(
                        "start_translation_http_listener",
                        {
                          port: config.translation_local_server_port,
                          localModel: config.translation_local_server_model,
                        },
                      );
                  void command
                    .then(setTranslationListenerStatus)
                    .catch(() =>
                      invoke<TranslationHttpListenerStatus>(
                        "get_translation_http_listener_status",
                      ).then(setTranslationListenerStatus),
                    )
                    .finally(() => setTranslationListenerPending(false));
                }}
              >
                {t(
                  translationListenerRunning
                    ? "translationSettings.localServer.stop"
                    : "translationSettings.localServer.start",
                )}
              </Button>
            )}
          </Group>
          <Accordion variant="contained">
            <Accordion.Item value="ync-neo-setup">
              <Accordion.Control>
                {t("translationSettings.localServer.setup.title")}
              </Accordion.Control>
              <Accordion.Panel>
                <List size="sm" spacing="xs">
                  <List.Item>
                    {t("translationSettings.localServer.setup.engine")}
                  </List.Item>
                  <List.Item>
                    {t("translationSettings.localServer.setup.url", {
                      port: config.translation_local_server_port,
                    })}
                  </List.Item>
                  <List.Item>
                    {t("translationSettings.localServer.setup.postMode")}
                  </List.Item>
                  <List.Item>
                    {t("translationSettings.localServer.setup.model")}
                  </List.Item>
                  <List.Item>
                    {t("translationSettings.localServer.setup.avoidInputApi")}
                  </List.Item>
                </List>
              </Accordion.Panel>
            </Accordion.Item>
          </Accordion>
        </Stack>
      </Paper>
    </Stack>
  );
};

const TranslationMappingRows: React.FC<{
  mappings: TranslationMapping[];
  asrModelOptions: { value: string; label: string }[];
  yncPluginAvailable: boolean;
  disabled: boolean;
  onChange: (mappings: TranslationMapping[]) => void;
}> = ({
  mappings,
  asrModelOptions,
  yncPluginAvailable,
  disabled,
  onChange,
}) => {
  const { t } = useTranslation();
  const backendOptions = [
    {
      value: "ync",
      label: t("translationSettings.backend.ync"),
      disabled: !yncPluginAvailable,
    },
    {
      value: "local",
      label: t("translationSettings.backend.local"),
    },
  ];
  const defaultBackend: TranslationBackend = yncPluginAvailable
    ? "ync"
    : "local";

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
        backend: defaultBackend,
        local_model: defaultLocalTranslationModel,
        source_lang: "ja" as TranslationLanguage,
        target_lang: "en" as TranslationLanguage,
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
        <Stack gap="xs">
          <Select
            label={t("translationSettings.backend.label")}
            data={backendOptions}
            value={mapping.backend}
            allowDeselect={false}
            disabled={disabled}
            onChange={(value) =>
              updateMapping({
                backend: (value ?? defaultBackend) as TranslationBackend,
              })
            }
          />
          {mapping.backend === "local" ? (
            <Select
              label={t("translationSettings.localModel.label")}
              data={localTranslationModelOptions}
              value={mapping.local_model}
              allowDeselect={false}
              disabled={disabled}
              onChange={(value) =>
                updateMapping({
                  local_model: (value ??
                    defaultLocalTranslationModel) as LocalTranslationModel,
                })
              }
            />
          ) : null}
          <Group grow>
            <Select
              label={t("translationSettings.mapping.sourceLang")}
              data={translationLanguageOptions}
              value={mapping.source_lang}
              allowDeselect={false}
              disabled={disabled}
              onChange={(value) => {
                const sourceLang = (value ?? "ja") as TranslationLanguage;
                updateMapping({
                  source_lang: sourceLang,
                  target_lang:
                    mapping.target_lang === sourceLang
                      ? oppositeTranslationLanguage(sourceLang)
                      : mapping.target_lang,
                });
              }}
            />
            <Select
              label={t("translationSettings.mapping.targetLang")}
              data={translationLanguageOptions}
              value={mapping.target_lang}
              allowDeselect={false}
              disabled={disabled}
              onChange={(value) => {
                const targetLang = (value ?? "en") as TranslationLanguage;
                updateMapping({
                  target_lang: targetLang,
                  source_lang:
                    mapping.source_lang === targetLang
                      ? oppositeTranslationLanguage(targetLang)
                      : mapping.source_lang,
                });
              }}
            />
          </Group>
        </Stack>
      )}
    />
  );
};

const oppositeTranslationLanguage = (
  language: TranslationLanguage,
): TranslationLanguage => (language === "ja" ? "en" : "ja");
