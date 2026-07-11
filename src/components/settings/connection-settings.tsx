import {
  Accordion,
  Button,
  Code,
  Collapse,
  Group,
  NumberInput,
  Paper,
  PasswordInput,
  SegmentedControl,
  Stack,
  Switch,
  Text,
  TextInput,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { isMacOs } from "../../lib/platform";
import { notificationColor } from "../../lib/theme";
import type {
  DeveloperConnectionMode,
  ParapperConfig,
  StreamingRecognitionOutputMode,
} from "../../lib/types";

type ConnectionSettingsProps = {
  config: ParapperConfig;
  runtimeLocked: boolean;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
};

export const ConnectionSettings: React.FC<ConnectionSettingsProps> = ({
  config,
  runtimeLocked,
  onUpdateConfig,
}) => {
  const { t } = useTranslation();
  const [detectingNeoPort, setDetectingNeoPort] = useState(false);
  const [detectingPluginPort, setDetectingPluginPort] = useState(false);
  const nativeConnectionsDisabled = isMacOs();
  const neoEnabled = !nativeConnectionsDisabled && config.neo_http_enabled;
  const developerEnabled = config.streaming_recognition_enabled;

  const findPort = async (
    command: "find_neo_http_port" | "find_ync_plugin_http_port",
    key: "neo_http_port" | "ync_plugin_port",
    notificationKey: "neoPort" | "pluginPort",
    setDetecting: (value: boolean) => void,
  ) => {
    setDetecting(true);
    try {
      const port = await invoke<number | null>(command);
      if (!port) {
        notifications.show({
          title: t(`notifications.${notificationKey}NotFound.title`),
          message: t(`notifications.${notificationKey}NotFound.message`),
          color: notificationColor.warn,
        });
        return;
      }
      onUpdateConfig(key, port);
      notifications.show({
        title: t(`notifications.${notificationKey}Detected.title`),
        message: t(`notifications.${notificationKey}Detected.message`, {
          port,
        }),
      });
    } finally {
      setDetecting(false);
    }
  };

  return (
    <Stack gap="md">
      <ConnectionSection
        enabled={neoEnabled}
        title={t("connectionSettings.neoEnabled")}
        disabled={nativeConnectionsDisabled || runtimeLocked}
        onToggle={(enabled) => onUpdateConfig("neo_http_enabled", enabled)}
      >
        <PortSetting
          label={t("settings.neoHttpPort.label")}
          value={config.neo_http_port}
          loading={detectingNeoPort}
          disabled={runtimeLocked}
          findLabel={t("common.search")}
          onChange={(port) => onUpdateConfig("neo_http_port", port)}
          onFind={() =>
            void findPort(
              "find_neo_http_port",
              "neo_http_port",
              "neoPort",
              setDetectingNeoPort,
            )
          }
        />
        <PortSetting
          label={t("settings.translationPluginHttpPort.label")}
          value={config.ync_plugin_port}
          loading={detectingPluginPort}
          disabled={runtimeLocked}
          findLabel={t("common.search")}
          onChange={(port) => onUpdateConfig("ync_plugin_port", port)}
          onFind={() =>
            void findPort(
              "find_ync_plugin_http_port",
              "ync_plugin_port",
              "pluginPort",
              setDetectingPluginPort,
            )
          }
        />
        <Switch
          label={t("settings.oscQuery.muteSyncLabel")}
          checked={!nativeConnectionsDisabled && config.vrc_osc_micmute}
          disabled={nativeConnectionsDisabled || runtimeLocked}
          onChange={(event) =>
            onUpdateConfig("vrc_osc_micmute", event.currentTarget.checked)
          }
        />
      </ConnectionSection>

      <ConnectionSection
        enabled={developerEnabled}
        title={t("connectionSettings.developerEnabled")}
        disabled={runtimeLocked}
        onToggle={(enabled) =>
          onUpdateConfig("streaming_recognition_enabled", enabled)
        }
      >
        <Stack gap={6}>
          <Text size="sm" fw={500}>
            {t("connectionSettings.connectionMode")}
          </Text>
          <SegmentedControl
            fullWidth
            value={config.developer_connection_mode}
            disabled={runtimeLocked}
            data={[
              { value: "http", label: "HTTP" },
              { value: "web_socket", label: "WebSocket" },
            ]}
            onChange={(value) =>
              onUpdateConfig(
                "developer_connection_mode",
                value as DeveloperConnectionMode,
              )
            }
          />
        </Stack>

        {config.developer_connection_mode === "http" ? (
          <>
            <TextInput
              label={t("connectionSettings.httpUrl")}
              value={config.developer_http_url}
              disabled={runtimeLocked}
              onChange={(event) =>
                onUpdateConfig("developer_http_url", event.currentTarget.value)
              }
            />
            <Accordion variant="contained">
              <Accordion.Item value="http-payload-example">
                <Accordion.Control>
                  {t("connectionSettings.httpPayloadExample")}
                </Accordion.Control>
                <Accordion.Panel>
                  <Code block>{developerHttpPayloadExample}</Code>
                </Accordion.Panel>
              </Accordion.Item>
            </Accordion>
          </>
        ) : (
          <>
            <TextInput
              label={t("connectionSettings.bindAddress")}
              value={config.streaming_recognition_bind_address}
              disabled={runtimeLocked}
              onChange={(event) =>
                onUpdateConfig(
                  "streaming_recognition_bind_address",
                  event.currentTarget.value,
                )
              }
            />
            <NumberInput
              label={t("connectionSettings.port")}
              value={config.streaming_recognition_port}
              min={1}
              max={65535}
              disabled={runtimeLocked}
              onChange={(value) =>
                onUpdateConfig(
                  "streaming_recognition_port",
                  typeof value === "number" ? value : 18082,
                )
              }
            />
            <PasswordInput
              label={t("connectionSettings.apiKey")}
              placeholder={t("connectionSettings.apiKeyPlaceholder")}
              value={config.streaming_recognition_api_key ?? ""}
              disabled={runtimeLocked}
              onChange={(event) =>
                onUpdateConfig(
                  "streaming_recognition_api_key",
                  event.currentTarget.value || null,
                )
              }
            />
            <Stack gap={4}>
              <Text size="sm" fw={500}>
                {t("connectionSettings.outputMode")}
              </Text>
              <SegmentedControl
                value={config.streaming_recognition_output_mode}
                disabled={runtimeLocked}
                data={[
                  {
                    value: "web_socket_only",
                    label: t("connectionSettings.webSocketOnly"),
                  },
                  {
                    value: "web_socket_and_desktop",
                    label: t("connectionSettings.webSocketAndDesktop"),
                  },
                ]}
                onChange={(value) =>
                  onUpdateConfig(
                    "streaming_recognition_output_mode",
                    value as StreamingRecognitionOutputMode,
                  )
                }
              />
            </Stack>
            <Text size="xs" c="dimmed">
              {t("connectionSettings.endpoint", {
                address: config.streaming_recognition_bind_address,
                port: config.streaming_recognition_port,
              })}
            </Text>
          </>
        )}
      </ConnectionSection>
    </Stack>
  );
};

const ConnectionSection: React.FC<{
  enabled: boolean;
  title: string;
  disabled: boolean;
  onToggle: (enabled: boolean) => void;
  children: React.ReactNode;
}> = ({ enabled, title, disabled, onToggle, children }) => (
  <Paper withBorder radius="md" p="md">
    <Stack gap="sm">
      <Switch
        checked={enabled}
        disabled={disabled}
        label={<Text fw={600}>{title}</Text>}
        onChange={(event) => onToggle(event.currentTarget.checked)}
      />
      <Collapse in={enabled}>
        <Stack gap="sm" pt="xs">
          {children}
        </Stack>
      </Collapse>
    </Stack>
  </Paper>
);

const developerHttpPayloadExample = `{
  "version": 1,
  "type": "turn.final",
  "id": "turn-3",
  "text": "こんにちは。",
  "turn_session_id": 7,
  "turn_id": 3,
  "revision": 2,
  "output_sequence": 4,
  "segment_id": 8,
  "previous_segment_id": 7,
  "source_asr_model": "reazonspeech_k2_v2",
  "source_language": "japanese",
  "detected_language": null,
  "recognized_at_ms": 1000,
  "elapsed_ms": 96,
  "audio_duration_ms": 1280
}`;

const PortSetting: React.FC<{
  label: string;
  value: number;
  loading: boolean;
  disabled: boolean;
  findLabel: string;
  onChange: (value: number) => void;
  onFind: () => void;
}> = ({ label, value, loading, disabled, findLabel, onChange, onFind }) => (
  <Group align="end" gap="xs" wrap="nowrap">
    <NumberInput
      label={label}
      value={value}
      min={1}
      max={65535}
      disabled={disabled}
      style={{ flex: 1 }}
      onChange={(next) => onChange(typeof next === "number" ? next : value)}
    />
    <Button
      variant="light"
      loading={loading}
      disabled={disabled}
      onClick={onFind}
    >
      {findLabel}
    </Button>
  </Group>
);
