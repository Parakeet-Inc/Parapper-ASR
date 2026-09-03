import {
  Accordion,
  Button,
  Checkbox,
  Collapse,
  Code,
  ColorSwatch,
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
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { STT_PROFILE_DISPLAY_COLOR_CSS } from "../../lib/stt-profile-colors";
import { sttProfileDisplayName } from "../../lib/stt-profiles";
import { notificationColor } from "../../lib/theme";
import type {
  DeveloperConnectionMode,
  ParapperConfig,
  SttProfileConfig,
  StreamingRecognitionOutputMode,
} from "../../lib/types";

type ExternalAppSettingsProps = {
  config: ParapperConfig;
  sttProfiles: readonly SttProfileConfig[];
  runtimeLocked: boolean;
  nativeConnectionsAvailable: boolean;
  onFindNeoPort: () => Promise<number | null>;
  onFindYncPluginPort: () => Promise<number | null>;
  onSetDeveloperConnectionEnabled: (enabled: boolean) => void;
  onSetSttProfileNeoHttpEnabled: (profileId: string, enabled: boolean) => void;
  onSetSttProfileDeveloperHttpEnabled: (
    profileId: string,
    enabled: boolean,
  ) => void;
  onUpdateConfig: <K extends keyof ParapperConfig>(
    key: K,
    value: ParapperConfig[K],
  ) => void;
};

export const ExternalAppSettings: React.FC<ExternalAppSettingsProps> = ({
  config,
  sttProfiles,
  runtimeLocked,
  nativeConnectionsAvailable,
  onFindNeoPort,
  onFindYncPluginPort,
  onSetDeveloperConnectionEnabled,
  onSetSttProfileNeoHttpEnabled,
  onSetSttProfileDeveloperHttpEnabled,
  onUpdateConfig,
}) => {
  const { t } = useTranslation();
  const [detectingNeoPort, setDetectingNeoPort] = useState(false);
  const [detectingPluginPort, setDetectingPluginPort] = useState(false);
  const nativeConnectionsDisabled = !nativeConnectionsAvailable;
  const neoEnabled = !nativeConnectionsDisabled && config.neo_http_enabled;
  const developerEnabled = config.streaming_recognition_enabled;

  const findPort = async (
    find: () => Promise<number | null>,
    key: "neo_http_port" | "ync_plugin_port",
    notificationKey: "neoPort" | "pluginPort",
    setDetecting: (value: boolean) => void,
  ) => {
    setDetecting(true);
    try {
      const port = await find();
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
      <ExternalAppSection
        enabled={neoEnabled}
        title={t("externalAppSettings.yncEnabled")}
        disabled={nativeConnectionsDisabled || runtimeLocked}
        onToggle={(enabled) => onUpdateConfig("neo_http_enabled", enabled)}
        alwaysVisible={
          <Stack gap="sm">
            <Text size="sm" fw={500}>
              {t("externalAppSettings.yncPlugin")}
            </Text>
            <PortSetting
              label={t("settings.translationPluginHttpPort.label")}
              value={config.ync_plugin_port}
              loading={detectingPluginPort}
              disabled={nativeConnectionsDisabled || runtimeLocked}
              findLabel={t("common.search")}
              onChange={(port) => onUpdateConfig("ync_plugin_port", port)}
              onFind={() =>
                void findPort(
                  onFindYncPluginPort,
                  "ync_plugin_port",
                  "pluginPort",
                  setDetectingPluginPort,
                )
              }
            />
          </Stack>
        }
      >
        <>
          <PortSetting
            label={t("settings.neoHttpPort.label")}
            value={config.neo_http_port}
            loading={detectingNeoPort}
            disabled={runtimeLocked}
            findLabel={t("common.search")}
            onChange={(port) => onUpdateConfig("neo_http_port", port)}
            onFind={() =>
              void findPort(
                onFindNeoPort,
                "neo_http_port",
                "neoPort",
                setDetectingNeoPort,
              )
            }
          />
          <SttProfileDestinationList
            title={t("connectionSettings.neoProfiles")}
            profiles={sttProfiles}
            disabled={runtimeLocked}
            checked={(profile) => profile.neo_http_enabled}
            ariaLabel={(profileName) =>
              t("sttProfiles.neoHttpEnabledFor", { name: profileName })
            }
            onChange={onSetSttProfileNeoHttpEnabled}
          />
        </>
      </ExternalAppSection>

      <Paper withBorder radius="md" p="md">
        <Stack gap="sm">
          <Text fw={600}>{t("settings.oscQuery.label")}</Text>
          <Switch
            label={t("settings.oscQuery.muteSyncLabel")}
            checked={!nativeConnectionsDisabled && config.vrc_osc_micmute}
            disabled={nativeConnectionsDisabled || runtimeLocked}
            onChange={(event) =>
              onUpdateConfig("vrc_osc_micmute", event.currentTarget.checked)
            }
          />
        </Stack>
      </Paper>

      <ExternalAppSection
        enabled={developerEnabled}
        title={t("connectionSettings.streamingEnabled")}
        disabled={runtimeLocked}
        onToggle={onSetDeveloperConnectionEnabled}
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
            <SttProfileDestinationList
              title={t("connectionSettings.httpProfiles")}
              profiles={sttProfiles}
              disabled={runtimeLocked}
              checked={(profile) => profile.developer_http_enabled}
              ariaLabel={(profileName) =>
                t("sttProfiles.developerHttpEnabledFor", {
                  name: profileName,
                })
              }
              onChange={onSetSttProfileDeveloperHttpEnabled}
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
      </ExternalAppSection>
    </Stack>
  );
};

const SttProfileDestinationList: React.FC<{
  title: string;
  profiles: readonly SttProfileConfig[];
  disabled: boolean;
  checked: (profile: SttProfileConfig) => boolean;
  ariaLabel: (profileName: string) => string;
  onChange: (profileId: string, enabled: boolean) => void;
}> = ({ title, profiles, disabled, checked, ariaLabel, onChange }) => {
  const { t } = useTranslation();
  return (
    <Stack gap={6}>
      <Text size="sm" fw={500}>
        {title}
      </Text>
      {profiles.map((profile) => {
        const profileName = sttProfileDisplayName(profile, (number) =>
          t("sttProfiles.defaultName", { number }),
        );
        return (
          <Checkbox
            key={profile.id}
            label={
              <Group gap="xs" wrap="nowrap">
                <ColorSwatch
                  color={STT_PROFILE_DISPLAY_COLOR_CSS[profile.display_color]}
                  size={14}
                  style={{
                    border: "1px solid var(--mantine-color-default-border)",
                  }}
                />
                <Text size="sm">{profileName}</Text>
              </Group>
            }
            aria-label={ariaLabel(profileName)}
            checked={checked(profile)}
            disabled={disabled}
            onChange={(event) =>
              onChange(profile.id, event.currentTarget.checked)
            }
          />
        );
      })}
    </Stack>
  );
};

const ExternalAppSection: React.FC<{
  enabled: boolean;
  title: string;
  disabled: boolean;
  onToggle: (enabled: boolean) => void;
  alwaysVisible?: React.ReactNode;
  children: React.ReactNode;
}> = ({ enabled, title, disabled, onToggle, alwaysVisible, children }) => (
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
      {alwaysVisible}
    </Stack>
  </Paper>
);

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
