import { Button, Group, NumberInput, Stack, Switch } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { isMacOs } from "../../lib/platform";
import { notificationColor } from "../../lib/theme";
import type { ParapperConfig } from "../../lib/types";
import { DisabledReasonTooltip, settingLabel } from "../ui/display";

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
  const runtimeLockedTooltip = t("tooltip.runtimeLocked");
  const nativeConnectionsDisabled = isMacOs();

  const findNeoHttpPort = async () => {
    setDetectingNeoPort(true);
    try {
      const port = await invoke<number | null>("find_neo_http_port");
      if (!port) {
        notifications.show({
          title: t("notifications.neoPortNotFound.title"),
          message: t("notifications.neoPortNotFound.message"),
          color: notificationColor.warn,
        });
        return;
      }
      onUpdateConfig("neo_http_port", port);
      notifications.show({
        title: t("notifications.neoPortDetected.title"),
        message: t("notifications.neoPortDetected.message", { port }),
      });
    } finally {
      setDetectingNeoPort(false);
    }
  };

  return (
    <Stack gap="sm">
      <Stack gap={4}>
        {settingLabel(
          t("settings.neoHttpEnabled.label"),
          t("settings.neoHttpEnabled.description"),
        )}
        <Switch
          aria-label={t("settings.neoHttpEnabled.label")}
          checked={!nativeConnectionsDisabled && config.neo_http_enabled}
          disabled={nativeConnectionsDisabled}
          onChange={(event) =>
            onUpdateConfig("neo_http_enabled", event.currentTarget.checked)
          }
        />
      </Stack>
      <Group align="end" gap="xs" wrap="nowrap">
        <NumberInput
          label={settingLabel(
            t("settings.neoHttpPort.label"),
            t("settings.neoHttpPort.description"),
          )}
          value={config.neo_http_port}
          min={1}
          max={65535}
          disabled={nativeConnectionsDisabled}
          style={{ flex: 1 }}
          onChange={(value) =>
            onUpdateConfig(
              "neo_http_port",
              typeof value === "number" ? value : 15520,
            )
          }
        />
        <Button
          variant="light"
          loading={!nativeConnectionsDisabled && detectingNeoPort}
          disabled={nativeConnectionsDisabled}
          onClick={() => void findNeoHttpPort()}
        >
          {t("common.search")}
        </Button>
      </Group>
      <Stack gap={4}>
        {settingLabel(
          t("settings.oscQuery.muteSyncLabel"),
          t("settings.oscQuery.muteSyncDescription"),
        )}
        <DisabledReasonTooltip
          disabled={runtimeLocked}
          label={runtimeLockedTooltip}
        >
          <Switch
            aria-label={t("settings.oscQuery.muteSyncLabel")}
            checked={!nativeConnectionsDisabled && config.vrc_osc_micmute}
            disabled={nativeConnectionsDisabled || runtimeLocked}
            onChange={(event) =>
              onUpdateConfig("vrc_osc_micmute", event.currentTarget.checked)
            }
          />
        </DisabledReasonTooltip>
      </Stack>
    </Stack>
  );
};
