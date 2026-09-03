import { Badge, Group } from "@mantine/core";
import { useTranslation } from "react-i18next";

import type { RuntimeState } from "../hooks/use-app-state";
import { errorColor, getParapperErrorMessage } from "../lib/error";
import { notificationColor } from "../lib/theme";
import { RecognitionStatusBadge } from "../ui/recognition/recognition-status-badge";

type StatusBadgesProps = {
  runtime: RuntimeState;
  nativeConnectionsAvailable: boolean;
};

export const StatusBadges: React.FC<StatusBadgesProps> = ({
  runtime,
  nativeConnectionsAvailable,
}) => {
  const { t } = useTranslation();
  const nativeConnectionsDisabled = !nativeConnectionsAvailable;
  const vrcMicState =
    runtime.oscMuted === null
      ? t("status.idle")
      : runtime.oscMuted
        ? t("status.muted")
        : t("status.open");

  return (
    <Group gap="xs" justify="flex-end">
      <RecognitionStatusBadge
        status={runtime.status}
        label={t(`status.recognition.${runtime.status}`)}
      />
      <Badge
        color={
          runtime.vadState?.state === "speech" ? notificationColor.info : "gray"
        }
        variant="light"
      >
        {t("status.vad", {
          state: runtime.vadState
            ? t(`status.vadState.${runtime.vadState.state}`)
            : t("status.idle"),
        })}
      </Badge>
      {!nativeConnectionsDisabled ? (
        <Badge
          color={runtime.oscMuted ? notificationColor.error : "gray"}
          variant="light"
        >
          {t("status.vrcMic", {
            state: vrcMicState,
          })}
        </Badge>
      ) : null}
      {!nativeConnectionsDisabled && runtime.neoNotFound ? (
        <Badge color={notificationColor.warn} variant="light">
          {t("status.neoNotFound")}
        </Badge>
      ) : null}
      {!nativeConnectionsDisabled && runtime.vrcNotFound ? (
        <Badge color={notificationColor.warn} variant="light">
          {t("status.vrcNotFound")}
        </Badge>
      ) : null}
      {runtime.lastError ? (
        <Badge color={errorColor(runtime.lastError.severity)} variant="light">
          {getParapperErrorMessage(runtime.lastError)}
        </Badge>
      ) : null}
    </Group>
  );
};
