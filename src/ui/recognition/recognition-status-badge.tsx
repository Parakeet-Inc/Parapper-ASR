import { Badge } from "@mantine/core";

import { notificationColor } from "../../lib/theme";
import type { RecognitionStatus } from "../../lib/types";

const statusColor = (status: RecognitionStatus) => {
  switch (status) {
    case "listening":
      return notificationColor.ok;
    case "waiting_for_client":
      return notificationColor.info;
    case "draining":
      return notificationColor.warn;
    case "error":
      return notificationColor.error;
    case "stopped":
      return "gray";
    default:
      return notificationColor.info;
  }
};

type RecognitionStatusBadgeProps = {
  status: RecognitionStatus;
  label: string;
};

export const RecognitionStatusBadge: React.FC<RecognitionStatusBadgeProps> = ({
  status,
  label,
}) => (
  <Badge color={statusColor(status)} variant="light">
    {label}
  </Badge>
);
