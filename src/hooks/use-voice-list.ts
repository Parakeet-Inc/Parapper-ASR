import { notifications } from "@mantine/notifications";
import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";

import { notificationColor } from "../lib/theme";

export const useVoiceList = (
  port: number,
  fetchVoices: (port: number) => Promise<string[]>,
) => {
  const { t } = useTranslation();
  const [voiceList, setVoiceList] = useState<string[]>([]);
  const [refreshingVoiceList, setRefreshingVoiceList] = useState(false);

  const refreshVoiceList = useCallback(async () => {
    setRefreshingVoiceList(true);
    try {
      const loadedVoiceList = await fetchVoices(port);
      setVoiceList(loadedVoiceList);
    } catch (error) {
      notifications.show({
        title: t("notifications.voiceListLoadFailed.title"),
        message: String(error),
        color: notificationColor.error,
      });
    } finally {
      setRefreshingVoiceList(false);
    }
  }, [fetchVoices, port, t]);

  return { voiceList, refreshingVoiceList, refreshVoiceList };
};
