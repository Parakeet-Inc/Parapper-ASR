import { notifications } from "@mantine/notifications";
import { invoke } from "@tauri-apps/api/core";
import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";

import { notificationColor } from "../lib/theme";

export const useVoiceList = (port: number) => {
  const { t } = useTranslation();
  const [voiceList, setVoiceList] = useState<string[]>([]);
  const [refreshingVoiceList, setRefreshingVoiceList] = useState(false);

  const refreshVoiceList = useCallback(async () => {
    setRefreshingVoiceList(true);
    try {
      const loadedVoiceList = await invoke<string[]>("fetch_neo_voice_list", {
        port,
      });
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
  }, [port, t]);

  return { voiceList, refreshingVoiceList, refreshVoiceList };
};
