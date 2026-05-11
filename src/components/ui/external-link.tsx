import { Anchor } from "@mantine/core";
import { invoke } from "@tauri-apps/api/core";
import type React from "react";

import { notificationColor } from "../../lib/theme";

const openExternalUrl = async (url: string) => {
  try {
    await invoke("open_external_url", { url });
  } catch (error) {
    console.warn("Failed to open external URL through Tauri", error);
    window.open(url, "_blank", "noopener,noreferrer");
  }
};

export const ExternalLink: React.FC<{
  href: string;
  children: React.ReactNode;
}> = ({ href, children }) => (
  <Anchor
    href={href}
    target="_blank"
    rel="noreferrer"
    c={notificationColor.info}
    onClick={(event) => {
      event.preventDefault();
      void openExternalUrl(href);
    }}
  >
    {children}
  </Anchor>
);
