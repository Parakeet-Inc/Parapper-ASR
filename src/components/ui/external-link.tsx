import { Anchor } from "@mantine/core";
import type React from "react";

import { notificationColor } from "../../lib/theme";

export const ExternalLink: React.FC<{
  href: string;
  children: React.ReactNode;
  onOpen: (url: string) => Promise<void>;
}> = ({ href, children, onOpen }) => (
  <Anchor
    href={href}
    target="_blank"
    rel="noreferrer"
    c={notificationColor.info}
    onClick={(event) => {
      event.preventDefault();
      void onOpen(href);
    }}
  >
    {children}
  </Anchor>
);
