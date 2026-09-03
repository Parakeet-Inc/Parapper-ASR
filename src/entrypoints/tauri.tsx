import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";

import {
  ColorSchemeScript,
  localStorageColorSchemeManager,
  MantineProvider,
} from "@mantine/core";
import { Notifications } from "@mantine/notifications";
import { attachConsole } from "@tauri-apps/plugin-log";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "../i18n";
import { App } from "../app";
import { theme } from "../lib/theme";
import {
  createTauriFrontendServices,
  desktopCapabilities,
} from "../platform/tauri/frontend-services";

attachConsole();

const services = createTauriFrontendServices();
const capabilities = desktopCapabilities();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ColorSchemeScript defaultColorScheme="auto" />
    <MantineProvider
      theme={theme}
      defaultColorScheme="auto"
      colorSchemeManager={localStorageColorSchemeManager({
        key: "mantine-color-scheme",
      })}
    >
      <Notifications />
      <App services={services} capabilities={capabilities} />
    </MantineProvider>
  </StrictMode>,
);
