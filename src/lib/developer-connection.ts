import type { ParapperConfig } from "./types";

export const configWithDeveloperConnectionEnabled = (
  config: ParapperConfig,
  enabled: boolean,
): ParapperConfig => ({
  ...config,
  streaming_recognition_enabled: enabled,
});
