import { describe, expect, it } from "vitest";

import { configWithDeveloperConnectionEnabled } from "./developer-connection";
import type { ParapperConfig } from "./types";

describe("configWithDeveloperConnectionEnabled", () => {
  it("keeps HTTP mode when enabling all-profile delivery with multiple STT profiles", () => {
    const config = {
      streaming_recognition_enabled: false,
      developer_connection_mode: "http",
      stt_profiles: [{ id: "first" }, { id: "second" }],
    } as ParapperConfig;

    expect(configWithDeveloperConnectionEnabled(config, true)).toMatchObject({
      streaming_recognition_enabled: true,
      developer_connection_mode: "http",
    });
  });

  it("keeps HTTP mode when there is only one STT profile", () => {
    const config = {
      streaming_recognition_enabled: false,
      developer_connection_mode: "http",
      stt_profiles: [{ id: "only" }],
    } as ParapperConfig;

    expect(configWithDeveloperConnectionEnabled(config, true)).toMatchObject({
      streaming_recognition_enabled: true,
      developer_connection_mode: "http",
    });
  });
});
