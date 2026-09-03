import { describe, expect, it } from "vitest";

import { STT_PROFILE_DISPLAY_COLOR_CSS } from "./stt-profile-colors";

describe("STT_PROFILE_DISPLAY_COLOR_CSS", () => {
  it("maps every persisted profile color to the same Mantine palette shade", () => {
    expect(STT_PROFILE_DISPLAY_COLOR_CSS).toEqual({
      green: "var(--mantine-color-green-6)",
      blue: "var(--mantine-color-blue-6)",
      violet: "var(--mantine-color-violet-6)",
      red: "var(--mantine-color-red-6)",
      orange: "var(--mantine-color-orange-6)",
      yellow: "var(--mantine-color-yellow-6)",
      white: "transparent",
    });
  });
});
