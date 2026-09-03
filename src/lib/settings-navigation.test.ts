import { describe, expect, it } from "vitest";

import { settingsNavigationForGroup } from "./settings-navigation";

describe("settingsNavigationForGroup", () => {
  it("keeps the requested STT, output, and other navigation order", () => {
    expect(
      ["stt", "output", "other"].map((group) =>
        settingsNavigationForGroup(group as "stt" | "output" | "other").map(
          (item) => item.tab,
        ),
      ),
    ).toEqual([
      ["connection", "noise-cancellation", "vad", "asr"],
      ["external-apps", "translation", "speech"],
      ["other", "downloads", "licenses"],
    ]);
  });
});
