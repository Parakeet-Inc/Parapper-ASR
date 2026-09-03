import { describe, expect, it } from "vitest";

import { buildNoiseCancellationTargetOptions } from "./settings-options";

describe("noise cancellation target options", () => {
  it("offers VAD-only first and keeps the legacy VAD-and-ASR mode selectable", () => {
    const options = buildNoiseCancellationTargetOptions((key) => key);

    expect(options).toEqual([
      {
        label: "options.noiseCancellationTarget.vadOnly",
        value: "vad_only",
      },
      {
        label: "options.noiseCancellationTarget.vadAndAsr",
        value: "vad_and_asr",
      },
    ]);
  });
});
