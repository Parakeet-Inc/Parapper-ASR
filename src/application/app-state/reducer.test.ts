import { describe, expect, it } from "vitest";

import {
  applyConnectionAvailability,
  applyInputLevel,
  applyRecognitionStatus,
  initialRuntimeState,
} from "./reducer";

describe("application runtime reducer", () => {
  it("derives running state from the backend status and preserves unrelated diagnostics", () => {
    const withWarning = {
      ...initialRuntimeState,
      asrWarning: "model warning",
    };

    expect(applyRecognitionStatus(withWarning, "listening")).toEqual({
      ...withWarning,
      status: "listening",
      running: true,
      starting: false,
    });
    expect(applyRecognitionStatus(withWarning, "stopped")).toEqual({
      ...withWarning,
      status: "stopped",
      running: false,
      starting: false,
    });
  });

  it("updates only the connection target reported by the platform", () => {
    const unavailable = applyConnectionAvailability(
      initialRuntimeState,
      "neo",
      false,
    );

    expect(unavailable.neoNotFound).toBe(true);
    expect(unavailable.vrcNotFound).toBe(false);
    expect(
      applyConnectionAvailability(unavailable, "vrchat", false),
    ).toMatchObject({ neoNotFound: true, vrcNotFound: true });
  });

  it("clamps invalid negative input levels without changing runtime status", () => {
    expect(applyInputLevel(initialRuntimeState, -1, 0.25)).toMatchObject({
      status: "idle",
      inputLevelBeforeGain: 0,
      inputLevel: 0.25,
    });
  });

  it("retains source-scoped input levels while preserving the legacy global level", () => {
    const updated = applyInputLevel(initialRuntimeState, 0.4, 0.8, "profile-a");

    expect(updated.inputLevelsBySource).toEqual({
      "profile-a": { inputLevel: 0.8, inputLevelBeforeGain: 0.4 },
    });
    expect(updated.inputLevel).toBe(0.8);
    expect(updated.inputLevelBeforeGain).toBe(0.4);
  });

  it("keeps the global fallback when a legacy input-level event has no source", () => {
    const updated = applyInputLevel(initialRuntimeState, 0.2, 0.3);

    expect(updated.inputLevelsBySource).toEqual({});
    expect(updated.inputLevel).toBe(0.3);
    expect(updated.inputLevelBeforeGain).toBe(0.2);
  });

  it("clears source-scoped levels when a new recognition session starts", () => {
    const withLevel = applyInputLevel(
      initialRuntimeState,
      0.4,
      0.8,
      "profile-a",
    );

    expect(
      applyRecognitionStatus(withLevel, "waiting_for_client"),
    ).toMatchObject({
      inputLevel: 0,
      inputLevelBeforeGain: 0,
      inputLevelsBySource: {},
    });
  });
});
