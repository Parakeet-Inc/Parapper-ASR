import { describe, expect, it } from "vitest";

import { configStateReducer, initialConfigState } from "./reducer";
import type { ParapperConfig } from "../../lib/types";

const config = (asrNumThreads: number) =>
  ({ asr_num_threads: asrNumThreads }) as ParapperConfig;

describe("configStateReducer", () => {
  const loaded = configStateReducer(initialConfigState, {
    type: "loaded",
    config: config(1),
  });

  it("keeps a newer optimistic edit visible while recording an older successful save", () => {
    const first = configStateReducer(loaded, {
      type: "optimisticUpdate",
      config: config(2),
      revision: 1,
    });
    const second = configStateReducer(first, {
      type: "optimisticUpdate",
      config: config(4),
      revision: 2,
    });

    expect(
      configStateReducer(second, {
        type: "saveCompleted",
        config: config(2),
        revision: 1,
      }),
    ).toMatchObject({
      current: { asr_num_threads: 4 },
      applied: { asr_num_threads: 2 },
      revision: 2,
    });
    expect(
      configStateReducer(second, {
        type: "saveCompleted",
        config: config(4),
        revision: 2,
      }),
    ).toMatchObject({
      current: { asr_num_threads: 4 },
      applied: { asr_num_threads: 4 },
      revision: 2,
    });
  });

  it("rolls back only the latest failed optimistic edit to the last persisted config", () => {
    const first = configStateReducer(loaded, {
      type: "optimisticUpdate",
      config: config(2),
      revision: 1,
    });
    const second = configStateReducer(first, {
      type: "optimisticUpdate",
      config: config(4),
      revision: 2,
    });
    const firstSaved = configStateReducer(second, {
      type: "saveCompleted",
      config: config(2),
      revision: 1,
    });

    expect(
      configStateReducer(firstSaved, {
        type: "saveFailed",
        revision: 1,
      }),
    ).toEqual(firstSaved);
    expect(
      configStateReducer(firstSaved, {
        type: "saveFailed",
        revision: 2,
      }),
    ).toMatchObject({
      current: { asr_num_threads: 2 },
      applied: { asr_num_threads: 2 },
      revision: 2,
    });
  });
});
