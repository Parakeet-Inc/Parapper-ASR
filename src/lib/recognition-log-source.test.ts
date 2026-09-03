import { describe, expect, it } from "vitest";

import {
  resolveRecognitionLogSourceColor,
  resolveRecognitionLogSourceMetaColor,
} from "./recognition-log-source";
import type { RecognitionSourceMeta } from "./types";

const sourceMeta = (sourceId: string): RecognitionSourceMeta => ({
  identity: {
    source_id: sourceId,
    speaker_label: "not-used-for-color",
    capture_endpoint_id: "capture-1",
    channel_index: 0,
  },
  turn_session_id: 1,
  turn_id: 2,
  turn_revision: 0,
  output_sequence: 0,
  segment_id: 3,
  previous_segment_id: null,
});

describe("resolveRecognitionLogSourceColor", () => {
  it("uses the persisted color of the profile matching the source id", () => {
    expect(
      resolveRecognitionLogSourceColor(
        [
          {
            id: "mic-channel-2",
            display_color: "blue",
          },
        ],
        "mic-channel-2",
      ),
    ).toBe("blue");
  });

  it("uses green for a historical source after its profile is deleted", () => {
    expect(resolveRecognitionLogSourceColor([], "deleted-channel")).toBe(
      "green",
    );
  });

  it("uses green for the legacy single source", () => {
    expect(resolveRecognitionLogSourceColor([], "legacy-single-source")).toBe(
      "green",
    );
  });

  it("resolves translation rows from their source metadata with historical fallback", () => {
    const profiles = [{ id: "mic-channel-2", display_color: "blue" as const }];

    expect(
      ["mic-channel-2", "deleted-channel", "legacy-single-source"].map(
        (sourceId) =>
          resolveRecognitionLogSourceMetaColor(profiles, sourceMeta(sourceId)),
      ),
    ).toEqual(["blue", "green", "green"]);
  });
});
