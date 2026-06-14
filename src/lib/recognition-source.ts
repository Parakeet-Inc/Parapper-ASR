import type { RecognitionSourceMeta } from "./types";

export const recognitionSourceRowId = (source: RecognitionSourceMeta) =>
  `turn-${source.turn_session_id}-${source.turn_id}`;
