import type { RecognitionSourceMeta } from "./types";

export const sameRecognitionTurn = (
  left: Pick<RecognitionSourceMeta, "turn_session_id" | "turn_id" | "identity">,
  right: Pick<
    RecognitionSourceMeta,
    "turn_session_id" | "turn_id" | "identity"
  >,
) =>
  left.turn_session_id === right.turn_session_id &&
  left.turn_id === right.turn_id &&
  left.identity.source_id === right.identity.source_id;

export const recognitionSourceRowId = (source: RecognitionSourceMeta) =>
  `turn-${source.turn_session_id}-${source.identity.source_id}-${source.turn_id}`;
