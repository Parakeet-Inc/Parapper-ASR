import type {
  RecognitionSourceMeta,
  SttProfileConfig,
  SttProfileDisplayColor,
} from "./types";

type RecognitionLogProfile = Pick<SttProfileConfig, "id" | "display_color">;

const DEFAULT_SOURCE_COLOR: SttProfileDisplayColor = "green";

/**
 * Resolves a current profile by immutable source id. Historical, deleted, and
 * legacy sources keep a stable green fallback.
 */
export const resolveRecognitionLogSourceColor = (
  profiles: readonly RecognitionLogProfile[],
  sourceId: string,
): SttProfileDisplayColor =>
  profiles.find((candidate) => candidate.id === sourceId)?.display_color ??
  DEFAULT_SOURCE_COLOR;

export const resolveRecognitionLogSourceMetaColor = (
  profiles: readonly RecognitionLogProfile[],
  source: RecognitionSourceMeta,
): SttProfileDisplayColor =>
  resolveRecognitionLogSourceColor(profiles, source.identity.source_id);
