import type { AsrHotword } from "./types";

export type HotwordValidation = {
  emptySurface: boolean;
  blankReading: boolean;
  duplicateSurface: boolean;
  duplicateReading: boolean;
  pathCollision: boolean;
  terminalPrefix: boolean;
  invalidScore: boolean;
};

type HotwordValidationOptions = {
  checkTerminalPrefixes?: boolean;
};

/**
 * Normalize a pronunciation to the canonical hiragana form used by the
 * editor and persisted settings.
 */
export const normalizeHotwordReading = (value: string) => {
  const normalized = value.trim().normalize("NFKC");
  return Array.from(normalized)
    .map((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      // Katakana (including ヵ/ヶ) and hiragana share a fixed Unicode offset.
      if (codePoint >= 0x30a1 && codePoint <= 0x30f6) {
        return String.fromCodePoint(codePoint - 0x60);
      }
      return character;
    })
    .join("");
};

export const splitHotwordReadings = (value: string) =>
  value
    .split(/\r?\n/)
    .map((reading) => reading.trim())
    .filter((reading) => reading.length > 0);

export const mergeSuggestedHotwordReadings = (
  current: string[],
  suggested: string[],
) => {
  const merged = [...current];
  const normalized = new Set(current.map(normalizeHotwordReading));
  for (const reading of suggested
    .map((value) => value.trim())
    .filter(Boolean)) {
    const key = normalizeHotwordReading(reading);
    if (!normalized.has(key)) {
      normalized.add(key);
      merged.push(key);
    }
  }
  return merged;
};

const tokenPath = (value: string) =>
  Array.from(value)
    .filter((character) => !/\s/u.test(character))
    .join("");

const toKatakana = (value: string) =>
  Array.from(value)
    .map((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return codePoint >= 0x3041 && codePoint <= 0x3096
        ? String.fromCodePoint(codePoint + 0x60)
        : character;
    })
    .join("");

const hotwordTokenPaths = (hotword: AsrHotword) => {
  const paths = [tokenPath(hotword.surface.trim())];
  for (const reading of hotword.readings) {
    const hiragana = tokenPath(normalizeHotwordReading(reading));
    paths.push(hiragana);
    const katakana = toKatakana(hiragana);
    if (katakana !== hiragana) paths.push(katakana);
  }
  return [...new Set(paths.filter(Boolean))];
};

export const validateHotwords = (
  hotwords: AsrHotword[],
  { checkTerminalPrefixes = true }: HotwordValidationOptions = {},
): HotwordValidation[] => {
  const surfaceCounts = new Map<string, number>();
  const readingCounts = new Map<string, number>();
  const surfacePathOwners = new Map<string, Set<string>>();
  const readingPathOwners = new Map<string, Set<string>>();
  const pathsByRow = hotwords.map(hotwordTokenPaths);
  const terminalPrefixRows = new Set<number>();
  for (const hotword of hotwords) {
    const surface = hotword.surface.trim();
    surfaceCounts.set(surface, (surfaceCounts.get(surface) ?? 0) + 1);
    const surfacePath = normalizeHotwordReading(surface);
    const surfaceOwners =
      surfacePathOwners.get(surfacePath) ?? new Set<string>();
    surfaceOwners.add(surface);
    surfacePathOwners.set(surfacePath, surfaceOwners);
    for (const reading of hotword.readings) {
      const normalized = normalizeHotwordReading(reading);
      if (normalized) {
        readingCounts.set(normalized, (readingCounts.get(normalized) ?? 0) + 1);
        const readingOwners =
          readingPathOwners.get(normalized) ?? new Set<string>();
        readingOwners.add(surface);
        readingPathOwners.set(normalized, readingOwners);
      }
    }
  }

  if (checkTerminalPrefixes) {
    for (let left = 0; left < pathsByRow.length; left += 1) {
      for (let right = left; right < pathsByRow.length; right += 1) {
        for (const leftPath of pathsByRow[left]) {
          for (const rightPath of pathsByRow[right]) {
            if (leftPath === rightPath) continue;
            if (
              leftPath.startsWith(rightPath) ||
              rightPath.startsWith(leftPath)
            ) {
              terminalPrefixRows.add(left);
              terminalPrefixRows.add(right);
            }
          }
        }
      }
    }
  }

  return hotwords.map((hotword, index) => {
    const readings = hotword.readings.map((reading) => reading.trim());
    const normalizedReadings = readings
      .map(normalizeHotwordReading)
      .filter((reading) => reading.length > 0);
    const hasDuplicateReadingInRow =
      new Set(normalizedReadings).size !== normalizedReadings.length;
    const hasCrossSurfacePathCollision = normalizedReadings.some((reading) => {
      const surfaceOwners = surfacePathOwners.get(reading) ?? new Set<string>();
      const readingOwners = readingPathOwners.get(reading) ?? new Set<string>();
      return [...surfaceOwners, ...readingOwners].some(
        (owner) => owner !== hotword.surface.trim(),
      );
    });
    const normalizedSurface = normalizeHotwordReading(hotword.surface.trim());
    const hasSurfacePathCollision = [
      ...(readingPathOwners.get(normalizedSurface) ?? new Set<string>()),
    ].some((owner) => owner !== hotword.surface.trim());
    return {
      emptySurface: hotword.surface.trim().length === 0,
      blankReading: readings.some((reading) => reading.length === 0),
      duplicateSurface:
        surfaceCounts.get(hotword.surface.trim()) !== undefined &&
        surfaceCounts.get(hotword.surface.trim())! > 1,
      duplicateReading:
        hasDuplicateReadingInRow ||
        normalizedReadings.some(
          (reading) => (readingCounts.get(reading) ?? 0) > 1,
        ) ||
        hasCrossSurfacePathCollision,
      pathCollision: hasSurfacePathCollision || hasCrossSurfacePathCollision,
      terminalPrefix: terminalPrefixRows.has(index),
      invalidScore:
        hotword.score !== null &&
        (!Number.isFinite(hotword.score) || hotword.score <= 0),
    };
  });
};

export const hasInvalidHotwords = (
  hotwords: AsrHotword[],
  options: HotwordValidationOptions = {},
) =>
  validateHotwords(hotwords, options).some(
    ({
      emptySurface,
      blankReading,
      duplicateSurface,
      duplicateReading,
      pathCollision,
      terminalPrefix,
      invalidScore,
    }) =>
      emptySurface ||
      blankReading ||
      duplicateSurface ||
      duplicateReading ||
      pathCollision ||
      terminalPrefix ||
      invalidScore,
  );

export const sanitizeHotwords = (hotwords: AsrHotword[]): AsrHotword[] =>
  hotwords.map((hotword) => ({
    surface: hotword.surface.trim(),
    readings: hotword.readings
      .map(normalizeHotwordReading)
      .filter((reading) => reading.length > 0),
    score:
      hotword.score !== null && Number.isFinite(hotword.score)
        ? hotword.score
        : null,
  }));
