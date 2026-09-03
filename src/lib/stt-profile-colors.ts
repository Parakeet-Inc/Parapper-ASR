import type { SttProfileDisplayColor } from "./types";

/** Mantine default palette tokens at one shade for consistent source identity. */
export const STT_PROFILE_DISPLAY_COLOR_CSS: Record<
  SttProfileDisplayColor,
  string
> = {
  green: "var(--mantine-color-green-6)",
  blue: "var(--mantine-color-blue-6)",
  violet: "var(--mantine-color-violet-6)",
  red: "var(--mantine-color-red-6)",
  orange: "var(--mantine-color-orange-6)",
  yellow: "var(--mantine-color-yellow-6)",
  white: "transparent",
};
