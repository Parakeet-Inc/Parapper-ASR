export const zeroMinSize = {
  minWidth: 0,
  minHeight: 0,
} as const;

export const zeroMinWidth = {
  minWidth: 0,
} as const;

export const zeroMinHeight = {
  minHeight: 0,
} as const;

export const fullSizeZeroMin = {
  height: "100%",
  width: "100%",
  ...zeroMinSize,
} as const;
