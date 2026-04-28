import { createTheme } from "@mantine/core";

export const theme = createTheme({
  colors: {
    primary: [
      "#eafcf6",
      "#dbf3ec",
      "#b8e6d8",
      "#90d8c2",
      "#70ccb0",
      "#5bc4a4",
      "#4fc19e",
      "#3eaa89",
      "#329779",
      "#1e8367",
    ],
  },
  primaryColor: "primary",
  primaryShade: 8,
});

export const notificationColor = {
  primary: "#329779",
  info: "#4a82c0",
  ok: "#329779",
  warn: "#F2994A",
  error: "#EB5757",
} as const;
