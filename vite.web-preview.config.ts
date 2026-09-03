import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "esnext",
    outDir: "dist-web-preview",
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(__dirname, "web-preview.html"),
    },
  },
  plugins: [
    react(),
    Icons({
      compiler: "jsx",
      jsx: "react",
    }),
  ],
});
