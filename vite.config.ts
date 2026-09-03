import react from "@vitejs/plugin-react";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";

// Vite only needs to watch the frontend source and public assets. This
// repository also contains large Rust, model, cache, and work directories;
// watching them can exhaust Windows file handles before the first page loads.
const nonFrontendDirectories = [
  ".agents",
  ".claude",
  ".codex",
  ".codex-tmp",
  ".pytest_cache",
  ".ruff_cache",
  ".tmp",
  ".uv-cache",
  "artifacts",
  "crates",
  "diagnostics",
  "dist",
  "dist-web-preview",
  "documents",
  "hackathon",
  "licenses",
  "onnx-optimize",
  "parapper-mini",
  "plan",
  "review",
  "scripts",
  "spaces",
  "src-tauri",
  "target",
  "tmp",
  "tools",
  "vendor",
  "work",
] as const;

export default defineConfig({
  build: {
    target: "esnext",
  },
  server: {
    watch: {
      ignored: nonFrontendDirectories.flatMap((directory) => [
        `**/${directory}`,
        `**/${directory}/**`,
      ]),
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
