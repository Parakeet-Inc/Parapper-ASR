import { describe, expect, it } from "vitest";

import viteConfig from "../vite.config";

describe("Vite development server", () => {
  it("does not watch Rust and frontend build outputs while Tauri is running", () => {
    expect(viteConfig.server?.watch?.ignored).toEqual(
      expect.arrayContaining([
        "**/target",
        "**/target/**",
        "**/dist",
        "**/dist/**",
        "**/dist-web-preview",
        "**/dist-web-preview/**",
        "**/.uv-cache",
        "**/.uv-cache/**",
        "**/onnx-optimize",
        "**/onnx-optimize/**",
        "**/work",
        "**/work/**",
      ]),
    );
  });
});
