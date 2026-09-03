import { describe, expect, it, vi } from "vitest";

import { createTauriFrontendServices } from "./frontend-services";
import type { FrontendEvent } from "../../application/frontend-services";
import type { ParapperConfig } from "../../lib/types";

describe("Tauri frontend services", () => {
  it("keeps command names, payloads, and event channels inside the platform adapter", async () => {
    const calls: { command: string; args?: Record<string, unknown> }[] = [];
    const handlers = new Map<string, (event: { payload: unknown }) => void>();
    const unsubscribe = vi.fn();
    const config = { neo_http_port: 15520 } as ParapperConfig;
    const results: Record<string, unknown> = {
      save_config: config,
      check_neo_http_available: true,
      start_translation_http_listener: {
        state: "running",
        port: 18081,
        error: null,
      },
      neo_speech_stop: undefined,
      suggest_hotword_readings: ["パラッパー"],
    };
    const services = createTauriFrontendServices({
      invoke: async <T>(command: string, args?: Record<string, unknown>) => {
        calls.push({ command, args });
        return results[command] as T;
      },
      listen: async <T>(
        channel: string,
        handler: (event: { payload: T }) => void,
      ) => {
        handlers.set(channel, handler as (event: { payload: unknown }) => void);
        return unsubscribe;
      },
    });

    await services.config.save(config);
    await services.connections.checkNeo(true, 15520);
    await services.translationServer.start(18081, "lfm2_q4");
    await services.speech.stop(15520);
    expect(await services.hotwordReadings.suggest("Parapper")).toEqual([
      "パラッパー",
    ]);

    expect(calls).toEqual([
      { command: "save_config", args: { config } },
      {
        command: "check_neo_http_available",
        args: { neoHttpEnabled: true, neoHttpPort: 15520 },
      },
      {
        command: "start_translation_http_listener",
        args: { port: 18081, localModel: "lfm2_q4" },
      },
      { command: "neo_speech_stop", args: { port: 15520 } },
      {
        command: "suggest_hotword_readings",
        args: { surface: "Parapper" },
      },
    ]);

    const received: FrontendEvent[] = [];
    const stop = await services.events.subscribe((event) =>
      received.push(event),
    );
    expect([...handlers.keys()]).toEqual([
      "parapper://status",
      "parapper://input-level",
      "parapper://vad-state",
      "parapper://recognized-text",
      "parapper://translated-text",
      "parapper://speech-request",
      "parapper://asr-missing",
      "parapper://osc-mute-state",
      "parapper://connection-state",
      "parapper://model-download-progress",
      "parapper://error",
    ]);

    handlers.get("parapper://status")?.({ payload: "listening" });
    expect(received).toEqual([
      { type: "recognitionStatusChanged", payload: "listening" },
    ]);

    stop();
    expect(unsubscribe).toHaveBeenCalledTimes(11);
  });
});
