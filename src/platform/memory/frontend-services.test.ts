import { describe, expect, it } from "vitest";

import {
  createMemoryFrontendServices,
  memoryCapabilities,
} from "./frontend-services";
import type { FrontendEvent } from "../../application/frontend-services";

describe("web preview frontend services", () => {
  it("replays recognition through the same event boundary without claiming desktop capabilities", async () => {
    const services = createMemoryFrontendServices();
    const received: FrontendEvent[] = [];
    const unsubscribe = await services.events.subscribe((event) =>
      received.push(event),
    );

    expect(memoryCapabilities).toMatchObject({
      localAudioDevices: false,
      modelManagement: false,
      systemAudioPermission: false,
      externalConnectionProbe: false,
      fileExport: false,
      recognitionControl: true,
    });

    expect(await services.recognition.start()).toBe("listening");
    expect(received.map((event) => event.type)).toEqual([
      "recognitionStatusChanged",
      "recognizedTextReceived",
      "recognizedTextReceived",
      "recognizedTextReceived",
      "recognizedTextReceived",
      "applicationError",
    ]);

    unsubscribe();
    await services.recognition.stop();
    expect(received.at(-1)?.type).toBe("applicationError");
  });

  it("fails unsupported desktop operations instead of returning a successful no-op", async () => {
    const services = createMemoryFrontendServices();

    await expect(services.connections.findNeoPort()).rejects.toThrow(
      "external connection probe is not available",
    );
    await expect(
      services.system.saveRecognitionCsv({
        defaultFileName: "preview.csv",
        content: "text",
      }),
    ).rejects.toThrow("file export is not available");
  });
});
