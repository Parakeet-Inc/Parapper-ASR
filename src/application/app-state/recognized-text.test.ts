import { expect, it } from "vitest";

import { frontendPreviewEvents } from "./fixtures";
import { upsertRecognizedText } from "./recognized-text";
import { upsertTranslatedText } from "./translated-text";
import type {
  RecognizedTextEvent,
  TranslationTextEvent,
} from "../../lib/types";

const firstRecognizedEvent = () => {
  const event = frontendPreviewEvents.find(
    (
      candidate,
    ): candidate is Extract<
      (typeof frontendPreviewEvents)[number],
      { type: "recognizedTextReceived" }
    > => candidate.type === "recognizedTextReceived",
  );
  if (!event) {
    throw new Error("frontend fixture must contain a recognized text event");
  }
  return event.payload;
};

it("keeps a final transcript when a stale interim arrives later", () => {
  const recognizedTexts = frontendPreviewEvents.reduce<RecognizedTextEvent[]>(
    (texts, event) =>
      event.type === "recognizedTextReceived"
        ? upsertRecognizedText(texts, event.payload)
        : texts,
    [],
  );

  expect(recognizedTexts).toEqual([
    expect.objectContaining({
      id: "turn-1-final",
      text: "音声認識です。",
      is_final: true,
    }),
  ]);
});

it("keeps simultaneous turns from different sources as separate rows", () => {
  const sourceA = firstRecognizedEvent();
  const sourceB = {
    ...sourceA,
    id: "turn-2-interim-1",
    text: "別入力",
    source: {
      ...sourceA.source,
      identity: { ...sourceA.source.identity, source_id: "channel-2" },
    },
  };

  const rows = upsertRecognizedText(upsertRecognizedText([], sourceA), sourceB);

  expect(rows).toHaveLength(2);
  expect(rows.map((row) => row.text)).toEqual(["音声", "別入力"]);
});

it("keeps translated rows separate when source ids differ", () => {
  const recognized = firstRecognizedEvent();
  const translated = (source: RecognizedTextEvent["source"], text: string) =>
    ({
      id: `translation-${text}`,
      source_recognition_id: recognized.id,
      source,
      source_asr_model: recognized.source_asr_model,
      source_text: recognized.text,
      source_detected_language: recognized.detected_language,
      target_lang: "en",
      translated_text: text,
      is_final: false,
      update_mode: "replace",
      translated_at_millis: 1,
      elapsed_millis: 1,
      status: "success",
      error: null,
    }) satisfies TranslationTextEvent;

  const sourceB = {
    ...recognized.source,
    identity: { ...recognized.source.identity, source_id: "channel-2" },
  };
  const rows = upsertTranslatedText(
    [translated(recognized.source, "source-a")],
    translated(sourceB, "source-b"),
  );

  expect(rows).toHaveLength(2);
  expect(rows.map((row) => row.translated_text)).toEqual([
    "source-a",
    "source-b",
  ]);
});
