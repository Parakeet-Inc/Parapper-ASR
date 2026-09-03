import { sameRecognitionTurn } from "../../lib/recognition-source";
import type { RecognizedTextEvent } from "../../lib/types";

export const upsertRecognizedText = (
  texts: RecognizedTextEvent[],
  event: RecognizedTextEvent,
) => {
  if (event.update_mode !== "replace") {
    return [...texts, event];
  }

  const index = texts.findIndex((text) =>
    sameRecognitionTurn(text.source, event.source),
  );
  if (index < 0) {
    return [...texts, event];
  }

  const current = texts[index];
  if (!shouldReplaceRecognitionEvent(current, event)) {
    return texts;
  }

  return texts.map((text, currentIndex) =>
    currentIndex === index ? event : text,
  );
};

const shouldReplaceRecognitionEvent = (
  current: Pick<RecognizedTextEvent, "source" | "is_final">,
  incoming: Pick<RecognizedTextEvent, "source" | "is_final">,
) => {
  if (current.is_final && !incoming.is_final) {
    return false;
  }
  if (incoming.source.turn_revision !== current.source.turn_revision) {
    return incoming.source.turn_revision > current.source.turn_revision;
  }
  if (incoming.source.output_sequence !== current.source.output_sequence) {
    return incoming.source.output_sequence > current.source.output_sequence;
  }
  return incoming.is_final || !current.is_final;
};
