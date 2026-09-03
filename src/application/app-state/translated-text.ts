import { sameRecognitionTurn } from "../../lib/recognition-source";
import type { TranslationTextEvent } from "../../lib/types";

export const upsertTranslatedText = (
  texts: TranslationTextEvent[],
  event: TranslationTextEvent,
) => {
  if (event.update_mode !== "replace") {
    return [...texts, event];
  }

  const index = texts.findIndex(
    (text) =>
      sameRecognitionTurn(text.source, event.source) &&
      text.target_lang === event.target_lang,
  );
  if (index < 0) {
    return [...texts, event];
  }

  const current = texts[index];
  if (current.is_final && !event.is_final) {
    return texts;
  }
  if (event.source.turn_revision < current.source.turn_revision) {
    return texts;
  }
  if (
    event.source.turn_revision === current.source.turn_revision &&
    event.source.output_sequence < current.source.output_sequence
  ) {
    return texts;
  }

  return texts.map((text, currentIndex) =>
    currentIndex === index ? event : text,
  );
};
