import { MantineProvider } from "@mantine/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  buildTranslationLogRows,
  TranslationLogSourceCard,
} from "./translation-side-panel";
import type {
  ParapperConfig,
  RecognitionSourceMeta,
  RecognizedTextEvent,
  SttProfileConfig,
  TranslationTextEvent,
} from "../lib/types";

const sourceMeta = (sourceId: string): RecognitionSourceMeta => ({
  identity: {
    source_id: sourceId,
    speaker_label: `speaker-${sourceId}`,
    capture_endpoint_id: "capture-1",
    channel_index: 0,
  },
  turn_session_id: 1,
  turn_id: 2,
  turn_revision: 0,
  output_sequence: 0,
  segment_id: 3,
  previous_segment_id: null,
});

const recognized = (sourceId: string): RecognizedTextEvent => ({
  id: `recognition-${sourceId}`,
  source: sourceMeta(sourceId),
  is_final: true,
  update_mode: "replace",
  text: "こんにちは",
  source_asr_model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
  source_language: "japanese",
  detected_language: "ja",
  recognized_at_millis: 1,
  audio_seconds: 1,
  elapsed_millis: 1,
  audio_frames: 16_000,
  debug_asr_audio_sample_rate: null,
  debug_asr_audio_samples: null,
});

const translated = (sourceId: string): TranslationTextEvent => ({
  id: `translation-${sourceId}`,
  source_recognition_id: `recognition-${sourceId}`,
  source: sourceMeta(sourceId),
  source_asr_model: "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
  source_text: "こんにちは",
  source_detected_language: "ja",
  target_lang: "en",
  translated_text: "hello",
  is_final: true,
  update_mode: "replace",
  translated_at_millis: 2,
  elapsed_millis: 1,
  status: "success",
  error: null,
});

const translationConfig = {
  translation_enabled: true,
  translation_send_timing: "interim",
  translation_mappings: [
    {
      id: "ja-en",
      source_asr_model: null,
      backend: "local",
      local_model: "lfm2_q4",
      source_lang: "ja",
      target_lang: "en",
    },
  ],
} as ParapperConfig;

describe("translation source marker", () => {
  it("renders the persisted source color without rendering the profile name", () => {
    const markup = renderToStaticMarkup(
      createElement(
        MantineProvider,
        null,
        createElement(
          TranslationLogSourceCard,
          {
            rowId: "row-1",
            source: sourceMeta("mic-channel-2"),
            profiles: [
              {
                id: "mic-channel-2",
                name: "MUST_NOT_RENDER_PROFILE_NAME",
                display_color: "blue",
              } as SttProfileConfig,
            ],
          },
          createElement("span", null, "translated text"),
        ),
      ),
    );

    expect(markup).toContain("data-translation-source-color");
    expect(markup).toContain("background-color:var(--mantine-color-blue-6)");
    expect(markup).not.toContain("MUST_NOT_RENDER_PROFILE_NAME");
  });

  it("keeps source metadata on pending, ready, and orphan translation rows", () => {
    const pendingRows = buildTranslationLogRows(
      translationConfig,
      [recognized("pending-source")],
      [],
    );
    const readyRows = buildTranslationLogRows(
      translationConfig,
      [recognized("ready-source")],
      [translated("ready-source")],
    );
    const orphanRows = buildTranslationLogRows(
      translationConfig,
      [],
      [translated("orphan-source")],
    );

    expect(
      [pendingRows, readyRows, orphanRows].map(
        ([row]) => row.source.identity.source_id,
      ),
    ).toEqual(["pending-source", "ready-source", "orphan-source"]);
    expect(
      [pendingRows, readyRows, orphanRows].map(([row]) => row.entries[0].kind),
    ).toEqual(["pending", "ready", "ready"]);
  });
});
