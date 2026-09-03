import { describe, expect, it } from "vitest";

import {
  mergeSuggestedHotwordReadings,
  hasInvalidHotwords,
  normalizeHotwordReading,
  sanitizeHotwords,
  validateHotwords,
} from "./asr-hotwords";
import type { AsrHotword } from "./types";
import { ja } from "../i18n/locales/ja";

const hotword = (surface: string, readings: string[]): AsrHotword => ({
  surface,
  readings,
  score: null,
});

describe("ASR hotword editor validation", () => {
  it("uses the hiragana and priority multiplier guidance without a duplicate footer", () => {
    expect(ja.settings.hotwords.description).toBe(
      "高精度モードで優先したい表記と読みを登録します。\n" +
        "読みは1行に1つひらがなで入力してください。\n" +
        "優先倍率は未指定でx100、x1は中立、1未満は抑制です。",
    );
    expect(ja.settings.hotwords.scoreColumn).toBe("優先倍率");
    expect(ja.settings.hotwords.score).toBe("優先倍率（任意）");
    expect("scoreDescription" in ja.settings.hotwords).toBe(false);
  });

  it("appends autofill suggestions as hiragana without replacing existing readings", () => {
    expect(
      mergeSuggestedHotwordReadings(
        ["あまぞん", "てにゅうりょく"],
        ["アマゾン", "アマゾーン"],
      ),
    ).toEqual(["あまぞん", "てにゅうりょく", "あまぞーん"]);
  });

  it("treats hiragana, katakana, and half-width katakana as the same reading", () => {
    expect(normalizeHotwordReading("ｻｲﾄｳ")).toBe("さいとう");
    expect(normalizeHotwordReading("サイトウ")).toBe("さいとう");
    expect(normalizeHotwordReading("さいとう")).toBe("さいとう");
  });

  it("reports collisions across surfaces and normalized readings", () => {
    const result = validateHotwords([
      hotword("斎藤", ["さいとう"]),
      hotword("斎藤", ["ｻｲﾄｳ"]),
    ]);

    expect(result).toEqual([
      {
        emptySurface: false,
        blankReading: false,
        duplicateSurface: true,
        duplicateReading: true,
        pathCollision: false,
        terminalPrefix: false,
        invalidScore: false,
      },
      {
        emptySurface: false,
        blankReading: false,
        duplicateSurface: true,
        duplicateReading: true,
        pathCollision: false,
        terminalPrefix: false,
        invalidScore: false,
      },
    ]);
    expect(
      hasInvalidHotwords([
        hotword("斎藤", ["さいとう"]),
        hotword("斎藤", ["ｻｲﾄｳ"]),
      ]),
    ).toBe(true);
  });

  it("allows a hotword without a reading and saves kana variants as hiragana", () => {
    const saved = sanitizeHotwords([
      { surface: "  Parapper ", readings: [], score: null },
      { surface: " 斎藤", readings: [" ｻｲﾄｳ ", ""], score: 1.25 },
    ]);

    expect(saved).toEqual([
      { surface: "Parapper", readings: [], score: null },
      { surface: "斎藤", readings: ["さいとう"], score: 1.25 },
    ]);
    expect(hasInvalidHotwords([hotword("Parapper", [])])).toBe(false);
  });

  it("reports a reading that collides with another entry's surface path", () => {
    const result = validateHotwords([
      hotword("サイトウ", []),
      hotword("斎藤", ["さいとう"]),
    ]);

    expect(result[1].duplicateReading).toBe(true);
    expect(result[0].pathCollision).toBe(true);
    expect(result[1].pathCollision).toBe(true);
    expect(
      hasInvalidHotwords([
        { surface: "斎藤", readings: ["さいとう"], score: 0 },
      ]),
    ).toBe(true);
  });

  it("rejects terminal prefixes because non-strict decoding cannot reach the longer path", () => {
    const entries = [
      hotword("東京", []),
      hotword("東京都", []),
      hotword("Parapper", ["とうきょう", "とうきょうと"]),
    ];
    const result = validateHotwords(entries);

    expect(result.map(({ terminalPrefix }) => terminalPrefix)).toEqual([
      true,
      true,
      true,
    ]);
    expect(
      hasInvalidHotwords([hotword("東京", []), hotword("東京都", [])]),
    ).toBe(true);
    expect(
      validateHotwords(entries, { checkTerminalPrefixes: false }).map(
        ({ terminalPrefix }) => terminalPrefix,
      ),
    ).toEqual([false, false, false]);
    expect(hasInvalidHotwords(entries, { checkTerminalPrefixes: false })).toBe(
      false,
    );
  });
});
