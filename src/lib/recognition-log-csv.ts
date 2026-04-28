import type { RecognizedTextEvent } from "./types";

const writeAscii = (view: DataView, offset: number, text: string) => {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
};

export const formatLogTime = (millis: number, locale: string) =>
  new Date(millis).toLocaleTimeString(locale, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

export const formatCsvDateTime = (millis: number) => {
  const date = new Date(millis);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(
    date.getDate(),
  )} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(
    date.getSeconds(),
  )}`;
};

export const formatCsvFileTimestamp = () => {
  const date = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(
    date.getDate(),
  )}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
};

export const escapeCsvCell = (value: string | number) => {
  const text = String(value);
  if (!/[",\r\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
};

type RecognitionCsvHeaders = {
  text: string;
  time: string;
  seconds: string;
  elapsedMs: string;
};

export const buildRecognitionCsvExport = (
  entries: RecognizedTextEvent[],
  headers: RecognitionCsvHeaders,
) => {
  const rows = [
    [headers.text, headers.time, headers.seconds, headers.elapsedMs],
    ...entries.map((entry) => [
      entry.text,
      formatCsvDateTime(entry.recognized_at_millis),
      entry.audio_seconds.toFixed(3),
      entry.elapsed_millis,
    ]),
  ];
  const content = rows
    .map((row) => row.map((cell) => escapeCsvCell(cell)).join(","))
    .join("\r\n");

  return {
    defaultFileName: `parapper-recognition-log-${formatCsvFileTimestamp()}.csv`,
    content: `\uFEFF${content}`,
  };
};

export const float32SamplesToWavBytes = (
  samples: number[],
  sampleRate: number,
) => {
  const bytesPerSample = 2;
  const headerBytes = 44;
  const dataBytes = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(headerBytes + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  samples.forEach((sample, index) => {
    const clamped = Math.max(-1, Math.min(1, sample));
    const pcm = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(headerBytes + index * bytesPerSample, pcm, true);
  });

  return new Uint8Array(buffer);
};
