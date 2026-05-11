import { notifications } from "@mantine/notifications";

import { notificationColor } from "./theme";
import type {
  ErrorSeverity,
  ParapperErrorPayload,
  ParapperErrorType,
} from "./types";

const PARAPPER_ERROR_TYPES: readonly ParapperErrorType[] = [
  "AUDIO_INPUT",
  "RESAMPLER",
  "VAD",
  "ASR",
  "MODEL_DOWNLOAD",
  "NEO_HTTP",
  "OSC_QUERY",
  "FILE_SAVE",
  "CONFIG",
  "UNKNOWN",
];

const ERROR_MESSAGES: Record<ParapperErrorType, string> = {
  AUDIO_INPUT: "音声入力の開始または処理に失敗しました。",
  RESAMPLER: "音声のサンプルレート変換に失敗しました。",
  VAD: "音声区間の判定に失敗しました。",
  ASR: "音声認識に失敗しました。",
  MODEL_DOWNLOAD: "モデルのダウンロードに失敗しました。",
  NEO_HTTP: "ゆかコネNEOへの送信に失敗しました。",
  OSC_QUERY: "VRChat OSCQueryの状態取得に失敗しました。",
  FILE_SAVE: "ファイルの保存に失敗しました。",
  CONFIG: "設定の読み書きに失敗しました。",
  UNKNOWN: "Parapperでエラーが発生しました。",
};

const isErrorType = (value: unknown): value is ParapperErrorType =>
  typeof value === "string" &&
  PARAPPER_ERROR_TYPES.includes(value as ParapperErrorType);

export const normalizeParapperErrorPayload = (
  value: unknown,
): ParapperErrorPayload => {
  if (value && typeof value === "object") {
    const payload = value as {
      errorType?: unknown;
      severity?: unknown;
      detail?: unknown;
    };
    if (
      isErrorType(payload.errorType) &&
      (payload.severity === "warning" || payload.severity === "fatal") &&
      (payload.detail === undefined ||
        payload.detail === null ||
        typeof payload.detail === "string")
    ) {
      return {
        errorType: payload.errorType,
        severity: payload.severity,
        detail: payload.detail ?? null,
      };
    }
  }

  return {
    errorType: "UNKNOWN",
    severity: "fatal",
    detail: typeof value === "string" ? value : String(value),
  };
};

export const getParapperErrorMessage = (payload: ParapperErrorPayload) =>
  ERROR_MESSAGES[payload.errorType];

export const notifyParapperIssue = (payload: ParapperErrorPayload) => {
  const message = getParapperErrorMessage(payload);
  const warning = payload.severity === "warning";
  notifications.show({
    title: warning ? "警告" : "エラー",
    message: payload.detail ? `${message}\n${payload.detail}` : message,
    color: warning ? notificationColor.warn : notificationColor.error,
  });
  if (payload.detail) {
    console.error(payload.detail);
  }
  return message;
};

export const errorColor = (severity: ErrorSeverity) =>
  severity === "warning" ? notificationColor.warn : notificationColor.error;
