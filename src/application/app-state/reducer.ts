import type {
  ModelDownloadProgress,
  ModelStatus,
  ParapperErrorPayload,
  RecognitionStatus,
  VadStateEvent,
} from "../../lib/types";

export type RuntimeState = {
  status: RecognitionStatus;
  running: boolean;
  starting: boolean;
  inputLevel: number;
  inputLevelBeforeGain: number;
  inputLevelsBySource: Record<string, InputLevelSnapshot>;
  vadState: VadStateEvent | null;
  asrWarning: string | null;
  lastError: ParapperErrorPayload | null;
  oscMuted: boolean | null;
  neoNotFound: boolean;
  vrcNotFound: boolean;
  translationSpeechDelaySuspected: boolean;
};

export type InputLevelSnapshot = {
  inputLevel: number;
  inputLevelBeforeGain: number;
};

export type ModelState = {
  status: ModelStatus | null;
  downloading: boolean;
  progress: ModelDownloadProgress | null;
};

export const initialRuntimeState: RuntimeState = {
  status: "idle",
  running: false,
  starting: false,
  inputLevel: 0,
  inputLevelBeforeGain: 0,
  inputLevelsBySource: {},
  vadState: null,
  asrWarning: null,
  lastError: null,
  oscMuted: null,
  neoNotFound: false,
  vrcNotFound: false,
  translationSpeechDelaySuspected: false,
};

export const initialModelState: ModelState = {
  status: null,
  downloading: false,
  progress: null,
};

export const recognitionIsRunning = (status: RecognitionStatus) =>
  status === "waiting_for_client" ||
  status === "listening" ||
  status === "draining";

export const applyRecognitionStatus = (
  state: RuntimeState,
  status: RecognitionStatus,
): RuntimeState => {
  const resetInputLevels =
    status === "idle" ||
    status === "waiting_for_client" ||
    status === "stopped" ||
    status === "error";
  return {
    ...state,
    status,
    running: recognitionIsRunning(status),
    starting: false,
    ...(resetInputLevels
      ? {
          inputLevel: 0,
          inputLevelBeforeGain: 0,
          inputLevelsBySource: {},
        }
      : null),
  };
};

export const applyConnectionAvailability = (
  state: RuntimeState,
  target: "neo" | "vrchat",
  found: boolean,
): RuntimeState => ({
  ...state,
  neoNotFound: target === "neo" ? !found : state.neoNotFound,
  vrcNotFound: target === "vrchat" ? !found : state.vrcNotFound,
});

export const applyInputLevel = (
  state: RuntimeState,
  preGain: number,
  postGain: number,
  sourceId?: string | null,
): RuntimeState => ({
  ...state,
  inputLevel: Math.max(0, postGain),
  inputLevelBeforeGain: Math.max(0, preGain),
  inputLevelsBySource: sourceId
    ? {
        ...state.inputLevelsBySource,
        [sourceId]: {
          inputLevel: Math.max(0, postGain),
          inputLevelBeforeGain: Math.max(0, preGain),
        },
      }
    : state.inputLevelsBySource,
});

export const applyModelStatus = (
  state: ModelState,
  status: ModelStatus,
): ModelState => ({ ...state, status });

export const applyModelDownloadProgress = (
  state: ModelState,
  progress: ModelDownloadProgress,
): ModelState => ({ ...state, progress });

export const setModelDownloading = (
  state: ModelState,
  downloading: boolean,
): ModelState => ({
  ...state,
  downloading,
  progress: downloading ? null : state.progress,
});
