import type {
  AsrMissingEvent,
  AudioDeviceInfo,
  ConfigPreset,
  ConnectionStateEvent,
  InputLevelEvent,
  LocalTranslationModel,
  ModelDownloadProgress,
  ModelStatus,
  OscMuteStateEvent,
  ParapperConfig,
  ParapperErrorPayload,
  RecognizedTextEvent,
  RecognitionStatus,
  SpeechRequestEvent,
  TranslationHttpListenerStatus,
  TranslationTextEvent,
  VadStateEvent,
} from "../lib/types";

export type Unsubscribe = () => void;

export type FrontendEvent =
  | { type: "recognitionStatusChanged"; payload: RecognitionStatus }
  | { type: "inputLevelChanged"; payload: InputLevelEvent | number }
  | { type: "vadStateChanged"; payload: VadStateEvent }
  | { type: "recognizedTextReceived"; payload: RecognizedTextEvent }
  | { type: "translationTextReceived"; payload: TranslationTextEvent }
  | { type: "speechRequestReceived"; payload: SpeechRequestEvent }
  | { type: "asrMissing"; payload: AsrMissingEvent }
  | { type: "oscMuteStateChanged"; payload: OscMuteStateEvent }
  | { type: "connectionStateChanged"; payload: ConnectionStateEvent }
  | { type: "modelDownloadProgressed"; payload: ModelDownloadProgress }
  | { type: "applicationError"; payload: ParapperErrorPayload };

export interface ConfigService {
  load(): Promise<ParapperConfig>;
  save(config: ParapperConfig): Promise<ParapperConfig>;
  reset(): Promise<ParapperConfig>;
}

export interface RecognitionControlService {
  status(): Promise<RecognitionStatus>;
  start(): Promise<RecognitionStatus>;
  stop(): Promise<RecognitionStatus>;
}

export interface FrontendEventService {
  subscribe(listener: (event: FrontendEvent) => void): Promise<Unsubscribe>;
}

export interface ModelService {
  status(): Promise<ModelStatus>;
  hasAnyInstalled(): Promise<boolean>;
  download(config: ParapperConfig): Promise<ModelStatus>;
  downloadLocalTranslation(model: LocalTranslationModel): Promise<boolean>;
  isLocalTranslationInstalled(model: LocalTranslationModel): Promise<boolean>;
}

export interface AudioDeviceService {
  inputDevices(): Promise<AudioDeviceInfo[]>;
  outputDevices(): Promise<AudioDeviceInfo[]>;
  requestLoopbackPermission(): Promise<boolean>;
  openLoopbackPermissionSettings(): Promise<void>;
}

export interface PresetService {
  list(): Promise<ConfigPreset[]>;
  save(name: string, config: ParapperConfig): Promise<ConfigPreset[]>;
  delete(name: string): Promise<ConfigPreset[]>;
}

export interface ConnectionService {
  checkNeo(enabled: boolean, port: number): Promise<boolean>;
  checkVrchat(enabled: boolean): Promise<boolean>;
  findNeoPort(): Promise<number | null>;
  findYncPluginPort(): Promise<number | null>;
  fetchNeoVoices(port: number): Promise<string[]>;
}

export interface SpeechControlService {
  stop(port: number): Promise<void>;
}

export interface HotwordReadingService {
  suggest(surface: string): Promise<string[]>;
}

export interface TranslationServerService {
  status(): Promise<TranslationHttpListenerStatus>;
  start(
    port: number,
    model: LocalTranslationModel,
  ): Promise<TranslationHttpListenerStatus>;
  stop(): Promise<TranslationHttpListenerStatus>;
}

export type SaveTextFileRequest = {
  defaultFileName: string;
  content: string;
};

export type SaveBinaryFileRequest = {
  defaultFileName: string;
  content: number[];
};

export type RustLicensesDocument = {
  licenses: {
    name: string;
    text: string;
    used_by: { crate: { name: string; version: string } }[];
  }[];
};

export interface SystemIntegrationService {
  saveRecognitionCsv(request: SaveTextFileRequest): Promise<string | null>;
  saveAsrInputWav(request: SaveBinaryFileRequest): Promise<string | null>;
  openExternalUrl(url: string): Promise<void>;
  playAudio(samples: number[], sampleRate: number): Promise<void>;
  loadRustLicenses(): Promise<RustLicensesDocument>;
}

export type FrontendServices = {
  config: ConfigService;
  recognition: RecognitionControlService;
  events: FrontendEventService;
  models: ModelService;
  audioDevices: AudioDeviceService;
  presets: PresetService;
  connections: ConnectionService;
  speech: SpeechControlService;
  hotwordReadings: HotwordReadingService;
  translationServer: TranslationServerService;
  system: SystemIntegrationService;
};

export type FrontendCapabilities = {
  localAudioDevices: boolean;
  outputAudioDevices: boolean;
  modelManagement: boolean;
  systemAudioPermission: boolean;
  externalConnectionProbe: boolean;
  fileExport: boolean;
  recognitionControl: boolean;
  speechControl: boolean;
  localTranslationServer: boolean;
};
