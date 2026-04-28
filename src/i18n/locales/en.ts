export const en = {
  app: {
    loading: "Loading",
  },
  language: {
    label: "Display language",
    description:
      "Japanese and English are officially supported. Additional locale files can be added later.",
    ja: "日本語",
    en: "English",
  },
  onboarding: {
    title: "初期設定 / Initial setup",
    languageStep: "1. UIの使用言語 / UI language",
    modelStep: "2. Model",
    next: "Next",
    back: "Back",
    downloadAndClose: "Download model and close",
  },
  status: {
    vad: "VAD {{state}}",
    vrcMic: "VRC mic {{state}}",
    muted: "muted",
    open: "open",
    idle: "idle",
    ready: "ready",
    missing: "missing",
    neoNotFound: "NEO not found",
    vrcNotFound: "VRChat not found",
    asrWarning: "ASR warning",
    recognition: {
      idle: "idle",
      listening: "listening",
      stopped: "stopped",
      error: "error",
    },
    vadState: {
      speech: "speech",
      silence: "silence",
    },
  },
  tabs: {
    settings: "Settings",
    connection: "Connection",
    vad: "VAD",
    asr: "ASR",
    other: "Other",
    licenses: "Licenses",
    collapseSettings: "Collapse settings",
    expandSettings: "Expand settings",
  },
  licenses: {
    modelLicenses: "Model licenses",
    rustLicenses: "Rust dependency licenses",
    openRustLicenses: "Open Rust dependency licenses",
    loadingRustLicenses: "Loading Rust dependency licenses",
    usedBy: "Used by",
  },
  common: {
    search: "Find",
    resetLogs: "Reset logs",
    csvExport: "CSV export",
    start: "Start",
    stop: "Stop",
    volume: "Volume",
    unset: "Not set",
  },
  settings: {
    neoHttpEnabled: {
      label: "Send to YukakoneNEO",
      description:
        "When enabled, recognized text is sent to the YukakoneNEO HTTP API. Recognition logs are still shown when disabled.",
    },
    neoHttpPort: {
      label: "YukakoneNEO HTTP API port",
      description:
        "The HTTP port used by the built-in YukakoNEO API. The default is usually 15520.",
    },
    oscQuery: {
      label: "VRChat / OSCQuery settings",
      description:
        "Reads the VRChat mute state through OSCQuery and skips sending to YukakoneNEO while muted.",
      muteSyncLabel: "Do not send while muted in VRChat (requires OSC)",
      muteSyncDescription:
        "When enabled, Parapper checks MuteSelf through VRChat OSCQuery when speech recognition starts and does not bind an OSC UDP port.",
    },
    pauseThreshold: {
      label: "Silence duration before stop (ms)",
      description:
        "Silence duration before a phrase is considered complete. Larger values allow longer pauses.",
    },
    phraseThreshold: {
      label: "Speech duration before phrase (ms)",
      description:
        "Speech duration required before audio is treated as a phrase. Smaller values react to shorter voices.",
    },
    vadInterval: {
      label: "Decision interval",
      description:
        "How often VAD runs. Shorter intervals react faster but increase CPU load.",
    },
    vadThreshold: {
      label: "VAD threshold",
      description:
        "Speech confidence threshold. Higher values reduce false positives but may miss quiet speech.",
    },
    asrModel: {
      label: "ASR model",
      description:
        "Choose a language and sherpa-onnx model. NeMo models are heavier but may provide better accuracy.",
    },
    asrPrecision: {
      label: "ASR precision",
      description:
        "Quantization mode for the ASR model. int8-fp32 uses int8 encoder/joiner and fp32 decoder.",
    },
    asrThreads: {
      label: "CPU threads",
      description:
        "Intra CPU thread count used for ASR inference. max lets the runtime choose.",
      max: "max",
    },
    modelDir: {
      label: "Model directory",
      description:
        "Fixed model directory under Tauri app data, tied to the app identifier.",
    },
    downloadModels: {
      button: "Download models",
      tooltipReady:
        "Downloads model files and Silero VAD required by the selected ASR model and precision.",
    },
    debugAudioPlayback: {
      label: "ASR input audio playback debug",
      description:
        "When enabled, the 16 kHz mono audio passed to ASR is kept in the recognition log and can be played from log rows. Use this only while debugging because it increases memory usage.",
    },
    recognitionLogRetention: {
      label: "Recognition log retention",
      description:
        "Maximum number of recognition log rows kept in the UI. Unlimited retention increases memory usage during long sessions.",
    },
    debugAudioRetention: {
      label: "Debug audio retention",
      description:
        "Number of rows that keep audio samples when ASR input audio playback debug is enabled. Unlimited retention can use a large amount of memory.",
    },
    resetConfig: {
      button: "Reset settings",
      tooltipRunning:
        "Settings cannot be reset while recognition is running. Stop recognition first.",
      tooltipReady: "Restore default settings and apply them immediately.",
    },
    audioDevice: {
      label: "Device",
      placeholder: "Default input device",
      refreshAriaLabel: "Refresh device list",
      refreshTooltip: "Refresh the device list.",
    },
  },
  options: {
    retention: {
      limited: "Specify count",
      unlimited: "Unlimited",
    },
    asrModel: {
      reazonspeechK2V2: "Japanese (ReazonSpeech k2 v2)",
      nemoParakeetTdtV2Int8: "English (NeMo Parakeet TDT 0.6B v2 int8)",
      nemoParakeetTdtV3Int8:
        "European multilingual (NeMo Parakeet TDT 0.6B v3 int8)",
    },
  },
  recognitionLog: {
    title: "Recognition log",
    empty: "No recognized text",
    audioPlayTooltip: "Play the audio passed to ASR.",
    audioUnavailableTooltip:
      "This log row has no debug audio because ASR input audio playback debug was off.",
    audioSaveTooltip: "Save the audio passed to ASR as WAV.",
    playAriaLabel: "Play ASR input audio",
    downloadAriaLabel: "Save ASR input audio as WAV",
    csvHeaderText: "Recognized text",
    csvHeaderTime: "Time",
    csvHeaderSeconds: "Seconds",
    csvHeaderElapsedMs: "Elapsed time (ms)",
  },
  tooltip: {
    runtimeLocked: "This cannot be changed while recognition is running.",
    missingModels: "Models are not ready. Download models first.",
    startRecognition: "Start recognition.",
    stopRecognition: "Stop recognition.",
  },
  notifications: {
    configSaved: {
      title: "Changes applied",
      message: "Applied settings will also be used on the next launch.",
    },
    configSaveFailed: {
      title: "Failed to save settings",
    },
    audioDeviceSaveFailed: {
      title: "Failed to apply device settings",
    },
    audioDeviceRefreshFailed: {
      title: "Failed to refresh device list",
    },
    neoPortNotFound: {
      title: "NEO HTTP API port was not found",
      message: "Check the YukakoNEO settings.",
    },
    neoPortDetected: {
      title: "Detected NEO HTTP API port",
      message: "Applied {{port}} to the settings.",
    },
    connectionNotFound: {
      neo: {
        title: "NEO not found",
        message:
          "Sending to YukakoneNEO is enabled, but YukakoneNEO was not found.",
      },
      vrchat: {
        title: "VRChat not found",
        message:
          "VRChat integration is enabled, but VRChat OSCQuery was not found.",
      },
    },
    csvSaved: {
      title: "CSV saved",
    },
    csvSaveFailed: {
      title: "Failed to save CSV",
    },
    configReset: {
      title: "Settings reset",
      message:
        "Default settings were applied and will be used on the next launch.",
    },
    audioNotPlayable: {
      title: "No playable audio",
      message: "Enable ASR input audio playback debug before recognition.",
    },
    audioNotSavable: {
      title: "No audio to save",
      message: "Enable ASR input audio playback debug before recognition.",
    },
    audioSaved: {
      title: "ASR input audio saved",
    },
    audioSaveFailed: {
      title: "Failed to save ASR input audio",
    },
    modelsPrepared: {
      title: "Models prepared",
    },
  },
  downloadProgress: {
    label: "{{file}} ({{index}} / {{total}})",
  },
} as const;
