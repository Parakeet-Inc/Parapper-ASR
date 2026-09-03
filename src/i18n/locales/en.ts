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
    presetStep: "2. Settings preset",
    presetDescription:
      "Choose the workflow you want to start with. Parapper will apply it and download the required models.",
    next: "Next",
    back: "Back",
    downloadAndClose: "Apply, download, and close",
  },
  status: {
    vad: "VAD {{state}}",
    vrcMic: "VRC mic {{state}}",
    muted: "muted",
    open: "open",
    idle: "idle",
    ready: "ready",
    downloading: "downloading",
    missing: "missing",
    neoNotFound: "NEO not found",
    vrcNotFound: "VRChat not found",
    asrWarning: "ASR warning",
    recognition: {
      idle: "idle",
      waiting_for_client: "waiting for client",
      listening: "listening",
      draining: "draining",
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
    connection: "Input",
    vad: "VAD",
    asr: "ASR",
    noiseCancellation: "NC",
    other: "Other",
    downloads: "Downloads",
    licenses: "Licenses",
    translation: "MT",
    speech: "TTS",
    externalApps: "External apps",
    collapseSettings: "Collapse settings",
    expandSettings: "Expand settings",
  },
  settingsGroups: {
    stt: "STT",
    output: "Output",
    other: "Other",
  },
  sttProfiles: {
    selectorLabel: "STT profile",
    defaultName: "Profile {{number}}",
    add: "Add new",
    delete: "Delete profile",
    deleteFor: "Delete {{name}}",
    deleteConfirmationTitle: "Delete this profile?",
    deleteConfirmationBody:
      "{{name}} and its related STT settings will be deleted. This cannot be undone.",
    enabledFor: "Enable {{name}}",
    name: "Profile name",
    color: "Display color",
    volumeLabel: "Profile volume",
    mute: "Mute profile",
    unmute: "Unmute profile",
    inputLevel: "Input level",
    neoHttpEnabled: "Send to YNC NEO",
    neoHttpEnabledDescription:
      "Send recognition results from this STT profile to YNC NEO.",
    neoHttpEnabledFor: "Send {{name}} to YNC NEO",
    developerHttpEnabled: "Send over HTTP",
    developerHttpEnabledDescription:
      "Send recognition results from this STT profile to the developer HTTP connection.",
    developerHttpEnabledFor: "Send {{name}} over HTTP",
  },
  licenses: {
    modelLicenses: "Model licenses",
    rustLicenses: "Rust dependency licenses",
    openRustLicenses: "Open Rust dependency licenses",
    loadingRustLicenses: "Loading Rust dependency licenses",
    hotwordReadingNotices: "Built-in dictionary licenses",
    hotwordReadingNoticesDescription:
      "Attribution and license texts for the Morph UniDic CWJ 3.1.1 dictionary, SudachiDict Full/UniDic/NEologd, and CMUdict.",
    openHotwordReadingNotices: "Open built-in dictionary licenses",
    loadingHotwordNotices: "Loading dictionary licenses",
    failedToLoadHotwordNotices: "Could not load dictionary licenses.",
    thirdPartyNotices: "Third-party notices",
    thirdPartyNoticesDescription:
      "Attribution for upstream ASR models, reference implementations, and model distributions, including a description of Parapper's modifications.",
    openThirdPartyNotices: "Open third-party notices",
    loadingThirdPartyNotices: "Loading third-party notices",
    failedToLoadThirdPartyNotices: "Could not load third-party notices.",
    usedBy: "Used by",
  },
  common: {
    cancel: "Cancel",
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
      label: "Send to YNC",
      description:
        "When enabled, ASR text is sent to the YNC HTTP API. ASR logs are still shown when disabled.",
    },
    neoHttpPort: {
      label: "YNC HTTP API port",
      description:
        "The HTTP port used by the built-in YNC API. The default is usually 15520.",
    },
    translationPluginHttpPort: {
      label: "Translation/speech plugin HTTP port",
      description:
        "HTTP port used by the YNC translation/speech server plugin. The default is usually 8080.",
    },
    neoSendTiming: {
      label: "YNC send timing",
      description:
        "Choose whether interim recognition updates are sent or only finalized results are sent.",
    },
    oscQuery: {
      label: "VRChat / OSCQuery settings",
      muteSyncLabel: "Do not send while muted in VRChat (requires OSC)",
    },
    turnDetector: {
      label: "Turn Detector",
      description: "How Parapper detects completed speech turns.",
    },
    interimResult: {
      label: "Show interim results",
      description:
        "When enabled, short silence before completion sends the audio so far to ASR and updates the same turn as an interim result. When disabled, short silence does not run ASR and Parapper waits for completion silence.",
    },
    interimResultSilence: {
      label: "Silence before interim result (ms)",
      description:
        "Silence duration before the audio so far is sent to ASR for an interim display update.",
    },
    turnCheckSilence: {
      label: "Silence before completion (ms)",
      description:
        "Silence duration before Parapper checks whether the turn is complete. If Namo says continue, the same turn stays open.",
    },
    segmentStartSpeech: {
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
    inputVolume: {
      label: "Microphone input volume (dB)",
      description:
        "Gain applied to microphone audio before VAD and ASR. 0 dB keeps the input unchanged.",
    },
    asrNormalizeInput: {
      label: "Normalize volume before ASR input",
      description:
        "When enabled, audio passed to ASR and language detection is peak-normalized. VAD decisions are not affected.",
    },
    namoTurnConfidence: {
      label: "Namo confidence threshold",
      description: "Minimum confidence required for Namo to finalize a turn.",
    },
    namoContextMaxTokens: {
      label: "Namo context max tokens",
      description:
        "Maximum trailing token count passed to Namo. 0 sends the full accumulated text.",
    },
    turnRerecognizeFull: {
      label: "Re-recognize full turn on completion",
      description:
        "When a turn completes after long silence, run ASR again on the full turn audio and replace the combined text. Morph / Namo always re-recognize the full audio for completion decisions.",
    },
    asrModel: {
      label: "ASR model",
      description:
        "Choose the ASR model. Japanese models also provide a separate inference mode.",
    },
    asrMode: {
      label: "ASR mode",
      description:
        "Fast and lightweight uses greedy decoding; Accuracy uses additional beam-search rescoring. Models without an accuracy decoder offer fast mode only.",
      options: {
        fast: "Fast and lightweight",
        accurate: "Accuracy",
      },
    },
    interimAsrModel: {
      label: "Interim ASR model",
      description:
        "ASR model used only for interim display. By default, this uses the primary ASR model.",
      primary: "Same as ASR model",
      nemotronEnglish: "Nemotron (English)",
      nemotronMultilingual: "Nemotron 3.5 (multilingual)",
      latencyLabel: "Nemotron latency",
      latencyDescription:
        "Choose the balance between interim responsiveness and recognition accuracy. This cannot be changed while recognition is running.",
      latency80: "80 ms",
      latency160: "160 ms",
      latency320: "320 ms",
      latency560: "560 ms",
      latency1120: "1120 ms",
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
    hotwords: {
      label: "Hotwords",
      count: "{{count}} entries",
      manage: "Manage hotwords",
      enable: "Enable hotwords",
      enableDescription:
        "Apply registered hotwords during candidate selection in accuracy mode. Registered entries are kept when disabled.",
      accurateOnly: "Hotwords can only be enabled in accuracy mode.",
      title: "Hotword management",
      description:
        "Register surface forms and hiragana readings to prioritize in accuracy mode.\nEnter one hiragana reading per line.\nThe priority multiplier defaults to x100; x1 is neutral and values below 1 suppress a phrase.",
      entry: "Hotword {{number}}",
      surfaceColumn: "Surface form",
      readingsColumn: "Readings",
      scoreColumn: "Priority multiplier",
      surface: "Surface form (required)",
      surfacePlaceholder: "Example: 斎藤",
      surfaceRequired: "Enter a surface form.",
      duplicateSurface: "More than one hotword uses this surface form.",
      pathCollision:
        "This surface or reading has the same token path as another hotword.",
      terminalPrefix:
        "This is a prefix of another surface or reading. Change either the shorter or longer entry.",
      readings: "Readings (optional)",
      readingsDescription:
        "Enter multiple readings on separate lines. You may leave this blank.",
      readingsPlaceholder: "ぱらっぱー",
      autofill: "Autofill",
      autofillHelp:
        "Look up an exact English spelling, then estimate a reading from its pronunciation if needed.",
      autofillEmpty: "No reading suggestion was found.",
      autofillFailed: "Could not generate reading suggestions.",
      blankReading: "Remove the blank reading or enter a value.",
      duplicateReading: "This reading is duplicated.",
      score: "Priority multiplier (optional)",
      invalidScore: "Enter a finite priority multiplier greater than zero.",
      scorePlaceholder: "100 (default)",
      add: "Add hotword",
      delete: "Delete hotword",
      empty: "No hotwords have been registered.",
      cancel: "Cancel",
      save: "Save",
    },
    multilingualAsr: {
      label: "Use multilingual ASR routing",
      description:
        "Detect language at turn-final candidates and switch ASR model and turn detector only within the enabled model set.",
    },
    enabledAsrModels: {
      label: "Enabled ASR models",
      description:
        "ASR models allowed after language detection. If detection maps to a disabled language, the current ASR model is used instead.",
    },
    modelDir: {
      label: "Model directory",
      description:
        "Fixed model directory under Tauri app data, tied to the app identifier.",
    },
    downloadModels: {
      button: "Download models",
      openButton: "Download",
      tooltipReady:
        "Downloads the selected ASR model, Silero VAD, Namo Turn Detector, local TTS, and noise cancellation model when required.",
    },
    debugAudioPlayback: {
      label: "ASR input audio playback debug",
      description:
        "When enabled, the 16 kHz mono audio passed to ASR is kept in the ASR log and can be played from log rows. Use this only while debugging because it increases memory usage.",
    },
    recognitionLogRetention: {
      label: "ASR log retention",
      description:
        "Maximum number of ASR log rows kept in the UI. Unlimited retention increases memory usage during long sessions.",
    },
    debugAudioRetention: {
      label: "Debug audio retention",
      description:
        "Number of rows that keep audio samples when ASR input audio playback debug is enabled. Unlimited retention can use a large amount of memory.",
    },
    configPresets: {
      title: "Settings presets",
      description:
        "Save the current settings with a name. Existing names are overwritten.",
      nameLabel: "Preset name",
      namePlaceholder: "Streaming Japanese captions",
      saveButton: "Save",
      loadLabel: "Preset",
      loadPlaceholder: "Choose a preset",
      applyButton: "Apply",
      deleteButton: "Delete",
      builtInLabel: "{{name}} (default)",
      deleteBuiltInTooltip: "Default presets cannot be deleted.",
      emptyName: "Enter a preset name.",
    },
    resetConfig: {
      button: "Reset settings",
      tooltipRunning:
        "Settings cannot be reset while ASR is running. Stop ASR first.",
      tooltipReady: "Restore default settings and apply them immediately.",
    },
    audioDevice: {
      refreshAriaLabel: "Refresh device list",
      refreshTooltip: "Refresh the device list.",
    },
    inputAudioDevice: {
      label: "Input device",
      description: "Input device and channel assigned to this STT profile.",
      placeholder: "Default input device",
      channelLabel: "Input channel",
      channelPlaceholder: "Choose a channel",
      channel: "Channel {{number}}",
      loopbackGroup: "Speaker output (loopback)",
      networkGroup: "Network input",
      webSocket: "WebSocket (PCM 16 kHz)",
    },
  },
  connectionSettings: {
    streamingEnabled: "Enable HTTP / WebSocket connection (Developers)",
    connectionMode: "Connection mode",
    neoProfiles: "STT profiles sent to YNC NEO",
    httpUrl: "HTTP destination URL",
    httpProfiles: "STT profiles sent over HTTP",
    httpPayloadExample: "Example HTTP payload",
    bindAddress: "Bind address",
    port: "WebSocket port",
    apiKey: "API key",
    apiKeyPlaceholder: "Optional for local connections",
    outputMode: "Recognition output",
    webSocketOnly: "WebSocket only",
    webSocketAndDesktop: "WebSocket + desktop integrations",
    endpoint: "Endpoint: ws://{{address}}:{{port}}/ws/recognition",
  },
  externalAppSettings: {
    yncEnabled: "Connect to YNC",
    yncPlugin: "YNC translation plugin",
  },
  noiseCancellationSettings: {
    enable: {
      label: "Enable noise cancellation",
      description:
        "Use noise-cancelled audio for VAD. Choose separately whether ASR receives it.",
    },
    target: {
      label: "NC application",
      description:
        "VAD only keeps ASR audio unmodified while suppressing noise-triggered speech detection.",
      disabledTooltip:
        "Enable noise cancellation to choose where it is applied.",
    },
    model: {
      label: "NC model",
      description:
        "Choose the noise cancellation baseline. Additional methods and models can be added here later.",
      disabledTooltip: "Enable noise cancellation to choose a model.",
    },
  },
  options: {
    retention: {
      limited: "Specify count",
      unlimited: "Unlimited",
    },
    asrModel: {
      reazonspeechK2V2: "Japanese (ReazonSpeech k2 v2)",
      nemoParakeetTdtCtcJa35000Int8:
        "Japanese (NeMo Parakeet TDT-CTC 0.6B 35000h int8)",
      nemoParakeetTdtV2Int8: "English (NeMo Parakeet TDT 0.6B v2 int8)",
      nemoParakeetTdtV3Int8:
        "European multilingual (NeMo Parakeet TDT 0.6B v3 int8)",
      nemotronSpeechStreamingEn80MsInt8:
        "English (Nemotron Speech Streaming 0.6B 80ms int8)",
      nemotronSpeechStreamingEn160MsInt8:
        "English (Nemotron Speech Streaming 0.6B 160ms int8)",
      nemotronSpeechStreamingEn320MsInt8:
        "English (Nemotron Speech Streaming 0.6B 320ms int8)",
      nemotronSpeechStreamingEn560MsInt8:
        "English (Nemotron Speech Streaming 0.6B 560ms int8)",
      nemotronSpeechStreamingEn1120MsInt8:
        "English (Nemotron Speech Streaming 0.6B 1120ms int8)",
      nemotron35AsrStreaming80MsInt8:
        "Multilingual (Nemotron 3.5 ASR Streaming 0.6B 80ms int8)",
      nemotron35AsrStreaming160MsInt8:
        "Multilingual (Nemotron 3.5 ASR Streaming 0.6B 160ms int8)",
      nemotron35AsrStreaming320MsInt8:
        "Multilingual (Nemotron 3.5 ASR Streaming 0.6B 320ms int8)",
      nemotron35AsrStreaming560MsInt8:
        "Multilingual (Nemotron 3.5 ASR Streaming 0.6B 560ms int8)",
      nemotron35AsrStreaming1120MsInt8:
        "Multilingual (Nemotron 3.5 ASR Streaming 0.6B 1120ms int8)",
    },
    turnDetector: {
      simple: "Simple",
      morph: "Morph",
      namo: "Namo",
    },
    noiseCancellationModel: {
      ulUnas: "UL-UNAS",
    },
    noiseCancellationTarget: {
      vadOnly: "VAD only",
      vadAndAsr: "VAD and ASR",
    },
    neoSendTiming: {
      interim: "Send interim updates",
      final: "Final only",
    },
  },
  recognitionLog: {
    title: "ASR log",
    empty: "No ASR text",
    partial: "pending",
    audioPlayTooltip: "Play the audio passed to ASR.",
    audioUnavailableTooltip:
      "This log row has no debug audio because ASR input audio playback debug was off.",
    audioSaveTooltip: "Save the audio passed to ASR as WAV.",
    playAriaLabel: "Play ASR input audio",
    downloadAriaLabel: "Save ASR input audio as WAV",
    csvHeaderText: "ASR text",
    csvHeaderTime: "Time",
    csvHeaderSeconds: "Seconds",
    csvHeaderElapsedMs: "Elapsed time (ms)",
  },
  translationLog: {
    title: "Translation log",
    empty: "No translated text",
    errorBadge: "error",
  },
  translationSettings: {
    title: "Translation",
    enable: {
      label: "Enable translation",
      description:
        "Translate ASR text with either the YNC plugin or the local Japanese/English model.",
    },
    sendTiming: {
      label: "Translation timing",
    },
    localServer: {
      title: "Run a translation HTTP server locally",
      start: "Start",
      stop: "Stop",
      downloadModel: "Download model",
      status: {
        stopped: "Stopped",
        starting: "Starting…",
        running: "Listening on 127.0.0.1:{{port}}",
        stopping: "Stopping…",
        error: "Error",
        modelMissing: "The server model is not downloaded yet",
      },
      setup: {
        title: "Configure YNC NEO",
        engine:
          'In YNC NEO translation settings, select "Local Translation (Personal / Custom API)" for the target language engine.',
        url: "Set the URL to http://127.0.0.1:{{port}}.",
        postMode: 'Select "OpenAI Format" for POST mode.',
        model:
          'The Model field can contain any display name. Choose the actual model using "Server model" above.',
        avoidInputApi:
          "Do not use the YNC NEO built-in /api/input endpoint or port 15520.",
      },
      mode: {
        off: "Off",
        auto: "With MT",
        on: "Server only",
      },
      port: {
        label: "Listen port",
        description:
          "localhost port that receives translation requests at POST /v1/chat/completions. The usual value is 18081.",
      },
      model: {
        label: "Server model",
        description:
          "Local translation model used for requests from external apps.",
      },
    },
    mapping: {
      title: "Translation mappings",
      anyModel: "All ASR models",
      sourceModel: "Source ASR model",
      sourceLang: "Source language",
      targetLang: "Target language",
      addRow: "Add translation mapping",
      deleteRow: "Delete translation mapping",
      moveUp: "Move translation mapping up",
      moveDown: "Move translation mapping down",
    },
    backend: {
      label: "Translation backend",
      ync: "YNC",
      local: "Local",
    },
    localModel: {
      label: "Local translation model",
    },
  },
  speechSettings: {
    title: "Text-to-speech settings",
    talker: {
      label: "Talker",
      description: "YNC talker name, for example: Zundamon-normal/VOICEVOX.",
      refresh: "Refresh",
      empty: "Refresh the talker list",
    },
    backend: {
      label: "Speech backend",
      ync: "YNC",
      localTts: "Local TTS",
    },
    localTtsVoice: {
      label: "Model",
    },
    localTtsLanguage: {
      label: "Language",
    },
    localTtsSpeaker: {
      label: "Speaker",
    },
    outputAudioDevice: {
      label: "Output device",
      placeholder: "Default output device",
    },
    volume: {
      label: "Volume (dB)",
    },
    neoReadAloudWarning:
      "When sending text to YNC and using Parapper text-to-speech together, configure YNC to skip reading the main text aloud. Otherwise main-text speech and translation speech can enter the same speech queue and delay translation speech.",
    mapping: {
      title: "Text-to-speech mappings",
      anyModel: "All ASR models",
      sourceModel: "Source ASR model",
      targetLang: "Translation target",
      addRow: "Add text-to-speech mapping",
      deleteRow: "Delete text-to-speech mapping",
      mute: "Mute text-to-speech mapping",
      unmute: "Unmute text-to-speech mapping",
      moveUp: "Move text-to-speech mapping up",
      moveDown: "Move text-to-speech mapping down",
      sourceKind: {
        recognition: "ASR",
        translation: "MT",
      },
    },
    stopButton: "Interrupt speech",
  },
  tooltip: {
    runtimeLocked: "This cannot be changed while ASR is running.",
    neoHttpDisabled:
      "YNC sending is off. Enable it in the Connection tab to use this.",
    nativeConnectionsDisabled:
      "External tool connections are not available on this OS.",
    missingModels: "Models are not ready. Download models first.",
    startRecognition: "Start ASR.",
    stopRecognition: "Stop ASR.",
  },
  notifications: {
    configSaved: {
      title: "Changes applied",
      message: "Applied settings will also be used on the next launch.",
    },
    configSaveFailed: {
      title: "Failed to save settings",
    },
    configPresetsLoadFailed: {
      title: "Failed to load saved settings",
    },
    configPresetSaved: {
      title: "Settings saved",
      message: "Saved {{name}}.",
    },
    configPresetSaveFailed: {
      title: "Failed to save settings preset",
    },
    configPresetDeleted: {
      title: "Settings preset deleted",
      message: "Deleted {{name}}.",
    },
    configPresetDeleteFailed: {
      title: "Failed to delete settings preset",
    },
    configPresetApplied: {
      title: "Settings preset applied",
      message: "Applied {{name}}.",
    },
    audioDeviceSaveFailed: {
      title: "Failed to apply device settings",
    },
    sttProfileAddFailed: {
      title: "Could not add STT profile",
      message: "There is no unused input-device channel.",
    },
    audioDeviceRefreshFailed: {
      title: "Failed to refresh device list",
    },
    neoPortNotFound: {
      title: "NEO HTTP API port was not found",
      message: "Check the YNC settings.",
    },
    neoPortDetected: {
      title: "Detected NEO HTTP API port",
      message: "Applied {{port}} to the settings.",
    },
    pluginPortNotFound: {
      title: "Translation/speech plugin HTTP port was not found",
      message: "Check the YNC translation/speech server plugin.",
    },
    pluginPortDetected: {
      title: "Detected translation/speech plugin HTTP port",
      message: "Applied {{port}} to the settings.",
    },
    connectionNotFound: {
      neo: {
        title: "NEO not found",
        message: "Sending to YNC is enabled, but YNC was not found.",
        detectedPortMessage:
          "The configured NEO HTTP API port is {{configuredPort}}, but YNC is running on {{detectedPort}}. Press Search in connection settings or change the port to {{detectedPort}}.",
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
      message: "Enable ASR input audio playback debug before ASR.",
    },
    audioNotSavable: {
      title: "No audio to save",
      message: "Enable ASR input audio playback debug before ASR.",
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
    voiceListLoadFailed: {
      title: "Failed to load voice list",
    },
    speechStopFailed: {
      title: "Failed to interrupt speech",
    },
    loopbackPermissionRequired: {
      title: "System Audio Recording permission required",
      message:
        "Allow Parapper to record system audio in System Settings → Privacy & Security → System Audio Recording, otherwise speaker capture stays silent.",
    },
  },
  downloadProgress: {
    label: "{{file}} ({{index}} / {{total}})",
  },
} as const;
