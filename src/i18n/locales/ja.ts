export const ja = {
  app: {
    loading: "Loading",
  },
  language: {
    label: "表示言語",
    description:
      "公式対応は日本語と英語です。将来は locales に対訳ファイルを追加して拡張できます。",
    ja: "日本語",
    en: "English",
  },
  onboarding: {
    title: "初期設定 / Initial setup",
    languageStep: "1. UIの使用言語 / UI language",
    modelStep: "2. モデル",
    next: "次へ",
    back: "戻る",
    downloadAndClose: "モデルをダウンロードして閉じる",
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
    settings: "設定",
    connection: "接続",
    vad: "VAD",
    asr: "ASR",
    other: "その他",
    licenses: "ライセンス",
    collapseSettings: "設定を折りたたむ",
    expandSettings: "設定を展開",
  },
  licenses: {
    modelLicenses: "モデルライセンス",
    rustLicenses: "Rust依存ライセンス",
    openRustLicenses: "Rust依存ライセンスを開く",
    loadingRustLicenses: "Rust依存ライセンスを読み込み中",
    usedBy: "Used by",
  },
  common: {
    search: "探す",
    resetLogs: "Reset logs",
    csvExport: "CSV export",
    start: "スタート",
    stop: "ストップ",
    volume: "音量",
    unset: "未設定",
  },
  settings: {
    neoHttpEnabled: {
      label: "ゆかコネNEOへ送信",
      description:
        "ONにすると認識した文字をゆかコネNEOのHTTP APIへ送信します。OFFでも認識ログには表示されます。",
    },
    neoHttpPort: {
      label: "ゆかこねNEO HTTP API port",
      description:
        "ゆかコネNEO の本体内蔵 API が待ち受けている HTTP ポート番号です。通常は 15520 です。",
    },
    oscQuery: {
      label: "VRChat / OSCQuery設定",
      description:
        "VRChat のミュート状態を OSCQuery で読み取り、ミュート中はゆかこねNEOへ送信しないための設定です。",
      muteSyncLabel: "VRChatでミュートの時は送信しない(要OSC)",
      muteSyncDescription:
        "ON にすると音声認識開始時に VRChat OSCQuery から MuteSelf を確認します。Parapper は OSC UDP 受信ポートを bind しません。",
    },
    pauseThreshold: {
      label: "無音判定までの時間(ms)",
      description:
        "無音が何 ms 続いたら発話の終了とみなすかを指定します。大きいほど長い間を許容します。",
    },
    phraseThreshold: {
      label: "発話判定までの時間(ms)",
      description:
        "音声検知が何 ms 続いたら発話として扱うかを指定します。小さいほど短い声にも反応します。",
    },
    vadInterval: {
      label: "判定間隔",
      description:
        "VAD を実行する間隔です。短いほど反応は速くなりますが CPU 負荷が増えます。",
    },
    vadThreshold: {
      label: "VAD threshold",
      description:
        "音声と判定する確信度のしきい値です。高いほど誤検知は減りますが、小さい声を拾いにくくなります。",
    },
    asrModel: {
      label: "ASRモデル",
      description:
        "言語と sherpa-onnx モデルを選びます。NeMo は重い代わりに精度が期待できます。",
    },
    asrPrecision: {
      label: "ASR precision",
      description:
        "ASR モデルの量子化設定です。int8-fp32 は encoder/joiner を int8、decoder を fp32 で使います。",
    },
    asrThreads: {
      label: "CPUスレッド数",
      description:
        "ASR 推論で使う intra CPU スレッド数です。max はランタイムに任せます。",
      max: "max",
    },
    modelDir: {
      label: "モデル保存先",
      description:
        "Tauri のアプリデータ領域に作成される固定のモデル保存先です。identifier に紐づきます。",
    },
    downloadModels: {
      button: "モデルをダウンロード",
      tooltipReady:
        "選択中の ASRモデルと precision に必要なモデルファイルと Silero VAD をダウンロードします。",
    },
    debugAudioPlayback: {
      label: "ASR入力音声の再生デバッグ",
      description:
        "ONにすると、ASRへ渡した16kHz mono音声を認識ログに保持し、ログ行の再生ボタンで確認できます。メモリ使用量が増えるためデバッグ時だけ使ってください。",
    },
    recognitionLogRetention: {
      label: "認識ログの保持",
      description:
        "認識ログを何件まで画面に保持するかを指定します。上限なしは長時間利用時にメモリ使用量が増えます。",
    },
    debugAudioRetention: {
      label: "Debug音声の保持",
      description:
        "ASR入力音声の再生デバッグがONのとき、音声サンプルを何件分保持するかを指定します。上限なしはメモリ使用量が大きくなります。",
    },
    resetConfig: {
      button: "設定をリセットする",
      tooltipRunning:
        "認識中は設定をリセットできません。先に停止してください。",
      tooltipReady: "設定をデフォルト値に戻し、すぐに反映します。",
    },
    audioDevice: {
      label: "デバイス",
      placeholder: "既定の入力デバイス",
      refreshAriaLabel: "デバイス一覧を更新",
      refreshTooltip: "デバイスの一覧を更新する",
    },
  },
  options: {
    retention: {
      limited: "件数を指定",
      unlimited: "上限なし",
    },
    asrModel: {
      reazonspeechK2V2: "日本語 (ReazonSpeech k2 v2)",
      nemoParakeetTdtV2Int8: "英語 (NeMo Parakeet TDT 0.6B v2 int8)",
      nemoParakeetTdtV3Int8:
        "ヨーロッパ系他言語 (NeMo Parakeet TDT 0.6B v3 int8)",
    },
  },
  recognitionLog: {
    title: "認識ログ",
    empty: "No recognized text",
    audioPlayTooltip: "ASRに渡した音声を再生します。",
    audioUnavailableTooltip: "ASR入力音声の再生デバッグがOFFのログです。",
    audioSaveTooltip: "ASRに渡した音声をWAVで保存します。",
    playAriaLabel: "ASR入力音声を再生",
    downloadAriaLabel: "ASR入力音声をWAVで保存",
    csvHeaderText: "認識文字",
    csvHeaderTime: "時刻",
    csvHeaderSeconds: "秒数",
    csvHeaderElapsedMs: "処理時間(ms)",
  },
  tooltip: {
    runtimeLocked: "音声認識中は変更できません。",
    missingModels: "モデルが未準備です。先にモデルをダウンロードしてください。",
    startRecognition: "認識を開始します。",
    stopRecognition: "認識を停止します。",
  },
  notifications: {
    configSaved: {
      title: "変更を反映しました",
      message: "反映した設定は次回起動時にも使われます。",
    },
    configSaveFailed: {
      title: "設定の保存に失敗しました",
    },
    audioDeviceSaveFailed: {
      title: "デバイス設定の反映に失敗しました",
    },
    audioDeviceRefreshFailed: {
      title: "デバイス一覧の更新に失敗しました",
    },
    neoPortNotFound: {
      title: "NEO HTTP API portが見つかりません",
      message: "ゆかコネNEOの設定を確認してください。",
    },
    neoPortDetected: {
      title: "NEO HTTP API portを検出しました",
      message: "{{port}} を設定に反映しました。",
    },
    connectionNotFound: {
      neo: {
        title: "NEO not found",
        message: "ゆかコネNEOへの送信がONですが、ゆかコネNEOが見つかりません。",
      },
      vrchat: {
        title: "VRChat not found",
        message: "VRChat連携がONですが、VRChat OSCQueryが見つかりません。",
      },
    },
    csvSaved: {
      title: "CSVを保存しました",
    },
    csvSaveFailed: {
      title: "CSV保存に失敗しました",
    },
    configReset: {
      title: "設定をリセットしました",
      message: "デフォルト設定を反映し、次回起動時にも使われます。",
    },
    audioNotPlayable: {
      title: "再生できる音声がありません",
      message: "ASR入力音声の再生デバッグをONにしてから認識してください。",
    },
    audioNotSavable: {
      title: "保存できる音声がありません",
      message: "ASR入力音声の再生デバッグをONにしてから認識してください。",
    },
    audioSaved: {
      title: "ASR入力音声を保存しました",
    },
    audioSaveFailed: {
      title: "ASR入力音声の保存に失敗しました",
    },
    modelsPrepared: {
      title: "モデルを準備しました",
    },
  },
  downloadProgress: {
    label: "{{file}} ({{index}} / {{total}})",
  },
} as const;
