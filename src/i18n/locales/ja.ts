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
    presetStep: "2. 設定プリセット",
    presetDescription:
      "最初に使うワークフローを選択します。選んだ設定を反映し、必要なモデルをダウンロードします。",
    next: "次へ",
    back: "戻る",
    downloadAndClose: "反映してモデルをダウンロード",
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
    noiseCancellation: "NC",
    other: "その他",
    licenses: "ライセンス",
    translation: "MT",
    speech: "TTS",
    collapseSettings: "設定を折りたたむ",
    expandSettings: "設定を展開",
  },
  licenses: {
    modelLicenses: "モデルライセンス",
    rustLicenses: "Rust依存ライセンス",
    openRustLicenses: "Rust依存ライセンスを開く",
    loadingRustLicenses: "Rust依存ライセンスを読み込み中",
    usedBy: "使用元",
  },
  common: {
    search: "探す",
    resetLogs: "ログを消去",
    csvExport: "CSV出力",
    start: "スタート",
    stop: "ストップ",
    volume: "音量",
    unset: "未設定",
  },
  settings: {
    neoHttpEnabled: {
      label: "ゆかコネNEOへ送信",
      description:
        "ONにするとASRテキストをゆかコネNEOのHTTP APIへ送信します。OFFでもASRログには表示されます。",
    },
    neoHttpPort: {
      label: "ゆかコネNEO HTTP API port",
      description:
        "ゆかコネNEO の本体内蔵 API が待ち受けている HTTP ポート番号です。通常は 15520 です。",
    },
    translationPluginHttpPort: {
      label: "翻訳/発話プラグイン HTTP port",
      description:
        "ゆかコネNEOの翻訳/発話連携サーバプラグインが待ち受けるHTTPポートです。通常は 8080 です。",
    },
    neoSendTiming: {
      label: "ゆかコネNEO送信タイミング",
      description: "ASR途中の更新も送るか、確定した結果だけ送るかを選びます。",
    },
    oscQuery: {
      label: "VRChat / OSCQuery設定",
      description:
        "VRChat のミュート状態を OSCQuery で読み取り、ミュート中はゆかコネNEOへ送信しないための設定です。",
      muteSyncLabel: "VRChatでミュートの時は送信しない(要OSC)",
      muteSyncDescription:
        "ON にするとASR開始時に VRChat OSCQuery から MuteSelf を確認します。Parapper は OSC UDP 受信ポートを bind しません。",
    },
    turnDetector: {
      label: "Turn Detector",
      description: "発話ターンの完了判定に使う方式です。",
    },
    interimResult: {
      label: "途中経過を表示する",
      description:
        "ON にすると、完了前の短い無音でそこまでの音声をASRに渡し、同じ発話ターンの途中経過として表示します。OFF の場合、短い無音ではASRを実行せず、完了までの無音時間まで待ちます。",
    },
    interimResultSilence: {
      label: "途中表示までの無音時間(ms)",
      description:
        "無音が何 ms 続いたら途中経過表示用に、そこまでの音声をASRへ渡すかを指定します。",
    },
    turnCheckSilence: {
      label: "完了までの無音時間(ms)",
      description:
        "無音が何 ms 続いたら発話完了判定に進むかを指定します。Namo が継続と判断した場合は同じ発話ターンとして保持します。",
    },
    segmentStartSpeech: {
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
    inputVolume: {
      label: "マイク入力音量 (dB)",
      description:
        "VAD と ASR に渡すマイク入力音声のゲインです。0 dB は入力をそのまま使います。",
    },
    asrNormalizeInput: {
      label: "入力前に音量を正規化する",
      description:
        "ON にすると、ASR と言語判定へ渡す音声をピーク基準で正規化します。VAD の判定には影響しません。",
    },
    namoTurnConfidence: {
      label: "Namo confidence threshold",
      description: "Namo が発話完了と判定するために必要な信頼度です。",
    },
    namoContextMaxTokens: {
      label: "Namo context max tokens",
      description:
        "Namo に渡す蓄積テキスト末尾の最大トークン数です。0 の場合は全文を渡します。",
    },
    turnRerecognizeFull: {
      label: "完了時に全体を再ASRする",
      description:
        "長無音で発話が完了したとき、全体音声をもう一度 ASR にかけて結果を上書きします。Namo が途中確定した場合は常に全体を再ASRします。",
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
    multilingualAsr: {
      label: "多言語ASRを使い分ける",
      description:
        "発話確定候補で言語判定し、有効化したASRモデルの範囲内でモデルとturn detectorを切り替えます。",
    },
    enabledAsrModels: {
      label: "使用するASRモデル",
      description:
        "言語判定後に使用してよいASRモデルです。未選択の言語に判定された場合は現在のASRモデルに戻します。",
    },
    modelDir: {
      label: "モデル保存先",
      description:
        "Tauri のアプリデータ領域に作成される固定のモデル保存先です。identifier に紐づきます。",
    },
    downloadModels: {
      button: "モデルをダウンロード",
      openButton: "ダウンロード",
      tooltipReady:
        "選択中の ASRモデル、Silero VAD、必要に応じて Namo Turn Detector、ローカルTTS、ノイズキャンセリングモデルをダウンロードします。",
    },
    debugAudioPlayback: {
      label: "ASR入力音声の再生デバッグ",
      description:
        "ONにすると、ASRへ渡した16kHz mono音声をASRログに保持し、ログ行の再生ボタンで確認できます。メモリ使用量が増えるためデバッグ時だけ使ってください。",
    },
    recognitionLogRetention: {
      label: "ASRログの保持",
      description:
        "ASRログを何件まで画面に保持するかを指定します。上限なしは長時間利用時にメモリ使用量が増えます。",
    },
    debugAudioRetention: {
      label: "Debug音声の保持",
      description:
        "ASR入力音声の再生デバッグがONのとき、音声サンプルを何件分保持するかを指定します。上限なしはメモリ使用量が大きくなります。",
    },
    configPresets: {
      title: "設定プリセット",
      description:
        "現在の設定を名前付きで保存できます。同じ名前は上書きされます。",
      nameLabel: "設定名",
      namePlaceholder: "配信用 日本語字幕",
      saveButton: "保存",
      loadLabel: "設定プリセット",
      loadPlaceholder: "設定を選択",
      applyButton: "反映",
      deleteButton: "削除",
      builtInLabel: "{{name}} (デフォルト)",
      deleteBuiltInTooltip: "デフォルトプリセットは削除できません。",
      emptyName: "設定名を入力してください。",
    },
    resetConfig: {
      button: "設定をリセットする",
      tooltipRunning: "ASR中は設定をリセットできません。先に停止してください。",
      tooltipReady: "設定をデフォルト値に戻し、すぐに反映します。",
    },
    audioDevice: {
      refreshAriaLabel: "デバイス一覧を更新",
      refreshTooltip: "デバイスの一覧を更新する",
    },
    inputAudioDevice: {
      label: "入力デバイス",
      placeholder: "既定の入力デバイス",
    },
  },
  noiseCancellationSettings: {
    enable: {
      label: "ノイズキャンセリングを有効化",
      description:
        "マイク音声を VAD と ASR に渡す前に、ノイズキャンセリングモデルで処理します。",
    },
    model: {
      label: "NCモデル",
      description:
        "ノイズキャンセリングのベースラインを選びます。今後ここに手法やモデルを追加できます。",
      disabledTooltip: "ノイズキャンセリングを有効化するとモデルを選べます。",
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
    turnDetector: {
      simple: "Simple",
      namo: "Namo",
    },
    noiseCancellationModel: {
      ulUnas: "UL-UNAS",
    },
    neoSendTiming: {
      interim: "ASR途中も送る",
      final: "確定後に送る",
    },
  },
  recognitionLog: {
    title: "ASRログ",
    empty: "ASRテキストはまだありません",
    partial: "判定中",
    audioPlayTooltip: "ASRに渡した音声を再生します。",
    audioUnavailableTooltip: "ASR入力音声の再生デバッグがOFFのログです。",
    audioSaveTooltip: "ASRに渡した音声をWAVで保存します。",
    playAriaLabel: "ASR入力音声を再生",
    downloadAriaLabel: "ASR入力音声をWAVで保存",
    csvHeaderText: "ASRテキスト",
    csvHeaderTime: "時刻",
    csvHeaderSeconds: "秒数",
    csvHeaderElapsedMs: "処理時間(ms)",
  },
  translationLog: {
    title: "翻訳ログ",
    empty: "翻訳はまだありません",
    errorBadge: "エラー",
  },
  translationSettings: {
    title: "翻訳",
    enable: {
      label: "翻訳を有効化",
      description:
        "ONにするとASRテキストをゆかコネNEOの翻訳/発話連携サーバプラグインへ送ります。",
    },
    sendTiming: {
      label: "翻訳タイミング",
    },
    mapping: {
      title: "翻訳マッピング",
      anyModel: "すべてのASRモデル",
      sourceModel: "元ASRモデル",
      targetLang: "翻訳先言語",
      addRow: "翻訳マッピングを追加",
      deleteRow: "翻訳マッピングを削除",
      moveUp: "翻訳マッピングを上へ移動",
      moveDown: "翻訳マッピングを下へ移動",
    },
  },
  speechSettings: {
    title: "テキスト読み上げ設定",
    talker: {
      label: "話者",
      description: "ゆかコネNEOの話者名です。例: ずんだもん-ノーマル/VOICEVOX",
      refresh: "更新",
      empty: "話者一覧を更新してください",
    },
    backend: {
      label: "読み上げ先",
      ync: "ゆかコネNEO",
      localTts: "ローカルTTS",
    },
    localTtsVoice: {
      label: "モデル",
    },
    localTtsLanguage: {
      label: "言語",
    },
    localTtsSpeaker: {
      label: "話者",
    },
    outputAudioDevice: {
      label: "出力デバイス",
      placeholder: "既定の出力デバイス",
    },
    volume: {
      label: "音量 (dB)",
    },
    neoReadAloudWarning:
      "ゆかコネNEOへ送信とParapperのテキスト読み上げを併用する場合は、ゆかコネNEO側で本文読み上げをスキップしてください。ONのままだと本文読み上げと翻訳読み上げが同じ読み上げキューに入り、翻訳読み上げが遅れることがあります。",
    mapping: {
      title: "テキスト読み上げマッピング",
      anyModel: "すべてのASRモデル",
      sourceModel: "元ASRモデル",
      targetLang: "翻訳先言語",
      addRow: "テキスト読み上げマッピングを追加",
      deleteRow: "テキスト読み上げマッピングを削除",
      mute: "テキスト読み上げマッピングをミュート",
      unmute: "テキスト読み上げマッピングのミュートを解除",
      moveUp: "テキスト読み上げマッピングを上へ移動",
      moveDown: "テキスト読み上げマッピングを下へ移動",
      sourceKind: {
        recognition: "ASR",
        translation: "MT",
      },
    },
    stopButton: "読み上げ中断",
  },
  tooltip: {
    runtimeLocked: "ASR中は変更できません。",
    neoHttpDisabled:
      "ゆかコネNEOへの送信がOFFのため利用できません。接続タブでONにしてください。",
    nativeConnectionsDisabled: "このOSでは外部ツール連携を利用できません。",
    missingModels: "モデルが未準備です。先にモデルをダウンロードしてください。",
    startRecognition: "ASRを開始します。",
    stopRecognition: "ASRを停止します。",
  },
  notifications: {
    configSaved: {
      title: "変更を反映しました",
      message: "反映した設定は次回起動時にも使われます。",
    },
    configSaveFailed: {
      title: "設定の保存に失敗しました",
    },
    configPresetsLoadFailed: {
      title: "保存済み設定の読み込みに失敗しました",
    },
    configPresetSaved: {
      title: "設定を保存しました",
      message: "{{name}} を保存しました。",
    },
    configPresetSaveFailed: {
      title: "設定プリセットの保存に失敗しました",
    },
    configPresetDeleted: {
      title: "設定プリセットを削除しました",
      message: "{{name}} を削除しました。",
    },
    configPresetDeleteFailed: {
      title: "設定プリセットの削除に失敗しました",
    },
    configPresetApplied: {
      title: "設定プリセットを反映しました",
      message: "{{name}} を反映しました。",
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
    pluginPortNotFound: {
      title: "翻訳/発話プラグイン HTTP portが見つかりません",
      message: "ゆかコネNEOの翻訳/発話連携サーバプラグインを確認してください。",
    },
    pluginPortDetected: {
      title: "翻訳/発話プラグイン HTTP portを検出しました",
      message: "{{port}} を設定に反映しました。",
    },
    connectionNotFound: {
      neo: {
        title: "NEO not found",
        message: "ゆかコネNEOへの送信がONですが、ゆかコネNEOが見つかりません。",
        detectedPortMessage:
          "設定されているNEO HTTP API portは {{configuredPort}} ですが、ゆかコネNEOは {{detectedPort}} で起動しています。接続設定で検索を押すか、portを {{detectedPort}} に変更してください。",
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
      message: "ASR入力音声の再生デバッグをONにしてからASRしてください。",
    },
    audioNotSavable: {
      title: "保存できる音声がありません",
      message: "ASR入力音声の再生デバッグをONにしてからASRしてください。",
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
    voiceListLoadFailed: {
      title: "話者リストの取得に失敗しました",
    },
    speechStopFailed: {
      title: "読み上げ中断に失敗しました",
    },
  },
  downloadProgress: {
    label: "{{file}} ({{index}} / {{total}})",
  },
} as const;
