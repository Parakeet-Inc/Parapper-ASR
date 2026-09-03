[ドキュメント一覧](documents/README.md) | [English](documents/README.en.md) | 日本語

# Parapper

<!-- cspell:words parapper Silero ReazonSpeech OSCQuery UNAS Supertonic CTC SpeechBrain VoxLingua Vibrato UniDic Nemotron ENJP OpenMDW OpenRAIL ECAPA TDNN WebSocket -->

Parapperは、音声認識・翻訳・読み上げをCPU上でリアルタイムに実行するデスクトップアプリケーションです。マイクやシステム音声の文字起こしに加えて、字幕・翻訳・読み上げ結果を外部アプリへ渡したり、外部アプリからWebSocketで音声を受け取ったりできます。

[ゆかりネットコネクターNEO](https://nmori.github.io/yncneo-Docs/)(以下ゆかコネNEO)に対応しており、動画配信やVRSNSでのコミュニケーションを支えます。

## デモ

<https://github.com/user-attachments/assets/57383500-09a9-4668-953c-41a956db6971>

- [Paravo](https://parakeet-inc.com/paravo)で、ずんだもんへ音声変換しながらリアルタイムに録画しています。
- 翻訳にはGPT-5.4 nanoを使用しています。
- [ゆかりネットコネクターNEO](https://nmori.github.io/yncneo-Docs/)からOBSへ字幕を送っています。
- 3Dモデルは[ミニずんだもん公式VRMアバター](https://tohozunko.booth.pm/items/7304529)を使用しています。
- [VSeeFace](https://www.vseeface.icu/)で3Dモデルの動きをキャプチャしています。

## 特徴

Parapperは、配信やVRChatなど「同じPCで他のソフトと並行して動かす」場面で使いやすいことを目指しています。

- **CPUだけで動く**: GPUを使わずに音声認識・翻訳・読み上げまで完結。配信ソフトやゲーム、VRChatと同じPCで動かしてもグラフィック性能を取り合いません。
- **動作が軽い**: メモリやCPUの使用量を控えめにしているので、裏で動かしても他のソフトの邪魔をしにくいです。
- **オフラインで動く**: 一度モデルをダウンロードすれば、音声認識から翻訳・読み上げまで通信なしで使えます。ブラウザの状態や通信環境に左右されません。
- **反応が速い**: 話し終わってから字幕が出るまでの遅延が短く、会話や配信のテンポを保ちやすい設計です。
- **話しながら字幕が流れる**: 途中表示用にストリーミングASRモデル(Nemotron)を指定すると、無音を待たずに発話中も字幕が連続で更新されます。
- **発話区切りを柔軟に判定**: 無音検出(VAD)に加えて、日本語の文法境界で判定するMorph、AIで発話の完了を判定するNamoのTurn Detectorに対応。短い間を挟む話し方でも字幕が途中で切れにくくなります。
- **多言語対応**: 日本語、英語、その他ヨーロッパ系を含む多言語のASRに対応。UIも日本語/英語に対応しています。
- **設定プリセットですぐ使える**: 「文字起こしだけ」「翻訳もする」「読み上げまでする」など、用途別のプリセットから選んで始められます。

## インストール

[Releases](https://github.com/Parakeet-Inc/Parapper-ASR/releases)ページから最新のWindows x64向け`.msi`をダウンロードし、実行してください。

ソースから実行する場合は[開発・配布手順](documents/developer/development-help.md)を参照してください。

## 主な機能

Parapperは、配信やVRChatなど、同じPCで他のソフトと並行して動かす場面を想定しています。

- **ASR(音声認識)**: マイクやシステム音声をリアルタイムに文字起こしします。Nemotronを途中表示専用モデルとして組み合わせると、発話中も字幕を更新できます。
- **VAD / Turn Detector**: 無音を使うSimple、日本語の文法境界を使うMorph、AIで発話完了を判断するNamoから区切り方を選べます。
- **MT(翻訳)**: 日本語と英語の間をローカルモデルで翻訳するか、ゆかコネNEOの翻訳プラグインへ送信します。ローカル翻訳はOpenAI互換のlocalhost APIとして明示的に起動し、他のアプリから利用することもできます。
- **TTS(読み上げ)**: 認識結果または翻訳結果を、ローカルTTSかゆかコネNEOの読み上げプラグインで再生します。ローカルTTSでは出力デバイス、音量、声と言語を設定できます。
- **NC(ノイズキャンセリング)**: マイク環境のノイズを抑えてから認識します。
- **外部連携**: ゆかコネNEOへ字幕を送信し、VRChatのミュート状態に合わせて送信を止められます。開発者向けには、HTTPによる認識event送信と、WebSocketによるPCM入力・認識event返却を提供します。
- **プリセットとログ**: 組み込みプリセットに加えて現在の設定を名前付きで保存できます。認識履歴のCSV保存や、認識に使った音声の確認にも対応します。

詳しい画面操作、モデル選択、ゆかコネNEO・VRChat連携は[使い方](documents/how-to-use.md)を参照してください。

## 開発者向け情報

ビルド方法・配布手順・モデル詳細は[documents/developer/development-help.md](documents/developer/development-help.md)を参照してください。

日本語形態素辞書の生成・配布仕様は
[scripts/morph-dictionary/README.md](scripts/morph-dictionary/README.md)を参照してください。

外部接続を実装する場合は[開発者向け文書](documents/developer/README.md)、[ストリーミング音声認識プロトコル v1](documents/developer/streaming-recognition-protocol-v1.md)、[セキュリティ上の注意](documents/developer/security.md)を参照してください。

## 配信・動画でのクレジット表記について

本ソフトウェアはMIT Licenseで公開されています。

配信、動画等で本ソフトウェアをご利用いただく場合、クレジットを記載していただけるとモチベーションになります。

## 関連プロダクト: Paravo

[Paravo](https://parakeet-inc.com/paravo)は、Parakeet株式会社が開発する軽量・高品質なリアルタイムAIボイスチェンジャーです。Parapperと同じくCPUだけで動作し、低遅延で配信・ゲーム・VRChatに組み込みやすいことを重視しています。

「字幕はParapper、声はParavo」のように組み合わせると、CPUだけで完結するリアルタイム配信環境を作れます。詳細は[Paravo公式ページ](https://parakeet-inc.com/paravo)をご覧ください。

## ライセンス

- [Parapper](./LICENSE): MIT
- [ReazonSpeech K2 v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2): Apache-2.0
- [Parapper NeMo Parakeet TDT CTC 0.6B Ja ONNX](https://huggingface.co/nadare/parakeet-tdt_ctc-0.6b-ja-onnx-dynamic-int8): CC-BY-4.0。元モデルは[NVIDIA Parakeet TDT CTC 0.6B Ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja)で、ONNX再エクスポート・エンコーダ共有化・一部重みの量子化を行っています
- [NeMo Parakeet TDT 0.6B v2 int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8): CC-BY-4.0
- [NeMo Parakeet TDT 0.6B v3 int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8): CC-BY-4.0
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo): Apache-2.0（ASRの挙動参照。アプリへのソースコード組み込みはありません）
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): Apache-2.0（モデル配布元および互換性参照。実行時依存ではありません）
- [Nemotron Speech Streaming 0.6B English](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b): NVIDIA Open Model License
- [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b): OpenMDW-1.1
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): MIT
- [Silero VAD](https://github.com/snakers4/silero-vad): MIT
- [Namo Turn Detector v1 Japanese](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Japanese): Apache-2.0
- [Namo Turn Detector v1 English](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-English): Apache-2.0
- [Namo Turn Detector v1 Multilingual](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual): Apache-2.0
- [SpeechBrain ECAPA-TDNN VoxLingua107](https://huggingface.co/drakulavich/SpeechBrain-coreml): Apache-2.0
- [Vibrato UniDic CWJ 3.1.1 dictionary](https://clrd.ninjal.ac.jp/unidic_archive/cwj/3.1.1/): BSD-3-Clause
- [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas): MIT
- [static-embedding-japanese](https://huggingface.co/hotchpotch/static-embedding-japanese): MIT
- [LFM2-350M-ENJP-MT ONNX (ONNX Community conversion)](https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX): LFM Open License v1.0 (base model: `LiquidAI/LFM2-350M-ENJP-MT`; annual revenue above US$10M requires a separate commercial license)
- [CAT-Translate 0.8B ONNX Q4 block16](https://huggingface.co/nadare/CAT-Translate-0.8b-onnx-q4-k-quant): MIT (base model: `cyberagent/CAT-Translate-0.8b`)
- [Supertonic 2](https://huggingface.co/Supertone/supertonic-2): OpenRAIL-M
- [Supertonic 3](https://huggingface.co/Supertone/supertonic-3): OpenRAIL-M
- [Supertonic 3 ONNX Q4](https://huggingface.co/nadare/supertonic-3-onnx-q4): OpenRAIL-M (unofficial quantized derivative)
- Built-in hotword reading dictionaries (SudachiDict Full/UniDic/NEologd and CMUdict): [NOTICE](src-tauri/resources/hotword-reading/NOTICE.md), including Apache-2.0 and upstream attributions. The complete Apache-2.0 text is bundled with the application and available from Settings > Licenses.
- [JSUT corpus BASIC5000 text](https://sites.google.com/site/shinnosuketakamichi/publication/jsut): selected diagnostic references are attributed to Ryosuke Sonobe, Shinnosuke Takamichi, and Hiroshi Saruwatari under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). They are reproduced unmodified; no JSUT audio is included.
- 詳細な上流帰属と変更内容は[第三者通知](public/licenses/THIRD_PARTY_NOTICES.md)を参照してください。
