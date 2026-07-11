[English](documents/README.en.md) | 日本語

# Parapper

<!-- cspell:words parapper Silero ReazonSpeech sherpa OSCQuery UNAS Piper Supertonic CTC SpeechBrain VoxLingua Vibrato UniDic Nemotron ENJP OpenMDW ECAPA TDNN -->

ParapperはCPU上でリアルタイムに動く音声AIをまとめた、様々なアプリケーションと接続できるデスクトップ向けのアプリケーションです。

[ゆかりネットコネクターNEO](https://nmori.github.io/yncneo-Docs/)(以下ゆかコネNEO)に対応しており、動画配信やVRSNSでのコミュニケーションを支えます。

## デモ

<https://github.com/user-attachments/assets/57383500-09a9-4668-953c-41a956db6971>

- [Paravo](https://parakeet-inc.com/paravo): ずんだもんで音声変換しながらリアルタイムに録画しています。
- 翻訳にはGPT5.4 nanoを使用しています
- [ゆかりネットコネクターNEO](https://nmori.github.io/yncneo-Docs/)からOBSに字幕をつけています。
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

## できること

- **接続**: 認識・翻訳・読み上げの結果をゆかコネNEOに送り、配信の字幕表示や他ツールとの連携に活用できます。VRChatのミュート状態に合わせて送信のON/OFFも切り替えられます。
- **NC(ノイズキャンセリング)**: マイク環境のノイズを抑え、聞き取りやすい音声で認識します。
- **VAD / Turn Detector**: 喋っているかどうかを判定し、発話の区切りを判定します。
- **ASR(音声認識)**: マイクからの発話をリアルタイムに文字起こしします。途中表示専用にストリーミングASRモデルを組み合わせることもできます。
- **MT(翻訳)**: 認識した文を別の言語に翻訳します。ゆかコネNEOの翻訳プラグインに加えて、Parapper内蔵のローカル翻訳モデル(日本語⇔英語)も選べます。ローカル翻訳はOpenAI互換のlocalhost APIとして他のアプリへ公開することもできます。
- **TTS(読み上げ)**: 認識結果や翻訳結果を音声で読み上げます。
- **その他**: 用途別の設定をプリセットとして保存・切り替えできるほか、認識履歴の保存や音声の確認といったログ機能にも対応します。

## 対応モデル

初回起動時にプリセットを選ぶと、必要なモデルが自動でダウンロードされます。モデルはアプリのデータ領域に保存され、以降はオフラインで利用できます。

- **ASR**: 日本語(ReazonSpeech / NeMo Parakeet TDT CTC) / 英語 / 多言語(英語を含むヨーロッパ系25言語)
- **ストリーミングASR(途中表示用)**: 英語(Nemotron Speech Streaming) / 多言語(Nemotron 3.5 ASR Streaming、日本語・英語を含む29言語)に対応(任意)
- **VAD**: 発話区間の検出に対応
- **Turn Detector**: 日本語 / 英語 / 多言語の発話完了判定モデルに対応(任意)
- **日本語形態素辞書**: Morph / Namo 使用時の日本語 grammar boundary 判定に対応
- **ローカル翻訳**: 日本語⇔英語の翻訳モデル(LFM2-350M-ENJP-MT ONNX Community Q4)に対応(任意)
- **ノイズキャンセリング**: 軽量NCモデルに対応(任意)
- **ローカルTTS**: 日本語・英語を含む多言語の音声合成モデルに対応(任意)

各モデルの名称・対応言語・サイズなどの詳細は[documents/how-to-use.md](documents/how-to-use.md)を参照してください。

## インストール

### Windows

[Releases](https://github.com/Parakeet-Inc/Parapper-ASR/releases)ページから最新の`.msi`インストーラーをダウンロードして実行してください。

### Mac

[Releases](https://github.com/Parakeet-Inc/Parapper-ASR/releases)ページから最新の`.zip`ファイルを展開して実行してください。

> ゆかコネNEO連携(字幕送信・翻訳プラグイン・読み上げプラグイン)はWindowsのみ対応です。macOSでは翻訳・読み上げにローカル翻訳・ローカルTTSをご利用ください。

## 使い方

1. アプリを起動します。初回はオンボーディング画面が開きます。
2. `UIの使用言語 / UI language` で表示言語を選びます。
3. `設定プリセット` で最初に使うワークフローを選び、「反映してモデルをダウンロード」を押します。
   - 選んだプリセットに必要な ASR、VAD、Turn Detector、ノイズキャンセリング、TTS モデルがダウンロードされます。
   - 補足: 日本語モデル(ReazonSpeech K2 v2)を`int8-fp32`で使う場合、ダウンロードする容量はVADと合わせて約170MBです。回線状況によっては少し時間がかかります。
4. メイン画面で入力デバイスを選び、入力音量を確認します。
5. 必要に応じて設定パネルを開き、`接続` / `NC` / `VAD` / `ASR` / `MT` / `TTS` / `その他` / `ライセンス` を調整します。
6. 「スタート」を押すと認識が始まります。

Turn Detector、ゆかコネNEO、VRChat連携などの詳しい設定は[documents/how-to-use.md](documents/how-to-use.md)を参照してください。

## 開発者向け情報

ビルド方法・配布手順・モデル詳細は[documents/developer/development-help.md](documents/developer/development-help.md)を参照してください。

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
- [NeMo Parakeet TDT CTC 0.6B Ja 35000 int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8): CC-BY-4.0
- [NeMo Parakeet TDT 0.6B v2 int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8): CC-BY-4.0
- [NeMo Parakeet TDT 0.6B v3 int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8): CC-BY-4.0
- [Nemotron Speech Streaming 0.6B English](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b): OpenMDW-1.1
- [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b): OpenMDW-1.1
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): Apache-2.0
- [Silero VAD](https://github.com/snakers4/silero-vad): MIT
- [Namo Turn Detector v1 Japanese](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Japanese): Apache-2.0
- [Namo Turn Detector v1 English](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-English): Apache-2.0
- [Namo Turn Detector v1 Multilingual](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual): Apache-2.0
- [SpeechBrain ECAPA-TDNN VoxLingua107](https://huggingface.co/drakulavich/SpeechBrain-coreml): Apache-2.0
- [Vibrato UniDic CWJ 3.1.1 dictionary](https://github.com/daac-tools/vibrato/releases/tag/v0.5.0): see archive license files
- [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas): MIT
- [LFM2-350M-ENJP-MT ONNX (ONNX Community conversion)](https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX): LFM Open License v1.0 (base model: `LiquidAI/LFM2-350M-ENJP-MT`)
- [Piper voices](https://huggingface.co/rhasspy/piper-voices): MIT
- [espeak-ng-data](https://github.com/espeak-ng/espeak-ng/tree/master/espeak-ng-data): GPL-3.0-or-later
- [Supertonic 2](https://huggingface.co/Supertone/supertonic-2): OpenRAIL-M
- [Supertonic 3](https://huggingface.co/Supertone/supertonic-3): OpenRAIL-M
