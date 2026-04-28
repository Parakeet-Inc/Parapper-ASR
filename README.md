# Parapper

<!-- cspell:words parapper Silero ReazonSpeech sherpa OSCQuery -->

Parapperは、マイクの音声をリアルタイムに文字起こしし、[ゆかりネットコネクターNEO](https://nmori.github.io/yncneo-Docs/)(以下ゆかコネNEO)連携できるデスクトップ向けの音声認識アプリケーションです。

動画配信やVRChatでのコミュニケーションで、自分の発話を字幕として送り出したいときに使えます。

## デモ

https://github.com/user-attachments/assets/57383500-09a9-4668-953c-41a956db6971

- [Paravo](https://parakeet-inc.com/paravo): ずんだもんで音声変換しながらリアルタイムに録画しています。
- 翻訳にはGPT5.4 nanoを使用しています
- [ゆかりネットコネクターNEO](https://nmori.github.io/yncneo-Docs/)からOBSに字幕をつけています。
- 3Dモデルは[ミニずんだもん公式VRMアバター](https://tohozunko.booth.pm/items/7304529)を使用しています。
- [VSeeFace](https://www.vseeface.icu/)で3Dモデルの動きをキャプチャしています。

## 特徴

ゆかコネNEOと組み合わせて使う音声認識には、ブラウザ音声認識やWhisperと組み合わせた音声認識など、いくつかの選択肢があります。Parapperは次のような場面で選びやすい一つになることを目指しています。

- **CPUだけで動く**: GPUがなくても使える。配信やゲームと同じPCで動かしても、グラフィック性能を取り合わずに済む。
- **動作が軽い**: メモリやCPUの使用量を控えめにしているので、配信中に裏で動かしても他のソフトの邪魔をしにくい。
- **インターネットがなくても動く**: 一度モデルをダウンロードすれば、あとはオフラインで使える。ブラウザの状態や通信環境に左右されない。
- **反応が速い**: 話し終わってから字幕が出るまでが短く、会話や配信のテンポを保ちやすい。
- **精度も実用的**: 軽さを優先しつつ、配信や会話用途で実用しやすい認識精度。
- **日本語と、英語を含む一部のヨーロッパ系言語に対応**: UIは日本語/英語に対応。認識モデルを切り替えることで、日本語配信でも英語配信でも運用できる。

## できること

- マイクから取り込んだ音声をリアルタイムに認識して字幕として表示します
- 認識結果をゆかコネNEOのHTTP APIに送信し、字幕配信に活用できます
- VRChatのミュート状態をOSCQueryで読み取り、ミュート中は送信を止めます
- 日本語と、英語を含む一部のヨーロッパ系言語のモデルを選んで使えます
- 認識ログの保存(CSVエクスポート)や、ASRに渡した音声の再生・WAV保存にも対応します

## 対応モデル

初回起動時に使うモデルを選び、ダウンロードして使います。モデルはアプリのデータ領域に保存されます。

- 日本語: ReazonSpeech K2 v2
- 英語: NeMo Parakeet TDT 0.6B v2
- 英語を含む一部のヨーロッパ系言語: NeMo Parakeet TDT 0.6B v3
- 音声区間検出: Silero VAD

日本語モデルでは精度と速度のバランスを3段階(`int8`, `int8-fp32`, `float32`)から選べます。

## インストール

### Windows

[Releases](../../releases)ページから最新の`.msi`インストーラーをダウンロードして実行してください。

### Mac

[Releases](../../releases)ページから最新の`.zip`ファイルを展開して実行してください。

## 使い方

1. アプリを起動します。初回はオンボーディング画面が開きます。
2. 表示言語(日本語/English)を選びます。
3. ASRモデルを選び、「モデルをダウンロード」を押します。ダウンロードが完了するまで待ってください。
   - 補足: 日本語モデル(ReazonSpeech K2 v2)を`int8-fp32`(デフォルト)で使う場合、ダウンロードする容量はVADと合わせて約170MBです。回線状況によっては少し時間がかかります。
4. 必要に応じて「設定」タブで以下を調整します。
   - **接続**: ゆかコネNEOへの送信、VRChatミュート連動
   - **VAD**: 無音判定までの時間、発話判定までの時間、判定間隔、しきい値
   - **ASR**: モデル、量子化精度、CPUスレッド数
   - **その他**: 認識ログの保持件数、ASR入力音声の再生デバッグ
5. メイン画面で入力デバイスを選び、「スタート」を押すと認識が始まります。

### ゆかコネNEOとの連携

「ゆかコネNEOへ送信」をONにすると、認識した文字をゆかコネNEOのHTTP APIに送ります。ポート番号は通常`15520`で、本体側の設定に合わせて変更できます。

OSC連携の仕様上、ゆかコネNEOはv2.3.58の安定版以降が必要です。古いバージョンを使っている場合は、先にゆかコネNEOをアップデートしてください。

### VRChatとの連携

「VRChatでミュートの時は送信しない」をONにすると、認識開始時にVRChatのOSCQueryからMuteSelf(自分のミュート状態)を読み取り、ミュート中は送信をスキップします。

アプリ内の「設定 > ライセンス」タブから、利用しているモデルおよびRust依存クレートのライセンス一覧を確認できます。

## 開発者向け情報

ビルド方法・配布手順・モデル詳細は[documents/development.md](documents/development.md)を参照してください。

## 配信・動画でのクレジット表記について

本ソフトウェアはMIT Licenseで公開されています。

配信、動画等で本ソフトウェアをご利用いただく場合、クレジットを記載していただけるとモチベーションになります。

## 関連プロダクト: Paravo

[Paravo](https://parakeet-inc.com/paravo)は、Parakeet株式会社が開発する軽量・高品質なリアルタイムAIボイスチェンジャーです。Parapperと同じくCPUだけで動作し、低遅延で配信・ゲーム・VRChatに組み込みやすいことを重視しています。

「字幕はParapper、声はParavo」のように組み合わせると、CPUだけで完結するリアルタイム配信環境を作れます。詳細は[Paravo公式ページ](https://parakeet-inc.com/paravo)をご覧ください。

## ライセンス

- [Parapper](./LICENSE): MIT
- [ReazonSpeech K2 v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2): Apache-2.0
- [NeMo Parakeet TDT 0.6B v2 int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8): CC-BY-4.0
- [NeMo Parakeet TDT 0.6B v3 int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8): CC-BY-4.0
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): Apache-2.0
- [Silero VAD](https://github.com/snakers4/silero-vad): MIT
