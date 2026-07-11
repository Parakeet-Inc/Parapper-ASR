English | [日本語](../README.md)

# Parapper

<!-- cspell:words parapper Silero ReazonSpeech sherpa OSCQuery UNAS Piper Supertonic CTC SpeechBrain VoxLingua Vibrato UniDic Nemotron ENJP OpenMDW ECAPA TDNN Paravo Zundamon VSeeFace VRSNS -->

Parapper is a desktop application that bundles voice AI running in real time on CPU and connects to a variety of other applications.

It supports [Yukarinette Connector NEO](https://nmori.github.io/yncneo-Docs/) ("YNC NEO" below) and helps you communicate in video streaming and VR social platforms.

## Demo

<https://github.com/user-attachments/assets/57383500-09a9-4668-953c-41a956db6971>

- [Paravo](https://parakeet-inc.com/paravo): recorded in real time while converting the voice to Zundamon.
- GPT5.4 nano is used for translation.
- Subtitles are displayed in OBS via [Yukarinette Connector NEO](https://nmori.github.io/yncneo-Docs/).
- The 3D model is the [official Mini Zundamon VRM avatar](https://tohozunko.booth.pm/items/7304529).
- [VSeeFace](https://www.vseeface.icu/) captures the 3D model's motion.

## Features

Parapper aims to be easy to use in situations where it runs alongside other software on the same PC, such as streaming or VRChat.

- **Runs on CPU only**: Speech recognition, translation, and text-to-speech all work without a GPU, so it never competes for graphics performance with streaming software, games, or VRChat on the same PC.
- **Lightweight**: Memory and CPU usage are kept modest, so it stays out of the way of other software even when running in the background.
- **Works offline**: Once the models are downloaded, everything from recognition to translation and speech runs without a network connection — unaffected by browser state or network conditions.
- **Fast response**: The delay from the end of an utterance to the subtitle is short, making it easy to keep up the tempo of conversations and streams.
- **Subtitles flow while you speak**: Assign a streaming ASR model (Nemotron) for interim display, and captions update continuously while you are speaking, without waiting for silence.
- **Flexible turn detection**: In addition to silence detection (VAD), Turn Detectors can decide utterance completion from Japanese grammatical boundaries (Morph) or with an AI model (Namo), so subtitles are less likely to be cut off mid-sentence even if you pause briefly while speaking.
- **Multilingual**: ASR supports Japanese, English, and other languages including European ones. The UI is available in Japanese and English.
- **Ready-to-use presets**: Pick a preset for your use case — transcription only, with translation, or all the way to text-to-speech — and start right away.

## What it can do

- **Connect**: Send recognition, translation, and speech results to YNC NEO for stream subtitles and integration with other tools. Sending can be toggled automatically based on your VRChat mute state.
- **NC (noise cancellation)**: Suppresses noise from your microphone environment so recognition works on clean audio.
- **VAD / Turn Detector**: Detects whether you are speaking and decides where utterances end.
- **ASR (speech recognition)**: Transcribes your speech from the microphone in real time. A dedicated streaming ASR model can be combined for interim display.
- **MT (translation)**: Translates recognized sentences into another language. Alongside the YNC NEO translation plugin, you can use Parapper's built-in local translation models (Japanese⇔English). Local translation can also be exposed to other apps as an OpenAI-compatible localhost API.
- **TTS (text-to-speech)**: Reads out recognition and translation results.
- **More**: Save and switch per-use-case setting presets, keep recognition history, and review captured audio with the logging features.

## Supported models

When you pick a preset on first launch, the required models are downloaded automatically. Models are stored in the app's data directory and can be used offline afterwards.

- **ASR**: Japanese (ReazonSpeech / NeMo Parakeet TDT CTC) / English / multilingual (25 European languages including English)
- **Streaming ASR (for interim display)**: English (Nemotron Speech Streaming) / multilingual (Nemotron 3.5 ASR Streaming, 29 languages including Japanese and English) (optional)
- **VAD**: Speech segment detection
- **Turn Detector**: Utterance-completion models for Japanese / English / multilingual (optional)
- **Japanese morphological dictionary**: Japanese grammar boundary detection for Morph / Namo
- **Local translation**: Japanese⇔English translation model (LFM2-350M-ENJP-MT ONNX Community Q4) (optional)
- **Noise cancellation**: Lightweight NC model (optional)
- **Local TTS**: Multilingual speech synthesis models including Japanese and English (optional)

For model names, supported languages, and sizes, see [documents/how-to-use.md](./how-to-use.md) (Japanese).

## Installation

### Windows

Download and run the latest `.msi` installer from the [Releases](https://github.com/Parakeet-Inc/Parapper-ASR/releases) page.

### Mac

Download and extract the latest `.zip` file from the [Releases](https://github.com/Parakeet-Inc/Parapper-ASR/releases) page, then run the app.

> YNC NEO integration (subtitle sending, translation plugin, speech plugin) is Windows-only. On macOS, use local translation and local TTS for translation and speech output.

## Usage

1. Launch the app. The onboarding screen opens on first launch.
2. Choose the display language in `UIの使用言語 / UI language`.
3. Pick your first workflow in `Config preset` and press "Apply and download models".
   - The ASR, VAD, Turn Detector, noise cancellation, and TTS models required by the preset are downloaded.
   - Note: using the Japanese model (ReazonSpeech K2 v2) at `int8-fp32` downloads about 170 MB including the VAD model. This may take a while depending on your connection.
4. Select the input device on the main screen and check the input level.
5. Open the settings panel as needed and adjust `Connect` / `NC` / `VAD` / `ASR` / `MT` / `TTS` / `Other` / `License`.
6. Press "Start" to begin recognition.

For details on the Turn Detector, YNC NEO, and VRChat integration, see [documents/how-to-use.md](./how-to-use.md) (Japanese).

## For developers

For build instructions, distribution steps, and model details, see [documents/developer/development-help.md](./developer/development-help.md) (Japanese).

For external integrations, see the [developer documentation](./developer/README.md), [Streaming Recognition Protocol v1](./developer/streaming-recognition-protocol-v1.md), and [security notes](./developer/security.md).

## Credits in streams and videos

This software is released under the MIT License.

If you use it in streams, videos, and similar content, a credit mention would be greatly appreciated and keeps us motivated.

## Related product: Paravo

[Paravo](https://parakeet-inc.com/paravo) is a lightweight, high-quality real-time AI voice changer developed by Parakeet Inc. Like Parapper, it runs on CPU only, with a focus on low latency and easy integration into streaming, gaming, and VRChat.

Combine them — "subtitles by Parapper, voice by Paravo" — to build a real-time streaming setup that runs entirely on CPU. See the [Paravo official page](https://parakeet-inc.com/paravo) for details.

## Licenses

- [Parapper](../LICENSE): MIT
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
