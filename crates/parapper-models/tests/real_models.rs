#![cfg(feature = "real-asr-tests")]
// Checked-in model fixtures use small frame/token values and f32 transcript timestamps.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    fs,
    path::{Path, PathBuf},
};

use parapper_models::asr::{
    AsrEngine, AsrLanguage, AsrModel, AsrPrecision, AsrSpeechRangeSamples, AsrStreamConfig,
    AsrTranscript, StreamingSessionId,
    backend::{
        REAZON_STATIC_EMBEDDING_DIR_NAME, REAZON_STATIC_EMBEDDING_REQUIRED_FILES,
        direct_ort::NvidiaCtcOrtAsrEngine,
        direct_tdt::{NvidiaTdtOrtAsrEngine, TdtDecodingStrategy},
        nemotron_ort::NemotronOrtAsrEngine,
        reazon_ort::{ReazonDecodingStrategy, ReazonSpeechOrtAsrEngine},
    },
    decoder::rnnt::StatelessRnntSearchProfile,
};

fn models_root() -> PathBuf {
    std::env::var_os("PARAPPER_MODELS_ROOT").map_or_else(
        || {
            PathBuf::from(
                std::env::var_os("APPDATA")
                    .expect("APPDATA or PARAPPER_MODELS_ROOT is required for real ASR tests"),
            )
            .join("com.parakeet-inc.parapper")
            .join("models")
        },
        PathBuf::from,
    )
}

fn japanese_wave() -> Vec<f32> {
    let path = models_root()
        .join("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11")
        .join("test_wavs")
        .join("ja.wav");
    let (samples, sample_rate) = read_pcm16_wav_mono_f32(&path);
    assert_eq!(sample_rate, 16_000);
    assert_eq!(samples.len(), 115_200);
    samples
}

fn english_wave() -> Vec<f32> {
    let path = models_root()
        .join("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11")
        .join("test_wavs")
        .join("en.wav");
    let (samples, sample_rate) = read_pcm16_wav_mono_f32(&path);
    assert_eq!(sample_rate, 16_000);
    samples
}

fn read_pcm16_wav_mono_f32(path: &Path) -> (Vec<f32>, u32) {
    let bytes = fs::read(path).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", path.display());
    });
    assert!(bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut cursor = 12;
    let mut format = None;
    let mut data = None;
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;
        let end = cursor.checked_add(size).unwrap();
        assert!(
            end <= bytes.len(),
            "invalid WAV chunk in {}",
            path.display()
        );
        if id == b"fmt " {
            assert!(size >= 16);
            format = Some((
                u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()),
                u16::from_le_bytes(bytes[cursor + 2..cursor + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()),
                u16::from_le_bytes(bytes[cursor + 14..cursor + 16].try_into().unwrap()),
            ));
        } else if id == b"data" {
            data = Some(cursor..end);
        }
        cursor = end + size % 2;
    }
    let (encoding, channels, sample_rate, bits) = format.expect("missing WAV fmt chunk");
    assert_eq!((encoding, bits), (1, 16), "test WAV must be PCM16");
    let data = data.expect("missing WAV data chunk");
    let frame_bytes = usize::from(channels) * 2;
    let samples = bytes[data]
        .chunks_exact(frame_bytes)
        .map(|frame| {
            (0..usize::from(channels))
                .map(|channel| {
                    let offset = channel * 2;
                    f32::from(i16::from_le_bytes(
                        frame[offset..offset + 2].try_into().unwrap(),
                    )) / 32_768.0
                })
                .sum::<f32>()
                / f32::from(channels)
        })
        .collect();
    (samples, sample_rate)
}

struct StagedModelDir(PathBuf);

impl std::ops::Deref for StagedModelDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for StagedModelDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn stage_reazon_accuracy_model() -> StagedModelDir {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after Unix epoch")
        .as_nanos();
    let staged = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/real-asr-tests")
        .join(format!("reazon-one-splice-{unique}"));
    let source = models_root().join("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01");
    for file in [
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx",
        "tokens.txt",
    ] {
        let destination = staged.join(file);
        fs::create_dir_all(destination.parent().expect("model file has a parent"))
            .expect("failed to create staged Reazon model dir");
        fs::hard_link(source.join(file), destination)
            .expect("failed to hard-link downloaded Reazon model");
    }
    let snapshot = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../target/hf-cache/models--hotchpotch--static-embedding-japanese/snapshots/95b3d9c80a7ccf604e2b5daee7b1b3eed6b1a9d3",
    );
    for file in REAZON_STATIC_EMBEDDING_REQUIRED_FILES {
        let source = fs::canonicalize(snapshot.join(file))
            .expect("pinned static embedding snapshot should exist");
        let destination = staged.join(REAZON_STATIC_EMBEDDING_DIR_NAME).join(file);
        fs::create_dir_all(destination.parent().expect("reranker file has a parent"))
            .expect("failed to create staged reranker dir");
        fs::hard_link(source, destination).expect("failed to hard-link static reranker model");
    }
    StagedModelDir(staged)
}

#[test]
#[ignore = "requires downloaded Parakeet TDT v2/v3 and Nemotron archive test audio"]
fn direct_ort_tdt_v2_and_v3_decode_the_same_real_audio() {
    let wave = english_wave();
    for (model, directory, fixture_name) in [
        (
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
            "parakeet-tdt-v2-int8-greedy.json",
        ),
        (
            AsrModel::NemoParakeetTdt0_6BV3Int8,
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            "parakeet-tdt-v3-int8-greedy.json",
        ),
    ] {
        let mut engine = NvidiaTdtOrtAsrEngine::new(
            &models_root().join(directory),
            model,
            AsrPrecision::Int8,
            2,
        )
        .expect("direct ONNX Runtime TDT engine must load");
        let transcript = engine
            .recognize(&wave)
            .expect("direct ONNX Runtime TDT recognition must succeed");
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../diagnostics/nemo-reference/fixtures")
            .join(fixture_name);
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixture_path).expect("TDT fixture must be readable"),
        )
        .expect("TDT fixture must be JSON");
        let output = &fixture["output"];
        let token_ids = output["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let token_table = load_token_table(&models_root().join(directory).join("tokens.txt"));
        let expected_token_texts = token_ids
            .iter()
            .map(|&id| token_table[id].replace('▁', " "))
            .collect::<Vec<_>>();
        let expected_timestamps = output["timestamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as f32 * 0.08)
            .collect::<Vec<_>>();
        let mut expected_durations = output["token_durations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as f32 * 0.08)
            .collect::<Vec<_>>();
        if model == AsrModel::NemoParakeetTdt0_6BV2Int8 {
            // The first v2 duration is the only observed provider-build
            // numeric delta: the bundled production ORT selects 4 while the
            // official Python ORT 1.24.4 wheel selects 3. Tokens, timestamps,
            // and every subsequent duration are identical.
            expected_durations[0] = 0.32;
        }

        assert_eq!(transcript.text, output["text"].as_str().unwrap());
        assert_eq!(
            transcript
                .tokens
                .iter()
                .map(|token| token.text.clone())
                .collect::<Vec<_>>(),
            expected_token_texts
        );
        assert_eq!(
            transcript
                .tokens
                .iter()
                .map(|token| token.start_sec.unwrap())
                .collect::<Vec<_>>(),
            expected_timestamps
        );
        assert_eq!(
            transcript
                .tokens
                .iter()
                .map(|token| token.duration_sec.unwrap())
                .collect::<Vec<_>>(),
            expected_durations
        );
    }
}

fn load_token_table(path: &std::path::Path) -> Vec<String> {
    let mut tokens = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let (token, raw_id) = line.rsplit_once(' ').unwrap();
        let id = raw_id.parse::<usize>().unwrap();
        if tokens.len() <= id {
            tokens.resize(id + 1, String::new());
        }
        tokens[id] = token.to_string();
    }
    tokens
}

#[test]
#[ignore = "performance comparison: requires Parakeet TDT v2 and real audio"]
fn direct_ort_tdt_default_beam_two_preserves_the_real_audio_text() {
    let model = AsrModel::NemoParakeetTdt0_6BV2Int8;
    let model_dir = models_root().join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8");
    let wave = english_wave();
    let mut engine = NvidiaTdtOrtAsrEngine::new_with_decoding(
        &model_dir,
        model,
        AsrPrecision::Int8,
        2,
        TdtDecodingStrategy::DefaultBeam { beam_size: 2 },
    )
    .expect("direct TDT beam engine must load");
    let started = std::time::Instant::now();
    let transcript = engine.recognize(&wave).expect("beam must decode");
    eprintln!("direct TDT v2 beam=2 elapsed={:?}", started.elapsed());
    assert_eq!(
        transcript.text,
        "The tribal chieftain called for the boy, and presented him with fifty pieces of gold."
    );
    assert!(
        transcript
            .tokens
            .iter()
            .all(|token| token.duration_sec.is_none()),
        "pinned NVIDIA default beam does not preserve token durations"
    );
}

#[test]
#[ignore = "requires downloaded Parakeet TDT CTC JA and Nemotron archive test audio"]
fn direct_ort_ctc_matches_the_pinned_nvidia_text_and_timestamp_contract() {
    let model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
    let model_dir = models_root().join("sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8");
    let wave = japanese_wave();
    let mut engine = NvidiaCtcOrtAsrEngine::new(&model_dir, model, AsrPrecision::Int8, 2)
        .expect("direct ONNX Runtime CTC engine must load without a Tauri host");

    let transcript = engine
        .recognize(&wave)
        .expect("direct ONNX Runtime CTC recognition must succeed");

    assert_eq!(
        transcript.text,
        "うちの中学は弁当制で持っていけない場合は50円の学校販売のパンを買う。"
    );
    assert_token_contract(
        &transcript,
        &[
            " ", "うち", "の", "中", "学", "は", "弁", "当", "制", "で", "持", "って", "い", "け",
            "ない", "場合", "は", "50", "円", "の", "学校", "販", "売", "の", "パン", "を", "買",
            "う", "。",
        ],
        &[
            0.0, 0.24, 0.4, 0.64, 0.88, 1.04, 1.28, 1.44, 1.68, 1.92, 2.24, 2.4, 2.56, 2.72, 2.88,
            3.04, 3.36, 3.6, 3.92, 4.16, 4.32, 4.72, 4.88, 5.12, 5.28, 5.6, 5.76, 5.92, 7.12,
        ],
        &[
            None,
            Some(0..2),
            Some(2..3),
            Some(3..4),
            Some(4..5),
            Some(5..6),
            Some(6..7),
            Some(7..8),
            Some(8..9),
            Some(9..10),
            Some(10..11),
            Some(11..13),
            Some(13..14),
            Some(14..15),
            Some(15..17),
            Some(17..19),
            Some(19..20),
            Some(20..22),
            Some(22..23),
            Some(23..24),
            Some(24..26),
            Some(26..27),
            Some(27..28),
            Some(28..29),
            Some(29..31),
            Some(31..32),
            Some(32..33),
            Some(33..34),
            Some(34..35),
        ],
    );
}

#[test]
#[ignore = "requires downloaded ReazonSpeech and Nemotron archive test audio"]
fn direct_ort_reazonspeech_matches_the_pinned_icefall_onnx_contract() {
    let model = AsrModel::ReazonSpeechK2V2;
    let model_dir = models_root().join("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01");
    let wave = japanese_wave();
    let mut direct = ReazonSpeechOrtAsrEngine::new(&model_dir, model, AsrPrecision::Int8, 2)
        .expect("direct ReazonSpeech engine must load");
    let transcript = direct.recognize(&wave).expect("direct recognition");
    let encoded = direct
        .encode(&wave)
        .expect("the reusable encoder output must be produced");
    let split_transcript = direct
        .decode_encoded(&encoded, ReazonDecodingStrategy::Greedy)
        .expect("the reusable encoder output must support greedy decoding");

    assert_eq!(split_transcript, transcript);

    assert_eq!(
        transcript.text,
        "うちの中学は弁当制で持っていけない場合は五十円の学校販売のパンを買う"
    );
    assert_eq!(
        transcript
            .tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>(),
        [
            "う", "ち", "の", "中", "学", "は", "弁", "当", "制", "で", "持", "っ", "て", "い",
            "け", "な", "い", "場", "合", "は", "五", "十", "円", "の", "学", "校", "販", "売",
            "の", "パ", "ン", "を", "買", "う",
        ]
    );
    assert_eq!(
        transcript
            .tokens
            .iter()
            .map(|token| (token.start_sec.unwrap() / 0.04).round() as u32)
            .collect::<Vec<_>>(),
        [
            0, 7, 10, 17, 22, 27, 33, 37, 43, 47, 53, 56, 57, 62, 65, 69, 71, 78, 81, 84, 92, 96,
            101, 105, 114, 118, 125, 128, 132, 141, 145, 149, 153, 173,
        ]
    );
    assert_eq!(
        transcript
            .tokens
            .iter()
            .map(|token| token.char_range.clone().unwrap())
            .collect::<Vec<_>>(),
        (0..34).map(|index| index..index + 1).collect::<Vec<_>>()
    );
    assert!(
        direct
            .start_stream(StreamingSessionId::new(1, None), AsrStreamConfig::default())
            .is_err(),
        "ReazonSpeech is an offline model"
    );
}

#[test]
#[ignore = "requires downloaded ReazonSpeech and Nemotron archive test audio"]
fn one_reazon_encoding_supports_greedy_and_multiple_state_beam_widths() {
    let model = AsrModel::ReazonSpeechK2V2;
    let model_dir = models_root().join("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01");
    let wave = japanese_wave();
    let mut direct = ReazonSpeechOrtAsrEngine::new_with_decoding(
        &model_dir,
        model,
        AsrPrecision::Float32,
        2,
        ReazonDecodingStrategy::ModifiedBeam { beam_size: 8 },
    )
    .expect("direct ReazonSpeech engine must load");
    let encoded = direct
        .encode(&wave)
        .expect("the utterance must be encoded once");

    let texts = [
        ReazonDecodingStrategy::Greedy,
        ReazonDecodingStrategy::ModifiedBeam { beam_size: 2 },
        ReazonDecodingStrategy::ModifiedBeam { beam_size: 4 },
        ReazonDecodingStrategy::ModifiedBeam { beam_size: 8 },
    ]
    .into_iter()
    .map(|strategy| {
        direct
            .decode_encoded(&encoded, strategy)
            .expect("every configured search must decode the same encoder output")
            .text
    })
    .collect::<Vec<_>>();

    assert_eq!(
        texts,
        ["うちの中学は弁当制で持っていけない場合は五十円の学校販売のパンを買う"; 4]
    );
}

#[test]
#[ignore = "requires downloaded ReazonSpeech and Nemotron archive test audio"]
fn production_reazon_beam_uses_dynamic_ort_top_k_and_gather_across_widths() {
    let model = AsrModel::ReazonSpeechK2V2;
    let model_dir = models_root().join("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01");
    let wave = japanese_wave();
    let strategy = ReazonDecodingStrategy::ModifiedBeam { beam_size: 4 };
    let mut direct = ReazonSpeechOrtAsrEngine::new_with_decoding(
        &model_dir,
        model,
        AsrPrecision::Int8,
        2,
        strategy,
    )
    .expect("production ReazonSpeech beam engine must load");
    let encoded = direct.encode(&wave).expect("encoder output");
    for beam_size in [2, 4, 8] {
        let mut profile = StatelessRnntSearchProfile::default();
        let transcript = direct
            .decode_encoded_with_search_profile(
                &encoded,
                ReazonDecodingStrategy::ModifiedBeam { beam_size },
                &mut profile,
            )
            .expect("production beam decoding");

        assert_eq!(
            transcript.text,
            "うちの中学は弁当制で持っていけない場合は五十円の学校販売のパンを買う"
        );
        assert_eq!(profile.log_softmax, std::time::Duration::ZERO);
        assert_eq!(profile.scalar_exp_terms_evaluated, 0);
        assert_eq!(profile.top_token_selection, std::time::Duration::ZERO);
        assert!(profile.network_output_bytes < profile.logit_values * std::mem::size_of::<f32>());
    }
}

#[test]
#[ignore = "requires downloaded ReazonSpeech, pinned static embedding, and Japanese test audio"]
fn app_reazon_accuracy_strategy_runs_width_four_one_splice_with_retained_width_two() {
    let model_dir = stage_reazon_accuracy_model();
    let mut engine = ReazonSpeechOrtAsrEngine::new_with_decoding(
        &model_dir,
        AsrModel::ReazonSpeechK2V2,
        AsrPrecision::Float32,
        2,
        ReazonDecodingStrategy::OneSpliceRerank {
            beam_size: 4,
            retained_candidates: 2,
        },
    )
    .expect("app Reazon accuracy engine should load");

    let transcript = engine
        .recognize(&japanese_wave())
        .expect("app Reazon accuracy strategy should decode real audio");

    assert_eq!(
        transcript.text,
        "うちの中学は弁当制で持っていけない場合は五十円の学校販売のパンを買う"
    );
    assert_eq!(transcript.tokens.len(), transcript.text.chars().count());
}

#[test]
#[ignore = "requires downloaded Nemotron 3.5 160ms model and archive test audio"]
fn direct_ort_nemotron_streaming_emits_deltas_finishes_and_resets_a_real_session() {
    for (index, model, directory, audio_name, fixture_name) in [
        (
            0,
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
            "0.wav",
            "nemotron-en-160-int8-streaming.json",
        ),
        (
            1,
            AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
            "0.wav",
            "nemotron-en-560-int8-streaming.json",
        ),
        (
            2,
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11",
            "ja.wav",
            "nemotron-3.5-160-int8-streaming.json",
        ),
        (
            3,
            AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
            "ja.wav",
            "nemotron-3.5-560-int8-streaming.json",
        ),
    ] {
        let model_dir = models_root().join(directory);
        let wave_path = model_dir.join("test_wavs").join(audio_name);
        let (wave, sample_rate) = read_pcm16_wav_mono_f32(&wave_path);
        assert_eq!(sample_rate, 16_000);
        let mut engine = NemotronOrtAsrEngine::new(&model_dir, model, AsrPrecision::Int8, 2)
            .expect("direct Nemotron engine must load without Tauri");
        let session = StreamingSessionId::new(43 + index, Some(8));
        let (deltas, final_transcript) = recognize_stream(&mut engine, session, &wave);
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../diagnostics/nemo-reference/fixtures")
            .join(fixture_name);
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture_path).unwrap()).unwrap();
        let provider_text = |text: &str| {
            if model == AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8 {
                // Official and bundled ORT 1.24.4 builds disagree at one
                // near-tied logit. The bundled production build emits the
                // acoustically correct 制; the official Python wheel emits 性.
                text.replace('性', "制")
            } else {
                text.to_string()
            }
        };
        assert_eq!(
            deltas
                .iter()
                .map(|value| value.text.clone())
                .collect::<Vec<_>>(),
            fixture["output"]["partials"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| provider_text(value.as_str().unwrap()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            final_transcript.text,
            provider_text(fixture["output"]["text"].as_str().unwrap())
        );
        let mut expected_timestamps = fixture["output"]["timestamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as f32 * 0.08)
            .collect::<Vec<_>>();
        if model == AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8 {
            // A second provider-build tie changes 買 by one frame only.
            expected_timestamps[32] = 5.92;
        }
        assert_eq!(
            final_transcript
                .tokens
                .iter()
                .map(|token| token.start_sec.unwrap())
                .collect::<Vec<_>>(),
            expected_timestamps
        );
        assert_stream_can_restart(&mut engine, session, &wave);
    }
}

#[test]
#[ignore = "requires the latest downloaded 560ms Nemotron archives"]
fn one_base_archive_runs_every_nemotron_latency_and_cache_shape() {
    for (index, model, directory) in [
        (
            0,
            AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        ),
        (
            1,
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        ),
        (
            2,
            AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        ),
        (
            3,
            AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        ),
        (
            4,
            AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
        ),
        (
            5,
            AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        ),
        (
            6,
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        ),
        (
            7,
            AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        ),
        (
            8,
            AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        ),
        (
            9,
            AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        ),
    ] {
        let model_dir = models_root().join(directory);
        let mut engine = NemotronOrtAsrEngine::new(&model_dir, model, AsrPrecision::Int8, 2)
            .unwrap_or_else(|error| panic!("{model:?} contract must load: {error:#}"));
        let session = StreamingSessionId::new(90 + index, None);
        let audio = vec![0.0; 20_000];
        engine
            .start_stream(
                session,
                AsrStreamConfig {
                    speech_range_samples: Some(AsrSpeechRangeSamples {
                        start: 0,
                        end: audio.len(),
                    }),
                    language_hint: Some(AsrLanguage::Japanese),
                },
            )
            .unwrap();
        engine
            .push_stream(session, &audio)
            .expect("latency-adjusted Nemotron graph must run streaming audio");
        engine
            .finish_stream(session)
            .expect("latency-adjusted Nemotron graph must finish");
    }
}

#[test]
#[ignore = "requires the latest downloaded 160ms and 560ms Nemotron archives"]
fn patched_560ms_archive_matches_the_official_160ms_graph() {
    for (index, model, official_directory, base_directory, audio_name) in [
        (
            0,
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
            "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
            "0.wav",
        ),
        (
            1,
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11",
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
            "ar.wav",
        ),
    ] {
        let official_dir = models_root().join(official_directory);
        let base_dir = models_root().join(base_directory);
        let (wave, sample_rate) =
            read_pcm16_wav_mono_f32(&official_dir.join("test_wavs").join(audio_name));
        assert_eq!(sample_rate, 16_000);

        let mut official = NemotronOrtAsrEngine::new(&official_dir, model, AsrPrecision::Int8, 2)
            .expect("official 160ms graph must load");
        let mut patched = NemotronOrtAsrEngine::new(&base_dir, model, AsrPrecision::Int8, 2)
            .expect("560ms graph must support the 160ms runtime contract");
        let official_output = recognize_stream(
            &mut official,
            StreamingSessionId::new(110 + index, None),
            &wave,
        );
        let patched_output = recognize_stream(
            &mut patched,
            StreamingSessionId::new(120 + index, None),
            &wave,
        );

        assert_eq!(patched_output, official_output);
    }
}

fn recognize_stream(
    engine: &mut dyn AsrEngine,
    session: StreamingSessionId,
    samples: &[f32],
) -> (Vec<AsrTranscript>, AsrTranscript) {
    engine
        .start_stream(session, AsrStreamConfig::default())
        .expect("session must start");
    assert!(
        engine
            .start_stream(session, AsrStreamConfig::default())
            .is_err(),
        "duplicate start must fail"
    );
    let deltas = samples
        .chunks(2_560)
        .map(|chunk| {
            engine
                .push_stream(session, chunk)
                .expect("chunk must decode")
        })
        .collect();
    let final_transcript = engine.finish_stream(session).expect("session must finish");
    (deltas, final_transcript)
}

fn assert_stream_can_restart(
    engine: &mut dyn AsrEngine,
    session: StreamingSessionId,
    samples: &[f32],
) {
    engine
        .start_stream(session, AsrStreamConfig::default())
        .expect("session must restart");
    engine
        .push_stream(session, &samples[..2_560])
        .expect("restarted session must accept audio");
    engine.cancel_stream(session);
    assert!(
        engine.push_stream(session, &samples[..2_560]).is_err(),
        "cancelled session must reject more audio until restarted"
    );
    engine
        .start_stream(session, AsrStreamConfig::default())
        .expect("cancelled session must restart");
    engine.cancel_stream(session);
}

fn assert_token_contract(
    transcript: &AsrTranscript,
    expected_texts: &[&str],
    expected_start_seconds: &[f32],
    expected_ranges: &[Option<std::ops::Range<usize>>],
) {
    assert_eq!(
        transcript
            .tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>(),
        expected_texts
    );
    assert_eq!(
        transcript
            .tokens
            .iter()
            .map(|token| token
                .start_sec
                .expect("baseline token must have a timestamp"))
            .collect::<Vec<_>>(),
        expected_start_seconds
    );
    assert!(
        transcript
            .tokens
            .iter()
            .all(|token| token.duration_sec.is_none())
    );
    assert_eq!(
        transcript
            .tokens
            .iter()
            .map(|token| token.char_range.clone())
            .collect::<Vec<_>>(),
        expected_ranges
    );
}
