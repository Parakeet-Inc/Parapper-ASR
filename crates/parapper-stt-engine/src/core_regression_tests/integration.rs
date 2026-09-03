use super::*;

// These cross-boundary regressions intentionally complement the focused unit tests. They keep
// the production producer sequence (SegmentStarted -> SegmentExtended -> timeout check ->
// SegmentClosed) visible through request scheduling, result application, and final output.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

fn replay_vad_frames_for_runtime(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    frames: impl IntoIterator<Item = (Vec<f32>, VadResult)>,
) {
    runtime.update_runtime_parameters(config.runtime_parameters());
    for (samples, vad_result) in frames {
        runtime.push_vad_frame(&samples, vad_result);
        runtime.step();
    }
}

fn test_env_path(key: &str) -> PathBuf {
    std::env::var_os(key).map_or_else(
        || {
            control_test_dotenv().get(key).map_or_else(
                || {
                    panic!(
                        "{key} must be set in the process environment or a local .env file for this diagnostic test"
                    )
                },
                PathBuf::from,
            )
        },
        PathBuf::from,
    )
}

fn control_test_dotenv() -> &'static HashMap<String, String> {
    static ENV: OnceLock<HashMap<String, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        control_test_dotenv_paths()
            .into_iter()
            .find_map(|path| path.is_file().then(|| parse_dotenv_file(&path)))
            .unwrap_or_default()
    })
}

fn control_test_dotenv_paths() -> [PathBuf; 2] {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("engine crate should be nested under the workspace crates directory");
    [workspace_root.join(".env"), manifest_dir.join(".env")]
}

fn parse_dotenv_file(path: &Path) -> HashMap<String, String> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    parse_dotenv_contents(&contents)
}

fn parse_dotenv_contents(contents: &str) -> HashMap<String, String> {
    contents
        .lines()
        .filter_map(parse_dotenv_line)
        .collect::<HashMap<_, _>>()
}

fn parse_dotenv_line(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let line = line.strip_prefix("export ").unwrap_or(line);
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_string(), unquote_dotenv_value(value.trim())))
}

fn unquote_dotenv_value(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(value)
        .to_string()
}

struct JvsPart {
    id: String,
    text: String,
    samples: Vec<f32>,
    sample_rate: u32,
}

struct FleursPart {
    locale: String,
    wav_path: PathBuf,
    samples: Vec<f32>,
    sample_rate: u32,
}

fn read_jvs_nonparallel_part(id: &str) -> JvsPart {
    let jvs_root = test_env_path("JVS_ROOT");
    let nonpara = jvs_root.join("jvs001").join("nonpara30");
    assert!(
        nonpara.is_dir(),
        "JVS nonparallel directory does not exist: {}",
        nonpara.display()
    );
    let text = read_jvs_transcript(&nonpara.join("transcripts_utf8.txt"), id);
    let wav = read_pcm16_wav_mono_f32(&nonpara.join("wav24kHz16bit").join(format!("{id}.wav")));
    JvsPart {
        id: id.to_string(),
        text,
        samples: wav.0,
        sample_rate: wav.1,
    }
}

fn read_jvs_transcript(path: &Path, id: &str) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()))
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find_map(|(line_id, text)| (line_id == id).then(|| text.to_string()))
        .unwrap_or_else(|| panic!("JVS transcript id {id} was not found in {}", path.display()))
}

fn read_short_fleurs_dev_parts(locale: &str, count: usize) -> Vec<FleursPart> {
    let fleurs_root = test_env_path("FLEURS_R_ROOT");
    let split_dir = fleurs_root.join(locale).join("dev").join("dev");
    assert!(
        split_dir.is_dir(),
        "FLEURS-R dev wav directory does not exist: {}",
        split_dir.display()
    );
    let mut wav_paths = fs::read_dir(&split_dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", split_dir.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        })
        .collect::<Vec<_>>();
    wav_paths.sort_by_key(|path| {
        fs::metadata(path)
            .unwrap_or_else(|err| panic!("failed to stat {}: {err}", path.display()))
            .len()
    });

    wav_paths
        .into_iter()
        .take(count)
        .map(|wav_path| read_fleurs_part(locale, wav_path))
        .collect()
}

fn read_fleurs_part(locale: &str, wav_path: PathBuf) -> FleursPart {
    let (samples, sample_rate) = read_pcm16_wav_mono_f32(&wav_path);
    let samples = resample_linear_for_test(&samples, sample_rate, crate::SAMPLE_RATE_HZ);
    FleursPart {
        locale: locale.to_string(),
        wav_path,
        samples,
        sample_rate: crate::SAMPLE_RATE_HZ,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test WAV resampling converts bounded sample positions between integer indices and fractional interpolation weights"
)]
fn resample_linear_for_test(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return samples.to_vec();
    }
    let target_len = (samples.len() as u128 * u128::from(target_rate))
        .div_ceil(u128::from(source_rate)) as usize;
    (0..target_len)
        .map(|index| {
            let position = index as f64 * f64::from(source_rate) / f64::from(target_rate);
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len().saturating_sub(1));
            let fraction = (position - left as f64) as f32;
            samples.get(left).copied().unwrap_or(0.0) * (1.0 - fraction)
                + samples.get(right).copied().unwrap_or(0.0) * fraction
        })
        .collect()
}

fn read_pcm16_wav_mono_f32(path: &Path) -> (Vec<f32>, u32) {
    let bytes =
        fs::read(path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    assert!(bytes.len() >= 12, "wav is too short: {}", path.display());
    assert_eq!(
        &bytes[0..4],
        b"RIFF",
        "wav must be RIFF: {}",
        path.display()
    );
    assert_eq!(
        &bytes[8..12],
        b"WAVE",
        "wav must be WAVE: {}",
        path.display()
    );

    let mut cursor = 12;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data = None;
    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = usize::try_from(u32::from_le_bytes(
            bytes[cursor + 4..cursor + 8]
                .try_into()
                .expect("chunk size should have 4 bytes"),
        ))
        .expect("wav chunk size should fit usize");
        cursor += 8;
        let chunk_end = cursor.saturating_add(chunk_size).min(bytes.len());
        match chunk_id {
            b"fmt " => {
                assert!(
                    chunk_size >= 16,
                    "fmt chunk is too small in {}",
                    path.display()
                );
                let audio_format = u16::from_le_bytes(
                    bytes[cursor..cursor + 2]
                        .try_into()
                        .expect("audio format should have 2 bytes"),
                );
                assert_eq!(audio_format, 1, "wav must be PCM: {}", path.display());
                channels = Some(u16::from_le_bytes(
                    bytes[cursor + 2..cursor + 4]
                        .try_into()
                        .expect("channels should have 2 bytes"),
                ));
                sample_rate = Some(u32::from_le_bytes(
                    bytes[cursor + 4..cursor + 8]
                        .try_into()
                        .expect("sample rate should have 4 bytes"),
                ));
                bits_per_sample = Some(u16::from_le_bytes(
                    bytes[cursor + 14..cursor + 16]
                        .try_into()
                        .expect("bits per sample should have 2 bytes"),
                ));
            }
            b"data" => {
                data = Some(cursor..chunk_end);
            }
            _ => {}
        }
        cursor = chunk_end + (chunk_size % 2);
    }

    let channels = channels.expect("wav fmt chunk should define channels");
    let channel_count = usize::from(channels);
    let sample_rate = sample_rate.expect("wav fmt chunk should define sample rate");
    assert_eq!(
        bits_per_sample,
        Some(16),
        "wav must be 16-bit PCM: {}",
        path.display()
    );
    let data = data.unwrap_or_else(|| panic!("wav data chunk not found: {}", path.display()));
    let frame_bytes = channel_count * 2;
    let mut samples = Vec::with_capacity((data.end - data.start) / frame_bytes);
    for frame in bytes[data].chunks_exact(frame_bytes) {
        let mut sum = 0.0_f32;
        for channel in 0..channel_count {
            let offset = channel * 2;
            let sample = i16::from_le_bytes(
                frame[offset..offset + 2]
                    .try_into()
                    .expect("PCM16 sample should have 2 bytes"),
            );
            sum += f32::from(sample) / 32768.0;
        }
        samples.push(sum / f32::from(channels));
    }
    (samples, sample_rate)
}

fn push_jvs_speech_chunks(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    samples: &[f32],
    sample_rate: u32,
) {
    let chunk_len = frames_for_millis(sample_rate, config.segmentation.vad_interval_ms);
    for chunk in samples.chunks(chunk_len) {
        runtime.push_vad_frame(chunk, vad(true));
        runtime.step();
    }
}

fn push_fleurs_speech_chunks(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    part: &FleursPart,
) {
    push_jvs_speech_chunks(runtime, config, &part.samples, part.sample_rate);
}

fn push_silence_chunks(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    sample_rate: u32,
    chunks: usize,
) {
    let chunk_len = frames_for_millis(sample_rate, config.segmentation.vad_interval_ms);
    let silence = vec![0.0; chunk_len];
    for _ in 0..chunks {
        runtime.push_vad_frame(&silence, vad(false));
        runtime.step();
    }
}

fn frames_for_millis(sample_rate: u32, millis: u32) -> usize {
    usize::try_from((u64::from(sample_rate) * u64::from(millis)).div_ceil(1000))
        .expect("test sample count should fit usize")
}

fn assert_output_phrase_contains_jvs_parts(output: &PhraseOutputSnapshot, parts: &[JvsPart]) {
    let mut search_from = 0;
    for part in parts {
        assert!(
            output.text.contains(part.text.trim_end_matches('。'))
                || output.text.contains(&part.text),
            "UI text for turn {} segment {} should keep JVS part {} visible\ntext: {}",
            output.turn_id,
            output.segment_id,
            part.id,
            output.text
        );
        let fingerprint = jvs_audio_fingerprint(part);
        let position = find_subsequence_approx(&output.phrase, &fingerprint, search_from)
                .unwrap_or_else(|| {
                    panic!(
                        "UI phrase audio for turn {} segment {} is missing JVS part {} ({} samples); output phrase has {} samples",
                        output.turn_id,
                        output.segment_id,
                        part.id,
                        part.samples.len(),
                        output.phrase.len()
                    )
                });
        search_from = position + fingerprint.len();
    }
}

fn assert_output_phrase_contains_fleurs_parts(output: &PhraseOutputSnapshot, parts: &[FleursPart]) {
    let mut search_from = 0;
    for part in parts {
        for fingerprint in fleurs_audio_fingerprints(part) {
            let position = find_subsequence_approx(&output.phrase, &fingerprint, search_from)
                .unwrap_or_else(|| {
                    panic!(
                        "UI phrase audio for turn {} segment {} is missing FLEURS part {} ({}, {} samples); output phrase has {} samples",
                        output.turn_id,
                        output.segment_id,
                        part.locale,
                        part.wav_path.display(),
                        part.samples.len(),
                        output.phrase.len()
                    )
                });
            search_from = position + fingerprint.len();
        }
    }
}

fn jvs_audio_fingerprint(part: &JvsPart) -> Vec<f32> {
    let len = part.samples.len().min(2048);
    assert!(len > 0, "JVS part {} should not be empty", part.id);
    let start = part.samples.len() / 2 - len / 2;
    part.samples[start..start + len].to_vec()
}

fn fleurs_audio_fingerprints(part: &FleursPart) -> [Vec<f32>; 2] {
    [
        fleurs_audio_fingerprint_at(part, 1, 4),
        fleurs_audio_fingerprint_at(part, 1, 2),
    ]
}

fn fleurs_audio_fingerprint_at(
    part: &FleursPart,
    numerator: usize,
    denominator: usize,
) -> Vec<f32> {
    let len = part.samples.len().min(2048);
    assert!(
        len > 0,
        "FLEURS part {} should not be empty",
        part.wav_path.display()
    );
    let center = part.samples.len().saturating_mul(numerator) / denominator.max(1);
    let start = center
        .saturating_sub(len / 2)
        .min(part.samples.len().saturating_sub(len));
    part.samples[start..start + len].to_vec()
}

fn find_subsequence_approx(haystack: &[f32], needle: &[f32], start: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() || start > haystack.len() - needle.len() {
        return None;
    }
    (start..=haystack.len() - needle.len()).find(|&index| {
        haystack[index..index + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(left, right)| (left - right).abs() <= 1.0e-6)
    })
}

#[test]
fn turn_runtime_completion_request_preserves_production_sized_vad_frame_range() {
    const SILERO_CHUNK_SAMPLES: usize = 512;
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, SILERO_CHUNK_SAMPLES, true),
            fixed_vad_frame(0.0, SILERO_CHUNK_SAMPLES, false),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("a production-sized speech/silence pair should dispatch completion ASR");
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(
        dispatched.target.range.end_sample,
        GlobalSampleIndex((SILERO_CHUNK_SAMPLES * 2) as u64)
    );
    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("completion request should remain in flight");
    assert_eq!(request.audio.len(), SILERO_CHUNK_SAMPLES * 2);
    assert_eq!(request.vad_results.len(), 2);
}

#[test]
fn turn_runtime_enqueues_completion_asr_from_closed_segment() {
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("closed segment must enqueue and dispatch one ASR request");
    assert_eq!(dispatched.request_id, AsrRequestId(1));
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.turn_id, TurnId(1));
    assert_eq!(dispatched.target.turn_revision, TurnRevision(0));
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(dispatched.target.range.end_sample, GlobalSampleIndex(2));
    assert_eq!(dispatched.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
}

#[test]
fn turn_runtime_enqueues_interim_asr_without_finishing_turn() {
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("interim silence must enqueue and dispatch one interim ASR request");
    assert_eq!(dispatched.request_id, AsrRequestId(1));
    assert_eq!(dispatched.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(dispatched.target.turn_id, TurnId(1));
    assert_eq!(dispatched.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
}

#[test]
fn turn_runtime_nemotron_interim_dispatches_first_raw_source_delta_without_native_chunk_wait() {
    const FRAME_SAMPLES: usize = 256;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0; FRAME_SAMPLES], vad(true))],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    let request = &submitted[0];
    assert_eq!(request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        request.route,
        RecognitionRoute::from_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
    );
    assert_eq!(
        request.close_reason,
        Some(SegmentCloseReason::InterimChunkReached)
    );
    assert_eq!(
        request.audio.len(),
        FRAME_SAMPLES,
        "SourceRuntime must forward the raw 16 kHz delta; Nemotron owns native chunk buffering"
    );
    assert_eq!(request.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(
        request.target.range.end_sample,
        GlobalSampleIndex(FRAME_SAMPLES as u64)
    );
}

#[test]
fn turn_runtime_streaming_interim_ignores_silence_snapshot_request() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(320);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0; 256], vad(true)),
            (vec![0.0; 256], vad(false)),
            (vec![2.0; 256], vad(true)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(
        submitted[0].close_reason,
        Some(SegmentCloseReason::InterimChunkReached)
    );
    assert_eq!(submitted[0].audio, vec![1.0; 256]);
}

#[test]
fn turn_runtime_streaming_interim_silence_threshold_does_not_split_completion_request() {
    const FRAME_SAMPLES: usize = 256;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(64);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![1.0; FRAME_SAMPLES], vad(true))),
    );
    let streaming_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("first streaming interim request should be in flight");
    assert_eq!(streaming_request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        streaming_request.close_reason,
        Some(SegmentCloseReason::InterimChunkReached)
    );
    asr_handle.complete_request_with_text(&streaming_request, "途中");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![0.0; FRAME_SAMPLES], vad(false)))
            .chain(std::iter::once((vec![2.0; FRAME_SAMPLES], vad(true))))
            .chain((0..4).map(|_| (vec![0.0; FRAME_SAMPLES], vad(false)))),
    );

    while runtime
        .requests
        .in_flight_request
        .as_ref()
        .is_some_and(|request| request.kind == AsrTaskKind::InterimDisplay)
    {
        let request = runtime.requests.in_flight_request.clone().unwrap();
        asr_handle.complete_request_with_text(&request, "途中");
        runtime.step();
    }
    runtime.step();

    let completion = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("turn-check silence should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        completion.target.first_segment_id,
        Some(SegmentId(1)),
        "streaming interim must not let interim_result_silence_ms split the logical completion segment"
    );
    assert_eq!(
        completion.target.last_segment_id,
        Some(SegmentId(1)),
        "completion should still target the original segment instead of a silence-threshold child segment"
    );
    assert_eq!(
        completion.source_audio.len(),
        FRAME_SAMPLES * 7,
        "completion source audio should cover the whole utterance regardless of interim_result_silence_ms"
    );
}

#[test]
fn turn_runtime_streaming_interim_continues_across_interim_silence_without_duplicate_prespeech() {
    const FRAME_SAMPLES: usize = 256;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(320);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![1.0; FRAME_SAMPLES], vad(true))),
    );
    let first_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("first raw streaming interim delta should be in flight");
    asr_handle.complete_request_with_text(&first_request, "最初");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![0.0; FRAME_SAMPLES], vad(false)))
            .chain(std::iter::once((vec![2.0; FRAME_SAMPLES], vad(true)))),
    );

    let combined_delta_request = runtime.requests.in_flight_request.clone().expect(
        "the continued speech delta should be dispatched after the first interim completes",
    );
    asr_handle.complete_request_with_text(&combined_delta_request, "最初");
    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 2);
    let second_request = &submitted[1];
    assert_eq!(second_request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        second_request.close_reason,
        Some(SegmentCloseReason::InterimChunkReached)
    );
    assert_eq!(
        second_request.target.first_segment_id,
        Some(SegmentId(1)),
        "streaming interim should keep updating the logical utterance segment across interim-threshold silence"
    );
    assert_eq!(
        second_request.target.last_segment_id,
        Some(SegmentId(1)),
        "streaming interim output should replace the same draft segment instead of appending duplicate cumulative audio"
    );
    assert_eq!(
        second_request.audio,
        vec![0.0; FRAME_SAMPLES],
        "the source-side delta preserves real silence but does not synthesize pre-roll"
    );
    assert_eq!(second_request.source_audio.len(), FRAME_SAMPLES * 2);
    assert!(
        second_request.source_audio[..FRAME_SAMPLES]
            .iter()
            .all(|sample| sample.to_bits() == 1.0_f32.to_bits())
    );
    assert!(
        second_request.source_audio[FRAME_SAMPLES..FRAME_SAMPLES * 2]
            .iter()
            .all(|sample| sample.to_bits() == 0.0_f32.to_bits())
    );
}

#[test]
fn turn_runtime_end_silence_discards_queued_nemotron_streaming_interim_chunks() {
    const FRAME_SAMPLES: usize = 256;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .turn_check_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..20).map(|_| (vec![1.0; FRAME_SAMPLES], vad(true))),
    );

    assert!(
        runtime.requests.in_flight_request.is_some(),
        "the first Nemotron interim chunk should already be in flight"
    );
    assert!(
        runtime
            .pending
            .asr_segments
            .iter()
            .any(|segment| { segment.reason == SegmentCloseReason::InterimChunkReached }),
        "the second Nemotron interim chunk should be queued before the utterance closes"
    );

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..2).map(|_| (vec![0.0; FRAME_SAMPLES], vad(false))),
    );

    assert!(
        runtime
            .pending
            .asr_segments
            .iter()
            .all(|segment| { segment.reason != SegmentCloseReason::InterimChunkReached }),
        "queued Nemotron interim audio that was not submitted yet must be discarded when the utterance closes"
    );
    assert!(
        runtime
            .pending
            .asr_segments
            .iter()
            .any(|segment| { segment.reason == SegmentCloseReason::EndSilenceReached }),
        "the final completion candidate must remain queued after discarding interim-only audio"
    );
    assert_eq!(
        asr_handle.streaming_reset_count(),
        1,
        "closing the utterance must reset the Nemotron streaming cache before the next interim session"
    );
}

#[test]
fn turn_runtime_non_streaming_interim_keeps_silence_snapshot_request() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::NemoParakeetTdt0_6BV2Int8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(320);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0; 256], vad(true)),
            (vec![0.0; 256], vad(false)),
            (vec![2.0; 256], vad(true)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        submitted[0].close_reason,
        Some(SegmentCloseReason::InterimResultSilenceReached)
    );
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8)
    );
}

#[test]
fn turn_runtime_nemotron_interim_updates_same_segment_without_duplicating_turn_audio() {
    const FRAME_SAMPLES: usize = 256;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![1.0; FRAME_SAMPLES], vad(true))),
    );
    let first_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("the first raw Nemotron interim delta should be in flight");
    asr_handle.complete_request_with_text(&first_request, "あ");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![2.0; FRAME_SAMPLES], vad(true))),
    );
    runtime.step();
    let second_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("the next raw delta should dispatch another interim request");
    assert_eq!(second_request.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(second_request.target.last_segment_id, Some(SegmentId(1)));
    assert_eq!(
        second_request.audio.len(),
        FRAME_SAMPLES,
        "Nemotron streaming input must send only the next source delta to the ASR worker"
    );
    assert_eq!(
        second_request.source_audio.len(),
        FRAME_SAMPLES * 2,
        "Turn replacement must still keep the cumulative source audio for UI output"
    );
    asr_handle.complete_request_with_text(&second_request, "あいう");
    runtime.step();

    let outputs = outputs.lock().expect("phrase outputs should be readable");
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].text, "あ...");
    assert_eq!(outputs[0].phrase.len(), FRAME_SAMPLES);
    assert_eq!(outputs[1].text, "あいう...");
    assert_eq!(
        outputs[1].phrase.len(),
        FRAME_SAMPLES * 2,
        "same-segment interim updates must replace the previous source audio instead of appending duplicate audio"
    );
}

#[test]
fn turn_runtime_keeps_only_one_asr_request_in_flight() {
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
            (vec![0.0], vad(false)),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("first closed segment must dispatch");
    assert_eq!(dispatched.request_id, AsrRequestId(1));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
    runtime.step();
    assert!(
        runtime.take_last_dispatched().is_none(),
        "second closed segment must stay queued while the first ASR request is in flight"
    );
}

#[test]
fn turn_runtime_applies_interim_asr_result_to_output_sink() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .scripted_asr_texts(vec!["途中"]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "途中...".to_string(),
            is_final: false,
            turn_id: 1,
            segment_id: 1,
        }]
    );
}

#[test]
fn turn_runtime_interim_punctuation_does_not_run_grammar_finalization() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .scripted_asr_texts(vec!["はい。次です"]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("はい。次です...", false, 1, 1)],
        "interim display must not run Mecab/grammar finalization even if punctuation is present"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_multilingual_turn_check_after_interim_uses_sli_route_for_rerecognition() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .segment_start_speech_ms(64)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .turn_check_silence_ms(64)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0; 16_000], vad(true)),
            (vec![1.0; 512], vad(true)),
            (vec![0.0; 512], vad(false)),
            (vec![1.0; 512], vad(true)),
        ],
    );
    let interim_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("interim ASR should dispatch before turn-check silence");
    assert_eq!(interim_request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        interim_request.route,
        RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
        "interim display should keep the default route until turn-check SLI"
    );
    asr_handle.complete_request_with_text(&interim_request, "hello");
    runtime.step();

    replay_vad_frames_for_runtime(&mut runtime, &config, vec![(vec![0.0; 512], vad(false))]);

    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("turn-check silence should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        vec![17_024],
        "turn-check after interim must run SLI over the accumulated turn audio before rerecognition"
    );
    assert_eq!(
        rerecognition.route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
        "full-turn rerecognition must switch to the SLI-selected English route"
    );
}

#[test]
fn turn_runtime_namo_completion_and_rerecognition_elapsed_millis_are_accumulated() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8)
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅"]);
    let asr_handle = builder.use_manual_asr();
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_request_with_text_elapsed(&completion, "句読点つき。", 41);

    runtime.step();

    let rerecognition =
        runtime.requests.in_flight_request.clone().expect(
            "Namo completion must dispatch full-turn rerecognition even if punctuation exists",
        );
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        rerecognition.route.model,
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8
    );
    asr_handle.complete_request_with_text_elapsed(&rerecognition, "再認識後。", 59);

    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "再認識後。");
    assert!(outputs[0].is_final);
    assert_eq!(
        outputs[0].elapsed_millis, 100,
        "final output should report completion ASR plus rerecognition ASR elapsed time"
    );
}

#[test]
fn turn_runtime_suppresses_late_interim_when_turn_check_already_reached() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty(),
        "silence that reaches turn-check must dispatch completion without first showing interim"
    );

    asr_handle.complete_next_with_text("hello");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new(),
        "completion ASR must wait for full-turn rerecognition before final output"
    );
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::Rerecognition);

    asr_handle.complete_next_with_text("hello");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("hello。", true, 1, 1)]
    );
}

#[test]
fn turn_runtime_interim_silence_does_not_emit_final_before_final_asr_result() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::InterimDisplay);

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です...", false, 1, 1)]
    );

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です...", false, 1, 1)],
        "turn-check silence must not finalize text from the interim ASR result"
    );

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です...", false, 1, 1)],
        "completion-check ASR must still wait for rerecognition before final output"
    );

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![
            output_snapshot("五月五日はこどもの日です...", false, 1, 1),
            output_snapshot("五月五日はこどもの日です。", true, 1, 2),
        ]
    );
}

#[test]
fn turn_runtime_interim_display_asr_pads_edges_without_persisting_padding() {
    const CHUNK: usize = 512;
    const EDGE_CHUNKS: usize = 10;
    const FADE_SAMPLES: usize = 160;
    let builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, CHUNK, true),
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(2.0, CHUNK, true),
        ],
    );
    let interim = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("speech after interim silence should dispatch interim ASR");
    assert_eq!(interim.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        interim.source_audio,
        [vec![1.0; CHUNK], vec![0.0; CHUNK]].concat(),
        "interim display must preserve only the observed source audio for the turn"
    );
    assert_eq!(
        interim.audio.len(),
        CHUNK * (EDGE_CHUNKS + 2 + EDGE_CHUNKS - 1),
        "interim ASR should add missing 320ms leading and trailing silence to the request audio"
    );
    assert!(
        interim.audio[..CHUNK * EDGE_CHUNKS]
            .iter()
            .all(|sample| sample_is_zero(*sample)),
        "interim request should have synthetic leading silence"
    );
    assert!(
        interim.audio[CHUNK * (EDGE_CHUNKS + 2)..]
            .iter()
            .all(|sample| sample_is_zero(*sample)),
        "interim request should have synthetic trailing silence after the natural silent chunk"
    );
    assert_sample_close(
        interim.audio[CHUNK * EDGE_CHUNKS],
        0.0,
        "speech edge should be faded in from the synthetic leading silence",
    );
    assert_sample_close(
        interim.audio[CHUNK * EDGE_CHUNKS + FADE_SAMPLES],
        1.0,
        "fade-in should not rewrite the steady speech body",
    );
    assert_eq!(
        interim.source_vad_results.len(),
        2,
        "synthetic ASR padding must not be persisted as turn-source VAD"
    );
    assert_eq!(
        interim.vad_results.len(),
        EDGE_CHUNKS + 2 + EDGE_CHUNKS - 1,
        "ASR VAD should cover the synthetic request padding"
    );
}

#[test]
fn turn_runtime_interim_display_rerecognition_uses_source_audio_before_padding() {
    const CHUNK: usize = 512;
    const EDGE_CHUNKS: usize = 10;
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, CHUNK, true),
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(2.0, CHUNK, true),
        ],
    );
    let interim = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("speech after interim silence should dispatch interim ASR");

    asr_handle.complete_request_with_text(&interim, "途中表示");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(0.0, CHUNK, false),
        ],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("turn-check silence should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        completion.audio,
        [
            vec![0.0; CHUNK],
            vec![2.0; CHUNK],
            vec![0.0; CHUNK],
            vec![0.0; CHUNK],
        ]
        .concat(),
        "completion ASR may keep the copied trailing silence as local padding"
    );

    asr_handle.complete_request_with_text(&completion, "初回認識");
    runtime.step();

    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion result should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        &rerecognition.audio[..CHUNK * 5],
        [
            vec![1.0; CHUNK],
            vec![0.0; CHUNK],
            vec![2.0; CHUNK],
            vec![0.0; CHUNK],
            vec![0.0; CHUNK],
        ]
        .concat()
        .as_slice(),
        "full-turn rerecognition must use the continuous turn audio, not stitched interim/completion ASR buffers"
    );
    assert_eq!(
        rerecognition.audio.len(),
        CHUNK * (5 + EDGE_CHUNKS - 2),
        "full-turn rerecognition should add only the missing trailing silence to the ASR request"
    );
    assert!(
        rerecognition.audio[CHUNK * 5..]
            .iter()
            .all(|sample| sample_is_zero(*sample)),
        "rerecognition padding should stay outside the persisted turn audio"
    );
}

fn sample_is_zero(sample: f32) -> bool {
    sample.abs() <= f32::EPSILON
}

fn assert_sample_close(actual: f32, expected: f32, context: &str) {
    assert!(
        (actual - expected).abs() <= f32::EPSILON,
        "{context}: actual={actual}, expected={expected}"
    );
}

#[test]
fn turn_runtime_interim_open_turn_rerecognition_adds_missing_fixed_edge_silence() {
    const CHUNK: usize = 512;
    const EDGE_CHUNKS: usize = 10;
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(64)
        .interim_display(true)
        .interim_result_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    }]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, CHUNK, true),
            fixed_vad_frame(2.0, CHUNK, true),
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(3.0, CHUNK, true),
        ],
    );
    let interim = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("speech after interim silence should dispatch interim ASR");
    assert_eq!(interim.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(interim.source_audio.len(), CHUNK * 3);

    asr_handle.complete_request_with_text(&interim, "interim");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(0.0, CHUNK, false),
        ],
    );
    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("turn-check silence after interim should dispatch full-turn rerecognition");

    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        rerecognition.audio.len(),
        CHUNK * (3 + EDGE_CHUNKS - 1),
        "rerecognition should append the missing fixed 320ms edge silence instead of waiting for a completion segment"
    );
    assert_eq!(
        &rerecognition.audio[CHUNK * 3..],
        vec![0.0; CHUNK * (EDGE_CHUNKS - 1)].as_slice()
    );
    assert_eq!(
        rerecognition.target.range.end_sample,
        GlobalSampleIndex((CHUNK * 4) as u64),
        "synthetic ASR-only silence must not expand the source audio range"
    );
}

#[test]
fn turn_runtime_interim_disabled_waits_for_turn_check_before_completion_asr() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .interim_result_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );

    assert!(
        asr_handle.submitted_requests().is_empty(),
        "interim_result_enabled=false must not dispatch ASR at interim_result_silence_ms"
    );
    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty()
    );

    replay_vad_frames_for_runtime(&mut runtime, &config, vec![(vec![0.0], vad(false))]);

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です。", true, 1, 1)]
    );
}

#[test]
fn turn_runtime_following_simple_interim_after_completed_turn_is_emitted_as_next_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();
    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です。", true, 1, 1)],
        "silence that reaches turn-check must finalize the first turn before the following root interim"
    );

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![2.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![3.0], vad(true)),
        ],
    );
    asr_handle.complete_next_with_text("すごいね");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );
    asr_handle.complete_next_with_text("すごいね");
    runtime.step();
    asr_handle.complete_next_with_text("すごいね");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![
            output_snapshot("五月五日はこどもの日です。", true, 1, 1),
            output_snapshot("すごいね...", false, 2, 2),
            output_snapshot("すごいね。", true, 2, 3),
        ]
    );
}

#[test]
#[ignore = "requires local JVS corpus; verifies UI output phrase audio coverage, not ASR accuracy"]
fn jvs_ui_phrase_audio_coverage_keeps_each_spoken_part_visible_across_interim_overwrites() {
    // This is a RecognitionSession/UI-output coverage test, not an ASR accuracy test.
    // ASR text is scripted so the assertion can focus on whether each JVS wav part
    // remains present in RecognizedTextOutput.phrase across interim replacement/finalization.
    let part_ids = ["BASIC5000_0408", "BASIC5000_1140"];
    let jvs_parts = part_ids
        .iter()
        .map(|id| read_jvs_nonparallel_part(id))
        .collect::<Vec<_>>();
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(192)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(64)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    push_jvs_speech_chunks(
        &mut runtime,
        &config,
        &jvs_parts[0].samples,
        jvs_parts[0].sample_rate,
    );
    push_silence_chunks(&mut runtime, &config, jvs_parts[0].sample_rate, 2);
    let second_part_first_chunk_len = frames_for_millis(
        jvs_parts[1].sample_rate,
        config.segmentation.vad_interval_ms,
    )
    .min(jvs_parts[1].samples.len());
    runtime.push_vad_frame(
        &jvs_parts[1].samples[..second_part_first_chunk_len],
        vad(true),
    );
    runtime.step();
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::InterimDisplay);
    asr_handle.complete_next_with_text(&jvs_parts[0].text);
    runtime.step();

    {
        let current_outputs = outputs.lock().expect("outputs should be readable");
        let latest = current_outputs
            .last()
            .expect("interim output should be emitted");
        assert!(!latest.is_final);
        assert_output_phrase_contains_jvs_parts(latest, &jvs_parts[..1]);
    }

    if second_part_first_chunk_len < jvs_parts[1].samples.len() {
        push_jvs_speech_chunks(
            &mut runtime,
            &config,
            &jvs_parts[1].samples[second_part_first_chunk_len..],
            jvs_parts[1].sample_rate,
        );
    }

    push_silence_chunks(
        &mut runtime,
        &config,
        jvs_parts[0].sample_rate,
        chunks_for_ms(jvs_parts[0].sample_rate, config.turn.check_silence_ms),
    );
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_next_with_text(&jvs_parts[1].text);
    runtime.step();
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::Rerecognition);
    let final_text = jvs_parts
        .iter()
        .map(|part| part.text.as_str())
        .collect::<String>();
    asr_handle.complete_next_with_text(&final_text);
    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 2);
    assert!(!outputs[0].is_final);
    assert!(outputs[1].is_final);
    assert_output_phrase_contains_jvs_parts(&outputs[0], &jvs_parts[..1]);
    assert_output_phrase_contains_jvs_parts(&outputs[1], &jvs_parts[..2]);
    assert!(
        outputs[0].phrase.len() < outputs[1].phrase.len(),
        "UI phrase audio should grow across interim replacement and final: {:?}",
        outputs
            .iter()
            .map(|output| output.phrase.len())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires local FLEURS-R corpus; verifies final saved phrase audio coverage across timing combinations, not ASR accuracy"]
fn fleurs_short_sentence_sequence_final_phrase_keeps_source_audio_in_order_for_timing_matrix() {
    let parts = read_short_fleurs_dev_parts("en_us", FLEURS_MATRIX_PART_COUNT);
    assert_eq!(parts.len(), FLEURS_MATRIX_PART_COUNT);

    for interim_silence_ms in [None, Some(64), Some(128)] {
        for turn_check_silence_ms in [128, 256, 512] {
            for gap_ms in [64, 128, 256] {
                run_fleurs_timing_matrix_case(
                    &parts,
                    FleursTimingCase {
                        interim_silence: interim_silence_ms,
                        turn_check_silence: turn_check_silence_ms,
                        gap: gap_ms,
                    },
                );
            }
        }
    }
}

const FLEURS_MATRIX_PART_COUNT: usize = 3;
const FLEURS_MATRIX_SEGMENT_START_MS: u32 = 128;

#[derive(Clone, Copy)]
struct FleursTimingCase {
    interim_silence: Option<u32>,
    turn_check_silence: u32,
    gap: u32,
}

fn run_fleurs_timing_matrix_case(parts: &[FleursPart], case: FleursTimingCase) {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(case.turn_check_silence)
        .segment_start_speech_ms(FLEURS_MATRIX_SEGMENT_START_MS)
        .interim_display(case.interim_silence.is_some())
        .rerecognize_full_on_complete(true);
    if let Some(interim_silence_ms) = case.interim_silence {
        builder = builder.interim_result_silence_ms(interim_silence_ms);
    }
    let internal_completion_count = usize::from(case.gap >= case.turn_check_silence) * 2;
    let mut decisions = vec![
        TurnDecision {
            is_end_of_turn: false,
            confidence: 0.99,
        };
        internal_completion_count
    ];
    decisions.push(TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    });
    let _ = builder.use_scripted_decisions(decisions);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();
    let chunk_len = frames_for_millis(parts[0].sample_rate, config.segmentation.vad_interval_ms);
    let expected_final_len = parts.iter().map(|part| part.samples.len()).sum::<usize>()
        + (chunks_for_ms(parts[0].sample_rate, case.gap) * 2
            + chunks_for_ms(parts[0].sample_rate, case.turn_check_silence))
            * chunk_len;

    push_fleurs_speech_chunks(&mut runtime, &config, &parts[0]);
    push_fleurs_gap_then_next_part(
        &mut runtime,
        &config,
        &asr_handle,
        &parts[1],
        case,
        "fleurs-0",
    );
    push_fleurs_gap_then_next_part(
        &mut runtime,
        &config,
        &asr_handle,
        &parts[2],
        case,
        "fleurs-1",
    );

    push_silence_chunks(
        &mut runtime,
        &config,
        parts[2].sample_rate,
        chunks_for_ms(parts[2].sample_rate, case.turn_check_silence),
    );
    complete_namo_turn_check_asr(
        &mut runtime,
        &asr_handle,
        case,
        "first pass",
        "fleurs final",
    );

    let outputs = outputs.lock().expect("phrase outputs should be readable");
    let final_output = outputs
        .last()
        .expect("final output should be emitted after rerecognition");
    assert!(final_output.is_final);
    assert_eq!(
        final_output.phrase.len(),
        expected_final_len,
        "{}: final phrase audio must contain observed FLEURS audio plus observed silence only",
        case.label()
    );
    assert_output_phrase_contains_fleurs_parts(final_output, parts);
    println!(
        "{} final phrase len={} parts={:?}",
        case.label(),
        final_output.phrase.len(),
        parts
            .iter()
            .map(|part| (part.wav_path.display().to_string(), part.samples.len()))
            .collect::<Vec<_>>()
    );
}

fn push_fleurs_gap_then_next_part(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    asr_handle: &ManualAsrHandle,
    next_part: &FleursPart,
    case: FleursTimingCase,
    transcript: &str,
) {
    push_silence_chunks(
        runtime,
        config,
        next_part.sample_rate,
        chunks_for_ms(next_part.sample_rate, case.gap),
    );
    if case.gap >= case.turn_check_silence {
        complete_namo_turn_check_asr(runtime, asr_handle, case, transcript, transcript);
        push_fleurs_speech_chunks(runtime, config, next_part);
        return;
    }
    if case
        .interim_silence
        .is_some_and(|interim_silence_ms| case.gap >= interim_silence_ms)
    {
        push_first_fleurs_speech_chunk(runtime, config, next_part);
        assert_next_asr_kind(asr_handle, AsrTaskKind::InterimDisplay, &case);
        asr_handle.complete_next_with_text(transcript);
        runtime.step();
        push_remaining_fleurs_speech_chunks(runtime, config, next_part);
        return;
    }
    push_fleurs_speech_chunks(runtime, config, next_part);
}

fn complete_namo_turn_check_asr(
    runtime: &mut RecognitionDriver,
    asr_handle: &ManualAsrHandle,
    case: FleursTimingCase,
    completion_text: &str,
    rerecognition_text: &str,
) {
    assert_next_asr_kind(asr_handle, AsrTaskKind::CompletionCheck, &case);
    asr_handle.complete_next_with_text(completion_text);
    runtime.step();
    assert_next_asr_kind(asr_handle, AsrTaskKind::Rerecognition, &case);
    asr_handle.complete_next_with_text(rerecognition_text);
    runtime.step();
}

fn push_first_fleurs_speech_chunk(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    part: &FleursPart,
) {
    let chunk_len = frames_for_millis(part.sample_rate, config.segmentation.vad_interval_ms);
    let first_len = chunk_len.min(part.samples.len());
    runtime.push_vad_frame(&part.samples[..first_len], vad(true));
    runtime.step();
}

fn push_remaining_fleurs_speech_chunks(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    part: &FleursPart,
) {
    let chunk_len = frames_for_millis(part.sample_rate, config.segmentation.vad_interval_ms);
    if part.samples.len() <= chunk_len {
        return;
    }
    push_jvs_speech_chunks(
        runtime,
        config,
        &part.samples[chunk_len..],
        part.sample_rate,
    );
}

fn assert_next_asr_kind(
    asr_handle: &ManualAsrHandle,
    expected: AsrTaskKind,
    case: &FleursTimingCase,
) {
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1, "{}", case.label());
    assert_eq!(submitted[0].kind, expected, "{}", case.label());
}

fn chunks_for_ms(sample_rate: u32, ms: u32) -> usize {
    let chunk_len = frames_for_millis(sample_rate, 32);
    frames_for_millis(sample_rate, ms).div_ceil(chunk_len)
}

impl FleursTimingCase {
    fn label(self) -> String {
        format!(
            "interim={:?}ms turn_check={}ms gap={}ms",
            self.interim_silence, self.turn_check_silence, self.gap
        )
    }
}

#[test]
fn turn_runtime_applies_completion_asr_result_to_output_sink() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["完了"]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "完了。".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }]
    );
}

#[test]
fn simple_completion_check_rerecognition_flag_controls_final_output_source() {
    for (case_name, rerecognize_full_on_complete) in [
        ("rerecognition disabled", false),
        ("rerecognition enabled", true),
    ] {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Simple)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(false)
            .rerecognize_full_on_complete(rerecognize_full_on_complete);
        let asr_handle = builder.use_manual_asr();
        let outputs = builder.use_recording_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        let completion_request = runtime
            .requests
            .in_flight_request
            .clone()
            .unwrap_or_else(|| {
                panic!("{case_name}: closed segment should dispatch completion ASR")
            });
        assert_eq!(
            completion_request.kind,
            AsrTaskKind::CompletionCheck,
            "{case_name}"
        );

        asr_handle.complete_request_with_text(&completion_request, "初回結果");
        runtime.step();

        if rerecognize_full_on_complete {
            assert_eq!(
                *outputs.lock().expect("outputs should be readable"),
                Vec::<OutputSnapshot>::new(),
                "{case_name}: completion output must wait for full-turn rerecognition"
            );
            let rerecognition_request =
                runtime
                    .requests
                    .in_flight_request
                    .clone()
                    .unwrap_or_else(|| {
                        panic!(
                            "{case_name}: completion result should dispatch full-turn rerecognition"
                        )
                    });
            assert_eq!(
                rerecognition_request.kind,
                AsrTaskKind::Rerecognition,
                "{case_name}"
            );
            assert_eq!(
                rerecognition_request.target.turn_id,
                TurnId(1),
                "{case_name}"
            );
            assert_eq!(
                &rerecognition_request.audio[..completion_request.source_audio.len()],
                completion_request.source_audio.as_slice(),
                "{case_name}: rerecognition should start from the continuous source audio"
            );
            assert!(
                rerecognition_request.audio[completion_request.source_audio.len()..]
                    .iter()
                    .all(|sample| *sample == 0.0),
                "{case_name}: missing ASR-only trailing silence should be synthesized"
            );

            asr_handle.complete_request_with_text(&rerecognition_request, "再認識結果");
            runtime.step();

            assert_eq!(
                *outputs.lock().expect("outputs should be readable"),
                vec![output_snapshot("再認識結果。", true, 1, 1)],
                "{case_name}"
            );
        } else {
            assert_eq!(
                *outputs.lock().expect("outputs should be readable"),
                vec![output_snapshot("初回結果。", true, 1, 1)],
                "{case_name}"
            );
            assert!(
                runtime.requests.in_flight_request.is_none(),
                "{case_name}: disabled rerecognition must not dispatch a second ASR request"
            );
            assert_eq!(asr_handle.submitted_requests().len(), 1, "{case_name}");
        }
    }
}

#[test]
fn turn_runtime_parakeet_models_dispatch_rerecognition_after_namo_completion_check() {
    for model in [
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
        AsrModel::NemoParakeetTdt0_6BV3Int8,
    ] {
        let mut builder = RecognitionSessionTestBuilder::new()
            .asr_model(model)
            .turn_detector(TurnDetector::Namo)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(false);
        let asr_handle = builder.use_manual_asr();
        let outputs = builder.use_recording_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        let completion_request = runtime
            .requests
            .in_flight_request
            .clone()
            .expect("closed segment should dispatch completion ASR");
        assert_eq!(completion_request.kind, AsrTaskKind::CompletionCheck);
        assert_eq!(completion_request.route.model, model, "model={model:?}");

        asr_handle.complete_request_with_text(&completion_request, "first pass");
        runtime.step();

        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            Vec::<OutputSnapshot>::new(),
            "model={model:?} must wait for rerecognition before final output"
        );
        let rerecognition_request = runtime
            .requests
            .in_flight_request
            .clone()
            .expect("Namo completion result should dispatch full-turn rerecognition");
        assert_eq!(
            rerecognition_request.kind,
            AsrTaskKind::Rerecognition,
            "model={model:?}"
        );
        assert_eq!(rerecognition_request.route.model, model, "model={model:?}");
        assert_eq!(
            &rerecognition_request.audio[..completion_request.source_audio.len()],
            completion_request.source_audio.as_slice(),
            "single-segment model={model:?} should rerecognize the same full-turn source audio before ASR-only padding"
        );
        assert!(
            rerecognition_request.audio[completion_request.source_audio.len()..]
                .iter()
                .all(|sample| *sample == 0.0),
            "single-segment model={model:?} should synthesize only missing trailing silence"
        );
    }
}

#[test]
fn english_punctuation_after_rerecognition_finalizes_as_strong_end_without_namo() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::NemoParakeetTdt0_6BV2Int8)
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("closed English segment should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);

    asr_handle.complete_request_with_text(&completion, "we should keep going");
    runtime.step();

    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("Namo completion should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    asr_handle.push_completed_result(AsrResult {
        request_id: rerecognition.request_id,
        kind: rerecognition.kind,
        target: rerecognition.target.clone(),
        route: rerecognition.route,
        status: AsrResultStatus::Ok(english_sentence_end_transcript("We should keep going.")),
        completed_at_frame: VadFrameIndex(0),
        elapsed_millis: 0,
    });
    runtime.step();

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        Vec::<String>::new(),
        "English sentence punctuation should finalize as StrongEnd without asking Namo"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "We should keep going.".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }],
        "Namo must finalize English sentence punctuation as grammar StrongEnd"
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

fn english_sentence_end_transcript(text: &str) -> AsrTranscript {
    let tokens = text.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
    let timestamps = (0..tokens.len())
        .map(|index| {
            f32::from(u16::try_from(index).expect("test transcript should have few tokens")) / 100.0
        })
        .collect::<Vec<_>>();
    let durations = vec![0.01; tokens.len()];
    AsrTranscript::from_parts(
        text.to_string(),
        tokens,
        Some(&timestamps),
        Some(&durations),
    )
}

#[test]
fn turn_runtime_internal_strong_boundary_keeps_turn_open_until_terminal_candidate() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_transcripts(vec![
            AsrTranscript::from_text("はい次です"),
            japanese_punctuation_transcript(),
        ]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    let rerecognition = runtime
        .take_last_dispatched()
        .expect("completion result should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);

    runtime.step();

    assert_eq!(*outputs.lock().expect("outputs should be readable"), vec![]);
    assert_eq!(
        runtime.turn_store.open_turn_id,
        Some(1),
        "internal grammar boundary must keep the full turn open"
    );
    assert!(
        runtime.turn_store.turns.contains_key(&1),
        "turn should remain open"
    );
}

#[test]
fn turn_runtime_namo_complete_without_boundary_emits_final() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅", "続き", "続き再認識", "さらに続き"]);
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    runtime.step();

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        vec!["東京駅".to_string()]
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "東京駅。".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }]
    );
}

#[test]
fn namo_decision_confidence_threshold_controls_finalization() {
    struct Case {
        name: &'static str,
        decision: TurnDecision,
        expected_output: Option<(&'static str, bool)>,
        expected_open_turn_id: Option<u64>,
    }

    let cases = vec![
        Case {
            name: "end decision at threshold finalizes",
            decision: TurnDecision {
                is_end_of_turn: true,
                confidence: 0.8,
            },
            expected_output: Some(("東京駅。", true)),
            expected_open_turn_id: None,
        },
        Case {
            name: "continue decision above threshold stays open",
            decision: TurnDecision {
                is_end_of_turn: false,
                confidence: 0.99,
            },
            expected_output: None,
            expected_open_turn_id: Some(1),
        },
        Case {
            name: "end decision below threshold stays open",
            decision: TurnDecision {
                is_end_of_turn: true,
                confidence: 0.79,
            },
            expected_output: None,
            expected_open_turn_id: Some(1),
        },
    ];

    for Case {
        name,
        decision,
        expected_output,
        expected_open_turn_id,
    } in cases
    {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Namo)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(false)
            .namo_turn_confidence_threshold(0.8)
            .scripted_asr_texts(vec!["東京駅", "東京駅"]);
        let _ = builder.use_scripted_decisions(vec![decision]);
        let outputs = builder.use_recording_phrase_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        runtime.step();
        runtime.step();

        let outputs = outputs.lock().expect("outputs should be readable");
        match expected_output {
            Some((expected_text, expected_is_final)) => {
                assert_eq!(outputs.len(), 1, "{name}");
                assert_eq!(outputs[0].text, expected_text, "{name}");
                assert_eq!(outputs[0].is_final, expected_is_final, "{name}");
            }
            None => assert!(outputs.is_empty(), "{name}: got outputs {outputs:?}"),
        }
        assert_eq!(
            runtime.turn_store.open_turn_id, expected_open_turn_id,
            "{name}"
        );
    }
}

#[test]
fn namo_continue_interim_display_flag_controls_partial_output_while_turn_stays_open() {
    struct Case {
        name: &'static str,
        interim_display: bool,
        expected_output: Option<(&'static str, bool)>,
    }

    let cases = vec![
        Case {
            name: "interim display disabled",
            interim_display: false,
            expected_output: None,
        },
        Case {
            name: "interim display enabled",
            interim_display: true,
            expected_output: Some(("東京駅...", false)),
        },
    ];

    for Case {
        name,
        interim_display,
        expected_output,
    } in cases
    {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Namo)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(interim_display)
            .interim_result_silence_ms(32)
            .scripted_asr_texts(vec!["東京駅", "東京駅"]);
        let _ = builder.use_scripted_decisions(vec![TurnDecision {
            is_end_of_turn: false,
            confidence: 0.01,
        }]);
        let outputs = builder.use_recording_phrase_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        runtime.step();
        runtime.step();

        let outputs = outputs.lock().expect("outputs should be readable");
        match expected_output {
            Some((expected_text, expected_is_final)) => {
                assert_eq!(outputs.len(), 1, "{name}");
                assert_eq!(outputs[0].text, expected_text, "{name}");
                assert_eq!(outputs[0].is_final, expected_is_final, "{name}");
            }
            None => assert!(outputs.is_empty(), "{name}: got outputs {outputs:?}"),
        }
        assert_eq!(runtime.turn_store.open_turn_id, Some(1), "{name}");
    }
}

#[test]
fn namo_turn_decision_error_keeps_turn_open_without_final_output() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅"]);
    let decision_texts = builder.use_scripted_decisions(Vec::new());
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    runtime.step();

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        vec!["東京駅".to_string()],
        "the decision runner should receive the draft text before its error is handled"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new(),
        "a Namo decision error must continue the turn instead of finalizing with stale confidence"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}

#[test]
fn turn_runtime_timeout_after_namo_continue_rerecognizes_then_finalizes() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅", "東京駅再認識"]);
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 0.01,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    runtime.step();
    let timeout_chunks =
        usize::try_from(runtime.timeout_ticks()).expect("timeout ticks should fit usize");
    push_silence_chunks(&mut runtime, &config, 16_000, timeout_chunks);
    let timeout_rerecognition = runtime
        .take_last_dispatched()
        .expect("timeout should dispatch rerecognition before final");
    assert_eq!(timeout_rerecognition.kind, AsrTaskKind::Rerecognition);
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "東京駅再認識。".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }]
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_activity_after_namo_continue_delays_timeout() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false);
    let asr_handle = builder.use_manual_asr();
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 0.01,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("first segment should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_request_with_text(&completion, "東京駅");
    runtime.step();
    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion result should dispatch rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    asr_handle.complete_request_with_text(&rerecognition, "東京駅");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![2.0], vad(true)),
            (vec![2.0], vad(true)),
            (vec![0.0], vad(false)),
        ],
    );

    let next_completion = runtime
        .take_last_dispatched()
        .expect("following active speech should close as the next segment before timeout");
    assert_eq!(next_completion.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        next_completion.target.turn_id,
        TurnId(1),
        "the segment after Namo Continue must stay attached to the open turn"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new()
    );
}

#[test]
fn simple_turn_check_rerecognition_flag_controls_existing_interim_finalization() {
    for (case_name, rerecognize_full_on_complete) in [
        ("rerecognition disabled", false),
        ("rerecognition enabled", true),
    ] {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Simple)
            .vad_interval_ms(32)
            .turn_check_silence_ms(64)
            .segment_start_speech_ms(1)
            .interim_display(true)
            .interim_result_silence_ms(32)
            .rerecognize_full_on_complete(rerecognize_full_on_complete)
            .scripted_asr_texts(vec!["途中", "確定"]);
        let outputs = builder.use_recording_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![
                (vec![1.0], vad(true)),
                (vec![0.0], vad(false)),
                (vec![0.0], vad(false)),
            ],
        );

        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            Vec::<OutputSnapshot>::new(),
            "{case_name}: turn check should not emit before the queued work is stepped"
        );

        runtime.step();

        let expected_after_first_step = if rerecognize_full_on_complete {
            Vec::<OutputSnapshot>::new()
        } else {
            vec![output_snapshot("途中。", true, 1, 1)]
        };
        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            expected_after_first_step,
            "{case_name}"
        );

        runtime.step();

        let expected_after_second_step = if rerecognize_full_on_complete {
            vec![output_snapshot("確定。", true, 1, 1)]
        } else {
            vec![output_snapshot("途中。", true, 1, 1)]
        };
        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            expected_after_second_step,
            "{case_name}"
        );
    }
}
