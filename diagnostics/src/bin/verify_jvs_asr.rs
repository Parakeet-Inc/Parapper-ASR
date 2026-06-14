#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

//! JVS nonparallel wav を sherpa-onnx / `ReazonSpeech` に直接かける ASR engine 単体評価ツール。
//!
//! この binary は VAD plot、連結 wav、CER、token timestamp の診断用であり、
//! `RecognitionSession` の segmenter / `TurnDraft` / UI output 統合経路は通さない。

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    execution_providers::CPUExecutionProvider,
    inputs,
    session::Session,
    value::{Tensor, TensorRef},
};
use rubato::{
    Async, FixedAsync, PolynomialDegree, Resampler,
    audioadapter::{Adapter, AdapterMut},
};
use serde::Serialize;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

#[path = "../verify_jvs_asr_constants.rs"]
mod verify_jvs_asr_constants;

use verify_jvs_asr_constants::LEGACY_INTERIM_SUMMARY_PREFIX;

const ASR_SAMPLE_RATE: u32 = 16_000;
const SILERO_CHUNK_SAMPLES: usize = 512;
const SILERO_CONTEXT_SAMPLES: usize = 64;
const SILERO_INPUT_SAMPLES: usize = SILERO_CONTEXT_SAMPLES + SILERO_CHUNK_SAMPLES;
const SILERO_STATE_LEN: usize = 2 * 128;
const DEFAULT_ASR_EDGE_SILENCE_MS: u32 = 320;
const ASR_EDGE_FADE_MS: u32 = 10;
const REAZONSPEECH_MODEL_DIR_NAME: &str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01";
const ENCODER_FILE: &str = "encoder-epoch-99-avg-1.int8.onnx";
const DECODER_FILE: &str = "decoder-epoch-99-avg-1.onnx";
const JOINER_FILE: &str = "joiner-epoch-99-avg-1.int8.onnx";
const TOKENS_FILE: &str = "tokens.txt";
const APP_IDENTIFIER: &str = "com.parakeet-inc.parapper";
const MODELS_ROOT_ENV: &str = "PARAPPER_MODELS_ROOT";
const ASR_MODEL_DIR_ENV: &str = "PARAPPER_ASR_MODEL_DIR";
const VAD_MODEL_ENV: &str = "PARAPPER_VAD_MODEL";
const JVS_ROOT_ENV: &str = "JVS_ROOT";

fn main() -> Result<()> {
    let args = Args::parse()?;
    let samples = group_samples(collect_jvs_nonparallel_samples(&args)?, &args);
    if samples.is_empty() {
        bail!(
            "No JVS nonparallel samples found under {}",
            args.jvs_root.display()
        );
    }

    validate_model_dir(&args.model_dir)?;
    let recognizer = create_recognizer(&args.model_dir, args.num_threads)?;
    let mut raw_vad = if args.plot_json.is_some() || args.trim_by_vad {
        Some(RawSileroVadEngine::new(
            &args.vad_model,
            args.vad_threshold,
        )?)
    } else {
        None
    };

    let mut evaluated = Vec::new();
    let mut plot_records = Vec::new();
    for sample in samples {
        let mut audio = Vec::new();
        let mut part_diagnostics = Vec::new();
        let mut speech_ranges = Vec::new();
        let mut sample_cursor = 0usize;
        for (index, part) in sample.parts.iter().enumerate() {
            if index > 0 && args.concat_silence_ms > 0 {
                let silence_len = frames_for_millis(ASR_SAMPLE_RATE, args.concat_silence_ms);
                audio.resize(audio.len() + silence_len, 0.0);
                sample_cursor += silence_len;
            }
            let wav = read_wav_mono_f32(&part.wav_path)
                .with_context(|| format!("Failed to read {}", part.wav_path.display()))?;
            let mut resampled = resample_to_asr_rate(&wav.samples, wav.sample_rate)
                .with_context(|| format!("Failed to resample {}", part.wav_path.display()))?;
            if args.trim_by_vad {
                let frames = raw_vad
                    .as_mut()
                    .expect("raw VAD engine must be available when --trim-by-vad is set")
                    .process_audio(&resampled)?;
                if let Some(range) = speech_span_from_raw_vad(&frames, resampled.len()) {
                    resampled = resampled[range].to_vec();
                }
            }
            if args.verbose {
                let standalone_recognized = transcribe(&recognizer, &resampled)
                    .with_context(|| format!("Failed to transcribe standalone {}", part.id))?;
                let standalone_expected_normalized = normalize_for_jvs_asr_check(&part.expected);
                let standalone_recognized_normalized =
                    normalize_for_jvs_asr_check(&standalone_recognized.text);
                let standalone_cer = char_error_rate(
                    &standalone_expected_normalized,
                    &standalone_recognized_normalized,
                );
                part_diagnostics.push(PartDiagnostic {
                    id: part.id.clone(),
                    source_sample_rate: wav.sample_rate,
                    source_samples: wav.samples.len(),
                    asr_samples: resampled.len(),
                    timeline_start_sample: sample_cursor,
                    timeline_end_sample: sample_cursor + resampled.len(),
                    expected: part.expected.clone(),
                    standalone_recognized,
                    standalone_cer,
                });
            }
            sample_cursor += resampled.len();
            speech_ranges.push(sample_cursor - resampled.len()..sample_cursor);
            audio.extend(resampled);
        }
        if args.trailing_silence_ms > 0 {
            let silence_len = frames_for_millis(ASR_SAMPLE_RATE, args.trailing_silence_ms);
            audio.resize(audio.len() + silence_len, 0.0);
        }
        if args.asr_edge_silence_ms > 0 {
            add_asr_edge_silence(
                &mut audio,
                &mut speech_ranges,
                args.asr_edge_silence_ms,
                ASR_EDGE_FADE_MS,
            );
        }
        let audio_stats = AudioStats::from_audio_and_speech_ranges(&audio, &speech_ranges);
        let raw_vad_frames = raw_vad
            .as_mut()
            .map(|vad| vad.process_audio(&audio))
            .transpose()?
            .unwrap_or_default();
        let raw_recognized = transcribe(&recognizer, &audio)
            .with_context(|| format!("Failed to transcribe {}", sample.id))?;
        let display_transcript = raw_recognized.clone();
        let expected_normalized = normalize_for_jvs_asr_check(&sample.expected);
        let recognized_normalized = normalize_for_jvs_asr_check(&display_transcript.text);
        let legacy_interim = args
            .compare_interim_rerecognition
            .then(|| {
                evaluate_legacy_interim_rerecognition(
                    &recognizer,
                    &sample,
                    &audio,
                    &speech_ranges,
                    &expected_normalized,
                    &args,
                )
            })
            .transpose()?;
        let first_token_start_sample = raw_recognized.first_token_start_sample();
        let last_token_end_sample = raw_recognized.last_token_end_sample();
        let token_count = raw_recognized.tokens.len();
        let cer = char_error_rate(&expected_normalized, &recognized_normalized);
        if args.plot_json.is_some() {
            plot_records.push(PlotRecord {
                id: sample.id.clone(),
                expected: sample.expected.clone(),
                raw_text: raw_recognized.text.clone(),
                recognized_text: display_transcript.text.clone(),
                sample_rate: ASR_SAMPLE_RATE,
                vad_threshold: args.vad_threshold,
                audio: audio.clone(),
                speech_ranges: speech_ranges
                    .iter()
                    .map(|range| PlotRange {
                        start: range.start,
                        end: range.end,
                    })
                    .collect(),
                raw_vad_frames,
                tokens: raw_recognized
                    .tokens
                    .iter()
                    .map(|token| PlotToken {
                        text: token.text.clone(),
                        sample: token.start_sample(),
                    })
                    .collect(),
            });
        }
        evaluated.push(EvaluatedSample {
            speaker: sample.speaker,
            id: sample.id,
            expected: sample.expected,
            recognized: display_transcript,
            raw_recognized,
            expected_normalized,
            recognized_normalized,
            cer,
            asr_samples: audio.len(),
            speech_ranges,
            parts: part_diagnostics,
            audio_stats,
            first_token_start_sample,
            last_token_end_sample,
            token_count,
            legacy_interim,
        });
    }

    let total_cer = evaluated.iter().map(|sample| sample.cer).sum::<f64>() / evaluated.len() as f64;
    let exact_count = evaluated
        .iter()
        .filter(|sample| sample.expected_normalized == sample.recognized_normalized)
        .count();
    let exact_rate = exact_count as f64 / evaluated.len() as f64;

    if let Some(path) = &args.plot_json {
        write_plot_json(path, &plot_records)?;
        println!("plot json: {}", path.display());
    }

    println!("JVS root: {}", args.jvs_root.display());
    println!("model dir: {}", args.model_dir.display());
    println!("samples: {}", evaluated.len());
    println!(
        "concat size: {}, concat silence: {} ms, trailing silence: {} ms",
        args.concat_size, args.concat_silence_ms, args.trailing_silence_ms
    );
    println!("mean CER: {total_cer:.4}");
    println!(
        "exact normalized match: {exact_count}/{} ({exact_rate:.2})",
        evaluated.len()
    );
    if args.compare_interim_rerecognition {
        print_legacy_interim_summary(&evaluated, total_cer);
    }

    for sample in evaluated
        .iter()
        .filter(|sample| args.verbose || sample.expected_normalized != sample.recognized_normalized)
        .take(args.print_failures)
    {
        println!();
        if sample.expected_normalized == sample.recognized_normalized {
            println!("match {} {}", sample.speaker, sample.id);
        } else {
            println!("mismatch {} {}", sample.speaker, sample.id);
        }
        println!(
            "concat audio: {} samples ({:.2} sec)",
            sample.asr_samples,
            sample.asr_samples as f32 / ASR_SAMPLE_RATE as f32
        );
        println!("{}", sample.audio_stats.report());
        for part in &sample.parts {
            println!(
                "  part {} timeline={}..{} source={}Hz/{} samples asr={} samples ({:.2} sec) standalone_CER={:.4}",
                part.id,
                part.timeline_start_sample,
                part.timeline_end_sample,
                part.source_sample_rate,
                part.source_samples,
                part.asr_samples,
                part.asr_samples as f32 / ASR_SAMPLE_RATE as f32,
                part.standalone_cer
            );
            println!("    standalone expected:   {}", part.expected);
            println!(
                "    standalone recognized: {}",
                part.standalone_recognized.text
            );
            println!(
                "    standalone tokens: count={} first={:?} last={:?}",
                part.standalone_recognized.tokens.len(),
                part.standalone_recognized.first_token_start_sample(),
                part.standalone_recognized.last_token_end_sample()
            );
            println!(
                "    standalone first tokens: {}",
                part.standalone_recognized.token_preview()
            );
        }
        println!(
            "concat tokens: count={} first={:?} last={:?}",
            sample.token_count, sample.first_token_start_sample, sample.last_token_end_sample
        );
        println!(
            "raw concat first tokens: {}",
            sample.raw_recognized.token_preview()
        );
        println!(
            "raw concat last tokens: {}",
            sample.raw_recognized.last_token_preview()
        );
        println!(
            "estimated first token from same speech run: {:?}",
            sample
                .raw_recognized
                .estimate_first_token_from_following_speech_run(&sample.speech_ranges())
        );
        println!(
            "estimated last token from same speech run: {:?}",
            sample
                .raw_recognized
                .estimate_last_token_from_previous_speech_run(&sample.speech_ranges())
        );
        println!(
            "timestamp/VAD disagreement range: leading={:?} trailing={:?}",
            sample.timestamp_vad_leading_disagreement(),
            sample.timestamp_vad_trailing_disagreement()
        );
        println!(
            "tokens on VAD silence: {}",
            sample
                .raw_recognized
                .silence_token_report(&sample.speech_ranges(), sample.asr_samples)
        );
        println!(
            "timestamp/VAD disagreement after estimated first token: leading={:?}",
            sample.timestamp_vad_leading_disagreement_after_first_token_estimation()
        );
        if let Some(legacy) = &sample.legacy_interim {
            println!(
                "legacy interim duplicated-padding: splits={} duplicated={} samples asr={} samples ({:.2} sec) CER={:.4}",
                legacy.split_count,
                legacy.duplicated_samples,
                legacy.asr_samples,
                legacy.asr_samples as f32 / ASR_SAMPLE_RATE as f32,
                legacy.cer
            );
            println!("legacy interim recognized: {}", legacy.recognized.text);
            println!(
                "legacy interim normalized: {}",
                legacy.recognized_normalized
            );
        }
        println!("expected:   {}", sample.expected);
        println!("raw recognized: {}", sample.raw_recognized.text);
        println!("recognized: {}", sample.recognized.text);
        println!("expected normalized:   {}", sample.expected_normalized);
        println!("recognized normalized: {}", sample.recognized_normalized);
        println!("CER: {:.4}", sample.cer);
    }

    if total_cer > args.max_cer {
        bail!(
            "JVS ASR mean CER {total_cer:.4} exceeded threshold {:.4}",
            args.max_cer
        );
    }
    if exact_rate < args.min_exact_rate {
        bail!(
            "JVS ASR exact normalized match rate {exact_rate:.4} was below threshold {:.4}",
            args.min_exact_rate
        );
    }

    Ok(())
}

fn evaluate_legacy_interim_rerecognition(
    asr: &OfflineRecognizer,
    sample: &JvsSample,
    continuous_audio: &[f32],
    speech_ranges: &[std::ops::Range<usize>],
    expected_normalized: &str,
    args: &Args,
) -> Result<LegacyInterimEvaluation> {
    let scenario = legacy_interim_rerecognition_audio(
        continuous_audio,
        speech_ranges,
        args.interim_result_silence_ms,
        args.turn_check_silence_ms,
    );
    let recognized = transcribe(asr, &scenario.audio)
        .with_context(|| format!("Failed to transcribe legacy interim scenario {}", sample.id))?;
    let recognized_normalized = normalize_for_jvs_asr_check(&recognized.text);
    let cer = char_error_rate(expected_normalized, &recognized_normalized);
    Ok(LegacyInterimEvaluation {
        recognized,
        recognized_normalized,
        cer,
        asr_samples: scenario.audio.len(),
        duplicated_samples: scenario.duplicated_samples,
        split_count: scenario.split_count,
    })
}

fn legacy_interim_rerecognition_audio(
    continuous_audio: &[f32],
    speech_ranges: &[std::ops::Range<usize>],
    interim_result_silence_ms: u32,
    turn_check_silence_ms: u32,
) -> LegacyInterimAudio {
    let interim_result_silence_samples =
        frames_for_millis(ASR_SAMPLE_RATE, interim_result_silence_ms);
    let turn_check_silence_samples = frames_for_millis(ASR_SAMPLE_RATE, turn_check_silence_ms);
    let mut audio = Vec::with_capacity(continuous_audio.len());
    let mut cursor = 0usize;
    let mut duplicated_samples = 0usize;
    let mut split_count = 0usize;

    for pair in speech_ranges.windows(2) {
        let previous = &pair[0];
        let next = &pair[1];
        if previous.end > next.start || next.start > continuous_audio.len() {
            continue;
        }
        let silence = previous.end..next.start;
        let silence_samples = silence.end.saturating_sub(silence.start);
        if silence_samples >= interim_result_silence_samples
            && silence_samples < turn_check_silence_samples
        {
            audio.extend_from_slice(&continuous_audio[cursor..next.start]);
            audio.extend_from_slice(&continuous_audio[silence]);
            duplicated_samples = duplicated_samples.saturating_add(silence_samples);
            split_count = split_count.saturating_add(1);
            cursor = next.start;
        }
    }

    audio.extend_from_slice(&continuous_audio[cursor..]);
    LegacyInterimAudio {
        audio,
        duplicated_samples,
        split_count,
    }
}

fn print_legacy_interim_summary(evaluated: &[EvaluatedSample], continuous_cer: f64) {
    let legacy = evaluated
        .iter()
        .filter_map(|sample| sample.legacy_interim.as_ref())
        .collect::<Vec<_>>();
    if legacy.is_empty() {
        return;
    }

    let legacy_cer = legacy.iter().map(|sample| sample.cer).sum::<f64>() / legacy.len() as f64;
    let duplicated_samples = legacy
        .iter()
        .map(|sample| sample.duplicated_samples)
        .sum::<usize>();
    let split_count = legacy
        .iter()
        .map(|sample| sample.split_count)
        .sum::<usize>();
    println!(
        "{LEGACY_INTERIM_SUMMARY_PREFIX}: {legacy_cer:.4} (delta {:+.4})",
        legacy_cer - continuous_cer
    );
    println!(
        "legacy interim duplicated-padding splits: {split_count}, duplicated samples: {duplicated_samples}"
    );
}

#[derive(Debug)]
struct Args {
    jvs_root: PathBuf,
    model_dir: PathBuf,
    vad_model: PathBuf,
    vad_threshold: f32,
    max_speakers: usize,
    max_utterances_per_speaker: usize,
    max_total: usize,
    max_cer: f64,
    min_exact_rate: f64,
    num_threads: i32,
    print_failures: usize,
    concat_size: usize,
    concat_silence_ms: u32,
    trailing_silence_ms: u32,
    asr_edge_silence_ms: u32,
    compare_interim_rerecognition: bool,
    interim_result_silence_ms: u32,
    turn_check_silence_ms: u32,
    trim_by_vad: bool,
    verbose: bool,
    ids: Option<Vec<String>>,
    plot_json: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut jvs_root = env_path(JVS_ROOT_ENV);
        let mut args = Self {
            jvs_root: PathBuf::new(),
            model_dir: env_path(ASR_MODEL_DIR_ENV).unwrap_or_default(),
            vad_model: env_path(VAD_MODEL_ENV).unwrap_or_default(),
            vad_threshold: 0.5,
            max_speakers: 1,
            max_utterances_per_speaker: 30,
            max_total: usize::MAX,
            max_cer: 0.15,
            min_exact_rate: 0.0,
            num_threads: 4,
            print_failures: 10,
            concat_size: 1,
            concat_silence_ms: 0,
            trailing_silence_ms: 0,
            asr_edge_silence_ms: 0,
            compare_interim_rerecognition: false,
            interim_result_silence_ms: 64,
            turn_check_silence_ms: 192,
            trim_by_vad: false,
            verbose: false,
            ids: None,
            plot_json: None,
        };

        let mut raw = env::args().skip(1);
        while let Some(flag) = raw.next() {
            match flag.as_str() {
                "--jvs-root" => jvs_root = Some(next_path(&mut raw, &flag)?),
                "--model-dir" => args.model_dir = next_path(&mut raw, &flag)?,
                "--vad-model" => args.vad_model = next_path(&mut raw, &flag)?,
                "--vad-threshold" => args.vad_threshold = next_parse(&mut raw, &flag)?,
                "--max-speakers" => args.max_speakers = next_parse(&mut raw, &flag)?,
                "--max-utterances-per-speaker" => {
                    args.max_utterances_per_speaker = next_parse(&mut raw, &flag)?;
                }
                "--max-total" => args.max_total = next_parse(&mut raw, &flag)?,
                "--max-cer" => args.max_cer = next_parse(&mut raw, &flag)?,
                "--min-exact-rate" => args.min_exact_rate = next_parse(&mut raw, &flag)?,
                "--num-threads" => args.num_threads = next_parse(&mut raw, &flag)?,
                "--print-failures" => args.print_failures = next_parse(&mut raw, &flag)?,
                "--concat-size" => args.concat_size = next_parse(&mut raw, &flag)?,
                "--concat-silence-ms" => args.concat_silence_ms = next_parse(&mut raw, &flag)?,
                "--trailing-silence-ms" => {
                    args.trailing_silence_ms = next_parse(&mut raw, &flag)?;
                }
                "--asr-edge-silence-ms" => {
                    args.asr_edge_silence_ms = next_parse(&mut raw, &flag)?;
                }
                "--production-asr-padding" => {
                    args.asr_edge_silence_ms = DEFAULT_ASR_EDGE_SILENCE_MS;
                }
                "--compare-interim-rerecognition" => {
                    args.compare_interim_rerecognition = true;
                }
                "--interim-result-silence-ms" => {
                    args.interim_result_silence_ms = next_parse(&mut raw, &flag)?;
                }
                "--turn-check-silence-ms" => {
                    args.turn_check_silence_ms = next_parse(&mut raw, &flag)?;
                }
                "--trim-by-vad" => args.trim_by_vad = true,
                "--verbose" => args.verbose = true,
                "--plot-json" => args.plot_json = Some(next_path(&mut raw, &flag)?),
                "--ids" => {
                    let value = raw
                        .next()
                        .ok_or_else(|| anyhow!("Missing value for {flag}"))?;
                    args.ids = Some(
                        value
                            .split(',')
                            .map(str::trim)
                            .filter(|id| !id.is_empty())
                            .map(str::to_string)
                            .collect(),
                    );
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("Unknown argument: {other}"),
            }
        }
        if args.concat_size == 0 {
            bail!("--concat-size must be 1 or greater");
        }
        if args.ids.as_ref().is_some_and(Vec::is_empty) {
            bail!("--ids must include at least one transcript id");
        }
        if !(0.0..=1.0).contains(&args.vad_threshold) {
            bail!("--vad-threshold must be between 0.0 and 1.0");
        }
        if args.compare_interim_rerecognition
            && args.interim_result_silence_ms >= args.turn_check_silence_ms
        {
            bail!("--interim-result-silence-ms must be less than --turn-check-silence-ms");
        }
        args.jvs_root = jvs_root
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| {
                anyhow!("JVS root is required. Pass --jvs-root PATH or set JVS_ROOT.")
            })?;
        if args.model_dir.as_os_str().is_empty() || args.vad_model.as_os_str().is_empty() {
            let default_models_root = default_models_root()?;
            if args.model_dir.as_os_str().is_empty() {
                args.model_dir = default_models_root.join(REAZONSPEECH_MODEL_DIR_NAME);
            }
            if args.vad_model.as_os_str().is_empty() {
                args.vad_model = default_models_root
                    .join("silero_vad_v6")
                    .join("silero_vad.onnx");
            }
        }

        Ok(args)
    }
}

fn next_path(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    raw.next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("Missing value for {flag}"))
}

fn next_parse<T: std::str::FromStr>(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    raw.next()
        .ok_or_else(|| anyhow!("Missing value for {flag}"))?
        .parse()
        .map_err(|err| anyhow!("Invalid value for {flag}: {err}"))
}

fn print_help() {
    println!(
        "verify_jvs_asr --jvs-root PATH [--model-dir PATH] [--vad-model PATH] [--vad-threshold FLOAT] [--max-speakers N] [--max-utterances-per-speaker N] [--concat-size N] [--ids ID1,ID2] [--concat-silence-ms MS] [--trailing-silence-ms MS] [--trim-by-vad] [--production-asr-padding] [--asr-edge-silence-ms MS] [--compare-interim-rerecognition] [--interim-result-silence-ms MS] [--turn-check-silence-ms MS] [--max-total N] [--max-cer FLOAT] [--plot-json PATH] [--verbose]\n\nEnvironment defaults: JVS_ROOT, PARAPPER_MODELS_ROOT, PARAPPER_ASR_MODEL_DIR, PARAPPER_VAD_MODEL."
    );
}

fn write_plot_json(path: &Path, records: &[PlotRecord]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(records)?;
    fs::write(path, json).with_context(|| format!("Failed to write {}", path.display()))
}

fn default_models_root() -> Result<PathBuf> {
    if let Some(path) = env_path(MODELS_ROOT_ENV) {
        return Ok(path);
    }
    Ok(default_app_data_dir()?.join("models"))
}

fn env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

#[cfg(target_os = "windows")]
fn default_app_data_dir() -> Result<PathBuf> {
    env_path("APPDATA")
        .map(|path| path.join(APP_IDENTIFIER))
        .ok_or_else(|| anyhow!("APPDATA is not set; set {MODELS_ROOT_ENV} explicitly"))
}

#[cfg(target_os = "macos")]
fn default_app_data_dir() -> Result<PathBuf> {
    env_path("HOME")
        .map(|path| {
            path.join("Library")
                .join("Application Support")
                .join(APP_IDENTIFIER)
        })
        .ok_or_else(|| anyhow!("HOME is not set; set {MODELS_ROOT_ENV} explicitly"))
}

#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
fn default_app_data_dir() -> Result<PathBuf> {
    if let Some(path) = env_path("XDG_DATA_HOME") {
        return Ok(path.join(APP_IDENTIFIER));
    }
    env_path("HOME")
        .map(|path| path.join(".local").join("share").join(APP_IDENTIFIER))
        .ok_or_else(|| {
            anyhow!("XDG_DATA_HOME and HOME are not set; set {MODELS_ROOT_ENV} explicitly")
        })
}

#[derive(Debug)]
struct JvsSample {
    speaker: String,
    id: String,
    parts: Vec<JvsSamplePart>,
    expected: String,
}

#[derive(Debug)]
struct JvsSamplePart {
    id: String,
    wav_path: PathBuf,
    expected: String,
}

fn collect_jvs_nonparallel_samples(args: &Args) -> Result<Vec<JvsSample>> {
    if !args.jvs_root.is_dir() {
        eprintln!(
            "warning: JVS root does not exist or is not a directory: {}",
            args.jvs_root.display()
        );
        return Ok(Vec::new());
    }

    let mut speaker_dirs = fs::read_dir(&args.jvs_root)
        .with_context(|| format!("Failed to read {}", args.jvs_root.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    speaker_dirs.sort_by_key(std::fs::DirEntry::file_name);

    let mut samples = Vec::new();
    for speaker in speaker_dirs
        .into_iter()
        .filter(|entry| entry.path().is_dir())
    {
        if samples.len() >= args.max_total {
            break;
        }
        let speaker_name = speaker.file_name().to_string_lossy().into_owned();
        if !speaker_name.starts_with("jvs") {
            continue;
        }
        if samples
            .iter()
            .map(|sample: &JvsSample| sample.speaker.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            >= args.max_speakers
        {
            break;
        }

        let nonpara_dir = speaker.path().join("nonpara30");
        let transcripts = nonpara_dir.join("transcripts_utf8.txt");
        let wav_dir = nonpara_dir.join("wav24kHz16bit");
        if !transcripts.is_file() || !wav_dir.is_dir() {
            continue;
        }

        let texts = read_transcripts(&transcripts)?;
        let ids = selected_transcript_ids(args, &texts);
        for id in ids {
            if samples.len() >= args.max_total {
                break;
            }
            let wav_path = wav_dir.join(format!("{id}.wav"));
            if !wav_path.is_file() {
                continue;
            }
            let expected = texts
                .get(&id)
                .expect("id was collected from transcript map")
                .clone();
            samples.push(JvsSample {
                speaker: speaker_name.clone(),
                id: id.clone(),
                parts: vec![JvsSamplePart {
                    id,
                    wav_path,
                    expected: expected.clone(),
                }],
                expected,
            });
        }
    }

    Ok(samples)
}

fn selected_transcript_ids(args: &Args, texts: &HashMap<String, String>) -> Vec<String> {
    if let Some(ids) = &args.ids {
        return ids
            .iter()
            .filter(|id| texts.contains_key(*id))
            .cloned()
            .collect();
    }

    let mut ids = texts.keys().cloned().collect::<Vec<_>>();
    ids.sort();
    ids.truncate(args.max_utterances_per_speaker);
    ids
}

fn group_samples(samples: Vec<JvsSample>, args: &Args) -> Vec<JvsSample> {
    if args.concat_size == 1 {
        return samples;
    }

    let mut grouped = Vec::new();
    let mut current_speaker = String::new();
    let mut current = Vec::new();
    for sample in samples {
        if current_speaker != sample.speaker {
            grouped.extend(drain_grouped_samples(&mut current, args.concat_size));
            current_speaker.clone_from(&sample.speaker);
        }
        current.push(sample);
    }
    grouped.extend(drain_grouped_samples(&mut current, args.concat_size));
    grouped
}

fn drain_grouped_samples(samples: &mut Vec<JvsSample>, concat_size: usize) -> Vec<JvsSample> {
    let mut grouped = Vec::new();
    let drained = std::mem::take(samples);
    for chunk in drained.chunks(concat_size) {
        if chunk.is_empty() {
            continue;
        }
        let speaker = chunk[0].speaker.clone();
        let id = chunk
            .iter()
            .map(|sample| sample.id.as_str())
            .collect::<Vec<_>>()
            .join("+");
        let wav_paths = chunk
            .iter()
            .flat_map(|sample| sample.parts.iter())
            .map(|part| JvsSamplePart {
                id: part.id.clone(),
                wav_path: part.wav_path.clone(),
                expected: part.expected.clone(),
            })
            .collect();
        let expected = chunk
            .iter()
            .map(|sample| sample.expected.as_str())
            .collect::<String>();
        grouped.push(JvsSample {
            speaker,
            id,
            parts: wav_paths,
            expected,
        });
    }
    grouped
}

fn read_transcripts(path: &Path) -> Result<HashMap<String, String>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let mut transcripts = HashMap::new();
    for line in content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let (id, text) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("Invalid transcript line in {}: {line}", path.display()))?;
        transcripts.insert(id.to_string(), text.to_string());
    }
    Ok(transcripts)
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    for file_name in [ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE] {
        let path = model_dir.join(file_name);
        if !path.is_file() {
            bail!("ASR model file not found: {}", path.display());
        }
    }
    Ok(())
}

fn create_recognizer(model_dir: &Path, num_threads: i32) -> Result<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(model_dir.join(ENCODER_FILE).display().to_string()),
        decoder: Some(model_dir.join(DECODER_FILE).display().to_string()),
        joiner: Some(model_dir.join(JOINER_FILE).display().to_string()),
    };
    config.model_config.tokens = Some(model_dir.join(TOKENS_FILE).display().to_string());
    config.model_config.provider = Some("cpu".to_string());
    config.model_config.modeling_unit = Some("cjkchar".to_string());
    config.model_config.num_threads = num_threads;
    config.decoding_method = Some("greedy_search".to_string());
    config.max_active_paths = 1;

    OfflineRecognizer::create(&config)
        .ok_or_else(|| anyhow!("Failed to create sherpa-onnx recognizer"))
}

#[derive(Debug, Clone)]
struct RecognizedTranscript {
    text: String,
    tokens: Vec<RecognizedToken>,
}

impl RecognizedTranscript {
    fn first_token_start_sample(&self) -> Option<usize> {
        self.tokens
            .iter()
            .find_map(|token| token.start_sec)
            .map(seconds_to_sample)
    }

    fn last_token_end_sample(&self) -> Option<usize> {
        self.tokens
            .iter()
            .filter_map(RecognizedToken::end_sec)
            .next_back()
            .map(seconds_to_sample)
    }

    fn token_preview(&self) -> String {
        self.tokens
            .iter()
            .take(8)
            .map(|token| {
                format!(
                    "{:?}@{:?}+{:?}",
                    token.text,
                    token.start_sec.map(seconds_to_sample),
                    token.duration_sec.map(seconds_to_sample)
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn last_token_preview(&self) -> String {
        let mut tokens = self.tokens.iter().rev().take(8).collect::<Vec<_>>();
        tokens.reverse();
        tokens
            .into_iter()
            .map(|token| {
                format!(
                    "{:?}@{:?}+{:?}",
                    token.text,
                    token.start_sec.map(seconds_to_sample),
                    token.duration_sec.map(seconds_to_sample)
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn estimate_first_token_from_following_speech_run(
        &self,
        speech_ranges: &[std::ops::Range<usize>],
    ) -> Option<usize> {
        let first = self.tokens.first()?.start_sample()?;
        let second = self.tokens.get(1)?.start_sample()?;
        let second_run = speech_ranges.iter().find(|range| range.contains(&second))?;
        if second_run.contains(&first) {
            return None;
        }

        let points = self
            .tokens
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(index, token)| {
                let sample = token.start_sample()?;
                second_run
                    .contains(&sample)
                    .then_some((index as f64, sample as f64))
            })
            .take(8)
            .collect::<Vec<_>>();
        estimate_y_at_x_zero(&points)
            .map(|sample| sample.clamp(second_run.start, second_run.end.saturating_sub(1)))
    }

    fn estimate_last_token_from_previous_speech_run(
        &self,
        speech_ranges: &[std::ops::Range<usize>],
    ) -> Option<usize> {
        let last_index = self.tokens.len().checked_sub(1)?;
        let last = self.tokens.get(last_index)?.start_sample()?;
        let previous_run = self.tokens.iter().rev().find_map(|token| {
            let sample = token.start_sample()?;
            speech_ranges.iter().find(|range| range.contains(&sample))
        })?;
        if previous_run.contains(&last) {
            return None;
        }

        let points = self
            .tokens
            .iter()
            .enumerate()
            .rev()
            .skip(1)
            .filter_map(|(index, token)| {
                let sample = token.start_sample()?;
                previous_run
                    .contains(&sample)
                    .then_some((index as f64, sample as f64))
            })
            .take(8)
            .collect::<Vec<_>>();
        estimate_y_at_x(&points, last_index as f64)?;
        Some(previous_run.end)
    }

    fn silence_token_report(
        &self,
        speech_ranges: &[std::ops::Range<usize>],
        audio_len: usize,
    ) -> String {
        let mut tokens = self
            .tokens
            .iter()
            .filter_map(|token| {
                let sample = token.start_sample()?;
                (!is_in_any_range(sample, speech_ranges)).then_some((sample, token.text.as_str()))
            })
            .collect::<Vec<_>>();
        tokens.sort_by_key(|(sample, _)| *sample);
        if tokens.is_empty() {
            return "none".to_string();
        }
        let text = tokens
            .iter()
            .map(|(_, token)| *token)
            .collect::<Vec<_>>()
            .join("");
        let preview = tokens
            .iter()
            .take(16)
            .map(|(sample, token)| format!("{token}@{sample}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "count={} chars={} audio_len={} preview=[{}]",
            tokens.len(),
            text,
            audio_len,
            preview
        )
    }
}

#[derive(Debug, Clone)]
struct RecognizedToken {
    text: String,
    start_sec: Option<f32>,
    duration_sec: Option<f32>,
}

impl RecognizedToken {
    fn start_sample(&self) -> Option<usize> {
        self.start_sec.map(seconds_to_sample)
    }

    fn end_sec(&self) -> Option<f32> {
        let start = self.start_sec?;
        Some(start + self.duration_sec.unwrap_or(0.0).max(0.0))
    }
}

fn seconds_to_sample(seconds: f32) -> usize {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * ASR_SAMPLE_RATE as f32).round().max(0.0) as usize
}

fn estimate_y_at_x_zero(points: &[(f64, f64)]) -> Option<usize> {
    estimate_y_at_x(points, 0.0)
}

fn estimate_y_at_x(points: &[(f64, f64)], x: f64) -> Option<usize> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let denominator = points
        .iter()
        .map(|(x, _)| (x - mean_x).powi(2))
        .sum::<f64>();
    if denominator <= f64::EPSILON {
        return None;
    }
    let slope = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / denominator;
    let y = mean_y + slope * (x - mean_x);
    y.is_finite().then_some(y.round().max(0.0) as usize)
}

fn is_in_any_range(sample: usize, ranges: &[std::ops::Range<usize>]) -> bool {
    ranges.iter().any(|range| range.contains(&sample))
}

fn transcribe(recognizer: &OfflineRecognizer, samples: &[f32]) -> Result<RecognizedTranscript> {
    let stream = recognizer.create_stream();
    stream.accept_waveform(
        i32::try_from(ASR_SAMPLE_RATE).expect("ASR sample rate should fit i32"),
        samples,
    );
    recognizer.decode(&stream);
    let result = stream
        .get_result()
        .ok_or_else(|| anyhow!("Failed to fetch sherpa-onnx result"))?;
    let tokens = result
        .tokens
        .into_iter()
        .enumerate()
        .map(|(index, text)| RecognizedToken {
            text,
            start_sec: result
                .timestamps
                .as_ref()
                .and_then(|timestamps| timestamps.get(index))
                .copied(),
            duration_sec: result
                .durations
                .as_ref()
                .and_then(|durations| durations.get(index))
                .copied(),
        })
        .collect();
    Ok(RecognizedTranscript {
        text: result.text.trim().to_string(),
        tokens,
    })
}

#[derive(Serialize)]
struct PlotRecord {
    id: String,
    expected: String,
    raw_text: String,
    recognized_text: String,
    sample_rate: u32,
    vad_threshold: f32,
    audio: Vec<f32>,
    speech_ranges: Vec<PlotRange>,
    raw_vad_frames: Vec<RawVadFrame>,
    tokens: Vec<PlotToken>,
}

#[derive(Serialize)]
struct PlotRange {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RawVadFrame {
    start: usize,
    end: usize,
    probability: f32,
    is_speech: bool,
}

#[derive(Serialize)]
struct PlotToken {
    text: String,
    sample: Option<usize>,
}

struct RawSileroVadEngine {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
    threshold: f32,
}

impl RawSileroVadEngine {
    fn new(model_path: &Path, threshold: f32) -> Result<Self> {
        if !model_path.is_file() {
            bail!("VAD model not found: {}", model_path.display());
        }

        ort::init()
            .with_name("verify_jvs_asr")
            .with_telemetry(false)
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit();

        let session = Session::builder()
            .map_err(|err| anyhow!("Failed to create VAD session builder: {err}"))?
            .with_intra_threads(1)
            .map_err(|err| anyhow!("Failed to configure VAD session: {err}"))?
            .commit_from_file(model_path)
            .map_err(|err| anyhow!("Failed to load VAD model {}: {err}", model_path.display()))?;

        Ok(Self {
            session,
            state: vec![0.0; SILERO_STATE_LEN],
            context: vec![0.0; SILERO_CONTEXT_SAMPLES],
            threshold,
        })
    }

    fn process_audio(&mut self, audio: &[f32]) -> Result<Vec<RawVadFrame>> {
        self.reset_state();
        let mut frames = Vec::new();
        for start in (0..audio.len()).step_by(SILERO_CHUNK_SAMPLES) {
            let end = (start + SILERO_CHUNK_SAMPLES).min(audio.len());
            let probability = self.process_chunk(&audio[start..end])?;
            frames.push(RawVadFrame {
                start,
                end,
                probability,
                is_speech: probability > self.threshold,
            });
        }
        Ok(frames)
    }

    fn reset_state(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    fn process_chunk(&mut self, samples: &[f32]) -> Result<f32> {
        if samples.is_empty() {
            return Ok(0.0);
        }

        let mut chunk = [0.0; SILERO_CHUNK_SAMPLES];
        chunk[..samples.len()].copy_from_slice(samples);

        let mut input_samples = Vec::with_capacity(SILERO_INPUT_SAMPLES);
        input_samples.extend_from_slice(&self.context);
        input_samples.extend_from_slice(&chunk);

        let input = TensorRef::from_array_view((
            [1_usize, SILERO_INPUT_SAMPLES],
            input_samples.as_slice(),
        ))?;
        let sr = Tensor::from_array(((), vec![i64::from(ASR_SAMPLE_RATE)]))?;
        let state = TensorRef::from_array_view(([2_usize, 1, 128], self.state.as_slice()))?;

        let outputs = self.session.run(inputs![
            "input" => input,
            "sr" => sr,
            "state" => state,
        ])?;

        let (_, out) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, state_out) = outputs[1].try_extract_tensor::<f32>()?;

        if state_out.len() == self.state.len() {
            self.state.copy_from_slice(state_out);
        }
        self.context
            .copy_from_slice(&chunk[SILERO_CHUNK_SAMPLES - SILERO_CONTEXT_SAMPLES..]);

        Ok(out.first().copied().unwrap_or(0.0))
    }
}

fn speech_span_from_raw_vad(
    frames: &[RawVadFrame],
    audio_len: usize,
) -> Option<std::ops::Range<usize>> {
    let first = frames.iter().find(|frame| frame.is_speech)?;
    let last = frames.iter().rev().find(|frame| frame.is_speech)?;
    Some(first.start.min(audio_len)..last.end.min(audio_len)).filter(|range| !range.is_empty())
}

fn add_asr_edge_silence(
    audio: &mut Vec<f32>,
    speech_ranges: &mut [std::ops::Range<usize>],
    edge_silence_ms: u32,
    fade_ms: u32,
) {
    if audio.is_empty() {
        return;
    }
    let edge_silence = frames_for_millis(ASR_SAMPLE_RATE, edge_silence_ms);
    let fade_samples = frames_for_millis(ASR_SAMPLE_RATE, fade_ms);
    if edge_silence == 0 {
        return;
    }

    let original = std::mem::take(audio);
    audio.reserve(edge_silence + original.len() + edge_silence);
    audio.resize(edge_silence, 0.0);
    audio.extend_from_slice(&original);
    audio.resize(edge_silence + original.len() + edge_silence, 0.0);

    let original_start = edge_silence;
    let original_end = original_start + original.len();
    apply_fade_in(&mut audio[original_start..original_end], fade_samples);
    apply_fade_out(&mut audio[original_start..original_end], fade_samples);

    for range in speech_ranges {
        range.start += edge_silence;
        range.end += edge_silence;
    }
}

fn apply_fade_in(audio: &mut [f32], fade_samples: usize) {
    let fade_samples = fade_samples.min(audio.len());
    if fade_samples == 0 {
        return;
    }
    for (index, sample) in audio.iter_mut().take(fade_samples).enumerate() {
        let gain = index as f32 / fade_samples as f32;
        *sample *= gain;
    }
}

fn apply_fade_out(audio: &mut [f32], fade_samples: usize) {
    let fade_samples = fade_samples.min(audio.len());
    if fade_samples == 0 {
        return;
    }
    let start = audio.len() - fade_samples;
    for (index, sample) in audio[start..].iter_mut().enumerate() {
        let gain = (fade_samples - index) as f32 / fade_samples as f32;
        *sample *= gain;
    }
}

fn leading_gap_before(
    before_sample: usize,
    speech_ranges: &[std::ops::Range<usize>],
) -> Option<std::ops::Range<usize>> {
    if before_sample == 0 {
        return None;
    }
    let previous = speech_ranges
        .iter()
        .filter(|range| range.end <= before_sample)
        .collect::<Vec<_>>();
    if let (Some(first), Some(last)) = (previous.first(), previous.last()) {
        return Some(first.start..last.end).filter(|range| !range.is_empty());
    }
    speech_ranges
        .iter()
        .find(|range| range.start < before_sample && before_sample <= range.end)
        .map(|range| range.start..range.end.min(before_sample))
        .filter(|range| !range.is_empty())
}

#[derive(Debug)]
struct WavAudio {
    sample_rate: u32,
    samples: Vec<f32>,
}

fn read_wav_mono_f32(path: &Path) -> Result<WavAudio> {
    let bytes = fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("Unsupported WAV header: {}", path.display());
    }

    let mut cursor = 12usize;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data = None;

    while cursor + 8 <= bytes.len() {
        let id = [
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ];
        let len = read_le_u32_at(&bytes, cursor + 4, path, "chunk length")? as usize;
        cursor += 8;
        if cursor + len > bytes.len() {
            bail!("Invalid WAV chunk length in {}", path.display());
        }

        match &id {
            b"fmt " => {
                if len < 16 {
                    bail!("Invalid WAV fmt chunk in {}", path.display());
                }
                let audio_format = read_le_u16_at(&bytes, cursor, path, "audio format")?;
                if audio_format != 1 {
                    bail!("Only PCM WAV is supported: {}", path.display());
                }
                channels = Some(read_le_u16_at(&bytes, cursor + 2, path, "channel count")?);
                sample_rate = Some(read_le_u32_at(&bytes, cursor + 4, path, "sample rate")?);
                bits_per_sample = Some(read_le_u16_at(
                    &bytes,
                    cursor + 14,
                    path,
                    "bits per sample",
                )?);
            }
            b"data" => data = Some(bytes[cursor..cursor + len].to_vec()),
            _ => {}
        }

        let padded_len = len
            .checked_add(len % 2)
            .ok_or_else(|| anyhow!("Invalid WAV chunk padding length in {}", path.display()))?;
        cursor = cursor
            .checked_add(padded_len)
            .ok_or_else(|| anyhow!("Invalid WAV cursor overflow in {}", path.display()))?;
        if cursor > bytes.len() {
            bail!("Invalid WAV chunk padding in {}", path.display());
        }
    }

    let channels =
        channels.ok_or_else(|| anyhow!("WAV fmt chunk not found: {}", path.display()))?;
    let sample_rate =
        sample_rate.ok_or_else(|| anyhow!("WAV sample rate not found: {}", path.display()))?;
    let bits_per_sample =
        bits_per_sample.ok_or_else(|| anyhow!("WAV bits not found: {}", path.display()))?;
    if bits_per_sample != 16 {
        bail!("Only 16-bit WAV is supported: {}", path.display());
    }
    let data = data.ok_or_else(|| anyhow!("WAV data chunk not found: {}", path.display()))?;
    if channels == 0 {
        bail!("WAV channel count is zero: {}", path.display());
    }

    let frame_bytes = usize::from(channels) * 2;
    if data.len() % frame_bytes != 0 {
        bail!("WAV data is not aligned to frames: {}", path.display());
    }

    let mut samples = Vec::with_capacity(data.len() / frame_bytes);
    for frame in data.chunks_exact(frame_bytes) {
        let mut sum = 0.0f32;
        for channel in 0..usize::from(channels) {
            let offset = channel * 2;
            let value = i16::from_le_bytes([frame[offset], frame[offset + 1]]);
            sum += f32::from(value) / 32_768.0;
        }
        samples.push(sum / f32::from(channels));
    }

    Ok(WavAudio {
        sample_rate,
        samples,
    })
}

fn read_le_u16_at(bytes: &[u8], offset: usize, path: &Path, field: &str) -> Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| anyhow!("Invalid WAV {field} in {}", path.display()))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_le_u32_at(bytes: &[u8], offset: usize, path: &Path, field: &str) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("Invalid WAV {field} in {}", path.display()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn resample_to_asr_rate(samples: &[f32], source_sample_rate: u32) -> Result<Vec<f32>> {
    if source_sample_rate == ASR_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }

    let input_chunk_size = frames_for_millis(source_sample_rate, 100);
    let mut resampler = Async::<f32>::new_poly(
        f64::from(ASR_SAMPLE_RATE) / f64::from(source_sample_rate),
        1.0,
        PolynomialDegree::Cubic,
        input_chunk_size,
        1,
        FixedAsync::Input,
    )
    .context("Failed to create resampler")?;
    let mut output_buffer = vec![0.0; resampler.output_frames_max()];
    let mut output = Vec::new();
    let mut cursor = 0usize;

    while cursor < samples.len() {
        let end = (cursor + input_chunk_size).min(samples.len());
        let mut input = samples[cursor..end].to_vec();
        if input.len() < input_chunk_size {
            input.resize(input_chunk_size, 0.0);
        }

        let input_adapter = SingleChannelInputAdapter::new(&input);
        let mut output_adapter = SingleChannelOutputAdapter::new(&mut output_buffer);
        let (_, written) =
            resampler.process_into_buffer(&input_adapter, &mut output_adapter, None)?;
        output.extend_from_slice(&output_buffer[..written]);
        cursor = end;
    }

    Ok(output)
}

fn frames_for_millis(sample_rate: u32, millis: u32) -> usize {
    ((u64::from(sample_rate) * u64::from(millis)) / 1000)
        .try_into()
        .unwrap_or(1)
}

struct SingleChannelInputAdapter<'a> {
    data: &'a [f32],
}

impl<'a> SingleChannelInputAdapter<'a> {
    fn new(data: &'a [f32]) -> Self {
        Self { data }
    }
}

impl<'a> Adapter<'a, f32> for SingleChannelInputAdapter<'a> {
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> f32 {
        debug_assert_eq!(channel, 0);
        // SAFETY: rubato calls this adapter with frame < self.frames() and the
        // adapter exposes exactly one channel.
        unsafe { *self.data.get_unchecked(frame) }
    }

    fn channels(&self) -> usize {
        1
    }

    fn frames(&self) -> usize {
        self.data.len()
    }
}

struct SingleChannelOutputAdapter<'a> {
    data: &'a mut [f32],
}

impl<'a> SingleChannelOutputAdapter<'a> {
    fn new(data: &'a mut [f32]) -> Self {
        Self { data }
    }
}

impl<'a> Adapter<'a, f32> for SingleChannelOutputAdapter<'a> {
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> f32 {
        debug_assert_eq!(channel, 0);
        // SAFETY: rubato calls this adapter with frame < self.frames() and the
        // adapter exposes exactly one channel.
        unsafe { *self.data.get_unchecked(frame) }
    }

    fn channels(&self) -> usize {
        1
    }

    fn frames(&self) -> usize {
        self.data.len()
    }
}

impl<'a> AdapterMut<'a, f32> for SingleChannelOutputAdapter<'a> {
    unsafe fn write_sample_unchecked(&mut self, channel: usize, frame: usize, value: &f32) -> bool {
        debug_assert_eq!(channel, 0);
        // SAFETY: rubato calls this adapter with frame < self.frames() and the
        // adapter exposes exactly one mutable output channel.
        unsafe {
            *self.data.get_unchecked_mut(frame) = *value;
        }
        false
    }
}

#[derive(Debug)]
struct EvaluatedSample {
    speaker: String,
    id: String,
    expected: String,
    recognized: RecognizedTranscript,
    raw_recognized: RecognizedTranscript,
    expected_normalized: String,
    recognized_normalized: String,
    cer: f64,
    asr_samples: usize,
    speech_ranges: Vec<std::ops::Range<usize>>,
    parts: Vec<PartDiagnostic>,
    audio_stats: AudioStats,
    first_token_start_sample: Option<usize>,
    last_token_end_sample: Option<usize>,
    token_count: usize,
    legacy_interim: Option<LegacyInterimEvaluation>,
}

#[derive(Debug)]
struct LegacyInterimEvaluation {
    recognized: RecognizedTranscript,
    recognized_normalized: String,
    cer: f64,
    asr_samples: usize,
    duplicated_samples: usize,
    split_count: usize,
}

struct LegacyInterimAudio {
    audio: Vec<f32>,
    duplicated_samples: usize,
    split_count: usize,
}

impl EvaluatedSample {
    fn speech_ranges(&self) -> Vec<std::ops::Range<usize>> {
        self.speech_ranges.clone()
    }

    fn timestamp_vad_leading_disagreement(&self) -> Option<std::ops::Range<usize>> {
        let before_sample = self.first_token_start_sample?;
        self.timestamp_vad_leading_disagreement_before(before_sample)
    }

    fn timestamp_vad_leading_disagreement_after_first_token_estimation(
        &self,
    ) -> Option<std::ops::Range<usize>> {
        let before_sample = self
            .raw_recognized
            .estimate_first_token_from_following_speech_run(&self.speech_ranges())?;
        self.timestamp_vad_leading_disagreement_before(before_sample)
    }

    fn timestamp_vad_leading_disagreement_before(
        &self,
        before_sample: usize,
    ) -> Option<std::ops::Range<usize>> {
        leading_gap_before(before_sample, &self.speech_ranges())
    }

    fn timestamp_vad_trailing_disagreement(&self) -> Option<std::ops::Range<usize>> {
        let after_sample = self.last_token_end_sample?;
        if after_sample >= self.asr_samples {
            return None;
        }
        self.speech_ranges
            .iter()
            .find(|range| range.end > after_sample)
            .map(|range| range.start.max(after_sample)..range.end)
            .filter(|range| !range.is_empty())
    }
}

#[derive(Debug)]
struct AudioStats {
    speech: SignalStats,
    silence: SignalStats,
    silence_ranges: Vec<std::ops::Range<usize>>,
}

impl AudioStats {
    fn from_audio_and_speech_ranges(
        audio: &[f32],
        speech_ranges: &[std::ops::Range<usize>],
    ) -> Self {
        let silence_ranges = complement_ranges(audio.len(), speech_ranges);
        Self {
            speech: SignalStats::from_ranges(audio, speech_ranges),
            silence: SignalStats::from_ranges(audio, &silence_ranges),
            silence_ranges,
        }
    }

    fn report(&self) -> String {
        format!(
            "volume speech={} silence={} silence_ranges={}",
            self.speech.report(),
            self.silence.report(),
            self.silence_ranges
                .iter()
                .map(|range| format!("{}..{}", range.start, range.end))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

#[derive(Debug)]
struct SignalStats {
    samples: usize,
    rms: f32,
    peak: f32,
    mean_abs: f32,
}

impl SignalStats {
    fn from_ranges(audio: &[f32], ranges: &[std::ops::Range<usize>]) -> Self {
        let mut samples = 0usize;
        let mut sum_square = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut peak = 0.0f32;
        for range in ranges {
            for sample in &audio[range.start.min(audio.len())..range.end.min(audio.len())] {
                let abs = sample.abs();
                samples += 1;
                sum_square += f64::from(*sample) * f64::from(*sample);
                sum_abs += f64::from(abs);
                peak = peak.max(abs);
            }
        }
        if samples == 0 {
            return Self {
                samples: 0,
                rms: 0.0,
                peak: 0.0,
                mean_abs: 0.0,
            };
        }
        Self {
            samples,
            rms: (sum_square / samples as f64).sqrt() as f32,
            peak,
            mean_abs: (sum_abs / samples as f64) as f32,
        }
    }

    fn report(&self) -> String {
        format!(
            "samples={} rms={:.6} peak={:.6} mean_abs={:.6}",
            self.samples, self.rms, self.peak, self.mean_abs
        )
    }
}

fn complement_ranges(len: usize, ranges: &[std::ops::Range<usize>]) -> Vec<std::ops::Range<usize>> {
    let mut sorted = ranges.to_vec();
    sorted.sort_by_key(|range| range.start);
    let mut complement = Vec::new();
    let mut cursor = 0usize;
    for range in sorted {
        let start = range.start.min(len);
        let end = range.end.min(len);
        if cursor < start {
            complement.push(cursor..start);
        }
        cursor = cursor.max(end);
    }
    if cursor < len {
        complement.push(cursor..len);
    }
    complement
}

#[derive(Debug)]
struct PartDiagnostic {
    id: String,
    source_sample_rate: u32,
    source_samples: usize,
    asr_samples: usize,
    timeline_start_sample: usize,
    timeline_end_sample: usize,
    expected: String,
    standalone_recognized: RecognizedTranscript,
    standalone_cer: f64,
}

fn normalize_for_jvs_asr_check(text: &str) -> String {
    text.nfkc()
        .filter(|ch| !ch.is_whitespace() && !is_combining_mark(*ch) && !is_ignored_punctuation(*ch))
        .collect()
}

fn is_ignored_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '、' | '。'
            | '，'
            | '．'
            | ','
            | '.'
            | '!'
            | '?'
            | '！'
            | '？'
            | ':'
            | ';'
            | '：'
            | '；'
            | '"'
            | '\''
            | '“'
            | '”'
            | '‘'
            | '’'
            | '('
            | ')'
            | '（'
            | '）'
            | '['
            | ']'
            | '「'
            | '」'
            | '『'
            | '』'
    )
}

fn char_error_rate(expected: &str, actual: &str) -> f64 {
    let expected = expected.chars().collect::<Vec<_>>();
    let actual = actual.chars().collect::<Vec<_>>();
    if expected.is_empty() {
        return if actual.is_empty() { 0.0 } else { 1.0 };
    }

    let mut previous = (0..=actual.len()).collect::<Vec<_>>();
    let mut current = vec![0; actual.len() + 1];
    for (expected_index, expected_char) in expected.iter().enumerate() {
        current[0] = expected_index + 1;
        for (actual_index, actual_char) in actual.iter().enumerate() {
            let substitution = usize::from(expected_char != actual_char);
            current[actual_index + 1] = (previous[actual_index + 1] + 1)
                .min(current[actual_index] + 1)
                .min(previous[actual_index] + substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[actual.len()] as f64 / expected.len() as f64
}
