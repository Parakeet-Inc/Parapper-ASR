//! Derives a noise-cancelled and/or edge-padded variant of an ASR eval split.
//!
//! The tool reads an existing runner manifest plus its canonical PCM16 WAVs and
//! writes a new split directory containing rewritten audio and a derived
//! manifest. Two independent transforms are available:
//!
//! * `--apply-nc` runs the production UL-UNAS noise-cancellation model over each
//!   utterance with a fresh (per-utterance) engine, then removes the model's
//!   one-hop algorithmic delay so the derived audio stays sample-aligned with
//!   the source.
//! * `--pad-ms` bakes zero-valued edge silence into both edges.
//!
//! When both are requested the order is NC first, then padding, matching the
//! production NC -> VAD -> edge-silence pipeline ordering.
//!
//! The derived split is a drop-in input for the unmodified `run_asr_eval` bin,
//! which enables an A/B comparison of noise cancellation without touching any
//! production code path.
#![allow(clippy::cast_possible_truncation)]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{
    DerivedAudio, RunnerManifestV1, RunnerSampleV1, decode_canonical_pcm16_wav,
};
use parapper_models::nc::{NoiseCancellationEngine, UlUnasNoiseCancellationEngine};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// UL-UNAS streams a 512-point STFT with this hop; it is also the exact output
/// lag of `UlUnasNoiseCancellationEngine::process`.
const HOP_SAMPLES: usize = 256;
const SAMPLE_RATE_HZ: usize = 16_000;
const NC_MODEL_FILE: &str = "ulunas_stream_simple.onnx";
const NC_ALIGNMENT_NOTE: &str =
    "dropped 256-sample leading lag; flushed with trailing zeros; output length == source length";
const PROGRESS_INTERVAL: usize = 50;

const USAGE: &str = "Usage:
  prepare_nc_derived_split \\
    --manifest <source-manifest.json> \\
    --split-id <source-split-id> \\
    --new-split-id <derived-split-id> \\
    --output-dir <derived-split-directory> \\
    [--apply-nc --nc-model-dir <ul-unas-directory> --ort-dylib <onnxruntime-library>] \\
    [--pad-ms <milliseconds-of-edge-silence>] \\
    [--limit <positive-integer>]

At least one of --apply-nc / --pad-ms must be given.";

fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };

    let started = Instant::now();
    let manifest_bytes = fs::read(&args.manifest)
        .with_context(|| format!("failed to read {}", args.manifest.display()))?;
    let source_manifest_sha256 = hex_sha256(&manifest_bytes);
    let manifest = RunnerManifestV1::parse(&manifest_bytes)
        .with_context(|| format!("failed to preflight {}", args.manifest.display()))?;
    if manifest.split_id != args.split_id {
        bail!(
            "split ID mismatch: CLI requested {:?}, manifest contains {:?}",
            args.split_id,
            manifest.split_id
        );
    }
    let manifest_root = args
        .manifest
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);

    let derived_manifest_path = args.output_dir.join("manifest.json");
    if derived_manifest_path.exists() {
        bail!(
            "derived manifest already exists (overwrite is not implemented): {}",
            derived_manifest_path.display()
        );
    }
    let wav_dir = args.output_dir.join("wav");
    fs::create_dir_all(&wav_dir)
        .with_context(|| format!("failed to create {}", wav_dir.display()))?;

    let nc_model_sha256 = if args.apply_nc {
        Some(arm_noise_cancellation(&args)?)
    } else {
        None
    };

    let selected = args.limit.unwrap_or(manifest.samples.len());
    let sources = &manifest.samples[..selected.min(manifest.samples.len())];
    let pad_samples = pad_edge_samples(args.pad_ms);

    let mut derived_samples = Vec::with_capacity(sources.len());
    let mut total_input_samples = 0_u64;
    let mut total_output_samples = 0_u64;
    let mut clipped_sample_count = 0_u64;
    let transform = Transform {
        manifest_root: &manifest_root,
        wav_dir: &wav_dir,
        nc_model_dir: if args.apply_nc {
            Some(
                args.nc_model_dir
                    .as_deref()
                    .context("--apply-nc requires --nc-model-dir")?,
            )
        } else {
            None
        },
        pad_samples,
    };
    for (index, sample) in sources.iter().enumerate() {
        let (derived, stats) = transform
            .apply(sample)
            .with_context(|| format!("failed to derive {}", sample.utterance_id))?;

        total_input_samples += stats.input_samples;
        total_output_samples += stats.output_samples;
        clipped_sample_count += stats.clipped;
        derived_samples.push(derived);

        if (index + 1) % PROGRESS_INTERVAL == 0 || index + 1 == sources.len() {
            eprintln!("processed {}/{} utterances", index + 1, sources.len());
        }
    }

    let provenance = build_provenance(
        &manifest.split_id,
        &source_manifest_sha256,
        args.apply_nc,
        nc_model_sha256.as_deref(),
        args.pad_ms,
    );
    let derived_manifest =
        build_derived_manifest(&manifest, &args.new_split_id, derived_samples, provenance)?;
    let reparsed = write_and_verify_manifest(
        &derived_manifest_path,
        &derived_manifest,
        &args.new_split_id,
    )?;

    println!(
        "{}",
        serde_json::to_string_pretty(&CommandSummary {
            schema_version: 1,
            new_split_id: &args.new_split_id,
            source_split_id: &manifest.split_id,
            samples_written: reparsed.samples.len(),
            apply_nc: args.apply_nc,
            pad_ms: args.pad_ms,
            total_input_samples,
            total_output_samples,
            clipped_sample_count,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        })?
    );
    Ok(())
}

/// Points ORT at the requested dynamic library and hashes the NC model.
///
/// Returns the model's SHA-256 so the derived manifest can record exactly which
/// artifact produced its audio.
fn arm_noise_cancellation(args: &Args) -> Result<String> {
    let model_dir = args
        .nc_model_dir
        .as_ref()
        .context("--apply-nc requires --nc-model-dir")?;
    let ort_dylib = args
        .ort_dylib
        .as_ref()
        .context("--apply-nc requires --ort-dylib")?;
    if !ort_dylib.is_file() {
        bail!(
            "ONNX Runtime library does not exist: {}",
            ort_dylib.display()
        );
    }
    let model_path = model_dir.join(NC_MODEL_FILE);
    let model_bytes = fs::read(&model_path)
        .with_context(|| format!("failed to read {}", model_path.display()))?;

    // SAFETY: this CLI is single-threaded until after this assignment and no
    // ONNX Runtime API is touched before it. ORT's dynamic loader reads this
    // process-global variable on the first API use.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", ort_dylib) };

    Ok(hex_sha256(&model_bytes))
}

/// Writes the derived manifest, then re-reads it through the runner's own
/// preflight so a split that `run_asr_eval` would reject never ships.
fn write_and_verify_manifest(
    path: &Path,
    manifest: &Value,
    expected_split_id: &str,
) -> Result<RunnerManifestV1> {
    let bytes =
        serde_json::to_vec_pretty(manifest).context("failed to serialize the derived manifest")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;

    let written =
        fs::read(path).with_context(|| format!("failed to re-read {}", path.display()))?;
    let reparsed = RunnerManifestV1::parse(&written).with_context(|| {
        format!(
            "derived manifest failed its own preflight: {}",
            path.display()
        )
    })?;
    if reparsed.split_id != expected_split_id {
        bail!(
            "derived manifest split ID is {:?}, expected {expected_split_id:?}",
            reparsed.split_id
        );
    }
    Ok(reparsed)
}

/// The per-utterance transform, applied identically to every sample.
#[derive(Debug)]
struct Transform<'a> {
    manifest_root: &'a Path,
    wav_dir: &'a Path,
    /// `Some` enables UL-UNAS noise cancellation with a fresh per-utterance engine.
    nc_model_dir: Option<&'a Path>,
    pad_samples: usize,
}

#[derive(Debug)]
struct SampleStats {
    input_samples: u64,
    output_samples: u64,
    clipped: u64,
}

impl Transform<'_> {
    /// Rewrites one utterance and returns its derived manifest entry.
    fn apply(&self, sample: &RunnerSampleV1) -> Result<(RunnerSampleV1, SampleStats)> {
        let source_path = self.manifest_root.join(&sample.audio.relative_path);
        let source_bytes = fs::read(&source_path)
            .with_context(|| format!("failed to read {}", source_path.display()))?;
        let actual_sha256 = hex_sha256(&source_bytes);
        if actual_sha256 != sample.audio.sha256 {
            bail!(
                "audio SHA-256 mismatch: manifest {}, file {actual_sha256}",
                sample.audio.sha256
            );
        }
        let wav = decode_canonical_pcm16_wav(&source_bytes)
            .context("failed to decode canonical audio")?;

        let denoised = match self.nc_model_dir {
            Some(model_dir) => {
                let mut engine = UlUnasNoiseCancellationEngine::new(model_dir)
                    .context("failed to construct the UL-UNAS engine")?;
                denoise_utterance(&mut engine, &wav.samples)
                    .context("failed to noise-cancel the utterance")?
            }
            None => wav.samples.clone(),
        };
        let padded = pad_edges(&denoised, self.pad_samples);
        let converted = to_pcm16(&padded);
        if converted.non_finite > 0 {
            bail!(
                "processing produced {} non-finite samples",
                converted.non_finite
            );
        }

        let file_name = Path::new(&sample.audio.relative_path)
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| {
                format!(
                    "audio path has no file name: {}",
                    sample.audio.relative_path
                )
            })?
            .to_owned();
        let bytes = pcm16_wav(&converted.samples);
        let derived_path = self.wav_dir.join(&file_name);
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&derived_path)
            .with_context(|| format!("failed to create {}", derived_path.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("failed to write {}", derived_path.display()))?;
        file.flush()
            .with_context(|| format!("failed to flush {}", derived_path.display()))?;

        Ok((
            RunnerSampleV1 {
                utterance_id: sample.utterance_id.clone(),
                audio: DerivedAudio {
                    relative_path: format!("wav/{file_name}"),
                    sha256: hex_sha256(&bytes),
                    duration_samples: converted.samples.len() as u64,
                },
                reference: sample.reference.clone(),
            },
            SampleStats {
                input_samples: wav.samples.len() as u64,
                output_samples: converted.samples.len() as u64,
                clipped: converted.clipped,
            },
        ))
    }
}

#[derive(Debug, Serialize)]
struct CommandSummary<'a> {
    schema_version: u32,
    new_split_id: &'a str,
    source_split_id: &'a str,
    samples_written: usize,
    apply_nc: bool,
    pad_ms: u32,
    total_input_samples: u64,
    total_output_samples: u64,
    clipped_sample_count: u64,
    elapsed_seconds: f64,
}

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    new_split_id: String,
    output_dir: PathBuf,
    nc_model_dir: Option<PathBuf>,
    ort_dylib: Option<PathBuf>,
    apply_nc: bool,
    pad_ms: u32,
    limit: Option<usize>,
}

impl Args {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Option<Self>> {
        let arguments = arguments.collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            return Ok(None);
        }
        for argument in arguments
            .iter()
            .filter(|argument| argument.starts_with('-'))
        {
            if ![
                "--manifest",
                "--split-id",
                "--new-split-id",
                "--output-dir",
                "--nc-model-dir",
                "--ort-dylib",
                "--apply-nc",
                "--pad-ms",
                "--limit",
            ]
            .contains(&argument.as_str())
            {
                bail!("unknown argument {argument}\n{USAGE}");
            }
        }
        let optional = |name: &str| -> Result<Option<&str>> {
            let Some(index) = arguments.iter().position(|argument| argument == name) else {
                return Ok(None);
            };
            arguments
                .get(index + 1)
                .filter(|candidate| !candidate.starts_with("--"))
                .map(String::as_str)
                .map(Some)
                .with_context(|| format!("missing value for {name}\n{USAGE}"))
        };
        let required = |name: &str| -> Result<&str> {
            optional(name)?.with_context(|| format!("missing required argument {name}\n{USAGE}"))
        };

        let apply_nc = arguments.iter().any(|argument| argument == "--apply-nc");
        let pad_ms = match optional("--pad-ms")? {
            Some(raw) => raw
                .parse::<u32>()
                .context("--pad-ms must be a non-negative integer")?,
            None => 0,
        };
        if !apply_nc && pad_ms == 0 {
            bail!("at least one of --apply-nc / --pad-ms must be given\n{USAGE}");
        }
        let nc_model_dir = optional("--nc-model-dir")?.map(PathBuf::from);
        let ort_dylib = optional("--ort-dylib")?.map(PathBuf::from);
        if apply_nc && nc_model_dir.is_none() {
            bail!("--apply-nc requires --nc-model-dir\n{USAGE}");
        }
        if apply_nc && ort_dylib.is_none() {
            bail!("--apply-nc requires --ort-dylib\n{USAGE}");
        }
        let limit = match optional("--limit")? {
            Some(raw) => {
                let parsed = raw
                    .parse::<usize>()
                    .context("--limit must be a positive integer")?;
                if parsed == 0 {
                    bail!("--limit must be a positive integer");
                }
                Some(parsed)
            }
            None => None,
        };

        Ok(Some(Self {
            manifest: PathBuf::from(required("--manifest")?),
            split_id: required("--split-id")?.to_owned(),
            new_split_id: required("--new-split-id")?.to_owned(),
            output_dir: PathBuf::from(required("--output-dir")?),
            nc_model_dir,
            ort_dylib,
            apply_nc,
            pad_ms,
            limit,
        }))
    }
}

/// Number of trailing zeros required to flush every source sample out of the
/// streaming engine.
///
/// `process` only emits whole hops and lags its input by exactly one hop, so the
/// engine must consume one extra hop plus whatever it takes to round the source
/// length up to a hop boundary.
fn flush_padding_len(source_len: usize) -> usize {
    HOP_SAMPLES + ((HOP_SAMPLES - source_len % HOP_SAMPLES) % HOP_SAMPLES)
}

/// Removes the engine's one-hop algorithmic delay and truncates to the source length.
fn realign_denoised(output: &[f32], source_len: usize) -> Result<&[f32]> {
    let end = HOP_SAMPLES
        .checked_add(source_len)
        .context("realignment window overflowed")?;
    output.get(HOP_SAMPLES..end).with_context(|| {
        format!(
            "noise-cancelled stream is too short: got {} samples, need {end} to recover {source_len}",
            output.len()
        )
    })
}

/// Runs one utterance through a dedicated engine and returns a sample-aligned result.
///
/// The engine is fully stateful and exposes no reset, so callers must pass a
/// freshly constructed engine per utterance.
fn denoise_utterance(
    engine: &mut dyn NoiseCancellationEngine,
    samples: &[f32],
) -> Result<Vec<f32>> {
    let mut output = engine.process(samples)?;
    let flush = vec![0.0_f32; flush_padding_len(samples.len())];
    output.extend(engine.process(&flush)?);
    let aligned = realign_denoised(&output, samples.len())?.to_vec();
    if aligned.len() != samples.len() {
        bail!(
            "realigned length {} does not match source length {}",
            aligned.len(),
            samples.len()
        );
    }
    Ok(aligned)
}

fn pad_edge_samples(pad_ms: u32) -> usize {
    pad_ms as usize * (SAMPLE_RATE_HZ / 1_000)
}

fn pad_edges(samples: &[f32], pad_samples: usize) -> Vec<f32> {
    if pad_samples == 0 {
        return samples.to_vec();
    }
    let mut padded = Vec::with_capacity(samples.len() + pad_samples * 2);
    padded.extend(std::iter::repeat_n(0.0_f32, pad_samples));
    padded.extend_from_slice(samples);
    padded.extend(std::iter::repeat_n(0.0_f32, pad_samples));
    padded
}

#[derive(Debug)]
struct Pcm16Conversion {
    samples: Vec<i16>,
    clipped: u64,
    non_finite: u64,
}

/// Converts float audio back to the canonical PCM16 contract.
///
/// The scale is the exact inverse of `decode_canonical_pcm16_wav` (`i16 / 32768.0`),
/// so an untouched decode/encode round trip is bit-identical.
fn to_pcm16(samples: &[f32]) -> Pcm16Conversion {
    let mut converted = Vec::with_capacity(samples.len());
    let mut clipped = 0_u64;
    let mut non_finite = 0_u64;
    for &value in samples {
        if !value.is_finite() {
            non_finite += 1;
            converted.push(0);
            continue;
        }
        if value.abs() >= 1.0 {
            clipped += 1;
        }
        converted.push((value * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
    }
    Pcm16Conversion {
        samples: converted,
        clipped,
        non_finite,
    }
}

/// Serializes mono 16 kHz PCM16 into a canonical 44-byte-header RIFF/WAVE file.
fn pcm16_wav(samples: &[i16]) -> Vec<u8> {
    let data_len = u32::try_from(samples.len() * 2).expect("PCM16 payload must fit a RIFF chunk");
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&16_000_u32.to_le_bytes());
    bytes.extend_from_slice(&32_000_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn build_provenance(
    source_split_id: &str,
    source_manifest_sha256: &str,
    apply_nc: bool,
    nc_model_sha256: Option<&str>,
    pad_ms: u32,
) -> Value {
    json!({
        "source_split_id": source_split_id,
        "source_manifest_sha256": source_manifest_sha256,
        "tool": "prepare_nc_derived_split",
        "nc": {
            "applied": apply_nc,
            "model_dir_file": NC_MODEL_FILE,
            "model_sha256": nc_model_sha256,
            "alignment": NC_ALIGNMENT_NOTE,
        },
        "pad_ms": pad_ms,
    })
}

/// Builds the derived manifest document.
///
/// Identity blocks are copied verbatim from the source manifest so the derived
/// split stays comparable, and the extra `provenance` object is tolerated by the
/// runner's parser (unknown fields are ignored).
fn build_derived_manifest(
    source: &RunnerManifestV1,
    new_split_id: &str,
    samples: Vec<RunnerSampleV1>,
    provenance: Value,
) -> Result<Value> {
    let derived = RunnerManifestV1 {
        schema_version: source.schema_version,
        split_id: new_split_id.to_owned(),
        dataset: source.dataset.clone(),
        normalization: source.normalization.clone(),
        audio_format: source.audio_format.clone(),
        samples,
    };
    let mut value =
        serde_json::to_value(&derived).context("failed to serialize the derived manifest")?;
    value
        .as_object_mut()
        .context("derived manifest is not a JSON object")?
        .insert("provenance".to_owned(), provenance);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use parapper_diagnostics::asr_eval::{
        AudioFormatContract, DatasetIdentity, DerivedAudio, NormalizationIdentity, ReferenceText,
        RunnerManifestV1, RunnerSampleV1, decode_canonical_pcm16_wav,
    };

    use super::{
        HOP_SAMPLES, build_derived_manifest, build_provenance, flush_padding_len, pad_edge_samples,
        pad_edges, pcm16_wav, realign_denoised, to_pcm16,
    };

    fn source_manifest() -> RunnerManifestV1 {
        RunnerManifestV1 {
            schema_version: 1,
            split_id: "cv26-ja-dev-smoke-8-v1".to_owned(),
            dataset: DatasetIdentity {
                id: "common_voice_ja".to_owned(),
                release: "26.0".to_owned(),
                source_split: "dev".to_owned(),
                language: "ja".to_owned(),
            },
            normalization: NormalizationIdentity {
                id: "identity_smoke".to_owned(),
                version: "1".to_owned(),
            },
            audio_format: AudioFormatContract {
                encoding: "pcm_s16le".to_owned(),
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![RunnerSampleV1 {
                utterance_id: "common_voice_ja_41787999".to_owned(),
                audio: DerivedAudio {
                    relative_path: "wav/common_voice_ja_41787999.wav".to_owned(),
                    sha256: "a".repeat(64),
                    duration_samples: 79_488,
                },
                reference: ReferenceText {
                    raw: "今日は晴れです。".to_owned(),
                    normalized: "今日は晴れです".to_owned(),
                },
            }],
        }
    }

    #[test]
    fn flush_padding_rounds_the_source_up_to_a_hop_and_adds_the_lag_hop() {
        for source_len in [1_usize, 255, 256, 257, 512, 79_488, 79_489] {
            let padding = flush_padding_len(source_len);
            let fed = source_len + padding;

            assert_eq!(fed % HOP_SAMPLES, 0, "fed length must be a whole hop count");
            assert!(
                fed >= source_len + HOP_SAMPLES,
                "engine must consume the lag hop for {source_len}"
            );
            // The engine only emits whole hops, so emitted == fed here.
            assert!(
                fed >= HOP_SAMPLES + source_len,
                "emitted stream must cover the realignment window for {source_len}"
            );
        }

        assert_eq!(flush_padding_len(256), 256);
        assert_eq!(flush_padding_len(257), 511);
        assert_eq!(flush_padding_len(1), 511);
    }

    #[test]
    fn realignment_drops_exactly_one_hop_and_truncates_to_the_source_length() {
        let source_len = 700_usize;
        let emitted = source_len + flush_padding_len(source_len);
        #[allow(clippy::cast_precision_loss)]
        let stream = (0..emitted).map(|index| index as f32).collect::<Vec<_>>();

        let aligned = realign_denoised(&stream, source_len).unwrap();

        assert_eq!(aligned.len(), source_len);
        assert!((aligned[0] - 256.0).abs() < f32::EPSILON);
        assert!((aligned[source_len - 1] - 955.0).abs() < f32::EPSILON);
    }

    #[test]
    fn realignment_refuses_a_stream_that_was_never_flushed() {
        let stream = vec![0.0_f32; 512];

        let error = realign_denoised(&stream, 400).unwrap_err().to_string();

        assert!(error.contains("too short"), "unexpected error: {error}");
    }

    #[test]
    fn pcm16_conversion_clamps_rounds_and_round_trips_the_canonical_decoder() {
        let converted = to_pcm16(&[0.0, 1.0, -1.0, 2.5, -3.0, 1.5 / 32768.0, -1.5 / 32768.0]);

        assert_eq!(
            converted.samples,
            vec![0, 32_767, -32_768, 32_767, -32_768, 2, -2]
        );
        assert_eq!(converted.clipped, 4);
        assert_eq!(converted.non_finite, 0);

        let extremes = [i16::MIN, -1, 0, 1, i16::MAX];
        let decoded = decode_canonical_pcm16_wav(&pcm16_wav(&extremes)).unwrap();
        assert_eq!(to_pcm16(&decoded.samples).samples, extremes);
        assert_eq!(
            to_pcm16(&decoded.samples).clipped,
            1,
            "i16::MIN maps to -1.0"
        );
    }

    #[test]
    fn pcm16_conversion_reports_non_finite_samples_instead_of_writing_them() {
        let converted = to_pcm16(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25]);

        assert_eq!(converted.non_finite, 3);
        assert_eq!(converted.samples, vec![0, 0, 0, 8_192]);
    }

    #[test]
    fn edge_padding_adds_the_requested_silence_to_both_edges() {
        assert_eq!(pad_edge_samples(0), 0);
        assert_eq!(pad_edge_samples(1_000), 16_000);
        assert_eq!(pad_edge_samples(250), 4_000);

        let padded = pad_edges(&[0.5, -0.5], pad_edge_samples(1));

        assert_eq!(padded.len(), 2 + 16 * 2);
        assert!(
            padded[..16]
                .iter()
                .all(|sample| sample.abs() < f32::EPSILON)
        );
        assert!((padded[16] - 0.5).abs() < f32::EPSILON);
        assert!((padded[17] + 0.5).abs() < f32::EPSILON);
        assert!(
            padded[18..]
                .iter()
                .all(|sample| sample.abs() < f32::EPSILON)
        );
        assert_eq!(pad_edges(&[0.5, -0.5], 0), vec![0.5, -0.5]);
    }

    #[test]
    fn derived_manifest_carries_provenance_and_still_passes_the_runner_preflight() {
        let source = source_manifest();
        let derived_samples = vec![RunnerSampleV1 {
            utterance_id: source.samples[0].utterance_id.clone(),
            audio: DerivedAudio {
                relative_path: "wav/common_voice_ja_41787999.wav".to_owned(),
                sha256: "b".repeat(64),
                duration_samples: 79_488 + 32_000,
            },
            reference: source.samples[0].reference.clone(),
        }];
        let provenance = build_provenance(
            &source.split_id,
            &"c".repeat(64),
            true,
            Some(&"d".repeat(64)),
            1_000,
        );

        let value = build_derived_manifest(
            &source,
            "cv26-ja-dev-smoke-8-nc-ulunas-pad1000-v1",
            derived_samples,
            provenance,
        )
        .unwrap();
        let bytes = serde_json::to_vec_pretty(&value).unwrap();
        let reparsed = RunnerManifestV1::parse(&bytes).unwrap();

        assert_eq!(
            reparsed.split_id,
            "cv26-ja-dev-smoke-8-nc-ulunas-pad1000-v1"
        );
        assert_eq!(reparsed.dataset, source.dataset);
        assert_eq!(reparsed.normalization, source.normalization);
        assert_eq!(reparsed.audio_format, source.audio_format);
        assert_eq!(reparsed.samples[0].reference, source.samples[0].reference);
        assert_eq!(reparsed.samples[0].audio.duration_samples, 111_488);
        assert_eq!(value["provenance"]["pad_ms"], 1_000);
        assert_eq!(value["provenance"]["nc"]["applied"], true);
        assert_eq!(
            value["provenance"]["nc"]["model_dir_file"],
            "ulunas_stream_simple.onnx"
        );
        assert_eq!(value["provenance"]["source_split_id"], source.split_id);
    }

    #[test]
    fn provenance_records_a_missing_model_hash_when_noise_cancellation_is_skipped() {
        let provenance = build_provenance("split-a", &"c".repeat(64), false, None, 1_000);

        assert_eq!(provenance["nc"]["applied"], false);
        assert!(provenance["nc"]["model_sha256"].is_null());
        assert_eq!(provenance["tool"], "prepare_nc_derived_split");
    }
}
