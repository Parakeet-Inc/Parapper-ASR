use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{RunnerManifestV1, decode_canonical_pcm16_wav};
use parapper_models::asr::{
    AsrModel, AsrPrecision,
    backend::reazon_ort::{ReazonDecodingStrategy, ReazonSpeechOrtAsrEngine},
    decoder::rnnt::StatelessRnntSearchProfile,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const BEAM_SIZE: usize = 4;
const USAGE: &str = "Usage:
  profile_reazon_search \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --model-dir <model-directory> \\
    --precision <fp32|int8> \\
    --threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --profile-output <profile.json>";

fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    run(&args)
}

#[allow(
    clippy::too_many_lines,
    reason = "the diagnostic command keeps identity validation, inference, and its auditable report together"
)]
fn run(args: &Args) -> Result<()> {
    if !args.ort_dylib.is_file() {
        bail!(
            "ONNX Runtime library does not exist: {}",
            args.ort_dylib.display()
        );
    }
    if args.profile_output.exists() {
        bail!(
            "profile output already exists: {}",
            args.profile_output.display()
        );
    }
    let manifest_bytes = fs::read(&args.manifest)
        .with_context(|| format!("failed to read {}", args.manifest.display()))?;
    let manifest = RunnerManifestV1::parse(&manifest_bytes)
        .with_context(|| format!("failed to preflight {}", args.manifest.display()))?;
    if manifest.split_id != args.split_id {
        bail!(
            "split ID mismatch: CLI requested {:?}, manifest contains {:?}",
            args.split_id,
            manifest.split_id
        );
    }

    // SAFETY: the process is single-threaded and has not touched ORT yet.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };
    let mut engine = ReazonSpeechOrtAsrEngine::new_with_decoding(
        &args.model_dir,
        AsrModel::ReazonSpeechK2V2,
        args.precision,
        args.threads,
        ReazonDecodingStrategy::ModifiedBeam {
            beam_size: BEAM_SIZE,
        },
    )
    .with_context(|| {
        format!(
            "failed to construct {:?} ReazonSpeech from {}",
            args.precision,
            args.model_dir.display(),
        )
    })?;

    let manifest_root = args.manifest.parent().unwrap_or_else(|| Path::new("."));
    let mut profile = StatelessRnntSearchProfile::default();
    let mut encode_wall = Duration::ZERO;
    let mut decode_wall = Duration::ZERO;
    let mut audio_samples = 0_u64;
    let mut hypothesis_hasher = Sha256::new();
    for sample in &manifest.samples {
        let audio_path = manifest_root.join(&sample.audio.relative_path);
        let bytes = fs::read(&audio_path)
            .with_context(|| format!("failed to read {}", audio_path.display()))?;
        let actual_sha = format!("{:x}", Sha256::digest(&bytes));
        if actual_sha != sample.audio.sha256 {
            bail!("audio SHA-256 mismatch for {}", sample.utterance_id);
        }
        let wav = decode_canonical_pcm16_wav(&bytes)
            .with_context(|| format!("invalid canonical WAV for {}", sample.utterance_id))?;
        if wav.samples.len() as u64 != sample.audio.duration_samples {
            bail!("audio duration mismatch for {}", sample.utterance_id);
        }
        audio_samples += sample.audio.duration_samples;

        let encode_started = Instant::now();
        let encoded = engine
            .encode(&wav.samples)
            .with_context(|| format!("encoder failed for {}", sample.utterance_id))?;
        encode_wall += encode_started.elapsed();
        let started = Instant::now();
        let transcript = engine
            .decode_encoded_with_search_profile(
                &encoded,
                ReazonDecodingStrategy::ModifiedBeam {
                    beam_size: BEAM_SIZE,
                },
                &mut profile,
            )
            .with_context(|| format!("decoder failed for {}", sample.utterance_id))?;
        decode_wall += started.elapsed();
        hypothesis_hasher.update(sample.utterance_id.as_bytes());
        hypothesis_hasher.update([0]);
        hypothesis_hasher.update(transcript.text.as_bytes());
        hypothesis_hasher.update(b"\n");
    }

    let stage_ms = StageMilliseconds::from(&profile);
    let timed_stage_ms = stage_ms.total();
    let search_management_ms = stage_ms.search_management();
    let encode_wall_ms = milliseconds(encode_wall);
    let decode_wall_ms = milliseconds(decode_wall);
    let total_inference_wall_ms = encode_wall_ms + decode_wall_ms;
    let audio_duration = Duration::from_secs(audio_samples / 16_000)
        + Duration::from_nanos((audio_samples % 16_000) * 1_000_000_000 / 16_000);
    let report = ProfileReport {
        schema_version: 2,
        split_id: &manifest.split_id,
        samples: manifest.samples.len(),
        audio_samples,
        model: AsrModel::ReazonSpeechK2V2,
        precision: args.precision,
        provider: "cpu",
        threads: args.threads,
        beam_size: BEAM_SIZE,
        search: "state_dominance_cached_unique",
        top_token_algorithm: "ort_top_k",
        score_normalization: "ort_log_softmax",
        network_score_output: "ort_top_k_dynamic_gather",
        predictor_cache_scope: "one_utterance",
        encode_wall_ms,
        decode_wall_ms,
        total_inference_wall_ms,
        rtf: total_inference_wall_ms / 1_000.0 / audio_duration.as_secs_f64(),
        timed_stage_ms,
        unattributed_ms: decode_wall_ms - timed_stage_ms,
        search_management_ms,
        network_ms: stage_ms.network_logits,
        stage_ms,
        counts: Counts::from(&profile),
        hypothesis_sha256: format!("{:x}", hypothesis_hasher.finalize()),
    };
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.profile_output)
        .with_context(|| format!("failed to create {}", args.profile_output.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(output), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[derive(Debug, Serialize)]
struct ProfileReport<'a> {
    schema_version: u32,
    split_id: &'a str,
    samples: usize,
    audio_samples: u64,
    model: AsrModel,
    precision: AsrPrecision,
    provider: &'static str,
    threads: i32,
    beam_size: usize,
    search: &'static str,
    top_token_algorithm: &'static str,
    score_normalization: &'static str,
    network_score_output: &'static str,
    predictor_cache_scope: &'static str,
    encode_wall_ms: f64,
    decode_wall_ms: f64,
    total_inference_wall_ms: f64,
    rtf: f64,
    timed_stage_ms: f64,
    unattributed_ms: f64,
    search_management_ms: f64,
    network_ms: f64,
    stage_ms: StageMilliseconds,
    counts: Counts,
    hypothesis_sha256: String,
}

#[derive(Debug, Serialize)]
struct StageMilliseconds {
    context_layout: f64,
    network_logits: f64,
    log_softmax: f64,
    top_token_selection: f64,
    candidate_generation_and_exact_prefix_merge: f64,
    state_dominance: f64,
    score_pruning: f64,
    survivor_materialization: f64,
    final_ranking_and_reconstruction: f64,
}

impl StageMilliseconds {
    fn total(&self) -> f64 {
        self.network_logits + self.search_management()
    }

    fn search_management(&self) -> f64 {
        self.context_layout
            + self.log_softmax
            + self.top_token_selection
            + self.candidate_generation_and_exact_prefix_merge
            + self.state_dominance
            + self.score_pruning
            + self.survivor_materialization
            + self.final_ranking_and_reconstruction
    }
}

impl From<&StatelessRnntSearchProfile> for StageMilliseconds {
    fn from(profile: &StatelessRnntSearchProfile) -> Self {
        Self {
            context_layout: milliseconds(profile.context_layout),
            network_logits: milliseconds(profile.network_logits),
            log_softmax: milliseconds(profile.log_softmax),
            top_token_selection: milliseconds(profile.top_token_selection),
            candidate_generation_and_exact_prefix_merge: milliseconds(profile.candidate_generation),
            state_dominance: milliseconds(profile.state_dominance),
            score_pruning: milliseconds(profile.score_pruning),
            survivor_materialization: milliseconds(profile.materialization),
            final_ranking_and_reconstruction: milliseconds(
                profile.final_ranking_and_reconstruction,
            ),
        }
    }
}

#[derive(Debug, Serialize)]
struct Counts {
    frames: usize,
    active_hypotheses: usize,
    network_context_rows: usize,
    logit_values: usize,
    network_output_values: usize,
    network_output_bytes: usize,
    scalar_exp_terms_evaluated: usize,
    scalar_exp_terms_skipped: usize,
    candidates_generated: usize,
    states_after_dominance: usize,
}

impl From<&StatelessRnntSearchProfile> for Counts {
    fn from(profile: &StatelessRnntSearchProfile) -> Self {
        Self {
            frames: profile.frames,
            active_hypotheses: profile.active_hypotheses,
            network_context_rows: profile.network_context_rows,
            logit_values: profile.logit_values,
            network_output_values: profile.network_output_values,
            network_output_bytes: profile.network_output_bytes,
            scalar_exp_terms_evaluated: profile.scalar_exp_terms_evaluated,
            scalar_exp_terms_skipped: profile.scalar_exp_terms_skipped,
            candidates_generated: profile.candidates_generated,
            states_after_dominance: profile.states_after_dominance,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    model_dir: PathBuf,
    precision: AsrPrecision,
    threads: i32,
    ort_dylib: PathBuf,
    profile_output: PathBuf,
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
                "--model-dir",
                "--precision",
                "--threads",
                "--ort-dylib",
                "--profile-output",
            ]
            .contains(&argument.as_str())
            {
                bail!("unknown argument {argument}\n{USAGE}");
            }
        }
        let value = |name: &str| -> Result<&str> {
            let index = arguments
                .iter()
                .position(|argument| argument == name)
                .with_context(|| format!("missing required argument {name}\n{USAGE}"))?;
            arguments
                .get(index + 1)
                .filter(|candidate| !candidate.starts_with("--"))
                .map(String::as_str)
                .with_context(|| format!("missing value for {name}\n{USAGE}"))
        };
        let threads = value("--threads")?
            .parse::<i32>()
            .context("--threads must be a positive integer")?;
        if threads <= 0 {
            bail!("--threads must be a positive integer");
        }
        let precision = parse_precision(value("--precision")?)?;
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            model_dir: PathBuf::from(value("--model-dir")?),
            precision,
            threads,
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            profile_output: PathBuf::from(value("--profile-output")?),
        }))
    }
}

fn parse_precision(value: &str) -> Result<AsrPrecision> {
    match value {
        "fp32" => Ok(AsrPrecision::Float32),
        "int8" => Ok(AsrPrecision::Int8),
        value => bail!("unsupported --precision {value:?}; expected fp32 or int8"),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use parapper_models::asr::AsrPrecision;

    use super::Args;

    #[test]
    fn cli_requires_every_input_that_identifies_the_width_four_measurement() {
        let args = Args::parse(
            [
                "--manifest",
                "data/manifest.json",
                "--split-id",
                "fixed-100",
                "--model-dir",
                "models/reazon",
                "--precision",
                "fp32",
                "--threads",
                "4",
                "--ort-dylib",
                "runtime/onnxruntime.dll",
                "--profile-output",
                "results/profile.json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            args,
            Args {
                manifest: PathBuf::from("data/manifest.json"),
                split_id: "fixed-100".to_owned(),
                model_dir: PathBuf::from("models/reazon"),
                precision: AsrPrecision::Float32,
                threads: 4,
                ort_dylib: PathBuf::from("runtime/onnxruntime.dll"),
                profile_output: PathBuf::from("results/profile.json"),
            }
        );
    }

    #[test]
    fn cli_rejects_removed_ablation_switches() {
        let common = vec![
            "--manifest",
            "data/manifest.json",
            "--split-id",
            "fixed-100",
            "--model-dir",
            "models/reazon",
            "--precision",
            "fp32",
            "--threads",
            "4",
            "--ort-dylib",
            "runtime/onnxruntime.dll",
            "--profile-output",
            "results/profile.json",
        ];
        for (flag, value) in [
            ("--top-token-algorithm", Some("cutoff_binary_search")),
            ("--score-normalization", Some("ort_log_softmax")),
            (
                "--log-softmax-joiner",
                Some("models/joiner-logsoftmax.onnx"),
            ),
            ("--ort-top-k-gather", None),
        ] {
            let mut arguments = common.clone();
            arguments.push(flag);
            arguments.extend(value);
            let error = Args::parse(arguments.into_iter().map(str::to_owned)).unwrap_err();
            assert!(
                error
                    .to_string()
                    .starts_with(&format!("unknown argument {flag}")),
                "flag={flag} error={error}"
            );
        }
    }
}
