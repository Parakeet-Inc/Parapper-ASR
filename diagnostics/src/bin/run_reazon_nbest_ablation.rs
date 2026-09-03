use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{
    ReazonNBestCondition, ReazonNBestSearchNormalization, RunnerManifestV1, run_reazon_nbest_sweep,
};
use parapper_models::asr::{
    AsrModel, AsrPrecision,
    backend::reazon_ort::{ReazonDecodingStrategy, ReazonSpeechOrtAsrEngine},
};
use serde::Serialize;

const USAGE: &str = "Usage:
  run_reazon_nbest_ablation \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --model-dir <model-directory> \\
    --threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --output-dir <directory> \\
    [--shard-index <zero-based-index> --shard-count <positive-integer>]";

const CONDITIONS: [NamedCondition; 4] = [
    NamedCondition {
        label: "beam4_raw_during",
        file_name: "reazon-fp32-full-prefix-beam4-raw-during.jsonl",
        condition: ReazonNBestCondition {
            beam_size: 4,
            search_normalization: ReazonNBestSearchNormalization::Raw,
        },
    },
    NamedCondition {
        label: "beam4_per_token_during",
        file_name: "reazon-fp32-full-prefix-beam4-per-token-during.jsonl",
        condition: ReazonNBestCondition {
            beam_size: 4,
            search_normalization: ReazonNBestSearchNormalization::PerToken,
        },
    },
    NamedCondition {
        label: "beam8_raw_during",
        file_name: "reazon-fp32-full-prefix-beam8-raw-during.jsonl",
        condition: ReazonNBestCondition {
            beam_size: 8,
            search_normalization: ReazonNBestSearchNormalization::Raw,
        },
    },
    NamedCondition {
        label: "beam8_per_token_during",
        file_name: "reazon-fp32-full-prefix-beam8-per-token-during.jsonl",
        condition: ReazonNBestCondition {
            beam_size: 8,
            search_normalization: ReazonNBestSearchNormalization::PerToken,
        },
    },
];

fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    run(&args)
}

#[allow(
    clippy::too_many_lines,
    reason = "the standalone diagnostic keeps loader, model, output, and audit metadata together"
)]
fn run(args: &Args) -> Result<()> {
    if !args.ort_dylib.is_file() {
        bail!(
            "ONNX Runtime library does not exist: {}",
            args.ort_dylib.display()
        );
    }
    let manifest_bytes = fs::read(&args.manifest)
        .with_context(|| format!("failed to read {}", args.manifest.display()))?;
    let mut manifest = RunnerManifestV1::parse(&manifest_bytes)
        .with_context(|| format!("failed to preflight {}", args.manifest.display()))?;
    if manifest.split_id != args.split_id {
        bail!(
            "split ID mismatch: CLI requested {:?}, manifest contains {:?}",
            args.split_id,
            manifest.split_id
        );
    }
    manifest.samples = manifest
        .samples
        .into_iter()
        .enumerate()
        .filter_map(|(index, sample)| {
            (index % args.shard_count == args.shard_index).then_some(sample)
        })
        .collect();
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;
    let paths = output_paths(&args.output_dir);
    for path in &paths {
        if path.exists() {
            bail!(
                "output already exists (resume/overwrite is not implemented): {}",
                path.display()
            );
        }
    }

    // SAFETY: no ORT API is touched before this process-global loader path is
    // fixed, and the diagnostic stays single-threaded during initialization.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };
    let mut engine = ReazonSpeechOrtAsrEngine::new_with_decoding(
        &args.model_dir,
        AsrModel::ReazonSpeechK2V2,
        AsrPrecision::Float32,
        args.threads,
        ReazonDecodingStrategy::ModifiedBeam { beam_size: 8 },
    )
    .with_context(|| {
        format!(
            "failed to construct FP32 ReazonSpeech from {}",
            args.model_dir.display()
        )
    })?;

    let mut writers = paths
        .iter()
        .map(|path| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map(BufWriter::new)
                .with_context(|| format!("failed to create {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let conditions = CONDITIONS.map(|condition| condition.condition);
    let mut outputs = writers
        .iter_mut()
        .map(|writer| writer as &mut dyn Write)
        .collect::<Vec<_>>();
    let summaries = run_reazon_nbest_sweep(
        &args.manifest,
        &manifest,
        &mut engine,
        &conditions,
        &mut outputs,
    )?;
    drop(outputs);
    for writer in &mut writers {
        writer.flush().context("failed to flush N-best output")?;
    }

    let outputs = CONDITIONS
        .iter()
        .zip(paths.iter())
        .zip(summaries)
        .map(|((condition, path), summary)| OutputSummary {
            condition: condition.label,
            beam_size: condition.condition.beam_size,
            search_normalization: condition.condition.search_normalization,
            path,
            total: summary.total,
            completed: summary.completed,
            failed: summary.failed,
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&CommandSummary {
            schema_version: 1,
            split_id: &manifest.split_id,
            model: AsrModel::ReazonSpeechK2V2,
            precision: AsrPrecision::Float32,
            backend: "direct_ort",
            provider: "cpu",
            threads: args.threads,
            shard_index: args.shard_index,
            shard_count: args.shard_count,
            accuracy_only: true,
            timing_comparable: false,
            encoder_runs_per_utterance: 1,
            full_prefix_history: true,
            outputs,
        })?
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct NamedCondition {
    label: &'static str,
    file_name: &'static str,
    condition: ReazonNBestCondition,
}

fn output_paths(output_dir: &Path) -> [PathBuf; 4] {
    CONDITIONS.map(|condition| output_dir.join(condition.file_name))
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    model_dir: PathBuf,
    threads: i32,
    ort_dylib: PathBuf,
    output_dir: PathBuf,
    shard_index: usize,
    shard_count: usize,
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
                "--threads",
                "--ort-dylib",
                "--output-dir",
                "--shard-index",
                "--shard-count",
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
        let optional_value = |name: &str| -> Result<Option<&str>> {
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
        let shard_index = optional_value("--shard-index")?
            .unwrap_or("0")
            .parse::<usize>()
            .context("--shard-index must be a nonnegative integer")?;
        let shard_count = optional_value("--shard-count")?
            .unwrap_or("1")
            .parse::<usize>()
            .context("--shard-count must be a positive integer")?;
        if shard_count == 0 || shard_index >= shard_count {
            bail!("shard index must be less than a positive shard count");
        }
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            model_dir: PathBuf::from(value("--model-dir")?),
            threads,
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            output_dir: PathBuf::from(value("--output-dir")?),
            shard_index,
            shard_count,
        }))
    }
}

#[derive(Serialize)]
struct OutputSummary<'a> {
    condition: &'static str,
    beam_size: usize,
    search_normalization: ReazonNBestSearchNormalization,
    path: &'a Path,
    total: usize,
    completed: usize,
    failed: usize,
}

#[derive(Serialize)]
struct CommandSummary<'a> {
    schema_version: u32,
    split_id: &'a str,
    model: AsrModel,
    precision: AsrPrecision,
    backend: &'static str,
    provider: &'static str,
    threads: i32,
    shard_index: usize,
    shard_count: usize,
    accuracy_only: bool,
    timing_comparable: bool,
    encoder_runs_per_utterance: usize,
    full_prefix_history: bool,
    outputs: Vec<OutputSummary<'a>>,
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{Args, output_paths};

    #[test]
    fn cli_pins_fp32_width_four_and_eight_full_prefix_outputs() {
        let args = Args::parse(
            [
                "--manifest",
                "data/manifest.json",
                "--split-id",
                "jsut-basic5000-test-v1",
                "--model-dir",
                "models/reazon",
                "--threads",
                "4",
                "--ort-dylib",
                "runtime/onnxruntime.dll",
                "--output-dir",
                "results/nbest",
                "--shard-index",
                "2",
                "--shard-count",
                "4",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();

        assert_eq!(args.threads, 4);
        assert_eq!((args.shard_index, args.shard_count), (2, 4));
        assert_eq!(
            output_paths(Path::new("results/nbest")),
            [
                PathBuf::from("results/nbest/reazon-fp32-full-prefix-beam4-raw-during.jsonl"),
                PathBuf::from("results/nbest/reazon-fp32-full-prefix-beam4-per-token-during.jsonl"),
                PathBuf::from("results/nbest/reazon-fp32-full-prefix-beam8-raw-during.jsonl"),
                PathBuf::from("results/nbest/reazon-fp32-full-prefix-beam8-per-token-during.jsonl"),
            ]
        );
    }
}
