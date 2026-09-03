use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{
    ReazonNBestCondition, ReazonNBestSearchNormalization, RunnerManifestV1,
    run_reazon_monotonic_alignment,
};
use parapper_models::asr::{
    AsrModel, AsrPrecision,
    backend::reazon_ort::{ReazonDecodingStrategy, ReazonSpeechOrtAsrEngine},
};
use serde::Serialize;

const BEAM_SIZE: usize = 8;
const OUTPUT_FILE: &str = "reazon-fp32-full-prefix-beam8-raw-monotonic-alignment.jsonl";
const USAGE: &str = "Usage:
  run_reazon_monotonic_alignment \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --model-dir <model-directory> \\
    --threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --output-dir <directory> \\
    [--include-utterance-ids <comma-separated-ids>] \\
    [--shard-index <zero-based-index> --shard-count <positive-integer>]";

fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    run(&args)
}

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
    select_manifest_samples(&mut manifest, args)?;
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;
    let output_path = args
        .output_dir
        .join(shard_output_name(args.shard_index, args.shard_count));
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut writer = BufWriter::new(output);

    // SAFETY: no ORT API is touched before this process-global loader path is
    // fixed, and initialization remains single-threaded.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };
    let mut engine = ReazonSpeechOrtAsrEngine::new_with_decoding(
        &args.model_dir,
        AsrModel::ReazonSpeechK2V2,
        AsrPrecision::Float32,
        args.threads,
        ReazonDecodingStrategy::ModifiedBeam {
            beam_size: BEAM_SIZE,
        },
    )
    .with_context(|| {
        format!(
            "failed to construct FP32 ReazonSpeech from {}",
            args.model_dir.display()
        )
    })?;
    let condition = ReazonNBestCondition {
        beam_size: BEAM_SIZE,
        search_normalization: ReazonNBestSearchNormalization::Raw,
    };
    let summary = run_reazon_monotonic_alignment(
        &args.manifest,
        &manifest,
        &mut engine,
        condition,
        &mut writer,
    )?;
    writer.flush().context("failed to flush alignment output")?;
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
            included_utterances: args.include_utterance_ids.as_ref().map(Vec::len),
            beam_size: BEAM_SIZE,
            search_normalization: ReazonNBestSearchNormalization::Raw,
            alignment_graph: "one_symbol_per_frame_forward_backward",
            output: &output_path,
            total: summary.total,
            completed: summary.completed,
            failed: summary.failed,
        })?
    );
    Ok(())
}

fn select_manifest_samples(manifest: &mut RunnerManifestV1, args: &Args) -> Result<()> {
    if let Some(included) = &args.include_utterance_ids {
        let missing = included
            .iter()
            .filter(|utterance_id| {
                !manifest
                    .samples
                    .iter()
                    .any(|sample| sample.utterance_id == **utterance_id)
            })
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!("included utterance IDs are absent from the manifest: {missing:?}");
        }
        manifest
            .samples
            .retain(|sample| included.contains(&sample.utterance_id));
    }
    let samples = std::mem::take(&mut manifest.samples);
    manifest.samples = samples
        .into_iter()
        .enumerate()
        .filter_map(|(index, sample)| {
            (index % args.shard_count == args.shard_index).then_some(sample)
        })
        .collect();
    Ok(())
}

fn shard_output_name(shard_index: usize, shard_count: usize) -> String {
    if shard_count == 1 {
        OUTPUT_FILE.to_owned()
    } else {
        OUTPUT_FILE.replace(
            ".jsonl",
            &format!("-part-{shard_index:02}-of-{shard_count:02}.jsonl"),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    model_dir: PathBuf,
    threads: i32,
    ort_dylib: PathBuf,
    output_dir: PathBuf,
    include_utterance_ids: Option<Vec<String>>,
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
                "--include-utterance-ids",
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
        let threads = value("--threads")?
            .parse::<i32>()
            .context("--threads must be a positive integer")?;
        if threads <= 0 {
            bail!("--threads must be a positive integer");
        }
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
        let include_utterance_ids = optional_value("--include-utterance-ids")?
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .filter(|values| !values.is_empty());
        if include_utterance_ids.as_ref().is_some_and(|values| {
            values.iter().any(String::is_empty)
                || values
                    .iter()
                    .enumerate()
                    .any(|(index, value)| values[..index].contains(value))
        }) {
            bail!("--include-utterance-ids must contain unique nonempty IDs");
        }
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            model_dir: PathBuf::from(value("--model-dir")?),
            threads,
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            output_dir: PathBuf::from(value("--output-dir")?),
            include_utterance_ids,
            shard_index,
            shard_count,
        }))
    }
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
    included_utterances: Option<usize>,
    beam_size: usize,
    search_normalization: ReazonNBestSearchNormalization,
    alignment_graph: &'static str,
    output: &'a Path,
    total: usize,
    completed: usize,
    failed: usize,
}

#[cfg(test)]
mod tests {
    use super::{Args, shard_output_name};

    #[test]
    fn cli_pins_width_eight_raw_alignment_and_gives_each_shard_a_distinct_file() {
        let args = Args::parse(
            [
                "--manifest",
                "data/manifest.json",
                "--split-id",
                "jsut-basic5000-test-v1",
                "--model-dir",
                "models/reazon",
                "--threads",
                "2",
                "--ort-dylib",
                "runtime/onnxruntime.dll",
                "--output-dir",
                "results/alignment",
                "--include-utterance-ids",
                "BASIC5000_0001,BASIC5000_0002",
                "--shard-index",
                "3",
                "--shard-count",
                "4",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            (args.threads, args.shard_index, args.shard_count),
            (2, 3, 4)
        );
        assert_eq!(
            args.include_utterance_ids,
            Some(vec![
                "BASIC5000_0001".to_owned(),
                "BASIC5000_0002".to_owned()
            ])
        );
        assert_eq!(
            shard_output_name(args.shard_index, args.shard_count),
            "reazon-fp32-full-prefix-beam8-raw-monotonic-alignment-part-03-of-04.jsonl"
        );
    }
}
