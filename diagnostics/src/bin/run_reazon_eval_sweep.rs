use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{RunnerManifestV1, run_reazon_accuracy_sweep};
use parapper_models::asr::{
    AsrModel, AsrPrecision,
    backend::reazon_ort::{ReazonDecodingStrategy, ReazonSpeechOrtAsrEngine},
};
use serde::Serialize;

const USAGE: &str = "Usage:
  run_reazon_eval_sweep \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --model-dir <model-directory> \\
    --threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --output-dir <directory>";

const CONDITIONS: [Condition; 4] = [
    Condition {
        label: "greedy",
        strategy: ReazonDecodingStrategy::Greedy,
        file_name: "reazon-fp32-greedy-accuracy-cached.jsonl",
    },
    Condition {
        label: "state2",
        strategy: ReazonDecodingStrategy::ModifiedBeam { beam_size: 2 },
        file_name: "reazon-fp32-state2-accuracy-cached.jsonl",
    },
    Condition {
        label: "state4",
        strategy: ReazonDecodingStrategy::ModifiedBeam { beam_size: 4 },
        file_name: "reazon-fp32-state4-accuracy-cached.jsonl",
    },
    Condition {
        label: "state8",
        strategy: ReazonDecodingStrategy::ModifiedBeam { beam_size: 8 },
        file_name: "reazon-fp32-state8-accuracy-cached.jsonl",
    },
];

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
    let manifest = RunnerManifestV1::parse(&manifest_bytes)
        .with_context(|| format!("failed to preflight {}", args.manifest.display()))?;
    if manifest.split_id != args.split_id {
        bail!(
            "split ID mismatch: CLI requested {:?}, manifest contains {:?}",
            args.split_id,
            manifest.split_id
        );
    }
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

    // SAFETY: this CLI remains single-threaded until after this assignment and
    // no ONNX Runtime API is touched first. The dynamic loader reads this
    // process-global value on its first API use.
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
    let strategies = CONDITIONS.map(|condition| condition.strategy);
    let mut output_refs = writers
        .iter_mut()
        .map(|writer| writer as &mut dyn Write)
        .collect::<Vec<_>>();
    let summaries = run_reazon_accuracy_sweep(
        &args.manifest,
        &manifest,
        &mut engine,
        &strategies,
        &mut output_refs,
    )?;
    drop(output_refs);
    for writer in &mut writers {
        writer.flush().context("failed to flush sweep output")?;
    }

    let outputs = CONDITIONS
        .iter()
        .zip(paths.iter())
        .zip(summaries)
        .map(|((condition, path), summary)| OutputSummary {
            condition: condition.label,
            beam_size: condition.beam_size(),
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
            accuracy_only: true,
            timing_comparable: false,
            encoder_runs_per_utterance: 1,
            predictor_cache_scope: "engine_lifetime_all_utterances_and_conditions",
            outputs,
        })?
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Condition {
    label: &'static str,
    strategy: ReazonDecodingStrategy,
    file_name: &'static str,
}

impl Condition {
    const fn beam_size(self) -> Option<usize> {
        match self.strategy {
            ReazonDecodingStrategy::Greedy => None,
            ReazonDecodingStrategy::ModifiedBeam { beam_size }
            | ReazonDecodingStrategy::OneSpliceRerank { beam_size, .. } => Some(beam_size),
            ReazonDecodingStrategy::ModifiedBeamAblation { config } => Some(config.beam_size),
        }
    }
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
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            model_dir: PathBuf::from(value("--model-dir")?),
            threads,
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            output_dir: PathBuf::from(value("--output-dir")?),
        }))
    }
}

#[derive(Serialize)]
struct OutputSummary<'a> {
    condition: &'static str,
    beam_size: Option<usize>,
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
    accuracy_only: bool,
    timing_comparable: bool,
    encoder_runs_per_utterance: usize,
    predictor_cache_scope: &'static str,
    outputs: Vec<OutputSummary<'a>>,
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{Args, output_paths};

    #[test]
    fn cli_requires_every_cache_sweep_identity_and_uses_only_fp32_conditions() {
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
                "results/jsut",
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
                split_id: "jsut-basic5000-test-v1".to_owned(),
                model_dir: PathBuf::from("models/reazon"),
                threads: 4,
                ort_dylib: PathBuf::from("runtime/onnxruntime.dll"),
                output_dir: PathBuf::from("results/jsut"),
            }
        );
        assert_eq!(
            output_paths(Path::new("results/jsut")),
            [
                PathBuf::from("results/jsut/reazon-fp32-greedy-accuracy-cached.jsonl"),
                PathBuf::from("results/jsut/reazon-fp32-state2-accuracy-cached.jsonl"),
                PathBuf::from("results/jsut/reazon-fp32-state4-accuracy-cached.jsonl"),
                PathBuf::from("results/jsut/reazon-fp32-state8-accuracy-cached.jsonl"),
            ]
        );
    }

    #[test]
    fn cli_rejects_nonpositive_threads_and_unknown_options() {
        let base = [
            "--manifest",
            "manifest.json",
            "--split-id",
            "jsut",
            "--model-dir",
            "model",
            "--threads",
            "0",
            "--ort-dylib",
            "onnxruntime.dll",
            "--output-dir",
            "out",
        ];
        let error = Args::parse(base.into_iter().map(str::to_owned)).unwrap_err();
        assert!(error.to_string().contains("positive integer"));

        let error = Args::parse(
            base.into_iter()
                .chain(["--precision", "int8"])
                .map(str::to_owned),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown argument --precision"));
    }
}
