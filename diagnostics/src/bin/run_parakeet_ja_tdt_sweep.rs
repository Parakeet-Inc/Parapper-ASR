//! Runs the Japanese Parakeet hybrid model's CTC and TDT branches over one
//! evaluation manifest, one JSONL output per condition.
//!
//! Conditions:
//! - `prod_ctc_greedy`: the legacy sherpa-layout CTC export
//!   (`model.int8.onnx` + `tokens.txt`) retained as a comparison baseline.
//! - `onnx_asr_ctc_greedy`: the onnx-asr CTC export, same graph contract but
//!   a different export/quantization pipeline.
//! - `fused_tdt_greedy`: the onnx-asr fused TDT export decoded with the
//!   NVIDIA-compatible greedy TDT loop.
//!
//! Comparing `onnx_asr_ctc_greedy` against `fused_tdt_greedy` isolates the
//! decoder branch (both graphs come from one export pipeline); comparing
//! against `prod_ctc_greedy` anchors the sweep to the existing baselines.

use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{
    EvalTranscriber, FusedTdtDagConfig, FusedTdtDurationExpansion, FusedTdtJaOrtEngine,
    HybridParakeetJaOrtEngine, OnnxAsrCtcJaOrtEngine, ParakeetJaDecodingStrategy, RunnerManifestV1,
    run_manifest,
};
use parapper_models::asr::{
    AsrEngine, AsrModel, AsrPrecision, backend::direct_ort::NvidiaCtcOrtAsrEngine,
};
use parapper_stt_engine::{AsrModelRegistry, OfflineTranscriptionService};
use serde::Serialize;

const USAGE: &str = "Usage:
  run_parakeet_ja_tdt_sweep \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --prod-ctc-model-dir <sherpa-layout model directory> \\
    --fused-model-dir <onnx-asr layout model directory> \\
    [--conditions <comma list of prod_ctc_greedy,onnx_asr_ctc_greedy,fused_tdt_greedy,shared_ctc_greedy,shared_tdt_greedy,shared_tdt_dag2,shared_tdt_dag4,shared_tdt_dag2_duration_argmax,shared_tdt_dag4_duration_argmax,shared_tdt_dag8_duration_argmax,shared_tdt_dag16_duration_argmax,shared_tdt_dag2_duration_argmax_static_embedding,shared_tdt_dag4_duration_argmax_static_embedding,shared_tdt_dag8_duration_argmax_static_embedding>] \\
    [--limit <sample-count, smoke runs only>] \\
    --threads <positive-integer> \\
    [--encoder-threads <positive-integer, defaults to --threads>] \\
    [--decoder-threads <positive-integer, defaults to --threads>] \\
    [--precision <float32|int8, describes the fused/shared artifacts; defaults to int8>] \\
    [--static-embedding-dir <static-embedding-japanese snapshot directory>] \\
    --ort-dylib <onnxruntime-library> \\
    --output-dir <directory for <condition>.jsonl>

A condition whose output file already exists is skipped, so an interrupted
sweep can be resumed at condition granularity.";

const DEFAULT_CONDITIONS: [&str; 11] = [
    "prod_ctc_greedy",
    "onnx_asr_ctc_greedy",
    "fused_tdt_greedy",
    "shared_ctc_greedy",
    "shared_tdt_greedy",
    "shared_tdt_dag2",
    "shared_tdt_dag4",
    "shared_tdt_dag2_duration_argmax",
    "shared_tdt_dag4_duration_argmax",
    "shared_tdt_dag8_duration_argmax",
    "shared_tdt_dag16_duration_argmax",
];

const ALL_CONDITIONS: [&str; 14] = [
    "prod_ctc_greedy",
    "onnx_asr_ctc_greedy",
    "fused_tdt_greedy",
    "shared_ctc_greedy",
    "shared_tdt_greedy",
    "shared_tdt_dag2",
    "shared_tdt_dag4",
    "shared_tdt_dag2_duration_argmax",
    "shared_tdt_dag4_duration_argmax",
    "shared_tdt_dag8_duration_argmax",
    "shared_tdt_dag16_duration_argmax",
    "shared_tdt_dag2_duration_argmax_static_embedding",
    "shared_tdt_dag4_duration_argmax_static_embedding",
    "shared_tdt_dag8_duration_argmax_static_embedding",
];

#[allow(
    clippy::too_many_lines,
    reason = "the per-condition engine construction and sweep loop are kept together"
)]
fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };

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
    if let Some(limit) = args.limit {
        manifest.samples.truncate(limit);
    }
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;

    // SAFETY: this CLI is single-threaded until after this assignment and no
    // ONNX Runtime API is touched before it. ORT's dynamic loader reads this
    // process-global variable on the first API use.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };

    let model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
    let mut condition_summaries = Vec::new();
    for condition in &args.conditions {
        let output_path = args.output_dir.join(format!("{condition}.jsonl"));
        if output_path.exists() {
            eprintln!(
                "skipping {condition}: output already exists at {}",
                output_path.display()
            );
            continue;
        }
        let engine: Box<dyn AsrEngine> = match condition.as_str() {
            "prod_ctc_greedy" => Box::new(
                NvidiaCtcOrtAsrEngine::new(
                    &args.prod_ctc_model_dir,
                    model,
                    AsrPrecision::Int8,
                    args.threads,
                )
                .with_context(|| {
                    format!(
                        "failed to construct the legacy monolithic CTC baseline from {}",
                        args.prod_ctc_model_dir.display()
                    )
                })?,
            ),
            "onnx_asr_ctc_greedy" => Box::new(
                OnnxAsrCtcJaOrtEngine::new(&args.fused_model_dir, args.encoder_threads)
                    .with_context(|| {
                        format!(
                            "failed to construct the onnx-asr CTC engine from {}",
                            args.fused_model_dir.display()
                        )
                    })?,
            ),
            "fused_tdt_greedy" => Box::new(
                FusedTdtJaOrtEngine::new(&args.fused_model_dir, args.encoder_threads)
                    .with_context(|| {
                        format!(
                            "failed to construct the fused TDT engine from {}",
                            args.fused_model_dir.display()
                        )
                    })?,
            ),
            "shared_ctc_greedy" => Box::new(
                HybridParakeetJaOrtEngine::new(
                    &args.fused_model_dir,
                    args.encoder_threads,
                    args.decoder_threads,
                    ParakeetJaDecodingStrategy::CtcGreedy,
                )
                .with_context(|| {
                    format!(
                        "failed to construct the shared-encoder CTC engine from {}",
                        args.fused_model_dir.display()
                    )
                })?,
            ),
            "shared_tdt_greedy" => Box::new(
                HybridParakeetJaOrtEngine::new(
                    &args.fused_model_dir,
                    args.encoder_threads,
                    args.decoder_threads,
                    ParakeetJaDecodingStrategy::TdtGreedy,
                )
                .with_context(|| {
                    format!(
                        "failed to construct the shared-encoder TDT engine from {}",
                        args.fused_model_dir.display()
                    )
                })?,
            ),
            condition if dag_config_for_condition(condition).is_some() => {
                let config = dag_config_for_condition(condition)
                    .expect("the match guard established a DAG condition");
                let static_embedding = condition.ends_with("static_embedding");
                let strategy = if static_embedding {
                    ParakeetJaDecodingStrategy::TdtVariableDagStaticEmbedding(config)
                } else {
                    ParakeetJaDecodingStrategy::TdtVariableDag(config)
                };
                let engine = if static_embedding {
                    let static_embedding_dir =
                        args.static_embedding_dir.as_deref().ok_or_else(|| {
                            anyhow::anyhow!("condition {condition} requires --static-embedding-dir")
                        })?;
                    HybridParakeetJaOrtEngine::new_with_static_embedding(
                        &args.fused_model_dir,
                        args.encoder_threads,
                        args.decoder_threads,
                        strategy,
                        static_embedding_dir,
                    )
                } else {
                    HybridParakeetJaOrtEngine::new(
                        &args.fused_model_dir,
                        args.encoder_threads,
                        args.decoder_threads,
                        strategy,
                    )
                }
                .with_context(|| {
                    format!(
                        "failed to construct the shared-encoder TDT DAG{} engine from {}",
                        config.beam_size,
                        args.fused_model_dir.display()
                    )
                })?;
                Box::new(engine)
            }
            other => bail!("unsupported condition: {other}\n{USAGE}"),
        };
        let mut models = AsrModelRegistry::default();
        models.insert(model, engine)?;
        let mut service = OfflineTranscriptionService::new(models);

        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?;
        let mut output_writer = BufWriter::new(output);
        let summary = run_manifest(
            &args.manifest,
            &manifest,
            model,
            &mut service as &mut dyn EvalTranscriber,
            &mut output_writer,
        )?;
        eprintln!(
            "{condition}: total {} completed {} failed {}",
            summary.total, summary.completed, summary.failed
        );
        condition_summaries.push(ConditionSummary {
            condition: condition.clone(),
            output: output_path,
            total: summary.total,
            completed: summary.completed,
            failed: summary.failed,
        });
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&CommandSummary {
            schema_version: 1,
            split_id: &manifest.split_id,
            model_registry_key: model,
            artifact_family: "nvidia/parakeet-tdt_ctc-0.6b-ja",
            fused_artifact_precision: args.precision,
            production_ctc_precision: AsrPrecision::Int8,
            backend: "direct_ort",
            provider: "cpu",
            threads: args.threads,
            encoder_threads: args.encoder_threads,
            decoder_threads: args.decoder_threads,
            static_embedding_dir: args.static_embedding_dir.as_deref(),
            limit: args.limit,
            conditions: condition_summaries,
        })?
    );
    Ok(())
}

fn dag_config_for_condition(condition: &str) -> Option<FusedTdtDagConfig> {
    let (beam_size, duration_expansion) = match condition {
        "shared_tdt_dag2" => (2, FusedTdtDurationExpansion::All),
        "shared_tdt_dag4" => (4, FusedTdtDurationExpansion::All),
        "shared_tdt_dag2_duration_argmax" => (2, FusedTdtDurationExpansion::Argmax),
        "shared_tdt_dag4_duration_argmax" => (4, FusedTdtDurationExpansion::Argmax),
        "shared_tdt_dag8_duration_argmax" => (8, FusedTdtDurationExpansion::Argmax),
        "shared_tdt_dag16_duration_argmax" => (16, FusedTdtDurationExpansion::Argmax),
        "shared_tdt_dag2_duration_argmax_static_embedding" => {
            (2, FusedTdtDurationExpansion::Argmax)
        }
        "shared_tdt_dag4_duration_argmax_static_embedding" => {
            (4, FusedTdtDurationExpansion::Argmax)
        }
        "shared_tdt_dag8_duration_argmax_static_embedding" => {
            (8, FusedTdtDurationExpansion::Argmax)
        }
        _ => return None,
    };
    Some(FusedTdtDagConfig {
        beam_size,
        max_symbols_per_step: 10,
        duration_expansion,
    })
}

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    prod_ctc_model_dir: PathBuf,
    fused_model_dir: PathBuf,
    conditions: Vec<String>,
    limit: Option<usize>,
    threads: i32,
    encoder_threads: i32,
    decoder_threads: i32,
    precision: AsrPrecision,
    static_embedding_dir: Option<PathBuf>,
    ort_dylib: PathBuf,
    output_dir: PathBuf,
}

impl Args {
    #[allow(
        clippy::too_many_lines,
        reason = "the auditable CLI keeps validation for every reproducibility setting together"
    )]
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Option<Self>> {
        let arguments = arguments.collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            return Ok(None);
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
        for argument in arguments
            .iter()
            .filter(|argument| argument.starts_with('-'))
        {
            if ![
                "--manifest",
                "--split-id",
                "--prod-ctc-model-dir",
                "--fused-model-dir",
                "--conditions",
                "--limit",
                "--threads",
                "--encoder-threads",
                "--decoder-threads",
                "--precision",
                "--static-embedding-dir",
                "--ort-dylib",
                "--output-dir",
            ]
            .contains(&argument.as_str())
            {
                bail!("unknown argument {argument}\n{USAGE}");
            }
        }

        let threads = value("--threads")?
            .parse::<i32>()
            .context("--threads must be a positive integer")?;
        if threads <= 0 {
            bail!("--threads must be a positive integer");
        }
        let parse_optional_threads = |name: &str| -> Result<i32> {
            let value = optional_value(name)?
                .map(|raw| {
                    raw.parse::<i32>()
                        .with_context(|| format!("{name} must be a positive integer"))
                })
                .transpose()?
                .unwrap_or(threads);
            if value <= 0 {
                bail!("{name} must be a positive integer");
            }
            Ok(value)
        };
        let encoder_threads = parse_optional_threads("--encoder-threads")?;
        let decoder_threads = parse_optional_threads("--decoder-threads")?;
        let precision = match optional_value("--precision")?.unwrap_or("int8") {
            "float32" => AsrPrecision::Float32,
            "int8" => AsrPrecision::Int8,
            other => bail!("unsupported --precision {other:?}; expected float32 or int8"),
        };
        let conditions = match optional_value("--conditions")? {
            None => DEFAULT_CONDITIONS.map(str::to_owned).to_vec(),
            Some(list) => {
                let conditions = list
                    .split(',')
                    .map(str::trim)
                    .filter(|condition| !condition.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                if conditions.is_empty() {
                    bail!("--conditions must name at least one condition\n{USAGE}");
                }
                for condition in &conditions {
                    if !ALL_CONDITIONS.contains(&condition.as_str()) {
                        bail!("unsupported condition: {condition}\n{USAGE}");
                    }
                }
                conditions
            }
        };
        let limit = optional_value("--limit")?
            .map(|raw| {
                raw.parse::<usize>()
                    .context("--limit must be a positive integer")
            })
            .transpose()?;
        if limit == Some(0) {
            bail!("--limit must be a positive integer");
        }
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            prod_ctc_model_dir: PathBuf::from(value("--prod-ctc-model-dir")?),
            fused_model_dir: PathBuf::from(value("--fused-model-dir")?),
            conditions,
            limit,
            threads,
            encoder_threads,
            decoder_threads,
            precision,
            static_embedding_dir: optional_value("--static-embedding-dir")?.map(PathBuf::from),
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            output_dir: PathBuf::from(value("--output-dir")?),
        }))
    }
}

#[derive(Serialize)]
struct ConditionSummary {
    condition: String,
    output: PathBuf,
    total: usize,
    completed: usize,
    failed: usize,
}

#[derive(Serialize)]
struct CommandSummary<'a> {
    schema_version: u32,
    split_id: &'a str,
    model_registry_key: AsrModel,
    artifact_family: &'static str,
    fused_artifact_precision: AsrPrecision,
    production_ctc_precision: AsrPrecision,
    backend: &'static str,
    provider: &'static str,
    threads: i32,
    encoder_threads: i32,
    decoder_threads: i32,
    static_embedding_dir: Option<&'a std::path::Path>,
    limit: Option<usize>,
    conditions: Vec<ConditionSummary>,
}

#[cfg(test)]
mod tests {
    use super::{Args, dag_config_for_condition};
    use parapper_diagnostics::asr_eval::FusedTdtDurationExpansion;
    use parapper_models::asr::AsrPrecision;

    fn base_arguments() -> Vec<&'static str> {
        vec![
            "--manifest",
            "manifest.json",
            "--split-id",
            "jsut-basic5000-test-v1",
            "--prod-ctc-model-dir",
            "models/prod",
            "--fused-model-dir",
            "models/fused",
            "--threads",
            "4",
            "--ort-dylib",
            "onnxruntime.dll",
            "--output-dir",
            "results",
        ]
    }

    #[test]
    fn cli_defaults_to_every_condition_in_a_fixed_order() {
        let args = Args::parse(base_arguments().into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();
        assert_eq!(
            args.conditions,
            vec![
                "prod_ctc_greedy",
                "onnx_asr_ctc_greedy",
                "fused_tdt_greedy",
                "shared_ctc_greedy",
                "shared_tdt_greedy",
                "shared_tdt_dag2",
                "shared_tdt_dag4",
                "shared_tdt_dag2_duration_argmax",
                "shared_tdt_dag4_duration_argmax",
                "shared_tdt_dag8_duration_argmax",
                "shared_tdt_dag16_duration_argmax"
            ]
        );
        assert_eq!(args.limit, None);
        assert_eq!(args.threads, 4);
        assert_eq!(args.encoder_threads, 4);
        assert_eq!(args.decoder_threads, 4);
        assert_eq!(args.precision, AsrPrecision::Int8);
    }

    #[test]
    fn cli_allows_the_small_decoder_heads_to_use_fewer_threads_than_the_encoder() {
        let mut arguments = base_arguments();
        arguments.extend([
            "--encoder-threads",
            "4",
            "--decoder-threads",
            "1",
            "--precision",
            "float32",
        ]);
        let args = Args::parse(arguments.into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();

        assert_eq!(args.encoder_threads, 4);
        assert_eq!(args.decoder_threads, 1);
        assert_eq!(args.precision, AsrPrecision::Float32);
    }

    #[test]
    fn dag_conditions_map_to_the_requested_width_and_duration_expansion() {
        for (condition, beam_size, duration_expansion) in [
            ("shared_tdt_dag2", 2, FusedTdtDurationExpansion::All),
            ("shared_tdt_dag4", 4, FusedTdtDurationExpansion::All),
            (
                "shared_tdt_dag2_duration_argmax",
                2,
                FusedTdtDurationExpansion::Argmax,
            ),
            (
                "shared_tdt_dag4_duration_argmax",
                4,
                FusedTdtDurationExpansion::Argmax,
            ),
            (
                "shared_tdt_dag8_duration_argmax",
                8,
                FusedTdtDurationExpansion::Argmax,
            ),
            (
                "shared_tdt_dag16_duration_argmax",
                16,
                FusedTdtDurationExpansion::Argmax,
            ),
            (
                "shared_tdt_dag2_duration_argmax_static_embedding",
                2,
                FusedTdtDurationExpansion::Argmax,
            ),
            (
                "shared_tdt_dag4_duration_argmax_static_embedding",
                4,
                FusedTdtDurationExpansion::Argmax,
            ),
            (
                "shared_tdt_dag8_duration_argmax_static_embedding",
                8,
                FusedTdtDurationExpansion::Argmax,
            ),
        ] {
            let config = dag_config_for_condition(condition).unwrap();
            assert_eq!(config.beam_size, beam_size);
            assert_eq!(config.max_symbols_per_step, 10);
            assert_eq!(config.duration_expansion, duration_expansion);
        }
        assert!(dag_config_for_condition("shared_tdt_greedy").is_none());
    }

    #[test]
    fn cli_rejects_an_unknown_condition_and_a_zero_limit() {
        let mut unknown = base_arguments();
        unknown.extend(["--conditions", "prod_ctc_greedy,tdt_beam"]);
        assert!(
            Args::parse(unknown.into_iter().map(str::to_owned))
                .unwrap_err()
                .to_string()
                .contains("unsupported condition: tdt_beam")
        );

        let mut zero = base_arguments();
        zero.extend(["--limit", "0"]);
        assert!(
            Args::parse(zero.into_iter().map(str::to_owned))
                .unwrap_err()
                .to_string()
                .contains("--limit must be a positive integer")
        );

        let mut zero_decoder_threads = base_arguments();
        zero_decoder_threads.extend(["--decoder-threads", "0"]);
        assert!(
            Args::parse(zero_decoder_threads.into_iter().map(str::to_owned))
                .unwrap_err()
                .to_string()
                .contains("--decoder-threads must be a positive integer")
        );
    }
}
