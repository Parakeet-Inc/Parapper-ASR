use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{EvalTranscriber, RunnerManifestV1, run_manifest};
use parapper_models::asr::{AsrModel, AsrPrecision, backend};
use parapper_stt_engine::{
    AsrModelRegistry, OfflineTranscriptionService, StreamingFileTranscriptionService,
};
use serde::Serialize;

const USAGE: &str = "Usage:
  run_asr_eval \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --model-dir <model-directory> \\
    --model <model-id> \\
    --precision <int8|int8_float32|float32> \\
    --decoding <greedy|modified_beam> \\
    [--beam-size <positive-integer>] \\
    [--initial-context <model_reference|sherpa>] \\
    [--unknown-token <emitting|non_emitting>] \\
    [--beam-pruning <state_dominance|full_prefix|sherpa>] \\
    [--network-batching <cached_unique|sherpa>] \\
    --threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --output <utterances.jsonl>";

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
    if args.output.exists() {
        bail!(
            "output already exists (resume/overwrite is not implemented yet): {}",
            args.output.display()
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
    if !args.model.supports_precision(args.precision) {
        bail!(
            "{:?} does not support precision {:?}",
            args.model,
            args.precision
        );
    }

    // SAFETY: this CLI is single-threaded until after this assignment and no
    // ONNX Runtime API is touched before it. ORT's dynamic loader reads this
    // process-global variable on the first API use.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };

    let engine = backend::build_engine_with_decoding(
        &args.model_dir,
        args.model,
        args.precision,
        args.threads,
        args.decoding,
    )
    .with_context(|| {
        format!(
            "failed to construct {:?} from {}",
            args.model,
            args.model_dir.display()
        )
    })?;
    let mut models = AsrModelRegistry::default();
    models.insert(args.model, engine)?;
    let mut service: Box<dyn EvalTranscriber> = if args.model.is_nemotron() {
        Box::new(StreamingFileTranscriptionService::new(models))
    } else {
        Box::new(OfflineTranscriptionService::new(models))
    };

    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;
    let mut output = BufWriter::new(output);
    let summary = run_manifest(
        &args.manifest,
        &manifest,
        args.model,
        service.as_mut(),
        &mut output,
    )?;
    let ablation = ablation_summary(args.decoding);

    println!(
        "{}",
        serde_json::to_string_pretty(&CommandSummary {
            schema_version: 1,
            split_id: &manifest.split_id,
            model: args.model,
            precision: args.precision,
            backend: if args.model == AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 {
                "shared_encoder_ctc_ort"
            } else {
                "direct_ort"
            },
            provider: "cpu",
            decoding: args.decoding.label(),
            beam_size: args.decoding.beam_size(),
            initial_context: ablation.initial_context,
            unknown_token: ablation.unknown_token,
            beam_pruning: ablation.beam_pruning,
            network_batching: ablation.network_batching,
            output: &args.output,
            total: summary.total,
            completed: summary.completed,
            failed: summary.failed,
        })?
    );
    Ok(())
}

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    model_dir: PathBuf,
    model: AsrModel,
    precision: AsrPrecision,
    decoding: backend::AsrDecodingStrategy,
    threads: i32,
    ort_dylib: PathBuf,
    output: PathBuf,
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
        for argument in arguments
            .iter()
            .filter(|argument| argument.starts_with('-'))
        {
            if ![
                "--manifest",
                "--split-id",
                "--model-dir",
                "--model",
                "--precision",
                "--decoding",
                "--beam-size",
                "--initial-context",
                "--unknown-token",
                "--beam-pruning",
                "--network-batching",
                "--threads",
                "--ort-dylib",
                "--output",
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
        let model = parse_model(value("--model")?)?;
        let decoding = parse_decoding(&arguments, model)?;
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            model_dir: PathBuf::from(value("--model-dir")?),
            model,
            precision: parse_precision(value("--precision")?)?,
            decoding,
            threads,
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            output: PathBuf::from(value("--output")?),
        }))
    }
}

fn parse_decoding(arguments: &[String], model: AsrModel) -> Result<backend::AsrDecodingStrategy> {
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
    match value("--decoding")? {
        "greedy" => {
            if [
                "--beam-size",
                "--initial-context",
                "--unknown-token",
                "--beam-pruning",
                "--network-batching",
            ]
            .iter()
            .any(|name| arguments.iter().any(|argument| argument == name))
            {
                bail!("beam options are valid only with --decoding modified_beam");
            }
            Ok(backend::AsrDecodingStrategy::Greedy)
        }
        "modified_beam" => {
            if model != AsrModel::ReazonSpeechK2V2 {
                bail!("modified_beam currently supports only reazonspeech_k2_v2");
            }
            let beam_size = optional_value("--beam-size")?
                .context("--beam-size is required with --decoding modified_beam")?
                .parse::<usize>()
                .context("--beam-size must be a positive integer")?;
            if beam_size == 0 {
                bail!("--beam-size must be a positive integer");
            }
            let initial_context = optional_value("--initial-context")?;
            let unknown_token = optional_value("--unknown-token")?;
            let beam_pruning = optional_value("--beam-pruning")?;
            let network_batching = optional_value("--network-batching")?;
            if initial_context.is_none()
                && unknown_token.is_none()
                && beam_pruning.is_none()
                && network_batching.is_none()
            {
                Ok(backend::AsrDecodingStrategy::ReazonModifiedBeam { beam_size })
            } else {
                let mut config = backend::ReazonBeamAblation::production(beam_size);
                if let Some(value) = initial_context {
                    config.initial_context = match value {
                        "model_reference" => backend::ReazonInitialContext::ModelReference,
                        "sherpa" => backend::ReazonInitialContext::Sherpa,
                        value => bail!("unsupported Reazon initial context: {value}"),
                    };
                }
                if let Some(value) = unknown_token {
                    config.unknown_is_non_emitting = match value {
                        "emitting" => false,
                        "non_emitting" => true,
                        value => bail!("unsupported Reazon unknown-token mode: {value}"),
                    };
                }
                if let Some(value) = beam_pruning {
                    config.pruning = match value {
                        "state_dominance" => backend::ReazonBeamPruning::StateDominance,
                        "full_prefix" => backend::ReazonBeamPruning::FullPrefix,
                        "sherpa" => backend::ReazonBeamPruning::Sherpa,
                        value => bail!("unsupported Reazon beam pruning: {value}"),
                    };
                }
                if let Some(value) = network_batching {
                    config.network_batching = match value {
                        "cached_unique" => backend::ReazonNetworkBatching::CachedUnique,
                        "sherpa" => backend::ReazonNetworkBatching::Sherpa,
                        value => bail!("unsupported Reazon network batching: {value}"),
                    };
                }
                Ok(backend::AsrDecodingStrategy::ReazonModifiedBeamAblation(
                    config,
                ))
            }
        }
        value => bail!("unsupported decoding strategy: {value}"),
    }
}

fn parse_model(value: &str) -> Result<AsrModel> {
    match value {
        "reazonspeech_k2_v2" => Ok(AsrModel::ReazonSpeechK2V2),
        "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8" => {
            Ok(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8)
        }
        "nemo_parakeet_tdt_0_6b_v2_int8" => Ok(AsrModel::NemoParakeetTdt0_6BV2Int8),
        "nemo_parakeet_tdt_0_6b_v3_int8" => Ok(AsrModel::NemoParakeetTdt0_6BV3Int8),
        "nemotron_speech_streaming_en_0_6b_160ms_int8" => {
            Ok(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8)
        }
        "nemotron_speech_streaming_en_0_6b_560ms_int8" => {
            Ok(AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8)
        }
        "nemotron_3_5_asr_streaming_0_6b_160ms_int8" => {
            Ok(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        }
        "nemotron_3_5_asr_streaming_0_6b_560ms_int8" => {
            Ok(AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8)
        }
        _ => bail!("unsupported offline model ID: {value}"),
    }
}

fn parse_precision(value: &str) -> Result<AsrPrecision> {
    match value {
        "int8" => Ok(AsrPrecision::Int8),
        "int8_float32" => Ok(AsrPrecision::Int8Float32),
        "float32" => Ok(AsrPrecision::Float32),
        _ => bail!("unsupported precision: {value}"),
    }
}

struct AblationSummary {
    initial_context: Option<&'static str>,
    unknown_token: Option<&'static str>,
    beam_pruning: Option<&'static str>,
    network_batching: Option<&'static str>,
}

fn ablation_summary(decoding: backend::AsrDecodingStrategy) -> AblationSummary {
    let Some(config) = decoding.reazon_ablation() else {
        return AblationSummary {
            initial_context: None,
            unknown_token: None,
            beam_pruning: None,
            network_batching: None,
        };
    };
    AblationSummary {
        initial_context: Some(match config.initial_context {
            backend::ReazonInitialContext::ModelReference => "model_reference",
            backend::ReazonInitialContext::Sherpa => "sherpa",
        }),
        unknown_token: Some(if config.unknown_is_non_emitting {
            "non_emitting"
        } else {
            "emitting"
        }),
        beam_pruning: Some(match config.pruning {
            backend::ReazonBeamPruning::StateDominance => "state_dominance",
            backend::ReazonBeamPruning::FullPrefix => "full_prefix",
            backend::ReazonBeamPruning::Sherpa => "sherpa",
        }),
        network_batching: Some(match config.network_batching {
            backend::ReazonNetworkBatching::CachedUnique => "cached_unique",
            backend::ReazonNetworkBatching::Sherpa => "sherpa",
        }),
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
    decoding: &'static str,
    beam_size: Option<usize>,
    initial_context: Option<&'static str>,
    unknown_token: Option<&'static str>,
    beam_pruning: Option<&'static str>,
    network_batching: Option<&'static str>,
    output: &'a PathBuf,
    total: usize,
    completed: usize,
    failed: usize,
}

#[cfg(test)]
mod tests {
    use parapper_models::asr::{AsrModel, AsrPrecision, backend};

    use super::Args;

    #[test]
    fn cli_requires_explicit_reproducibility_inputs() {
        let args = Args::parse(
            [
                "--manifest",
                "data/manifest.json",
                "--split-id",
                "cv26-ja-dev-strict-1000-v1",
                "--model-dir",
                "models/reazon",
                "--model",
                "reazonspeech_k2_v2",
                "--precision",
                "int8_float32",
                "--decoding",
                "greedy",
                "--threads",
                "4",
                "--ort-dylib",
                "runtime/onnxruntime.dll",
                "--output",
                "results/utterances.jsonl",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();

        assert_eq!(args.split_id, "cv26-ja-dev-strict-1000-v1");
        assert_eq!(args.model, AsrModel::ReazonSpeechK2V2);
        assert_eq!(args.precision, AsrPrecision::Int8Float32);
        assert_eq!(args.decoding, backend::AsrDecodingStrategy::Greedy);
        assert_eq!(args.threads, 4);
    }

    #[test]
    fn cli_rejects_implicit_runtime_and_nonpositive_threads() {
        let missing_runtime = Args::parse(
            [
                "--manifest",
                "manifest.json",
                "--split-id",
                "dev",
                "--model-dir",
                "model",
                "--model",
                "reazonspeech_k2_v2",
                "--precision",
                "int8",
                "--decoding",
                "greedy",
                "--threads",
                "1",
                "--output",
                "result.jsonl",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err()
        .to_string();
        assert!(missing_runtime.contains("--ort-dylib"));

        let invalid_threads = Args::parse(
            [
                "--manifest",
                "manifest.json",
                "--split-id",
                "dev",
                "--model-dir",
                "model",
                "--model",
                "reazonspeech_k2_v2",
                "--precision",
                "int8",
                "--decoding",
                "greedy",
                "--threads",
                "0",
                "--ort-dylib",
                "onnxruntime.dll",
                "--output",
                "result.jsonl",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err()
        .to_string();
        assert!(invalid_threads.contains("positive integer"));
    }

    #[test]
    fn cli_requires_a_positive_beam_only_for_reazon_modified_beam() {
        let base = [
            "--manifest",
            "manifest.json",
            "--split-id",
            "dev",
            "--model-dir",
            "model",
            "--model",
            "reazonspeech_k2_v2",
            "--precision",
            "int8",
            "--decoding",
            "modified_beam",
            "--threads",
            "4",
            "--ort-dylib",
            "onnxruntime.dll",
            "--output",
            "result.jsonl",
        ];
        let missing = Args::parse(base.into_iter().map(str::to_owned))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("--beam-size is required"));

        let mut valid = base.to_vec();
        valid.extend(["--beam-size", "4"]);
        let args = Args::parse(valid.into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();
        assert_eq!(
            args.decoding,
            backend::AsrDecodingStrategy::ReazonModifiedBeam { beam_size: 4 }
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cli_keeps_every_reazon_ablation_axis_explicit_in_the_decoder_config() {
        let args = Args::parse(
            [
                "--manifest",
                "manifest.json",
                "--split-id",
                "dev",
                "--model-dir",
                "model",
                "--model",
                "reazonspeech_k2_v2",
                "--precision",
                "int8",
                "--decoding",
                "modified_beam",
                "--beam-size",
                "8",
                "--initial-context",
                "sherpa",
                "--unknown-token",
                "non_emitting",
                "--beam-pruning",
                "sherpa",
                "--network-batching",
                "sherpa",
                "--threads",
                "4",
                "--ort-dylib",
                "onnxruntime.dll",
                "--output",
                "result.jsonl",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            args.decoding,
            backend::AsrDecodingStrategy::ReazonModifiedBeamAblation(backend::ReazonBeamAblation {
                beam_size: 8,
                initial_context: backend::ReazonInitialContext::Sherpa,
                unknown_is_non_emitting: true,
                pruning: backend::ReazonBeamPruning::Sherpa,
                network_batching: backend::ReazonNetworkBatching::Sherpa,
            })
        );

        let base = [
            "--manifest",
            "manifest.json",
            "--split-id",
            "dev",
            "--model-dir",
            "model",
            "--model",
            "reazonspeech_k2_v2",
            "--precision",
            "int8",
            "--decoding",
            "modified_beam",
            "--beam-size",
            "8",
            "--threads",
            "4",
            "--ort-dylib",
            "onnxruntime.dll",
            "--output",
            "result.jsonl",
        ];
        let cases = [
            (
                ["--initial-context", "sherpa"],
                backend::ReazonBeamAblation {
                    beam_size: 8,
                    initial_context: backend::ReazonInitialContext::Sherpa,
                    unknown_is_non_emitting: false,
                    pruning: backend::ReazonBeamPruning::StateDominance,
                    network_batching: backend::ReazonNetworkBatching::CachedUnique,
                },
            ),
            (
                ["--unknown-token", "non_emitting"],
                backend::ReazonBeamAblation {
                    beam_size: 8,
                    initial_context: backend::ReazonInitialContext::ModelReference,
                    unknown_is_non_emitting: true,
                    pruning: backend::ReazonBeamPruning::StateDominance,
                    network_batching: backend::ReazonNetworkBatching::CachedUnique,
                },
            ),
            (
                ["--beam-pruning", "sherpa"],
                backend::ReazonBeamAblation {
                    beam_size: 8,
                    initial_context: backend::ReazonInitialContext::ModelReference,
                    unknown_is_non_emitting: false,
                    pruning: backend::ReazonBeamPruning::Sherpa,
                    network_batching: backend::ReazonNetworkBatching::CachedUnique,
                },
            ),
        ];
        for (axis, expected) in cases {
            let args = Args::parse(base.into_iter().chain(axis).map(str::to_owned))
                .unwrap()
                .unwrap();
            assert_eq!(
                args.decoding,
                backend::AsrDecodingStrategy::ReazonModifiedBeamAblation(expected)
            );
        }
    }
}
