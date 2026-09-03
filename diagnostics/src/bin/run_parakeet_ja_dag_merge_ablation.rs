//! Compares TDT DAG merge keys and prune/merge ordering while sharing one
//! encoder output across all four decoder conditions.

use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{
    EvalRecordV1, FusedTdtDagConfig, FusedTdtDagMergeAblation, FusedTdtDagMergeOrder,
    FusedTdtDagStats, FusedTdtDurationExpansion, HybridParakeetJaOrtEngine, RunnerManifestV1,
    RunnerSampleV1, decode_canonical_pcm16_wav,
};
use parapper_models::asr::AsrModel;
use parapper_stt_engine::prepare_offline_model_input_audio;
use serde::Serialize;
use sha2::{Digest, Sha256};

const MODEL: AsrModel = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
const CONDITIONS: [Condition; 4] = [
    Condition {
        label: "current",
        merge: FusedTdtDagMergeAblation {
            order: FusedTdtDagMergeOrder::PruneThenMerge,
            include_symbols_since_advance_in_key: false,
        },
    },
    Condition {
        label: "symbols_key",
        merge: FusedTdtDagMergeAblation {
            order: FusedTdtDagMergeOrder::PruneThenMerge,
            include_symbols_since_advance_in_key: true,
        },
    },
    Condition {
        label: "merge_then_prune",
        merge: FusedTdtDagMergeAblation {
            order: FusedTdtDagMergeOrder::MergeThenPrune,
            include_symbols_since_advance_in_key: false,
        },
    },
    Condition {
        label: "merge_then_prune_symbols_key",
        merge: FusedTdtDagMergeAblation {
            order: FusedTdtDagMergeOrder::MergeThenPrune,
            include_symbols_since_advance_in_key: true,
        },
    },
];
const USAGE: &str = "Usage:
  run_parakeet_ja_dag_merge_ablation \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --fused-model-dir <shared FP32 model directory> \\
    --encoder-threads <positive-integer> \\
    --decoder-threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --output-dir <directory> \\
    [--limit <positive-sample-count>]";

fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    run(&args)
}

#[allow(
    clippy::too_many_lines,
    reason = "manifest validation, shared encoding, alternating decoding, and records form one audit"
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
    let manifest = RunnerManifestV1::parse(&manifest_bytes)
        .with_context(|| format!("failed to preflight {}", args.manifest.display()))?;
    if manifest.split_id != args.split_id {
        bail!(
            "split ID mismatch: CLI requested {:?}, manifest contains {:?}",
            args.split_id,
            manifest.split_id
        );
    }
    let selected = manifest
        .samples
        .iter()
        .take(args.limit.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("the selected manifest subset is empty");
    }
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;
    let mut outputs = CONDITIONS
        .iter()
        .map(|condition| {
            Ok(ConditionOutput {
                condition: *condition,
                writer: create_new(&args.output_dir.join(format!("{}.jsonl", condition.label)))?,
                totals: ConditionTotals::default(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // SAFETY: this diagnostic is single-threaded and has not called ORT yet.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };
    let config = FusedTdtDagConfig {
        beam_size: 2,
        max_symbols_per_step: 10,
        duration_expansion: FusedTdtDurationExpansion::Argmax,
    };
    let mut engine = HybridParakeetJaOrtEngine::new_tdt_dag(
        &args.fused_model_dir,
        args.encoder_threads,
        args.decoder_threads,
        config,
        false,
    )?;
    let root = args.manifest.parent().unwrap_or_else(|| Path::new("."));
    let mut encoder_elapsed_ms = 0.0;
    let mut total_audio_samples = 0_u64;
    let mut completed_samples = 0_usize;
    let mut output_differences_vs_current = [0_usize; CONDITIONS.len()];

    for (sample_index, sample) in selected.iter().enumerate() {
        let wav = match load_audio(root, sample) {
            Ok(wav) => wav,
            Err(error) => {
                write_all_failed(&mut outputs, &sample.utterance_id, "audio", &error)?;
                continue;
            }
        };
        let model_input = prepare_offline_model_input_audio(MODEL, &wav);
        let started = Instant::now();
        let encoded = match engine.encode(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                write_all_failed(&mut outputs, &sample.utterance_id, "encoder", &error)?;
                continue;
            }
        };
        encoder_elapsed_ms += started.elapsed().as_secs_f64() * 1_000.0;
        total_audio_samples += sample.audio.duration_samples;
        let mut hypotheses = vec![None; CONDITIONS.len()];

        // Rotate the first condition per utterance so warm-cache and thermal
        // effects do not systematically favor one decoder condition.
        for offset in 0..CONDITIONS.len() {
            let condition_index = (sample_index + offset) % CONDITIONS.len();
            let output = &mut outputs[condition_index];
            let started = Instant::now();
            let result = engine.decode_encoded_tdt_dag_with_merge_ablation(
                &encoded,
                config,
                output.condition.merge,
            );
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            match result {
                Ok((transcript, stats)) => {
                    hypotheses[condition_index] = Some(transcript.text.clone());
                    output.totals.add(elapsed_ms, stats);
                    write_record(
                        &mut output.writer,
                        &EvalRecordV1::completed(
                            &sample.utterance_id,
                            &sample.reference.normalized,
                            transcript.text,
                            sample.audio.duration_samples,
                            elapsed_ms,
                        ),
                    )?;
                }
                Err(error) => {
                    output.totals.failed_samples += 1;
                    write_record(
                        &mut output.writer,
                        &EvalRecordV1::failed(&sample.utterance_id, "decoder", error.to_string()),
                    )?;
                }
            }
        }
        if let Some(current) = hypotheses[0].as_deref() {
            for (index, hypothesis) in hypotheses.iter().enumerate().skip(1) {
                if hypothesis.as_deref().is_some_and(|text| text != current) {
                    output_differences_vs_current[index] += 1;
                }
            }
        }
        completed_samples += 1;
        if completed_samples.is_multiple_of(25) || completed_samples == selected.len() {
            eprintln!("completed {completed_samples}/{}", selected.len());
        }
    }

    for output in &mut outputs {
        output.writer.flush()?;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "the selected evaluation duration is far below f64's exact integer range"
    )]
    let audio_seconds = total_audio_samples as f64 / 16_000.0;
    let summaries = outputs
        .iter()
        .enumerate()
        .map(|(index, output)| ConditionSummary {
            label: output.condition.label,
            merge_order: match output.condition.merge.order {
                FusedTdtDagMergeOrder::PruneThenMerge => "prune_then_merge",
                FusedTdtDagMergeOrder::MergeThenPrune => "merge_then_prune",
            },
            include_symbols_since_advance_in_key: output
                .condition
                .merge
                .include_symbols_since_advance_in_key,
            decoder_elapsed_ms: output.totals.decoder_elapsed_ms,
            decoder_rtf: rtf(output.totals.decoder_elapsed_ms, audio_seconds),
            combined_encoder_decoder_rtf: rtf(
                encoder_elapsed_ms + output.totals.decoder_elapsed_ms,
                audio_seconds,
            ),
            failed_samples: output.totals.failed_samples,
            output_differences_vs_current: output_differences_vs_current[index],
            stats: &output.totals.stats,
        })
        .collect();
    let summary = Summary {
        schema_version: 1,
        split_id: &manifest.split_id,
        selected_samples: selected.len(),
        completed_samples,
        beam_size: config.beam_size,
        duration_expansion: "argmax",
        precision: "float32_model_artifacts",
        audio_seconds,
        encoder_elapsed_ms,
        encoder_rtf: rtf(encoder_elapsed_ms, audio_seconds),
        conditions: summaries,
    };
    let mut summary_writer = create_new(&args.output_dir.join("summary.json"))?;
    serde_json::to_writer_pretty(&mut summary_writer, &summary)?;
    summary_writer.write_all(b"\n")?;
    summary_writer.flush()?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn rtf(elapsed_ms: f64, audio_seconds: f64) -> f64 {
    elapsed_ms / 1_000.0 / audio_seconds
}

fn load_audio(root: &Path, sample: &RunnerSampleV1) -> Result<Vec<f32>> {
    let path = root.join(&sample.audio.relative_path);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if hash != sample.audio.sha256 {
        bail!("SHA-256 mismatch for {}", path.display());
    }
    let wav = decode_canonical_pcm16_wav(&bytes)?;
    if wav.samples.len() as u64 != sample.audio.duration_samples {
        bail!("duration mismatch for {}", path.display());
    }
    Ok(wav.samples)
}

fn create_new(path: &Path) -> Result<BufWriter<File>> {
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map(BufWriter::new)
        .with_context(|| format!("failed to create {}", path.display()))
}

fn write_record(writer: &mut BufWriter<File>, record: &EvalRecordV1) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_all_failed(
    outputs: &mut [ConditionOutput],
    utterance_id: &str,
    stage: &str,
    error: &anyhow::Error,
) -> Result<()> {
    for output in outputs {
        output.totals.failed_samples += 1;
        write_record(
            &mut output.writer,
            &EvalRecordV1::failed(utterance_id, stage, error.to_string()),
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Condition {
    label: &'static str,
    merge: FusedTdtDagMergeAblation,
}

struct ConditionOutput {
    condition: Condition,
    writer: BufWriter<File>,
    totals: ConditionTotals,
}

#[derive(Default)]
struct ConditionTotals {
    decoder_elapsed_ms: f64,
    failed_samples: usize,
    stats: StatsSummary,
}

impl ConditionTotals {
    fn add(&mut self, elapsed_ms: f64, stats: FusedTdtDagStats) {
        self.decoder_elapsed_ms += elapsed_ms;
        self.stats.add(stats);
    }
}

#[derive(Debug, Default, Serialize)]
struct StatsSummary {
    network_batches: u64,
    network_hypotheses: u64,
    generated_candidates: u64,
    max_active_width: usize,
    merge_calls: u64,
    merge_input_nodes: u64,
    duplicate_nodes_merged: u64,
    nodes_pruned: u64,
    underfilled_merge_calls: u64,
    symbol_budget_conflicts: u64,
}

impl StatsSummary {
    fn add(&mut self, stats: FusedTdtDagStats) {
        self.network_batches += stats.network_batches as u64;
        self.network_hypotheses += stats.network_hypotheses as u64;
        self.generated_candidates += stats.generated_candidates as u64;
        self.max_active_width = self.max_active_width.max(stats.max_active_width);
        self.merge_calls += stats.merge_calls as u64;
        self.merge_input_nodes += stats.merge_input_nodes as u64;
        self.duplicate_nodes_merged += stats.duplicate_nodes_merged as u64;
        self.nodes_pruned += stats.nodes_pruned as u64;
        self.underfilled_merge_calls += stats.underfilled_merge_calls as u64;
        self.symbol_budget_conflicts += stats.symbol_budget_conflicts as u64;
    }
}

#[derive(Serialize)]
struct Summary<'a> {
    schema_version: u32,
    split_id: &'a str,
    selected_samples: usize,
    completed_samples: usize,
    beam_size: usize,
    duration_expansion: &'static str,
    precision: &'static str,
    audio_seconds: f64,
    encoder_elapsed_ms: f64,
    encoder_rtf: f64,
    conditions: Vec<ConditionSummary<'a>>,
}

#[derive(Serialize)]
struct ConditionSummary<'a> {
    label: &'a str,
    merge_order: &'static str,
    include_symbols_since_advance_in_key: bool,
    decoder_elapsed_ms: f64,
    decoder_rtf: f64,
    combined_encoder_decoder_rtf: f64,
    failed_samples: usize,
    output_differences_vs_current: usize,
    stats: &'a StatsSummary,
}

struct Args {
    manifest: PathBuf,
    split_id: String,
    fused_model_dir: PathBuf,
    encoder_threads: i32,
    decoder_threads: i32,
    ort_dylib: PathBuf,
    output_dir: PathBuf,
    limit: Option<usize>,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Option<Self>> {
        let mut manifest = None;
        let mut split_id = None;
        let mut fused_model_dir = None;
        let mut encoder_threads = None;
        let mut decoder_threads = None;
        let mut ort_dylib = None;
        let mut output_dir = None;
        let mut limit = None;
        while let Some(flag) = args.next() {
            if matches!(flag.as_str(), "-h" | "--help") {
                return Ok(None);
            }
            let value = args
                .next()
                .with_context(|| format!("missing value for {flag}\n{USAGE}"))?;
            match flag.as_str() {
                "--manifest" => manifest = Some(PathBuf::from(value)),
                "--split-id" => split_id = Some(value),
                "--fused-model-dir" => fused_model_dir = Some(PathBuf::from(value)),
                "--encoder-threads" => encoder_threads = Some(parse_positive(&flag, &value)?),
                "--decoder-threads" => decoder_threads = Some(parse_positive(&flag, &value)?),
                "--ort-dylib" => ort_dylib = Some(PathBuf::from(value)),
                "--output-dir" => output_dir = Some(PathBuf::from(value)),
                "--limit" => limit = Some(parse_positive_usize(&flag, &value)?),
                _ => bail!("unknown argument {flag}\n{USAGE}"),
            }
        }
        Ok(Some(Self {
            manifest: manifest.with_context(|| format!("missing --manifest\n{USAGE}"))?,
            split_id: split_id.with_context(|| format!("missing --split-id\n{USAGE}"))?,
            fused_model_dir: fused_model_dir
                .with_context(|| format!("missing --fused-model-dir\n{USAGE}"))?,
            encoder_threads: encoder_threads
                .with_context(|| format!("missing --encoder-threads\n{USAGE}"))?,
            decoder_threads: decoder_threads
                .with_context(|| format!("missing --decoder-threads\n{USAGE}"))?,
            ort_dylib: ort_dylib.with_context(|| format!("missing --ort-dylib\n{USAGE}"))?,
            output_dir: output_dir.with_context(|| format!("missing --output-dir\n{USAGE}"))?,
            limit,
        }))
    }
}

fn parse_positive(flag: &str, value: &str) -> Result<i32> {
    let parsed = value
        .parse::<i32>()
        .with_context(|| format!("{flag} must be a positive integer"))?;
    if parsed <= 0 {
        bail!("{flag} must be positive");
    }
    Ok(parsed)
}

fn parse_positive_usize(flag: &str, value: &str) -> Result<usize> {
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("{flag} must be a positive integer"))?;
    if parsed == 0 {
        bail!("{flag} must be positive");
    }
    Ok(parsed)
}
