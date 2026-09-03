use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use parapper_diagnostics::asr_eval::{
    EvalRecordV1, FusedTdtDagConfig, FusedTdtDurationExpansion, FusedTdtHotwordCandidatePolicy,
    HybridParakeetJaOrtEngine, ParakeetJaDecodingStrategy, RunnerManifestV1,
    ctc_gate_hotword_paths, decode_canonical_pcm16_wav, select_ctc_anchor_candidate,
};
use parapper_models::asr::{
    AsrModel, AsrTranscript,
    backend::JapaneseStaticEmbeddingModel,
    decoder::hotword::{HotwordContextGraph, HotwordPathKind, HotwordTokenPath},
};
use parapper_stt_engine::prepare_offline_model_input_audio;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const MODEL: AsrModel = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
const USAGE: &str = "Usage:
  run_parakeet_ja_hotword_eval \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --fused-model-dir <shared FP32 model directory> \\
    --encoder-threads <positive-integer> \\
    --decoder-threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --hotword-config <reazon_proper_noun_hotwords_v1.json> \\
    --corpus <jsut|common_voice> \\
    [--beam-sizes <comma-separated-positive-integers, defaults to 1,2,4,8,16>] \\
    [--multipliers <comma-separated-positive-floats, defaults to 1,100>] \\
    [--path-sets <surface,surface-readings; defaults to both>] \\
    [--candidate-policy <injected-all|direct-pre-topk; defaults to injected-all>] \\
    [--acoustic-top-k <positive-integer; defaults to 8>] \\
    [--ctc-gate-thresholds <comma-separated-non-positive-floats>] \\
    [--ctc-max-gap-frames <positive-integer; defaults to 8>] \\
    [--static-embedding-dir <snapshot-directory>] \\
    [--static-weights <comma-separated-non-negative-floats; defaults to 0>] \\
    [--limit <positive-sample-count; performance diagnostics only>] \\
    --output-dir <directory> \\
    [--target-only]";

fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    run(&args)
}

#[allow(
    clippy::too_many_lines,
    reason = "manifest preflight, shared encoding, and the diagnostic matrix stay auditable together"
)]
fn run(args: &Args) -> Result<()> {
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
    let fixture: HotwordFixture = serde_json::from_slice(
        &fs::read(&args.hotword_config)
            .with_context(|| format!("failed to read {}", args.hotword_config.display()))?,
    )
    .with_context(|| format!("failed to parse {}", args.hotword_config.display()))?;
    if fixture.version != 1 {
        bail!("unsupported hotword fixture version {}", fixture.version);
    }
    let corpus = fixture
        .corpora
        .get(&args.corpus)
        .with_context(|| format!("hotword fixture has no corpus {:?}", args.corpus))?;
    let path_sets = args
        .path_sets
        .iter()
        .copied()
        .map(|path_set| Ok((path_set, build_hotword_paths(corpus, path_set)?)))
        .collect::<Result<Vec<_>>>()?;
    if path_sets.iter().any(|(_, paths)| paths.is_empty()) || corpus.oracle_by_utterance.is_empty()
    {
        bail!("hotword corpus must contain fixed and oracle entries");
    }
    if !args.ort_dylib.is_file() {
        bail!(
            "ONNX Runtime library does not exist: {}",
            args.ort_dylib.display()
        );
    }
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;
    let mut outputs = create_outputs(args, &path_sets)?;

    // SAFETY: this diagnostic is single-threaded and has not called ORT yet.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };
    let mut engine = HybridParakeetJaOrtEngine::new(
        &args.fused_model_dir,
        args.encoder_threads,
        args.decoder_threads,
        ParakeetJaDecodingStrategy::TdtVariableDag(dag_config(1)),
    )?;
    let static_embedding = args
        .static_embedding_dir
        .as_deref()
        .map(JapaneseStaticEmbeddingModel::load)
        .transpose()?;
    let needs_ctc = args.ctc_gate_thresholds.is_some()
        || args.static_weights.iter().any(|&weight| weight > 0.0);
    let root = args.manifest.parent().unwrap_or_else(|| Path::new("."));
    let selected = manifest
        .samples
        .iter()
        .filter(|sample| {
            !args.target_only
                || corpus
                    .oracle_by_utterance
                    .contains_key(&sample.utterance_id)
        })
        .take(args.limit.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("no manifest samples matched the requested hotword evaluation");
    }

    for sample in &selected {
        let wav = match load_audio(root, sample) {
            Ok(wav) => wav,
            Err(error) => {
                write_all_failed(&mut outputs, &sample.utterance_id, "audio", &error)?;
                continue;
            }
        };
        let model_input = prepare_offline_model_input_audio(MODEL, &wav);
        let encoder_started = Instant::now();
        let encoded = match engine.encode(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                write_all_failed(&mut outputs, &sample.utterance_id, "encoder", &error)?;
                continue;
            }
        };
        let encoder_elapsed_ms = encoder_started.elapsed().as_secs_f64() * 1_000.0;
        let ctc_started = Instant::now();
        let ctc_output = if needs_ctc {
            match engine.ctc_output(&encoded) {
                Ok(output) => Some(output),
                Err(error) => {
                    write_all_failed(&mut outputs, &sample.utterance_id, "ctc", &error)?;
                    continue;
                }
            }
        } else {
            None
        };
        let anchor_embedding = if args.static_weights.iter().any(|&weight| weight > 0.0) {
            let transcript = match engine.decode_ctc_output(
                ctc_output
                    .as_ref()
                    .expect("static reranking requested CTC output"),
            ) {
                Ok(transcript) => transcript,
                Err(error) => {
                    write_all_failed(&mut outputs, &sample.utterance_id, "ctc", &error)?;
                    continue;
                }
            };
            let model = static_embedding
                .as_ref()
                .expect("positive static weight validated embedding model");
            match model.sentence_embedding(&transcript.text) {
                Ok(embedding) => Some(embedding),
                Err(error) => {
                    write_all_failed(&mut outputs, &sample.utterance_id, "static", &error)?;
                    continue;
                }
            }
        } else {
            None
        };
        let shared_elapsed_ms = encoder_elapsed_ms
            + if needs_ctc {
                ctc_started.elapsed().as_secs_f64() * 1_000.0
            } else {
                0.0
            };

        for condition in &mut outputs {
            let decoder_started = Instant::now();
            let config = dag_config(condition.beam_size);
            let active_paths = if let Some(threshold) = condition.gate_threshold {
                ctc_gate_hotword_paths(
                    ctc_output.as_ref().expect("CTC gate requested CTC output"),
                    &condition.paths,
                    threshold,
                    args.ctc_max_gap_frames,
                )
            } else {
                Ok(condition.paths.clone())
            };
            let active_paths = match active_paths {
                Ok(paths) => paths,
                Err(error) => {
                    write_condition_failed(condition, &sample.utterance_id, "ctc_gate", &error)?;
                    continue;
                }
            };
            let retained_entries = active_paths
                .iter()
                .map(|path| path.entry_id)
                .collect::<std::collections::HashSet<_>>()
                .len();
            condition.retained_entries_total += retained_entries;
            condition.zero_entry_utterances += usize::from(retained_entries == 0);
            let hotwords = match HotwordContextGraph::from_token_paths_with_phrase_multiplier(
                active_paths,
                condition.multiplier,
            ) {
                Ok(graph) => graph,
                Err(error) => {
                    write_condition_failed(condition, &sample.utterance_id, "hotword", &error)?;
                    continue;
                }
            };
            let candidates = match engine.decode_encoded_tdt_dag_hotword_candidates(
                &encoded,
                config,
                &hotwords,
                condition.candidate_policy.decoder_policy(),
            ) {
                Ok(candidates) => candidates,
                Err(error) => {
                    write_condition_failed(condition, &sample.utterance_id, "decoder", &error)?;
                    continue;
                }
            };
            let acoustic_scores = candidates
                .iter()
                .map(|candidate| candidate.acoustic_score)
                .collect::<Vec<_>>();
            let decoder_elapsed_ms =
                shared_elapsed_ms + decoder_started.elapsed().as_secs_f64() * 1_000.0;
            let static_started = Instant::now();
            let similarities = if condition
                .writers
                .iter()
                .any(|output| output.static_weight > 0.0)
            {
                let model = static_embedding
                    .as_ref()
                    .expect("positive static weight validated embedding model");
                let anchor = anchor_embedding
                    .as_ref()
                    .expect("positive static weight requested CTC anchor");
                let values = candidates
                    .iter()
                    .map(|candidate| {
                        let embedding = model.sentence_embedding(&candidate.transcript.text)?;
                        JapaneseStaticEmbeddingModel::cosine_similarity(anchor, &embedding)
                    })
                    .collect::<Result<Vec<_>>>();
                match values {
                    Ok(values) => Some(values),
                    Err(error) => {
                        write_condition_failed(condition, &sample.utterance_id, "static", &error)?;
                        continue;
                    }
                }
            } else {
                None
            };
            let static_elapsed_ms = static_started.elapsed().as_secs_f64() * 1_000.0;
            for output in &mut condition.writers {
                let selected = if output.static_weight == 0.0 {
                    Ok(0)
                } else {
                    select_ctc_anchor_candidate(
                        &acoustic_scores,
                        similarities
                            .as_ref()
                            .expect("positive static weight computed similarities"),
                        output.static_weight,
                    )
                };
                let result = selected.and_then(|index| {
                    candidates
                        .get(index)
                        .map(|candidate| candidate.transcript.clone())
                        .ok_or_else(|| anyhow::anyhow!("static reranker selected no candidate"))
                });
                write_result(
                    &mut output.writer,
                    sample,
                    wav.len() as u64,
                    decoder_elapsed_ms
                        + if output.static_weight > 0.0 {
                            static_elapsed_ms
                        } else {
                            0.0
                        },
                    result,
                )?;
            }
        }
    }
    for condition in &mut outputs {
        for output in &mut condition.writers {
            output.writer.flush()?;
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "schema_version": 1,
            "split_id": manifest.split_id,
            "corpus": args.corpus,
            "beam_sizes": args.beam_sizes,
            "phrase_multipliers": args.multipliers,
            "path_sets": args.path_sets.iter().map(|path_set| path_set.label()).collect::<Vec<_>>(),
            "candidate_policy": args.candidate_policy.label(),
            "acoustic_top_k": match args.candidate_policy {
                CandidatePolicy::InjectedAll => None,
                CandidatePolicy::DirectPreTopK { acoustic_top_k } => Some(acoustic_top_k),
            },
            "ctc_gate_thresholds": args.ctc_gate_thresholds,
            "ctc_max_gap_frames": args.ctc_max_gap_frames,
            "static_embedding_dir": args.static_embedding_dir,
            "static_weights": args.static_weights,
            "duration_expansion": "argmax",
            "target_only": args.target_only,
            "limit": args.limit,
            "selected_samples": selected.len(),
            "fixed_hotwords": corpus.fixed_hotwords.len(),
            "token_paths": path_sets.iter().map(|(path_set, paths)| (path_set.label(), paths.len())).collect::<HashMap<_, _>>(),
            "encoder_threads": args.encoder_threads,
            "decoder_threads": args.decoder_threads,
            "gate_retention": outputs.iter().map(|condition| serde_json::json!({
                "beam_size": condition.beam_size,
                "multiplier": condition.multiplier,
                "threshold": condition.gate_threshold,
                "retained_entries_total": condition.retained_entries_total,
                "zero_entry_utterances": condition.zero_entry_utterances,
            })).collect::<Vec<_>>(),
            "timing_scope": if needs_ctc {
                "shared_encoder_ctc_elapsed_plus_condition_decoder_and_rerank_elapsed"
            } else {
                "shared_encoder_elapsed_plus_condition_decoder_elapsed"
            },
            "encoder_runs_per_utterance": 1,
            "ctc_runs_per_utterance": usize::from(needs_ctc),
        })
    );
    Ok(())
}

fn dag_config(beam_size: usize) -> FusedTdtDagConfig {
    FusedTdtDagConfig {
        beam_size,
        max_symbols_per_step: 10,
        duration_expansion: FusedTdtDurationExpansion::Argmax,
    }
}

fn create_outputs(
    args: &Args,
    path_sets: &[(PathSet, Vec<HotwordTokenPath>)],
) -> Result<Vec<ConditionOutput>> {
    let mut outputs = Vec::new();
    let thresholds = args.ctc_gate_thresholds.as_deref().map_or_else(
        || vec![None],
        |values| values.iter().copied().map(Some).collect(),
    );
    for (path_set, paths) in path_sets {
        for &beam_size in &args.beam_sizes {
            for &multiplier in &args.multipliers {
                for &gate_threshold in &thresholds {
                    let multiplier_label = float_label(multiplier);
                    let policy_suffix = args.candidate_policy.output_suffix();
                    let gate_suffix = gate_threshold.map_or_else(String::new, |threshold| {
                        format!(
                            "-ctcgate-{}-gap{}",
                            float_label(threshold),
                            args.ctc_max_gap_frames,
                        )
                    });
                    let mut writers = Vec::new();
                    for &static_weight in &args.static_weights {
                        let static_suffix = if static_weight == 0.0 {
                            String::new()
                        } else {
                            format!("-ctcanchor-staticw{}", float_label(static_weight))
                        };
                        let output = args.output_dir.join(format!(
                            "parakeet-tdt-fp32-dag-argmax-beam{beam_size}{policy_suffix}-hotword-paths-{}-x{multiplier_label}{gate_suffix}{static_suffix}.jsonl",
                            path_set.label()
                        ));
                        writers.push(RerankOutput {
                            static_weight,
                            writer: create_new(&output)?,
                        });
                    }
                    outputs.push(ConditionOutput {
                        beam_size,
                        paths: paths.clone(),
                        multiplier,
                        gate_threshold,
                        candidate_policy: args.candidate_policy,
                        writers,
                        retained_entries_total: 0,
                        zero_entry_utterances: 0,
                    });
                }
            }
        }
    }
    Ok(outputs)
}

fn float_label(value: f32) -> String {
    value.to_string().replace('-', "m").replace('.', "p")
}

fn build_hotword_paths(
    corpus: &CorpusHotwords,
    path_set: PathSet,
) -> Result<Vec<HotwordTokenPath>> {
    let mut paths = Vec::new();
    for (entry_id, surface) in corpus.fixed_hotwords.iter().enumerate() {
        let surface_tokens = corpus
            .parakeet_tdt_token_ids
            .get(surface)
            .with_context(|| format!("missing Parakeet TDT tokens for {surface:?}"))?;
        if surface_tokens.is_empty() {
            bail!("Parakeet TDT token path for {surface:?} must not be empty");
        }
        paths.push(HotwordTokenPath {
            tokens: surface_tokens.clone(),
            entry_id,
            surface: surface.clone(),
            kind: HotwordPathKind::Surface,
            phrase_score: None,
        });
        if path_set == PathSet::SurfaceReadings {
            for reading_path in corpus
                .parakeet_tdt_reading_paths
                .get(surface)
                .into_iter()
                .flatten()
            {
                if reading_path.reading.trim().is_empty() || reading_path.token_ids.is_empty() {
                    bail!("reading path for {surface:?} must contain text and tokens");
                }
                if paths
                    .iter()
                    .any(|path| path.entry_id == entry_id && path.tokens == reading_path.token_ids)
                {
                    continue;
                }
                paths.push(HotwordTokenPath {
                    tokens: reading_path.token_ids.clone(),
                    entry_id,
                    surface: surface.clone(),
                    kind: HotwordPathKind::Reading,
                    phrase_score: None,
                });
            }
        }
    }
    Ok(paths)
}

fn create_new(path: &Path) -> Result<BufWriter<File>> {
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map(BufWriter::new)
        .with_context(|| format!("failed to create {}", path.display()))
}

fn load_audio(
    root: &Path,
    sample: &parapper_diagnostics::asr_eval::RunnerSampleV1,
) -> Result<Vec<f32>> {
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

fn write_result(
    writer: &mut BufWriter<File>,
    sample: &parapper_diagnostics::asr_eval::RunnerSampleV1,
    duration_samples: u64,
    inference_elapsed_ms: f64,
    result: Result<AsrTranscript>,
) -> Result<()> {
    let record = match result {
        Ok(transcript) => EvalRecordV1::completed(
            &sample.utterance_id,
            &sample.reference.normalized,
            transcript.text,
            duration_samples,
            inference_elapsed_ms,
        ),
        Err(error) => EvalRecordV1::failed(&sample.utterance_id, "decoder", error.to_string()),
    };
    serde_json::to_writer(&mut *writer, &record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_all_failed(
    outputs: &mut [ConditionOutput],
    utterance_id: &str,
    stage: &str,
    error: &anyhow::Error,
) -> Result<()> {
    let record = EvalRecordV1::failed(utterance_id, stage, error.to_string());
    for condition in outputs {
        for output in &mut condition.writers {
            serde_json::to_writer(&mut output.writer, &record)?;
            output.writer.write_all(b"\n")?;
        }
    }
    Ok(())
}

fn write_condition_failed(
    condition: &mut ConditionOutput,
    utterance_id: &str,
    stage: &str,
    error: &anyhow::Error,
) -> Result<()> {
    let record = EvalRecordV1::failed(utterance_id, stage, error.to_string());
    for output in &mut condition.writers {
        serde_json::to_writer(&mut output.writer, &record)?;
        output.writer.write_all(b"\n")?;
    }
    Ok(())
}

struct ConditionOutput {
    beam_size: usize,
    paths: Vec<HotwordTokenPath>,
    multiplier: f32,
    gate_threshold: Option<f32>,
    candidate_policy: CandidatePolicy,
    writers: Vec<RerankOutput>,
    retained_entries_total: usize,
    zero_entry_utterances: usize,
}

struct RerankOutput {
    static_weight: f32,
    writer: BufWriter<File>,
}

#[derive(Deserialize)]
struct HotwordFixture {
    version: u32,
    corpora: HashMap<String, CorpusHotwords>,
}

#[derive(Deserialize)]
struct CorpusHotwords {
    fixed_hotwords: Vec<String>,
    parakeet_tdt_token_ids: HashMap<String, Vec<usize>>,
    #[serde(default)]
    parakeet_tdt_reading_paths: HashMap<String, Vec<HotwordReadingPath>>,
    oracle_by_utterance: HashMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct HotwordReadingPath {
    reading: String,
    token_ids: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathSet {
    Surface,
    SurfaceReadings,
}

impl PathSet {
    const fn label(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::SurfaceReadings => "surface-readings",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "surface" => Ok(Self::Surface),
            "surface-readings" => Ok(Self::SurfaceReadings),
            _ => bail!("--path-sets must contain surface or surface-readings"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidatePolicy {
    InjectedAll,
    DirectPreTopK { acoustic_top_k: usize },
}

impl CandidatePolicy {
    const fn label(self) -> &'static str {
        match self {
            Self::InjectedAll => "injected-all",
            Self::DirectPreTopK { .. } => "direct-pre-topk",
        }
    }

    fn output_suffix(self) -> String {
        match self {
            Self::InjectedAll => String::new(),
            Self::DirectPreTopK { acoustic_top_k } => {
                format!("-direct-pretopk-a{acoustic_top_k}")
            }
        }
    }

    const fn decoder_policy(self) -> FusedTdtHotwordCandidatePolicy {
        match self {
            Self::InjectedAll => FusedTdtHotwordCandidatePolicy::InjectedAll,
            Self::DirectPreTopK { acoustic_top_k } => {
                FusedTdtHotwordCandidatePolicy::DirectPreTopK { acoustic_top_k }
            }
        }
    }
}

#[derive(Debug, PartialEq)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    fused_model_dir: PathBuf,
    encoder_threads: i32,
    decoder_threads: i32,
    ort_dylib: PathBuf,
    hotword_config: PathBuf,
    corpus: String,
    beam_sizes: Vec<usize>,
    multipliers: Vec<f32>,
    path_sets: Vec<PathSet>,
    candidate_policy: CandidatePolicy,
    ctc_gate_thresholds: Option<Vec<f32>>,
    ctc_max_gap_frames: usize,
    static_embedding_dir: Option<PathBuf>,
    static_weights: Vec<f32>,
    limit: Option<usize>,
    output_dir: PathBuf,
    target_only: bool,
}

impl Args {
    #[allow(
        clippy::too_many_lines,
        reason = "the standalone diagnostic keeps its complete CLI contract in one parser"
    )]
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Option<Self>> {
        let arguments = arguments.collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            return Ok(None);
        }
        let allowed = [
            "--manifest",
            "--split-id",
            "--fused-model-dir",
            "--encoder-threads",
            "--decoder-threads",
            "--ort-dylib",
            "--hotword-config",
            "--corpus",
            "--beam-sizes",
            "--multipliers",
            "--path-sets",
            "--candidate-policy",
            "--acoustic-top-k",
            "--ctc-gate-thresholds",
            "--ctc-max-gap-frames",
            "--static-embedding-dir",
            "--static-weights",
            "--limit",
            "--output-dir",
            "--target-only",
        ];
        for argument in arguments.iter().filter(|value| value.starts_with("--")) {
            if !allowed.contains(&argument.as_str()) {
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
        let optional = |name: &str| -> Result<Option<&str>> {
            arguments
                .iter()
                .position(|argument| argument == name)
                .map(|index| {
                    arguments
                        .get(index + 1)
                        .filter(|candidate| !candidate.starts_with("--"))
                        .map(String::as_str)
                        .with_context(|| format!("missing value for {name}\n{USAGE}"))
                })
                .transpose()
        };
        let encoder_threads = value("--encoder-threads")?.parse::<i32>()?;
        let decoder_threads = value("--decoder-threads")?.parse::<i32>()?;
        if encoder_threads <= 0 || decoder_threads <= 0 {
            bail!("thread counts must be positive");
        }
        let beam_sizes = optional("--beam-sizes")?
            .unwrap_or("1,2,4,8,16")
            .split(',')
            .map(|value| value.parse::<usize>().context("invalid --beam-sizes value"))
            .collect::<Result<Vec<_>>>()?;
        if beam_sizes.is_empty() || beam_sizes.contains(&0) {
            bail!("--beam-sizes must contain positive integers");
        }
        let multipliers = optional("--multipliers")?
            .unwrap_or("1,100")
            .split(',')
            .map(|value| value.parse::<f32>().context("invalid --multipliers value"))
            .collect::<Result<Vec<_>>>()?;
        if multipliers.is_empty()
            || multipliers
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("--multipliers must contain positive finite values");
        }
        let path_sets = optional("--path-sets")?
            .unwrap_or("surface,surface-readings")
            .split(',')
            .map(PathSet::parse)
            .collect::<Result<Vec<_>>>()?;
        if path_sets.is_empty() {
            bail!("--path-sets must not be empty");
        }
        let acoustic_top_k = optional("--acoustic-top-k")?
            .unwrap_or("8")
            .parse::<usize>()
            .context("invalid --acoustic-top-k value")?;
        let candidate_policy = match optional("--candidate-policy")?.unwrap_or("injected-all") {
            "injected-all" => CandidatePolicy::InjectedAll,
            "direct-pre-topk" => CandidatePolicy::DirectPreTopK { acoustic_top_k },
            _ => bail!("--candidate-policy must be injected-all or direct-pre-topk"),
        };
        if let CandidatePolicy::DirectPreTopK { acoustic_top_k } = candidate_policy
            && beam_sizes
                .iter()
                .any(|&beam_size| beam_size > acoustic_top_k)
        {
            bail!("--acoustic-top-k must be at least every requested beam size");
        }
        let ctc_gate_thresholds = optional("--ctc-gate-thresholds")?
            .map(|values| {
                values
                    .split(',')
                    .map(|value| {
                        value
                            .parse::<f32>()
                            .context("invalid --ctc-gate-thresholds value")
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        if ctc_gate_thresholds.as_ref().is_some_and(|values| {
            values.is_empty()
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value > 0.0)
        }) {
            bail!("--ctc-gate-thresholds must contain finite non-positive values");
        }
        if ctc_gate_thresholds.is_some()
            && !matches!(candidate_policy, CandidatePolicy::DirectPreTopK { .. })
        {
            bail!("CTC hotword gating requires --candidate-policy direct-pre-topk");
        }
        let ctc_max_gap_frames = optional("--ctc-max-gap-frames")?
            .unwrap_or("8")
            .parse::<usize>()
            .context("invalid --ctc-max-gap-frames value")?;
        if ctc_max_gap_frames == 0 {
            bail!("--ctc-max-gap-frames must be positive");
        }
        let static_weights = optional("--static-weights")?
            .unwrap_or("0")
            .split(',')
            .map(|value| {
                value
                    .parse::<f32>()
                    .context("invalid --static-weights value")
            })
            .collect::<Result<Vec<_>>>()?;
        if static_weights.is_empty()
            || static_weights
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            bail!("--static-weights must contain finite non-negative values");
        }
        let static_embedding_dir = optional("--static-embedding-dir")?.map(PathBuf::from);
        if static_weights.iter().any(|&weight| weight > 0.0) && static_embedding_dir.is_none() {
            bail!("positive --static-weights require --static-embedding-dir");
        }
        let limit = optional("--limit")?
            .map(|value| value.parse::<usize>().context("invalid --limit value"))
            .transpose()?;
        if limit == Some(0) {
            bail!("--limit must be positive");
        }
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            fused_model_dir: PathBuf::from(value("--fused-model-dir")?),
            encoder_threads,
            decoder_threads,
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            hotword_config: PathBuf::from(value("--hotword-config")?),
            corpus: value("--corpus")?.to_owned(),
            beam_sizes,
            multipliers,
            path_sets,
            candidate_policy,
            ctc_gate_thresholds,
            ctc_max_gap_frames,
            static_embedding_dir,
            static_weights,
            limit,
            output_dir: PathBuf::from(value("--output-dir")?),
            target_only: arguments.iter().any(|argument| argument == "--target-only"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use parapper_models::asr::decoder::hotword::{HotwordContextGraph, HotwordPathKind};

    use super::{
        Args, CandidatePolicy, CorpusHotwords, HotwordFixture, HotwordReadingPath, PathSet,
        build_hotword_paths,
    };

    #[test]
    fn surface_readings_adds_pronunciation_as_an_alternative_path_for_the_same_entry() {
        let corpus = CorpusHotwords {
            fixed_hotwords: vec!["固有名詞".to_owned()],
            parakeet_tdt_token_ids: HashMap::from([("固有名詞".to_owned(), vec![10, 11])]),
            parakeet_tdt_reading_paths: HashMap::from([(
                "固有名詞".to_owned(),
                vec![HotwordReadingPath {
                    reading: "こゆうめいし".to_owned(),
                    token_ids: vec![20, 21, 22],
                }],
            )]),
            oracle_by_utterance: HashMap::new(),
        };

        let surface = build_hotword_paths(&corpus, PathSet::Surface).unwrap();
        assert_eq!(surface.len(), 1);
        assert_eq!(surface[0].kind, HotwordPathKind::Surface);
        assert_eq!(surface[0].tokens, [10, 11]);

        let with_readings = build_hotword_paths(&corpus, PathSet::SurfaceReadings).unwrap();
        assert_eq!(with_readings.len(), 2);
        assert_eq!(with_readings[0].entry_id, with_readings[1].entry_id);
        assert_eq!(with_readings[1].kind, HotwordPathKind::Reading);
        assert_eq!(with_readings[1].surface, "固有名詞");
        assert_eq!(with_readings[1].tokens, [20, 21, 22]);
    }

    #[test]
    fn proper_noun_fixture_gives_every_surface_one_valid_reading_path() {
        let fixture: HotwordFixture = serde_json::from_str(include_str!(
            "../../fixtures/reazon_proper_noun_hotwords_v1.json"
        ))
        .unwrap();
        for corpus in fixture.corpora.values() {
            let surface = build_hotword_paths(corpus, PathSet::Surface).unwrap();
            let with_readings = build_hotword_paths(corpus, PathSet::SurfaceReadings).unwrap();
            assert_eq!(with_readings.len(), surface.len() * 2);
            HotwordContextGraph::from_token_paths_with_phrase_multiplier(with_readings, 100.0)
                .unwrap();
        }
    }

    #[test]
    fn kanji_proper_noun_fixture_tests_surface_against_registered_reading() {
        let fixture: HotwordFixture = serde_json::from_str(include_str!(
            "../../fixtures/kanji_proper_noun_hotwords_v1.json"
        ))
        .unwrap();
        let corpus = fixture.corpora.get("common_voice_train_kanji").unwrap();

        assert_eq!(corpus.fixed_hotwords.len(), 18);
        assert!(corpus.fixed_hotwords.iter().all(|surface| {
            surface.chars().any(|character| {
                matches!(
                    character,
                    '\u{3400}'..='\u{9fff}' | '\u{f900}'..='\u{faff}'
                )
            })
        }));
        assert!(corpus.fixed_hotwords.iter().all(|surface| {
            corpus
                .parakeet_tdt_reading_paths
                .get(surface)
                .is_some_and(|paths| paths.len() == 1 && !paths[0].token_ids.is_empty())
        }));
        HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            build_hotword_paths(corpus, PathSet::SurfaceReadings).unwrap(),
            100.0,
        )
        .unwrap();
    }

    #[test]
    fn latin_proper_noun_fixture_has_hiragana_and_katakana_paths() {
        let fixture: HotwordFixture = serde_json::from_str(include_str!(
            "../../fixtures/latin_proper_noun_hotwords_v1.json"
        ))
        .unwrap();
        let corpus = fixture.corpora.get("common_voice_train_latin").unwrap();

        assert_eq!(corpus.fixed_hotwords.len(), 6);
        assert!(corpus.fixed_hotwords.iter().all(|surface| {
            surface
                .chars()
                .any(|character| character.is_ascii_alphabetic())
        }));
        assert!(corpus.fixed_hotwords.iter().all(|surface| {
            corpus
                .parakeet_tdt_reading_paths
                .get(surface)
                .is_some_and(|paths| paths.len() == 2)
        }));
        HotwordContextGraph::from_token_paths_with_phrase_multiplier(
            build_hotword_paths(corpus, PathSet::SurfaceReadings).unwrap(),
            1.0,
        )
        .unwrap();
    }

    #[test]
    fn defaults_cover_neutral_and_x100_across_the_requested_widths() {
        let arguments = [
            "--manifest",
            "m.json",
            "--split-id",
            "split",
            "--fused-model-dir",
            "model",
            "--encoder-threads",
            "4",
            "--decoder-threads",
            "1",
            "--ort-dylib",
            "ort.dll",
            "--hotword-config",
            "h.json",
            "--corpus",
            "common_voice",
            "--output-dir",
            "out",
            "--target-only",
        ];
        let args = Args::parse(arguments.into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();
        assert_eq!(args.beam_sizes, [1, 2, 4, 8, 16]);
        assert_eq!(args.multipliers, [1.0, 100.0]);
        assert_eq!(args.path_sets, [PathSet::Surface, PathSet::SurfaceReadings]);
        assert_eq!(args.candidate_policy, CandidatePolicy::InjectedAll);
        assert_eq!(args.ctc_gate_thresholds, None);
        assert_eq!(args.ctc_max_gap_frames, 8);
        assert_eq!(args.static_weights, [0.0]);
        assert_eq!(args.static_embedding_dir, None);
        assert_eq!(args.limit, None);
        assert_eq!((args.encoder_threads, args.decoder_threads), (4, 1));
        assert!(args.target_only);
    }

    #[test]
    fn direct_pre_topk_requires_an_acoustic_shortlist_at_least_as_wide_as_the_beam() {
        let base = [
            "--manifest",
            "m.json",
            "--split-id",
            "split",
            "--fused-model-dir",
            "model",
            "--encoder-threads",
            "4",
            "--decoder-threads",
            "1",
            "--ort-dylib",
            "ort.dll",
            "--hotword-config",
            "h.json",
            "--corpus",
            "common_voice",
            "--beam-sizes",
            "2,4",
            "--candidate-policy",
            "direct-pre-topk",
            "--acoustic-top-k",
            "8",
            "--output-dir",
            "out",
        ];
        let args = Args::parse(base.into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();
        assert_eq!(
            args.candidate_policy,
            CandidatePolicy::DirectPreTopK { acoustic_top_k: 8 }
        );

        let invalid = base.map(|value| {
            if value == "8" {
                "1".to_owned()
            } else {
                value.to_owned()
            }
        });
        assert!(Args::parse(invalid.into_iter()).is_err());
    }

    #[test]
    fn ctc_gate_and_static_rerank_require_explicit_valid_dependencies() {
        let arguments = [
            "--manifest",
            "m.json",
            "--split-id",
            "split",
            "--fused-model-dir",
            "model",
            "--encoder-threads",
            "4",
            "--decoder-threads",
            "1",
            "--ort-dylib",
            "ort.dll",
            "--hotword-config",
            "h.json",
            "--corpus",
            "common_voice",
            "--beam-sizes",
            "4",
            "--candidate-policy",
            "direct-pre-topk",
            "--ctc-gate-thresholds",
            "-0.25,-1",
            "--static-embedding-dir",
            "static",
            "--static-weights",
            "0,0.1",
            "--limit",
            "100",
            "--output-dir",
            "out",
        ];
        let args = Args::parse(arguments.into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();

        assert_eq!(args.ctc_gate_thresholds, Some(vec![-0.25, -1.0]));
        assert_eq!(args.static_weights, [0.0, 0.1]);
        assert_eq!(args.static_embedding_dir, Some(PathBuf::from("static")));
        assert_eq!(args.limit, Some(100));
    }
}
