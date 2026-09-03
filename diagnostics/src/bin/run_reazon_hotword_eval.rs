use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use parapper_diagnostics::asr_eval::{EvalRecordV1, RunnerManifestV1, decode_canonical_pcm16_wav};
use parapper_models::asr::{
    AsrModel, AsrPrecision,
    backend::reazon_ort::{ReazonDecodingStrategy, ReazonSpeechOrtAsrEngine},
    decoder::hotword::{HotwordEntry, normalize_reading},
};
use parapper_stt_engine::prepare_offline_model_input_audio;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const USAGE: &str = "Usage:
  run_reazon_hotword_eval \\
    --manifest <manifest.json> \\
    --split-id <split-id> \\
    --model-dir <model-directory> \\
    --threads <positive-integer> \\
    --ort-dylib <onnxruntime-library> \\
    --hotword-config <reazon_proper_noun_hotwords_v1.json> \\
    --corpus <jsut|common_voice> \\
    [--beam-size <positive-integer, defaults to 4>] \\
    [--score-mode <token-score|phrase-multiplier>] \\
    [--path-sets <surface|surface-readings,...>] \\
    --scores <comma-separated-positive-floats> \\
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
    reason = "the diagnostic command keeps manifest preflight, cached inference, and JSONL output in one auditable flow"
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
    if corpus.fixed_hotwords.is_empty() || corpus.oracle_by_utterance.is_empty() {
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

    let mut outputs = create_outputs(
        &args.output_dir,
        args.beam_size,
        &args.scores,
        &args.path_sets,
    )?;
    // SAFETY: the command stays single-threaded and has not touched ORT yet.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib) };
    let mut engine = ReazonSpeechOrtAsrEngine::new_with_decoding(
        &args.model_dir,
        AsrModel::ReazonSpeechK2V2,
        AsrPrecision::Float32,
        args.threads,
        ReazonDecodingStrategy::ModifiedBeam {
            beam_size: args.beam_size,
        },
    )?;

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
        let model_input = prepare_offline_model_input_audio(AsrModel::ReazonSpeechK2V2, &wav);
        let encoder_started = Instant::now();
        let encoded = match engine.encode(model_input.as_ref()) {
            Ok(encoded) => encoded,
            Err(error) => {
                write_all_failed(&mut outputs, &sample.utterance_id, "encoder", &error)?;
                continue;
            }
        };
        let encoder_elapsed_ms = encoder_started.elapsed().as_secs_f64() * 1_000.0;
        let decoder_started = Instant::now();
        let baseline = engine.decode_encoded(
            &encoded,
            ReazonDecodingStrategy::ModifiedBeam {
                beam_size: args.beam_size,
            },
        );
        let baseline_elapsed_ms =
            encoder_elapsed_ms + decoder_started.elapsed().as_secs_f64() * 1_000.0;
        let baseline_transcript = baseline.as_ref().ok().cloned();
        write_result(
            &mut outputs.baseline,
            sample,
            wav.len() as u64,
            baseline_elapsed_ms,
            baseline,
        )?;

        for condition in &mut outputs.conditions {
            let surfaces = match condition.mode {
                HotwordMode::Fixed => Some(corpus.fixed_hotwords.as_slice()),
                HotwordMode::Oracle => corpus
                    .oracle_by_utterance
                    .get(&sample.utterance_id)
                    .map(Vec::as_slice),
            };
            let hotwords = surfaces
                .map(|surfaces| build_hotword_entries(corpus, surfaces, condition.path_set))
                .transpose()?;
            let decoder_started = Instant::now();
            let reused_baseline = args.score_mode == ScoreMode::PhraseMultiplier
                && condition.score.to_bits() == 1.0_f32.to_bits()
                && condition.path_set == PathSet::Surface;
            let result = if reused_baseline {
                baseline_transcript
                    .clone()
                    .ok_or_else(|| anyhow!("baseline decode failed for neutral hotword condition"))
            } else {
                match hotwords.as_deref() {
                    Some(hotwords) => match args.score_mode {
                        ScoreMode::TokenScore => engine.decode_encoded_with_hotword_entries(
                            &encoded,
                            args.beam_size,
                            hotwords,
                            condition.score,
                        ),
                        ScoreMode::PhraseMultiplier => engine
                            .decode_encoded_with_hotword_entries_phrase_multiplier(
                                &encoded,
                                args.beam_size,
                                hotwords,
                                condition.score,
                            ),
                    },
                    None => baseline_transcript
                        .clone()
                        .ok_or_else(|| anyhow!("baseline decode failed for non-oracle sample")),
                }
            };
            let elapsed_ms = if reused_baseline || hotwords.is_none() {
                baseline_elapsed_ms
            } else {
                encoder_elapsed_ms + decoder_started.elapsed().as_secs_f64() * 1_000.0
            };
            write_result(
                &mut condition.writer,
                sample,
                wav.len() as u64,
                elapsed_ms,
                result,
            )?;
        }
    }
    outputs.baseline.flush()?;
    for condition in &mut outputs.conditions {
        condition.writer.flush()?;
    }
    println!(
        "{}",
        serde_json::json!({
            "schema_version": 1,
            "split_id": manifest.split_id,
            "corpus": args.corpus,
            "beam_size": args.beam_size,
            "score_mode": args.score_mode.label(),
            "scores": args.scores,
            "path_sets": args.path_sets.iter().map(|path_set| path_set.label()).collect::<Vec<_>>(),
            "target_only": args.target_only,
            "selected_samples": selected.len(),
            "fixed_hotwords": corpus.fixed_hotwords.len(),
            "oracle_samples": corpus.oracle_by_utterance.len(),
            "timing_scope": "shared_encoder_elapsed_plus_condition_decoder_elapsed",
            "encoder_runs_per_utterance": 1,
        })
    );
    Ok(())
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
    result: Result<parapper_models::asr::AsrTranscript>,
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
    outputs: &mut Outputs,
    utterance_id: &str,
    stage: &str,
    error: &anyhow::Error,
) -> Result<()> {
    let record = EvalRecordV1::failed(utterance_id, stage, error.to_string());
    serde_json::to_writer(&mut outputs.baseline, &record)?;
    outputs.baseline.write_all(b"\n")?;
    for condition in &mut outputs.conditions {
        serde_json::to_writer(&mut condition.writer, &record)?;
        condition.writer.write_all(b"\n")?;
    }
    Ok(())
}

fn create_outputs(
    output_dir: &Path,
    beam_size: usize,
    scores: &[f32],
    path_sets: &[PathSet],
) -> Result<Outputs> {
    let baseline =
        create_new(&output_dir.join(format!("reazon-fp32-beam{beam_size}-baseline.jsonl")))?;
    let mut conditions = Vec::new();
    for &score in scores {
        for &path_set in path_sets {
            for mode in [HotwordMode::Fixed, HotwordMode::Oracle] {
                let score_label = score.to_string().replace('.', "p");
                let path_set_label = match path_set {
                    PathSet::Surface => String::new(),
                    PathSet::SurfaceReadings => format!("-paths-{}", path_set.label()),
                };
                let path = output_dir.join(format!(
                "reazon-fp32-beam{beam_size}-hotword-{}{path_set_label}-score{score_label}.jsonl",
                mode.label(),
            ));
                conditions.push(ConditionOutput {
                    mode,
                    path_set,
                    score,
                    writer: create_new(&path)?,
                });
            }
        }
    }
    Ok(Outputs {
        baseline,
        conditions,
    })
}

fn create_new(path: &Path) -> Result<BufWriter<File>> {
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map(BufWriter::new)
        .with_context(|| format!("failed to create {}", path.display()))
}

struct Outputs {
    baseline: BufWriter<File>,
    conditions: Vec<ConditionOutput>,
}

struct ConditionOutput {
    mode: HotwordMode,
    path_set: PathSet,
    score: f32,
    writer: BufWriter<File>,
}

#[derive(Clone, Copy)]
enum HotwordMode {
    Fixed,
    Oracle,
}

impl HotwordMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Oracle => "oracle",
        }
    }
}

#[derive(Deserialize)]
struct HotwordFixture {
    version: u32,
    corpora: HashMap<String, CorpusHotwords>,
}

#[derive(Deserialize)]
struct CorpusHotwords {
    fixed_hotwords: Vec<String>,
    #[serde(default)]
    parakeet_tdt_reading_paths: HashMap<String, Vec<HotwordReadingPath>>,
    oracle_by_utterance: HashMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct HotwordReadingPath {
    reading: String,
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

fn build_hotword_entries(
    corpus: &CorpusHotwords,
    surfaces: &[String],
    path_set: PathSet,
) -> Result<Vec<HotwordEntry>> {
    surfaces
        .iter()
        .map(|surface| {
            let mut readings = if path_set == PathSet::SurfaceReadings {
                corpus
                    .parakeet_tdt_reading_paths
                    .get(surface)
                    .with_context(|| format!("missing reading paths for {surface:?}"))?
                    .iter()
                    .map(|path| normalize_reading(&path.reading))
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            readings.sort();
            readings.dedup();
            Ok(HotwordEntry {
                surface: surface.clone(),
                readings,
                phrase_score: None,
            })
        })
        .collect()
}

#[derive(Debug, PartialEq)]
struct Args {
    manifest: PathBuf,
    split_id: String,
    model_dir: PathBuf,
    threads: i32,
    ort_dylib: PathBuf,
    hotword_config: PathBuf,
    corpus: String,
    beam_size: usize,
    score_mode: ScoreMode,
    scores: Vec<f32>,
    path_sets: Vec<PathSet>,
    output_dir: PathBuf,
    target_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreMode {
    TokenScore,
    PhraseMultiplier,
}

impl ScoreMode {
    const fn label(self) -> &'static str {
        match self {
            Self::TokenScore => "token-score",
            Self::PhraseMultiplier => "phrase-multiplier",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "token-score" => Ok(Self::TokenScore),
            "phrase-multiplier" => Ok(Self::PhraseMultiplier),
            _ => bail!("--score-mode must be token-score or phrase-multiplier"),
        }
    }
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
            "--model-dir",
            "--threads",
            "--ort-dylib",
            "--hotword-config",
            "--corpus",
            "--beam-size",
            "--score-mode",
            "--path-sets",
            "--scores",
            "--output-dir",
            "--target-only",
        ];
        for argument in arguments.iter().filter(|value| value.starts_with('-')) {
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
        let threads = value("--threads")?.parse::<i32>()?;
        if threads <= 0 {
            bail!("--threads must be positive");
        }
        let beam_size = arguments
            .iter()
            .position(|argument| argument == "--beam-size")
            .map(|index| {
                arguments
                    .get(index + 1)
                    .filter(|candidate| !candidate.starts_with("--"))
                    .context("missing value for --beam-size")?
                    .parse::<usize>()
                    .context("invalid --beam-size value")
            })
            .transpose()?
            .unwrap_or(4);
        if beam_size == 0 {
            bail!("--beam-size must be positive");
        }
        let scores = value("--scores")?
            .split(',')
            .map(|score| score.parse::<f32>().context("invalid --scores value"))
            .collect::<Result<Vec<_>>>()?;
        if scores.is_empty()
            || scores
                .iter()
                .any(|score| !score.is_finite() || *score <= 0.0)
        {
            bail!("--scores must contain positive finite values");
        }
        let score_mode = arguments
            .iter()
            .position(|argument| argument == "--score-mode")
            .map(|index| {
                arguments
                    .get(index + 1)
                    .filter(|candidate| !candidate.starts_with("--"))
                    .context("missing value for --score-mode")
                    .and_then(|value| ScoreMode::parse(value))
            })
            .transpose()?
            .unwrap_or(ScoreMode::TokenScore);
        let path_sets = arguments
            .iter()
            .position(|argument| argument == "--path-sets")
            .map(|index| {
                arguments
                    .get(index + 1)
                    .filter(|candidate| !candidate.starts_with("--"))
                    .context("missing value for --path-sets")?
                    .split(',')
                    .map(PathSet::parse)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_else(|| vec![PathSet::Surface]);
        if path_sets.is_empty() {
            bail!("--path-sets must not be empty");
        }
        Ok(Some(Self {
            manifest: PathBuf::from(value("--manifest")?),
            split_id: value("--split-id")?.to_owned(),
            model_dir: PathBuf::from(value("--model-dir")?),
            threads,
            ort_dylib: PathBuf::from(value("--ort-dylib")?),
            hotword_config: PathBuf::from(value("--hotword-config")?),
            corpus: value("--corpus")?.to_owned(),
            beam_size,
            score_mode,
            scores,
            path_sets,
            output_dir: PathBuf::from(value("--output-dir")?),
            target_only: arguments.iter().any(|argument| argument == "--target-only"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{Args, CorpusHotwords, HotwordFixture, PathSet, ScoreMode, build_hotword_entries};

    #[test]
    fn target_only_is_a_flag_and_scores_are_validated() {
        let arguments = [
            "--manifest",
            "m.json",
            "--split-id",
            "split",
            "--model-dir",
            "model",
            "--threads",
            "4",
            "--ort-dylib",
            "ort.dll",
            "--hotword-config",
            "h.json",
            "--corpus",
            "jsut",
            "--score-mode",
            "phrase-multiplier",
            "--scores",
            "0.5,1.0",
            "--output-dir",
            "out",
            "--target-only",
        ];
        let args = Args::parse(arguments.into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();
        assert_eq!(args.scores, vec![0.5, 1.0]);
        assert_eq!(args.beam_size, 4);
        assert_eq!(args.score_mode, ScoreMode::PhraseMultiplier);
        assert_eq!(args.path_sets, [PathSet::Surface]);
        assert!(args.target_only);
    }

    #[test]
    fn token_score_remains_the_default_for_existing_commands() {
        let arguments = [
            "--manifest",
            "m.json",
            "--split-id",
            "split",
            "--model-dir",
            "model",
            "--threads",
            "4",
            "--ort-dylib",
            "ort.dll",
            "--hotword-config",
            "h.json",
            "--corpus",
            "jsut",
            "--beam-size",
            "8",
            "--scores",
            "0.7",
            "--output-dir",
            "out",
        ];
        let args = Args::parse(arguments.into_iter().map(str::to_owned))
            .unwrap()
            .unwrap();
        assert_eq!(args.score_mode, ScoreMode::TokenScore);
        assert_eq!(args.beam_size, 8);
    }

    #[test]
    fn latin_fixture_deduplicates_hiragana_and_katakana_as_one_spoken_reading() {
        let fixture: HotwordFixture = serde_json::from_str(include_str!(
            "../../fixtures/latin_proper_noun_hotwords_v1.json"
        ))
        .unwrap();
        let corpus: &CorpusHotwords = fixture.corpora.get("common_voice_train_latin").unwrap();
        let entries =
            build_hotword_entries(corpus, &["Slovenia".to_owned()], PathSet::SurfaceReadings)
                .unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].surface, "Slovenia");
        assert_eq!(entries[0].readings, ["すろべにあ"]);
    }
}
