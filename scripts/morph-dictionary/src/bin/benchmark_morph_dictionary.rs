use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    io::BufReader,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use vibrato_rkyv::{Dictionary, LoadMode, Tokenizer, tokenizer::worker::Worker};

const LOAD_ITERATIONS: usize = 5;
const TOKENIZE_ITERATIONS: usize = 7;
const WARMUP_ITERATIONS: usize = 1;

struct DictionaryBenchmark {
    label: String,
    path: PathBuf,
    tokenizer: Tokenizer,
    tokenize_times: Vec<Duration>,
    materialize_times: Vec<Duration>,
}

#[derive(Debug, PartialEq, Eq)]
struct PathToken {
    surface: String,
    char_range: std::ops::Range<usize>,
    word_cost: i16,
    total_cost: i32,
}

#[derive(Debug, Default)]
struct CorpusComparison {
    path_different: usize,
    cost_only_different: usize,
    identical: usize,
}

#[derive(Debug)]
struct Timings {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let Some(command) = args.next() else {
        bail!(usage());
    };
    match command.to_string_lossy().as_ref() {
        "prepare-legacy" => {
            let input = args.next().context(usage())?;
            let output = args.next().context(usage())?;
            if args.next().is_some() {
                bail!(usage());
            }
            prepare_legacy_dictionary(Path::new(&input), Path::new(&output))
        }
        "run" => {
            let corpus = args.next().context(usage())?;
            let dictionaries = args
                .map(|arg| parse_dictionary_arg(&arg))
                .collect::<Result<Vec<_>>>()?;
            if dictionaries.len() < 2 {
                bail!("run requires at least two LABEL=SYSTEM_DIC arguments");
            }
            run_benchmark(Path::new(&corpus), dictionaries)
        }
        "run-per-line" => {
            let corpus = args.next().context(usage())?;
            let dictionaries = args
                .map(|arg| parse_dictionary_arg(&arg))
                .collect::<Result<Vec<_>>>()?;
            if dictionaries.is_empty() {
                bail!("run-per-line requires at least one LABEL=SYSTEM_DIC argument");
            }
            run_per_line_benchmark(Path::new(&corpus), dictionaries)
        }
        _ => bail!(usage()),
    }
}

fn usage() -> &'static str {
    "usage:\n  benchmark_morph_dictionary prepare-legacy INPUT_SYSTEM_DIC_ZST OUTPUT_SYSTEM_DIC\n  benchmark_morph_dictionary run CORPUS LABEL=SYSTEM_DIC [LABEL=SYSTEM_DIC ...]\n  benchmark_morph_dictionary run-per-line CORPUS LABEL=SYSTEM_DIC [LABEL=SYSTEM_DIC ...]"
}

fn parse_dictionary_arg(arg: &std::ffi::OsStr) -> Result<(String, PathBuf)> {
    let arg = arg.to_string_lossy();
    let Some((label, path)) = arg.split_once('=') else {
        bail!("dictionary argument must be LABEL=SYSTEM_DIC: {arg}");
    };
    if label.is_empty() || path.is_empty() {
        bail!("dictionary argument must have a non-empty label and path: {arg}");
    }
    Ok((label.to_string(), PathBuf::from(path)))
}

fn prepare_legacy_dictionary(input: &Path, output: &Path) -> Result<()> {
    let input_file = fs::File::open(input)
        .with_context(|| format!("failed to open legacy dictionary {}", input.display()))?;
    let decoder = zstd::Decoder::new(BufReader::new(input_file))
        .with_context(|| format!("failed to decompress {}", input.display()))?;
    let dictionary = unsafe { Dictionary::from_legacy_reader(decoder) }
        .with_context(|| format!("failed to read legacy dictionary {}", input.display()))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temporary = output.with_extension("dic.writing");
    let mut writer = fs::File::create(&temporary)
        .with_context(|| format!("failed to create {}", temporary.display()))?;
    dictionary
        .write(&mut writer)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    drop(writer);
    if output.exists() {
        fs::remove_file(output)
            .with_context(|| format!("failed to replace {}", output.display()))?;
    }
    fs::rename(&temporary, output).with_context(|| {
        format!(
            "failed to move {} to {}",
            temporary.display(),
            output.display()
        )
    })?;
    Dictionary::from_path(output, LoadMode::Validate).with_context(|| {
        format!(
            "prepared dictionary failed validation: {}",
            output.display()
        )
    })?;
    println!(
        "PREPARED path={} bytes={}",
        output.display(),
        fs::metadata(output)?.len()
    );
    Ok(())
}

fn run_benchmark(corpus_path: &Path, dictionaries: Vec<(String, PathBuf)>) -> Result<()> {
    let corpus = fs::read_to_string(corpus_path)
        .with_context(|| format!("failed to read corpus {}", corpus_path.display()))?;
    let lines = corpus.lines().collect::<Vec<_>>();
    println!(
        "CORPUS path={} bytes={} lines={}",
        corpus_path.display(),
        corpus.len(),
        lines.len()
    );

    let mut benchmarks = Vec::with_capacity(dictionaries.len());
    for (label, path) in dictionaries {
        let file_bytes = fs::metadata(&path)
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len();
        let validate_times = benchmark_load(&path, LoadMode::Validate, LOAD_ITERATIONS)?;

        let trust_first_start = Instant::now();
        drop(load_tokenizer(&path, LoadMode::TrustCache)?);
        let trust_first = trust_first_start.elapsed();
        let trust_warm_times = benchmark_load(&path, LoadMode::TrustCache, LOAD_ITERATIONS)?;

        let tokenizer = load_tokenizer(&path, LoadMode::TrustCache)?;
        let fingerprint = corpus_path_fingerprint(&tokenizer, &lines);

        print_load_result(
            &label,
            &path,
            file_bytes,
            trust_first,
            &validate_times,
            &trust_warm_times,
            fingerprint,
        );
        benchmarks.push(DictionaryBenchmark {
            label,
            path,
            tokenizer,
            tokenize_times: Vec::with_capacity(TOKENIZE_ITERATIONS),
            materialize_times: Vec::with_capacity(TOKENIZE_ITERATIONS),
        });
    }

    for left in 0..benchmarks.len() {
        for right in (left + 1)..benchmarks.len() {
            let comparison = compare_corpus_paths(
                &benchmarks[left].tokenizer,
                &benchmarks[right].tokenizer,
                &lines,
            );
            println!(
                "COMPARE left={} right={} path_different_lines={} cost_only_different_lines={} identical_lines={} total_lines={}",
                benchmarks[left].label,
                benchmarks[right].label,
                comparison.path_different,
                comparison.cost_only_different,
                comparison.identical,
                lines.len(),
            );
        }
    }

    for _ in 0..WARMUP_ITERATIONS {
        for benchmark in &benchmarks {
            std::hint::black_box(tokenize_corpus(&benchmark.tokenizer, &lines));
            std::hint::black_box(materialize_corpus(&benchmark.tokenizer, &lines));
        }
    }

    for iteration in 0..TOKENIZE_ITERATIONS {
        for offset in 0..benchmarks.len() {
            let index = (iteration + offset) % benchmarks.len();
            let start = Instant::now();
            let result = tokenize_corpus(&benchmarks[index].tokenizer, &lines);
            benchmarks[index].tokenize_times.push(start.elapsed());
            std::hint::black_box(result);

            let start = Instant::now();
            let result = materialize_corpus(&benchmarks[index].tokenizer, &lines);
            benchmarks[index].materialize_times.push(start.elapsed());
            std::hint::black_box(result);
        }
    }

    for benchmark in benchmarks {
        print_processing_result(
            &benchmark.label,
            &benchmark.path,
            corpus.len(),
            &benchmark.tokenize_times,
            &benchmark.materialize_times,
        );
    }
    Ok(())
}

fn run_per_line_benchmark(corpus_path: &Path, dictionaries: Vec<(String, PathBuf)>) -> Result<()> {
    let corpus = fs::read_to_string(corpus_path)
        .with_context(|| format!("failed to read corpus {}", corpus_path.display()))?;
    let lines = corpus.lines().collect::<Vec<_>>();
    let mut benchmarks = dictionaries
        .into_iter()
        .map(|(label, path)| {
            Ok(DictionaryBenchmark {
                label,
                tokenizer: load_tokenizer(&path, LoadMode::TrustCache)?,
                path,
                tokenize_times: Vec::new(),
                materialize_times: Vec::with_capacity(TOKENIZE_ITERATIONS),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    for benchmark in &benchmarks {
        std::hint::black_box(materialize_corpus_per_line(&benchmark.tokenizer, &lines));
    }
    for iteration in 0..TOKENIZE_ITERATIONS {
        for offset in 0..benchmarks.len() {
            let index = (iteration + offset) % benchmarks.len();
            let start = Instant::now();
            let result = materialize_corpus_per_line(&benchmarks[index].tokenizer, &lines);
            benchmarks[index].materialize_times.push(start.elapsed());
            std::hint::black_box(result);
        }
    }

    for benchmark in benchmarks {
        let measured = timings(&benchmark.materialize_times);
        let line_count = u32::try_from(lines.len()).expect("line count should fit u32");
        println!(
            "PER_LINE label={} corpus_bytes={} lines={} total_min_ms={:.3} total_median_ms={:.3} total_mean_ms={:.3} total_max_ms={:.3} average_per_line_us={:.3} path={}",
            benchmark.label,
            corpus.len(),
            line_count,
            duration_ms(measured.min),
            duration_ms(measured.median),
            duration_ms(measured.mean),
            duration_ms(measured.max),
            measured.median.as_secs_f64() * 1_000_000.0 / f64::from(line_count),
            benchmark.path.display(),
        );
    }
    Ok(())
}

fn load_tokenizer(path: &Path, mode: LoadMode) -> Result<Tokenizer> {
    let dictionary = Dictionary::from_path(path, mode)
        .with_context(|| format!("failed to mmap-load {}", path.display()))?;
    Ok(Tokenizer::new(dictionary))
}

fn benchmark_load(path: &Path, mode: LoadMode, iterations: usize) -> Result<Vec<Duration>> {
    (0..iterations)
        .map(|_| {
            let start = Instant::now();
            drop(load_tokenizer(path, mode)?);
            Ok(start.elapsed())
        })
        .collect()
}

fn tokenize_corpus(tokenizer: &Tokenizer, lines: &[&str]) -> (usize, i64) {
    let mut worker = tokenizer.new_worker();
    let mut token_count = 0;
    let mut total_cost = 0_i64;
    for line in lines {
        worker.reset_sentence(line);
        worker.tokenize();
        token_count += worker.num_tokens();
        for index in 0..worker.num_tokens() {
            total_cost += i64::from(worker.token(index).total_cost());
        }
    }
    (token_count, total_cost)
}

fn materialize_corpus(tokenizer: &Tokenizer, lines: &[&str]) -> (usize, usize) {
    let mut worker = tokenizer.new_worker();
    let mut token_count = 0;
    let mut owned_bytes = 0;
    for line in lines {
        worker.reset_sentence(line);
        worker.tokenize();
        let tokens = (0..worker.num_tokens())
            .map(|index| {
                let token = worker.token(index);
                (
                    token.surface().to_string(),
                    token.range_char(),
                    token.feature().to_string(),
                )
            })
            .collect::<Vec<_>>();
        token_count += tokens.len();
        owned_bytes += tokens
            .iter()
            .map(|(surface, _, feature)| surface.len() + feature.len())
            .sum::<usize>();
        std::hint::black_box(tokens);
    }
    (token_count, owned_bytes)
}

fn materialize_corpus_per_line(tokenizer: &Tokenizer, lines: &[&str]) -> (usize, usize) {
    let mut token_count = 0;
    let mut owned_bytes = 0;
    for line in lines {
        let mut worker = tokenizer.new_worker();
        worker.reset_sentence(line);
        worker.tokenize();
        let tokens = (0..worker.num_tokens())
            .map(|index| {
                let token = worker.token(index);
                (
                    token.surface().to_string(),
                    token.range_char(),
                    token.feature().to_string(),
                )
            })
            .collect::<Vec<_>>();
        token_count += tokens.len();
        owned_bytes += tokens
            .iter()
            .map(|(surface, _, feature)| surface.len() + feature.len())
            .sum::<usize>();
        std::hint::black_box(tokens);
    }
    (token_count, owned_bytes)
}

fn corpus_path_fingerprint(tokenizer: &Tokenizer, lines: &[&str]) -> u64 {
    let mut worker = tokenizer.new_worker();
    let mut hasher = DefaultHasher::new();
    for line in lines {
        worker.reset_sentence(line);
        worker.tokenize();
        worker.num_tokens().hash(&mut hasher);
        for index in 0..worker.num_tokens() {
            let token = worker.token(index);
            token.surface().hash(&mut hasher);
            token.range_char().hash(&mut hasher);
            token.word_cost().hash(&mut hasher);
            token.total_cost().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn compare_corpus_paths(left: &Tokenizer, right: &Tokenizer, lines: &[&str]) -> CorpusComparison {
    let mut left_worker = left.new_worker();
    let mut right_worker = right.new_worker();
    let mut comparison = CorpusComparison::default();
    for line in lines {
        left_worker.reset_sentence(line);
        left_worker.tokenize();
        right_worker.reset_sentence(line);
        right_worker.tokenize();
        let left_tokens = path_tokens(&left_worker);
        let right_tokens = path_tokens(&right_worker);
        let same_path = left_tokens.len() == right_tokens.len()
            && left_tokens.iter().zip(&right_tokens).all(|(left, right)| {
                left.surface == right.surface && left.char_range == right.char_range
            });
        if !same_path {
            comparison.path_different += 1;
        } else if left_tokens != right_tokens {
            comparison.cost_only_different += 1;
        } else {
            comparison.identical += 1;
        }
    }
    comparison
}

fn path_tokens(worker: &Worker) -> Vec<PathToken> {
    (0..worker.num_tokens())
        .map(|index| {
            let token = worker.token(index);
            PathToken {
                surface: token.surface().to_string(),
                char_range: token.range_char(),
                word_cost: token.word_cost(),
                total_cost: token.total_cost(),
            }
        })
        .collect()
}

fn timings(values: &[Duration]) -> Timings {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let total = sorted.iter().sum::<Duration>();
    Timings {
        min: sorted[0],
        median: sorted[sorted.len() / 2],
        mean: total / u32::try_from(sorted.len()).expect("sample count should fit u32"),
        max: sorted[sorted.len() - 1],
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn print_load_result(
    label: &str,
    path: &Path,
    file_bytes: u64,
    trust_first: Duration,
    validate_times: &[Duration],
    trust_warm_times: &[Duration],
    fingerprint: u64,
) {
    let validate = timings(validate_times);
    let trust_warm = timings(trust_warm_times);
    println!(
        "LOAD label={label} bytes={file_bytes} validate_min_ms={:.3} validate_median_ms={:.3} validate_mean_ms={:.3} validate_max_ms={:.3} trust_first_ms={:.3} trust_warm_min_ms={:.3} trust_warm_median_ms={:.3} trust_warm_mean_ms={:.3} trust_warm_max_ms={:.3} fingerprint={fingerprint:016x} path={}",
        duration_ms(validate.min),
        duration_ms(validate.median),
        duration_ms(validate.mean),
        duration_ms(validate.max),
        duration_ms(trust_first),
        duration_ms(trust_warm.min),
        duration_ms(trust_warm.median),
        duration_ms(trust_warm.mean),
        duration_ms(trust_warm.max),
        path.display(),
    );
}

fn print_processing_result(
    label: &str,
    path: &Path,
    corpus_bytes: usize,
    tokenize_times: &[Duration],
    materialize_times: &[Duration],
) {
    let tokenize = timings(tokenize_times);
    let materialize = timings(materialize_times);
    let corpus_bytes = u32::try_from(corpus_bytes).expect("benchmark corpus size should fit u32");
    let throughput_mib_s =
        f64::from(corpus_bytes) / 1024.0 / 1024.0 / tokenize.median.as_secs_f64();
    println!(
        "PROCESS label={label} tokenize_min_ms={:.3} tokenize_median_ms={:.3} tokenize_mean_ms={:.3} tokenize_max_ms={:.3} tokenize_median_mib_s={throughput_mib_s:.3} materialize_min_ms={:.3} materialize_median_ms={:.3} materialize_mean_ms={:.3} materialize_max_ms={:.3} path={}",
        duration_ms(tokenize.min),
        duration_ms(tokenize.median),
        duration_ms(tokenize.mean),
        duration_ms(tokenize.max),
        duration_ms(materialize.min),
        duration_ms(materialize.median),
        duration_ms(materialize.mean),
        duration_ms(materialize.max),
        path.display(),
    );
}
