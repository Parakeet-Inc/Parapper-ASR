//! Dumps compact `UniDic` tokenizations for evaluation reference texts.
//!
//! Input is a UTF-8 TSV of `utterance_id<TAB>text` lines (one per utterance,
//! typically diagnostic-normalized references). Output is JSONL with one
//! record per line: `{"id", "tokens": [{"surface", "start", "end",
//! "feature"}]}` where `start`/`end` are character offsets into the input
//! text and `feature` is the packaged dictionary's compact four-digit
//! `[PP][S][F]` feature code.

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use vibrato_rkyv::{Dictionary, LoadMode, Tokenizer};

const USAGE: &str = "Usage:
  dump_ja_morphology \\
    --dictionary <system.dic> \\
    --input <references.tsv (utterance_id<TAB>text)> \\
    --output <morphology.jsonl>";

#[derive(Serialize)]
struct MorphToken {
    surface: String,
    start: usize,
    end: usize,
    feature: String,
}

#[derive(Serialize)]
struct MorphRecord<'a> {
    id: &'a str,
    tokens: Vec<MorphToken>,
}

fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{USAGE}");
        return Ok(());
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
        if !["--dictionary", "--input", "--output"].contains(&argument.as_str()) {
            bail!("unknown argument {argument}\n{USAGE}");
        }
    }
    let dictionary_path = PathBuf::from(value("--dictionary")?);
    let input_path = PathBuf::from(value("--input")?);
    let output_path = PathBuf::from(value("--output")?);
    if output_path.exists() {
        bail!("output already exists: {}", output_path.display());
    }

    let dictionary = Dictionary::from_path(&dictionary_path, LoadMode::TrustCache)
        .with_context(|| format!("failed to read {}", dictionary_path.display()))?;
    let tokenizer = Tokenizer::new(dictionary);
    let mut worker = tokenizer.new_worker();

    let input = fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut output = BufWriter::new(output);

    let mut records = 0_usize;
    for (line_index, line) in input.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let Some((id, text)) = line.split_once('\t') else {
            bail!(
                "invalid TSV line {} in {}: missing tab separator",
                line_index + 1,
                input_path.display()
            );
        };
        worker.reset_sentence(text);
        worker.tokenize();
        let tokens = (0..worker.num_tokens())
            .map(|index| {
                let token = worker.token(index);
                let range = token.range_char();
                MorphToken {
                    surface: token.surface().to_string(),
                    start: range.start,
                    end: range.end,
                    feature: token.feature().to_string(),
                }
            })
            .collect::<Vec<_>>();
        serde_json::to_writer(&mut output, &MorphRecord { id, tokens })
            .context("failed to serialize a morphology record")?;
        output
            .write_all(b"\n")
            .context("failed to write a morphology record")?;
        records += 1;
    }
    output.flush().context("failed to flush the output")?;
    eprintln!("wrote {records} records to {}", output_path.display());
    Ok(())
}
