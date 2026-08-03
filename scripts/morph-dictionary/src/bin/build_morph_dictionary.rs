use std::{
    fs,
    io::{BufRead, BufReader, BufWriter, Cursor, Read, Write as _},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use parapper_morph_dictionary::morph_dictionary::compact_unidic_feature;
use vibrato_rkyv::{Dictionary, SystemDictionaryBuilder};

struct BuildArgs {
    lexicon: PathBuf,
    feature: PathBuf,
    right_id: PathBuf,
    left_id: PathBuf,
    model: PathBuf,
    dicrc: PathBuf,
    character: PathBuf,
    unknown: PathBuf,
    output: PathBuf,
}

struct BigramInfo {
    right: Vec<u8>,
    left: Vec<u8>,
    cost: Vec<u8>,
}

#[derive(Clone, Copy)]
enum DefinitionKind {
    Feature,
    Model,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let cost_factor = parse_cost_factor(BufReader::new(open_source(&args.dicrc)?))?;
    let bigram = generate_bigram_info(
        BufReader::new(open_source(&args.feature)?),
        BufReader::new(open_source(&args.right_id)?),
        BufReader::new(open_source(&args.left_id)?),
        BufReader::new(open_source(&args.model)?),
        cost_factor,
    )?;
    let lexicon = compact_source_csv(BufReader::new(open_source(&args.lexicon)?), "lexicon")?;
    let unknown = compact_source_csv(BufReader::new(open_source(&args.unknown)?), "unknown")?;
    let dictionary = build_compact_raw_native_dictionary(
        &lexicon,
        &bigram,
        BufReader::new(open_source(&args.character)?),
        &unknown,
    )?;
    write_zstd_dictionary(&dictionary, &args.output)?;
    println!(
        "wrote {} bytes to {}",
        fs::metadata(&args.output)?.len(),
        args.output.display()
    );
    Ok(())
}

fn parse_args() -> Result<BuildArgs> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: build_morph_dictionary LEXICON_CSV FEATURE_DEF RIGHT_ID_DEF LEFT_ID_DEF MODEL_DEF DICRC CHAR_DEF UNK_DEF OUTPUT_SYSTEM_DIC_ZST";
    let Some(lexicon) = args.next() else {
        bail!(usage)
    };
    let Some(feature) = args.next() else {
        bail!(usage)
    };
    let Some(right_id) = args.next() else {
        bail!(usage)
    };
    let Some(left_id) = args.next() else {
        bail!(usage)
    };
    let Some(model) = args.next() else {
        bail!(usage)
    };
    let Some(dicrc) = args.next() else {
        bail!(usage)
    };
    let Some(character) = args.next() else {
        bail!(usage)
    };
    let Some(unknown) = args.next() else {
        bail!(usage)
    };
    let Some(output) = args.next() else {
        bail!(usage)
    };
    if args.next().is_some() {
        bail!(usage);
    }
    Ok(BuildArgs {
        lexicon: lexicon.into(),
        feature: feature.into(),
        right_id: right_id.into(),
        left_id: left_id.into(),
        model: model.into(),
        dicrc: dicrc.into(),
        character: character.into(),
        unknown: unknown.into(),
        output: output.into(),
    })
}

fn open_source(path: &Path) -> Result<fs::File> {
    fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))
}

fn build_compact_raw_native_dictionary(
    lexicon: &[u8],
    bigram: &BigramInfo,
    character: impl Read,
    unknown: &[u8],
) -> Result<Dictionary> {
    let inner = SystemDictionaryBuilder::from_readers_with_bigram_info(
        Cursor::new(lexicon),
        bigram.right.as_slice(),
        bigram.left.as_slice(),
        bigram.cost.as_slice(),
        character,
        Cursor::new(unknown),
        false,
    )
    .context("failed to build compact Raw dictionary through vibrato-rkyv")?;
    Ok(Dictionary::from_inner(inner))
}

fn parse_cost_factor(reader: impl BufRead) -> Result<f64> {
    for line in reader.lines() {
        let line = line.context("failed to read dicrc")?;
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == "cost-factor" {
            return value
                .trim()
                .parse()
                .context("dicrc cost-factor must be a number");
        }
    }
    bail!("dicrc does not define cost-factor")
}

fn generate_bigram_info(
    feature: impl Read,
    right_id: impl Read,
    left_id: impl Read,
    model: impl Read,
    cost_factor: f64,
) -> Result<BigramInfo> {
    let feature = normalize_definition(feature, DefinitionKind::Feature)?;
    let model = normalize_definition(model, DefinitionKind::Model)?;
    let mut right = Vec::new();
    let mut left = Vec::new();
    let mut cost = Vec::new();
    vibrato::mecab::generate_bigram_info(
        Cursor::new(feature),
        right_id,
        left_id,
        Cursor::new(model),
        cost_factor,
        &mut right,
        &mut left,
        &mut cost,
    )
    .context("failed to generate small-dic bigram connection info")?;
    Ok(BigramInfo { right, left, cost })
}

fn normalize_definition(reader: impl Read, kind: DefinitionKind) -> Result<Vec<u8>> {
    let reader = BufReader::new(reader);
    let mut output = Vec::new();
    for line in reader.lines() {
        let line = line.context("failed to read MeCab model definition")?;
        let normalized = match kind {
            DefinitionKind::Feature => [" F_F ", " I_I "]
                .into_iter()
                .find_map(|marker| {
                    line.find(marker).map(|index| {
                        let label = marker.trim();
                        format!("{} {label}/{label}", &line[..index])
                    })
                })
                .unwrap_or(line),
            DefinitionKind::Model => line
                .strip_suffix("\tF_F")
                .map(|prefix| format!("{prefix}\tF_F/F_F"))
                .or_else(|| {
                    line.strip_suffix("\tI_I")
                        .map(|prefix| format!("{prefix}\tI_I/I_I"))
                })
                .unwrap_or(line),
        };
        writeln!(&mut output, "{normalized}")?;
    }
    Ok(output)
}

fn compact_source_csv(reader: impl Read, source_name: &str) -> Result<Vec<u8>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(reader);
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());

    for (row_index, record) in reader.records().enumerate() {
        let record = record
            .with_context(|| format!("failed to parse {source_name} CSV row {}", row_index + 1))?;
        if record.len() < 10 {
            bail!(
                "{source_name} CSV row {} has {} columns; expected four dictionary columns and at least six UniDic feature columns",
                row_index + 1,
                record.len()
            );
        }
        let feature = record.iter().skip(4).collect::<Vec<_>>().join(",");
        let compact = compact_unidic_feature(&feature).with_context(|| {
            format!("failed to compact {source_name} CSV row {}", row_index + 1)
        })?;
        let mut output_record = record.iter().take(4).collect::<Vec<_>>();
        output_record.push(&compact);
        writer.write_record(output_record).with_context(|| {
            format!(
                "failed to write compact {source_name} CSV row {}",
                row_index + 1
            )
        })?;
    }

    writer
        .into_inner()
        .map_err(csv::IntoInnerError::into_error)
        .with_context(|| format!("failed to finish compact {source_name} CSV"))
}

fn write_zstd_dictionary(dictionary: &Dictionary, output_path: &Path) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temporary_path = output_path.with_extension("zst.writing");
    let output = fs::File::create(&temporary_path)
        .with_context(|| format!("failed to create {}", temporary_path.display()))?;
    let writer = BufWriter::new(output);
    let mut encoder = zstd::Encoder::new(writer, 19)
        .with_context(|| format!("failed to start zstd encoder for {}", output_path.display()))?;
    dictionary.write(&mut encoder).with_context(|| {
        format!(
            "failed to serialize dictionary to {}",
            output_path.display()
        )
    })?;
    let mut writer = encoder
        .finish()
        .with_context(|| format!("failed to finish zstd output {}", output_path.display()))?;
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", output_path.display()))?;
    drop(writer);

    if output_path.exists() {
        fs::remove_file(output_path)
            .with_context(|| format!("failed to replace {}", output_path.display()))?;
    }
    fs::rename(&temporary_path, output_path).with_context(|| {
        format!(
            "failed to move {} to {}",
            temporary_path.display(),
            output_path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        io::{BufReader, Read as _},
        ops::Range,
        path::Path,
    };

    use parapper_morph_dictionary::morph_dictionary::compact_unidic_feature;
    use tempfile::tempdir;
    use vibrato_rkyv::{Dictionary, SystemDictionaryBuilder, Tokenizer, dictionary::MODEL_MAGIC};

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct TokenSnapshot {
        surface: String,
        char_range: Range<usize>,
        left_id: u16,
        right_id: u16,
        word_cost: i16,
        total_cost: i32,
        feature: String,
    }

    fn token_snapshots(
        dictionary: Dictionary,
        text: &str,
        compact_expected_feature: bool,
    ) -> Vec<TokenSnapshot> {
        let tokenizer = Tokenizer::new(dictionary);
        let mut worker = tokenizer.new_worker();
        worker.reset_sentence(text);
        worker.tokenize();
        (0..worker.num_tokens())
            .map(|index| {
                let token = worker.token(index);
                TokenSnapshot {
                    surface: token.surface().to_string(),
                    char_range: token.range_char(),
                    left_id: token.left_id(),
                    right_id: token.right_id(),
                    word_cost: token.word_cost(),
                    total_cost: token.total_cost(),
                    feature: if compact_expected_feature {
                        compact_unidic_feature(token.feature()).unwrap()
                    } else {
                        token.feature().to_string()
                    },
                }
            })
            .collect()
    }

    fn native_token_snapshots(tokenizer: &Tokenizer, text: &str) -> Vec<TokenSnapshot> {
        let mut worker = tokenizer.new_worker();
        worker.reset_sentence(text);
        worker.tokenize();
        (0..worker.num_tokens())
            .map(|index| {
                let token = worker.token(index);
                TokenSnapshot {
                    surface: token.surface().to_string(),
                    char_range: token.range_char(),
                    left_id: token.left_id(),
                    right_id: token.right_id(),
                    word_cost: token.word_cost(),
                    total_cost: token.total_cost(),
                    feature: token.feature().to_string(),
                }
            })
            .collect()
    }

    fn upstream_token_snapshots(tokenizer: &vibrato::Tokenizer, text: &str) -> Vec<TokenSnapshot> {
        let mut worker = tokenizer.new_worker();
        worker.reset_sentence(text);
        worker.tokenize();
        (0..worker.num_tokens())
            .map(|index| {
                let token = worker.token(index);
                TokenSnapshot {
                    surface: token.surface().to_string(),
                    char_range: token.range_char(),
                    left_id: token.left_id(),
                    right_id: token.right_id(),
                    word_cost: token.word_cost(),
                    total_cost: token.total_cost(),
                    feature: token.feature().to_string(),
                }
            })
            .collect()
    }

    fn release_path_snapshot(
        tokens: &[TokenSnapshot],
    ) -> Vec<(String, Range<usize>, i16, i32, String)> {
        tokens
            .iter()
            .map(|token| {
                (
                    token.surface.clone(),
                    token.char_range.clone(),
                    token.word_cost,
                    token.total_cost,
                    token.feature.clone(),
                )
            })
            .collect()
    }

    fn serialized_dictionary(dictionary: &Dictionary) -> Vec<u8> {
        let mut bytes = Vec::new();
        dictionary
            .write(&mut bytes)
            .expect("dictionary should serialize");
        bytes
    }

    fn test_bigram_info() -> BigramInfo {
        BigramInfo {
            right: b"1\tAB,*,CD,*,EF,*,GH,*,IJ,*,KL,*,MN,*,OP,*,QR,*,ST\n2\tUV,*,WX,*,YZ,*,12,*,34,*,56,*,78,*,90,*,*,*,*\n"
                .to_vec(),
            left: b"1\tuv,*,wx,*,yz,*,12,*,34,*,56,*,78,*,90,*,*,*,*\n2\tab,*,cd,*,ef,*,gh,*,ij,*,kl,*,mn,*,op,*,qr,*,st\n"
                .to_vec(),
            cost: b"AB/ab\t0\nCD/cd\t0\nEF/ef\t0\nGH/gh\t0\nIJ/ij\t0\nKL/kl\t0\nMN/mn\t0\nOP/op\t0\nQR/qr\t0\nST/st\t0\nUV/uv\t0\nWX/wx\t0\nYZ/yz\t0\n12/12\t0\n34/34\t0\n56/56\t0\n78/78\t0\n90/90\t0\n"
                .to_vec(),
        }
    }

    #[test]
    fn native_distribution_artifact_uses_raw_connector_and_preserves_token_paths() {
        let lexicon_csv = "東京,1,1,2,名詞,固有名詞,地名,一般,*,*\n行く,1,1,1,動詞,一般,*,*,五段-カ行,終止形-一般";
        let char_def = "DEFAULT 0 1 0";
        let unk_def = "DEFAULT,1,1,100,名詞,普通名詞,一般,*,*,*";
        let compact_lexicon =
            compact_source_csv(lexicon_csv.as_bytes(), "lexicon").expect("lexicon should compact");
        let compact_unknown =
            compact_source_csv(unk_def.as_bytes(), "unknown").expect("unknown should compact");
        let bigram = test_bigram_info();

        let native = build_compact_raw_native_dictionary(
            &compact_lexicon,
            &bigram,
            char_def.as_bytes(),
            &compact_unknown,
        )
        .expect("native compact Raw dictionary should build");
        let raw_inner = SystemDictionaryBuilder::from_readers_with_bigram_info(
            Cursor::new(&compact_lexicon),
            bigram.right.as_slice(),
            bigram.left.as_slice(),
            bigram.cost.as_slice(),
            char_def.as_bytes(),
            Cursor::new(&compact_unknown),
            false,
        )
        .expect("raw compact baseline should build");
        let raw = Dictionary::from_inner(raw_inner);

        assert_eq!(
            serialized_dictionary(&native),
            serialized_dictionary(&raw),
            "native distribution artifact must use the audited Raw connector representation"
        );
        assert_eq!(
            token_snapshots(native, "東京行く", false),
            token_snapshots(raw, "東京行く", false),
            "the native Raw artifact must preserve the complete token path"
        );
    }

    #[test]
    fn source_dictionary_build_preserves_forward_paths_and_stores_four_digit_features() {
        let lexicon_csv = "東京,1,1,2,名詞,固有名詞,地名,一般,*,*\n行く,1,1,1,動詞,一般,*,*,五段-カ行,終止形-一般";
        let matrix_def = "2 2\n0 0 0\n0 1 0\n1 0 0\n1 1 0";
        let char_def = "DEFAULT 0 1 0";
        let unk_def = "DEFAULT,1,1,100,名詞,普通名詞,一般,*,*,*";
        let compact_lexicon =
            compact_source_csv(lexicon_csv.as_bytes(), "lexicon").expect("lexicon should compact");
        let compact_unknown =
            compact_source_csv(unk_def.as_bytes(), "unknown").expect("unknown should compact");
        let bigram = test_bigram_info();

        let compact = build_compact_raw_native_dictionary(
            &compact_lexicon,
            &bigram,
            char_def.as_bytes(),
            &compact_unknown,
        )
        .expect("compact dictionary should build through upstream vibrato-rkyv");
        let dir = tempdir().expect("temporary directory should be created");
        let output_path = dir.path().join("system.dic.zst");
        write_zstd_dictionary(&compact, &output_path)
            .expect("compact dictionary should be written as zstd rkyv");

        let mut decoder = zstd::Decoder::new(BufReader::new(
            fs::File::open(&output_path).expect("compact output should open"),
        ))
        .expect("compact output should decompress");
        let mut magic = vec![0; MODEL_MAGIC.len()];
        decoder
            .read_exact(&mut magic)
            .expect("native rkyv magic should be readable");
        assert_eq!(magic, MODEL_MAGIC);

        for text in ["東京行く", "未知語"] {
            let original_inner = SystemDictionaryBuilder::from_readers(
                lexicon_csv.as_bytes(),
                matrix_def.as_bytes(),
                char_def.as_bytes(),
                unk_def.as_bytes(),
            )
            .expect("source dictionary should build");
            let expected = token_snapshots(Dictionary::from_inner(original_inner), text, true);

            let converted_input = fs::File::open(&output_path).expect("compact output should open");
            let converted_decoder = zstd::Decoder::new(BufReader::new(converted_input))
                .expect("compact output should decompress");
            let converted =
                Dictionary::read(converted_decoder).expect("compact rkyv dictionary should load");
            let actual = token_snapshots(converted, text, false);

            assert_eq!(
                actual, expected,
                "the complete forward token path changed for {text}"
            );
            assert!(
                actual
                    .iter()
                    .all(|token| token.feature.len() == 4 && token.feature.is_ascii()),
                "every distributed feature must use the four-digit format"
            );
        }
    }

    #[test]
    fn vibrato_release_model_special_bigram_features_are_normalized_before_generation() {
        let feature = b"BIGRAM B:%L[0]/%R[0]\nBIGRAM I_I %L?[11]/%R?[9],%R?[10]\nBIGRAM F_F %L?[9],%L?[10]/%R?[11]\n";
        let model = b"1.0\tB:a/b\n-3.0\tF_F\n-4.0\tI_I\n";

        assert_eq!(
            normalize_definition(feature.as_slice(), DefinitionKind::Feature).unwrap(),
            b"BIGRAM B:%L[0]/%R[0]\nBIGRAM I_I/I_I\nBIGRAM F_F/F_F\n"
        );
        assert_eq!(
            normalize_definition(model.as_slice(), DefinitionKind::Model).unwrap(),
            b"1.0\tB:a/b\n-3.0\tF_F/F_F\n-4.0\tI_I/I_I\n"
        );
        let cost_factor =
            parse_cost_factor(BufReader::new(b"; comment\ncost-factor = 700\n".as_slice()))
                .unwrap();
        assert_eq!(cost_factor.to_bits(), 700.0_f64.to_bits());
    }

    #[test]
    #[ignore = "requires generated artifacts and the official Vibrato v0.5.0 compact dictionary"]
    fn generated_real_dictionaries_match_vibrato_v0_5_0_compact_paths() {
        let artifact_dir = env::var_os("PARAPPER_MORPH_ARTIFACT_DIR")
            .expect("PARAPPER_MORPH_ARTIFACT_DIR must point to generated artifacts");
        let artifact_dir = Path::new(&artifact_dir);
        let upstream_path = env::var_os("PARAPPER_MORPH_UPSTREAM_ZST")
            .expect("PARAPPER_MORPH_UPSTREAM_ZST must point to the official compact dictionary");

        let native_decoder = zstd::Decoder::new(BufReader::new(
            fs::File::open(artifact_dir.join("system.dic.zst"))
                .expect("native compact dictionary should open"),
        ))
        .expect("native compact dictionary should decompress");
        let native_dictionary =
            Dictionary::read(native_decoder).expect("native compact dictionary should load");
        let upstream_decoder = zstd::Decoder::new(BufReader::new(
            fs::File::open(upstream_path).expect("official compact dictionary should open"),
        ))
        .expect("official compact dictionary should decompress");
        let upstream_dictionary = vibrato::Dictionary::read(upstream_decoder)
            .expect("official compact dictionary should load through upstream vibrato");
        let native_tokenizer = Tokenizer::new(native_dictionary);
        let upstream_tokenizer = vibrato::Tokenizer::new(upstream_dictionary);

        for text in [
            "今日は良い天気ですね。",
            "東京駅へ行きます",
            "行けると思うけど、まだ分かりません",
            "本がある",
            "行くか",
            "静か",
            "橋",
            "食べない",
            "これは私の",
            "これはABC123",
            "商品X9",
            "Version2β",
            "商品🙂",
            "東京𠮷野家",
            "はい",
            "これはABC123です！",
            "本とカレーの街神保町へようこそ。",
        ] {
            let native = native_token_snapshots(&native_tokenizer, text);
            let mut upstream = upstream_token_snapshots(&upstream_tokenizer, text);
            for token in &mut upstream {
                token.feature = compact_unidic_feature(&token.feature)
                    .expect("official UniDic feature should compact");
            }
            assert_eq!(
                release_path_snapshot(&native),
                release_path_snapshot(&upstream),
                "generated token path differs from Vibrato v0.5.0 compact for {text}"
            );
            assert!(
                native.iter().all(|token| {
                    token.feature.len() == 4
                        && token.feature.as_bytes().iter().all(u8::is_ascii_digit)
                }),
                "every observed real-dictionary feature must be four ASCII digits for {text}"
            );
        }
    }
}
