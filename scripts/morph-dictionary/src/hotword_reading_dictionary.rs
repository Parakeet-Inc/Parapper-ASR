use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
};

use anyhow::{Context, Result};
use unicode_normalization::UnicodeNormalization;

const MAX_SUDACHI_READINGS: usize = 3;
const MAX_CMU_PRONUNCIATIONS: usize = 2;

fn normalized_english_key(value: &str) -> Option<String> {
    let mut key = String::new();
    let mut previous_space = false;
    for character in value.nfkc().flat_map(char::to_lowercase) {
        if character.is_whitespace() {
            if !key.is_empty() && !previous_space {
                key.push(' ');
            }
            previous_space = true;
            continue;
        }
        if !character.is_ascii()
            || !(character.is_ascii_alphanumeric()
                || matches!(character, '-' | '_' | '.' | '+' | '#' | '\'' | '&'))
        {
            return None;
        }
        previous_space = false;
        key.push(character);
    }
    while key.ends_with(' ') {
        key.pop();
    }
    key.chars()
        .any(|character| character.is_ascii_alphabetic())
        .then_some(key)
}

fn is_kana_reading(value: &str) -> bool {
    !value.is_empty()
        && value.chars().all(|character| {
            matches!(character, '\u{3041}'..='\u{309f}' | '\u{30a0}'..='\u{30ff}' | ' ')
        })
}

/// Builds the compressed, sorted lookup tables embedded by the desktop app.
///
/// # Errors
///
/// Returns an error when either source cannot be parsed or an output cannot be written.
pub fn build_hotword_reading_dictionaries(
    sudachi_full_winfo: &Path,
    cmudict: &Path,
    output_directory: &Path,
) -> Result<()> {
    fs::create_dir_all(output_directory)
        .with_context(|| format!("failed to create {}", output_directory.display()))?;
    let (sudachi_english, sudachi_kanji) =
        collect_sudachi_dictionaries(BufReader::new(fs::File::open(sudachi_full_winfo)?))?;
    let cmu = collect_cmudict(BufReader::new(fs::File::open(cmudict)?))?;
    write_zstd_tsv(
        &output_directory.join("sudachi-en-readings.tsv.zst"),
        ranked_sudachi_rows(sudachi_english),
    )?;
    write_zstd_tsv(
        &output_directory.join("sudachi-kanji-readings.tsv.zst"),
        ranked_sudachi_rows(sudachi_kanji),
    )?;
    write_zstd_tsv(
        &output_directory.join("cmudict-arpabet.tsv.zst"),
        cmu.into_iter()
            .map(|(key, pronunciations)| (key, pronunciations.join("|"))),
    )
}

#[cfg(test)]
fn collect_sudachi_english(reader: impl BufRead) -> Result<SudachiReadings> {
    collect_sudachi(reader, normalized_english_key)
}

#[cfg(test)]
fn collect_sudachi_kanji(reader: impl BufRead) -> Result<SudachiReadings> {
    collect_sudachi(reader, normalized_kanji_key)
}

type SudachiReadings = BTreeMap<String, BTreeMap<String, i32>>;

fn collect_sudachi_dictionaries(
    reader: impl BufRead,
) -> Result<(SudachiReadings, SudachiReadings)> {
    let mut english = SudachiReadings::new();
    let mut kanji = SudachiReadings::new();
    visit_sudachi_rows(reader, |surface, reading, cost| {
        if let Some(key) = normalized_english_key(surface) {
            insert_sudachi_reading(&mut english, key, reading, cost);
        }
        if let Some(key) = normalized_kanji_key(surface) {
            insert_sudachi_reading(&mut kanji, key, reading, cost);
        }
    })?;
    Ok((english, kanji))
}

#[cfg(test)]
fn collect_sudachi(
    reader: impl BufRead,
    key_for_surface: fn(&str) -> Option<String>,
) -> Result<SudachiReadings> {
    let mut result = SudachiReadings::new();
    visit_sudachi_rows(reader, |surface, reading, cost| {
        if let Some(key) = key_for_surface(surface) {
            insert_sudachi_reading(&mut result, key, reading, cost);
        }
    })?;
    Ok(result)
}

fn visit_sudachi_rows(
    reader: impl BufRead,
    mut visitor: impl FnMut(&str, &str, i32),
) -> Result<()> {
    let mut rows = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(reader);
    for row in rows.records() {
        let row = row.context("invalid Sudachi winfo CSV row")?;
        if row.len() <= 11 {
            continue;
        }
        let reading = row[11].trim();
        if !is_kana_reading(reading) {
            continue;
        }
        let cost = row[3].parse::<i32>().unwrap_or(i32::MAX);
        visitor(&row[0], reading, cost);
    }
    Ok(())
}

fn insert_sudachi_reading(dictionary: &mut SudachiReadings, key: String, reading: &str, cost: i32) {
    let previous = dictionary
        .entry(key)
        .or_default()
        .entry(reading.into())
        .or_insert(cost);
    *previous = (*previous).min(cost);
}

fn normalized_kanji_key(value: &str) -> Option<String> {
    let key = value.nfkc().collect::<String>().trim().to_owned();
    contains_kanji(&key).then_some(key)
}

fn contains_kanji(value: &str) -> bool {
    value.chars().any(|character| {
        matches!(
            character,
            '\u{3400}'..='\u{4dbf}'
                | '\u{4e00}'..='\u{9fff}'
                | '\u{f900}'..='\u{faff}'
                | '\u{20000}'..='\u{2fa1f}'
        )
    })
}

fn ranked_sudachi_rows(rows: SudachiReadings) -> impl Iterator<Item = (String, String)> {
    rows.into_iter().map(|(key, values)| {
        let mut values = values.into_iter().collect::<Vec<_>>();
        values.sort_by(|(left_reading, left_cost), (right_reading, right_cost)| {
            left_cost
                .cmp(right_cost)
                .then_with(|| left_reading.cmp(right_reading))
        });
        let readings = values
            .into_iter()
            .take(MAX_SUDACHI_READINGS)
            .map(|(reading, _)| reading)
            .collect::<Vec<_>>()
            .join(",");
        (key, readings)
    })
}

fn collect_cmudict(reader: impl BufRead) -> Result<BTreeMap<String, Vec<String>>> {
    let mut result = BTreeMap::<String, Vec<String>>::new();
    for line in reader.lines() {
        let line = line.context("invalid CMUdict line")?;
        if line.starts_with(";;; ") || line.trim().is_empty() {
            continue;
        }
        let Some((raw_word, pronunciation)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let word = raw_word
            .split_once('(')
            .map_or(raw_word, |(base, _)| base)
            .to_ascii_lowercase();
        if normalized_english_key(&word).as_deref() != Some(word.as_str()) {
            continue;
        }
        let pronunciation = pronunciation.trim().to_owned();
        let pronunciations = result.entry(word).or_default();
        if !pronunciation.is_empty()
            && !pronunciations.contains(&pronunciation)
            && pronunciations.len() < MAX_CMU_PRONUNCIATIONS
        {
            pronunciations.push(pronunciation);
        }
    }
    Ok(result)
}

fn write_zstd_tsv(path: &Path, rows: impl IntoIterator<Item = (String, String)>) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = zstd::Encoder::new(file, 19)?;
    for (key, value) in rows {
        writeln!(writer, "{key}\t{value}")?;
    }
    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read};

    #[test]
    fn sudachi_entries_are_nfkc_case_folded_ranked_and_limited() {
        let input = [
            "ＡＭＡＺＯＮ,0,0,10,ＡＭＡＺＯＮ,名詞,普通名詞,一般,*,*,*,アマゾン",
            "amazon,0,0,5,amazon,名詞,普通名詞,一般,*,*,*,アマゾーン",
            "amazon,0,0,20,amazon,名詞,普通名詞,一般,*,*,*,アマゾニア",
            "amazon,0,0,30,amazon,名詞,普通名詞,一般,*,*,*,アマゾニック",
            "amazon,0,0,40,amazon,名詞,普通名詞,一般,*,*,*,アマゾン",
            "東京,0,0,1,東京,名詞,固有名詞,地名,*,*,*,トウキョウ",
        ]
        .join("\n");

        let rows = collect_sudachi_english(Cursor::new(input)).unwrap();
        let mut ranked = rows["amazon"]
            .iter()
            .map(|(reading, cost)| (*cost, reading.clone()))
            .collect::<Vec<_>>();
        ranked.sort();
        ranked.truncate(3);
        assert_eq!(
            ranked,
            vec![
                (5, "アマゾーン".into()),
                (10, "アマゾン".into()),
                (20, "アマゾニア".into())
            ]
        );
        assert!(!rows.contains_key("東京"));
    }

    #[test]
    fn sudachi_kanji_entries_keep_exact_surface_readings_without_hiragana_only_rows() {
        let input = [
            "雪崩,0,0,20,雪崩,名詞,普通名詞,一般,*,*,*,ナダレ",
            "雪崩,0,0,10,雪崩,名詞,普通名詞,一般,*,*,*,セッポウ",
            "雪崩,0,0,30,雪崩,名詞,普通名詞,一般,*,*,*,ナダレ",
            "なだれ,0,0,1,なだれ,名詞,普通名詞,一般,*,*,*,ナダレ",
            "Ａ東京,0,0,1,Ａ東京,名詞,固有名詞,地名,*,*,*,エートウキョウ",
        ]
        .join("\n");

        let kanji = collect_sudachi_kanji(Cursor::new(input)).unwrap();
        assert_eq!(
            kanji["雪崩"],
            BTreeMap::from([("セッポウ".into(), 10), ("ナダレ".into(), 20)])
        );
        assert!(!kanji.contains_key("なだれ"));
        assert!(kanji.contains_key("A東京"));
    }

    #[test]
    fn cmudict_keeps_two_exact_word_pronunciations_without_splitting_names() {
        let input = "open  OW1 P AH0 N\nopen(2)  OW1 P N\nai  EY1 AY1\nopenai  OW1 P AH0 N EY1 AY1\nopen(3)  AA1\n";
        let rows = collect_cmudict(Cursor::new(input)).unwrap();
        assert_eq!(rows["open"], ["OW1 P AH0 N", "OW1 P N"]);
        assert_eq!(rows["openai"], ["OW1 P AH0 N EY1 AY1"]);
    }

    #[test]
    fn build_uses_full_as_the_single_source_for_english_and_kanji() {
        let directory = tempfile::tempdir().unwrap();
        let full = directory.path().join("full.csv");
        let cmu = directory.path().join("cmudict.dict");
        fs::write(
            &full,
            "GitHub,0,0,1,GitHub,名詞,固有名詞,一般,*,*,*,ギットハブ\n雪崩,0,0,1,雪崩,名詞,普通名詞,一般,*,*,*,ナダレ\n",
        )
        .unwrap();
        fs::write(&cmu, "cat  K AE1 T\n").unwrap();

        build_hotword_reading_dictionaries(&full, &cmu, directory.path()).unwrap();

        assert_eq!(
            decode_zstd(&directory.path().join("sudachi-en-readings.tsv.zst")),
            "github\tギットハブ\n"
        );
        assert_eq!(
            decode_zstd(&directory.path().join("sudachi-kanji-readings.tsv.zst")),
            "雪崩\tナダレ\n"
        );
    }

    fn decode_zstd(path: &Path) -> String {
        let mut output = String::new();
        zstd::Decoder::new(fs::File::open(path).unwrap())
            .unwrap()
            .read_to_string(&mut output)
            .unwrap();
        output
    }
}
