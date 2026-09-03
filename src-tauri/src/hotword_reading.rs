use std::{io::Read, sync::OnceLock};

use unicode_normalization::UnicodeNormalization;

mod spelling;

const SUDACHI_DICTIONARY: &[u8] =
    include_bytes!("../resources/hotword-reading/sudachi-en-readings.tsv.zst");
const SUDACHI_KANJI_DICTIONARY: &[u8] =
    include_bytes!("../resources/hotword-reading/sudachi-kanji-readings.tsv.zst");
const CMU_DICTIONARY: &[u8] =
    include_bytes!("../resources/hotword-reading/cmudict-arpabet.tsv.zst");
const MAX_SUGGESTIONS: usize = 3;

static DICTIONARIES: OnceLock<Result<ReadingDictionaries, String>> = OnceLock::new();

#[derive(Debug)]
struct ReadingDictionaries {
    kanji: ReadingDictionary,
    sudachi: ReadingDictionary,
    cmu: ReadingDictionary,
}

#[derive(Debug)]
struct ReadingDictionary {
    text: String,
    line_offsets: Vec<u32>,
    value_separator: char,
}

impl ReadingDictionaries {
    fn embedded() -> Result<Self, String> {
        Ok(Self {
            kanji: ReadingDictionary::from_zstd(SUDACHI_KANJI_DICTIONARY, ',')?,
            sudachi: ReadingDictionary::from_zstd(SUDACHI_DICTIONARY, ',')?,
            cmu: ReadingDictionary::from_zstd(CMU_DICTIONARY, '|')?,
        })
    }

    fn suggest(&self, surface: &str) -> Vec<String> {
        let exact_key = normalize_exact_key(surface);
        if let Some(readings) = self.kanji.get(&exact_key) {
            return readings;
        }
        let key = normalize_english_key(surface);
        if key.is_empty() {
            return Vec::new();
        }
        if let Some(readings) = self.sudachi.get(&key) {
            return readings;
        }
        if key.contains(' ') {
            return self.suggest_space_delimited(&key).into_iter().collect();
        }
        let cmu = self.cmu_readings(&key);
        if cmu.is_empty() {
            spelling::suggest(&key)
        } else {
            cmu
        }
    }

    fn suggest_space_delimited(&self, key: &str) -> Option<String> {
        key.split(' ')
            .map(|part| {
                self.sudachi
                    .get(part)
                    .and_then(|values| values.into_iter().next())
                    .or_else(|| self.cmu_readings(part).into_iter().next())
                    .or_else(|| spelling::suggest(part).into_iter().next())
            })
            .collect::<Option<String>>()
    }

    fn cmu_readings(&self, key: &str) -> Vec<String> {
        let mut result = Vec::new();
        for pronunciation in self.cmu.get(key).into_iter().flatten() {
            if let Some(reading) = arpabet_to_katakana(&pronunciation)
                && !result.contains(&reading)
            {
                result.push(reading);
            }
            if result.len() == MAX_SUGGESTIONS {
                break;
            }
        }
        result
    }
}

impl ReadingDictionary {
    fn from_zstd(bytes: &[u8], value_separator: char) -> Result<Self, String> {
        let mut decoder = zstd::Decoder::new(bytes).map_err(|error| error.to_string())?;
        let mut text = String::new();
        decoder
            .read_to_string(&mut text)
            .map_err(|error| error.to_string())?;
        Self::from_text(text, value_separator)
    }

    fn from_text(text: String, value_separator: char) -> Result<Self, String> {
        let mut line_offsets = Vec::new();
        let mut byte_offset = 0usize;
        let mut previous_key: Option<&str> = None;
        for line in text.split_terminator('\n') {
            let (key, values) = line
                .split_once('\t')
                .ok_or_else(|| "invalid hotword reading dictionary row".to_owned())?;
            if key.is_empty() || values.is_empty() || previous_key.is_some_and(|value| value >= key)
            {
                return Err("hotword reading dictionary keys must be strictly sorted".into());
            }
            line_offsets.push(
                u32::try_from(byte_offset)
                    .map_err(|_| "hotword reading dictionary exceeds 4 GiB".to_owned())?,
            );
            previous_key = Some(key);
            byte_offset += line.len() + 1;
        }
        Ok(Self {
            text,
            line_offsets,
            value_separator,
        })
    }

    fn get(&self, key: &str) -> Option<Vec<String>> {
        let index = self
            .line_offsets
            .binary_search_by(|offset| self.key_at(*offset).cmp(key))
            .ok()?;
        let line = self.line_at(self.line_offsets[index]);
        let (_, values) = line.split_once('\t')?;
        Some(
            values
                .split(self.value_separator)
                .map(str::to_owned)
                .collect(),
        )
    }

    fn key_at(&self, offset: u32) -> &str {
        self.line_at(offset)
            .split_once('\t')
            .map_or("", |(key, _)| key)
    }

    fn line_at(&self, offset: u32) -> &str {
        let suffix = &self.text[offset as usize..];
        suffix.split_once('\n').map_or(suffix, |(line, _)| line)
    }
}

pub fn suggest_hotword_readings(surface: &str) -> Result<Vec<String>, String> {
    DICTIONARIES
        .get_or_init(ReadingDictionaries::embedded)
        .as_ref()
        .map(|dictionaries| {
            dictionaries
                .suggest(surface)
                .into_iter()
                .map(|reading| katakana_to_hiragana(&reading))
                .collect()
        })
        .map_err(Clone::clone)
}

fn katakana_to_hiragana(value: &str) -> String {
    value
        .nfkc()
        .map(|character| {
            let code = character as u32;
            if (0x30a1..=0x30f6).contains(&code) {
                char::from_u32(code - 0x60).unwrap_or(character)
            } else {
                character
            }
        })
        .collect()
}

fn normalize_english_key(value: &str) -> String {
    let mut result = String::new();
    let mut previous_space = false;
    for character in value.nfkc().flat_map(char::to_lowercase) {
        if character.is_whitespace() {
            if !result.is_empty() && !previous_space {
                result.push(' ');
            }
            previous_space = true;
        } else {
            previous_space = false;
            result.push(character);
        }
    }
    result.trim_end().to_owned()
}

fn normalize_exact_key(value: &str) -> String {
    value.nfkc().collect::<String>().trim().to_owned()
}

fn arpabet_to_katakana(pronunciation: &str) -> Option<String> {
    let phones = pronunciation
        .split_whitespace()
        .map(|phone| phone.trim_end_matches(|character: char| character.is_ascii_digit()))
        .collect::<Vec<_>>();
    if phones.is_empty() {
        return None;
    }
    let mut output = String::new();
    let mut pending = Vec::new();
    let mut last_vowel = None;
    for phone in phones {
        if let Some(vowel) = vowel_kana(phone, pending.last().copied()) {
            flush_consonant_cluster(&mut output, &pending, Some(vowel), false);
            pending.clear();
            last_vowel = Some(phone);
        } else if is_consonant(phone) {
            pending.push(phone);
        } else {
            return None;
        }
    }
    flush_consonant_cluster(
        &mut output,
        &pending,
        None,
        last_vowel.is_some_and(is_short_vowel),
    );
    (!output.is_empty()).then_some(output)
}

fn vowel_kana(phone: &str, preceding: Option<&str>) -> Option<&'static str> {
    Some(match (phone, preceding) {
        ("AE", Some("K" | "G")) => "ャ",
        ("AA" | "AE" | "AH", _) => "ア",
        ("AO", _) => "オ",
        ("AW", _) => "アウ",
        ("AY", _) => "アイ",
        ("EH", _) => "エ",
        ("ER", _) => "アー",
        ("EY", _) => "エイ",
        ("IH", _) => "イ",
        ("IY", _) => "イー",
        ("OW", _) => "オウ",
        ("OY", _) => "オイ",
        ("UH", _) => "ウ",
        ("UW", _) => "ウー",
        _ => return None,
    })
}

fn is_consonant(phone: &str) -> bool {
    matches!(
        phone,
        "B" | "CH"
            | "D"
            | "DH"
            | "F"
            | "G"
            | "HH"
            | "JH"
            | "K"
            | "L"
            | "M"
            | "N"
            | "NG"
            | "P"
            | "R"
            | "S"
            | "SH"
            | "T"
            | "TH"
            | "V"
            | "W"
            | "Y"
            | "Z"
            | "ZH"
    )
}

fn is_short_vowel(phone: &str) -> bool {
    matches!(phone, "AE" | "EH" | "IH" | "AH" | "UH")
}

fn flush_consonant_cluster(
    output: &mut String,
    consonants: &[&str],
    vowel: Option<&str>,
    geminate_final_stop: bool,
) {
    if consonants.is_empty() {
        if let Some(vowel) = vowel {
            output.push_str(vowel);
        }
        return;
    }
    for consonant in &consonants[..consonants.len().saturating_sub(1)] {
        output.push_str(final_consonant_kana(consonant));
    }
    let consonant = consonants[consonants.len() - 1];
    if let Some(vowel) = vowel {
        output.push_str(consonant_vowel_kana(consonant, vowel));
        output.push_str(vowel_suffix(vowel));
    } else {
        if geminate_final_stop && matches!(consonant, "K" | "P" | "T" | "CH") {
            output.push('ッ');
        }
        output.push_str(final_consonant_kana(consonant));
    }
}

fn vowel_suffix(vowel: &str) -> &'static str {
    match vowel {
        "アウ" | "オウ" => "ウ",
        "アイ" | "エイ" | "オイ" => "イ",
        "アー" | "イー" | "ウー" => "ー",
        _ => "",
    }
}

#[expect(
    clippy::match_same_arms,
    reason = "phoneme rows stay auditable when each articulation is listed together"
)]
fn consonant_vowel_kana(consonant: &str, vowel: &str) -> &'static str {
    match (consonant, vowel) {
        ("K", "ャ") => "キャ",
        ("G", "ャ") => "ギャ",
        ("K", "ア") => "カ",
        ("K", "イ" | "イー") => "キ",
        ("K", "ウ" | "ウー") => "ク",
        ("K", "エ" | "エイ") => "ケ",
        ("K", _) => "コ",
        ("G", "ア") => "ガ",
        ("G", "イ" | "イー") => "ギ",
        ("G", "ウ" | "ウー") => "グ",
        ("G", "エ" | "エイ") => "ゲ",
        ("G", _) => "ゴ",
        ("S" | "TH", "イ" | "イー") => "シ",
        ("S" | "TH", "ウ" | "ウー") => "ス",
        ("S" | "TH", "エ" | "エイ") => "セ",
        ("S" | "TH", "オ" | "オウ" | "オイ") => "ソ",
        ("S" | "TH", _) => "サ",
        ("SH", _) => "シ",
        ("CH", _) => "チ",
        ("JH" | "ZH", _) => "ジ",
        ("T", "イ" | "イー") => "ティ",
        ("T", "ウ" | "ウー") => "トゥ",
        ("T", "エ" | "エイ") => "テ",
        ("T", "オ" | "オウ" | "オイ") => "ト",
        ("T", _) => "タ",
        ("D" | "DH", "イ" | "イー") => "ディ",
        ("D" | "DH", "ウ" | "ウー") => "ドゥ",
        ("D" | "DH", "エ" | "エイ") => "デ",
        ("D" | "DH", "オ" | "オウ" | "オイ") => "ド",
        ("D" | "DH", _) => "ダ",
        ("N" | "NG", "イ" | "イー") => "ニ",
        ("N" | "NG", "ウ" | "ウー") => "ヌ",
        ("N" | "NG", "エ" | "エイ") => "ネ",
        ("N" | "NG", "オ" | "オウ" | "オイ") => "ノ",
        ("N" | "NG", _) => "ナ",
        ("HH", "イ" | "イー") => "ヒ",
        ("HH", "ウ" | "ウー") => "フ",
        ("HH", "エ" | "エイ") => "ヘ",
        ("HH", "オ" | "オウ" | "オイ") => "ホ",
        ("HH", _) => "ハ",
        ("M", "イ" | "イー") => "ミ",
        ("M", "ウ" | "ウー") => "ム",
        ("M", "エ" | "エイ") => "メ",
        ("M", "オ" | "オウ" | "オイ") => "モ",
        ("M", _) => "マ",
        ("Y", "ウ" | "ウー") => "ユ",
        ("Y", "オ" | "オウ" | "オイ") => "ヨ",
        ("Y", _) => "ヤ",
        ("R" | "L", "イ" | "イー") => "リ",
        ("R" | "L", "ウ" | "ウー") => "ル",
        ("R" | "L", "エ" | "エイ") => "レ",
        ("R" | "L", "オ" | "オウ" | "オイ") => "ロ",
        ("R" | "L", _) => "ラ",
        ("W", "イ" | "イー") => "ウィ",
        ("W", "エ" | "エイ") => "ウェ",
        ("W", "オ" | "オウ" | "オイ") => "ウォ",
        ("W", _) => "ワ",
        ("F", _) => "フ",
        ("V", _) => "ヴ",
        ("P", "イ" | "イー") => "ピ",
        ("P", "ウ" | "ウー") => "プ",
        ("P", "エ" | "エイ") => "ペ",
        ("P", "オ" | "オウ" | "オイ") => "ポ",
        ("P", _) => "パ",
        ("B", "イ" | "イー") => "ビ",
        ("B", "ウ" | "ウー") => "ブ",
        ("B", "エ" | "エイ") => "ベ",
        ("B", "オ" | "オウ" | "オイ") => "ボ",
        ("B", _) => "バ",
        ("Z", "イ" | "イー") => "ジ",
        ("Z", "ウ" | "ウー") => "ズ",
        ("Z", "エ" | "エイ") => "ゼ",
        ("Z", "オ" | "オウ" | "オイ") => "ゾ",
        ("Z", _) => "ザ",
        (_, _) => "",
    }
}

fn final_consonant_kana(consonant: &str) -> &'static str {
    match consonant {
        "M" | "N" | "NG" => "ン",
        "K" => "ク",
        "G" => "グ",
        "P" => "プ",
        "B" | "V" => "ブ",
        "T" => "ト",
        "CH" => "チ",
        "D" | "DH" => "ド",
        "S" | "SH" | "TH" => "ス",
        "Z" | "ZH" | "JH" => "ズ",
        "F" | "HH" => "フ",
        "L" | "R" => "ル",
        "W" => "ウ",
        "Y" => "イ",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    fn dictionaries() -> ReadingDictionaries {
        ReadingDictionaries {
            kanji: dictionary([("雪崩", "ナダレ")], ','),
            sudachi: dictionary(
                [
                    ("ai", "エーアイ"),
                    ("amazon", "アマゾン"),
                    ("github", "ギットハブ"),
                    ("open", "オープン"),
                ],
                ',',
            ),
            cmu: dictionary(
                [("cat", "K AE1 T"), ("parakeet", "P EH1 R AH0 K IY2 T")],
                '|',
            ),
        }
    }

    fn dictionary<const N: usize>(rows: [(&str, &str); N], separator: char) -> ReadingDictionary {
        let mut text = String::new();
        for (key, value) in rows {
            writeln!(text, "{key}\t{value}").unwrap();
        }
        ReadingDictionary::from_text(text, separator).unwrap()
    }

    #[test]
    fn nfkc_and_case_are_normalized_before_exact_lookup() {
        assert_eq!(dictionaries().suggest(" Ａｍａｚｏｎ "), ["アマゾン"]);
    }

    #[test]
    fn exact_kanji_surface_uses_the_sudachi_reading_baseline() {
        assert_eq!(dictionaries().suggest(" 雪崩 "), ["ナダレ"]);
    }

    #[test]
    fn full_sudachi_dictionary_precedes_the_unknown_word_phoneme_fallback() {
        assert_eq!(dictionaries().suggest("GitHub"), ["ギットハブ"]);
        assert_eq!(dictionaries().suggest("PARAKEET"), ["ペラキート"]);
    }

    #[test]
    fn alphabetic_unknown_word_uses_spelling_rules_after_both_dictionaries_miss() {
        assert_eq!(dictionaries().suggest("fiction"), ["フィクション"]);
    }

    #[test]
    fn only_explicit_space_delimited_words_are_composed() {
        assert_eq!(dictionaries().suggest("open ai"), ["オープンエーアイ"]);
        assert!(
            !dictionaries()
                .suggest("OpenAI")
                .contains(&"オープンエーアイ".to_owned())
        );
        assert!(dictionaries().suggest("open-ai").is_empty());
    }

    #[test]
    fn cmudict_pronunciation_is_used_only_after_sudachi_misses() {
        assert_eq!(dictionaries().suggest("cat"), ["キャット"]);
        assert_eq!(dictionaries().suggest("amazon"), ["アマゾン"]);
        assert_eq!(
            arpabet_to_katakana("P EH1 R AH0 K IY2 T"),
            Some("ペラキート".into())
        );
        assert_eq!(arpabet_to_katakana("M AE1 CH"), Some("マッチ".into()));
    }

    #[test]
    fn every_embedded_reading_dictionary_decodes_and_serves_its_own_entry() {
        let dictionaries = ReadingDictionaries::embedded().unwrap();

        assert_eq!(dictionaries.kanji.get("雪崩").unwrap(), ["ナダレ"]);
        assert!(dictionaries.sudachi.get("github").is_some());
        assert!(dictionaries.cmu.get("cat").is_some());
    }

    #[test]
    fn embedded_compact_dictionary_serves_a_common_loanword() {
        assert!(
            suggest_hotword_readings("ＡＭＡＺＯＮ")
                .unwrap()
                .contains(&"あまぞん".to_owned())
        );
        assert_eq!(suggest_hotword_readings("雪崩").unwrap(), ["なだれ"]);
        assert_eq!(suggest_hotword_readings("GitHub").unwrap(), ["ぎっとはぶ"]);
        assert_eq!(
            suggest_hotword_readings("Parakeet").unwrap(),
            ["ぺらきーと"]
        );
    }

    #[test]
    #[ignore = "diagnostic sweep over the embedded Sudachi Full English table"]
    fn evaluate_spelling_rules_against_sudachi_full_known_readings() {
        let dictionaries = ReadingDictionaries::embedded().unwrap();
        let mut all = SpellingEvaluation::default();
        let mut cmu_missing = SpellingEvaluation::default();
        let mut cmu_missing_wordlike = SpellingEvaluation::default();

        for offset in &dictionaries.sudachi.line_offsets {
            let line = dictionaries.sudachi.line_at(*offset);
            let (word, values) = line.split_once('\t').unwrap();
            if word.len() < 2 || !word.bytes().all(|byte| byte.is_ascii_lowercase()) {
                continue;
            }
            let references = values.split(',').collect::<Vec<_>>();
            let candidates = spelling::suggest(word);
            all.observe(&references, &candidates);
            if dictionaries.cmu.get(word).is_none() {
                cmu_missing.observe(&references, &candidates);
                if word.len() >= 4 && word.bytes().any(is_ascii_vowel) {
                    cmu_missing_wordlike.observe(&references, &candidates);
                }
            }
        }

        assert!(all.samples > 50_000);
        assert!(cmu_missing.samples > 10_000);
        println!("all={}", all.summary());
        println!("cmu_missing={}", cmu_missing.summary());
        println!("cmu_missing_wordlike={}", cmu_missing_wordlike.summary());
    }

    fn is_ascii_vowel(byte: u8) -> bool {
        matches!(byte, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
    }

    #[derive(Default)]
    struct SpellingEvaluation {
        samples: usize,
        top1_exact: usize,
        top_k_exact: usize,
        edits: usize,
        reference_chars: usize,
    }

    impl SpellingEvaluation {
        fn observe(&mut self, references: &[&str], candidates: &[String]) {
            self.samples += 1;
            self.top1_exact += usize::from(
                candidates
                    .first()
                    .is_some_and(|candidate| references.contains(&candidate.as_str())),
            );
            self.top_k_exact += usize::from(
                candidates
                    .iter()
                    .any(|candidate| references.contains(&candidate.as_str())),
            );
            let Some(candidate) = candidates.first() else {
                return;
            };
            let (distance, length) = references
                .iter()
                .map(|reference| {
                    (
                        character_edit_distance(reference, candidate),
                        reference.chars().count(),
                    )
                })
                .min_by_key(|(distance, _)| *distance)
                .unwrap();
            self.edits += distance;
            self.reference_chars += length;
        }

        fn summary(&self) -> String {
            format!(
                "samples={} top1_exact={:.4}% top_k_exact={:.4}% micro_cer={:.4}%",
                self.samples,
                percentage(self.top1_exact, self.samples),
                percentage(self.top_k_exact, self.samples),
                percentage(self.edits, self.reference_chars),
            )
        }
    }

    fn percentage(numerator: usize, denominator: usize) -> f64 {
        100.0 * f64::from(u32::try_from(numerator).unwrap())
            / f64::from(u32::try_from(denominator).unwrap())
    }

    fn character_edit_distance(left: &str, right: &str) -> usize {
        let right = right.chars().collect::<Vec<_>>();
        let mut previous = (0..=right.len()).collect::<Vec<_>>();
        let mut current = vec![0; right.len() + 1];
        for (left_index, left_character) in left.chars().enumerate() {
            current[0] = left_index + 1;
            for (right_index, right_character) in right.iter().enumerate() {
                current[right_index + 1] = (previous[right_index + 1] + 1)
                    .min(current[right_index] + 1)
                    .min(previous[right_index] + usize::from(left_character != *right_character));
            }
            std::mem::swap(&mut previous, &mut current);
        }
        previous[right.len()]
    }
}
