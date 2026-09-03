const MAX_LETTER_NAME_LENGTH: usize = 6;

pub(super) fn suggest(word: &str) -> Vec<String> {
    let word = word.to_ascii_lowercase();
    if word.len() < 2 || !word.bytes().all(|byte| byte.is_ascii_lowercase()) {
        return Vec::new();
    }

    let mut result = Vec::new();
    if let Some(reading) = english_chunks_to_katakana(&word) {
        result.push(reading);
    }
    if word.len() <= MAX_LETTER_NAME_LENGTH {
        let reading = letter_names(&word);
        if !result.contains(&reading) {
            result.push(reading);
        }
    }
    result
}

fn english_chunks_to_katakana(word: &str) -> Option<String> {
    romaji_to_katakana(&english_chunks_to_romaji(word))
}

fn english_chunks_to_romaji(word: &str) -> String {
    let bytes = word.as_bytes();
    let mut output = String::with_capacity(word.len() + 4);
    let mut index = 0;
    while index < bytes.len() {
        let rest = &word[index..];
        let (replacement, consumed) = if rest.starts_with("ssion") {
            ("shon", 5)
        } else if rest.starts_with("tion") {
            ("shon", 4)
        } else if rest.starts_with("sion") {
            ("jon", 4)
        } else if rest.starts_with("tch") {
            ("ch", 3)
        } else if rest.starts_with("ph") {
            ("f", 2)
        } else if rest.starts_with("sh") {
            ("sh", 2)
        } else if rest.starts_with("ch") {
            ("ch", 2)
        } else if rest.starts_with("th") {
            ("s", 2)
        } else if rest.starts_with("ck") {
            ("kk", 2)
        } else if rest.starts_with("qu") {
            ("kw", 2)
        } else if rest.starts_with("ee") || rest.starts_with("ea") {
            ("i~", 2)
        } else if rest.starts_with("oo") {
            ("u~", 2)
        } else if rest.starts_with("ai") || rest.starts_with("ay") {
            ("ei", 2)
        } else if rest.starts_with("oa") {
            ("o~", 2)
        } else if rest.starts_with("oi") || rest.starts_with("oy") {
            ("oi", 2)
        } else if rest.starts_with("ou") || rest.starts_with("ow") {
            ("au", 2)
        } else {
            let current = bytes[index];
            if current == b'e'
                && index + 1 == bytes.len()
                && index > 1
                && !is_vowel(bytes[index - 1])
            {
                index += 1;
                continue;
            }
            if is_vowel(current)
                && index + 2 == bytes.len() - 1
                && !is_vowel(bytes[index + 1])
                && bytes[index + 2] == b'e'
            {
                output.push_str(match current {
                    b'a' => "ei",
                    b'e' => "i~",
                    b'i' | b'y' => "ai",
                    b'o' => "ou",
                    b'u' => "yu~",
                    _ => "",
                });
                index += 1;
                continue;
            }
            let next = bytes.get(index + 1).copied();
            let replacement = match current {
                b'c' if next.is_some_and(|byte| matches!(byte, b'e' | b'i' | b'y')) => "s",
                b'c' => "k",
                b'g' if next.is_some_and(|byte| matches!(byte, b'e' | b'i' | b'y')) => "j",
                b'g' => "g",
                b'x' => "ks",
                b'y' if index + 1 == bytes.len() => "i",
                b's' if index > 0 && is_vowel(bytes[index - 1]) && next.is_some_and(is_vowel) => {
                    "z"
                }
                _ => {
                    output.push(char::from(current));
                    index += 1;
                    continue;
                }
            };
            (replacement, 1)
        };
        output.push_str(replacement);
        index += consumed;
    }
    output
}

fn is_vowel(byte: u8) -> bool {
    matches!(byte, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

fn romaji_to_katakana(romaji: &str) -> Option<String> {
    let bytes = romaji.as_bytes();
    let mut output = String::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            output.push('ー');
            index += 1;
            continue;
        }
        if index + 1 < bytes.len()
            && bytes[index] == bytes[index + 1]
            && !is_vowel(bytes[index])
            && bytes[index] != b'n'
        {
            output.push('ッ');
            index += 1;
            continue;
        }
        if bytes[index] == b'n'
            && (index + 1 == bytes.len()
                || (!is_vowel(bytes[index + 1]) && bytes[index + 1] != b'y'))
        {
            output.push('ン');
            index += 1;
            continue;
        }
        if let Some((consumed, kana)) = extended_mora(&romaji[index..]) {
            output.push_str(kana);
            index += consumed;
            continue;
        }
        if let Some(kana) = vowel_kana(bytes[index]) {
            output.push_str(kana);
            index += 1;
            continue;
        }
        if let Some(vowel) = bytes.get(index + 1).and_then(|byte| vowel_kana(*byte)) {
            output.push_str(basic_mora(bytes[index], vowel)?);
            index += 2;
            continue;
        }
        output.push_str(coda_kana(bytes[index])?);
        index += 1;
    }
    (!output.is_empty()).then_some(output)
}

fn extended_mora(rest: &str) -> Option<(usize, &'static str)> {
    for (pattern, kana) in [
        ("kya", "キャ"),
        ("kyu", "キュ"),
        ("kyo", "キョ"),
        ("gya", "ギャ"),
        ("gyu", "ギュ"),
        ("gyo", "ギョ"),
        ("sha", "シャ"),
        ("shu", "シュ"),
        ("sho", "ショ"),
        ("cha", "チャ"),
        ("chu", "チュ"),
        ("cho", "チョ"),
        ("nya", "ニャ"),
        ("nyu", "ニュ"),
        ("nyo", "ニョ"),
        ("hya", "ヒャ"),
        ("hyu", "ヒュ"),
        ("hyo", "ヒョ"),
        ("bya", "ビャ"),
        ("byu", "ビュ"),
        ("byo", "ビョ"),
        ("pya", "ピャ"),
        ("pyu", "ピュ"),
        ("pyo", "ピョ"),
        ("mya", "ミャ"),
        ("myu", "ミュ"),
        ("myo", "ミョ"),
        ("rya", "リャ"),
        ("ryu", "リュ"),
        ("ryo", "リョ"),
        ("shi", "シ"),
        ("she", "シェ"),
        ("chi", "チ"),
        ("che", "チェ"),
        ("tsu", "ツ"),
        ("ji", "ジ"),
        ("ja", "ジャ"),
        ("ju", "ジュ"),
        ("jo", "ジョ"),
        ("fa", "ファ"),
        ("fi", "フィ"),
        ("fe", "フェ"),
        ("fo", "フォ"),
        ("va", "ヴァ"),
        ("vi", "ヴィ"),
        ("vu", "ヴ"),
        ("ve", "ヴェ"),
        ("vo", "ヴォ"),
    ] {
        if rest.starts_with(pattern) {
            return Some((pattern.len(), kana));
        }
    }
    None
}

fn vowel_kana(vowel: u8) -> Option<&'static str> {
    Some(match vowel {
        b'a' => "ア",
        b'i' | b'y' => "イ",
        b'u' => "ウ",
        b'e' => "エ",
        b'o' => "オ",
        _ => return None,
    })
}

fn basic_mora(consonant: u8, vowel: &str) -> Option<&'static str> {
    let row = match consonant {
        b'k' => ["カ", "キ", "ク", "ケ", "コ"],
        b'g' => ["ガ", "ギ", "グ", "ゲ", "ゴ"],
        b's' => ["サ", "シ", "ス", "セ", "ソ"],
        b'z' => ["ザ", "ジ", "ズ", "ゼ", "ゾ"],
        b't' => ["タ", "ティ", "トゥ", "テ", "ト"],
        b'd' => ["ダ", "ディ", "ドゥ", "デ", "ド"],
        b'n' => ["ナ", "ニ", "ヌ", "ネ", "ノ"],
        b'h' => ["ハ", "ヒ", "フ", "ヘ", "ホ"],
        b'b' => ["バ", "ビ", "ブ", "ベ", "ボ"],
        b'p' => ["パ", "ピ", "プ", "ペ", "ポ"],
        b'm' => ["マ", "ミ", "ム", "メ", "モ"],
        b'r' | b'l' => ["ラ", "リ", "ル", "レ", "ロ"],
        b'w' => ["ワ", "ウィ", "ウ", "ウェ", "ウォ"],
        b'f' => ["ファ", "フィ", "フ", "フェ", "フォ"],
        b'v' => ["ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"],
        b'j' => ["ジャ", "ジ", "ジュ", "ジェ", "ジョ"],
        b'y' => ["ヤ", "イ", "ユ", "イェ", "ヨ"],
        _ => return None,
    };
    Some(
        row[match vowel {
            "ア" => 0,
            "イ" => 1,
            "ウ" => 2,
            "エ" => 3,
            "オ" => 4,
            _ => return None,
        }],
    )
}

fn coda_kana(consonant: u8) -> Option<&'static str> {
    Some(match consonant {
        b'b' | b'v' => "ブ",
        b'c' | b'k' | b'q' => "ク",
        b'd' => "ド",
        b'f' | b'h' => "フ",
        b'g' => "グ",
        b'j' => "ジ",
        b'l' | b'r' => "ル",
        b'm' | b'n' => "ン",
        b'p' => "プ",
        b's' => "ス",
        b't' => "ト",
        b'w' => "ウ",
        b'x' => "クス",
        b'y' => "イ",
        b'z' => "ズ",
        _ => return None,
    })
}

fn letter_names(word: &str) -> String {
    word.bytes()
        .map(|letter| match letter {
            b'a' => "エー",
            b'b' => "ビー",
            b'c' => "シー",
            b'd' => "ディー",
            b'e' => "イー",
            b'f' => "エフ",
            b'g' => "ジー",
            b'h' => "エイチ",
            b'i' => "アイ",
            b'j' => "ジェイ",
            b'k' => "ケー",
            b'l' => "エル",
            b'm' => "エム",
            b'n' => "エヌ",
            b'o' => "オー",
            b'p' => "ピー",
            b'q' => "キュー",
            b'r' => "アール",
            b's' => "エス",
            b't' => "ティー",
            b'u' => "ユー",
            b'v' => "ブイ",
            b'w' => "ダブリュー",
            b'x' => "エックス",
            b'y' => "ワイ",
            b'z' => "ゼット",
            _ => "",
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_english_chunks_produce_editable_katakana_candidates() {
        for (word, expected) in [
            ("fiction", "フィクション"),
            ("phase", "フェイズ"),
            ("check", "チェック"),
            ("vision", "ヴィジョン"),
            ("parakeet", "パラキート"),
            ("abaca", "アバカ"),
        ] {
            assert_eq!(suggest(word).first().map(String::as_str), Some(expected));
        }
    }

    #[test]
    fn fallback_is_case_insensitive_but_does_not_split_unspaced_compounds() {
        assert_eq!(suggest("PARAKEET"), suggest("parakeet"));
        assert!(suggest("open-ai").is_empty());
        assert!(suggest("open_ai").is_empty());
    }

    #[test]
    fn short_unknown_spellings_also_offer_a_letter_name_candidate() {
        assert!(suggest("xyz").contains(&"エックスワイゼット".to_owned()));
    }
}
