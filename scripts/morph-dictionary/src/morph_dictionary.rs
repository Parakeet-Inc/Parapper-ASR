use anyhow::{Context, Result};

/// Encodes the `UniDic` fields consumed by Parapper's Morph boundary detector.
///
/// The output is a four-digit `[PP][S][F]` code. Fields that do not affect
/// boundary classification are encoded as zero.
///
/// # Errors
///
/// Returns an error when the feature has fewer than the six standard `UniDic`
/// columns.
pub fn compact_unidic_feature(feature: &str) -> Result<String> {
    let mut fields = feature.split(',');
    let pos1 = required_field(&mut fields, "pos1")?;
    let pos2 = required_field(&mut fields, "pos2")?;
    required_field(&mut fields, "pos3")?;
    required_field(&mut fields, "pos4")?;
    required_field(&mut fields, "cType")?;
    let cform = required_field(&mut fields, "cForm")?;

    let (pos1, pos2) = normalized_pos_fields(pos1, pos2);
    Ok(format!(
        "{:02}{}{}",
        pos1_code(pos1),
        pos2_code(pos1, pos2),
        cform_code(cform)
    ))
}

fn required_field<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    name: &'static str,
) -> Result<&'a str> {
    fields
        .next()
        .with_context(|| format!("UniDic feature is missing the {name} column"))
}

fn normalized_pos_fields<'a>(pos1: &'a str, pos2: &'a str) -> (&'a str, &'a str) {
    let pos1 = pos1.trim();
    let pos2 = pos2.trim();
    if let Some((pos1, embedded_pos2)) = pos1.split_once('-') {
        let pos1 = pos1.trim();
        let embedded_pos2 = embedded_pos2.trim();
        if !pos1.is_empty() && !embedded_pos2.is_empty() {
            return (pos1, embedded_pos2);
        }
    }
    (pos1, pos2)
}

fn pos1_code(pos1: &str) -> u8 {
    match pos1 {
        "補助記号" => 1,
        "助詞" => 2,
        "動詞" => 3,
        "形容詞" => 4,
        "助動詞" => 5,
        "名詞" => 6,
        "代名詞" => 7,
        "接尾辞" => 8,
        "形状詞" => 9,
        "感動詞" => 10,
        "接頭辞" => 11,
        "連体詞" => 12,
        _ => 0,
    }
}

fn pos2_code(pos1: &str, pos2: &str) -> u8 {
    match (pos1, pos2) {
        ("補助記号", "句点") => 1,
        ("補助記号", "読点") => 2,
        ("助詞", "終助詞") => 3,
        ("助詞", "格助詞") => 4,
        ("助詞", "係助詞") => 5,
        ("助詞", "副助詞") => 6,
        ("助詞", "準体助詞") => 7,
        ("助詞", "接続助詞") => 8,
        ("接尾辞", pos2) if pos2.starts_with("名詞的") || pos2.contains("名詞") => 9,
        _ => 0,
    }
}

fn cform_code(cform: &str) -> u8 {
    let cform = cform
        .trim()
        .split_once('-')
        .map_or(cform.trim(), |(base, _)| base.trim());
    match cform {
        "未然形" => 1,
        "連用形" => 2,
        "仮定形" => 3,
        "連体形" => 4,
        "終止形" => 5,
        "命令形" => 6,
        "意志推量形" => 7,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_unidic_feature_encodes_only_boundary_relevant_categories() {
        let cases = [
            ("動詞,一般,*,*,五段-カ行,終止形-一般,イク,行く", "0305"),
            ("助詞,終助詞,*,*,*,*,ネ,ね", "0230"),
            ("補助記号,読点,*,*,*,*,、,、", "0120"),
            ("接尾辞-名詞的,一般,*,*,*,*,エキ,駅", "0890"),
            ("名詞,固有名詞,地名,一般,*,*,トウキョウ,東京", "0600"),
        ];

        for (feature, expected) in cases {
            assert_eq!(
                compact_unidic_feature(feature).unwrap(),
                expected,
                "{feature}"
            );
        }
    }

    #[test]
    fn compact_unidic_feature_rejects_rows_without_the_cform_column() {
        assert!(compact_unidic_feature("動詞,一般").is_err());
    }
}
