use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use ort::{inputs, session::Session, value::TensorRef};
use tokenizers::Tokenizer;

use super::vad::init_onnx_runtime;

const MODEL_FILE: &str = "model_quant.onnx";
const VOCAB_FILE: &str = "vocab.txt";
const TOKENIZER_FILE: &str = "tokenizer.json";
const MAX_SEQUENCE_LEN: usize = 512;

#[derive(Clone, Copy)]
pub enum NamoTokenizerKind {
    Character,
    TokenizerJson,
}

pub struct NamoTurnDetectorEngine {
    session: Session,
    tokenizer: NamoTokenizer,
    pad_id: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NamoTurnDecision {
    pub is_end_of_turn: bool,
    pub confidence: f32,
}

impl NamoTurnDetectorEngine {
    pub fn new(model_dir: &Path, tokenizer_kind: NamoTokenizerKind) -> Result<Self> {
        init_onnx_runtime();

        let model_path = model_dir.join(MODEL_FILE);
        if !model_path.is_file() {
            return Err(anyhow!(
                "Namo turn detector model not found: {}",
                model_path.display()
            ));
        }

        let tokenizer = match tokenizer_kind {
            NamoTokenizerKind::Character => {
                let vocab = read_vocab(&model_dir.join(VOCAB_FILE))?;
                NamoTokenizer::Character(CharacterTokenizer {
                    cls_id: token_id(&vocab, "[CLS]", 101),
                    sep_id: token_id(&vocab, "[SEP]", 102),
                    unk_id: token_id(&vocab, "[UNK]", 100),
                    vocab,
                })
            }
            NamoTokenizerKind::TokenizerJson => {
                let tokenizer_path = model_dir.join(TOKENIZER_FILE);
                let tokenizer = context_display(
                    Tokenizer::from_file(&tokenizer_path),
                    format!("Failed to load Namo tokenizer {}", tokenizer_path.display()),
                )?;
                NamoTokenizer::TokenizerJson(Box::new(tokenizer))
            }
        };
        let pad_id = tokenizer.compute_pad_id();
        let builder = context_display(Session::builder(), "Failed to create Namo session builder")?;
        let mut builder = context_display(
            builder.with_intra_threads(1),
            "Failed to configure Namo session",
        )?;
        let session = context_display(
            builder.commit_from_file(&model_path),
            format!("Failed to load Namo model {}", model_path.display()),
        )?;

        Ok(Self {
            session,
            tokenizer,
            pad_id,
        })
    }

    pub fn decide(&mut self, text: &str, max_context_tokens: u32) -> Result<NamoTurnDecision> {
        let mut token_ids = self.tokenizer.tokenize(text)?;
        trim_to_context(&mut token_ids, max_context_tokens);
        token_ids.truncate(MAX_SEQUENCE_LEN);

        let mut attention_mask = vec![1_i64; token_ids.len()];
        while token_ids.len() < MAX_SEQUENCE_LEN {
            token_ids.push(self.pad_id);
            attention_mask.push(0);
        }

        let input_ids =
            TensorRef::from_array_view(([1_usize, MAX_SEQUENCE_LEN], token_ids.as_slice()))?;
        let attention_mask =
            TensorRef::from_array_view(([1_usize, MAX_SEQUENCE_LEN], attention_mask.as_slice()))?;
        let outputs = self.session.run(inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        ])?;

        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let [not_end, end, ..] = logits else {
            return Err(anyhow!("Namo output did not contain two logits"));
        };
        let end_probability = softmax_second(*not_end, *end);
        Ok(NamoTurnDecision {
            is_end_of_turn: end_probability >= 0.5,
            confidence: end_probability,
        })
    }
}

fn context_display<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    context: impl std::fmt::Display,
) -> Result<T> {
    result.map_err(|err| anyhow!("{context}: {err}"))
}

struct CharacterTokenizer {
    vocab: HashMap<String, i64>,
    cls_id: i64,
    sep_id: i64,
    unk_id: i64,
}

enum NamoTokenizer {
    Character(CharacterTokenizer),
    TokenizerJson(Box<Tokenizer>),
}

impl NamoTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        match self {
            Self::Character(tokenizer) => Ok(tokenizer.tokenize(text)),
            Self::TokenizerJson(tokenizer) => tokenizer
                .encode(text, true)
                .map(|encoding| encoding.get_ids().iter().map(|id| i64::from(*id)).collect())
                .map_err(|err| anyhow!("Failed to tokenize text for Namo: {err}")),
        }
    }

    fn compute_pad_id(&self) -> i64 {
        match self {
            Self::Character(tokenizer) => token_id(&tokenizer.vocab, "[PAD]", 0),
            Self::TokenizerJson(tokenizer) => tokenizer
                .get_vocab(true)
                .get("[PAD]")
                .copied()
                .map_or(0, i64::from),
        }
    }
}

impl CharacterTokenizer {
    fn tokenize(&self, text: &str) -> Vec<i64> {
        let mut token_ids = Vec::new();
        token_ids.push(self.cls_id);
        for character in text.chars().filter(|character| !character.is_whitespace()) {
            let token = character.to_string();
            token_ids.push(*self.vocab.get(&token).unwrap_or(&self.unk_id));
        }
        token_ids.push(self.sep_id);
        token_ids
    }
}

fn trim_to_context(token_ids: &mut Vec<i64>, max_context_tokens: u32) {
    if token_ids.len() <= 2 {
        return;
    }
    let leading_special_tokens = 1;
    let trailing_special_tokens = 1;
    let payload_start = leading_special_tokens;
    let payload_end = token_ids.len().saturating_sub(trailing_special_tokens);
    let payload_len = payload_end.saturating_sub(payload_start);
    if payload_len == 0 {
        return;
    }
    let max_payload_len =
        MAX_SEQUENCE_LEN.saturating_sub(leading_special_tokens + trailing_special_tokens);
    let effective_cap = if max_context_tokens > 0 {
        max_payload_len.min(usize::try_from(max_context_tokens).unwrap_or(MAX_SEQUENCE_LEN))
    } else {
        max_payload_len
    };
    if payload_len > effective_cap {
        token_ids.drain(payload_start..payload_end - effective_cap);
    }
}

fn read_vocab(path: &Path) -> Result<HashMap<String, i64>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read Namo vocab: {}", path.display()))?;
    Ok(content
        .lines()
        .enumerate()
        .map(|(index, token)| {
            (
                token.to_string(),
                i64::try_from(index).expect("vocab index fits in i64"),
            )
        })
        .collect())
}

fn token_id(vocab: &HashMap<String, i64>, token: &str, default_id: i64) -> i64 {
    vocab.get(token).copied().unwrap_or(default_id)
}

fn softmax_second(first: f32, second: f32) -> f32 {
    let max = first.max(second);
    let first_exp = (first - max).exp();
    let second_exp = (second - max).exp();
    second_exp / (first_exp + second_exp)
}

#[cfg(test)]
mod tests {
    use super::softmax_second;

    #[test]
    fn softmax_second_returns_probability() {
        let probability = softmax_second(0.0, 1.0);

        assert!((0.73..=0.74).contains(&probability));
    }
}
