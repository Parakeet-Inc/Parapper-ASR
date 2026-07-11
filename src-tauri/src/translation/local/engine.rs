use std::{borrow::Cow, collections::HashMap, path::PathBuf};

use anyhow::{Context, Result, anyhow};
use ort::{
    memory::Allocator,
    session::{Session, SessionInputValue, SessionOutputs},
    value::{DynTensor, DynValue, Tensor, TensorElementType, ValueType},
};
use tokenizers::Tokenizer;

use crate::{
    config::{LocalTranslationModel, TranslationLanguage},
    model::onnx_runtime::init_onnx_runtime,
};

const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
const MAX_LFM2_INPUT_TOKENS: usize = 512;
const MAX_CAT_TRANSLATE_INPUT_TOKENS: usize = 4_096;
const MAX_NEW_TOKENS: usize = 256;

pub(super) struct LocalTranslationEngine {
    local_model: LocalTranslationModel,
    tokenizer: Tokenizer,
    session: Session,
    input_specs: Vec<OnnxInputSpec>,
}

#[derive(Clone)]
struct OnnxInputSpec {
    name: String,
    element_type: TensorElementType,
    shape: Vec<i64>,
}

struct DecoderState {
    past_values: HashMap<String, DynValue>,
    total_sequence_len: usize,
}

struct DecoderStep {
    next_token: u32,
    state: DecoderState,
}

impl LocalTranslationEngine {
    pub(super) fn load(model_dir: PathBuf, local_model: LocalTranslationModel) -> Result<Self> {
        Self::load_with_intra_threads(model_dir, local_model, 4)
    }

    fn load_with_intra_threads(
        model_dir: PathBuf,
        local_model: LocalTranslationModel,
        intra_threads: usize,
    ) -> Result<Self> {
        init_onnx_runtime()?;
        let tokenizer_path = model_dir.join(TOKENIZER_FILE_NAME);
        let model_path = model_dir.join(local_model.onnx_file_name());
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("{err}"))
            .with_context(|| {
                format!(
                    "Failed to load local translation tokenizer: {}",
                    tokenizer_path.display()
                )
            })?;
        let session = Session::builder()
            .map_err(|err| anyhow!("Failed to create local translation session builder: {err}"))?
            .with_intra_threads(intra_threads)
            .map_err(|err| anyhow!("Failed to configure local translation session: {err}"))?
            .commit_from_file(&model_path)
            .map_err(|err| {
                anyhow!(
                    "Failed to load local translation ONNX model {}: {err}",
                    model_path.display()
                )
            })
            .with_context(|| {
                format!(
                    "Failed to load local translation ONNX model: {}",
                    model_path.display()
                )
            })?;
        let input_specs = session
            .inputs()
            .iter()
            .map(|input| {
                let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
                    anyhow::bail!("Unsupported local translation input type: {}", input.name());
                };
                Ok(OnnxInputSpec {
                    name: input.name().to_string(),
                    element_type: *ty,
                    shape: shape.iter().copied().collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if !input_specs.iter().any(|input| input.name == "input_ids")
            || !input_specs
                .iter()
                .any(|input| input.name == "attention_mask")
        {
            anyhow::bail!("Local translation model is missing input_ids or attention_mask");
        }

        Ok(Self {
            local_model,
            tokenizer,
            session,
            input_specs,
        })
    }

    pub(super) fn translate(
        &mut self,
        source_lang: TranslationLanguage,
        target_lang: TranslationLanguage,
        text: &str,
    ) -> Result<String> {
        if source_lang == target_lang {
            return Ok(text.to_string());
        }

        let prompt = local_translation_prompt(self.local_model, source_lang, target_lang, text);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow!("{err}"))
            .context("Failed to tokenize local translation prompt")?;
        let prompt_token_ids = encoding.get_ids().to_vec();
        let max_input_tokens = local_translation_max_input_tokens(self.local_model);
        if prompt_token_ids.len() > max_input_tokens {
            anyhow::bail!(
                "Local translation prompt is too long: {} tokens > {}",
                prompt_token_ids.len(),
                max_input_tokens
            );
        }

        let mut generated_token_ids = Vec::new();
        let mut decoder_state = None;
        for _ in 0..MAX_NEW_TOKENS {
            let step = self.next_token(
                decoder_input_tokens(&prompt_token_ids, &generated_token_ids),
                decoder_state.as_ref(),
            )?;
            decoder_state = Some(step.state);
            if step.next_token == local_translation_eos_token_id(self.local_model) {
                break;
            }
            generated_token_ids.push(step.next_token);
        }

        let decoded = self
            .tokenizer
            .decode(&generated_token_ids, true)
            .map_err(|err| anyhow!("{err}"))
            .context("Failed to decode local translation output")?;
        Ok(trim_local_translation_output(self.local_model, &decoded))
    }

    fn next_token(
        &mut self,
        input_token_ids: &[u32],
        decoder_state: Option<&DecoderState>,
    ) -> Result<DecoderStep> {
        let total_sequence_len = decoder_total_sequence_len(input_token_ids.len(), decoder_state);
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(self.input_specs.len());
        for spec in &self.input_specs {
            let value =
                build_input_value(spec, input_token_ids, total_sequence_len, decoder_state)?;
            inputs.push((Cow::Owned(spec.name.clone()), value));
        }

        let mut outputs = self.session.run(inputs)?;
        if outputs.len() == 0 {
            anyhow::bail!("Local translation model did not return logits");
        }
        let next_token = {
            let logits = outputs.get("logits").unwrap_or(&outputs[0]);
            let (shape, values) = logits
                .try_extract_tensor::<f32>()
                .context("Failed to extract local translation logits")?;
            greedy_next_token(shape, values)?
        };
        let state =
            decoder_state_from_outputs(&mut outputs, &self.input_specs, total_sequence_len)?;
        Ok(DecoderStep { next_token, state })
    }
}

fn build_input_value<'a>(
    spec: &OnnxInputSpec,
    input_token_ids: &[u32],
    total_sequence_len: usize,
    decoder_state: Option<&'a DecoderState>,
) -> Result<SessionInputValue<'a>> {
    match spec.name.as_str() {
        "input_ids" => {
            let ids = input_token_ids
                .iter()
                .map(|token_id| i64::from(*token_id))
                .collect::<Vec<_>>();
            Ok(Tensor::from_array((vec![1_i64, input_token_ids.len() as i64], ids))?.into())
        }
        "attention_mask" => {
            let mask = vec![1_i64; total_sequence_len];
            Ok(Tensor::from_array((vec![1_i64, total_sequence_len as i64], mask))?.into())
        }
        name if name.starts_with("past_conv.") => {
            if let Some(state) = decoder_state {
                let value = state.past_values.get(name).ok_or_else(|| {
                    anyhow!("Local translation decoder cache is missing input {name}")
                })?;
                return Ok(value.into());
            }
            let shape = concrete_past_shape(&spec.shape, total_sequence_len, false);
            build_zero_tensor(spec.element_type, shape)
        }
        name if name.starts_with("past_key_values.") => {
            if let Some(state) = decoder_state {
                let value = state.past_values.get(name).ok_or_else(|| {
                    anyhow!("Local translation decoder cache is missing input {name}")
                })?;
                return Ok(value.into());
            }
            let shape = concrete_past_shape(&spec.shape, total_sequence_len, true);
            build_zero_tensor(spec.element_type, shape)
        }
        _ => anyhow::bail!("Unsupported local translation model input: {}", spec.name),
    }
}

fn decoder_input_tokens<'a>(
    prompt_token_ids: &'a [u32],
    generated_token_ids: &'a [u32],
) -> &'a [u32] {
    generated_token_ids
        .last()
        .map(std::slice::from_ref)
        .unwrap_or(prompt_token_ids)
}

fn decoder_total_sequence_len(
    input_token_count: usize,
    decoder_state: Option<&DecoderState>,
) -> usize {
    decoder_state
        .map(|state| state.total_sequence_len + input_token_count)
        .unwrap_or(input_token_count)
}

fn decoder_state_from_outputs(
    outputs: &mut SessionOutputs<'_>,
    input_specs: &[OnnxInputSpec],
    total_sequence_len: usize,
) -> Result<DecoderState> {
    let mut past_values = HashMap::new();
    for spec in input_specs {
        let Some(output_name) = present_output_name_for_past_input(&spec.name) else {
            continue;
        };
        let value = outputs.remove(&output_name).ok_or_else(|| {
            anyhow!(
                "Local translation model did not return cache output {} for input {}; outputs: {}",
                output_name,
                spec.name,
                output_names(outputs)
            )
        })?;
        past_values.insert(spec.name.clone(), value);
    }

    if past_values.is_empty() {
        anyhow::bail!("Local translation model did not return any decoder cache outputs");
    }

    Ok(DecoderState {
        past_values,
        total_sequence_len,
    })
}

fn present_output_name_for_past_input(input_name: &str) -> Option<String> {
    input_name
        .strip_prefix("past_conv.")
        .map(|rest| format!("present_conv.{rest}"))
        .or_else(|| {
            input_name
                .strip_prefix("past_key_values.")
                .map(|rest| format!("present.{rest}"))
        })
}

fn output_names(outputs: &SessionOutputs<'_>) -> String {
    outputs.keys().collect::<Vec<_>>().join(", ")
}

fn concrete_past_shape(template: &[i64], seq_len: usize, empty_sequence_dim: bool) -> Vec<i64> {
    template
        .iter()
        .enumerate()
        .map(|(index, dim)| {
            if *dim > 0 {
                *dim
            } else if index == 0 {
                1
            } else if empty_sequence_dim && index == 2 {
                0
            } else if !empty_sequence_dim && index == 2 {
                3
            } else {
                seq_len as i64
            }
        })
        .collect()
}

fn build_zero_tensor(
    element_type: TensorElementType,
    shape: Vec<i64>,
) -> Result<SessionInputValue<'static>> {
    match element_type {
        TensorElementType::Float16 | TensorElementType::Float32 => {
            Ok(DynTensor::new(&Allocator::default(), element_type, shape)?.into())
        }
        _ => anyhow::bail!(
            "Unsupported local translation past tensor type: {}",
            element_type
        ),
    }
}

fn local_translation_prompt(
    local_model: LocalTranslationModel,
    source_lang: TranslationLanguage,
    target_lang: TranslationLanguage,
    text: &str,
) -> String {
    match local_model {
        LocalTranslationModel::Lfm2Q4 => lfm2_translation_prompt(target_lang, text),
        LocalTranslationModel::CatTranslate0_8BQ4KQuant => {
            cat_translate_prompt(source_lang, target_lang, text)
        }
    }
}

fn local_translation_eos_token_id(local_model: LocalTranslationModel) -> u32 {
    match local_model {
        LocalTranslationModel::Lfm2Q4 => 7,
        LocalTranslationModel::CatTranslate0_8BQ4KQuant => 2,
    }
}

fn local_translation_max_input_tokens(local_model: LocalTranslationModel) -> usize {
    match local_model {
        LocalTranslationModel::Lfm2Q4 => MAX_LFM2_INPUT_TOKENS,
        LocalTranslationModel::CatTranslate0_8BQ4KQuant => MAX_CAT_TRANSLATE_INPUT_TOKENS,
    }
}

fn lfm2_translation_prompt(target_lang: TranslationLanguage, text: &str) -> String {
    let system_prompt = match target_lang {
        TranslationLanguage::Ja => "Translate to Japanese.",
        TranslationLanguage::En => "Translate to English.",
    };
    format!(
        "<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn cat_translate_prompt(
    source_lang: TranslationLanguage,
    target_lang: TranslationLanguage,
    text: &str,
) -> String {
    let user_prompt = format!(
        "Translate the following {} text into {}.\n\n{}",
        translation_language_name(source_lang),
        translation_language_name(target_lang),
        text
    );
    format!("<|user|>{user_prompt}</s><|assistant|>")
}

fn translation_language_name(language: TranslationLanguage) -> &'static str {
    match language {
        TranslationLanguage::Ja => "Japanese",
        TranslationLanguage::En => "English",
    }
}

fn trim_local_translation_output(local_model: LocalTranslationModel, text: &str) -> String {
    match local_model {
        LocalTranslationModel::Lfm2Q4 => trim_lfm2_translation_output(text),
        LocalTranslationModel::CatTranslate0_8BQ4KQuant => trim_cat_translate_output(text),
    }
}

fn trim_lfm2_translation_output(text: &str) -> String {
    text.split("<|im_end|>")
        .next()
        .unwrap_or(text)
        .trim()
        .to_string()
}

fn trim_cat_translate_output(text: &str) -> String {
    text.split("</s>")
        .next()
        .unwrap_or(text)
        .trim()
        .trim_start_matches("<|assistant|>")
        .trim()
        .to_string()
}

fn greedy_next_token(shape: &[i64], values: &[f32]) -> Result<u32> {
    let [_, sequence_len, vocab_size] = shape else {
        anyhow::bail!("Unexpected local translation logits shape: {shape:?}");
    };
    let sequence_len = usize::try_from(*sequence_len)
        .ok()
        .filter(|len| *len > 0)
        .ok_or_else(|| anyhow!("Unexpected local translation sequence length: {sequence_len}"))?;
    let vocab_size = usize::try_from(*vocab_size)
        .ok()
        .filter(|len| *len > 0)
        .ok_or_else(|| anyhow!("Unexpected local translation vocab size: {vocab_size}"))?;
    let start = (sequence_len - 1) * vocab_size;
    let logits = values
        .get(start..start + vocab_size)
        .ok_or_else(|| anyhow!("Local translation logits buffer is shorter than its shape"))?;
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(token_id, _)| token_id as u32)
        .ok_or_else(|| anyhow!("Local translation logits were empty"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env, fs,
        path::{Path, PathBuf},
        time::Instant,
    };

    struct JvsBenchUtterance {
        id: String,
        text: String,
        wav_path: PathBuf,
    }

    #[derive(Clone, Copy)]
    struct Lfm2BenchCase {
        id: &'static str,
        source_lang: TranslationLanguage,
        target_lang: TranslationLanguage,
        input: &'static str,
    }

    #[test]
    fn lfm2_translation_prompt_uses_required_system_prompt_and_chat_template() {
        assert_eq!(
            lfm2_translation_prompt(TranslationLanguage::Ja, "hello"),
            "<|startoftext|><|im_start|>system\nTranslate to Japanese.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            lfm2_translation_prompt(TranslationLanguage::En, "こんにちは"),
            "<|startoftext|><|im_start|>system\nTranslate to English.<|im_end|>\n<|im_start|>user\nこんにちは<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn cat_translate_prompt_uses_model_chat_template_and_eos_token() {
        assert_eq!(
            local_translation_prompt(
                LocalTranslationModel::CatTranslate0_8BQ4KQuant,
                TranslationLanguage::Ja,
                TranslationLanguage::En,
                "こんにちは"
            ),
            "<|user|>Translate the following Japanese text into English.\n\nこんにちは</s><|assistant|>"
        );
        assert_eq!(
            local_translation_eos_token_id(LocalTranslationModel::CatTranslate0_8BQ4KQuant),
            2
        );
        assert_eq!(
            trim_local_translation_output(
                LocalTranslationModel::CatTranslate0_8BQ4KQuant,
                "<|assistant|>hello</s>"
            ),
            "hello"
        );
    }

    #[test]
    fn greedy_next_token_reads_last_sequence_position() {
        let logits = vec![
            0.0, 9.0, 1.0, //
            4.0, 3.0, 8.0,
        ];

        assert_eq!(greedy_next_token(&[1, 2, 3], &logits).unwrap(), 2);
    }

    #[test]
    fn decoder_cache_input_names_match_lfm2_present_output_names() {
        assert_eq!(
            present_output_name_for_past_input("past_conv.0").as_deref(),
            Some("present_conv.0")
        );
        assert_eq!(
            present_output_name_for_past_input("past_key_values.2.key").as_deref(),
            Some("present.2.key")
        );
        assert_eq!(
            present_output_name_for_past_input("past_key_values.5.value").as_deref(),
            Some("present.5.value")
        );
        assert!(present_output_name_for_past_input("input_ids").is_none());
    }

    #[test]
    fn decoder_input_replays_prompt_only_before_cache_then_uses_last_generated_token() {
        let prompt = vec![10, 11, 12];
        let generated = Vec::new();
        assert_eq!(decoder_input_tokens(&prompt, &generated), prompt.as_slice());

        let generated = vec![20, 21];
        assert_eq!(decoder_input_tokens(&prompt, &generated), &[21]);

        let cached = DecoderState {
            past_values: HashMap::new(),
            total_sequence_len: prompt.len() + generated.len() - 1,
        };
        assert_eq!(
            decoder_total_sequence_len(
                decoder_input_tokens(&prompt, &generated).len(),
                Some(&cached)
            ),
            prompt.len() + generated.len()
        );
    }

    #[test]
    #[ignore = "diagnostic benchmark: requires local LFM2 ONNX models and JVS_ROOT"]
    fn bench_jvs_reference_translation_q4_under_one_and_four_ort_threads() {
        let model_dir = local_translation_model_dir_from_env_or_default();
        let utterances = jvs_bench_utterances();
        let repeats = env_usize("PARAPPER_LOCAL_TRANSLATION_BENCH_REPEATS", 1);

        println!(
            "local translation benchmark: model_dir={} utterances={} repeats={}",
            model_dir.display(),
            utterances.len(),
            repeats
        );
        for utterance in &utterances {
            println!(
                "jvs utterance: id={} wav={} text={}",
                utterance.id,
                utterance.wav_path.display(),
                utterance.text
            );
        }

        for intra_threads in [1_usize, 4_usize] {
            for local_model in [LocalTranslationModel::Lfm2Q4] {
                let mut engine = LocalTranslationEngine::load_with_intra_threads(
                    model_dir.clone(),
                    local_model,
                    intra_threads,
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to load local translation model {:?} with {} threads: {err:#}",
                        local_model, intra_threads
                    )
                });

                let warmup = &utterances[0];
                let warmup_start = Instant::now();
                let warmup_output = engine
                    .translate(
                        TranslationLanguage::Ja,
                        TranslationLanguage::En,
                        &warmup.text,
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "warmup failed for {:?} with {} threads: {err:#}",
                            local_model, intra_threads
                        )
                    });
                println!(
                    "warmup model={:?} threads={} elapsed_ms={} output_chars={} output={}",
                    local_model,
                    intra_threads,
                    warmup_start.elapsed().as_millis(),
                    warmup_output.chars().count(),
                    warmup_output
                );

                let mut elapsed_ms = Vec::new();
                let mut output_chars = 0_usize;
                for repeat in 0..repeats {
                    for utterance in &utterances {
                        let started = Instant::now();
                        let output = engine
                            .translate(
                                TranslationLanguage::Ja,
                                TranslationLanguage::En,
                                &utterance.text,
                            )
                            .unwrap_or_else(|err| {
                                panic!(
                                    "benchmark failed for {:?} with {} threads on {}: {err:#}",
                                    local_model, intra_threads, utterance.id
                                )
                            });
                        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                        output_chars += output.chars().count();
                        elapsed_ms.push(elapsed);
                        println!(
                            "sample model={:?} threads={} repeat={} id={} elapsed_ms={:.3} output_chars={} output={}",
                            local_model,
                            intra_threads,
                            repeat + 1,
                            utterance.id,
                            elapsed,
                            output.chars().count(),
                            output
                        );
                    }
                }

                let total_ms = elapsed_ms.iter().sum::<f64>();
                let mean_ms = total_ms / elapsed_ms.len() as f64;
                let median_ms = median(&mut elapsed_ms);
                println!(
                    "summary model={:?} threads={} samples={} total_ms={:.3} mean_ms={:.3} median_ms={:.3} output_chars={}",
                    local_model,
                    intra_threads,
                    elapsed_ms.len(),
                    total_ms,
                    mean_ms,
                    median_ms,
                    output_chars
                );
            }
        }
    }

    #[test]
    #[ignore = "diagnostic benchmark: requires local LFM2 ONNX model dir"]
    fn bench_lfm2_reference_sentences_reports_wall_time_and_target_language() {
        let model_dir = local_translation_model_dir_from_env_or_default();
        let repeats = env_usize("PARAPPER_LOCAL_TRANSLATION_BENCH_REPEATS", 2);
        let model_label = env::var("PARAPPER_LOCAL_TRANSLATION_BENCH_MODEL_LABEL")
            .unwrap_or_else(|_| "lfm2".to_string());
        let case_set = env::var("PARAPPER_LOCAL_TRANSLATION_BENCH_CASE_SET")
            .unwrap_or_else(|_| "base".to_string());
        let cases = lfm2_bench_cases(&case_set);

        println!(
            "lfm2 reference benchmark: label={} model_dir={} case_set={} cases={} repeats={}",
            model_label,
            model_dir.display(),
            case_set,
            cases.len(),
            repeats
        );

        for intra_threads in [1_usize, 4_usize] {
            let mut engine = LocalTranslationEngine::load_with_intra_threads(
                model_dir.clone(),
                LocalTranslationModel::Lfm2Q4,
                intra_threads,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "failed to load LFM2 model {} with {} threads: {err:#}",
                    model_dir.display(),
                    intra_threads
                )
            });

            let warmup = cases[0];
            let warmup_started = Instant::now();
            let warmup_output = engine
                .translate(warmup.source_lang, warmup.target_lang, warmup.input)
                .unwrap_or_else(|err| {
                    panic!(
                        "warmup failed for label={} threads={} case={}: {err:#}",
                        model_label, intra_threads, warmup.id
                    )
                });
            println!(
                "lfm2 warmup label={} threads={} case={} elapsed_ms={:.3} output={}",
                model_label,
                intra_threads,
                warmup.id,
                warmup_started.elapsed().as_secs_f64() * 1000.0,
                warmup_output
            );

            let mut elapsed_ms = Vec::new();
            for repeat in 0..repeats {
                for case in cases.iter().copied() {
                    let started = Instant::now();
                    let output = engine
                        .translate(case.source_lang, case.target_lang, case.input)
                        .unwrap_or_else(|err| {
                            panic!(
                                "benchmark failed for label={} threads={} case={}: {err:#}",
                                model_label, intra_threads, case.id
                            )
                        });
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    let target_language_ok =
                        output_matches_target_language(case.target_lang, &output);
                    println!(
                        "lfm2 sample label={} threads={} repeat={} case={} source={:?} target={:?} elapsed_ms={:.3} target_language_ok={} output={}",
                        model_label,
                        intra_threads,
                        repeat + 1,
                        case.id,
                        case.source_lang,
                        case.target_lang,
                        elapsed,
                        target_language_ok,
                        output
                    );
                    assert!(
                        !output.trim().is_empty(),
                        "LFM2 benchmark output should not be empty for {}",
                        case.id
                    );
                    elapsed_ms.push(elapsed);
                }
            }

            let total_ms = elapsed_ms.iter().sum::<f64>();
            let mean_ms = total_ms / elapsed_ms.len() as f64;
            let median_ms = median(&mut elapsed_ms);
            println!(
                "lfm2 summary label={} threads={} samples={} wall_total_ms={:.3} mean_ms={:.3} median_ms={:.3}",
                model_label,
                intra_threads,
                elapsed_ms.len(),
                total_ms,
                mean_ms,
                median_ms
            );
        }
    }

    fn lfm2_bench_cases(case_set: &str) -> Vec<Lfm2BenchCase> {
        let mut cases = vec![
            Lfm2BenchCase {
                id: "ja_to_en_greeting",
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
                input: "こんにちは。",
            },
            Lfm2BenchCase {
                id: "ja_to_en_settings",
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
                input: "設定画面を開いてください。",
            },
            Lfm2BenchCase {
                id: "en_to_ja_greeting",
                source_lang: TranslationLanguage::En,
                target_lang: TranslationLanguage::Ja,
                input: "Hello.",
            },
            Lfm2BenchCase {
                id: "en_to_ja_help",
                source_lang: TranslationLanguage::En,
                target_lang: TranslationLanguage::Ja,
                input: "Thank you for your help.",
            },
            Lfm2BenchCase {
                id: "en_to_ja_meeting",
                source_lang: TranslationLanguage::En,
                target_lang: TranslationLanguage::Ja,
                input: "I will join the meeting tomorrow.",
            },
            Lfm2BenchCase {
                id: "en_to_ja_settings",
                source_lang: TranslationLanguage::En,
                target_lang: TranslationLanguage::Ja,
                input: "Please open the settings screen.",
            },
        ];

        match case_set {
            "base" => cases,
            "extended" => {
                cases.extend([
                    Lfm2BenchCase {
                        id: "ja_to_en_cancel_download",
                        source_lang: TranslationLanguage::Ja,
                        target_lang: TranslationLanguage::En,
                        input: "ダウンロードをキャンセルしました。",
                    },
                    Lfm2BenchCase {
                        id: "ja_to_en_server_error",
                        source_lang: TranslationLanguage::Ja,
                        target_lang: TranslationLanguage::En,
                        input: "サーバーに接続できませんでした。",
                    },
                    Lfm2BenchCase {
                        id: "ja_to_en_save_before_close",
                        source_lang: TranslationLanguage::Ja,
                        target_lang: TranslationLanguage::En,
                        input: "閉じる前に変更を保存しますか？",
                    },
                    Lfm2BenchCase {
                        id: "ja_to_en_shortcut",
                        source_lang: TranslationLanguage::Ja,
                        target_lang: TranslationLanguage::En,
                        input: "Ctrl+Sでファイルを保存できます。",
                    },
                    Lfm2BenchCase {
                        id: "ja_to_en_cpu_usage",
                        source_lang: TranslationLanguage::Ja,
                        target_lang: TranslationLanguage::En,
                        input: "CPU使用率が85%を超えています。",
                    },
                    Lfm2BenchCase {
                        id: "ja_to_en_named_product",
                        source_lang: TranslationLanguage::Ja,
                        target_lang: TranslationLanguage::En,
                        input: "Parapperの翻訳モデルを更新しました。",
                    },
                    Lfm2BenchCase {
                        id: "ja_to_en_quoted_button",
                        source_lang: TranslationLanguage::Ja,
                        target_lang: TranslationLanguage::En,
                        input: "「開始」ボタンを押してください。",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_cancel_download",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "The download was canceled.",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_server_error",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "Could not connect to the server.",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_save_before_close",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "Do you want to save changes before closing?",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_shortcut",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "Press Ctrl+S to save the file.",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_cpu_usage",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "CPU usage is above 85%.",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_named_product",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "Parapper updated the translation model.",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_quoted_button",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "Click the \"Start\" button.",
                    },
                    Lfm2BenchCase {
                        id: "en_to_ja_error_code",
                        source_lang: TranslationLanguage::En,
                        target_lang: TranslationLanguage::Ja,
                        input: "Error code 0x80070005 means access is denied.",
                    },
                ]);
                cases
            }
            other => panic!("unknown LFM2 benchmark case set: {other}"),
        }
    }

    #[test]
    #[ignore = "diagnostic smoke: requires the ONNX Community LFM2 Q4 model"]
    fn smoke_onnx_community_lfm2_q4_model_translates_reference_sentence() {
        let variants = [(
            LocalTranslationModel::Lfm2Q4,
            local_translation_model_dir_from_env_or_default(),
        )];

        for (model, model_dir) in variants {
            let mut engine =
                LocalTranslationEngine::load_with_intra_threads(model_dir.clone(), model, 4)
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to load LFM2 {model:?} model from {}: {err:#}",
                            model_dir.display()
                        )
                    });
            let output = engine
                .translate(
                    TranslationLanguage::Ja,
                    TranslationLanguage::En,
                    "設定画面を開いてください。",
                )
                .unwrap_or_else(|err| panic!("LFM2 {model:?} smoke translation failed: {err:#}"));
            println!(
                "lfm2 variant={model:?} dir={} output={output}",
                model_dir.display()
            );
            assert!(
                output_matches_target_language(TranslationLanguage::En, &output),
                "LFM2 {model:?} should return English output, got {output:?}"
            );
        }
    }

    #[test]
    #[ignore = "diagnostic smoke: requires copied Cat local translation ONNX model"]
    fn smoke_cat_translate_loads_copied_model_and_translates_text() {
        let model_dir = cat_translation_model_dir_from_env_or_appdata();
        assert!(
            model_dir.join(TOKENIZER_FILE_NAME).is_file(),
            "Cat tokenizer should be copied before smoke test: {}",
            model_dir.display()
        );
        assert!(
            model_dir
                .join(LocalTranslationModel::CatTranslate0_8BQ4KQuant.onnx_file_name())
                .is_file(),
            "Cat ONNX model should be copied before smoke test: {}",
            model_dir.display()
        );

        let mut engine = LocalTranslationEngine::load(
            model_dir,
            LocalTranslationModel::CatTranslate0_8BQ4KQuant,
        )
        .expect("Cat local translation engine should load copied model files");
        let output = engine
            .translate(
                TranslationLanguage::Ja,
                TranslationLanguage::En,
                "こんにちは",
            )
            .expect("Cat local translation should translate a short Japanese input");

        println!("cat translate output={output}");
        assert!(
            !output.trim().is_empty(),
            "Cat local translation should return non-empty text"
        );
    }

    #[test]
    #[ignore = "diagnostic smoke: requires copied Cat local translation ONNX model"]
    fn smoke_cat_translate_direction_table_returns_target_language() {
        let model_dir = cat_translation_model_dir_from_env_or_appdata();
        assert!(
            model_dir.join(TOKENIZER_FILE_NAME).is_file(),
            "Cat tokenizer should be copied before smoke test: {}",
            model_dir.display()
        );
        assert!(
            model_dir
                .join(LocalTranslationModel::CatTranslate0_8BQ4KQuant.onnx_file_name())
                .is_file(),
            "Cat ONNX model should be copied before smoke test: {}",
            model_dir.display()
        );

        let mut engine = LocalTranslationEngine::load(
            model_dir,
            LocalTranslationModel::CatTranslate0_8BQ4KQuant,
        )
        .expect("Cat local translation engine should load copied model files");
        let cases = [
            (
                "ja_to_en_greeting",
                TranslationLanguage::Ja,
                TranslationLanguage::En,
                "こんにちは。",
            ),
            (
                "ja_to_en_settings",
                TranslationLanguage::Ja,
                TranslationLanguage::En,
                "設定画面を開いてください。",
            ),
            (
                "en_to_ja_greeting",
                TranslationLanguage::En,
                TranslationLanguage::Ja,
                "Hello.",
            ),
            (
                "en_to_ja_help",
                TranslationLanguage::En,
                TranslationLanguage::Ja,
                "Thank you for your help.",
            ),
            (
                "en_to_ja_meeting",
                TranslationLanguage::En,
                TranslationLanguage::Ja,
                "I will join the meeting tomorrow.",
            ),
            (
                "en_to_ja_settings",
                TranslationLanguage::En,
                TranslationLanguage::Ja,
                "Please open the settings screen.",
            ),
        ];

        let mut failures = Vec::new();
        for (id, source_lang, target_lang, input) in cases {
            let output = engine
                .translate(source_lang, target_lang, input)
                .unwrap_or_else(|err| {
                    panic!("Cat local translation failed for {id}: {err:#}");
                });
            println!(
                "cat translate case={id} source={source_lang:?} target={target_lang:?} input={input} output={output}"
            );

            if output.trim().is_empty() {
                failures.push(format!("{id}: empty output"));
                continue;
            }
            match target_lang {
                TranslationLanguage::Ja => {
                    if !contains_japanese_text(&output) {
                        failures.push(format!("{id}: expected Japanese output, got {output:?}"));
                    }
                }
                TranslationLanguage::En => {
                    if !contains_ascii_alphabet(&output) || contains_japanese_text(&output) {
                        failures.push(format!("{id}: expected English output, got {output:?}"));
                    }
                }
            }
        }

        assert!(
            failures.is_empty(),
            "Cat local translation should return target-language output: {failures:?}"
        );
    }

    fn contains_japanese_text(text: &str) -> bool {
        text.chars().any(|ch| {
            matches!(
                ch as u32,
                0x3040..=0x30ff | 0x3400..=0x9fff | 0xff00..=0xffef
            )
        })
    }

    fn contains_ascii_alphabet(text: &str) -> bool {
        text.chars().any(|ch| ch.is_ascii_alphabetic())
    }

    fn output_matches_target_language(target_lang: TranslationLanguage, output: &str) -> bool {
        match target_lang {
            TranslationLanguage::Ja => contains_japanese_text(output),
            TranslationLanguage::En => {
                contains_ascii_alphabet(output) && !contains_japanese_text(output)
            }
        }
    }

    fn local_translation_model_dir_from_env_or_default() -> PathBuf {
        env::var_os("PARAPPER_LOCAL_TRANSLATION_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let appdata = env::var_os("APPDATA")
                    .map(PathBuf::from)
                    .expect("APPDATA must be set or PARAPPER_LOCAL_TRANSLATION_MODEL_DIR supplied");
                appdata
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                    .join("lfm2-350m-enjp-mt-onnx-q4")
            })
    }

    fn cat_translation_model_dir_from_env_or_appdata() -> PathBuf {
        env::var_os("PARAPPER_CAT_TRANSLATION_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let appdata = env::var_os("APPDATA")
                    .map(PathBuf::from)
                    .expect("APPDATA must be set or PARAPPER_CAT_TRANSLATION_MODEL_DIR supplied");
                appdata
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                    .join("cat-translate-0.8b-onnx-q4-k-quant")
            })
    }

    fn jvs_bench_utterances() -> Vec<JvsBenchUtterance> {
        let jvs_root = env::var_os("JVS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| jvs_root_from_dotenv().expect("set JVS_ROOT or define it in .env"));
        let speaker = env::var("PARAPPER_LOCAL_TRANSLATION_BENCH_JVS_SPEAKER")
            .unwrap_or_else(|_| "jvs001".to_string());
        let subset = env::var("PARAPPER_LOCAL_TRANSLATION_BENCH_JVS_SUBSET")
            .unwrap_or_else(|_| "nonpara30".to_string());
        let transcript_path = jvs_root
            .join(&speaker)
            .join(&subset)
            .join("transcripts_utf8.txt");
        let wav_dir = jvs_root.join(&speaker).join(&subset).join("wav24kHz16bit");
        let max_utterances = env_usize("PARAPPER_LOCAL_TRANSLATION_BENCH_MAX_UTTERANCES", 3);
        let transcript = fs::read_to_string(&transcript_path).unwrap_or_else(|err| {
            panic!(
                "failed to read JVS transcript {}: {err}",
                transcript_path.display()
            )
        });

        let utterances = transcript
            .lines()
            .filter_map(|line| line.split_once(':'))
            .map(|(id, text)| {
                let wav_path = wav_dir.join(format!("{id}.wav"));
                assert!(
                    wav_path.is_file(),
                    "JVS transcript id has no matching wav: {}",
                    wav_path.display()
                );
                JvsBenchUtterance {
                    id: id.to_string(),
                    text: text.to_string(),
                    wav_path,
                }
            })
            .take(max_utterances)
            .collect::<Vec<_>>();
        assert!(
            !utterances.is_empty(),
            "JVS transcript did not contain any utterances: {}",
            transcript_path.display()
        );
        utterances
    }

    fn jvs_root_from_dotenv() -> Option<PathBuf> {
        let dotenv = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join(".env"),
        )
        .ok()?;
        dotenv.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == "JVS_ROOT").then(|| PathBuf::from(value.trim()))
        })
    }

    fn env_usize(name: &str, default: usize) -> usize {
        env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|left, right| left.total_cmp(right));
        values[values.len() / 2]
    }
}
