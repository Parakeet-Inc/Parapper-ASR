#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array, Array3};
use ort::{session::Session, value::Value};
use rand::Rng;
use regex::Regex;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

const TOTAL_STEP: usize = 5;
const SPEED: f32 = 1.05;
const SILENCE_SECONDS: f32 = 0.3;
const MAX_CHUNK_LENGTH: usize = 300;
const MAX_KO_CHUNK_LENGTH: usize = 120;

pub(in crate::synthesis) struct SupertonicOnnxTtsEngine {
    config: SupertonicConfig,
    text_processor: UnicodeProcessor,
    duration_predictor: Session,
    text_encoder: Session,
    vector_estimator: Session,
    vocoder: Session,
    voice_styles_dir: PathBuf,
    speaker_id: Option<i32>,
    style: Style,
    supported_languages: &'static [&'static str],
    pub(in crate::synthesis) sample_rate: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct SupertonicConfig {
    ae: AutoEncoderConfig,
    ttl: TextToLatentConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct AutoEncoderConfig {
    sample_rate: i32,
    base_chunk_size: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct TextToLatentConfig {
    chunk_compress_factor: i32,
    latent_dim: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct VoiceStyleData {
    style_ttl: StyleComponent,
    style_dp: StyleComponent,
}

#[derive(Debug, Clone, Deserialize)]
struct StyleComponent {
    data: Vec<Vec<Vec<f32>>>,
    dims: Vec<usize>,
}

struct Style {
    ttl: Array3<f32>,
    dp: Array3<f32>,
}

struct UnicodeProcessor {
    indexer: Vec<i64>,
}

impl SupertonicOnnxTtsEngine {
    pub(in crate::synthesis) fn new(
        model_dir: &Path,
        speaker_id: Option<i32>,
        supported_languages: &'static [&'static str],
    ) -> Result<Self> {
        let onnx_dir = model_dir.join("onnx");
        let config = load_config(&onnx_dir)?;
        let text_processor = UnicodeProcessor::new(&onnx_dir.join("unicode_indexer.json"))?;
        let voice_style_path = model_dir
            .join("voice_styles")
            .join(voice_style_file_name(speaker_id));
        let style = load_voice_style(&voice_style_path)?;
        let sample_rate = config.ae.sample_rate;
        Ok(Self {
            config,
            text_processor,
            duration_predictor: load_session(&onnx_dir.join("duration_predictor.onnx"))?,
            text_encoder: load_session(&onnx_dir.join("text_encoder.onnx"))?,
            vector_estimator: load_session(&onnx_dir.join("vector_estimator.onnx"))?,
            vocoder: load_session(&onnx_dir.join("vocoder.onnx"))?,
            voice_styles_dir: model_dir.join("voice_styles"),
            speaker_id,
            style,
            supported_languages,
            sample_rate,
        })
    }

    fn set_speaker(&mut self, speaker_id: Option<i32>) -> Result<()> {
        if self.speaker_id == speaker_id {
            return Ok(());
        }
        self.style = load_voice_style(
            &self
                .voice_styles_dir
                .join(voice_style_file_name(speaker_id)),
        )?;
        self.speaker_id = speaker_id;
        Ok(())
    }

    pub(in crate::synthesis) fn synthesize(
        &mut self,
        text: &str,
        speaker_id: Option<i32>,
        language: Option<&str>,
    ) -> Result<Vec<f32>> {
        self.set_speaker(speaker_id)?;
        let language = normalize_language(language, self.supported_languages);
        self.synthesize_normalized(text, language)
    }

    fn synthesize_normalized(&mut self, text: &str, language: &str) -> Result<Vec<f32>> {
        let max_len = if language == "ko" {
            MAX_KO_CHUNK_LENGTH
        } else {
            MAX_CHUNK_LENGTH
        };
        let chunks = chunk_text(text, max_len);
        let mut output = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let (samples, duration) = self.infer_one(chunk, language)?;
            let sample_len = (self.sample_rate as f32 * duration) as usize;
            if index > 0 {
                output.extend(std::iter::repeat_n(
                    0.0,
                    (SILENCE_SECONDS * self.sample_rate as f32) as usize,
                ));
            }
            output.extend_from_slice(&samples[..sample_len.min(samples.len())]);
        }
        Ok(output)
    }

    fn infer_one(&mut self, text: &str, language: &str) -> Result<(Vec<f32>, f32)> {
        let text_list = [text.to_string()];
        let lang_list = [language.to_string()];
        let (text_ids, text_mask) =
            self.text_processor
                .process(&text_list, &lang_list, self.supported_languages)?;
        let text_ids_array = Array::from_shape_vec((1, text_ids[0].len()), text_ids[0].clone())?;

        let text_ids_value = Value::from_array(text_ids_array)?;
        let text_mask_value = Value::from_array(text_mask.clone())?;
        let style_dp_value = Value::from_array(self.style.dp.clone())?;
        let duration_outputs = self.duration_predictor.run(ort::inputs! {
            "text_ids" => &text_ids_value,
            "style_dp" => &style_dp_value,
            "text_mask" => &text_mask_value,
        })?;
        let (_, duration_data) = duration_outputs["duration"].try_extract_tensor::<f32>()?;
        let duration = duration_data
            .first()
            .copied()
            .ok_or_else(|| anyhow!("Supertonic duration output is empty"))?
            / SPEED;

        let style_ttl_value = Value::from_array(self.style.ttl.clone())?;
        let text_outputs = self.text_encoder.run(ort::inputs! {
            "text_ids" => &text_ids_value,
            "style_ttl" => &style_ttl_value,
            "text_mask" => &text_mask_value,
        })?;
        let (text_emb_shape, text_emb_data) =
            text_outputs["text_emb"].try_extract_tensor::<f32>()?;
        let text_emb = Array3::from_shape_vec(
            (
                text_emb_shape[0] as usize,
                text_emb_shape[1] as usize,
                text_emb_shape[2] as usize,
            ),
            text_emb_data.to_vec(),
        )?;

        let (mut latent, latent_mask) = sample_noisy_latent(
            duration,
            self.sample_rate,
            self.config.ae.base_chunk_size,
            self.config.ttl.chunk_compress_factor,
            self.config.ttl.latent_dim,
        );
        let total_step = Array::from_elem(1, TOTAL_STEP as f32);
        for step in 0..TOTAL_STEP {
            let latent_value = Value::from_array(latent.clone())?;
            let text_emb_value = Value::from_array(text_emb.clone())?;
            let latent_mask_value = Value::from_array(latent_mask.clone())?;
            let text_mask_value = Value::from_array(text_mask.clone())?;
            let current_step = Value::from_array(Array::from_elem(1, step as f32))?;
            let total_step_value = Value::from_array(total_step.clone())?;

            let outputs = self.vector_estimator.run(ort::inputs! {
                "noisy_latent" => &latent_value,
                "text_emb" => &text_emb_value,
                "style_ttl" => &style_ttl_value,
                "latent_mask" => &latent_mask_value,
                "text_mask" => &text_mask_value,
                "current_step" => &current_step,
                "total_step" => &total_step_value,
            })?;
            let (shape, data) = outputs["denoised_latent"].try_extract_tensor::<f32>()?;
            latent = Array3::from_shape_vec(
                (shape[0] as usize, shape[1] as usize, shape[2] as usize),
                data.to_vec(),
            )?;
        }

        let latent_value = Value::from_array(latent)?;
        let outputs = self.vocoder.run(ort::inputs! {
            "latent" => &latent_value,
        })?;
        let (_, samples) = outputs["wav_tts"].try_extract_tensor::<f32>()?;
        Ok((samples.to_vec(), duration))
    }
}

impl UnicodeProcessor {
    fn new(path: &Path) -> Result<Self> {
        let reader = BufReader::new(
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?,
        );
        let indexer = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse {}", path.display()))?;
        Ok(Self { indexer })
    }

    fn process(
        &self,
        text_list: &[String],
        lang_list: &[String],
        supported_languages: &[&str],
    ) -> Result<(Vec<Vec<i64>>, Array3<f32>)> {
        let processed_texts = text_list
            .iter()
            .zip(lang_list)
            .map(|(text, lang)| preprocess_text(text, lang, supported_languages))
            .collect::<Result<Vec<_>>>()?;
        let text_lengths = processed_texts
            .iter()
            .map(|text| text.chars().count())
            .collect::<Vec<_>>();
        let max_len = text_lengths.iter().copied().max().unwrap_or(0);
        let mut text_ids = Vec::with_capacity(processed_texts.len());
        for text in &processed_texts {
            let mut row = vec![0_i64; max_len];
            for (index, codepoint) in text.chars().map(|c| c as usize).enumerate() {
                row[index] = self.indexer.get(codepoint).copied().unwrap_or(-1);
            }
            text_ids.push(row);
        }
        Ok((text_ids, length_to_mask(&text_lengths, max_len)))
    }
}

fn load_config(onnx_dir: &Path) -> Result<SupertonicConfig> {
    let path = onnx_dir.join("tts.json");
    let reader = BufReader::new(
        File::open(&path).with_context(|| format!("Failed to open {}", path.display()))?,
    );
    serde_json::from_reader(reader).with_context(|| format!("Failed to parse {}", path.display()))
}

fn load_session(path: &Path) -> Result<Session> {
    Session::builder()?
        .commit_from_file(path)
        .with_context(|| format!("Failed to load ONNX model {}", path.display()))
}

fn load_voice_style(path: &Path) -> Result<Style> {
    let reader = BufReader::new(
        File::open(path).with_context(|| format!("Failed to open {}", path.display()))?,
    );
    let data: VoiceStyleData = serde_json::from_reader(reader)
        .with_context(|| format!("Failed to parse {}", path.display()))?;
    let ttl_dims = component_dims(&data.style_ttl, "style_ttl")?;
    let dp_dims = component_dims(&data.style_dp, "style_dp")?;
    Ok(Style {
        ttl: Array3::from_shape_vec(ttl_dims, flatten_component(&data.style_ttl))?,
        dp: Array3::from_shape_vec(dp_dims, flatten_component(&data.style_dp))?,
    })
}

fn component_dims(component: &StyleComponent, name: &str) -> Result<(usize, usize, usize)> {
    if component.dims.len() != 3 {
        bail!("{name} must have 3 dimensions");
    }
    Ok((component.dims[0], component.dims[1], component.dims[2]))
}

fn flatten_component(component: &StyleComponent) -> Vec<f32> {
    component
        .data
        .iter()
        .flat_map(|batch| batch.iter())
        .flat_map(|row| row.iter().copied())
        .collect()
}

fn preprocess_text(text: &str, language: &str, supported_languages: &[&str]) -> Result<String> {
    if !supported_languages.contains(&language) {
        bail!("Invalid Supertonic language: {language}");
    }
    let mut text = text.nfkd().collect::<String>();
    text = emoji_regex().replace_all(&text, "").to_string();
    for (from, to) in [
        ("\u{2013}", "-"),
        ("\u{2011}", "-"),
        ("\u{2014}", "-"),
        ("_", " "),
        ("\u{201C}", "\""),
        ("\u{201D}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
        ("\u{00b4}", "'"),
        ("`", "'"),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("/", " "),
        ("#", " "),
        ("\u{2192}", " "),
        ("\u{2190}", " "),
        ("@", " at "),
        ("e.g.,", "for example, "),
        ("i.e.,", "that is, "),
    ] {
        text = text.replace(from, to);
    }
    for symbol in ["\u{2665}", "\u{2606}", "\u{2661}", "\u{00a9}", "\\"] {
        text = text.replace(symbol, "");
    }
    for (from, to) in [
        (" ,", ","),
        (" .", "."),
        (" !", "!"),
        (" ?", "?"),
        (" ;", ";"),
        (" :", ":"),
        (" '", "'"),
    ] {
        text = text.replace(from, to);
    }
    while text.contains("\"\"") {
        text = text.replace("\"\"", "\"");
    }
    while text.contains("''") {
        text = text.replace("''", "'");
    }
    text = whitespace_regex()
        .replace_all(&text, " ")
        .trim()
        .to_string();
    if !text.is_empty() && !sentence_end_regex().is_match(&text) {
        text.push('.');
    }
    Ok(format!("<{language}>{text}</{language}>"))
}

fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() || text.chars().count() <= max_len {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let next_len =
            current.chars().count() + usize::from(!current.is_empty()) + word.chars().count();
        if next_len > max_len && !current.is_empty() {
            chunks.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn length_to_mask(lengths: &[usize], max_len: usize) -> Array3<f32> {
    let mut mask = Array3::<f32>::zeros((lengths.len(), 1, max_len));
    for (batch, length) in lengths.iter().copied().enumerate() {
        for index in 0..length.min(max_len) {
            mask[[batch, 0, index]] = 1.0;
        }
    }
    mask
}

fn sample_noisy_latent(
    duration: f32,
    sample_rate: i32,
    base_chunk_size: i32,
    chunk_compress: i32,
    latent_dim: i32,
) -> (Array3<f32>, Array3<f32>) {
    let wav_len = (duration * sample_rate as f32) as usize;
    let chunk_size = (base_chunk_size * chunk_compress) as usize;
    let latent_len = wav_len.div_ceil(chunk_size).max(1);
    let latent_dim = (latent_dim * chunk_compress) as usize;
    let mut latent = Array3::<f32>::zeros((1, latent_dim, latent_len));
    let mut rng = rand::rng();
    for value in &mut latent {
        *value = normal_sample(&mut rng);
    }
    let mask = Array3::<f32>::ones((1, 1, latent_len));
    (latent, mask)
}

fn normal_sample(rng: &mut impl Rng) -> f32 {
    let u1 = rng.random::<f32>().clamp(f32::MIN_POSITIVE, 1.0);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn normalize_language(
    language: Option<&str>,
    supported_languages: &'static [&'static str],
) -> &'static str {
    let normalized = language.unwrap_or("en").trim().to_ascii_lowercase();
    if supported_languages.contains(&normalized.as_str()) {
        match normalized.as_str() {
            "ar" => "ar",
            "bg" => "bg",
            "cs" => "cs",
            "da" => "da",
            "de" => "de",
            "el" => "el",
            "es" => "es",
            "et" => "et",
            "fi" => "fi",
            "fr" => "fr",
            "hi" => "hi",
            "hu" => "hu",
            "id" => "id",
            "it" => "it",
            "ja" => "ja",
            "ko" => "ko",
            "nl" => "nl",
            "pl" => "pl",
            "pt" => "pt",
            "ro" => "ro",
            "ru" => "ru",
            "vi" => "vi",
            _ => "en",
        }
    } else {
        "en"
    }
}

fn voice_style_file_name(speaker_id: Option<i32>) -> &'static str {
    match speaker_id.unwrap_or(0).clamp(0, 9) {
        0 => "F1.json",
        1 => "F2.json",
        2 => "F3.json",
        3 => "F4.json",
        4 => "F5.json",
        5 => "M1.json",
        6 => "M2.json",
        7 => "M3.json",
        8 => "M4.json",
        _ => "M5.json",
    }
}

fn emoji_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{1F1E6}-\x{1F1FF}]+",
        )
        .expect("emoji regex must compile")
    })
}

fn whitespace_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\s+").expect("whitespace regex must compile"))
}

fn sentence_end_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"[.!?;:,'"“”‘’)）\]}…。｣』】〉》›»]$"#)
            .expect("sentence end regex must compile")
    })
}
