use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow};
use ort::{inputs, session::Session, value::TensorRef};

use crate::runtime::init_onnx_runtime;

const SPEECHBRAIN_ECAPA_MODEL_FILE: &str = "lang-id-ecapa.onnx";
const SPEECHBRAIN_ECAPA_LABELS_FILE: &str = "labels.json";

pub struct SpokenLanguageIdentificationEngine {
    session: Session,
    labels: Vec<String>,
}

// The engine is owned by one host worker after construction. Its Session is
// never shared; moving the engine matches the ownership of other native models.
unsafe impl Send for SpokenLanguageIdentificationEngine {}

impl SpokenLanguageIdentificationEngine {
    /// Loads the `SpeechBrain` ECAPA model and label contract.
    ///
    /// # Errors
    ///
    /// Returns an error for missing or invalid model resources or when ORT
    /// cannot construct the Session.
    pub fn new(model_dir: &Path, num_threads: i32) -> Result<Self> {
        init_onnx_runtime();

        let model_path = model_dir.join(SPEECHBRAIN_ECAPA_MODEL_FILE);
        if !model_path.is_file() {
            return Err(anyhow!(
                "SpeechBrain language ID model not found: {}",
                model_path.display()
            ));
        }
        let labels_path = model_dir.join(SPEECHBRAIN_ECAPA_LABELS_FILE);
        let labels = read_speechbrain_labels(&labels_path)?;
        let builder = context_display(
            Session::builder(),
            "Failed to create SpeechBrain language ID builder",
        )?;
        let mut builder = context_display(
            builder.with_intra_threads(usize::try_from(num_threads.max(1)).unwrap_or(1)),
            "Failed to configure SpeechBrain language ID session",
        )?;
        let session = context_display(
            builder.commit_from_file(&model_path),
            format!(
                "Failed to load SpeechBrain language ID model {}",
                model_path.display()
            ),
        )?;
        Ok(Self { session, labels })
    }

    /// Selects the highest-probability language, optionally within candidates.
    ///
    /// # Errors
    ///
    /// Returns an error when ORT inference or tensor extraction fails.
    pub fn detect(&mut self, samples: &[f32], candidates: Option<&[&str]>) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let waveform = TensorRef::from_array_view(([1_usize, samples.len()], samples))?;
        let outputs = self.session.run(inputs!["waveform" => waveform])?;
        let (_, probabilities) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(select_label(&self.labels, probabilities, candidates).unwrap_or_default())
    }
}

fn select_label(
    labels: &[String],
    probabilities: &[f32],
    candidates: Option<&[&str]>,
) -> Option<String> {
    probabilities
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| {
            candidates.is_none_or(|candidates| {
                labels
                    .get(*index)
                    .is_some_and(|label| candidates.contains(&label.as_str()))
            })
        })
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .and_then(|(index, _)| labels.get(index).cloned())
}

fn context_display<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    context: impl std::fmt::Display,
) -> Result<T> {
    result.map_err(|error| anyhow!("{context}: {error}"))
}

fn read_speechbrain_labels(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path).with_context(|| {
        format!(
            "Failed to read SpeechBrain language labels: {}",
            path.display()
        )
    })?;
    serde_json::from_str::<Vec<String>>(&content).with_context(|| {
        format!(
            "Failed to parse SpeechBrain language labels: {}",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::select_label;

    #[test]
    fn candidate_filter_selects_the_highest_probability_allowed_language() {
        let labels = vec!["ja".to_string(), "en".to_string(), "fr".to_string()];
        let probabilities = [0.9, 0.7, 0.8];

        assert_eq!(
            select_label(&labels, &probabilities, Some(&["en", "fr"])),
            Some("fr".to_string())
        );
        assert_eq!(select_label(&labels, &probabilities, Some(&["de"])), None);
    }
}
