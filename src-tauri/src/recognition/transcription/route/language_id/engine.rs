use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow};
use ort::{inputs, session::Session, value::TensorRef};

use crate::model::onnx_runtime::init_onnx_runtime;

const SPEECHBRAIN_ECAPA_MODEL_FILE: &str = "lang-id-ecapa.onnx";
const SPEECHBRAIN_ECAPA_LABELS_FILE: &str = "labels.json";

pub struct SpokenLanguageIdentificationEngine {
    session: Session,
    labels: Vec<String>,
}

// The engine is owned by the ASR worker thread after construction. We do not
// share a Session across threads; this only allows moving the engine into that
// worker, matching the usage pattern of the other ONNX-backed engines.
unsafe impl Send for SpokenLanguageIdentificationEngine {}

impl SpokenLanguageIdentificationEngine {
    pub fn new(model_dir: &Path, num_threads: i32) -> Result<Self> {
        init_onnx_runtime()?;

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

    pub fn detect(&mut self, samples: &[f32], candidates: Option<&[&str]>) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let waveform = TensorRef::from_array_view(([1_usize, samples.len()], samples))?;
        let outputs = self.session.run(inputs!["waveform" => waveform])?;
        let (_, probabilities) = outputs[0].try_extract_tensor::<f32>()?;
        let Some((top_index, _)) = probabilities
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, _)| {
                candidates.is_none_or(|candidates| {
                    self.labels
                        .get(*index)
                        .is_some_and(|label| candidates.contains(&label.as_str()))
                })
            })
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
        else {
            return Ok(String::new());
        };
        Ok(self.labels.get(top_index).cloned().unwrap_or_default())
    }
}

fn context_display<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    context: impl std::fmt::Display,
) -> Result<T> {
    result.map_err(|err| anyhow!("{context}: {err}"))
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
