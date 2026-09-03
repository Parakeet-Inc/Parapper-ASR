#[cfg(feature = "vad-ort")]
use std::path::Path;

#[cfg(feature = "vad-ort")]
use anyhow::{Result, anyhow};
#[cfg(feature = "vad-ort")]
use ort::{
    inputs,
    session::Session,
    value::{Tensor, TensorRef},
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "vad-ort")]
use crate::{SAMPLE_RATE_HZ, init_onnx_runtime};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VadResult {
    pub probability: f32,
    pub is_speech: bool,
}

pub trait VadEngine: Send {
    /// Calculates speech probability for one audio chunk.
    ///
    /// # Errors
    ///
    /// Returns an error when the concrete VAD model cannot process the chunk.
    fn process(&mut self, samples: &[f32]) -> anyhow::Result<VadResult>;
    fn set_threshold(&mut self, _threshold: f32) {}
}

#[cfg(feature = "vad-ort")]
const SILERO_CHUNK_SAMPLES: usize = 512;
#[cfg(feature = "vad-ort")]
const SILERO_CONTEXT_SAMPLES: usize = 64;
#[cfg(feature = "vad-ort")]
const SILERO_INPUT_SAMPLES: usize = SILERO_CONTEXT_SAMPLES + SILERO_CHUNK_SAMPLES;
#[cfg(feature = "vad-ort")]
const SILERO_STATE_LEN: usize = 2 * 128;

#[cfg(feature = "vad-ort")]
pub struct OnnxRuntimeSileroVadEngine {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
    threshold: f32,
}

#[cfg(feature = "vad-ort")]
impl OnnxRuntimeSileroVadEngine {
    /// Loads a Silero VAD ONNX model.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact is missing or the ONNX session cannot be loaded.
    pub fn new(model_path: &Path, threshold: f32) -> Result<Self> {
        init_onnx_runtime();

        if !model_path.is_file() {
            return Err(anyhow!("VAD model not found: {}", model_path.display()));
        }

        let session = Session::builder()
            .map_err(|err| anyhow!("Failed to create VAD session builder: {err}"))?
            .with_intra_threads(1)
            .map_err(|err| anyhow!("Failed to configure VAD session: {err}"))?
            .commit_from_file(model_path)
            .map_err(|err| anyhow!("Failed to load VAD model {}: {err}", model_path.display()))?;

        Ok(Self {
            session,
            state: vec![0.0; SILERO_STATE_LEN],
            context: vec![0.0; SILERO_CONTEXT_SAMPLES],
            threshold,
        })
    }
}

#[cfg(feature = "vad-ort")]
impl VadEngine for OnnxRuntimeSileroVadEngine {
    fn process(&mut self, samples: &[f32]) -> Result<VadResult> {
        if samples.is_empty() {
            return Ok(VadResult {
                probability: 0.0,
                is_speech: false,
            });
        }

        let mut chunk = [0.0; SILERO_CHUNK_SAMPLES];
        let copy_len = samples.len().min(SILERO_CHUNK_SAMPLES);
        chunk[..copy_len].copy_from_slice(&samples[..copy_len]);

        let mut input_samples = Vec::with_capacity(SILERO_INPUT_SAMPLES);
        input_samples.extend_from_slice(&self.context);
        input_samples.extend_from_slice(&chunk);

        let input = TensorRef::from_array_view((
            [1_usize, SILERO_INPUT_SAMPLES],
            input_samples.as_slice(),
        ))?;
        let sr = Tensor::from_array(((), vec![i64::from(SAMPLE_RATE_HZ)]))?;
        let state = TensorRef::from_array_view(([2_usize, 1, 128], self.state.as_slice()))?;

        let outputs = self.session.run(inputs![
            "input" => input,
            "sr" => sr,
            "state" => state,
        ])?;

        let (_, out) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, state_out) = outputs[1].try_extract_tensor::<f32>()?;

        if state_out.len() == self.state.len() {
            self.state.copy_from_slice(state_out);
        }
        self.context
            .copy_from_slice(&chunk[SILERO_CHUNK_SAMPLES - SILERO_CONTEXT_SAMPLES..]);

        let probability = out.first().copied().unwrap_or(0.0);
        Ok(VadResult {
            probability,
            is_speech: probability > self.threshold,
        })
    }

    fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
}

#[cfg(all(test, feature = "vad-ort"))]
mod tests {
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    use crate::init_onnx_runtime;

    #[test]
    fn onnx_runtime_initializes_without_hanging() {
        let (sender, receiver) = mpsc::channel();
        let started_at = Instant::now();
        thread::spawn(move || {
            init_onnx_runtime();
            let _ = sender.send(());
        });

        match receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(()) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!(
                    "ONNX Runtime initialization did not finish within {:?}",
                    started_at.elapsed()
                );
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("ONNX Runtime initialization thread stopped without returning a result");
            }
        }
    }
}
