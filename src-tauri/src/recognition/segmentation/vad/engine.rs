use std::path::Path;

use anyhow::{Result, anyhow};
use ort::{
    inputs,
    session::Session,
    value::{Tensor, TensorRef},
};

use crate::{audio::ASR_SAMPLE_RATE, model::onnx_runtime::init_onnx_runtime};

const SILERO_CHUNK_SAMPLES: usize = 512;
const SILERO_CONTEXT_SAMPLES: usize = 64;
const SILERO_INPUT_SAMPLES: usize = SILERO_CONTEXT_SAMPLES + SILERO_CHUNK_SAMPLES;
const SILERO_STATE_LEN: usize = 2 * 128;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadResult {
    pub probability: f32,
    pub is_speech: bool,
}

pub trait VadEngine: Send {
    fn process(&mut self, samples: &[f32]) -> Result<VadResult>;
    fn set_threshold(&mut self, _threshold: f32) {}
}

pub struct OnnxRuntimeSileroVadEngine {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
    threshold: f32,
}

impl OnnxRuntimeSileroVadEngine {
    pub fn new(model_path: &Path, threshold: f32) -> Result<Self> {
        init_onnx_runtime()?;

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
        let sr = Tensor::from_array(((), vec![i64::from(ASR_SAMPLE_RATE)]))?;
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

#[cfg(test)]
mod tests {
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    use crate::model::onnx_runtime::init_onnx_runtime;

    #[test]
    fn onnx_runtime_initializes_without_hanging() {
        let (sender, receiver) = mpsc::channel();
        let started_at = Instant::now();
        thread::spawn(move || {
            let _ = sender.send(init_onnx_runtime());
        });

        match receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => panic!("ONNX Runtime initialization failed: {err:#}"),
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
