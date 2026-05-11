use std::{collections::VecDeque, path::Path, sync::Arc};

use anyhow::{Result, anyhow};
use num_complex::Complex32;
use ort::{inputs, session::Session, value::TensorRef};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::vad::init_onnx_runtime;

const MODEL_FILE: &str = "ulunas_stream_simple.onnx";
const FFT_SIZE: usize = 512;
const FFT_SIZE_U16: u16 = 512;
const FFT_SIZE_F32: f32 = 512.0;
const HOP_SIZE: usize = 256;
const FREQ_BINS: usize = FFT_SIZE / 2 + 1;
const COMPLEX_COMPONENTS: usize = 2;
const CONV_CACHE_LEN: usize = 5_358;
const TFA_CACHE_LEN: usize = 402;
const INTER_CACHE_LEN: usize = 1_056;

pub trait NoiseCancellationEngine: Send {
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>>;
}

pub struct UlUnasNoiseCancellationEngine {
    session: Session,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    previous_hop: Vec<f32>,
    pending: VecDeque<f32>,
    overlap: VecDeque<f32>,
    norm: VecDeque<f32>,
    conv_cache: Vec<f32>,
    tfa_cache: Vec<f32>,
    inter_cache: Vec<f32>,
}

impl UlUnasNoiseCancellationEngine {
    pub fn new(model_dir: &Path) -> Result<Self> {
        init_onnx_runtime();

        let model_path = model_dir.join(MODEL_FILE);
        if !model_path.is_file() {
            return Err(anyhow!(
                "Noise cancellation model not found: {}",
                model_path.display()
            ));
        }

        let session = Session::builder()
            .map_err(|err| anyhow!("Failed to create UL-UNAS session builder: {err}"))?
            .with_intra_threads(1)
            .map_err(|err| anyhow!("Failed to configure UL-UNAS session: {err}"))?
            .with_inter_threads(1)
            .map_err(|err| anyhow!("Failed to configure UL-UNAS inter-op threads: {err}"))?
            .with_parallel_execution(false)
            .map_err(|err| anyhow!("Failed to configure UL-UNAS execution mode: {err}"))?
            .with_intra_op_spinning(false)
            .map_err(|err| anyhow!("Failed to configure UL-UNAS intra-op spinning: {err}"))?
            .with_inter_op_spinning(false)
            .map_err(|err| anyhow!("Failed to configure UL-UNAS inter-op spinning: {err}"))?
            .commit_from_file(&model_path)
            .map_err(|err| {
                anyhow!(
                    "Failed to load UL-UNAS model {}: {err}",
                    model_path.display()
                )
            })?;
        let mut planner = RealFftPlanner::<f32>::new();

        Ok(Self {
            session,
            forward_fft: planner.plan_fft_forward(FFT_SIZE),
            inverse_fft: planner.plan_fft_inverse(FFT_SIZE),
            window: hann_window(),
            previous_hop: vec![0.0; HOP_SIZE],
            pending: VecDeque::new(),
            overlap: VecDeque::from(vec![0.0; FFT_SIZE]),
            norm: VecDeque::from(vec![0.0; FFT_SIZE]),
            conv_cache: vec![0.0; CONV_CACHE_LEN],
            tfa_cache: vec![0.0; TFA_CACHE_LEN],
            inter_cache: vec![0.0; INTER_CACHE_LEN],
        })
    }

    fn process_frame(&mut self, next_hop: &[f32]) -> Result<Vec<f32>> {
        let mut frame = Vec::with_capacity(FFT_SIZE);
        frame.extend_from_slice(&self.previous_hop);
        frame.extend_from_slice(next_hop);
        for (sample, window) in frame.iter_mut().zip(&self.window) {
            *sample *= *window;
        }

        let mut spectrum = self.forward_fft.make_output_vec();
        self.forward_fft.process(&mut frame, &mut spectrum)?;
        let enhanced_spectrum = self.run_model(&spectrum)?;

        let mut enhanced_frame = self.inverse_fft.make_output_vec();
        let mut enhanced_spectrum = enhanced_spectrum;
        self.inverse_fft
            .process(&mut enhanced_spectrum, &mut enhanced_frame)?;
        for ((sample, window), (overlap, norm)) in enhanced_frame
            .iter_mut()
            .zip(&self.window)
            .zip(self.overlap.iter_mut().zip(self.norm.iter_mut()))
        {
            *sample = (*sample / FFT_SIZE_F32) * *window;
            *overlap += *sample;
            *norm += *window * *window;
        }

        let output = pop_hop(&mut self.overlap, &mut self.norm);
        self.previous_hop.copy_from_slice(next_hop);
        Ok(output)
    }

    fn run_model(&mut self, spectrum: &[Complex32]) -> Result<Vec<Complex32>> {
        let mut mix = vec![0.0; FREQ_BINS * COMPLEX_COMPONENTS];
        for (index, value) in spectrum.iter().enumerate() {
            mix[index * COMPLEX_COMPONENTS] = value.re;
            mix[index * COMPLEX_COMPONENTS + 1] = value.im;
        }

        let mix = TensorRef::from_array_view(([1_usize, FREQ_BINS, 1, 2], mix.as_slice()))?;
        let conv_cache =
            TensorRef::from_array_view(([1_usize, CONV_CACHE_LEN], self.conv_cache.as_slice()))?;
        let tfa_cache =
            TensorRef::from_array_view(([1_usize, TFA_CACHE_LEN], self.tfa_cache.as_slice()))?;
        let inter_cache =
            TensorRef::from_array_view(([1_usize, INTER_CACHE_LEN], self.inter_cache.as_slice()))?;

        let outputs = self.session.run(inputs![
            "mix" => mix,
            "conv_cache" => conv_cache,
            "tfa_cache" => tfa_cache,
            "inter_cache" => inter_cache,
        ])?;

        let (_, enhanced) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, conv_cache_out) = outputs[1].try_extract_tensor::<f32>()?;
        let (_, tfa_cache_out) = outputs[2].try_extract_tensor::<f32>()?;
        let (_, inter_cache_out) = outputs[3].try_extract_tensor::<f32>()?;

        if enhanced.len() != FREQ_BINS * COMPLEX_COMPONENTS {
            return Err(anyhow!(
                "UL-UNAS returned unexpected enhanced spectrum size: {}",
                enhanced.len()
            ));
        }
        copy_cache(&mut self.conv_cache, conv_cache_out, "conv_cache_out")?;
        copy_cache(&mut self.tfa_cache, tfa_cache_out, "tfa_cache_out")?;
        copy_cache(&mut self.inter_cache, inter_cache_out, "inter_cache_out")?;

        Ok(enhanced
            .chunks_exact(COMPLEX_COMPONENTS)
            .map(|value| Complex32::new(value[0], value[1]))
            .collect())
    }
}

impl NoiseCancellationEngine for UlUnasNoiseCancellationEngine {
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        self.pending.extend(samples.iter().copied());
        let mut output = Vec::with_capacity(samples.len());
        while self.pending.len() >= HOP_SIZE {
            let next_hop = self.pending.drain(..HOP_SIZE).collect::<Vec<_>>();
            output.extend(self.process_frame(&next_hop)?);
        }
        Ok(output)
    }
}

fn pop_hop(overlap: &mut VecDeque<f32>, norm: &mut VecDeque<f32>) -> Vec<f32> {
    let mut output = Vec::with_capacity(HOP_SIZE);
    for _ in 0..HOP_SIZE {
        let sample = overlap.pop_front().unwrap_or(0.0);
        let weight = norm.pop_front().unwrap_or(0.0);
        output.push(if weight > 1.0e-6 {
            sample / weight
        } else {
            0.0
        });
        overlap.push_back(0.0);
        norm.push_back(0.0);
    }
    output
}

fn copy_cache(target: &mut [f32], source: &[f32], name: &str) -> Result<()> {
    if source.len() != target.len() {
        return Err(anyhow!(
            "UL-UNAS returned unexpected {name} size: {}",
            source.len()
        ));
    }
    target.copy_from_slice(source);
    Ok(())
}

fn hann_window() -> Vec<f32> {
    (0..FFT_SIZE_U16)
        .map(|index| {
            let phase = 2.0 * std::f32::consts::PI * f32::from(index) / FFT_SIZE_F32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{FFT_SIZE, HOP_SIZE, hann_window, pop_hop};

    #[test]
    fn hann_window_has_expected_edges() {
        let window = hann_window();

        assert_eq!(window.len(), FFT_SIZE);
        assert!(window[0].abs() < f32::EPSILON);
        assert!(window[HOP_SIZE] > 0.999);
    }

    #[test]
    fn pop_hop_normalizes_overlap_samples() {
        let mut overlap = std::collections::VecDeque::from(vec![2.0; FFT_SIZE]);
        let mut norm = std::collections::VecDeque::from(vec![4.0; FFT_SIZE]);

        let samples = pop_hop(&mut overlap, &mut norm);

        assert_eq!(samples, vec![0.5; HOP_SIZE]);
        assert_eq!(overlap.len(), FFT_SIZE);
        assert_eq!(norm.len(), FFT_SIZE);
    }
}
