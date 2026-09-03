// Pinned ONNX dimensions are tiny constants that are guaranteed to fit i64.
#![allow(clippy::cast_possible_wrap)]

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    session::Session,
    value::{Tensor, ValueType},
};

use crate::{
    AsrEngine, AsrModel, AsrPrecision, AsrTranscript,
    decoder::ctc::{CtcDecodingStrategy, decode_ctc, transcript_from_path},
    frontend::NemoMelFrontend,
    init_onnx_runtime,
};

const CTC_MODEL_FILE: &str = "model.int8.onnx";
const TOKEN_FILE: &str = "tokens.txt";
const CTC_NUM_MEL_BINS: i64 = 80;
const CTC_VOCAB_SIZE: usize = 3_073;
const CTC_BLANK_ID: usize = 3_072;
const CTC_SUBSAMPLING_FACTOR: f64 = 8.0;
const FEATURE_FRAME_SHIFT_SEC: f64 = 0.01;

/// Direct ONNX Runtime implementation of NVIDIA's exported CTC branch.
pub struct NvidiaCtcOrtAsrEngine {
    session: Session,
    frontend: NemoMelFrontend,
    tokens: Vec<String>,
    decoding: CtcDecodingStrategy,
}

impl NvidiaCtcOrtAsrEngine {
    /// Loads the pinned Japanese Parakeet TDT/CTC CTC export.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported model configuration, missing or invalid model
    /// artifacts, or an ONNX Runtime session initialization failure.
    pub fn new(
        model_dir: &Path,
        model: AsrModel,
        precision: AsrPrecision,
        num_threads: i32,
    ) -> Result<Self> {
        Self::new_with_decoding(
            model_dir,
            model,
            precision,
            num_threads,
            CtcDecodingStrategy::Greedy,
        )
    }

    /// Loads the pinned CTC export with an explicit decoding strategy.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported model configuration, missing or invalid model
    /// artifacts, or an ONNX Runtime session initialization failure.
    pub fn new_with_decoding(
        model_dir: &Path,
        model: AsrModel,
        precision: AsrPrecision,
        num_threads: i32,
        decoding: CtcDecodingStrategy,
    ) -> Result<Self> {
        if model != AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 {
            bail!("direct CTC backend does not support {model:?}");
        }
        if precision != AsrPrecision::Int8 {
            bail!("direct CTC backend only supports the int8 export");
        }
        if num_threads <= 0 {
            bail!("ASR thread count must be greater than zero");
        }
        let model_path = model_dir.join(CTC_MODEL_FILE);
        let token_path = model_dir.join(TOKEN_FILE);
        if !model_path.is_file() {
            bail!("ASR model file not found: {}", model_path.display());
        }
        let tokens = load_tokens(&token_path)?;

        init_onnx_runtime();
        let session = Session::builder()
            .map_err(|error| anyhow!("failed to create CTC ONNX session builder: {error}"))?
            .with_intra_threads(usize::try_from(num_threads).unwrap_or(1))
            .map_err(|error| anyhow!("failed to configure CTC intra-op threads: {error}"))?
            .with_inter_threads(1)
            .map_err(|error| anyhow!("failed to configure CTC inter-op threads: {error}"))?
            .commit_from_file(&model_path)
            .map_err(|error| {
                anyhow!(
                    "failed to load direct CTC ONNX model {}: {error}",
                    model_path.display()
                )
            })?;
        validate_model_contract(&session)?;

        Ok(Self {
            session,
            frontend: NemoMelFrontend::new(),
            tokens,
            decoding,
        })
    }
}

impl AsrEngine for NvidiaCtcOrtAsrEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let features = self.frontend.process(samples)?;
        let frames = i64::try_from(features.frames).context("too many CTC feature frames")?;
        let valid_frames =
            i64::try_from(features.valid_frames).context("too many valid CTC feature frames")?;
        let audio_signal =
            Tensor::from_array((vec![1_i64, CTC_NUM_MEL_BINS, frames], features.values))?;
        let length = Tensor::from_array((vec![1_i64], vec![valid_frames]))?;
        let outputs = self.session.run(ort::inputs![
            "audio_signal" => audio_signal,
            "length" => length,
        ])?;
        let output = outputs
            .get("logprobs")
            .ok_or_else(|| anyhow!("direct CTC model did not return logprobs"))?;
        let (shape, log_probs) = output
            .try_extract_tensor::<f32>()
            .context("failed to extract direct CTC logprobs")?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] != CTC_VOCAB_SIZE as i64 {
            bail!("unexpected direct CTC output shape: {shape:?}");
        }
        let output_frames = usize::try_from(shape[1]).context("invalid CTC output frame count")?;
        let path = decode_ctc(
            log_probs,
            output_frames,
            CTC_VOCAB_SIZE,
            CTC_BLANK_ID,
            self.decoding,
        )?;
        transcript_from_path(
            &path,
            &self.tokens,
            FEATURE_FRAME_SHIFT_SEC * CTC_SUBSAMPLING_FACTOR,
        )
    }
}

fn load_tokens(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read CTC tokens: {}", path.display()))?;
    let mut tokens = Vec::<Option<String>>::new();
    for (line_index, line) in contents.lines().enumerate() {
        let (token, id) = line.rsplit_once(' ').ok_or_else(|| {
            anyhow!(
                "invalid CTC token line {} in {}",
                line_index + 1,
                path.display()
            )
        })?;
        let id = id.parse::<usize>().with_context(|| {
            format!(
                "invalid CTC token id on line {} in {}",
                line_index + 1,
                path.display()
            )
        })?;
        if id >= CTC_VOCAB_SIZE {
            bail!("CTC token id {id} exceeds model vocabulary");
        }
        if tokens.len() <= id {
            tokens.resize(id + 1, None);
        }
        if tokens[id].replace(token.to_string()).is_some() {
            bail!("duplicate CTC token id {id}");
        }
    }
    if tokens.len() != CTC_VOCAB_SIZE || tokens.iter().any(Option::is_none) {
        bail!(
            "CTC token table must contain contiguous ids 0..{CTC_VOCAB_SIZE}, found {} entries",
            tokens.iter().filter(|token| token.is_some()).count()
        );
    }
    let tokens = tokens.into_iter().flatten().collect::<Vec<_>>();
    if tokens[CTC_BLANK_ID] != "<blk>" {
        bail!("CTC blank token must be <blk> at id {CTC_BLANK_ID}");
    }
    Ok(tokens)
}

fn validate_model_contract(session: &Session) -> Result<()> {
    let inputs = session.inputs();
    let outputs = session.outputs();
    if inputs.len() != 2 || inputs[0].name() != "audio_signal" || inputs[1].name() != "length" {
        bail!("direct CTC model input contract changed");
    }
    if outputs.len() != 1 || outputs[0].name() != "logprobs" {
        bail!("direct CTC model output contract changed");
    }
    validate_tensor_rank(inputs[0].dtype(), 3, "audio_signal")?;
    validate_tensor_rank(inputs[1].dtype(), 1, "length")?;
    validate_tensor_rank(outputs[0].dtype(), 3, "logprobs")?;

    let metadata = session
        .metadata()
        .context("failed to read direct CTC metadata")?;
    for (key, expected) in [
        ("vocab_size", "3072"),
        ("subsampling_factor", "8"),
        ("normalize_type", "per_feature"),
        ("model_type", "EncDecHybridRNNTCTCBPEModel"),
    ] {
        let actual = metadata.custom(key);
        if actual.as_deref() != Some(expected) {
            bail!("direct CTC metadata {key} changed: {actual:?}");
        }
    }
    Ok(())
}

fn validate_tensor_rank(value_type: &ValueType, expected_rank: usize, name: &str) -> Result<()> {
    let ValueType::Tensor { shape, .. } = value_type else {
        bail!("direct CTC {name} is not a tensor");
    };
    if shape.len() != expected_rank {
        bail!("direct CTC {name} rank changed: {shape:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{CTC_BLANK_ID, CTC_VOCAB_SIZE, load_tokens};

    fn temporary_token_file(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "parapper-models-{name}-{}-tokens.txt",
            std::process::id()
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn token_contract_rejects_duplicate_gap_and_wrong_blank() {
        let duplicate = temporary_token_file("duplicate", "a 0\nb 0\n");
        assert!(
            load_tokens(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        let gap = temporary_token_file("gap", "a 0\n<blk> 3072\n");
        assert!(
            load_tokens(&gap)
                .unwrap_err()
                .to_string()
                .contains("contiguous")
        );

        let mut complete = (0..CTC_VOCAB_SIZE)
            .map(|id| format!("t{id} {id}"))
            .collect::<Vec<_>>();
        complete[CTC_BLANK_ID] = format!("wrong {CTC_BLANK_ID}");
        let wrong_blank = temporary_token_file("blank", &(complete.join("\n") + "\n"));
        assert!(
            load_tokens(&wrong_blank)
                .unwrap_err()
                .to_string()
                .contains("<blk>")
        );

        for path in [duplicate, gap, wrong_blank] {
            let _ = fs::remove_file(path);
        }
    }
}
