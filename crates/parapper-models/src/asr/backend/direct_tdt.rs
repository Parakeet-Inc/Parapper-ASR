// Pinned tensor dimensions fit i64, and frame timestamps are represented as f32 by the ASR API.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    session::Session,
    value::{Tensor, ValueType},
};

use crate::{
    AsrEngine, AsrModel, AsrPrecision, AsrTranscript,
    decoder::tdt::{TdtHypothesis, TdtNetwork, default_beam_tdt, greedy_tdt},
    frontend::NemoMelFrontend,
    init_onnx_runtime,
};

const ENCODER_FILE: &str = "encoder.int8.onnx";
const DECODER_FILE: &str = "decoder.int8.onnx";
const JOINER_FILE: &str = "joiner.int8.onnx";
const TOKEN_FILE: &str = "tokens.txt";
const MEL_BINS: usize = 128;
const ENCODER_DIM: usize = 1_024;
const PREDICTOR_DIM: usize = 640;
const PREDICTOR_LAYERS: usize = 2;
const FEATURE_FRAME_SHIFT_SEC: f32 = 0.01;
const SUBSAMPLING_FACTOR: f32 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdtDecodingStrategy {
    Greedy,
    DefaultBeam { beam_size: usize },
}

#[derive(Debug, Clone, Copy)]
struct TdtContract {
    blank_id: usize,
    vocab_size: usize,
    joiner_width: usize,
}

impl TdtContract {
    fn for_model(model: AsrModel) -> Result<Self> {
        match model {
            AsrModel::NemoParakeetTdt0_6BV2Int8 => Ok(Self {
                blank_id: 1_024,
                vocab_size: 1_024,
                joiner_width: 1_030,
            }),
            AsrModel::NemoParakeetTdt0_6BV3Int8 => Ok(Self {
                blank_id: 8_192,
                vocab_size: 8_192,
                joiner_width: 8_198,
            }),
            _ => bail!("direct TDT backend does not support {model:?}"),
        }
    }
}

pub struct NvidiaTdtOrtAsrEngine {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    frontend: NemoMelFrontend,
    tokens: Vec<String>,
    contract: TdtContract,
    decoding: TdtDecodingStrategy,
}

impl NvidiaTdtOrtAsrEngine {
    /// Loads a pinned NVIDIA TDT export with greedy decoding.
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
            TdtDecodingStrategy::Greedy,
        )
    }

    /// Loads a pinned NVIDIA TDT export with an explicit decoding strategy.
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
        decoding: TdtDecodingStrategy,
    ) -> Result<Self> {
        let contract = TdtContract::for_model(model)?;
        if precision != AsrPrecision::Int8 {
            bail!("direct TDT backend only supports the int8 exports");
        }
        if num_threads <= 0 {
            bail!("ASR thread count must be greater than zero");
        }
        if matches!(decoding, TdtDecodingStrategy::DefaultBeam { beam_size: 0 }) {
            bail!("TDT beam size must be positive");
        }
        let tokens = load_tokens(&model_dir.join(TOKEN_FILE), contract)?;

        init_onnx_runtime();
        let threads = usize::try_from(num_threads).context("invalid ASR thread count")?;
        let encoder = load_session(&model_dir.join(ENCODER_FILE), threads, "TDT encoder")?;
        let decoder = load_session(&model_dir.join(DECODER_FILE), threads, "TDT predictor")?;
        let joiner = load_session(&model_dir.join(JOINER_FILE), threads, "TDT joiner")?;
        validate_contract(&encoder, &decoder, &joiner, contract)?;

        Ok(Self {
            encoder,
            decoder,
            joiner,
            frontend: NemoMelFrontend::with_mel_bins(MEL_BINS),
            tokens,
            contract,
            decoding,
        })
    }
}

impl AsrEngine for NvidiaTdtOrtAsrEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let features = self.frontend.process(samples)?;
        let feature_frames =
            i64::try_from(features.frames).context("too many TDT feature frames")?;
        let valid_frames =
            i64::try_from(features.valid_frames).context("too many valid TDT feature frames")?;
        let signal = Tensor::from_array((
            vec![1_i64, MEL_BINS as i64, feature_frames],
            features.values,
        ))?;
        let length = Tensor::from_array((vec![1_i64], vec![valid_frames]))?;
        let encoder_outputs = self.encoder.run(ort::inputs![
            "audio_signal" => signal,
            "length" => length,
        ])?;
        let (encoder_shape, encoder_values) = encoder_outputs
            .get("outputs")
            .ok_or_else(|| anyhow!("TDT encoder did not return outputs"))?
            .try_extract_tensor::<f32>()
            .context("failed to extract TDT encoder output")?;
        let encoded_length = encoder_outputs
            .get("encoded_lengths")
            .ok_or_else(|| anyhow!("TDT encoder did not return encoded_lengths"))?
            .try_extract_tensor::<i64>()
            .context("failed to extract TDT encoded length")?
            .1
            .first()
            .copied()
            .ok_or_else(|| anyhow!("TDT encoded length is empty"))?;
        if encoder_shape.len() != 3
            || encoder_shape[0] != 1
            || encoder_shape[1] != ENCODER_DIM as i64
        {
            bail!("unexpected TDT encoder output shape: {encoder_shape:?}");
        }
        let all_frames = usize::try_from(encoder_shape[2]).context("invalid TDT encoder frames")?;
        let frames = usize::try_from(encoded_length).context("invalid TDT encoded length")?;
        if frames == 0 || frames > all_frames {
            bail!("TDT encoded length {frames} exceeds output frames {all_frames}");
        }
        let mut trimmed = vec![0.0; ENCODER_DIM * frames];
        for feature in 0..ENCODER_DIM {
            trimmed[feature * frames..(feature + 1) * frames].copy_from_slice(
                &encoder_values[feature * all_frames..feature * all_frames + frames],
            );
        }
        drop(encoder_outputs);

        let mut network = OrtTdtNetwork {
            decoder: &mut self.decoder,
            joiner: &mut self.joiner,
            contract: self.contract,
        };
        let (hypothesis, include_durations) = match self.decoding {
            TdtDecodingStrategy::Greedy => (
                greedy_tdt(
                    &mut network,
                    &trimmed,
                    frames,
                    ENCODER_DIM,
                    self.contract.blank_id,
                )?,
                true,
            ),
            TdtDecodingStrategy::DefaultBeam { beam_size } => (
                default_beam_tdt(
                    &mut network,
                    &trimmed,
                    frames,
                    ENCODER_DIM,
                    self.contract.blank_id,
                    beam_size,
                )?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("TDT beam search returned no hypotheses"))?,
                false,
            ),
        };
        transcript_from_hypothesis(&hypothesis, &self.tokens, include_durations)
    }
}

#[derive(Clone)]
struct PredictorState {
    hidden: Vec<f32>,
    cell: Vec<f32>,
}

struct OrtTdtNetwork<'a> {
    decoder: &'a mut Session,
    joiner: &'a mut Session,
    contract: TdtContract,
}

impl TdtNetwork for OrtTdtNetwork<'_> {
    type State = PredictorState;

    fn initial_state(&self) -> Self::State {
        let elements = PREDICTOR_LAYERS * PREDICTOR_DIM;
        PredictorState {
            hidden: vec![0.0; elements],
            cell: vec![0.0; elements],
        }
    }

    fn predictor(&mut self, token: usize, state: &Self::State) -> Result<(Vec<f32>, Self::State)> {
        let token = i32::try_from(token).context("TDT token id exceeds i32")?;
        let targets = Tensor::from_array((vec![1_i64, 1], vec![token]))?;
        let target_length = Tensor::from_array((vec![1_i64], vec![1_i32]))?;
        let hidden =
            Tensor::from_array((vec![2_i64, 1, PREDICTOR_DIM as i64], state.hidden.clone()))?;
        let cell = Tensor::from_array((vec![2_i64, 1, PREDICTOR_DIM as i64], state.cell.clone()))?;
        let outputs = self.decoder.run(ort::inputs![
            "targets" => targets,
            "target_length" => target_length,
            "states.1" => hidden,
            "onnx::Slice_3" => cell,
        ])?;
        let prediction = extract_f32(&outputs, "outputs", PREDICTOR_DIM)?;
        let next_hidden = extract_f32(&outputs, "states", PREDICTOR_LAYERS * PREDICTOR_DIM)?;
        let next_cell = extract_f32(&outputs, "162", PREDICTOR_LAYERS * PREDICTOR_DIM)?;
        Ok((
            prediction,
            PredictorState {
                hidden: next_hidden,
                cell: next_cell,
            },
        ))
    }

    fn joiner(&mut self, encoder_frame: &[f32], predictor: &[f32]) -> Result<Vec<f32>> {
        if encoder_frame.len() != ENCODER_DIM || predictor.len() != PREDICTOR_DIM {
            bail!("invalid TDT joiner input shape");
        }
        let encoder =
            Tensor::from_array((vec![1_i64, ENCODER_DIM as i64, 1], encoder_frame.to_vec()))?;
        let decoder =
            Tensor::from_array((vec![1_i64, PREDICTOR_DIM as i64, 1], predictor.to_vec()))?;
        let outputs = self.joiner.run(ort::inputs![
            "encoder_outputs" => encoder,
            "decoder_outputs" => decoder,
        ])?;
        extract_f32(&outputs, "outputs", self.contract.joiner_width)
    }
}

fn extract_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let (_, values) = outputs
        .get(name)
        .ok_or_else(|| anyhow!("ONNX session did not return {name}"))?
        .try_extract_tensor::<f32>()
        .with_context(|| format!("failed to extract ONNX output {name}"))?;
    if values.len() != expected_elements {
        bail!(
            "ONNX output {name} has {} elements, expected {expected_elements}",
            values.len()
        );
    }
    Ok(values.to_vec())
}

fn transcript_from_hypothesis(
    hypothesis: &TdtHypothesis<PredictorState>,
    tokens: &[String],
    include_durations: bool,
) -> Result<AsrTranscript> {
    let token_texts = hypothesis
        .token_ids
        .iter()
        .map(|&id| {
            tokens
                .get(id)
                .map(|token| token.replace('▁', " "))
                .ok_or_else(|| anyhow!("TDT decoder emitted unknown token id {id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let text = token_texts.concat();
    let timestamps = hypothesis
        .timestamps
        .iter()
        .map(|&frame| frame as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
        .collect::<Vec<_>>();
    let durations = hypothesis
        .durations
        .iter()
        .map(|&frames| frames as f32 * FEATURE_FRAME_SHIFT_SEC * SUBSAMPLING_FACTOR)
        .collect::<Vec<_>>();
    Ok(AsrTranscript::from_parts(
        text,
        token_texts,
        Some(&timestamps),
        include_durations.then_some(durations.as_slice()),
    ))
}

fn load_session(path: &Path, threads: usize, label: &str) -> Result<Session> {
    if !path.is_file() {
        bail!("{label} file not found: {}", path.display());
    }
    Session::builder()
        .map_err(|error| anyhow!("failed to create {label} session builder: {error}"))?
        .with_intra_threads(threads)
        .map_err(|error| anyhow!("failed to configure {label} intra-op threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow!("failed to configure {label} inter-op threads: {error}"))?
        .commit_from_file(path)
        .map_err(|error| anyhow!("failed to load {label} {}: {error}", path.display()))
}

fn load_tokens(path: &Path, contract: TdtContract) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read TDT tokens: {}", path.display()))?;
    let mut tokens = vec![None; contract.blank_id + 1];
    for (line_index, line) in contents.lines().enumerate() {
        let (token, id) = line.rsplit_once(' ').ok_or_else(|| {
            anyhow!(
                "invalid TDT token line {} in {}",
                line_index + 1,
                path.display()
            )
        })?;
        let id = id.parse::<usize>().with_context(|| {
            format!(
                "invalid TDT token id on line {} in {}",
                line_index + 1,
                path.display()
            )
        })?;
        let slot = tokens
            .get_mut(id)
            .ok_or_else(|| anyhow!("TDT token id {id} exceeds model vocabulary"))?;
        if slot.replace(token.to_string()).is_some() {
            bail!("duplicate TDT token id {id}");
        }
    }
    if tokens.iter().any(Option::is_none) {
        bail!(
            "TDT token table must contain contiguous ids through blank {}",
            contract.blank_id
        );
    }
    let tokens = tokens.into_iter().flatten().collect::<Vec<_>>();
    if tokens[contract.blank_id] != "<blk>" {
        bail!("TDT blank token must be <blk> at id {}", contract.blank_id);
    }
    Ok(tokens)
}

fn validate_contract(
    encoder: &Session,
    decoder: &Session,
    joiner: &Session,
    contract: TdtContract,
) -> Result<()> {
    validate_names(
        encoder,
        &["audio_signal", "length"],
        &["outputs", "encoded_lengths"],
        "encoder",
    )?;
    validate_names(
        decoder,
        &["targets", "target_length", "states.1", "onnx::Slice_3"],
        &["outputs", "prednet_lengths", "states", "162"],
        "predictor",
    )?;
    validate_names(
        joiner,
        &["encoder_outputs", "decoder_outputs"],
        &["outputs"],
        "joiner",
    )?;
    let metadata = encoder
        .metadata()
        .context("failed to read TDT encoder metadata")?;
    for (key, expected) in [
        ("vocab_size", contract.vocab_size.to_string()),
        ("subsampling_factor", "8".to_string()),
        ("normalize_type", "per_feature".to_string()),
        ("pred_rnn_layers", "2".to_string()),
        ("pred_hidden", "640".to_string()),
        ("feat_dim", "128".to_string()),
    ] {
        let actual = metadata.custom(key);
        if actual.as_deref() != Some(expected.as_str()) {
            bail!("direct TDT metadata {key} changed: {actual:?}");
        }
    }
    Ok(())
}

fn validate_names(
    session: &Session,
    expected_inputs: &[&str],
    expected_outputs: &[&str],
    label: &str,
) -> Result<()> {
    let inputs = session.inputs();
    let outputs = session.outputs();
    if inputs
        .iter()
        .map(ort::value::Outlet::name)
        .collect::<Vec<_>>()
        != expected_inputs
        || outputs
            .iter()
            .map(ort::value::Outlet::name)
            .collect::<Vec<_>>()
            != expected_outputs
    {
        bail!("direct TDT {label} I/O contract changed");
    }
    for input in inputs {
        if !matches!(input.dtype(), ValueType::Tensor { .. }) {
            bail!("direct TDT {label} input {} is not a tensor", input.name());
        }
    }
    for output in outputs {
        if !matches!(output.dtype(), ValueType::Tensor { .. }) {
            bail!(
                "direct TDT {label} output {} is not a tensor",
                output.name()
            );
        }
    }
    Ok(())
}
