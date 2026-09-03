// Audio-sized tensor dimensions fit i64, and transcript timestamps use f32 seconds.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use onnx_protobuf::{Message, ModelProto, TensorProto};
use ort::{session::Session, value::Tensor};

use crate::{
    AsrEngine, AsrLanguage, AsrModel, AsrPrecision, AsrStreamConfig, AsrTranscript,
    StreamingSessionId,
    decoder::rnnt::{RnntHypothesis, RnntNetwork, greedy_rnnt},
    frontend::{NemoStreamingAdapter, NemoStreamingWindow},
    init_onnx_runtime,
};

const ENCODER_FILE: &str = "encoder.int8.onnx";
const DECODER_FILE: &str = "decoder.int8.onnx";
const JOINER_FILE: &str = "joiner.int8.onnx";
const TOKEN_FILE: &str = "tokens.txt";
const ENCODER_DIM: usize = 1_024;
const CACHE_LAYERS: usize = 24;
const ENGLISH_CACHE_CHANNEL_FRAMES: usize = 70;
const MULTILINGUAL_CACHE_CHANNEL_FRAMES: usize = 56;
const CACHE_TIME_FRAMES: usize = 8;
const PREDICTOR_DIM: usize = 640;
const PREDICTOR_STATE_ELEMENTS: usize = 2 * PREDICTOR_DIM;
const CACHE_TIME_ELEMENTS: usize = CACHE_LAYERS * ENCODER_DIM * CACHE_TIME_FRAMES;
const MAX_SYMBOLS_PER_FRAME: usize = 10;
const FRAME_SECONDS: f32 = 0.08;
// The legacy host adapter used this 160ms grid and 80ms fade for all pinned
// Nemotron variants. Keep the established contract here, where model-native
// buffering belongs, instead of teaching SourceRuntime its native shape.
const STREAM_BOOTSTRAP_CHUNK_SAMPLES: usize = 2_560;
const STREAM_BOOTSTRAP_FADE_SAMPLES: usize = 1_280;

#[derive(Debug, Clone, Copy)]
struct NemotronContract {
    window_frames: usize,
    shift_frames: usize,
    blank_id: usize,
    cache_channel_frames: usize,
    prompt_ids: Option<NemotronPromptIds>,
}

#[derive(Debug, Clone, Copy)]
struct NemotronPromptIds {
    auto: i64,
    japanese: i64,
}

impl NemotronContract {
    fn for_model(model: AsrModel) -> Result<Self> {
        let (window_frames, shift_frames) = match model {
            AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8
            | AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8 => (17, 8),
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8
            | AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8 => (25, 16),
            // NeMo accepts the English [70, 3] attention context at runtime even
            // though NVIDIA's exported English candidate list omits 320 ms. Its
            // mask builder uses integer division, so the 70-frame cache retains
            // 70 / 4 = 17 chunks, matching the Python implementation.
            AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8
            | AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8 => (41, 32),
            AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8
            | AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8 => (65, 56),
            AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8
            | AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8 => (121, 112),
            _ => bail!("direct Nemotron backend does not support {model:?}"),
        };
        let has_prompt = matches!(
            model,
            AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8
                | AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
                | AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8
                | AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8
                | AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8
        );
        Ok(Self {
            window_frames,
            shift_frames,
            blank_id: if has_prompt { 13_087 } else { 1_024 },
            cache_channel_frames: if has_prompt {
                MULTILINGUAL_CACHE_CHANNEL_FRAMES
            } else {
                ENGLISH_CACHE_CHANNEL_FRAMES
            },
            prompt_ids: None,
        })
    }

    fn cache_channel_elements(self) -> usize {
        CACHE_LAYERS * self.cache_channel_frames * ENCODER_DIM
    }

    fn prompt_id(self, language_hint: Option<AsrLanguage>) -> Option<i64> {
        self.prompt_ids.map(|ids| match language_hint {
            Some(AsrLanguage::Japanese) => ids.japanese,
            _ => ids.auto,
        })
    }
}

pub struct NemotronOrtAsrEngine {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    contract: NemotronContract,
    tokens: Vec<String>,
    streams: HashMap<StreamingSessionId, StreamState>,
}

struct StreamState {
    adapter: NemoStreamingAdapter,
    cache_channel: Vec<f32>,
    cache_time: Vec<f32>,
    cache_channel_len: i64,
    prompt_id: Option<i64>,
    hypothesis: Option<RnntHypothesis<PredictorState>>,
}

impl StreamState {
    fn new(contract: NemotronContract, config: AsrStreamConfig) -> Self {
        let mut adapter = NemoStreamingAdapter::new(
            contract.window_frames,
            contract.shift_frames,
            STREAM_BOOTSTRAP_CHUNK_SAMPLES,
            STREAM_BOOTSTRAP_FADE_SAMPLES,
        );
        adapter.start(config);
        Self {
            adapter,
            cache_channel: vec![0.0; contract.cache_channel_elements()],
            cache_time: vec![0.0; CACHE_TIME_ELEMENTS],
            cache_channel_len: 0,
            prompt_id: contract.prompt_id(config.language_hint),
            hypothesis: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PredictorState {
    hidden: Vec<f32>,
    cell: Vec<f32>,
}

impl NemotronOrtAsrEngine {
    /// Loads a pinned streaming Nemotron export.
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
        let contract = NemotronContract::for_model(model)?;
        if precision != AsrPrecision::Int8 {
            bail!("direct Nemotron backend only supports int8 exports");
        }
        let threads = usize::try_from(num_threads)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| anyhow!("ASR thread count must be greater than zero"))?;
        let tokens = load_tokens(&model_dir.join(TOKEN_FILE), contract.blank_id)?;
        init_onnx_runtime();
        let encoder = load_encoder_session(
            &model_dir.join(ENCODER_FILE),
            threads,
            "Nemotron encoder",
            contract,
        )?;
        let decoder = load_session(&model_dir.join(DECODER_FILE), 1, "Nemotron predictor")?;
        let joiner = load_session(&model_dir.join(JOINER_FILE), 1, "Nemotron joiner")?;
        let contract = validate_contract(&encoder, &decoder, &joiner, contract)?;
        Ok(Self {
            encoder,
            decoder,
            joiner,
            contract,
            tokens,
            streams: HashMap::new(),
        })
    }

    fn decode_windows(
        &mut self,
        state: &mut StreamState,
        windows: Vec<NemoStreamingWindow>,
    ) -> Result<()> {
        for window in windows {
            let signal =
                Tensor::from_array((vec![1_i64, 128, window.frames as i64], window.values))?;
            let length = Tensor::from_array((vec![1_i64], vec![window.frames as i64]))?;
            let channel = Tensor::from_array((
                vec![
                    1_i64,
                    CACHE_LAYERS as i64,
                    self.contract.cache_channel_frames as i64,
                    ENCODER_DIM as i64,
                ],
                state.cache_channel.clone(),
            ))?;
            let time = Tensor::from_array((
                vec![
                    1_i64,
                    CACHE_LAYERS as i64,
                    ENCODER_DIM as i64,
                    CACHE_TIME_FRAMES as i64,
                ],
                state.cache_time.clone(),
            ))?;
            let channel_len = Tensor::from_array((vec![1_i64], vec![state.cache_channel_len]))?;
            let outputs = if let Some(prompt_id) = state.prompt_id {
                let prompt = Tensor::from_array((vec![1_i64], vec![prompt_id]))?;
                self.encoder.run(ort::inputs![
                    "audio_signal" => signal,
                    "length" => length,
                    "cache_last_channel" => channel,
                    "cache_last_time" => time,
                    "cache_last_channel_len" => channel_len,
                    "prompt_index" => prompt,
                ])?
            } else {
                self.encoder.run(ort::inputs![
                    "audio_signal" => signal,
                    "length" => length,
                    "cache_last_channel" => channel,
                    "cache_last_time" => time,
                    "cache_last_channel_len" => channel_len,
                ])?
            };
            let (shape, encoded) = outputs
                .get("outputs")
                .ok_or_else(|| anyhow!("Nemotron encoder did not return outputs"))?
                .try_extract_tensor::<f32>()?;
            if shape.len() != 3 || shape[0] != 1 || shape[1] != ENCODER_DIM as i64 {
                bail!("unexpected Nemotron encoder output shape: {shape:?}");
            }
            let all_frames = usize::try_from(shape[2]).context("invalid Nemotron output frames")?;
            let frames = extract_i64(&outputs, "encoded_lengths")?
                .first()
                .copied()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| anyhow!("invalid Nemotron encoded length"))?;
            if frames == 0 || frames > all_frames {
                bail!("Nemotron encoded length {frames} exceeds {all_frames}");
            }
            let mut trimmed = vec![0.0; ENCODER_DIM * frames];
            for feature in 0..ENCODER_DIM {
                trimmed[feature * frames..(feature + 1) * frames]
                    .copy_from_slice(&encoded[feature * all_frames..feature * all_frames + frames]);
            }
            let next_channel = extract_f32(
                &outputs,
                "cache_last_channel_next",
                self.contract.cache_channel_elements(),
            )?;
            let next_time = extract_f32(&outputs, "cache_last_time_next", CACHE_TIME_ELEMENTS)?;
            let next_channel_len = extract_i64(&outputs, "cache_last_channel_next_len")?
                .first()
                .copied()
                .ok_or_else(|| anyhow!("empty Nemotron cache length"))?;
            drop(outputs);
            state.cache_channel = next_channel;
            state.cache_time = next_time;
            state.cache_channel_len = next_channel_len;

            let mut network = OrtRnntNetwork {
                decoder: &mut self.decoder,
                joiner: &mut self.joiner,
                blank_id: self.contract.blank_id,
            };
            state.hypothesis = Some(greedy_rnnt(
                &mut network,
                &trimmed,
                frames,
                ENCODER_DIM,
                self.contract.blank_id,
                MAX_SYMBOLS_PER_FRAME,
                state.hypothesis.take(),
            )?);
        }
        Ok(())
    }

    fn transcript(&self, state: &StreamState) -> Result<AsrTranscript> {
        let Some(hypothesis) = &state.hypothesis else {
            return Ok(AsrTranscript::from_text(""));
        };
        let mut token_texts = Vec::new();
        let mut timestamps = Vec::new();
        for (&id, &frame) in hypothesis.token_ids.iter().zip(&hypothesis.timestamps) {
            let token = self
                .tokens
                .get(id)
                .ok_or_else(|| anyhow!("Nemotron emitted unknown token id {id}"))?;
            if token.starts_with('<') && token.ends_with('>') {
                continue;
            }
            token_texts.push(token.replace('▁', " "));
            timestamps.push(frame as f32 * FRAME_SECONDS);
        }
        // Preserve the host's historical guard: some model exports already
        // report source-relative timestamps, so only compensate a transcript
        // whose first token is plausibly still on the synthetic prefix.
        let leading_padding_sec = state.adapter.leading_padding_samples() as f32 / 16_000.0;
        if leading_padding_sec > 0.0
            && timestamps
                .first()
                .is_some_and(|first| *first >= leading_padding_sec * 0.8)
        {
            for timestamp in &mut timestamps {
                *timestamp = (*timestamp - leading_padding_sec).max(0.0);
            }
        }
        let (text, token_texts) = detokenize_cjk_tokens(&token_texts);
        Ok(AsrTranscript::from_parts(
            text,
            token_texts,
            Some(&timestamps),
            None,
        ))
    }
}

impl AsrEngine for NemotronOrtAsrEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        let session = StreamingSessionId::new(u64::MAX, None);
        self.start_stream(session, AsrStreamConfig::default())?;
        let result = self.push_stream(session, samples);
        if result.is_err() {
            self.cancel_stream(session);
            return result;
        }
        self.finish_stream(session)
    }

    fn start_stream(&mut self, session: StreamingSessionId, config: AsrStreamConfig) -> Result<()> {
        if self.streams.contains_key(&session) {
            bail!("Nemotron stream {session:?} is already started");
        }
        self.streams
            .insert(session, StreamState::new(self.contract, config));
        Ok(())
    }

    fn push_stream(
        &mut self,
        session: StreamingSessionId,
        samples: &[f32],
    ) -> Result<AsrTranscript> {
        let mut state = self
            .streams
            .remove(&session)
            .ok_or_else(|| anyhow!("Nemotron stream {session:?} is not started"))?;
        let result = (|| {
            let windows = state.adapter.push(samples)?;
            self.decode_windows(&mut state, windows)?;
            self.transcript(&state)
        })();
        self.streams.insert(session, state);
        result
    }

    fn finish_stream(&mut self, session: StreamingSessionId) -> Result<AsrTranscript> {
        let state = self
            .streams
            .remove(&session)
            .ok_or_else(|| anyhow!("Nemotron stream {session:?} is not started"))?;
        let mut state = state;
        let windows = state.adapter.finish()?;
        self.decode_windows(&mut state, windows)?;
        self.transcript(&state)
    }

    fn cancel_stream(&mut self, session: StreamingSessionId) {
        self.streams.remove(&session);
    }

    fn cancel_all_streams(&mut self) {
        self.streams.clear();
    }
}

struct OrtRnntNetwork<'a> {
    decoder: &'a mut Session,
    joiner: &'a mut Session,
    blank_id: usize,
}

impl RnntNetwork for OrtRnntNetwork<'_> {
    type State = PredictorState;

    fn initial_state(&self) -> Self::State {
        PredictorState {
            hidden: vec![0.0; PREDICTOR_STATE_ELEMENTS],
            cell: vec![0.0; PREDICTOR_STATE_ELEMENTS],
        }
    }

    fn predictor(&mut self, token: usize, state: &Self::State) -> Result<(Vec<f32>, Self::State)> {
        let targets = Tensor::from_array((
            vec![1_i64, 1],
            vec![i32::try_from(token).context("Nemotron token exceeds i32")?],
        ))?;
        let target_length = Tensor::from_array((vec![1_i64], vec![1_i32]))?;
        let hidden = Tensor::from_array((vec![2_i64, 1, 640], state.hidden.clone()))?;
        let cell = Tensor::from_array((vec![2_i64, 1, 640], state.cell.clone()))?;
        let outputs = self.decoder.run(ort::inputs![
            "targets" => targets,
            "target_length" => target_length,
            "states.1" => hidden,
            "onnx::Slice_3" => cell,
        ])?;
        Ok((
            extract_f32(&outputs, "outputs", PREDICTOR_DIM)?,
            PredictorState {
                hidden: extract_f32(&outputs, "states", PREDICTOR_STATE_ELEMENTS)?,
                cell: extract_f32(&outputs, "162", PREDICTOR_STATE_ELEMENTS)?,
            },
        ))
    }

    fn joiner(&mut self, encoder_frame: &[f32], predictor: &[f32]) -> Result<Vec<f32>> {
        let encoder = Tensor::from_array((vec![1_i64, 1_024, 1], encoder_frame.to_vec()))?;
        let decoder = Tensor::from_array((vec![1_i64, 640, 1], predictor.to_vec()))?;
        let outputs = self.joiner.run(ort::inputs![
            "encoder_outputs" => encoder,
            "decoder_outputs" => decoder,
        ])?;
        extract_f32(&outputs, "outputs", self.blank_id + 1)
    }
}

fn extract_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>> {
    let (_, values) = outputs
        .get(name)
        .ok_or_else(|| anyhow!("Nemotron session did not return {name}"))?
        .try_extract_tensor::<f32>()?;
    if values.len() != elements {
        bail!(
            "Nemotron output {name} has {} elements, expected {elements}",
            values.len()
        );
    }
    Ok(values.to_vec())
}

fn extract_i64(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<Vec<i64>> {
    Ok(outputs
        .get(name)
        .ok_or_else(|| anyhow!("Nemotron session did not return {name}"))?
        .try_extract_tensor::<i64>()?
        .1
        .to_vec())
}

fn load_session(path: &Path, threads: usize, label: &str) -> Result<Session> {
    if !path.is_file() {
        bail!("{label} file not found: {}", path.display());
    }
    Session::builder()
        .map_err(|error| anyhow!("failed to create {label} builder: {error}"))?
        .with_intra_threads(threads)
        .map_err(|error| anyhow!("failed to configure {label} threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow!("failed to configure {label} inter-op threads: {error}"))?
        .commit_from_file(path)
        .map_err(|error| anyhow!("failed to load {label} {}: {error}", path.display()))
}

fn load_encoder_session(
    path: &Path,
    threads: usize,
    label: &str,
    contract: NemotronContract,
) -> Result<Session> {
    if !path.is_file() {
        bail!("{label} file not found: {}", path.display());
    }
    let mut builder = Session::builder()
        .map_err(|error| anyhow!("failed to create {label} builder: {error}"))?
        .with_intra_threads(threads)
        .map_err(|error| anyhow!("failed to configure {label} threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow!("failed to configure {label} inter-op threads: {error}"))?;
    if contract.shift_frames == 56 {
        return builder
            .commit_from_file(path)
            .map_err(|error| anyhow!("failed to load {label} {}: {error}", path.display()));
    }

    // Every latency in a family deliberately shares its 560 ms distribution.
    // Only NeMo's four latency-derived constants and descriptive metadata differ,
    // so patch those values in memory instead of downloading duplicate archives.
    let bytes = fs::read(path).with_context(|| {
        format!(
            "failed to read {label} for latency selection: {}",
            path.display()
        )
    })?;
    let mut model = ModelProto::parse_from_bytes(&bytes)
        .with_context(|| format!("failed to parse {label} ONNX: {}", path.display()))?;
    patch_encoder_latency(&mut model, contract)?;
    let bytes = model
        .write_to_bytes()
        .context("failed to serialize latency-adjusted Nemotron encoder")?;
    builder.commit_from_memory(&bytes).map_err(|error| {
        anyhow!(
            "failed to load latency-adjusted {label} {}: {error}",
            path.display()
        )
    })
}

fn patch_encoder_latency(model: &mut ModelProto, contract: NemotronContract) -> Result<()> {
    let chunk_frames = contract.shift_frames / 8;
    let left_chunks = contract.cache_channel_frames / chunk_frames;
    let graph = model
        .graph
        .as_mut()
        .context("Nemotron encoder ONNX has no graph")?;

    set_constant_input_i32(graph, "/Div", 1, i32::try_from(chunk_frames)?)?;
    set_constant_input_i64(graph, "/LessOrEqual", 1, i64::try_from(left_chunks)?)?;
    set_constant_input_i64(graph, "/Clip_2", 2, i64::try_from(chunk_frames)?)?;
    set_constant_input_i64(graph, "/Slice_6", 2, i64::try_from(chunk_frames)?)?;

    for (key, value) in [
        ("window_size", contract.window_frames.to_string()),
        ("chunk_size_ms", (contract.shift_frames * 10).to_string()),
        ("chunk_shift", contract.shift_frames.to_string()),
    ] {
        let property = model
            .metadata_props
            .iter_mut()
            .find(|property| property.key == key)
            .with_context(|| format!("Nemotron encoder metadata {key} is missing"))?;
        property.value = value;
    }
    Ok(())
}

fn set_constant_input_i32(
    graph: &mut onnx_protobuf::GraphProto,
    consumer_name: &str,
    input_index: usize,
    value: i32,
) -> Result<()> {
    let tensor = constant_input_tensor(graph, consumer_name, input_index)?;
    if tensor.data_type != 6 {
        bail!("Nemotron constant for {consumer_name} is not int32");
    }
    tensor.raw_data = value.to_le_bytes().to_vec();
    tensor.int32_data.clear();
    Ok(())
}

fn set_constant_input_i64(
    graph: &mut onnx_protobuf::GraphProto,
    consumer_name: &str,
    input_index: usize,
    value: i64,
) -> Result<()> {
    let tensor = constant_input_tensor(graph, consumer_name, input_index)?;
    if tensor.data_type != 7 {
        bail!("Nemotron constant for {consumer_name} is not int64");
    }
    tensor.raw_data = value.to_le_bytes().to_vec();
    tensor.int64_data.clear();
    Ok(())
}

fn constant_input_tensor<'a>(
    graph: &'a mut onnx_protobuf::GraphProto,
    consumer_name: &str,
    input_index: usize,
) -> Result<&'a mut TensorProto> {
    let input = graph
        .node
        .iter()
        .find(|node| node.name == consumer_name)
        .with_context(|| format!("Nemotron encoder node {consumer_name} is missing"))?
        .input
        .get(input_index)
        .with_context(|| format!("Nemotron encoder node {consumer_name} input changed"))?
        .clone();
    let producer = graph
        .node
        .iter_mut()
        .find(|node| node.output.iter().any(|output| output == &input))
        .with_context(|| format!("Nemotron constant producer for {consumer_name} is missing"))?;
    if producer.op_type != "Constant" {
        bail!("Nemotron producer for {consumer_name} is not Constant");
    }
    producer
        .attribute
        .iter_mut()
        .find(|attribute| attribute.name == "value")
        .and_then(|attribute| attribute.t.as_mut())
        .with_context(|| format!("Nemotron constant tensor for {consumer_name} is missing"))
}

fn load_tokens(path: &Path, blank_id: usize) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read Nemotron tokens: {}", path.display()))?;
    let mut tokens = vec![None; blank_id + 1];
    for line in contents.lines() {
        let (token, raw_id) = line
            .rsplit_once(' ')
            .ok_or_else(|| anyhow!("invalid Nemotron token line"))?;
        let id = raw_id.parse::<usize>()?;
        let slot = tokens
            .get_mut(id)
            .ok_or_else(|| anyhow!("Nemotron token id {id} exceeds vocabulary"))?;
        if slot.replace(token.to_string()).is_some() {
            bail!("duplicate Nemotron token id {id}");
        }
    }
    if tokens.iter().any(Option::is_none) {
        bail!("Nemotron token table is not contiguous");
    }
    let tokens = tokens.into_iter().flatten().collect::<Vec<_>>();
    if tokens[blank_id] != "<blk>" {
        bail!("Nemotron blank token must be <blk> at id {blank_id}");
    }
    Ok(tokens)
}

fn validate_contract(
    encoder: &Session,
    decoder: &Session,
    joiner: &Session,
    mut contract: NemotronContract,
) -> Result<NemotronContract> {
    let expects_prompt = contract.cache_channel_frames == MULTILINGUAL_CACHE_CHANNEL_FRAMES;
    let expected_encoder_inputs = if expects_prompt { 6 } else { 5 };
    if encoder.inputs().len() != expected_encoder_inputs
        || encoder.outputs().len() != 5
        || decoder.inputs().len() != 4
        || decoder.outputs().len() != 4
        || joiner.inputs().len() != 2
        || joiner.outputs().len() != 1
    {
        bail!("Nemotron ONNX I/O contract changed");
    }
    let metadata = encoder.metadata()?;
    for (key, expected) in [
        ("window_size", contract.window_frames.to_string()),
        ("chunk_shift", contract.shift_frames.to_string()),
        ("subsampling_factor", "8".to_string()),
        ("feat_dim", "128".to_string()),
        ("pred_hidden", "640".to_string()),
        ("cache_last_channel_dim1", CACHE_LAYERS.to_string()),
        (
            "cache_last_channel_dim2",
            contract.cache_channel_frames.to_string(),
        ),
        ("cache_last_channel_dim3", ENCODER_DIM.to_string()),
        ("cache_last_time_dim1", CACHE_LAYERS.to_string()),
        ("cache_last_time_dim2", ENCODER_DIM.to_string()),
        ("cache_last_time_dim3", CACHE_TIME_FRAMES.to_string()),
    ] {
        let actual = metadata.custom(key);
        if actual.as_deref() != Some(expected.as_str()) {
            bail!("Nemotron metadata {key} changed: expected {expected}, got {actual:?}");
        }
    }
    if expects_prompt {
        let auto = metadata
            .custom("auto_prompt_id")
            .context("Nemotron auto prompt metadata is missing")?
            .parse::<i64>()
            .context("Nemotron auto prompt metadata is invalid")?;
        let prompt_dictionary = metadata
            .custom("prompt_dictionary")
            .context("Nemotron prompt dictionary metadata is missing")?;
        let prompt_dictionary = serde_json::from_str::<HashMap<String, i64>>(&prompt_dictionary)
            .context("Nemotron prompt dictionary metadata is invalid")?;
        let dictionary_auto = prompt_dictionary
            .get("auto")
            .copied()
            .context("Nemotron prompt dictionary has no auto entry")?;
        if auto != dictionary_auto {
            bail!("Nemotron auto prompt metadata disagrees with its prompt dictionary");
        }
        let japanese = ["ja", "ja-JP", "ja-JA"]
            .into_iter()
            .find_map(|key| prompt_dictionary.get(key).copied())
            .context("Nemotron prompt dictionary has no Japanese entry")?;
        contract.prompt_ids = Some(NemotronPromptIds { auto, japanese });
    }
    Ok(contract)
}

fn detokenize_cjk_tokens(token_texts: &[String]) -> (String, Vec<String>) {
    let chars = token_texts
        .iter()
        .enumerate()
        .flat_map(|(token_index, text)| text.chars().map(move |character| (token_index, character)))
        .collect::<Vec<_>>();
    let mut normalized = vec![String::new(); token_texts.len()];
    for (index, &(token_index, character)) in chars.iter().enumerate() {
        let remove = character == ' '
            && index > 0
            && index + 1 < chars.len()
            && is_cjk(chars[index - 1].1)
            && is_cjk(chars[index + 1].1);
        if !remove {
            normalized[token_index].push(character);
        }
    }
    (normalized.concat(), normalized)
}

fn is_cjk(character: char) -> bool {
    matches!(character as u32, 0x3040..=0x30ff | 0x3400..=0x9fff | 0xac00..=0xd7af)
}

#[cfg(test)]
mod tests {
    use crate::{AsrLanguage, AsrModel, AsrTranscript};

    use super::{
        ENGLISH_CACHE_CHANNEL_FRAMES, MULTILINGUAL_CACHE_CHANNEL_FRAMES, NemotronContract,
        NemotronPromptIds, detokenize_cjk_tokens,
    };

    #[test]
    fn every_nemotron_family_supports_the_full_80_to_1120ms_latency_grid() {
        let cases = [
            (
                AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8,
                17,
                8,
                false,
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
                25,
                16,
                false,
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8,
                41,
                32,
                false,
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
                65,
                56,
                false,
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8,
                121,
                112,
                false,
            ),
            (AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8, 17, 8, true),
            (AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8, 25, 16, true),
            (AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8, 41, 32, true),
            (AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8, 65, 56, true),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8,
                121,
                112,
                true,
            ),
        ];

        for (model, window_frames, shift_frames, multilingual) in cases {
            let contract = NemotronContract::for_model(model).unwrap();
            assert_eq!(
                (contract.window_frames, contract.shift_frames),
                (window_frames, shift_frames)
            );
            assert_eq!(
                contract.cache_channel_frames,
                if multilingual {
                    MULTILINGUAL_CACHE_CHANNEL_FRAMES
                } else {
                    ENGLISH_CACHE_CHANNEL_FRAMES
                }
            );
        }
    }

    #[test]
    fn multilingual_prompt_contract_uses_japanese_hint_and_defaults_to_auto() {
        let mut contract =
            NemotronContract::for_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8).unwrap();
        contract.prompt_ids = Some(NemotronPromptIds {
            auto: 101,
            japanese: 10,
        });

        assert_eq!(contract.prompt_id(None), Some(101));
        assert_eq!(contract.prompt_id(Some(AsrLanguage::Japanese)), Some(10));
        assert_eq!(
            contract.prompt_id(Some(AsrLanguage::English)),
            Some(101),
            "only an explicitly requested Japanese completion route overrides auto detection"
        );
    }

    #[test]
    fn cjk_detokenization_removes_only_internal_cjk_spaces() {
        let (text, _) = detokenize_cjk_tokens(&[" う ち no 中 学 ".to_string()]);

        assert_eq!(text, " うち no 中学 ");
    }

    #[test]
    fn cjk_detokenization_keeps_token_ranges_inside_the_final_text() {
        let token_texts = vec![" う".to_string(), " ち".to_string()];
        let (text, token_texts) = detokenize_cjk_tokens(&token_texts);
        let transcript = AsrTranscript::from_parts(text, token_texts, None, None);

        assert_eq!(transcript.text, "うち");
        assert_eq!(transcript.tokens[0].char_range, Some(0..1));
        assert_eq!(transcript.tokens[1].char_range, Some(1..2));
    }
}
