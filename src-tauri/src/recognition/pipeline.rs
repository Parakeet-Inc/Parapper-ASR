use std::sync::{Arc, RwLock};

use anyhow::Result;
use tauri::{AppHandle, Emitter};

use super::{
    events::{VadState, VadStateEvent},
    segmenter::PhraseSegmenter,
    worker::AsrWorker,
};
use crate::{
    config::ParapperConfig,
    model::{OnnxRuntimeSileroVadEngine, VadEngine, vad_model_path},
};

pub struct RecognitionPipeline {
    handle: AppHandle,
    segmenter: PhraseSegmenter,
    vad: Box<dyn VadEngine>,
    asr_worker: AsrWorker,
}

impl RecognitionPipeline {
    pub fn new(
        handle: AppHandle,
        config: &ParapperConfig,
        runtime_config: &Arc<RwLock<ParapperConfig>>,
    ) -> Result<Self> {
        let vad_path = vad_model_path(&handle)?;
        let vad = OnnxRuntimeSileroVadEngine::new(&vad_path, config.vad_threshold)?;
        let asr_worker = AsrWorker::start(handle.clone(), config.clone(), runtime_config)?;

        Ok(Self {
            handle,
            segmenter: PhraseSegmenter::new(config),
            vad: Box::new(vad),
            asr_worker,
        })
    }

    pub fn process_chunk(&mut self, samples: &[f32]) -> Result<()> {
        let vad_result = self.vad.process(samples)?;
        let state = if vad_result.is_speech {
            VadState::Speech
        } else {
            VadState::Silence
        };
        let _ = self.handle.emit(
            "parapper://vad-state",
            VadStateEvent {
                state,
                probability: vad_result.probability,
            },
        );

        if let Some(phrase) = self.segmenter.push(samples, vad_result) {
            self.asr_worker.send(phrase);
        }

        Ok(())
    }

    pub fn update_config(&mut self, config: &ParapperConfig) {
        self.segmenter.update_config(config);
        self.vad.set_threshold(config.vad_threshold);
    }

    pub fn stop(mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        self.asr_worker.stop_inner();
    }
}

impl Drop for RecognitionPipeline {
    fn drop(&mut self) {
        self.stop_inner();
    }
}
