use std::collections::VecDeque;

use crate::{config::ParapperConfig, model::VadResult};

const MAX_PHRASE_MILLIS: u32 = 25_000;

pub(crate) struct PhraseSegmenter {
    pause_threshold: u32,
    phrase_threshold: u32,
    max_chunks: u32,
    silence_counter: u32,
    speech_counter: u32,
    audio: Vec<f32>,
    pending_silence_audio: Vec<f32>,
    pre_speech_chunks: VecDeque<Vec<f32>>,
    pre_speech_max_chunks: usize,
}

impl PhraseSegmenter {
    pub(crate) fn new(config: &ParapperConfig) -> Self {
        Self {
            pause_threshold: config.pause_threshold,
            phrase_threshold: config.phrase_threshold,
            max_chunks: max_chunks_for_interval(config.vad_interval_ms),
            silence_counter: 0,
            speech_counter: 0,
            audio: Vec::new(),
            pending_silence_audio: Vec::new(),
            pre_speech_chunks: VecDeque::new(),
            pre_speech_max_chunks: config.pause_threshold as usize,
        }
    }

    pub(crate) fn push(&mut self, samples: &[f32], vad_result: VadResult) -> Option<Vec<f32>> {
        if vad_result.is_speech {
            if self.speech_counter == 0 {
                self.audio.clear();
                self.pending_silence_audio.clear();
                self.drain_pre_speech_chunks();
            } else if self.silence_counter > 0 {
                self.audio.append(&mut self.pending_silence_audio);
            }
            self.audio.extend_from_slice(samples);
            self.speech_counter += 1;
            self.silence_counter = 0;
            return None;
        }

        if self.speech_counter == 0 {
            self.push_pre_speech_chunk(samples);
            return None;
        }

        self.silence_counter += 1;
        self.pending_silence_audio.extend_from_slice(samples);
        let current_chunks = self.speech_counter + self.silence_counter;
        if self.silence_counter < self.pause_threshold && current_chunks <= self.max_chunks {
            return None;
        }

        let phrase = if self.speech_counter >= self.phrase_threshold {
            self.audio.append(&mut self.pending_silence_audio);
            Some(std::mem::take(&mut self.audio))
        } else {
            self.audio.clear();
            None
        };
        self.speech_counter = 0;
        self.silence_counter = 0;
        self.pending_silence_audio.clear();
        self.push_pre_speech_chunk(samples);
        phrase
    }

    pub(crate) fn update_config(&mut self, config: &ParapperConfig) {
        self.pause_threshold = config.pause_threshold;
        self.phrase_threshold = config.phrase_threshold;
        self.max_chunks = max_chunks_for_interval(config.vad_interval_ms);
        self.pre_speech_max_chunks = config.pause_threshold as usize;
        while self.pre_speech_chunks.len() > self.pre_speech_max_chunks {
            self.pre_speech_chunks.pop_front();
        }
    }

    fn push_pre_speech_chunk(&mut self, samples: &[f32]) {
        self.pre_speech_chunks.push_back(samples.to_vec());
        while self.pre_speech_chunks.len() > self.pre_speech_max_chunks {
            self.pre_speech_chunks.pop_front();
        }
    }

    fn drain_pre_speech_chunks(&mut self) {
        for chunk in self.pre_speech_chunks.drain(..) {
            self.audio.extend_from_slice(&chunk);
        }
    }
}

fn max_chunks_for_interval(interval_ms: u32) -> u32 {
    MAX_PHRASE_MILLIS.div_ceil(interval_ms.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::PhraseSegmenter;
    use crate::{config::ParapperConfig, model::VadResult};

    #[test]
    fn phrase_segmenter_emits_after_speech_and_pause() {
        let config = ParapperConfig {
            pause_threshold: 2,
            phrase_threshold: 1,
            ..ParapperConfig::default()
        };
        let mut segmenter = PhraseSegmenter::new(&config);
        let speech = VadResult {
            probability: 0.9,
            is_speech: true,
        };
        let silence = VadResult {
            probability: 0.0,
            is_speech: false,
        };

        assert!(segmenter.push(&[1.0, 1.0], speech).is_none());
        assert!(segmenter.push(&[1.0, 1.0], speech).is_none());
        assert!(segmenter.push(&[0.0, 0.0], silence).is_none());
        let phrase = segmenter.push(&[0.0, 0.0], silence).unwrap();

        assert_eq!(phrase.len(), 8);
    }

    #[test]
    fn phrase_segmenter_uses_same_silence_amount_at_start_and_end() {
        let config = ParapperConfig {
            pause_threshold: 2,
            phrase_threshold: 2,
            ..ParapperConfig::default()
        };
        let mut segmenter = PhraseSegmenter::new(&config);
        let speech = VadResult {
            probability: 0.9,
            is_speech: true,
        };
        let silence = VadResult {
            probability: 0.0,
            is_speech: false,
        };

        assert!(segmenter.push(&[10.0], silence).is_none());
        assert!(segmenter.push(&[20.0], silence).is_none());
        assert!(segmenter.push(&[30.0], silence).is_none());
        assert!(segmenter.push(&[1.0], speech).is_none());
        assert!(segmenter.push(&[2.0], speech).is_none());
        assert!(segmenter.push(&[3.0], speech).is_none());
        assert!(segmenter.push(&[0.0], silence).is_none());
        let phrase = segmenter.push(&[0.0], silence).unwrap();

        assert_eq!(phrase, vec![20.0, 30.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn phrase_segmenter_keeps_short_silence_when_speech_resumes() {
        let config = ParapperConfig {
            pause_threshold: 2,
            phrase_threshold: 1,
            ..ParapperConfig::default()
        };
        let mut segmenter = PhraseSegmenter::new(&config);
        let speech = VadResult {
            probability: 0.9,
            is_speech: true,
        };
        let silence = VadResult {
            probability: 0.0,
            is_speech: false,
        };

        assert!(segmenter.push(&[1.0], speech).is_none());
        assert!(segmenter.push(&[0.1], silence).is_none());
        assert!(segmenter.push(&[2.0], speech).is_none());
        assert!(segmenter.push(&[0.0], silence).is_none());
        let phrase = segmenter.push(&[0.0], silence).unwrap();

        assert_eq!(phrase, vec![1.0, 0.1, 2.0, 0.0, 0.0]);
    }
}
