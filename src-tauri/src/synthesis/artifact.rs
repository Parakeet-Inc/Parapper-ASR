use crate::playback::{PlaybackEvent, PlaybackRequest};

pub(super) struct TtsArtifact {
    pub(super) request_id: String,
    pub(super) samples: Vec<f32>,
    pub(super) sample_rate: i32,
    pub(super) volume: f32,
    pub(super) output_device_host: Option<String>,
    pub(super) output_device_id: Option<String>,
}

impl TtsArtifact {
    pub(super) fn into_playback_request(
        self,
        on_finished: Box<dyn FnOnce(PlaybackEvent) + Send>,
    ) -> PlaybackRequest {
        PlaybackRequest::new(
            self.request_id,
            self.samples,
            self.sample_rate,
            self.volume,
            self.output_device_host,
            self.output_device_id,
            on_finished,
        )
    }
}
