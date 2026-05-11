use std::{
    sync::{Arc, OnceLock},
    thread,
    time::Instant,
};

use crate::{audio::play_mono_samples, playback::PlaybackEvent};

use super::queue::PlaybackQueue;

pub(crate) struct PlaybackRequest {
    request_id: String,
    samples: Vec<f32>,
    sample_rate: i32,
    volume: f32,
    output_device_host: Option<String>,
    output_device_id: Option<String>,
    on_finished: Box<dyn FnOnce(PlaybackEvent) + Send>,
}

pub(crate) struct PlaybackManager {
    queue: Arc<PlaybackQueue>,
}

static PLAYBACK_MANAGER: OnceLock<PlaybackManager> = OnceLock::new();

impl PlaybackRequest {
    pub(crate) fn new(
        request_id: String,
        samples: Vec<f32>,
        sample_rate: i32,
        volume: f32,
        output_device_host: Option<String>,
        output_device_id: Option<String>,
        on_finished: Box<dyn FnOnce(PlaybackEvent) + Send>,
    ) -> Self {
        Self {
            request_id,
            samples,
            sample_rate,
            volume,
            output_device_host,
            output_device_id,
            on_finished,
        }
    }
}

impl PlaybackManager {
    pub(crate) fn global() -> &'static Self {
        PLAYBACK_MANAGER.get_or_init(|| {
            let manager = Self {
                queue: Arc::new(PlaybackQueue::new()),
            };
            manager.start_worker();
            manager
        })
    }

    pub(crate) fn submit(&self, request: PlaybackRequest) {
        self.queue.push(request);
    }

    fn start_worker(&self) {
        let queue = Arc::clone(&self.queue);
        if let Err(err) = thread::Builder::new()
            .name("parapper-playback".to_string())
            .spawn(move || {
                loop {
                    let request = queue.pop_blocking();
                    log::info!("Playback start id={}", request.request_id);
                    let started_at = Instant::now();
                    let result = match play_mono_samples(
                        &request.samples,
                        request.sample_rate,
                        request.volume,
                        request.output_device_host.as_deref(),
                        request.output_device_id.as_deref(),
                    ) {
                        Ok(()) => PlaybackEvent::Finished {
                            request_id: request.request_id.clone(),
                            elapsed_millis: started_at.elapsed().as_millis(),
                        },
                        Err(err) => PlaybackEvent::Failed {
                            request_id: request.request_id.clone(),
                            elapsed_millis: started_at.elapsed().as_millis(),
                            error: err.to_string(),
                        },
                    };
                    (request.on_finished)(result);
                }
            })
        {
            log::warn!("Failed to spawn playback worker: {err}");
        }
    }
}
