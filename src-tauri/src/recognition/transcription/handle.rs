use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{SyncSender, TrySendError, sync_channel},
    },
    thread::{self, JoinHandle},
};

use anyhow::{Context, Result};
use tauri::AppHandle;

use super::{job::AsrJob, worker_runtime::run_asr_worker};
use crate::{config::ParapperConfig, recognition::segment_builder::SegmentCloseReason};

pub(crate) struct AsrWorker {
    sender: Option<SyncSender<AsrJob>>,
    segment_activity_epoch: Arc<AtomicU64>,
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

impl AsrWorker {
    pub(crate) fn start(
        handle: AppHandle,
        config: ParapperConfig,
        runtime_config: &Arc<RwLock<ParapperConfig>>,
    ) -> Result<Self> {
        let (sender, receiver) = sync_channel(4);
        let segment_activity_epoch = Arc::new(AtomicU64::new(0));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_activity_epoch = segment_activity_epoch.clone();
        let worker_stop = stop_requested.clone();
        let worker_config = runtime_config.clone();
        let join_handle = thread::Builder::new()
            .name("parapper-asr".to_string())
            .spawn(move || {
                run_asr_worker(
                    &handle,
                    &config,
                    &worker_config,
                    &receiver,
                    &worker_activity_epoch,
                    &worker_stop,
                );
            })
            .context("Failed to spawn ASR worker")?;

        Ok(Self {
            sender: Some(sender),
            segment_activity_epoch,
            stop_requested,
            join_handle: Some(join_handle),
        })
    }

    pub(crate) fn send_segment_closed(
        &self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: Vec<f32>,
        reason: SegmentCloseReason,
    ) {
        self.send_required(AsrJob::SegmentClosed {
            segment_id,
            previous_segment_id,
            full_audio,
            reason,
        });
    }

    pub(crate) fn send_segment_activity(&self) {
        self.segment_activity_epoch.fetch_add(1, Ordering::Release);
    }

    #[expect(
        clippy::unused_self,
        reason = "RecognitionPipeline からの periodic hook として ASR handle API を揃えている"
    )]
    pub(crate) fn tick(&self) {}

    pub(crate) fn send_turn_check_silence_reached(&self, previous_segment_id: u64) {
        self.send_required(AsrJob::TurnCheckSilenceReached {
            previous_segment_id,
        });
    }

    fn send_required(&self, job: AsrJob) {
        let Some(sender) = self.sender.as_ref() else {
            return;
        };
        match sender.try_send(job) {
            Ok(()) => {}
            Err(TrySendError::Full(job)) => {
                let sender = sender.clone();
                if let Err(err) = thread::Builder::new()
                    .name("parapper-asr-required-send".to_string())
                    .spawn(move || {
                        if let Err(err) = sender.send(job) {
                            log::warn!("Dropping required ASR job because worker stopped: {err}");
                        }
                    })
                {
                    log::warn!("Failed to queue required ASR job on helper thread: {err}");
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                log::warn!("Dropping required ASR job because worker stopped");
            }
        }
    }

    pub(crate) fn stop_inner(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        self.sender.take();
        if let Some(join_handle) = self.join_handle.take() {
            if let Err(err) = join_handle.join() {
                log::warn!("ASR worker thread panicked: {err:?}");
            }
        }
    }
}

impl Drop for AsrWorker {
    fn drop(&mut self) {
        self.stop_inner();
    }
}
