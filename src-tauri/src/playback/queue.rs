use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

use super::manager::PlaybackRequest;

pub(super) struct PlaybackQueue {
    state: Mutex<VecDeque<PlaybackRequest>>,
    ready: Condvar,
}

impl PlaybackQueue {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
        }
    }

    pub(super) fn push(&self, request: PlaybackRequest) {
        self.state
            .lock()
            .expect("playback queue lock poisoned")
            .push_back(request);
        self.ready.notify_one();
    }

    pub(super) fn pop_blocking(&self) -> PlaybackRequest {
        let mut state = self.state.lock().expect("playback queue lock poisoned");
        while state.is_empty() {
            state = self
                .ready
                .wait(state)
                .expect("playback queue lock poisoned");
        }
        state.pop_front().expect("playback request")
    }
}
