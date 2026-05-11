pub(crate) enum PlaybackEvent {
    Finished {
        request_id: String,
        elapsed_millis: u128,
    },
    Failed {
        request_id: String,
        elapsed_millis: u128,
        error: String,
    },
}
