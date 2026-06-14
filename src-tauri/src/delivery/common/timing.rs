use crate::{
    config::{NeoSendTiming, ParapperConfig},
    delivery::RecognizedTextOutput,
};

pub(crate) fn speech_timing_allows(is_final: bool) -> bool {
    is_final
}

pub(crate) fn translation_timing_allows(config: &ParapperConfig, is_final: bool) -> bool {
    config.translation.send_timing == NeoSendTiming::Interim || is_final
}

pub(crate) fn translation_timing_allows_output(
    config: &ParapperConfig,
    output: &RecognizedTextOutput,
) -> bool {
    translation_timing_allows(config, output.meta.is_final())
}
