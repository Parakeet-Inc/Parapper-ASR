mod buffer;
mod config;
mod facade;

pub(crate) use facade::{SegmentBuilder, SegmentBuilderEvent, SegmentCloseReason};

#[cfg(test)]
mod tests;
