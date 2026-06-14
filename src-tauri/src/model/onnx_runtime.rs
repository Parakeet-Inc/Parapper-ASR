use std::sync::OnceLock;

use anyhow::{Result, anyhow};
use ort::execution_providers::CPUExecutionProvider;

static ORT_INIT: OnceLock<Result<()>> = OnceLock::new();

pub(crate) fn init_onnx_runtime() -> Result<()> {
    match ORT_INIT.get_or_init(|| {
        let initialized = ort::init()
            .with_name("parapper")
            .with_telemetry(false)
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()
            .then_some(());
        initialized.ok_or_else(|| anyhow!("Failed to initialize ONNX Runtime"))
    }) {
        Ok(()) => Ok(()),
        Err(err) => Err(anyhow!("{err:#}")),
    }
}
