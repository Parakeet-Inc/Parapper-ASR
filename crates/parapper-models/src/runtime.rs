use std::sync::OnceLock;

use ort::execution_providers::CPUExecutionProvider;

static ORT_INIT: OnceLock<()> = OnceLock::new();

/// Initializes the process-wide ONNX Runtime environment once.
///
/// This operation is idempotent, including when another library constructor
/// initialized the process-wide ONNX Runtime environment first.
pub fn init_onnx_runtime() {
    ORT_INIT.get_or_init(|| {
        let _ = ort::init()
            .with_name("parapper")
            .with_telemetry(false)
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit();
    });
}

#[cfg(test)]
mod tests {
    use super::init_onnx_runtime;

    #[test]
    fn shared_initializer_accepts_runtime_initialized_by_an_asr_constructor() {
        let _ = ort::init()
            .with_name("direct-asr-constructor")
            .with_telemetry(false)
            .commit();

        init_onnx_runtime();
    }
}
