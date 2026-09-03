use std::{env, path::PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ort::{execution_providers::CPUExecutionProvider, session::Session};
use serde::Serialize;

#[derive(Serialize)]
struct ModelContract {
    path: String,
    inputs: Vec<OutletContract>,
    outputs: Vec<OutletContract>,
    metadata: ModelMetadataContract,
}

#[derive(Serialize)]
struct OutletContract {
    name: String,
    value_type: String,
}

#[derive(Serialize)]
struct ModelMetadataContract {
    name: Option<String>,
    producer: Option<String>,
    domain: Option<String>,
    version: Option<i64>,
    description: Option<String>,
    graph_description: Option<String>,
    custom: Vec<(String, Option<String>)>,
}

fn main() -> Result<()> {
    let paths = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty() {
        bail!("usage: inspect_onnx_contract <model.onnx> [model.onnx ...]");
    }

    let initialized = ort::init()
        .with_name("inspect_onnx_contract")
        .with_telemetry(false)
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit();
    if !initialized {
        bail!("failed to initialize ONNX Runtime");
    }

    let contracts = paths
        .iter()
        .map(inspect_model)
        .collect::<Result<Vec<_>>>()?;
    println!("{}", serde_json::to_string_pretty(&contracts)?);
    Ok(())
}

fn inspect_model(path: &PathBuf) -> Result<ModelContract> {
    let session = Session::builder()
        .context("failed to create ONNX session builder")?
        .with_intra_threads(1)
        .map_err(|error| anyhow!("failed to configure ONNX intra-op threads: {error}"))?
        .with_inter_threads(1)
        .map_err(|error| anyhow!("failed to configure ONNX inter-op threads: {error}"))?
        .commit_from_file(path)
        .with_context(|| format!("failed to load {}", path.display()))?;

    let inputs = session
        .inputs()
        .iter()
        .map(|outlet| OutletContract {
            name: outlet.name().to_string(),
            value_type: outlet.dtype().to_string(),
        })
        .collect();
    let outputs = session
        .outputs()
        .iter()
        .map(|outlet| OutletContract {
            name: outlet.name().to_string(),
            value_type: outlet.dtype().to_string(),
        })
        .collect();

    let metadata = session
        .metadata()
        .context("failed to read model metadata")?;
    let custom = metadata
        .custom_keys()
        .context("failed to enumerate custom metadata")?
        .into_iter()
        .map(|key| {
            let value = metadata.custom(&key);
            (key, value)
        })
        .collect();

    Ok(ModelContract {
        path: path.display().to_string(),
        inputs,
        outputs,
        metadata: ModelMetadataContract {
            name: metadata.name(),
            producer: metadata.producer(),
            domain: metadata.domain(),
            version: metadata.version(),
            description: metadata.description(),
            graph_description: metadata.graph_description(),
            custom,
        },
    })
}
