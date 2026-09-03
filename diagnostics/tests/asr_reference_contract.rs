use std::{collections::BTreeSet, fs, path::PathBuf};

use parapper_models::asr::{AsrModel, AsrPrecision};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ReferenceManifest {
    schema_version: u32,
    nemo: NemoContract,
    models: Vec<ModelContract>,
}

#[derive(Debug, Deserialize)]
struct NemoContract {
    commit: String,
    source_files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelContract {
    app_model: String,
    precisions: Vec<String>,
    oracle: String,
    reference: ArtifactReference,
    production_artifact: ProductionArtifact,
}

#[derive(Debug, Deserialize)]
struct ArtifactReference {
    revision: String,
}

#[derive(Debug, Deserialize)]
struct ProductionArtifact {
    kind: String,
    revision: String,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RuntimeLock {
    schema_version: u32,
    policy: String,
    rust: RustRuntime,
    native_onnx_runtime: NativeRuntime,
}

#[derive(Debug, Deserialize)]
struct RustRuntime {
    toolchain: String,
    ort: LockedCrate,
    ort_sys: LockedCrate,
}

#[derive(Debug, Deserialize)]
struct LockedCrate {
    version: String,
    checksum: String,
}

#[derive(Debug, Deserialize)]
struct NativeRuntime {
    version: String,
    provider: String,
    source: String,
    windows_x64: WindowsRuntime,
    macos_arm64: ArchiveRuntime,
}

#[derive(Debug, Deserialize)]
struct WindowsRuntime {
    archive_url: String,
    archive_size: u64,
    archive_sha256: String,
    library: String,
    library_size: u64,
    library_sha256: String,
    providers_shared_library: String,
    providers_shared_library_size: u64,
    providers_shared_library_sha256: String,
}

#[derive(Debug, Deserialize)]
#[allow(
    clippy::struct_field_names,
    reason = "field names mirror the checked-in runtime manifest schema"
)]
struct ArchiveRuntime {
    archive_url: String,
    archive_size: u64,
    archive_sha256: String,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("diagnostics crate must be inside the workspace")
        .to_path_buf()
}

fn read_json<T: for<'de> Deserialize<'de>>(relative: &str) -> T {
    let path = workspace_root().join(relative);
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&contents)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn serialized_model(model: AsrModel) -> String {
    serde_json::to_value(model)
        .expect("ASR model must serialize")
        .as_str()
        .expect("ASR model must serialize as a string")
        .to_owned()
}

fn serialized_precision(precision: AsrPrecision) -> String {
    serde_json::to_value(precision)
        .expect("ASR precision must serialize")
        .as_str()
        .expect("ASR precision must serialize as a string")
        .to_owned()
}

#[test]
fn reference_manifest_covers_the_product_asr_matrix_and_keeps_legacy_nemotron() {
    let manifest: ReferenceManifest =
        read_json("diagnostics/nemo-reference/reference-manifest.json");
    assert_eq!(manifest.schema_version, 1);

    let product_models = [
        AsrModel::ReazonSpeechK2V2,
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
        AsrModel::NemoParakeetTdt0_6BV3Int8,
        AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
        AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
        AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
    ];
    let expected: BTreeSet<_> = product_models
        .iter()
        .copied()
        .map(serialized_model)
        .collect();
    let actual: BTreeSet<_> = manifest
        .models
        .iter()
        .map(|model| model.app_model.clone())
        .collect();
    assert_eq!(actual, expected);

    for model in product_models.iter().copied() {
        let contract = manifest
            .models
            .iter()
            .find(|contract| contract.app_model == serialized_model(model))
            .expect("every product model must have a reference contract");
        let expected_precisions: BTreeSet<_> = [
            AsrPrecision::Int8,
            AsrPrecision::Int8Float32,
            AsrPrecision::Float32,
        ]
        .into_iter()
        .filter(|precision| model.supports_precision(*precision))
        .map(serialized_precision)
        .collect();
        assert_eq!(
            contract.precisions.iter().cloned().collect::<BTreeSet<_>>(),
            expected_precisions,
            "precision contract drifted for {}",
            contract.app_model
        );
    }

    assert!(actual.contains("nemotron_speech_streaming_en_0_6b_160ms_int8"));
    assert!(actual.contains("nemotron_speech_streaming_en_0_6b_560ms_int8"));
}

#[test]
fn nvidia_references_and_production_artifacts_are_immutable() {
    let manifest: ReferenceManifest =
        read_json("diagnostics/nemo-reference/reference-manifest.json");
    assert!(full_sha(&manifest.nemo.commit));
    assert_eq!(manifest.nemo.source_files.len(), 3);

    for model in &manifest.models {
        assert!(full_sha(&model.reference.revision));
        if model.oracle == "pinned_nvidia_python" {
            assert_ne!(model.app_model, "reazonspeech_k2_v2");
        }
        match model.production_artifact.kind.as_str() {
            "huggingface_snapshot" => assert!(full_sha(&model.production_artifact.revision)),
            "github_release_asset" => assert!(
                model
                    .production_artifact
                    .sha256
                    .as_deref()
                    .is_some_and(full_sha256)
            ),
            other => panic!("unknown production artifact kind: {other}"),
        }
    }
}

#[test]
fn runtime_lock_matches_the_workspace_and_native_bundle() {
    let runtime: RuntimeLock = read_json("diagnostics/nemo-reference/runtime-lock.json");
    assert_eq!(runtime.schema_version, 1);
    assert_eq!(runtime.policy, "fixed_for_v0_5");
    assert_eq!(runtime.rust.toolchain, "1.97.1");
    assert_eq!(runtime.rust.ort.version, "2.0.0-rc.12");
    assert_eq!(runtime.rust.ort_sys.version, "2.0.0-rc.12");
    assert!(full_sha256(&runtime.rust.ort.checksum));
    assert!(full_sha256(&runtime.rust.ort_sys.checksum));
    assert_eq!(runtime.native_onnx_runtime.version, "1.24.4");
    assert_eq!(runtime.native_onnx_runtime.provider, "cpu");
    assert_eq!(
        runtime.native_onnx_runtime.source,
        "Microsoft ONNX Runtime v1.24.4 official release assets"
    );
    assert_eq!(
        runtime.native_onnx_runtime.windows_x64.archive_size,
        74_442_783
    );
    assert!(
        runtime
            .native_onnx_runtime
            .windows_x64
            .archive_url
            .contains("github.com/microsoft/onnxruntime/releases/download/v1.24.4/")
    );
    assert!(full_sha256(
        &runtime.native_onnx_runtime.windows_x64.archive_sha256
    ));
    assert_eq!(
        runtime.native_onnx_runtime.windows_x64.library,
        "onnxruntime.dll"
    );
    assert_eq!(
        runtime.native_onnx_runtime.windows_x64.library_size,
        14_203_464
    );
    assert!(full_sha256(
        &runtime.native_onnx_runtime.windows_x64.library_sha256
    ));
    assert_eq!(
        runtime
            .native_onnx_runtime
            .windows_x64
            .providers_shared_library,
        "onnxruntime_providers_shared.dll"
    );
    assert_eq!(
        runtime
            .native_onnx_runtime
            .windows_x64
            .providers_shared_library_size,
        22_088
    );
    assert!(full_sha256(
        &runtime
            .native_onnx_runtime
            .windows_x64
            .providers_shared_library_sha256
    ));
    assert_eq!(
        runtime.native_onnx_runtime.macos_arm64.archive_size,
        30_937_282
    );
    assert!(
        runtime
            .native_onnx_runtime
            .macos_arm64
            .archive_url
            .contains("github.com/microsoft/onnxruntime/releases/download/v1.24.4/")
    );
    assert!(full_sha256(
        &runtime.native_onnx_runtime.macos_arm64.archive_sha256
    ));

    let workspace = workspace_root();
    let tauri_manifest = fs::read_to_string(workspace.join("src-tauri/Cargo.toml"))
        .expect("src-tauri Cargo.toml must be readable");
    assert!(tauri_manifest.contains("ort = { version = \"=2.0.0-rc.12\""));
    assert!(!tauri_manifest.contains("sherpa-onnx"));
    let workspace_lock = fs::read_to_string(workspace.join("Cargo.lock"))
        .expect("workspace Cargo.lock must be readable");
    assert!(!workspace_lock.contains("name = \"sherpa-onnx\""));
    assert!(!workspace_lock.contains("name = \"sherpa-onnx-sys\""));
    let build_script = fs::read_to_string(workspace.join("src-tauri/build.rs"))
        .expect("build.rs must be readable");
    assert!(build_script.contains("libonnxruntime.1.24.4.dylib"));
    let build_workflow = fs::read_to_string(workspace.join(".github/workflows/build.yml"))
        .expect("build workflow must be readable");
    assert!(build_workflow.contains(&runtime.native_onnx_runtime.windows_x64.archive_sha256));
    assert!(build_workflow.contains(&runtime.native_onnx_runtime.windows_x64.library_sha256));
    assert!(
        build_workflow.contains(
            &runtime
                .native_onnx_runtime
                .windows_x64
                .providers_shared_library_sha256
        )
    );
}

fn full_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn full_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
