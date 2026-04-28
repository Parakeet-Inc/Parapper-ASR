use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tokio::{fs::File, io::AsyncWriteExt};

use crate::{
    config::{AsrModel, ParapperConfig},
    model::SherpaOnnxTransducerModelFiles,
};

const VAD_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/refs/tags/v6.0/src/silero_vad/data/silero_vad.onnx";
const ASR_MODEL_BASE_URL: &str =
    "https://huggingface.co/reazon-research/reazonspeech-k2-v2/resolve/main";
const ASR_MODEL_DIR_NAME_JA: &str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01";
const ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V2_INT8: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main";
const ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V2_INT8: &str =
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
const ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V3_INT8: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main";
const ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V3_INT8: &str =
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub root_dir: String,
    pub vad: ModelAssetStatus,
    pub asr: ModelAssetStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAssetStatus {
    pub installed: bool,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadProgress {
    pub file_name: String,
    pub file_index: usize,
    pub total_files: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub progress: f64,
    pub finished: bool,
}

struct DownloadTarget {
    url: String,
    output_path: PathBuf,
    file_name: String,
}

pub fn models_root(handle: &AppHandle) -> Result<PathBuf> {
    Ok(handle.path().app_data_dir()?.join("models"))
}

pub fn vad_model_path_from_root(root: &Path) -> PathBuf {
    root.join("silero_vad_v6").join("silero_vad.onnx")
}

pub fn vad_model_path(handle: &AppHandle) -> Result<PathBuf> {
    Ok(vad_model_path_from_root(&models_root(handle)?))
}

pub fn default_asr_model_dir_from_root(root: &Path, model: AsrModel) -> PathBuf {
    root.join(asr_model_dir_name(model))
}

pub fn asr_model_dir(handle: &AppHandle, config: &ParapperConfig) -> Result<PathBuf> {
    Ok(asr_model_dir_from_root(&models_root(handle)?, config))
}

pub fn asr_model_dir_from_root(root: &Path, config: &ParapperConfig) -> PathBuf {
    match config
        .model_dir
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        Some(path) => PathBuf::from(path),
        None => default_asr_model_dir_from_root(root, config.asr_model),
    }
}

pub fn model_status(handle: &AppHandle, config: &ParapperConfig) -> Result<ModelStatus> {
    Ok(model_status_from_root(&models_root(handle)?, config))
}

pub fn model_status_from_root(root: &Path, config: &ParapperConfig) -> ModelStatus {
    let vad_path = vad_model_path_from_root(root);
    let asr_path = asr_model_dir_from_root(root, config);
    ModelStatus {
        root_dir: root.display().to_string(),
        vad: ModelAssetStatus {
            installed: vad_path.is_file(),
            path: vad_path.display().to_string(),
        },
        asr: ModelAssetStatus {
            installed: asr_model_installed(&asr_path, config),
            path: asr_path.display().to_string(),
        },
    }
}

pub fn any_model_installed_in(root: &Path) -> bool {
    if !root.is_dir() {
        return false;
    }

    let vad_installed = vad_model_path_from_root(root).is_file();
    if vad_installed {
        return true;
    }

    for model in [
        AsrModel::ReazonSpeechK2V2,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
        AsrModel::NemoParakeetTdt0_6BV3Int8,
    ] {
        let mut config = ParapperConfig {
            asr_language: model.language(),
            asr_model: model,
            ..ParapperConfig::default()
        };
        config.asr_precision = model.default_precision();
        let model_dir = default_asr_model_dir_from_root(root, model);
        if asr_model_installed(&model_dir, &config) {
            return true;
        }
    }

    false
}

pub async fn ensure_models_downloaded(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<ModelStatus> {
    let root = models_root(handle)?;
    fs::create_dir_all(&root)
        .with_context(|| format!("Failed to create model dir: {}", root.display()))?;

    let mut targets = Vec::new();
    let vad_path = vad_model_path(handle)?;
    if !vad_path.is_file() {
        if let Some(parent) = vad_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create VAD model dir: {}", parent.display()))?;
        }
        targets.push(DownloadTarget {
            url: VAD_MODEL_URL.to_string(),
            output_path: vad_path,
            file_name: "silero_vad.onnx".to_string(),
        });
    }

    let asr_path = asr_model_dir(handle, config)?;
    fs::create_dir_all(&asr_path)
        .with_context(|| format!("Failed to create ASR model dir: {}", asr_path.display()))?;
    for file_name in
        SherpaOnnxTransducerModelFiles::required_file_names(config.asr_model, config.asr_precision)
    {
        let output_path = asr_path.join(file_name);
        if !output_path.is_file() {
            let url = format!(
                "{}/{file_name}?download=true",
                asr_model_base_url(config.asr_model)
            );
            targets.push(DownloadTarget {
                url,
                output_path,
                file_name: (*file_name).to_string(),
            });
        }
    }

    let total_files = targets.len();
    for (index, target) in targets.into_iter().enumerate() {
        download_file(handle, &target, index, total_files).await?;
    }

    model_status(handle, config)
}

fn asr_model_installed(model_dir: &std::path::Path, config: &ParapperConfig) -> bool {
    SherpaOnnxTransducerModelFiles::required_file_names(config.asr_model, config.asr_precision)
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn asr_model_base_url(model: AsrModel) -> &'static str {
    match model {
        AsrModel::ReazonSpeechK2V2 => ASR_MODEL_BASE_URL,
        AsrModel::NemoParakeetTdt0_6BV2Int8 => ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V2_INT8,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V3_INT8,
    }
}

fn asr_model_dir_name(model: AsrModel) -> &'static str {
    match model {
        AsrModel::ReazonSpeechK2V2 => ASR_MODEL_DIR_NAME_JA,
        AsrModel::NemoParakeetTdt0_6BV2Int8 => ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V2_INT8,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V3_INT8,
    }
}

async fn download_file(
    handle: &AppHandle,
    target: &DownloadTarget,
    file_index: usize,
    total_files: usize,
) -> Result<()> {
    let temporary_path = target.output_path.with_extension("download");
    let mut response = reqwest::get(&target.url)
        .await
        .with_context(|| format!("Failed to start model download: {}", target.url))?
        .error_for_status()
        .with_context(|| format!("Model download returned an error: {}", target.url))?;
    let total_bytes = response.content_length();
    emit_download_progress(
        handle,
        &target.file_name,
        file_index,
        total_files,
        0,
        total_bytes,
        false,
    );
    let mut file = File::create(&temporary_path).await.with_context(|| {
        format!(
            "Failed to create download file: {}",
            temporary_path.display()
        )
    })?;

    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("Failed to read model download: {}", target.url))?
    {
        file.write_all(&chunk).await.with_context(|| {
            format!(
                "Failed to write download file: {}",
                temporary_path.display()
            )
        })?;
        let downloaded_bytes = temporary_path
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        emit_download_progress(
            handle,
            &target.file_name,
            file_index,
            total_files,
            downloaded_bytes,
            total_bytes,
            false,
        );
    }
    file.flush().await?;
    drop(file);

    fs::rename(&temporary_path, &target.output_path).with_context(|| {
        format!(
            "Failed to move downloaded model from {} to {}",
            temporary_path.display(),
            target.output_path.display()
        )
    })?;
    emit_download_progress(
        handle,
        &target.file_name,
        file_index,
        total_files,
        total_bytes.unwrap_or(0),
        total_bytes,
        file_index + 1 == total_files,
    );
    Ok(())
}

#[expect(clippy::cast_precision_loss)]
fn emit_download_progress(
    handle: &AppHandle,
    file_name: &str,
    file_index: usize,
    total_files: usize,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    finished: bool,
) {
    if total_files == 0 {
        return;
    }

    let file_progress = total_bytes.filter(|total| *total > 0).map_or(0.0, |total| {
        (downloaded_bytes as f64 / total as f64).clamp(0.0, 1.0)
    });
    let progress = ((file_index as f64 + file_progress) / total_files as f64).clamp(0.0, 1.0);
    let _ = handle.emit(
        "parapper://model-download-progress",
        ModelDownloadProgress {
            file_name: file_name.to_string(),
            file_index: file_index + 1,
            total_files,
            downloaded_bytes,
            total_bytes,
            progress: if finished { 1.0 } else { progress },
            finished,
        },
    );
}
