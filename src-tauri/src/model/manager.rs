use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use serde::{Deserialize, Serialize};
use tar::Archive;
use tauri::{AppHandle, Emitter, Manager};
use tokio::{fs::File, io::AsyncWriteExt};

use super::catalog::{
    ALL_ASR_MODELS, ALL_NAMO_TURN_DETECTOR_MODELS, ALL_NOISE_CANCELLATION_MODELS,
    NamoTurnDetectorModel, VAD_MODEL_URL, asr_model_base_url, asr_model_dir_name,
    asr_model_required_file_names, language_id_model_base_url, language_id_model_dir_name,
    language_id_model_files, local_tts_model_archive_name, local_tts_model_base_url,
    local_tts_model_required_dir_names, local_tts_model_required_file_names,
    namo_turn_detector_base_url, namo_turn_detector_dir_name, namo_turn_detector_files,
    noise_cancellation_model_base_url, noise_cancellation_model_dir_name,
    noise_cancellation_model_required_file_names, supertonic_tts_model_base_url,
};
use crate::config::{
    ALL_LOCAL_TTS_VOICES, AsrModel, AsrPrecision, LocalTtsFamily, LocalTtsVoice,
    NoiseCancellationModel, ParapperConfig, SpeechBackend,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub root_dir: String,
    pub vad: ModelAssetStatus,
    pub asr: ModelAssetStatus,
    pub language_id: Option<ModelAssetStatus>,
    pub turn_detectors: Vec<ModelAssetStatus>,
    pub tts: Vec<ModelAssetStatus>,
    pub noise_cancellation: Option<ModelAssetStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAssetStatus {
    pub installed: bool,
    pub path: String,
}

impl ModelAssetStatus {
    fn new(path: &Path, installed: bool) -> Self {
        Self {
            installed,
            path: path.display().to_string(),
        }
    }
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
    kind: DownloadTargetKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadTargetKind {
    File,
    TarBz2Directory,
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

pub fn default_asr_model_dir(handle: &AppHandle, model: AsrModel) -> Result<PathBuf> {
    Ok(default_asr_model_dir_from_root(
        &models_root(handle)?,
        model,
    ))
}

pub fn asr_model_dir(handle: &AppHandle, config: &ParapperConfig) -> Result<PathBuf> {
    Ok(asr_model_dir_from_root(&models_root(handle)?, config))
}

pub fn asr_model_dir_for(
    handle: &AppHandle,
    config: &ParapperConfig,
    model: AsrModel,
) -> Result<PathBuf> {
    if model == config.asr_model {
        asr_model_dir(handle, config)
    } else {
        default_asr_model_dir(handle, model)
    }
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

pub fn namo_turn_detector_model_dir_from_root(
    root: &Path,
    model: NamoTurnDetectorModel,
) -> PathBuf {
    root.join(namo_turn_detector_dir_name(model))
}

pub fn language_id_model_dir(handle: &AppHandle) -> Result<PathBuf> {
    Ok(language_id_model_dir_from_root(&models_root(handle)?))
}

fn language_id_model_dir_from_root(root: &Path) -> PathBuf {
    root.join(language_id_model_dir_name())
}

pub fn local_tts_model_dir_from_root(root: &Path, voice: LocalTtsVoice) -> PathBuf {
    root.join(voice.dir_name())
}

pub fn local_tts_model_dir(handle: &AppHandle, voice: LocalTtsVoice) -> Result<PathBuf> {
    Ok(local_tts_model_dir_from_root(&models_root(handle)?, voice))
}

pub fn noise_cancellation_model_dir_from_root(
    root: &Path,
    model: NoiseCancellationModel,
) -> PathBuf {
    root.join(noise_cancellation_model_dir_name(model))
}

pub fn noise_cancellation_model_dir(
    handle: &AppHandle,
    model: NoiseCancellationModel,
) -> Result<PathBuf> {
    Ok(noise_cancellation_model_dir_from_root(
        &models_root(handle)?,
        model,
    ))
}

pub fn model_status(handle: &AppHandle, config: &ParapperConfig) -> Result<ModelStatus> {
    Ok(model_status_from_root(&models_root(handle)?, config))
}

pub fn model_status_from_root(root: &Path, config: &ParapperConfig) -> ModelStatus {
    let vad_path = vad_model_path_from_root(root);
    let asr_path = asr_model_dir_from_root(root, config);
    let asr_installed = config.required_asr_models().into_iter().all(|model| {
        let model_dir = if model == config.asr_model {
            asr_path.clone()
        } else {
            default_asr_model_dir_from_root(root, model)
        };
        asr_model_installed_for(&model_dir, model, config.asr_precision_for(model))
    });
    ModelStatus {
        root_dir: root.display().to_string(),
        vad: ModelAssetStatus::new(&vad_path, vad_path.is_file()),
        asr: ModelAssetStatus::new(&asr_path, asr_installed),
        language_id: config.multilingual_asr_enabled.then(|| {
            let path = language_id_model_dir_from_root(root);
            ModelAssetStatus::new(&path, language_id_model_installed(&path))
        }),
        turn_detectors: if config.uses_namo_turn_detector() {
            namo_turn_detector_models_for_config(config)
                .into_iter()
                .map(|model| {
                    let path = namo_turn_detector_model_dir_from_root(root, model);
                    ModelAssetStatus::new(&path, namo_turn_detector_model_installed(&path, model))
                })
                .collect()
        } else {
            Vec::new()
        },
        tts: local_tts_voices_for_config(config)
            .into_iter()
            .map(|voice| {
                let path = local_tts_model_dir_from_root(root, voice);
                ModelAssetStatus::new(&path, local_tts_model_installed(&path, voice))
            })
            .collect(),
        noise_cancellation: config.noise_cancellation_enabled.then(|| {
            let path =
                noise_cancellation_model_dir_from_root(root, config.noise_cancellation_model);
            ModelAssetStatus::new(
                &path,
                noise_cancellation_model_installed(&path, config.noise_cancellation_model),
            )
        }),
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

    for model in ALL_ASR_MODELS {
        let mut config = ParapperConfig {
            asr_language: model.language(),
            asr_model: *model,
            ..ParapperConfig::default()
        };
        config.asr_precision = model.default_precision();
        let model_dir = default_asr_model_dir_from_root(root, *model);
        if asr_model_installed(&model_dir, &config) {
            return true;
        }
    }

    let language_id_dir = language_id_model_dir_from_root(root);
    if language_id_model_installed(&language_id_dir) {
        return true;
    }

    for model in ALL_NAMO_TURN_DETECTOR_MODELS {
        let namo_path = namo_turn_detector_model_dir_from_root(root, *model);
        if namo_turn_detector_model_installed(&namo_path, *model) {
            return true;
        }
    }

    for voice in ALL_LOCAL_TTS_VOICES {
        let tts_path = local_tts_model_dir_from_root(root, *voice);
        if local_tts_model_installed(&tts_path, *voice) {
            return true;
        }
    }

    for model in ALL_NOISE_CANCELLATION_MODELS {
        let noise_cancellation_path = noise_cancellation_model_dir_from_root(root, *model);
        if noise_cancellation_model_installed(&noise_cancellation_path, *model) {
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
    push_vad_download_targets(&mut targets, handle)?;
    push_asr_download_targets(&mut targets, handle, config)?;
    push_language_id_download_targets(&mut targets, &root, config)?;
    push_namo_download_targets(&mut targets, &root, config)?;
    push_local_tts_download_targets(&mut targets, &root, config)?;
    push_noise_cancellation_download_targets(&mut targets, &root, config)?;

    let total_files = targets.len();
    for (index, target) in targets.into_iter().enumerate() {
        download_file(handle, &target, index, total_files).await?;
    }

    model_status(handle, config)
}

fn push_vad_download_targets(targets: &mut Vec<DownloadTarget>, handle: &AppHandle) -> Result<()> {
    let vad_path = vad_model_path(handle)?;
    if vad_path.is_file() {
        return Ok(());
    }
    if let Some(parent) = vad_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create VAD model dir: {}", parent.display()))?;
    }
    targets.push(DownloadTarget {
        url: VAD_MODEL_URL.to_string(),
        output_path: vad_path,
        file_name: "silero_vad.onnx".to_string(),
        kind: DownloadTargetKind::File,
    });
    Ok(())
}

fn push_asr_download_targets(
    targets: &mut Vec<DownloadTarget>,
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<()> {
    for model in config.required_asr_models() {
        let asr_path = asr_model_dir_for(handle, config, model)?;
        fs::create_dir_all(&asr_path)
            .with_context(|| format!("Failed to create ASR model dir: {}", asr_path.display()))?;
        let precision = config.asr_precision_for(model);
        push_missing_file_targets(
            targets,
            &asr_path,
            asr_model_required_file_names(model, precision),
            asr_model_base_url(model),
        );
    }
    Ok(())
}

fn push_language_id_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    if !config.multilingual_asr_enabled {
        return Ok(());
    }

    let language_id_path = language_id_model_dir_from_root(root);
    fs::create_dir_all(&language_id_path).with_context(|| {
        format!(
            "Failed to create language identification model dir: {}",
            language_id_path.display()
        )
    })?;
    push_missing_file_targets(
        targets,
        &language_id_path,
        language_id_model_files(),
        language_id_model_base_url(),
    );
    Ok(())
}

fn push_namo_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    for model in namo_turn_detector_models_for_config(config) {
        let namo_path = namo_turn_detector_model_dir_from_root(root, model);
        fs::create_dir_all(&namo_path).with_context(|| {
            format!(
                "Failed to create Namo turn detector model dir: {}",
                namo_path.display()
            )
        })?;
        push_missing_file_targets(
            targets,
            &namo_path,
            namo_turn_detector_files(model),
            namo_turn_detector_base_url(model),
        );
    }
    Ok(())
}

fn push_missing_file_targets(
    targets: &mut Vec<DownloadTarget>,
    model_dir: &Path,
    file_names: &[&str],
    base_url: &str,
) {
    push_missing_file_targets_with_query(targets, model_dir, file_names, base_url, true);
}

fn push_missing_file_targets_with_query(
    targets: &mut Vec<DownloadTarget>,
    model_dir: &Path,
    file_names: &[&str],
    base_url: &str,
    append_download_query: bool,
) {
    for file_name in file_names {
        let output_path = model_dir.join(file_name);
        if output_path.is_file() {
            continue;
        }
        let url = if append_download_query {
            format!("{base_url}/{file_name}?download=true")
        } else {
            format!("{base_url}/{file_name}")
        };
        targets.push(DownloadTarget {
            url,
            output_path,
            file_name: (*file_name).to_string(),
            kind: DownloadTargetKind::File,
        });
    }
}

fn asr_model_installed(model_dir: &std::path::Path, config: &ParapperConfig) -> bool {
    asr_model_installed_for(model_dir, config.asr_model, config.asr_precision)
}

fn asr_model_installed_for(
    model_dir: &std::path::Path,
    model: AsrModel,
    precision: AsrPrecision,
) -> bool {
    asr_model_required_file_names(model, precision)
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn language_id_model_installed(model_dir: &std::path::Path) -> bool {
    language_id_model_files()
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn namo_turn_detector_model_installed(
    model_dir: &std::path::Path,
    model: NamoTurnDetectorModel,
) -> bool {
    namo_turn_detector_files(model)
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn local_tts_model_installed(model_dir: &Path, voice: LocalTtsVoice) -> bool {
    local_tts_model_required_file_names(voice)
        .iter()
        .all(|file| model_dir.join(file).is_file())
        && local_tts_model_required_dir_names(voice)
            .iter()
            .all(|dir| model_dir.join(dir).is_dir())
}

fn noise_cancellation_model_installed(model_dir: &Path, model: NoiseCancellationModel) -> bool {
    noise_cancellation_model_required_file_names(model)
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn push_local_tts_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    for voice in local_tts_voices_for_config(config) {
        let model_dir = local_tts_model_dir_from_root(root, voice);
        fs::create_dir_all(root)
            .with_context(|| format!("Failed to create model root dir: {}", root.display()))?;
        if voice.family() == LocalTtsFamily::Supertonic {
            push_missing_file_targets(
                targets,
                &model_dir,
                &local_tts_model_required_file_names(voice),
                supertonic_tts_model_base_url(voice),
            );
        } else if !local_tts_model_archive_installed(&model_dir, voice) {
            let archive_name = local_tts_model_archive_name(voice);
            targets.push(DownloadTarget {
                url: format!("{}/{}", local_tts_model_base_url(), archive_name),
                output_path: model_dir,
                file_name: archive_name,
                kind: DownloadTargetKind::TarBz2Directory,
            });
        }
    }
    Ok(())
}

fn push_noise_cancellation_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    if !config.noise_cancellation_enabled {
        return Ok(());
    }

    let model_dir = noise_cancellation_model_dir_from_root(root, config.noise_cancellation_model);
    fs::create_dir_all(&model_dir).with_context(|| {
        format!(
            "Failed to create noise cancellation model dir: {}",
            model_dir.display()
        )
    })?;
    push_missing_file_targets_with_query(
        targets,
        &model_dir,
        noise_cancellation_model_required_file_names(config.noise_cancellation_model),
        noise_cancellation_model_base_url(config.noise_cancellation_model),
        false,
    );
    Ok(())
}

fn local_tts_model_archive_installed(model_dir: &Path, voice: LocalTtsVoice) -> bool {
    local_tts_model_required_file_names(voice)
        .iter()
        .all(|file| model_dir.join(file).is_file())
        && local_tts_model_required_dir_names(voice)
            .iter()
            .all(|dir| model_dir.join(dir).is_dir())
}

fn namo_turn_detector_models_for_config(config: &ParapperConfig) -> Vec<NamoTurnDetectorModel> {
    config
        .required_namo_turn_detector_languages()
        .into_iter()
        .map(NamoTurnDetectorModel::for_asr_language)
        .collect()
}

fn local_tts_voices_for_config(config: &ParapperConfig) -> Vec<LocalTtsVoice> {
    let mut voices = config
        .speech_mappings
        .iter()
        .filter(|mapping| mapping.backend == SpeechBackend::LocalTts)
        .filter_map(|mapping| mapping.local_tts_voice)
        .collect::<Vec<_>>();
    voices.sort_by_key(|voice| match voice {
        LocalTtsVoice::Kristin => 0,
        LocalTtsVoice::John => 1,
        LocalTtsVoice::Norman => 2,
        LocalTtsVoice::Supertonic2Onnx => 3,
        LocalTtsVoice::Supertonic3Onnx => 4,
    });
    voices.dedup();
    voices
}

async fn download_file(
    handle: &AppHandle,
    target: &DownloadTarget,
    file_index: usize,
    total_files: usize,
) -> Result<()> {
    let temporary_path = target.output_path.with_extension("download");
    if let Some(parent) = target.output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create model output dir: {}", parent.display()))?;
    }
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

    match target.kind {
        DownloadTargetKind::File => {
            fs::rename(&temporary_path, &target.output_path).with_context(|| {
                format!(
                    "Failed to move downloaded model from {} to {}",
                    temporary_path.display(),
                    target.output_path.display()
                )
            })?;
        }
        DownloadTargetKind::TarBz2Directory => {
            extract_tar_bz2_directory(&temporary_path, &target.output_path)?;
        }
    }
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

fn extract_tar_bz2_directory(archive_path: &Path, output_path: &Path) -> Result<()> {
    let parent = output_path.parent().with_context(|| {
        format!(
            "TTS model output path has no parent directory: {}",
            output_path.display()
        )
    })?;
    let temp_dir = output_path.with_extension("extracting");
    if temp_dir.is_dir() {
        fs::remove_dir_all(&temp_dir).with_context(|| {
            format!(
                "Failed to remove temporary TTS extraction dir: {}",
                temp_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&temp_dir).with_context(|| {
        format!(
            "Failed to create temporary TTS extraction dir: {}",
            temp_dir.display()
        )
    })?;

    let file = fs::File::open(archive_path)
        .with_context(|| format!("Failed to open TTS archive: {}", archive_path.display()))?;
    let decoder = BzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive
        .unpack(&temp_dir)
        .with_context(|| format!("Failed to extract TTS archive: {}", archive_path.display()))?;

    let extracted_dir = temp_dir.join(output_path.file_name().with_context(|| {
        format!(
            "TTS model output path has no directory name: {}",
            output_path.display()
        )
    })?);
    if !extracted_dir.is_dir() {
        anyhow::bail!(
            "TTS archive did not contain expected directory: {}",
            extracted_dir.display()
        );
    }
    if output_path.is_dir() {
        fs::remove_dir_all(output_path).with_context(|| {
            format!(
                "Failed to replace existing TTS model dir: {}",
                output_path.display()
            )
        })?;
    }
    fs::rename(&extracted_dir, output_path).with_context(|| {
        format!(
            "Failed to move extracted TTS model from {} to {}",
            extracted_dir.display(),
            output_path.display()
        )
    })?;
    fs::remove_dir_all(&temp_dir).ok();
    fs::remove_file(archive_path).ok();

    if !output_path.starts_with(parent) {
        anyhow::bail!("TTS model extraction escaped model root");
    }
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

#[cfg(test)]
mod tests {
    use super::{
        NamoTurnDetectorModel, default_asr_model_dir_from_root, local_tts_voices_for_config,
        model_status_from_root, namo_turn_detector_models_for_config,
    };
    use crate::config::{
        AsrModel, AsrPrecision, LocalTtsVoice, NoiseCancellationModel, ParapperConfig,
        SpeechBackend, SpeechMapping, SpeechSourceKind, TurnDetector,
    };
    use crate::model::catalog::asr_model_required_file_names;
    use std::{fs, path::Path, time::SystemTime};

    #[test]
    fn namo_models_follow_required_asr_models() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            turn_detector: TurnDetector::Namo,
            enabled_asr_models: vec![
                AsrModel::NemoParakeetTdt0_6BV3Int8,
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
            ..ParapperConfig::default()
        };

        assert_eq!(
            namo_turn_detector_models_for_config(&config),
            vec![
                NamoTurnDetectorModel::Japanese,
                NamoTurnDetectorModel::English,
                NamoTurnDetectorModel::Multilingual,
            ]
        );
    }

    #[test]
    fn language_id_and_turn_detector_status_follow_mode_matrix() {
        for turn_detector in [TurnDetector::Simple, TurnDetector::Namo] {
            for multilingual_asr_enabled in [false, true] {
                let config = ParapperConfig {
                    multilingual_asr_enabled,
                    turn_detector,
                    ..ParapperConfig::default()
                };

                let status = model_status_from_root(std::path::Path::new("models"), &config);

                assert_eq!(
                    status.language_id.is_some(),
                    multilingual_asr_enabled,
                    "turn_detector={turn_detector:?}, multilingual={multilingual_asr_enabled}"
                );
                assert_eq!(
                    !status.turn_detectors.is_empty(),
                    config.uses_namo_turn_detector(),
                    "turn_detector={turn_detector:?}, multilingual={multilingual_asr_enabled}"
                );
                assert!(status.tts.is_empty());
            }
        }
    }

    #[test]
    fn noise_cancellation_status_only_appears_when_enabled() {
        let disabled =
            model_status_from_root(std::path::Path::new("models"), &ParapperConfig::default());
        assert!(disabled.noise_cancellation.is_none());

        let enabled = model_status_from_root(
            std::path::Path::new("models"),
            &ParapperConfig {
                noise_cancellation_enabled: true,
                noise_cancellation_model: NoiseCancellationModel::UlUnas,
                ..ParapperConfig::default()
            },
        );
        assert!(enabled.noise_cancellation.is_some());
    }

    #[test]
    fn asr_status_requires_all_enabled_asr_models() {
        let root = unique_test_models_root("model-status-asr");
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            asr_model: AsrModel::ReazonSpeechK2V2,
            asr_precision: AsrPrecision::Int8Float32,
            enabled_asr_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
            ..ParapperConfig::default()
        };

        write_required_asr_model_files(
            &root,
            AsrModel::ReazonSpeechK2V2,
            AsrPrecision::Int8Float32,
        );

        let status = model_status_from_root(&root, &config);
        assert!(!status.asr.installed);

        write_required_asr_model_files(
            &root,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            AsrPrecision::Int8,
        );

        let status = model_status_from_root(&root, &config);
        assert!(status.asr.installed);

        let _ = fs::remove_dir_all(root);
    }

    fn unique_test_models_root(name: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("{name}-{}-{timestamp}", std::process::id()))
    }

    fn write_required_asr_model_files(root: &Path, model: AsrModel, precision: AsrPrecision) {
        let model_dir = default_asr_model_dir_from_root(root, model);
        fs::create_dir_all(&model_dir).expect("failed to create test ASR model dir");
        for file in asr_model_required_file_names(model, precision) {
            let path = model_dir.join(file);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("failed to create test ASR file parent");
            }
            fs::write(path, b"test").expect("failed to write test ASR model file");
        }
    }

    #[test]
    fn local_tts_models_follow_speech_mappings() {
        let config = ParapperConfig {
            speech_mappings: vec![
                SpeechMapping {
                    id: "tts-kristin".to_string(),
                    source_kind: SpeechSourceKind::Recognition,
                    source_asr_model: None,
                    target_lang: None,
                    backend: SpeechBackend::LocalTts,
                    talker: String::new(),
                    local_tts_voice: Some(LocalTtsVoice::Kristin),
                    local_tts_language: None,
                    local_tts_speaker_id: None,
                    output_device_id: None,
                    output_device_host: None,
                    output_device_name: None,
                    muted: false,
                    volume: 1.0,
                },
                SpeechMapping {
                    id: "tts-kristin-2".to_string(),
                    source_kind: SpeechSourceKind::Translation,
                    source_asr_model: None,
                    target_lang: Some("en_US".to_string()),
                    backend: SpeechBackend::LocalTts,
                    talker: String::new(),
                    local_tts_voice: Some(LocalTtsVoice::Kristin),
                    local_tts_language: None,
                    local_tts_speaker_id: None,
                    output_device_id: None,
                    output_device_host: None,
                    output_device_name: None,
                    muted: false,
                    volume: 1.0,
                },
                SpeechMapping {
                    id: "tts-supertonic".to_string(),
                    source_kind: SpeechSourceKind::Translation,
                    source_asr_model: None,
                    target_lang: Some("en_US".to_string()),
                    backend: SpeechBackend::LocalTts,
                    talker: String::new(),
                    local_tts_voice: Some(LocalTtsVoice::Supertonic2Onnx),
                    local_tts_language: Some("en".to_string()),
                    local_tts_speaker_id: Some(0),
                    output_device_id: None,
                    output_device_host: None,
                    output_device_name: None,
                    muted: false,
                    volume: 1.0,
                },
                SpeechMapping {
                    id: "tts-neo".to_string(),
                    source_kind: SpeechSourceKind::Recognition,
                    source_asr_model: None,
                    target_lang: None,
                    backend: SpeechBackend::Ync,
                    talker: "Voice/Engine".to_string(),
                    local_tts_voice: None,
                    local_tts_language: None,
                    local_tts_speaker_id: None,
                    output_device_id: None,
                    output_device_host: None,
                    output_device_name: None,
                    muted: false,
                    volume: 1.0,
                },
            ],
            ..ParapperConfig::default()
        };

        assert_eq!(
            local_tts_voices_for_config(&config),
            vec![LocalTtsVoice::Kristin, LocalTtsVoice::Supertonic2Onnx]
        );
        assert_eq!(
            model_status_from_root(std::path::Path::new("models"), &config)
                .tts
                .len(),
            2
        );
    }
}
