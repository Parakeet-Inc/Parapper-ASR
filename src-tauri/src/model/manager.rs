use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tar::Archive;
use tauri::{AppHandle, Emitter, Manager};
use tokio::{fs::File, io::AsyncWriteExt};

use super::catalog::{
    ALL_ASR_MODELS, ALL_LOCAL_TRANSLATION_MODELS, ALL_NAMO_TURN_DETECTOR_MODELS,
    ALL_NOISE_CANCELLATION_MODELS, FileIntegrity, NamoTurnDetectorModel, VAD_MODEL_URL,
    asr_model_archive_integrity, asr_model_archive_name, asr_model_base_url, asr_model_dir_name,
    asr_model_file_integrity, asr_model_required_file_names, language_id_model_base_url,
    language_id_model_dir_name, language_id_model_files, local_translation_model_base_url,
    local_translation_model_dir_name, local_translation_model_file_base_url,
    local_translation_model_file_integrity, local_translation_model_required_file_names,
    local_tts_model_file_integrity, local_tts_model_required_dir_names,
    local_tts_model_required_file_names, namo_turn_detector_base_url, namo_turn_detector_dir_name,
    namo_turn_detector_files, noise_cancellation_model_base_url, noise_cancellation_model_dir_name,
    noise_cancellation_model_required_file_names, reazon_static_embedding_base_url,
    reazon_static_embedding_file_integrity, supertonic_tts_model_base_url,
};
use crate::config::{
    ALL_LOCAL_TTS_VOICES, AsrMode, AsrModel, AsrPrecision, LocalTranslationModel, LocalTtsVoice,
    NoiseCancellationModel, ParapperConfig, SpeechBackend, TranslationBackend,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub root_dir: String,
    pub vad: ModelAssetStatus,
    pub asr: ModelAssetStatus,
    pub japanese_morph: Option<ModelAssetStatus>,
    pub language_id: Option<ModelAssetStatus>,
    pub turn_detectors: Vec<ModelAssetStatus>,
    pub tts: Vec<ModelAssetStatus>,
    pub local_translation: Option<ModelAssetStatus>,
    pub noise_cancellation: Option<ModelAssetStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAssetStatus {
    pub installed: bool,
    pub preparing: bool,
    pub path: String,
}

impl ModelAssetStatus {
    fn new(path: &Path, installed: bool) -> Self {
        Self {
            installed,
            preparing: false,
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
    integrity: Option<FileIntegrity>,
}

const MODEL_ARCHIVE_SHA256_MARKER: &str = ".parapper-archive-sha256";
const BUILT_IN_JAPANESE_MORPH_DICTIONARY: &str = "built-in:unidic-cwj-3_1_1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadTargetKind {
    File,
    LocalFile,
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
    if model == config.asr.model {
        asr_model_dir(handle, config)
    } else {
        default_asr_model_dir(handle, model)
    }
}

pub fn asr_model_dir_from_root(root: &Path, config: &ParapperConfig) -> PathBuf {
    match config
        .models
        .dir
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        Some(path) => PathBuf::from(path),
        None => default_asr_model_dir_from_root(root, config.asr.model),
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

pub fn local_translation_model_dir_from_root(root: &Path, model: LocalTranslationModel) -> PathBuf {
    root.join(local_translation_model_dir_name(model))
}

pub fn local_translation_model_dir(
    handle: &AppHandle,
    model: LocalTranslationModel,
) -> Result<PathBuf> {
    Ok(local_translation_model_dir_from_root(
        &models_root(handle)?,
        model,
    ))
}

pub fn local_translation_model_is_installed(
    handle: &AppHandle,
    model: LocalTranslationModel,
) -> Result<bool> {
    let model_dir = local_translation_model_dir(handle, model)?;
    Ok(local_translation_model_installed(&model_dir, model))
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
    let stt_runtime_configs = stt_runtime_configs_for_model_management(config);
    let asr_path = asr_model_dir_from_root(root, &stt_runtime_configs[0]);
    let asr_installed = stt_runtime_configs.iter().all(|runtime_config| {
        runtime_config
            .required_asr_models()
            .into_iter()
            .all(|model| {
                let model_dir = asr_model_dir_for_runtime_config(root, runtime_config, model);
                asr_model_installed_for(&model_dir, model, runtime_config.asr_precision_for(model))
                    && (!reazon_accuracy_assets_required(runtime_config, model)
                        || reazon_accuracy_assets_installed(&model_dir))
                    && (!parakeet_accuracy_assets_required(runtime_config, model)
                        || parakeet_accuracy_assets_installed(&model_dir))
            })
    });
    let noise_cancellation_models = noise_cancellation_models_for_stt_runtime(&stt_runtime_configs);
    ModelStatus {
        root_dir: root.display().to_string(),
        vad: ModelAssetStatus::new(&vad_path, vad_path.is_file()),
        asr: ModelAssetStatus::new(&asr_path, asr_installed),
        japanese_morph: japanese_morph_required(config)
            .then(|| ModelAssetStatus::new(Path::new(BUILT_IN_JAPANESE_MORPH_DICTIONARY), true)),
        language_id: stt_runtime_configs
            .iter()
            .any(|runtime_config| runtime_config.asr.multilingual_enabled)
            .then(|| {
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
        local_translation: {
            let models = local_translation_models_for_config(config);
            (!models.is_empty()).then(|| {
                let path = local_translation_status_path(root, &models);
                ModelAssetStatus::new(&path, local_translation_models_installed(root, &models))
            })
        },
        noise_cancellation: (!noise_cancellation_models.is_empty()).then(|| {
            let path = noise_cancellation_model_dir_from_root(root, noise_cancellation_models[0]);
            ModelAssetStatus::new(
                &path,
                noise_cancellation_models.iter().all(|model| {
                    let path = noise_cancellation_model_dir_from_root(root, *model);
                    noise_cancellation_model_installed(&path, *model)
                }),
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
        let mut config = ParapperConfig::default();
        config.asr.language = model.language();
        config.asr.model = *model;
        config.asr.precision = model.default_precision();
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

    if ALL_LOCAL_TRANSLATION_MODELS.iter().any(|model| {
        let local_translation_path = local_translation_model_dir_from_root(root, *model);
        local_translation_model_installed(&local_translation_path, *model)
    }) {
        return true;
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
    // Japanese Morph and hotword-reading dictionaries are compiled into the application.
    push_asr_download_targets(&mut targets, handle, config)?;
    push_language_id_download_targets(&mut targets, &root, config)?;
    push_namo_download_targets(&mut targets, &root, config)?;
    push_local_tts_download_targets(&mut targets, &root, config)?;
    push_local_translation_download_targets(&mut targets, &root, config)?;
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
        integrity: None,
    });
    Ok(())
}

fn push_asr_download_targets(
    targets: &mut Vec<DownloadTarget>,
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<()> {
    push_asr_download_targets_from_root(targets, &models_root(handle)?, config)
}

fn push_asr_download_targets_from_root(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    for runtime_config in stt_runtime_configs_for_model_management(config) {
        push_asr_download_targets_for_runtime_config(targets, root, &runtime_config)?;
    }
    Ok(())
}

fn push_asr_download_targets_for_runtime_config(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    for model in config.required_asr_models() {
        let asr_path = asr_model_dir_for_runtime_config(root, config, model);
        fs::create_dir_all(&asr_path)
            .with_context(|| format!("Failed to create ASR model dir: {}", asr_path.display()))?;
        let precision = config.asr_precision_for(model);
        if !asr_model_installed_for(&asr_path, model, precision) {
            if let Some(archive_name) = asr_model_archive_name(model) {
                let archive_already_scheduled = targets.iter().any(|target| {
                    target.kind == DownloadTargetKind::TarBz2Directory
                        && target.output_path == asr_path
                });
                if !archive_already_scheduled {
                    targets.push(DownloadTarget {
                        url: format!("{}/{}", asr_model_base_url(model), archive_name),
                        output_path: asr_path.clone(),
                        file_name: archive_name,
                        kind: DownloadTargetKind::TarBz2Directory,
                        integrity: asr_model_archive_integrity(model),
                    });
                }
                continue;
            }
            push_missing_asr_file_targets(
                targets,
                &asr_path,
                model,
                asr_model_required_file_names(model, precision),
            );
        }
        push_reazon_accuracy_download_targets(targets, &asr_path, config, model);
        push_parakeet_accuracy_download_targets(targets, &asr_path, config, model);
    }
    Ok(())
}

fn push_reazon_accuracy_download_targets(
    targets: &mut Vec<DownloadTarget>,
    asr_model_dir: &Path,
    config: &ParapperConfig,
    model: AsrModel,
) {
    if !reazon_accuracy_assets_required(config, model) {
        return;
    }
    let model_dir =
        asr_model_dir.join(parapper_models::asr::backend::REAZON_STATIC_EMBEDDING_DIR_NAME);
    for file_name in parapper_models::asr::backend::REAZON_STATIC_EMBEDDING_REQUIRED_FILES {
        let output_path = model_dir.join(file_name);
        if output_path.is_file() || target_output_already_scheduled(targets, &output_path) {
            continue;
        }
        targets.push(DownloadTarget {
            url: format!(
                "{}/{file_name}?download=true",
                reazon_static_embedding_base_url()
            ),
            output_path,
            file_name: (*file_name).to_owned(),
            kind: DownloadTargetKind::File,
            integrity: reazon_static_embedding_file_integrity(file_name),
        });
    }
}

fn push_parakeet_accuracy_download_targets(
    targets: &mut Vec<DownloadTarget>,
    asr_model_dir: &Path,
    config: &ParapperConfig,
    model: AsrModel,
) {
    if !parakeet_accuracy_assets_required(config, model) {
        return;
    }
    push_missing_asr_file_targets(
        targets,
        asr_model_dir,
        model,
        parapper_models::asr::backend::parakeet_ja::HYBRID_REQUIRED_FILES,
    );
}

fn push_language_id_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    if !stt_runtime_configs_for_model_management(config)
        .iter()
        .any(|runtime_config| runtime_config.asr.multilingual_enabled)
    {
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

fn push_missing_asr_file_targets(
    targets: &mut Vec<DownloadTarget>,
    model_dir: &Path,
    model: AsrModel,
    file_names: &[&str],
) {
    let base_url = asr_model_base_url(model);
    for file_name in file_names {
        let output_path = model_dir.join(file_name);
        if output_path.is_file() || target_output_already_scheduled(targets, &output_path) {
            continue;
        }
        targets.push(DownloadTarget {
            url: format!("{base_url}/{file_name}?download=true"),
            output_path,
            file_name: (*file_name).to_owned(),
            kind: DownloadTargetKind::File,
            integrity: asr_model_file_integrity(model, file_name),
        });
    }
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
        if output_path.is_file() || target_output_already_scheduled(targets, &output_path) {
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
            integrity: None,
        });
    }
}

fn asr_model_installed(model_dir: &std::path::Path, config: &ParapperConfig) -> bool {
    asr_model_installed_for(model_dir, config.asr.model, config.asr.precision)
}

fn asr_model_installed_for(
    model_dir: &std::path::Path,
    model: AsrModel,
    precision: AsrPrecision,
) -> bool {
    let required_files_present = asr_model_required_file_names(model, precision)
        .iter()
        .all(|file| model_dir.join(file).is_file());
    required_files_present && current_nemotron_archive_installed(model_dir, model)
}

fn current_nemotron_archive_installed(model_dir: &Path, model: AsrModel) -> bool {
    if !model.is_nemotron() {
        return true;
    }
    let Some(expected) = asr_model_archive_integrity(model) else {
        return false;
    };
    fs::read_to_string(model_dir.join(MODEL_ARCHIVE_SHA256_MARKER))
        .is_ok_and(|value| value.trim() == expected.sha256)
}

fn reazon_accuracy_assets_required(config: &ParapperConfig, model: AsrModel) -> bool {
    config.asr.mode == AsrMode::Accurate
        && config.asr.model == AsrModel::ReazonSpeechK2V2
        && model == AsrModel::ReazonSpeechK2V2
}

fn reazon_accuracy_assets_installed(asr_model_dir: &Path) -> bool {
    let model_dir =
        asr_model_dir.join(parapper_models::asr::backend::REAZON_STATIC_EMBEDDING_DIR_NAME);
    parapper_models::asr::backend::REAZON_STATIC_EMBEDDING_REQUIRED_FILES
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn parakeet_accuracy_assets_required(config: &ParapperConfig, model: AsrModel) -> bool {
    config.asr.mode == AsrMode::Accurate
        && config.asr.model == AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8
        && model == AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8
}

fn parakeet_accuracy_assets_installed(asr_model_dir: &Path) -> bool {
    parapper_models::asr::backend::parakeet_ja::HYBRID_REQUIRED_FILES
        .iter()
        .all(|file| asr_model_dir.join(file).is_file())
}

fn language_id_model_installed(model_dir: &std::path::Path) -> bool {
    language_id_model_files()
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn japanese_morph_required(config: &ParapperConfig) -> bool {
    config.requires_japanese_morph_analyzer()
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

fn local_translation_model_installed(model_dir: &Path, model: LocalTranslationModel) -> bool {
    local_translation_model_required_file_names(model)
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn local_translation_models_installed(root: &Path, models: &[LocalTranslationModel]) -> bool {
    models.iter().all(|model| {
        let model_dir = local_translation_model_dir_from_root(root, *model);
        local_translation_model_installed(&model_dir, *model)
    })
}

fn local_translation_status_path(root: &Path, models: &[LocalTranslationModel]) -> PathBuf {
    match models {
        [model] => local_translation_model_dir_from_root(root, *model),
        _ => root.to_path_buf(),
    }
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
        let required_files = local_tts_model_required_file_names(voice);
        if voice == LocalTtsVoice::Supertonic3OnnxQuantized {
            push_missing_verified_local_tts_file_targets(
                targets,
                &model_dir,
                voice,
                &required_files,
                supertonic_tts_model_base_url(voice),
            )?;
        } else {
            push_missing_file_targets(
                targets,
                &model_dir,
                &required_files,
                supertonic_tts_model_base_url(voice),
            );
        }
    }
    Ok(())
}

fn push_missing_verified_local_tts_file_targets(
    targets: &mut Vec<DownloadTarget>,
    model_dir: &Path,
    voice: LocalTtsVoice,
    file_names: &[&str],
    base_url: &str,
) -> Result<()> {
    for file_name in file_names {
        let output_path = model_dir.join(file_name);
        let integrity = local_tts_model_file_integrity(voice, file_name);
        if output_path.is_file() {
            match integrity {
                Some(expected) => {
                    if verify_file_integrity(
                        &output_path,
                        expected,
                        "installed local TTS model file",
                    )
                    .is_ok()
                    {
                        continue;
                    }
                    log::warn!(
                        "Replacing local TTS model file that does not match the published distribution: {}",
                        output_path.display()
                    );
                    fs::remove_file(&output_path).with_context(|| {
                        format!(
                            "Failed to remove invalid local TTS model file: {}",
                            output_path.display()
                        )
                    })?;
                }
                None => continue,
            }
        }
        if target_output_already_scheduled(targets, &output_path) {
            continue;
        }
        targets.push(DownloadTarget {
            url: format!("{base_url}/{file_name}?download=true"),
            output_path,
            file_name: (*file_name).to_string(),
            kind: DownloadTargetKind::File,
            integrity,
        });
    }
    Ok(())
}

fn push_local_translation_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    push_local_translation_download_targets_with_source_resolver(
        targets,
        root,
        config,
        local_translation_model_local_source_dir,
    )
}

fn push_local_translation_download_targets_with_source_resolver(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
    local_source_dir: impl Fn(LocalTranslationModel) -> Option<PathBuf>,
) -> Result<()> {
    for model in local_translation_models_for_config(config) {
        push_local_translation_model_download_targets(targets, root, model, &local_source_dir)?;
    }
    Ok(())
}

fn push_local_translation_model_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    model: LocalTranslationModel,
    local_source_dir: &impl Fn(LocalTranslationModel) -> Option<PathBuf>,
) -> Result<()> {
    let model_dir = local_translation_model_dir_from_root(root, model);
    fs::create_dir_all(&model_dir).with_context(|| {
        format!(
            "Failed to create local translation model dir: {}",
            model_dir.display()
        )
    })?;
    let required_files = local_translation_model_required_file_names(model);
    if local_translation_model_base_url(model).is_some() {
        push_missing_local_translation_file_targets(targets, &model_dir, model, required_files)?;
    } else {
        let source_dir = local_source_dir(model)
            .with_context(|| format!("Local translation model {model:?} has no download source"))?;
        push_missing_local_file_targets(targets, &model_dir, required_files, &source_dir)?;
    }
    Ok(())
}

fn push_missing_local_translation_file_targets(
    targets: &mut Vec<DownloadTarget>,
    model_dir: &Path,
    model: LocalTranslationModel,
    file_names: &[&str],
) -> Result<()> {
    for file_name in file_names {
        let output_path = model_dir.join(file_name);
        let integrity = local_translation_model_file_integrity(model, file_name);
        if output_path.is_file() {
            match integrity {
                Some(expected) => {
                    if verify_file_integrity(
                        &output_path,
                        expected,
                        "installed local translation model file",
                    )
                    .is_ok()
                    {
                        continue;
                    }
                    log::warn!(
                        "Replacing local translation model file that does not match the published distribution: {}",
                        output_path.display()
                    );
                    fs::remove_file(&output_path).with_context(|| {
                        format!(
                            "Failed to remove invalid local translation model file: {}",
                            output_path.display()
                        )
                    })?;
                }
                None => continue,
            }
        }
        if target_output_already_scheduled(targets, &output_path) {
            continue;
        }
        let file_base_url = local_translation_model_file_base_url(model, file_name)
            .expect("downloadable local translation model file must have a source URL");
        targets.push(DownloadTarget {
            url: format!("{file_base_url}/{file_name}?download=true"),
            output_path,
            file_name: (*file_name).to_string(),
            kind: DownloadTargetKind::File,
            integrity,
        });
    }
    Ok(())
}

pub async fn ensure_local_translation_model_downloaded(
    handle: &AppHandle,
    model: LocalTranslationModel,
) -> Result<()> {
    let root = models_root(handle)?;
    fs::create_dir_all(&root)
        .with_context(|| format!("Failed to create model dir: {}", root.display()))?;

    let mut targets = Vec::new();
    push_local_translation_model_download_targets(
        &mut targets,
        &root,
        model,
        &local_translation_model_local_source_dir,
    )?;

    let total_files = targets.len();
    for (index, target) in targets.into_iter().enumerate() {
        download_file(handle, &target, index, total_files).await?;
    }
    Ok(())
}

fn push_missing_local_file_targets(
    targets: &mut Vec<DownloadTarget>,
    model_dir: &Path,
    file_names: &[&str],
    source_dir: &Path,
) -> Result<()> {
    for file_name in file_names {
        let output_path = model_dir.join(file_name);
        if output_path.is_file() || target_output_already_scheduled(targets, &output_path) {
            continue;
        }
        let source_path = source_dir.join(file_name);
        if !source_path.is_file() {
            anyhow::bail!(
                "Local translation model source file is missing: {}",
                source_path.display()
            );
        }
        targets.push(DownloadTarget {
            url: source_path.display().to_string(),
            output_path,
            file_name: (*file_name).to_string(),
            kind: DownloadTargetKind::LocalFile,
            integrity: None,
        });
    }
    Ok(())
}

fn target_output_already_scheduled(targets: &[DownloadTarget], output_path: &Path) -> bool {
    targets
        .iter()
        .any(|target| target.output_path == output_path)
}

fn local_translation_model_local_source_dir(model: LocalTranslationModel) -> Option<PathBuf> {
    match model {
        LocalTranslationModel::Lfm2Q4 | LocalTranslationModel::CatTranslate0_8BQ4KQuant => None,
    }
}

fn push_noise_cancellation_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    for model in
        noise_cancellation_models_for_stt_runtime(&stt_runtime_configs_for_model_management(config))
    {
        let model_dir = noise_cancellation_model_dir_from_root(root, model);
        fs::create_dir_all(&model_dir).with_context(|| {
            format!(
                "Failed to create noise cancellation model dir: {}",
                model_dir.display()
            )
        })?;
        push_missing_file_targets_with_query(
            targets,
            &model_dir,
            noise_cancellation_model_required_file_names(model),
            noise_cancellation_model_base_url(model),
            false,
        );
    }
    Ok(())
}

fn stt_runtime_configs_for_model_management(config: &ParapperConfig) -> Vec<ParapperConfig> {
    if config.stt_profiles.is_empty() {
        return vec![config.clone()];
    }
    config
        .stt_profiles
        .iter()
        .filter(|profile| profile.enabled)
        .map(|profile| {
            config
                .config_for_stt_profile(&profile.id)
                .expect("STT profile collected from this configuration must resolve")
        })
        .collect()
}

fn asr_model_dir_for_runtime_config(
    root: &Path,
    config: &ParapperConfig,
    model: AsrModel,
) -> PathBuf {
    if model == config.asr.model {
        asr_model_dir_from_root(root, config)
    } else {
        default_asr_model_dir_from_root(root, model)
    }
}

fn noise_cancellation_models_for_stt_runtime(
    runtime_configs: &[ParapperConfig],
) -> Vec<NoiseCancellationModel> {
    let mut models = Vec::new();
    for config in runtime_configs {
        if config.noise_cancellation.enabled && !models.contains(&config.noise_cancellation.model) {
            models.push(config.noise_cancellation.model);
        }
    }
    models
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
        .speech
        .mappings
        .iter()
        .filter(|mapping| mapping.backend == SpeechBackend::LocalTts)
        .filter_map(|mapping| mapping.local_tts_voice)
        .collect::<Vec<_>>();
    voices.sort_by_key(|voice| match voice {
        LocalTtsVoice::Supertonic2Onnx => 0,
        LocalTtsVoice::Supertonic3Onnx => 1,
        LocalTtsVoice::Supertonic3OnnxQuantized => 2,
    });
    voices.dedup();
    voices
}

fn local_translation_models_for_config(config: &ParapperConfig) -> Vec<LocalTranslationModel> {
    if !config.translation.enabled {
        return Vec::new();
    }

    let mut models = Vec::new();
    if config.translation.enabled {
        models.extend(
            config
                .translation
                .mappings
                .iter()
                .filter(|mapping| mapping.backend == TranslationBackend::Local)
                .filter(|mapping| ALL_LOCAL_TRANSLATION_MODELS.contains(&mapping.local_model))
                .map(|mapping| mapping.local_model),
        );
    }
    models.sort_by_key(|model| model.sort_key());
    models.dedup();
    models
}

async fn download_file<R: tauri::Runtime>(
    handle: &AppHandle<R>,
    target: &DownloadTarget,
    file_index: usize,
    total_files: usize,
) -> Result<()> {
    let temporary_path = target.output_path.with_extension("download");
    if let Some(parent) = target.output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create model output dir: {}", parent.display()))?;
    }
    if target.kind == DownloadTargetKind::LocalFile {
        copy_local_model_file(handle, target, file_index, total_files)?;
        return Ok(());
    }
    if try_install_cached_archive(handle, target, &temporary_path, file_index, total_files)? {
        return Ok(());
    }
    let mut response = reqwest::get(&target.url)
        .await
        .with_context(|| format!("Failed to start model download: {}", target.url))?
        .error_for_status()
        .with_context(|| format!("Model download returned an error: {}", target.url))?;
    let total_bytes = response.content_length();
    if let (Some(integrity), Some(actual_size)) = (target.integrity, total_bytes)
        && actual_size != integrity.size
    {
        anyhow::bail!(
            "Model download size header did not match {}: expected {} bytes, got {} bytes",
            target.url,
            integrity.size,
            actual_size
        );
    }
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
    let mut downloaded_bytes = 0_u64;

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
        downloaded_bytes =
            downloaded_bytes.saturating_add(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
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
    install_downloaded_target(target, &temporary_path)?;
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

fn try_install_cached_archive<R: tauri::Runtime>(
    handle: &AppHandle<R>,
    target: &DownloadTarget,
    temporary_path: &Path,
    file_index: usize,
    total_files: usize,
) -> Result<bool> {
    if !cached_archive_available(target, temporary_path)? {
        return Ok(false);
    }
    if target.integrity.is_some() {
        install_downloaded_archive(handle, target, temporary_path, file_index, total_files)?;
        return Ok(true);
    }

    match install_downloaded_archive(handle, target, temporary_path, file_index, total_files) {
        Ok(()) => Ok(true),
        Err(err) => {
            log::warn!(
                "Failed to install existing model archive {}; downloading it again: {err}",
                temporary_path.display()
            );
            fs::remove_file(temporary_path).ok();
            Ok(false)
        }
    }
}

fn cached_archive_available(target: &DownloadTarget, temporary_path: &Path) -> Result<bool> {
    if target.kind == DownloadTargetKind::File || !temporary_path.is_file() {
        return Ok(false);
    }
    if let Some(integrity) = target.integrity
        && let Err(err) = verify_file_integrity(temporary_path, integrity, "cached model archive")
    {
        log::warn!(
            "Discarding invalid cached model archive {}: {err}",
            temporary_path.display()
        );
        fs::remove_file(temporary_path).with_context(|| {
            format!(
                "Failed to remove invalid cached model archive: {}",
                temporary_path.display()
            )
        })?;
        return Ok(false);
    }
    Ok(true)
}

fn copy_local_model_file<R: tauri::Runtime>(
    handle: &AppHandle<R>,
    target: &DownloadTarget,
    file_index: usize,
    total_files: usize,
) -> Result<()> {
    let source_path = Path::new(&target.url);
    let total_bytes = source_path.metadata().map(|metadata| metadata.len()).ok();
    emit_download_progress(
        handle,
        &target.file_name,
        file_index,
        total_files,
        0,
        total_bytes,
        false,
    );
    let copied_bytes = fs::copy(source_path, &target.output_path).with_context(|| {
        format!(
            "Failed to copy local model file from {} to {}",
            source_path.display(),
            target.output_path.display()
        )
    })?;
    emit_download_progress(
        handle,
        &target.file_name,
        file_index,
        total_files,
        copied_bytes,
        total_bytes,
        file_index + 1 == total_files,
    );
    Ok(())
}

fn install_downloaded_archive<R: tauri::Runtime>(
    handle: &AppHandle<R>,
    target: &DownloadTarget,
    temporary_path: &Path,
    file_index: usize,
    total_files: usize,
) -> Result<()> {
    let downloaded_bytes = temporary_path
        .metadata()
        .map_or(0, |metadata| metadata.len());
    emit_download_progress(
        handle,
        &target.file_name,
        file_index,
        total_files,
        downloaded_bytes,
        Some(downloaded_bytes),
        false,
    );

    install_downloaded_target(target, temporary_path)?;

    emit_download_progress(
        handle,
        &target.file_name,
        file_index,
        total_files,
        downloaded_bytes,
        Some(downloaded_bytes),
        file_index + 1 == total_files,
    );
    Ok(())
}

fn install_downloaded_target(target: &DownloadTarget, temporary_path: &Path) -> Result<()> {
    if let Some(integrity) = target.integrity {
        verify_file_integrity(temporary_path, integrity, "downloaded model archive")?;
    }
    match target.kind {
        DownloadTargetKind::File => {
            fs::rename(temporary_path, &target.output_path).with_context(|| {
                format!(
                    "Failed to move downloaded model from {} to {}",
                    temporary_path.display(),
                    target.output_path.display()
                )
            })?;
        }
        DownloadTargetKind::LocalFile => unreachable!("local files are not archive-installed"),
        DownloadTargetKind::TarBz2Directory => {
            extract_tar_bz2_directory(temporary_path, &target.output_path)?;
            if let Some(integrity) = target.integrity {
                fs::write(
                    target.output_path.join(MODEL_ARCHIVE_SHA256_MARKER),
                    integrity.sha256,
                )
                .with_context(|| {
                    format!(
                        "Failed to record installed model archive revision: {}",
                        target.output_path.display()
                    )
                })?;
            }
        }
    }
    Ok(())
}

fn verify_file_integrity(path: &Path, expected: FileIntegrity, label: &str) -> Result<()> {
    let actual_size = path
        .metadata()
        .with_context(|| format!("Failed to inspect {label}: {}", path.display()))?
        .len();
    if actual_size != expected.size {
        anyhow::bail!(
            "{label} size mismatch for {}: expected {} bytes, got {} bytes",
            path.display(),
            expected.size,
            actual_size
        );
    }

    let actual_sha256 = sha256_file(path, label)?;
    if actual_sha256 != expected.sha256 {
        anyhow::bail!(
            "{label} SHA-256 mismatch for {}: expected {}, got {}",
            path.display(),
            expected.sha256,
            actual_sha256
        );
    }
    Ok(())
}

fn sha256_file(path: &Path, label: &str) -> Result<String> {
    let mut input = fs::File::open(path)
        .with_context(|| format!("Failed to open {label}: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .with_context(|| format!("Failed to hash {label}: {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_tar_bz2_directory(archive_path: &Path, output_path: &Path) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("Failed to open TTS archive: {}", archive_path.display()))?;
    let decoder = BzDecoder::new(file);
    extract_tar_directory(decoder, archive_path, output_path, "TTS")
}

fn extract_tar_directory(
    reader: impl std::io::Read,
    archive_path: &Path,
    output_path: &Path,
    label: &str,
) -> Result<()> {
    extract_archive_directory(archive_path, output_path, label, |temp_dir| {
        let mut archive = Archive::new(reader);
        unpack_tar_entries_within(&mut archive, temp_dir, label).with_context(|| {
            format!(
                "Failed to extract {label} archive: {}",
                archive_path.display()
            )
        })?;
        Ok(())
    })
}

fn unpack_tar_entries_within<R: std::io::Read>(
    archive: &mut Archive<R>,
    temp_dir: &Path,
    label: &str,
) -> Result<()> {
    for entry in archive.entries()? {
        let mut entry = entry?;
        let entry_type = entry.header().entry_type();
        if entry_type.is_symlink() || entry_type.is_hard_link() {
            anyhow::bail!("{label} archive contains unsupported link entry");
        }
        let entry_path = entry.path()?.into_owned();
        let output_path = contained_tar_entry_path(temp_dir, &entry_path).with_context(|| {
            format!(
                "{label} archive entry escaped output dir: {}",
                entry_path.display()
            )
        })?;
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create {label} extraction parent: {}",
                    parent.display()
                )
            })?;
        }
        entry.unpack(&output_path).with_context(|| {
            format!(
                "Failed to unpack {label} archive entry {}",
                entry_path.display()
            )
        })?;
    }
    Ok(())
}

fn contained_tar_entry_path(temp_dir: &Path, entry_path: &Path) -> Result<PathBuf> {
    let mut output_path = temp_dir.to_path_buf();
    for component in entry_path.components() {
        match component {
            Component::Normal(part) => output_path.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!("unsafe archive path component: {}", entry_path.display());
            }
        }
    }
    if !output_path.starts_with(temp_dir) {
        anyhow::bail!("archive path escaped output dir: {}", entry_path.display());
    }
    Ok(output_path)
}

fn extract_archive_directory(
    archive_path: &Path,
    output_path: &Path,
    label: &str,
    unpack: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let parent = output_path.parent().with_context(|| {
        format!(
            "{label} output path has no parent directory: {}",
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
            "Failed to create temporary {label} extraction dir: {}",
            temp_dir.display()
        )
    })?;

    unpack(&temp_dir)?;

    let extracted_dir = temp_dir.join(output_path.file_name().with_context(|| {
        format!(
            "{label} output path has no directory name: {}",
            output_path.display()
        )
    })?);
    if !extracted_dir.is_dir() {
        anyhow::bail!(
            "{label} archive did not contain expected directory: {}",
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
            "Failed to move extracted {label} from {} to {}",
            extracted_dir.display(),
            output_path.display()
        )
    })?;
    fs::remove_dir_all(&temp_dir).ok();
    fs::remove_file(archive_path).ok();

    if !output_path.starts_with(parent) {
        anyhow::bail!("{label} extraction escaped model root");
    }
    Ok(())
}

#[expect(clippy::cast_precision_loss)]
fn emit_download_progress<R: tauri::Runtime>(
    handle: &AppHandle<R>,
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
        DownloadTarget, DownloadTargetKind, MODEL_ARCHIVE_SHA256_MARKER, NamoTurnDetectorModel,
        asr_model_installed_for, contained_tar_entry_path, default_asr_model_dir_from_root,
        local_translation_model_dir_from_root, local_translation_model_local_source_dir,
        local_translation_models_for_config, local_tts_voices_for_config, model_status_from_root,
        namo_turn_detector_models_for_config, noise_cancellation_model_dir_from_root,
        push_asr_download_targets_from_root,
        push_local_translation_download_targets_with_source_resolver,
        push_local_tts_download_targets, push_noise_cancellation_download_targets,
        push_reazon_accuracy_download_targets,
    };
    use crate::config::{
        AsrConfig, AsrMode, AsrModel, AsrPrecision, LocalTranslationModel, LocalTtsVoice,
        NoiseCancellationConfig, NoiseCancellationModel, ParapperConfig, SegmentationConfig,
        SpeechBackend, SpeechMapping, SpeechSourceKind, SttProfileConfig, SttProfileDisplayColor,
        SttProfileInputConfig, TranslationBackend, TranslationLanguage, TranslationMapping,
        TurnConfig, TurnDetector,
    };
    use crate::model::catalog::{
        asr_model_archive_integrity, asr_model_archive_name, asr_model_required_file_names,
        local_translation_model_required_file_names, local_tts_model_required_file_names,
        noise_cancellation_model_required_file_names,
    };
    use std::{fs, path::Path, time::SystemTime};

    #[test]
    fn japanese_morph_is_ready_without_installed_files() {
        let root = unique_test_models_root("japanese-morph-built-in");
        let config = parapper_config! {
            turn_detector: TurnDetector::Morph,
            ..ParapperConfig::default()
        };

        let status = model_status_from_root(&root, &config)
            .japanese_morph
            .expect("Morph mode should report its built-in dictionary");
        assert!(status.installed);
        assert!(!status.preparing);
    }

    #[test]
    fn namo_models_follow_required_asr_models() {
        let config = parapper_config! {
            multilingual_asr_enabled: true,
            turn_detector: TurnDetector::Namo,
            enabled_asr_models: vec![
                AsrModel::NemoParakeetTdt0_6BV3Int8,
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
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

        assert!(
            namo_turn_detector_models_for_config(&parapper_config! {
                turn_detector: TurnDetector::Morph,
                ..ParapperConfig::default()
            })
            .is_empty()
        );
    }

    #[test]
    fn language_id_and_turn_detector_status_follow_mode_matrix() {
        for turn_detector in [
            TurnDetector::Simple,
            TurnDetector::Namo,
            TurnDetector::Morph,
        ] {
            for multilingual_asr_enabled in [false, true] {
                let config = parapper_config! {
                    multilingual_asr_enabled: multilingual_asr_enabled,
                    turn_detector: turn_detector,
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
                assert_eq!(
                    status.japanese_morph.is_some(),
                    config.requires_japanese_morph_analyzer(),
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
            &parapper_config! {
                noise_cancellation_enabled: true,
                noise_cancellation_model: NoiseCancellationModel::UlUnas,
                ..ParapperConfig::default()
            },
        );
        assert!(enabled.noise_cancellation.is_some());
    }

    #[test]
    fn profile_model_status_requires_each_profiles_asr_precision_and_nested_noise_model() {
        let root = unique_test_models_root("profile-model-status-assets");
        let mut first = stt_profile_for_model_assets("first", AsrModel::ReazonSpeechK2V2);
        first.asr.precision = AsrPrecision::Float32;
        let mut second =
            stt_profile_for_model_assets("second", AsrModel::NemoParakeetTdt0_6BV2Int8);
        second.asr.precision = AsrPrecision::Int8;
        second.noise_cancellation = NoiseCancellationConfig {
            enabled: true,
            model: NoiseCancellationModel::UlUnas,
            ..NoiseCancellationConfig::default()
        };
        let config = ParapperConfig {
            stt_profiles: vec![first.clone(), second.clone()],
            ..ParapperConfig::default()
        };

        write_required_asr_model_files(&root, first.asr.model, first.asr.precision);
        let status = model_status_from_root(&root, &config);
        assert!(
            !status.asr.installed,
            "a downloaded first profile model must not hide a missing second profile ASR asset"
        );
        assert!(
            status
                .noise_cancellation
                .as_ref()
                .is_some_and(|asset| !asset.installed),
            "nested profile NC must be reported even if global NC is disabled"
        );

        write_required_asr_model_files(&root, second.asr.model, second.asr.precision);
        let nc_dir = noise_cancellation_model_dir_from_root(&root, NoiseCancellationModel::UlUnas);
        fs::create_dir_all(&nc_dir).expect("failed to create nested NC model directory");
        for file in noise_cancellation_model_required_file_names(NoiseCancellationModel::UlUnas) {
            fs::write(nc_dir.join(file), b"model").expect("failed to write nested NC model file");
        }

        let status = model_status_from_root(&root, &config);
        assert!(status.asr.installed);
        assert!(
            status
                .noise_cancellation
                .as_ref()
                .is_some_and(|asset| asset.installed),
            "all nested profile assets installed must make the aggregate status ready"
        );
    }

    #[test]
    fn disabled_stt_profile_assets_are_not_reported_or_requested() {
        let root = unique_test_models_root("disabled-profile-assets");
        let enabled = stt_profile_for_model_assets("enabled", AsrModel::ReazonSpeechK2V2);
        let mut disabled =
            stt_profile_for_model_assets("disabled", AsrModel::NemoParakeetTdt0_6BV2Int8);
        disabled.enabled = false;
        disabled.noise_cancellation = NoiseCancellationConfig {
            enabled: true,
            model: NoiseCancellationModel::UlUnas,
            ..NoiseCancellationConfig::default()
        };
        let config = ParapperConfig {
            stt_profiles: vec![enabled.clone(), disabled],
            ..ParapperConfig::default()
        };
        write_required_asr_model_files(&root, enabled.asr.model, enabled.asr.precision);

        let status = model_status_from_root(&root, &config);
        assert!(status.asr.installed);
        assert!(status.noise_cancellation.is_none());

        let mut targets = Vec::new();
        push_asr_download_targets_from_root(&mut targets, &root, &config)
            .expect("disabled profile assets must not be requested");
        push_noise_cancellation_download_targets(&mut targets, &root, &config)
            .expect("disabled profile noise-cancellation assets must not be requested");
        assert!(
            targets.is_empty(),
            "only missing disabled-profile ASR and NC assets remain"
        );
    }

    #[test]
    fn profile_asr_download_deduplicates_a_shared_archive_target() {
        let root = unique_test_models_root("profile-asr-archive-dedup");
        let model = AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8;
        assert!(
            asr_model_archive_name(model).is_some(),
            "this regression requires a catalog model downloaded as an archive"
        );
        let mut first = stt_profile_for_model_assets("first", AsrModel::ReazonSpeechK2V2);
        first.asr.interim_model = Some(model);
        let mut second = stt_profile_for_model_assets("second", AsrModel::ReazonSpeechK2V2);
        second.asr.interim_model = Some(AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8);
        let config = ParapperConfig {
            stt_profiles: vec![first, second],
            ..ParapperConfig::default()
        };
        let mut targets = Vec::new();

        push_asr_download_targets_from_root(&mut targets, &root, &config)
            .expect("profile ASR targets should be collected");

        assert_eq!(
            targets
                .iter()
                .filter(|target| target.kind == DownloadTargetKind::TarBz2Directory)
                .count(),
            1,
            "different runtime latencies from one Nemotron family must enqueue one base archive"
        );
    }

    #[test]
    fn profile_asr_download_unions_required_files_for_same_model_with_different_precisions() {
        let root = unique_test_models_root("profile-asr-precision-union");
        let model = AsrModel::ReazonSpeechK2V2;
        assert!(
            asr_model_archive_name(model).is_none(),
            "this regression requires a catalog model downloaded as individual files"
        );
        let mut float = stt_profile_for_model_assets("float", model);
        float.asr.precision = AsrPrecision::Float32;
        let mut quantized = stt_profile_for_model_assets("int8", model);
        quantized.asr.precision = AsrPrecision::Int8;
        let config = ParapperConfig {
            stt_profiles: vec![float, quantized],
            ..ParapperConfig::default()
        };
        let mut targets = Vec::new();

        push_asr_download_targets_from_root(&mut targets, &root, &config)
            .expect("profile ASR targets should be collected");

        let model_dir = default_asr_model_dir_from_root(&root, model);
        let expected = asr_model_required_file_names(model, AsrPrecision::Float32)
            .iter()
            .chain(asr_model_required_file_names(model, AsrPrecision::Int8))
            .map(|file| model_dir.join(file))
            .collect::<std::collections::HashSet<_>>();
        let actual = targets
            .iter()
            .map(|target| target.output_path.clone())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            actual, expected,
            "same-model profile precisions must download the union of their required files exactly once"
        );
        assert_eq!(
            targets.len(),
            actual.len(),
            "shared files must be deduplicated by concrete output path"
        );
    }

    #[test]
    fn tar_entry_path_must_stay_inside_extraction_dir() {
        let root = Path::new("extracting");

        assert_eq!(
            contained_tar_entry_path(root, Path::new("model/file.onnx"))
                .expect("normal archive path should be accepted"),
            root.join("model").join("file.onnx")
        );
        assert!(
            contained_tar_entry_path(root, Path::new("../escape")).is_err(),
            "parent components must not escape extraction dir"
        );
        assert!(
            contained_tar_entry_path(root, Path::new("/absolute/path")).is_err(),
            "absolute paths must not escape extraction dir"
        );
    }

    #[test]
    fn asr_status_requires_all_enabled_asr_models() {
        let root = unique_test_models_root("model-status-asr");
        let config = parapper_config! {
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
    }

    #[test]
    fn reazon_accuracy_status_requires_static_reranker_assets_only_when_enabled() {
        let root = unique_test_models_root("model-status-reazon-accuracy");
        let mut config = parapper_config! {
            asr_model: AsrModel::ReazonSpeechK2V2,
            asr_precision: AsrPrecision::Int8Float32,
            ..ParapperConfig::default()
        };
        write_required_asr_model_files(
            &root,
            AsrModel::ReazonSpeechK2V2,
            AsrPrecision::Int8Float32,
        );

        assert!(model_status_from_root(&root, &config).asr.installed);

        config.asr.mode = AsrMode::Accurate;
        assert!(!model_status_from_root(&root, &config).asr.installed);

        let reranker_dir = default_asr_model_dir_from_root(&root, AsrModel::ReazonSpeechK2V2)
            .join(parapper_models::asr::backend::REAZON_STATIC_EMBEDDING_DIR_NAME);
        for file in parapper_models::asr::backend::REAZON_STATIC_EMBEDDING_REQUIRED_FILES {
            let path = reranker_dir.join(file);
            fs::create_dir_all(path.parent().expect("reranker file has a parent"))
                .expect("failed to create reranker test dir");
            fs::write(path, b"installed").expect("failed to create reranker test file");
        }

        assert!(model_status_from_root(&root, &config).asr.installed);
    }

    #[test]
    fn parakeet_status_uses_split_ctc_for_fast_and_adds_tdt_files_for_accurate() {
        use parapper_models::asr::backend::parakeet_ja::{
            HYBRID_REQUIRED_FILES, SHARED_CTC_REQUIRED_FILES,
        };

        let root = unique_test_models_root("model-status-parakeet-accuracy");
        let mut config = ParapperConfig::default();
        config.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        config.asr.precision = AsrPrecision::Int8;
        let dir =
            default_asr_model_dir_from_root(&root, AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8);
        fs::create_dir_all(&dir).expect("create Parakeet fixture directory");
        fs::write(dir.join("model.int8.onnx"), b"legacy").expect("write legacy model fixture");
        fs::write(dir.join("tokens.txt"), b"legacy").expect("write legacy tokens fixture");
        assert!(
            !model_status_from_root(&root, &config).asr.installed,
            "legacy monolithic CTC files must not satisfy fast mode"
        );

        for file in SHARED_CTC_REQUIRED_FILES {
            fs::write(dir.join(file), b"shared CTC").expect("write shared CTC fixture");
        }
        assert!(model_status_from_root(&root, &config).asr.installed);

        config.asr.mode = AsrMode::Accurate;
        assert!(!model_status_from_root(&root, &config).asr.installed);
        for file in HYBRID_REQUIRED_FILES {
            fs::write(dir.join(file), b"hybrid TDT").expect("write hybrid TDT fixture");
        }
        assert!(model_status_from_root(&root, &config).asr.installed);
    }

    #[test]
    fn parakeet_split_bundle_downloads_the_files_for_each_mode_from_the_pinned_hf_release() {
        use parapper_models::asr::backend::parakeet_ja::{
            HYBRID_REQUIRED_FILES, SHARED_CTC_REQUIRED_FILES,
        };

        let root = unique_test_models_root("parakeet-split-download-published");
        let model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        let model_dir = default_asr_model_dir_from_root(&root, model);
        let base_url = "https://huggingface.co/nadare/parakeet-tdt_ctc-0.6b-ja-onnx-dynamic-int8/resolve/ab9073e4b457a4eb3df4e362946404be8adc0b1e";

        for (mode, additional_files) in [
            (AsrMode::Fast, &[][..]),
            (AsrMode::Accurate, HYBRID_REQUIRED_FILES),
        ] {
            let config = parapper_config! {
                asr_model: model,
                asr_precision: AsrPrecision::Int8,
                asr_mode: mode,
                ..ParapperConfig::default()
            };
            let mut targets = Vec::new();

            push_asr_download_targets_from_root(&mut targets, &root, &config)
                .expect("published Parakeet files must be scheduled");

            let expected_files = SHARED_CTC_REQUIRED_FILES
                .iter()
                .chain(additional_files)
                .copied()
                .collect::<std::collections::HashSet<_>>();
            let actual_files = targets
                .iter()
                .filter(|target| target.output_path.starts_with(&model_dir))
                .map(|target| target.file_name.as_str())
                .collect::<std::collections::HashSet<_>>();
            assert_eq!(actual_files, expected_files, "mode={mode:?}");
            assert!(
                targets
                    .iter()
                    .filter(|target| target.output_path.starts_with(&model_dir))
                    .all(|target| {
                        target.kind == DownloadTargetKind::File
                            && target.integrity.is_some()
                            && target.url
                                == format!("{base_url}/{}?download=true", target.file_name)
                    }),
                "mode={mode:?}: every Parakeet file must come from the pinned public release"
            );
        }
    }

    #[test]
    fn reazon_accuracy_downloads_only_the_two_files_from_the_pinned_snapshot() {
        let mut config = ParapperConfig::default();
        config.asr.model = AsrModel::ReazonSpeechK2V2;
        let model_dir = Path::new("reazon-model");
        let mut targets = Vec::new();

        push_reazon_accuracy_download_targets(
            &mut targets,
            model_dir,
            &config,
            AsrModel::ReazonSpeechK2V2,
        );
        assert!(targets.is_empty());

        config.asr.mode = AsrMode::Accurate;
        push_reazon_accuracy_download_targets(
            &mut targets,
            model_dir,
            &config,
            AsrModel::ReazonSpeechK2V2,
        );

        assert_eq!(targets.len(), 2);
        assert!(targets.iter().all(|target| {
            target.url.starts_with("https://huggingface.co/hotchpotch/static-embedding-japanese/resolve/95b3d9c80a7ccf604e2b5daee7b1b3eed6b1a9d3/")
                && target.kind == DownloadTargetKind::File
                && target.output_path.starts_with(model_dir.join("static-embedding-japanese"))
                && target.integrity.is_some()
        }));
    }

    #[test]
    fn nemotron_asr_status_reuses_one_560ms_archive_for_every_runtime_latency() {
        for (model, archive_name) in [
            (
                AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8,
                "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
                "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8,
                "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
                "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8,
                "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8,
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8,
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8,
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
            ),
        ] {
            let root = unique_test_models_root("model-status-nemotron-asr");
            let config = parapper_config! {
                interim_asr_model: Some(model),
                ..ParapperConfig::default()
            }
            .normalized();

            assert_eq!(asr_model_archive_name(model).as_deref(), Some(archive_name));

            let missing = model_status_from_root(&root, &config);
            assert!(!missing.asr.installed);

            write_required_asr_model_files(&root, model, AsrPrecision::Int8);
            write_required_asr_model_files(&root, config.asr.model, config.asr.precision);
            let integrity = asr_model_archive_integrity(model).unwrap();
            fs::write(
                default_asr_model_dir_from_root(&root, model).join(MODEL_ARCHIVE_SHA256_MARKER),
                integrity.sha256,
            )
            .unwrap();

            let installed = model_status_from_root(&root, &config);
            assert!(installed.asr.installed);
        }
    }

    #[test]
    fn multilingual_nemotron_without_current_archive_marker_is_not_installed() {
        let root = unique_test_models_root("nemotron-current-archive-marker");
        let model = AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8;
        write_required_asr_model_files(&root, model, AsrPrecision::Int8);
        let model_dir = default_asr_model_dir_from_root(&root, model);

        assert!(!asr_model_installed_for(
            &model_dir,
            model,
            AsrPrecision::Int8
        ));

        let expected = asr_model_archive_integrity(model).unwrap();
        fs::write(model_dir.join(MODEL_ARCHIVE_SHA256_MARKER), expected.sha256).unwrap();
        assert!(asr_model_installed_for(
            &model_dir,
            model,
            AsrPrecision::Int8
        ));
    }

    struct TestModelsRoot {
        path: std::path::PathBuf,
    }

    impl std::ops::Deref for TestModelsRoot {
        type Target = Path;

        fn deref(&self) -> &Self::Target {
            &self.path
        }
    }

    impl Drop for TestModelsRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn unique_test_models_root(name: &str) -> TestModelsRoot {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("{name}-{}-{timestamp}", std::process::id()));
        TestModelsRoot { path }
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

    fn stt_profile_for_model_assets(id: &str, model: AsrModel) -> SttProfileConfig {
        let mut asr = AsrConfig::default();
        asr.language = model.language();
        asr.model = model;
        asr.precision = model.default_precision();
        asr.enabled_models = vec![model];
        SttProfileConfig {
            id: id.to_owned(),
            name: id.to_owned(),
            enabled: true,
            neo_http_enabled: true,
            developer_http_enabled: true,
            display_color: SttProfileDisplayColor::Green,
            input: SttProfileInputConfig {
                device_host: None,
                device_id: None,
                device_name: None,
                channel_index: 0,
                volume_percent: 100,
                muted: false,
            },
            noise_cancellation: NoiseCancellationConfig::default(),
            segmentation: SegmentationConfig::default(),
            turn: TurnConfig::default(),
            asr,
            delivery_profile_id: None,
        }
    }

    #[test]
    fn local_tts_models_follow_speech_mappings() {
        let config = parapper_config! {
            speech_mappings: vec![
                SpeechMapping {
                    id: "tts-kristin".to_string(),
                    source_kind: SpeechSourceKind::Recognition,
                    source_asr_model: None,
                    target_lang: None,
                    backend: SpeechBackend::LocalTts,
                    talker: String::new(),
                    local_tts_voice: Some(LocalTtsVoice::Supertonic2Onnx),
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
                    local_tts_voice: Some(LocalTtsVoice::Supertonic2Onnx),
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
                    local_tts_voice: Some(LocalTtsVoice::Supertonic3Onnx),
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
            vec![
                LocalTtsVoice::Supertonic2Onnx,
                LocalTtsVoice::Supertonic3Onnx
            ]
        );
        assert_eq!(
            model_status_from_root(std::path::Path::new("models"), &config)
                .tts
                .len(),
            2
        );
    }

    #[test]
    fn local_translation_status_uses_selected_distribution_model_files_without_requiring_other_variants()
     {
        let root = unique_test_models_root("model-status-local-translation");
        let disabled =
            model_status_from_root(std::path::Path::new("models"), &ParapperConfig::default());
        assert!(disabled.local_translation.is_none());

        for local_model in [
            LocalTranslationModel::Lfm2Q4,
            LocalTranslationModel::CatTranslate0_8BQ4KQuant,
        ] {
            let root = root.join(format!("{local_model:?}"));
            let config = parapper_config! {
                translation_enabled: true,
                translation_mappings: vec![TranslationMapping {
                    id: "translate-ja-en-local".to_string(),
                    source_asr_model: None,
                    backend: TranslationBackend::Local,
                    local_model,
                    source_lang: TranslationLanguage::Ja,
                    target_lang: TranslationLanguage::En,
                }],
                ..ParapperConfig::default()
            };
            assert_eq!(
                local_translation_models_for_config(&config),
                vec![local_model]
            );

            let missing = model_status_from_root(&root, &config);
            assert_eq!(
                missing
                    .local_translation
                    .as_ref()
                    .map(|status| status.installed),
                Some(false)
            );

            let model_dir = local_translation_model_dir_from_root(&root, local_model);
            for file in local_translation_model_required_file_names(local_model) {
                let path = model_dir.join(file);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)
                        .expect("failed to create local translation file parent");
                }
                fs::write(path, b"test").expect("failed to write local translation required file");
            }

            let installed = model_status_from_root(&root, &config);
            assert_eq!(
                installed
                    .local_translation
                    .as_ref()
                    .map(|status| status.installed),
                Some(true)
            );
        }
    }

    #[test]
    fn duplicate_lfm2_q4_translation_mappings_require_model_once() {
        let config = parapper_config! {
            translation_enabled: true,
            translation_mappings: vec![
                TranslationMapping {
                    id: "translate-ja-en-lfm2-1".to_string(),
                    source_asr_model: None,
                    backend: TranslationBackend::Local,
                    local_model: LocalTranslationModel::Lfm2Q4,
                    source_lang: TranslationLanguage::Ja,
                    target_lang: TranslationLanguage::En,
                },
                TranslationMapping {
                    id: "translate-ja-en-lfm2-2".to_string(),
                    source_asr_model: None,
                    backend: TranslationBackend::Local,
                    local_model: LocalTranslationModel::Lfm2Q4,
                    source_lang: TranslationLanguage::Ja,
                    target_lang: TranslationLanguage::En,
                },
            ],
            ..ParapperConfig::default()
        };

        assert_eq!(
            local_translation_models_for_config(&config),
            vec![LocalTranslationModel::Lfm2Q4]
        );
    }

    #[test]
    fn onnx_community_lfm2_q4_download_targets_use_hugging_face_files() {
        let root = unique_test_models_root("model-status-local-translation-targets");
        let models_root = root.join("models");
        let config = parapper_config! {
            translation_enabled: true,
            translation_mappings: vec![TranslationMapping {
                id: "translate-ja-en-lfm2".to_string(),
                source_asr_model: None,
                backend: TranslationBackend::Local,
                local_model: LocalTranslationModel::Lfm2Q4,
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
            }],
            ..ParapperConfig::default()
        };
        let mut targets = Vec::new();

        push_local_translation_download_targets_with_source_resolver(
            &mut targets,
            &models_root,
            &config,
            |_| None,
        )
        .expect("ONNX Community LFM2 Q4 should not require a local source");

        let expected_len =
            local_translation_model_required_file_names(LocalTranslationModel::Lfm2Q4).len();
        let mut output_paths = targets
            .iter()
            .map(|target| target.output_path.clone())
            .collect::<Vec<_>>();
        output_paths.sort();
        output_paths.dedup();
        assert_eq!(targets.len(), expected_len);
        assert_eq!(targets.len(), output_paths.len());
        assert!(
            targets
                .iter()
                .all(|target| target.kind == DownloadTargetKind::File)
        );
        assert!(
            targets
                .iter()
                .filter(|target| target.file_name != "LICENSE")
                .all(|target| target.url.starts_with(
                    "https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX/resolve/main/"
                ))
        );
        assert_eq!(
            targets
                .iter()
                .find(|target| target.file_name == "LICENSE")
                .map(|target| target.url.as_str()),
            Some(
                "https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT/resolve/80367784d525777ad7565b24534ba5810eeac59f/LICENSE?download=true"
            )
        );
    }

    #[test]
    fn cat_translate_download_targets_use_pinned_hugging_face_files_with_published_integrity() {
        let root = unique_test_models_root("model-status-cat-translation-targets");
        let models_root = root.join("models");
        let config = parapper_config! {
            translation_enabled: true,
            translation_mappings: vec![TranslationMapping {
                id: "translate-ja-en-cat".to_string(),
                source_asr_model: None,
                backend: TranslationBackend::Local,
                local_model: LocalTranslationModel::CatTranslate0_8BQ4KQuant,
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
            }],
            ..ParapperConfig::default()
        };
        let mut targets = Vec::new();

        push_local_translation_download_targets_with_source_resolver(
            &mut targets,
            &models_root,
            &config,
            |_| panic!("published CAT translation model must not use a local export source"),
        )
        .expect("published CAT translation model should use Hugging Face");

        let required_files = local_translation_model_required_file_names(
            LocalTranslationModel::CatTranslate0_8BQ4KQuant,
        );
        assert_eq!(targets.len(), required_files.len());
        assert!(targets.iter().all(|target| {
            target.url.starts_with(
                "https://huggingface.co/nadare/CAT-Translate-0.8b-onnx-q4-k-quant/resolve/a6369bfcaa1f7c9a8df7294c6b2011286e5dc843/",
            ) && target.kind == DownloadTargetKind::File
                && target.integrity.is_some()
        }));
        assert_eq!(
            targets
                .iter()
                .find(|target| target.file_name == "model_q4.onnx.data")
                .and_then(|target| target.integrity),
            Some(crate::model::catalog::FileIntegrity {
                size: 596_894_720,
                sha256: "66839e48f81021eb3f6cf888b57411021914555f705024b15bd76a15e0956480",
            })
        );
    }

    fn quantized_supertonic3_config() -> ParapperConfig {
        parapper_config! {
            speech_mappings: vec![SpeechMapping {
                id: "speech-supertonic3-q4".to_string(),
                source_kind: SpeechSourceKind::Recognition,
                source_asr_model: None,
                target_lang: None,
                backend: SpeechBackend::LocalTts,
                talker: String::new(),
                local_tts_voice: Some(LocalTtsVoice::Supertonic3OnnxQuantized),
                local_tts_language: Some("ja".to_string()),
                local_tts_speaker_id: Some(0),
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: 1.0,
            }],
            ..ParapperConfig::default()
        }
    }

    #[test]
    fn quantized_supertonic3_download_targets_use_the_published_commit_and_integrity() {
        let root = unique_test_models_root("supertonic3-q4-targets");
        let config = quantized_supertonic3_config();
        let mut targets = Vec::new();

        push_local_tts_download_targets(&mut targets, &root, &config)
            .expect("published Supertonic 3 Q4 targets should be created");

        assert_eq!(
            targets.len(),
            local_tts_model_required_file_names(LocalTtsVoice::Supertonic3OnnxQuantized).len()
        );
        assert!(targets.iter().all(|target| {
            target.url.starts_with(
                "https://huggingface.co/nadare/supertonic-3-onnx-q4/resolve/0831a17d4f7de14ade46364ec447d50e24ff1f82/",
            ) && target.kind == DownloadTargetKind::File
                && target.integrity.is_some()
        }));
        assert_eq!(
            targets
                .iter()
                .find(|target| target.file_name == "onnx/vector_estimator.onnx")
                .and_then(|target| target.integrity),
            Some(crate::model::catalog::FileIntegrity {
                size: 51_663_166,
                sha256: "1564c34bdb897c0006349213655979f9a7c573f27effe7ea1417f984d2315b04",
            })
        );
    }

    #[test]
    fn corrupt_quantized_supertonic3_file_is_replaced_from_the_same_distribution() {
        let root = unique_test_models_root("supertonic3-q4-corrupt");
        let model_dir =
            super::local_tts_model_dir_from_root(&root, LocalTtsVoice::Supertonic3OnnxQuantized);
        fs::create_dir_all(model_dir.join("onnx")).expect("failed to create model dir");
        fs::write(model_dir.join("onnx/vector_estimator.onnx"), b"corrupt")
            .expect("failed to write corrupt model fixture");
        let config = quantized_supertonic3_config();
        let mut targets = Vec::new();

        push_local_tts_download_targets(&mut targets, &root, &config)
            .expect("corrupt published file should schedule a replacement");

        assert!(!model_dir.join("onnx/vector_estimator.onnx").exists());
        assert!(targets.iter().any(|target| {
            target.file_name == "onnx/vector_estimator.onnx"
                && target.integrity
                    == Some(crate::model::catalog::FileIntegrity {
                        size: 51_663_166,
                        sha256: "1564c34bdb897c0006349213655979f9a7c573f27effe7ea1417f984d2315b04",
                    })
        }));
    }

    #[test]
    fn listener_model_selection_does_not_change_internal_translation_model_requirements() {
        let config = parapper_config! {
            translation_enabled: false,
            translation_local_server_model: LocalTranslationModel::CatTranslate0_8BQ4KQuant,
            ..ParapperConfig::default()
        };

        assert!(local_translation_models_for_config(&config).is_empty());
        assert!(
            model_status_from_root(std::path::Path::new("models"), &config)
                .local_translation
                .is_none()
        );
    }

    #[test]
    fn corrupt_cat_translate_file_is_replaced_from_the_published_distribution() {
        let root = unique_test_models_root("model-status-cat-translation-corrupt-file");
        let models_root = root.join("models");
        let model_dir = local_translation_model_dir_from_root(
            &models_root,
            LocalTranslationModel::CatTranslate0_8BQ4KQuant,
        );
        fs::create_dir_all(&model_dir).expect("failed to create CAT model dir");
        fs::write(model_dir.join("model_q4.onnx"), b"not the published model")
            .expect("failed to write corrupt CAT model");
        let config = parapper_config! {
            translation_enabled: true,
            translation_mappings: vec![TranslationMapping {
                id: "translate-ja-en-cat".to_string(),
                source_asr_model: None,
                backend: TranslationBackend::Local,
                local_model: LocalTranslationModel::CatTranslate0_8BQ4KQuant,
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
            }],
            ..ParapperConfig::default()
        };
        let mut targets = Vec::new();

        push_local_translation_download_targets_with_source_resolver(
            &mut targets,
            &models_root,
            &config,
            |_| None,
        )
        .expect("corrupt CAT file should schedule a verified replacement");

        assert!(!model_dir.join("model_q4.onnx").exists());
        assert!(targets.iter().any(|target| {
            target.file_name == "model_q4.onnx"
                && target.integrity
                    == Some(crate::model::catalog::FileIntegrity {
                        size: 211_164,
                        sha256: "af6fac6bb8df46ce7cffecde2fca833a92b64d4e46c5d873abb7fc3d60423fc3",
                    })
        }));
    }

    #[test]
    fn published_cat_translate_does_not_use_a_local_export_source() {
        assert_eq!(
            local_translation_model_local_source_dir(
                LocalTranslationModel::CatTranslate0_8BQ4KQuant
            ),
            None
        );
    }

    #[test]
    fn onnx_community_lfm2_q4_does_not_use_a_local_export_source() {
        assert_eq!(
            local_translation_model_local_source_dir(LocalTranslationModel::Lfm2Q4),
            None
        );
    }
}
