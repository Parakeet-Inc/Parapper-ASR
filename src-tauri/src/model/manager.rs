use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::{Duration, SystemTime},
};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tar::Archive;
use tauri::{AppHandle, Emitter, Manager};
use tokio::{fs::File, io::AsyncWriteExt};
use vibrato_rkyv::LoadMode;

use super::catalog::{
    ALL_ASR_MODELS, ALL_LOCAL_TRANSLATION_MODELS, ALL_NAMO_TURN_DETECTOR_MODELS,
    ALL_NOISE_CANCELLATION_MODELS, FileIntegrity, NamoTurnDetectorModel, VAD_MODEL_URL,
    VIBRATO_MODEL_MAGIC, asr_model_archive_name, asr_model_base_url, asr_model_dir_name,
    asr_model_required_file_names, language_id_model_base_url, language_id_model_dir_name,
    language_id_model_files, local_translation_model_base_url, local_translation_model_dir_name,
    local_translation_model_file_integrity, local_translation_model_required_file_names,
    local_tts_model_archive_name, local_tts_model_base_url, local_tts_model_file_integrity,
    local_tts_model_required_dir_names, local_tts_model_required_file_names,
    namo_turn_detector_base_url, namo_turn_detector_dir_name, namo_turn_detector_files,
    noise_cancellation_model_base_url, noise_cancellation_model_dir_name,
    noise_cancellation_model_required_file_names, supertonic_tts_model_base_url,
    vibrato_unidic_archive_integrity, vibrato_unidic_archive_name, vibrato_unidic_archive_url,
    vibrato_unidic_dir_name, vibrato_unidic_expanded_integrity,
};
use crate::config::{
    ALL_LOCAL_TTS_VOICES, AsrModel, AsrPrecision, LocalTranslationModel, LocalTtsFamily,
    LocalTtsVoice, NoiseCancellationModel, ParapperConfig, SpeechBackend, TranslationBackend,
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

    fn new_with_preparing(path: &Path, installed: bool, preparing: bool) -> Self {
        Self {
            installed,
            preparing,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadTargetKind {
    File,
    LocalFile,
    TarBz2Directory,
    TarZstdJapaneseMorphDirectory,
}

const STALE_EXTRACTION_MARKER_AGE: Duration = Duration::from_secs(60 * 60 * 6);
pub fn models_root(handle: &AppHandle) -> Result<PathBuf> {
    Ok(handle.path().app_data_dir()?.join("models"))
}

pub fn vad_model_path_from_root(root: &Path) -> PathBuf {
    root.join("silero_vad_v6").join("silero_vad.onnx")
}

pub fn vad_model_path(handle: &AppHandle) -> Result<PathBuf> {
    Ok(vad_model_path_from_root(&models_root(handle)?))
}

pub fn japanese_morph_model_dir_from_root(root: &Path) -> PathBuf {
    root.join(vibrato_unidic_dir_name())
}

pub fn japanese_morph_dictionary_paths_from_root(root: &Path) -> Vec<PathBuf> {
    japanese_morph_dictionary_paths_from_model_dir(&japanese_morph_model_dir_from_root(root))
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
    let asr_path = asr_model_dir_from_root(root, config);
    let asr_installed = config.required_asr_models().into_iter().all(|model| {
        let model_dir = if model == config.asr.model {
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
        japanese_morph: japanese_morph_required(config).then(|| {
            let path = japanese_morph_model_dir_from_root(root);
            let installed = japanese_morph_model_installed(&path);
            ModelAssetStatus::new_with_preparing(
                &path,
                installed,
                !installed && japanese_morph_model_preparing(&path),
            )
        }),
        language_id: config.asr.multilingual_enabled.then(|| {
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
        noise_cancellation: config.noise_cancellation.enabled.then(|| {
            let path =
                noise_cancellation_model_dir_from_root(root, config.noise_cancellation.model);
            ModelAssetStatus::new(
                &path,
                noise_cancellation_model_installed(&path, config.noise_cancellation.model),
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

    let japanese_morph_dir = japanese_morph_model_dir_from_root(root);
    if japanese_morph_model_installed(&japanese_morph_dir) {
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
    push_japanese_morph_download_targets(&mut targets, &root, config)?;
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

fn push_japanese_morph_download_targets(
    targets: &mut Vec<DownloadTarget>,
    root: &Path,
    config: &ParapperConfig,
) -> Result<()> {
    if !japanese_morph_required(config) {
        return Ok(());
    }

    let model_dir = japanese_morph_model_dir_from_root(root);
    if japanese_morph_model_installed(&model_dir) {
        return Ok(());
    }
    fs::create_dir_all(root)
        .with_context(|| format!("Failed to create model root dir: {}", root.display()))?;
    targets.push(DownloadTarget {
        url: vibrato_unidic_archive_url().to_string(),
        output_path: model_dir,
        file_name: vibrato_unidic_archive_name().to_string(),
        kind: DownloadTargetKind::TarZstdJapaneseMorphDirectory,
        integrity: Some(vibrato_unidic_archive_integrity()),
    });
    Ok(())
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
    for model in config.required_asr_models() {
        let asr_path = asr_model_dir_for(handle, config, model)?;
        fs::create_dir_all(&asr_path)
            .with_context(|| format!("Failed to create ASR model dir: {}", asr_path.display()))?;
        let precision = config.asr_precision_for(model);
        if asr_model_installed_for(&asr_path, model, precision) {
            continue;
        }
        if let Some(archive_name) = asr_model_archive_name(model) {
            targets.push(DownloadTarget {
                url: format!("{}/{}", asr_model_base_url(model), archive_name),
                output_path: asr_path,
                file_name: archive_name,
                kind: DownloadTargetKind::TarBz2Directory,
                integrity: None,
            });
            continue;
        }
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
    if !config.asr.multilingual_enabled {
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
    asr_model_required_file_names(model, precision)
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn language_id_model_installed(model_dir: &std::path::Path) -> bool {
    language_id_model_files()
        .iter()
        .all(|file| model_dir.join(file).is_file())
}

fn japanese_morph_model_installed(model_dir: &Path) -> bool {
    if let Err(err) = recover_interrupted_directory_replacement(model_dir) {
        log::warn!(
            "Failed to clean up interrupted Japanese morph dictionary replacement at {}: {err}",
            model_dir.display()
        );
    }
    japanese_morph_model_present_for_integrity(model_dir, vibrato_unidic_expanded_integrity())
}

fn japanese_morph_model_present_for_integrity(model_dir: &Path, expected: FileIntegrity) -> bool {
    let dictionary_path = model_dir.join("system.dic");
    japanese_morph_install_manifest_compatible(model_dir, expected)
        && dictionary_path
            .metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == expected.size)
}

fn japanese_morph_model_installed_for_integrity(model_dir: &Path, expected: FileIntegrity) -> bool {
    let dictionary_path = model_dir.join("system.dic");
    japanese_morph_install_manifest_compatible(model_dir, expected)
        && verify_file_integrity(
            &dictionary_path,
            expected,
            "installed Japanese morph dictionary",
        )
        .is_ok()
}

fn japanese_morph_install_manifest_compatible(model_dir: &Path, expected: FileIntegrity) -> bool {
    let Ok(contents) = fs::read(model_dir.join("manifest.json")) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_slice::<serde_json::Value>(&contents) else {
        return false;
    };
    manifest
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        == Some(1)
        && manifest
            .get("dictionary_id")
            .and_then(serde_json::Value::as_str)
            == Some(vibrato_unidic_dir_name())
        && manifest
            .get("representation")
            .and_then(serde_json::Value::as_str)
            == Some("compact-raw")
        && manifest
            .get("feature_encoding")
            .and_then(serde_json::Value::as_str)
            == Some("[PP][S][F]")
        && manifest
            .pointer("/expanded_dictionary/size")
            .and_then(serde_json::Value::as_u64)
            == Some(expected.size)
        && manifest
            .pointer("/expanded_dictionary/sha256")
            .and_then(serde_json::Value::as_str)
            == Some(expected.sha256)
}

fn japanese_morph_model_preparing(model_dir: &Path) -> bool {
    model_dir.with_extension("download").is_file()
        || extraction_marker_is_active(&model_dir.with_extension("extracting"))
}

fn extraction_marker_is_active(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    let Ok(modified) = metadata.modified() else {
        return true;
    };
    let Ok(age) = SystemTime::now().duration_since(modified) else {
        return true;
    };
    if age <= STALE_EXTRACTION_MARKER_AGE {
        return true;
    }
    if let Err(err) = fs::remove_dir_all(path) {
        log::warn!(
            "Failed to remove stale model extraction marker {}: {err}",
            path.display()
        );
    }
    false
}

fn japanese_morph_dictionary_paths_from_model_dir(model_dir: &Path) -> Vec<PathBuf> {
    vec![model_dir.join("system.dic")]
}

fn materialize_rkyv_japanese_morph_dictionary(model_dir: &Path) -> Result<()> {
    let compressed_path = model_dir.join("system.dic.zst");
    if !compressed_path.is_file() {
        anyhow::bail!(
            "Japanese morph archive did not contain expected dictionary: {}",
            compressed_path.display()
        );
    }
    let output_path = model_dir.join("system.dic");
    materialize_zstd_japanese_morph_dictionary_as_rkyv(&compressed_path, &output_path)?;
    Ok(())
}

fn materialize_zstd_japanese_morph_dictionary_as_rkyv(
    compressed_path: &Path,
    output_path: &Path,
) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create Japanese morph dictionary dir: {}",
                parent.display()
            )
        })?;
    }
    let temporary_path = output_path.with_extension("dic.transcoding");
    decompress_zstd_dictionary_to_file(compressed_path, &temporary_path)?;
    if japanese_morph_dictionary_compatible(&temporary_path)? {
        fs::rename(&temporary_path, output_path).with_context(|| {
            format!(
                "Failed to move rkyv Japanese morph dictionary from {} to {}",
                temporary_path.display(),
                output_path.display()
            )
        })?;
        return Ok(());
    }
    fs::remove_file(&temporary_path).ok();
    anyhow::bail!(
        "Japanese morph dictionary is not the expected Vibrato rkyv format: {}",
        compressed_path.display()
    )
}

fn decompress_zstd_dictionary_to_file(compressed_path: &Path, output_path: &Path) -> Result<()> {
    let input = fs::File::open(compressed_path).with_context(|| {
        format!(
            "Failed to open compressed Japanese morph dictionary: {}",
            compressed_path.display()
        )
    })?;
    let mut decoder = zstd::Decoder::new(input).with_context(|| {
        format!(
            "Failed to decode Japanese morph dictionary: {}",
            compressed_path.display()
        )
    })?;
    let mut output = fs::File::create(output_path).with_context(|| {
        format!(
            "Failed to create rkyv Japanese morph dictionary: {}",
            output_path.display()
        )
    })?;
    std::io::copy(&mut decoder, &mut output).with_context(|| {
        format!(
            "Failed to write decompressed Japanese morph dictionary: {}",
            output_path.display()
        )
    })?;
    output.flush().with_context(|| {
        format!(
            "Failed to flush rkyv Japanese morph dictionary: {}",
            output_path.display()
        )
    })?;
    Ok(())
}

fn wait_for_rkyv_dictionary_mmap(path: &Path) -> Result<()> {
    const ATTEMPTS: usize = 10;
    const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(20);
    const MAX_RETRY_DELAY: Duration = Duration::from_secs(1);

    let mut last_error = None;
    let mut retry_delay = INITIAL_RETRY_DELAY;
    for attempt in 0..ATTEMPTS {
        // Validate without writing Vibrato's global proof cache. Parallel installs can otherwise
        // contend on the same metadata-derived cache file on Windows and report access denied.
        match vibrato_rkyv::Dictionary::from_path(path, LoadMode::Validate) {
            Ok(dictionary) => {
                drop(dictionary);
                return Ok(());
            }
            Err(err) => {
                last_error = Some(err);
                if attempt + 1 < ATTEMPTS {
                    std::thread::sleep(retry_delay);
                    retry_delay = retry_delay.saturating_mul(2).min(MAX_RETRY_DELAY);
                }
            }
        }
    }
    Err(anyhow::anyhow!(
        "Failed to mmap materialized Japanese morph dictionary {} after {ATTEMPTS} attempts: {}",
        path.display(),
        last_error.expect("at least one mmap attempt")
    ))
}

fn japanese_morph_dictionary_compatible(path: &Path) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let mut file = fs::File::open(path).with_context(|| {
        format!(
            "Failed to open Japanese morph dictionary: {}",
            path.display()
        )
    })?;
    let mut magic = vec![0; VIBRATO_MODEL_MAGIC.len()];
    file.read_exact(&mut magic).with_context(|| {
        format!(
            "Failed to read Japanese morph dictionary: {}",
            path.display()
        )
    })?;
    Ok(magic == VIBRATO_MODEL_MAGIC)
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
        if voice.family() == LocalTtsFamily::Supertonic {
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
        } else if !local_tts_model_archive_installed(&model_dir, voice) {
            let archive_name = local_tts_model_archive_name(voice);
            targets.push(DownloadTarget {
                url: format!("{}/{}", local_tts_model_base_url(), archive_name),
                output_path: model_dir,
                file_name: archive_name,
                kind: DownloadTargetKind::TarBz2Directory,
                integrity: None,
            });
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
    if let Some(base_url) = local_translation_model_base_url(model) {
        push_missing_local_translation_file_targets(
            targets,
            &model_dir,
            model,
            required_files,
            base_url,
        )?;
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
    base_url: &str,
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
    if !config.noise_cancellation.enabled {
        return Ok(());
    }

    let model_dir = noise_cancellation_model_dir_from_root(root, config.noise_cancellation.model);
    fs::create_dir_all(&model_dir).with_context(|| {
        format!(
            "Failed to create noise cancellation model dir: {}",
            model_dir.display()
        )
    })?;
    push_missing_file_targets_with_query(
        targets,
        &model_dir,
        noise_cancellation_model_required_file_names(config.noise_cancellation.model),
        noise_cancellation_model_base_url(config.noise_cancellation.model),
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
        .speech
        .mappings
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
        LocalTtsVoice::Supertonic3OnnxQuantized => 5,
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
    if let Some(integrity) = target.integrity {
        if let Err(err) = verify_file_integrity(temporary_path, integrity, "cached model archive") {
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
        .map(|metadata| metadata.len())
        .unwrap_or(0);
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
        }
        DownloadTargetKind::TarZstdJapaneseMorphDirectory => {
            install_zstd_japanese_morph_directory(temporary_path, &target.output_path)?;
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

fn install_zstd_japanese_morph_directory(archive_path: &Path, output_path: &Path) -> Result<()> {
    recover_interrupted_directory_replacement(output_path)?;

    let temp_dir = output_path.with_extension("extracting");
    if temp_dir.is_dir() {
        fs::remove_dir_all(&temp_dir).with_context(|| {
            format!(
                "Failed to remove temporary Japanese morph extraction dir: {}",
                temp_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&temp_dir).with_context(|| {
        format!(
            "Failed to create temporary Japanese morph extraction dir: {}",
            temp_dir.display()
        )
    })?;

    let file = fs::File::open(archive_path).with_context(|| {
        format!(
            "Failed to open Japanese morph dictionary archive: {}",
            archive_path.display()
        )
    })?;
    let decoder = zstd::Decoder::new(file).with_context(|| {
        format!(
            "Failed to decode Japanese morph dictionary archive: {}",
            archive_path.display()
        )
    })?;
    let mut archive = Archive::new(decoder);
    unpack_tar_entries_within(&mut archive, &temp_dir, "Japanese morph dictionary").with_context(
        || {
            format!(
                "Failed to extract Japanese morph dictionary archive: {}",
                archive_path.display()
            )
        },
    )?;

    let extracted_dir = temp_dir.join(vibrato_unidic_dir_name());
    if !extracted_dir.is_dir() {
        anyhow::bail!(
            "Japanese morph archive did not contain expected directory: {}",
            extracted_dir.display()
        );
    }
    materialize_rkyv_japanese_morph_dictionary(&extracted_dir)?;
    let expanded_path = extracted_dir.join("system.dic");
    verify_file_integrity(
        &expanded_path,
        vibrato_unidic_expanded_integrity(),
        "expanded Japanese morph dictionary",
    )?;
    wait_for_rkyv_dictionary_mmap(&expanded_path)?;

    replace_directory_with_staged(&extracted_dir, output_path)?;
    fs::remove_dir_all(&temp_dir).ok();
    fs::remove_file(archive_path).ok();
    Ok(())
}

fn replacement_backup_path(output_path: &Path) -> PathBuf {
    output_path.with_extension("replacing")
}

fn recover_interrupted_directory_replacement(output_path: &Path) -> Result<()> {
    recover_interrupted_directory_replacement_with(output_path, |path| {
        japanese_morph_model_installed_for_integrity(path, vibrato_unidic_expanded_integrity())
    })
}

fn recover_interrupted_directory_replacement_with(
    output_path: &Path,
    output_is_valid: impl FnOnce(&Path) -> bool,
) -> Result<()> {
    let backup_path = replacement_backup_path(output_path);
    if !backup_path.is_dir() {
        return Ok(());
    }
    if output_path.is_dir() {
        if output_is_valid(output_path) {
            fs::remove_dir_all(&backup_path).with_context(|| {
                format!(
                    "Failed to remove stale Japanese morph backup: {}",
                    backup_path.display()
                )
            })?;
        } else {
            fs::remove_dir_all(output_path).with_context(|| {
                format!(
                    "Failed to remove incomplete Japanese morph dictionary: {}",
                    output_path.display()
                )
            })?;
            fs::rename(&backup_path, output_path).with_context(|| {
                format!(
                    "Failed to restore Japanese morph dictionary from {} to {}",
                    backup_path.display(),
                    output_path.display()
                )
            })?;
        }
    } else {
        fs::rename(&backup_path, output_path).with_context(|| {
            format!(
                "Failed to restore Japanese morph dictionary from {} to {}",
                backup_path.display(),
                output_path.display()
            )
        })?;
    }
    Ok(())
}

fn replace_directory_with_staged(staged_path: &Path, output_path: &Path) -> Result<()> {
    let backup_path = replacement_backup_path(output_path);
    let had_existing = output_path.is_dir();
    if had_existing {
        fs::rename(output_path, &backup_path).with_context(|| {
            format!(
                "Failed to preserve existing Japanese morph dictionary at {}",
                output_path.display()
            )
        })?;
    }

    if let Err(err) = fs::rename(staged_path, output_path) {
        if had_existing {
            fs::rename(&backup_path, output_path).with_context(|| {
                format!(
                    "Failed to restore existing Japanese morph dictionary after install error: {err}"
                )
            })?;
        }
        return Err(err).with_context(|| {
            format!(
                "Failed to install Japanese morph dictionary from {} to {}",
                staged_path.display(),
                output_path.display()
            )
        });
    }

    if had_existing && let Err(err) = fs::remove_dir_all(&backup_path) {
        log::warn!(
            "Failed to remove replaced Japanese morph dictionary {}: {err}",
            backup_path.display()
        );
    }
    Ok(())
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
        DownloadTarget, DownloadTargetKind, NamoTurnDetectorModel, cached_archive_available,
        contained_tar_entry_path, default_asr_model_dir_from_root, download_file,
        install_zstd_japanese_morph_directory, japanese_morph_dictionary_compatible,
        japanese_morph_dictionary_paths_from_model_dir, japanese_morph_model_installed,
        japanese_morph_model_installed_for_integrity, japanese_morph_model_present_for_integrity,
        local_translation_model_dir_from_root, local_translation_model_local_source_dir,
        local_translation_models_for_config, local_tts_voices_for_config,
        materialize_rkyv_japanese_morph_dictionary, model_status_from_root,
        namo_turn_detector_models_for_config, push_japanese_morph_download_targets,
        push_local_translation_download_targets_with_source_resolver,
        push_local_tts_download_targets, recover_interrupted_directory_replacement,
        recover_interrupted_directory_replacement_with, replace_directory_with_staged,
        replacement_backup_path, sha256_file, verify_file_integrity,
    };
    use crate::config::{
        AsrModel, AsrPrecision, LocalTranslationModel, LocalTtsVoice, NoiseCancellationModel,
        ParapperConfig, SpeechBackend, SpeechMapping, SpeechSourceKind, TranslationBackend,
        TranslationLanguage, TranslationMapping, TurnDetector,
    };
    use crate::model::catalog::{
        VIBRATO_MODEL_MAGIC, asr_model_archive_name, asr_model_required_file_names,
        local_translation_model_required_file_names, local_tts_model_required_file_names,
        vibrato_unidic_archive_integrity, vibrato_unidic_archive_name, vibrato_unidic_archive_url,
        vibrato_unidic_expanded_integrity,
    };
    use std::{
        fs,
        path::Path,
        time::{Duration, SystemTime},
    };
    use tauri::Manager as _;

    #[test]
    fn japanese_morph_download_uses_public_parapper_asr_release_asset() {
        let root = unique_test_models_root("japanese-morph-public-release-target");
        let config = parapper_config! {
            turn_detector: TurnDetector::Namo,
            ..ParapperConfig::default()
        };
        let mut targets = Vec::new();

        push_japanese_morph_download_targets(&mut targets, &root, &config)
            .expect("Japanese morph download target should be created");

        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].url,
            "https://github.com/Parakeet-Inc/Parapper-ASR/releases/download/morph-dictionary-unidic-cwj-3.1.1-v1/parapper-unidic-cwj-3_1_1-compact-raw-v1.tar.zst"
        );
        assert_eq!(
            targets[0].file_name,
            "parapper-unidic-cwj-3_1_1-compact-raw-v1.tar.zst"
        );
        assert_eq!(
            targets[0].kind,
            DownloadTargetKind::TarZstdJapaneseMorphDirectory
        );
        assert_eq!(
            targets[0].integrity,
            Some(crate::model::catalog::FileIntegrity {
                size: 7_434_191,
                sha256: "a1dd0e62ae87f4631ade3aa46cf00b7fdbed1827893dd1d1bde2765009e125ea",
            })
        );
    }

    #[test]
    fn invalid_compact_raw_payload_does_not_replace_existing_dictionary() {
        let root = unique_test_models_root("japanese-morph-invalid-package");
        let model_dir = root.join("unidic-cwj-3_1_1");
        fs::create_dir_all(&model_dir).expect("failed to create existing dictionary dir");
        fs::write(model_dir.join("system.dic"), b"known working dictionary")
            .expect("failed to write existing dictionary");
        fs::write(model_dir.join("keep.txt"), b"keep this installation")
            .expect("failed to write existing installation sentinel");

        let archive_path = root.join("invalid-compact-raw.tar.zst");
        write_invalid_compact_raw_archive(&archive_path);

        install_zstd_japanese_morph_directory(&archive_path, &model_dir)
            .expect_err("non-published dictionary payload must be rejected");
        assert_eq!(
            fs::read(model_dir.join("system.dic"))
                .expect("existing dictionary should still be readable"),
            b"known working dictionary"
        );
        assert_eq!(
            fs::read(model_dir.join("keep.txt"))
                .expect("existing installation sentinel should still exist"),
            b"keep this installation"
        );
        assert!(!root.join("unidic-cwj-3_1_1.replacing").exists());
    }

    #[test]
    fn cached_archive_is_retained_on_install_failure_and_removed_only_on_integrity_failure() {
        let root = unique_test_models_root("japanese-morph-cached-archive-policy");
        fs::create_dir_all(&*root).expect("failed to create cached archive test dir");
        let model_dir = root.join("unidic-cwj-3_1_1");
        fs::create_dir_all(&model_dir).expect("failed to create existing dictionary dir");
        fs::write(model_dir.join("keep.txt"), b"known working dictionary")
            .expect("failed to write existing dictionary sentinel");
        let archive_path = model_dir.with_extension("download");
        write_invalid_compact_raw_archive(&archive_path);
        let sha256 =
            sha256_file(&archive_path, "test cached archive").expect("failed to hash test archive");
        let target = DownloadTarget {
            url: "https://example.invalid/dictionary.tar.zst".to_string(),
            output_path: model_dir.clone(),
            file_name: "dictionary.tar.zst".to_string(),
            kind: DownloadTargetKind::TarZstdJapaneseMorphDirectory,
            integrity: Some(crate::model::catalog::FileIntegrity {
                size: archive_path
                    .metadata()
                    .expect("failed to inspect test archive")
                    .len(),
                sha256: Box::leak(sha256.into_boxed_str()),
            }),
        };
        assert!(
            cached_archive_available(&target, &archive_path)
                .expect("valid cache inspection should succeed")
        );
        install_zstd_japanese_morph_directory(&archive_path, &model_dir)
            .expect_err("inner dictionary contract should make installation fail");
        assert!(
            archive_path.is_file(),
            "a hash-valid cache should survive a temporary installation failure"
        );
        assert_eq!(
            fs::read(model_dir.join("keep.txt"))
                .expect("existing dictionary should survive cached install failure"),
            b"known working dictionary"
        );

        let mut corrupted = fs::read(&archive_path).expect("failed to read cached archive");
        corrupted[0] ^= 0xff;
        fs::write(&archive_path, corrupted).expect("failed to corrupt cached archive");
        assert!(
            !cached_archive_available(&target, &archive_path)
                .expect("integrity mismatch should be handled as a cache miss")
        );
        assert!(
            !archive_path.exists(),
            "only an integrity-invalid cached archive should be discarded"
        );
    }

    #[tokio::test]
    async fn model_download_path_requests_only_configured_url_and_preserves_existing_dictionary() {
        let root = unique_test_models_root("japanese-morph-http-download-contract");
        fs::create_dir_all(&*root).expect("failed to create HTTP download test dir");
        let source_archive = root.join("server-response.tar.zst");
        write_invalid_compact_raw_archive(&source_archive);
        let response_bytes =
            fs::read(&source_archive).expect("failed to read HTTP response fixture");
        let response_size =
            u64::try_from(response_bytes.len()).expect("HTTP response size should fit u64");
        fs::remove_file(&source_archive).expect("failed to remove HTTP response fixture");
        let response_sha256 = {
            let path = root.join("hash-input.tar.zst");
            fs::write(&path, &response_bytes).expect("failed to write hash input");
            let sha256 =
                sha256_file(&path, "HTTP response fixture").expect("failed to hash HTTP response");
            fs::remove_file(path).expect("failed to remove hash input");
            sha256
        };

        let server =
            tiny_http::Server::http(("127.0.0.1", 0)).expect("test server should bind locally");
        let address = server
            .server_addr()
            .to_ip()
            .expect("test server should use an IP address");
        let server_thread = std::thread::spawn(move || {
            let request = server
                .recv_timeout(Duration::from_secs(2))
                .expect("test server receive should succeed")
                .expect("production download should issue one request");
            let requested_path = request.url().to_string();
            request
                .respond(tiny_http::Response::from_data(response_bytes))
                .expect("test server should respond");
            let extra_request = server
                .recv_timeout(Duration::from_millis(300))
                .expect("extra request check should succeed")
                .map(|request| request.url().to_string());
            (requested_path, extra_request)
        });

        let model_dir = root.join("unidic-cwj-3_1_1");
        fs::create_dir_all(&model_dir).expect("failed to create existing dictionary dir");
        fs::write(model_dir.join("keep.txt"), b"known working dictionary")
            .expect("failed to write existing dictionary sentinel");
        let target = DownloadTarget {
            url: format!("http://{address}/only-this-dictionary.tar.zst"),
            output_path: model_dir.clone(),
            file_name: "only-this-dictionary.tar.zst".to_string(),
            kind: DownloadTargetKind::TarZstdJapaneseMorphDirectory,
            integrity: Some(crate::model::catalog::FileIntegrity {
                size: response_size,
                sha256: Box::leak(response_sha256.into_boxed_str()),
            }),
        };
        let app = tauri::test::mock_builder()
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .expect("failed to build test app");

        download_file(app.handle(), &target, 0, 1)
            .await
            .expect_err("inner dictionary identity should reject the HTTP fixture");
        let (requested_path, extra_request) =
            server_thread.join().expect("test server should stop");

        assert_eq!(requested_path, "/only-this-dictionary.tar.zst");
        assert_eq!(extra_request, None, "download must not try a fallback URL");
        assert_eq!(
            fs::read(model_dir.join("keep.txt"))
                .expect("existing dictionary should survive failed HTTP install"),
            b"known working dictionary"
        );
    }

    #[test]
    fn failed_final_directory_swap_restores_existing_dictionary() {
        let root = unique_test_models_root("japanese-morph-swap-rollback");
        let model_dir = root.join("unidic-cwj-3_1_1");
        fs::create_dir_all(&model_dir).expect("failed to create existing dictionary dir");
        fs::write(model_dir.join("keep.txt"), b"known working dictionary")
            .expect("failed to write existing dictionary sentinel");
        let missing_staged_dir = root.join("missing-staged-dictionary");

        replace_directory_with_staged(&missing_staged_dir, &model_dir)
            .expect_err("missing staged directory should make final rename fail");

        assert_eq!(
            fs::read(model_dir.join("keep.txt"))
                .expect("existing dictionary should be restored after swap failure"),
            b"known working dictionary"
        );
        assert!(
            !replacement_backup_path(&model_dir).exists(),
            "rollback should consume the temporary backup"
        );
    }

    #[test]
    fn interrupted_dictionary_swap_restores_backup_when_new_output_is_incomplete() {
        for output_exists in [false, true] {
            let root =
                unique_test_models_root(&format!("japanese-morph-swap-recovery-{output_exists}"));
            let model_dir = root.join("unidic-cwj-3_1_1");
            let backup_dir = replacement_backup_path(&model_dir);
            fs::create_dir_all(&backup_dir).expect("failed to create interrupted backup");
            fs::write(backup_dir.join("state.txt"), b"old")
                .expect("failed to write old backup state");
            if output_exists {
                fs::create_dir_all(&model_dir).expect("failed to create new install");
                fs::write(model_dir.join("state.txt"), b"new")
                    .expect("failed to write new install state");
            }

            recover_interrupted_directory_replacement(&model_dir)
                .expect("interrupted replacement should recover deterministically");

            assert_eq!(
                fs::read(model_dir.join("state.txt"))
                    .expect("recovered dictionary state should exist"),
                b"old"
            );
            assert!(
                !backup_dir.exists(),
                "recovery should leave a single canonical dictionary directory"
            );
        }
    }

    #[test]
    fn interrupted_dictionary_swap_discards_backup_only_after_new_output_is_valid() {
        let root = unique_test_models_root("japanese-morph-swap-valid-output");
        let model_dir = root.join("unidic-cwj-3_1_1");
        let backup_dir = replacement_backup_path(&model_dir);
        fs::create_dir_all(&backup_dir).expect("failed to create interrupted backup");
        fs::write(backup_dir.join("state.txt"), b"old").expect("failed to write old backup");
        fs::create_dir_all(&model_dir).expect("failed to create new install");
        fs::write(model_dir.join("state.txt"), b"new").expect("failed to write new install");

        recover_interrupted_directory_replacement_with(&model_dir, |path| {
            fs::read(path.join("state.txt")).is_ok_and(|state| state == b"new")
        })
        .expect("valid new output should complete interrupted replacement");

        assert_eq!(
            fs::read(model_dir.join("state.txt")).expect("new dictionary should remain"),
            b"new"
        );
        assert!(!backup_dir.exists());
    }

    #[tokio::test]
    #[ignore = "downloads the published GitHub Release asset"]
    async fn published_compact_raw_v1_asset_installs_and_mmap_loads() {
        let root = unique_test_models_root("japanese-morph-published-package");
        fs::create_dir_all(&*root).expect("failed to create published package test dir");
        let archive_path = root.join(vibrato_unidic_archive_name());
        let response = reqwest::get(vibrato_unidic_archive_url())
            .await
            .expect("published dictionary download should start")
            .error_for_status()
            .expect("published dictionary URL should return success");
        assert_eq!(
            response.content_length(),
            Some(vibrato_unidic_archive_integrity().size)
        );
        let bytes = response
            .bytes()
            .await
            .expect("published dictionary body should download");
        fs::write(&archive_path, &bytes).expect("downloaded archive should be saved");

        verify_file_integrity(
            &archive_path,
            vibrato_unidic_archive_integrity(),
            "published Japanese morph archive",
        )
        .expect("published archive should match the pinned release identity");
        let model_dir = root.join("unidic-cwj-3_1_1");
        install_zstd_japanese_morph_directory(&archive_path, &model_dir)
            .expect("published compact Raw dictionary should install");

        let dictionary_path = model_dir.join("system.dic");
        verify_file_integrity(
            &dictionary_path,
            vibrato_unidic_expanded_integrity(),
            "installed Japanese morph dictionary",
        )
        .expect("installed dictionary should match the pinned expanded identity");
        assert!(
            ["AUTHORS", "BSD", "NOTICE", "manifest.json", "SHA256SUMS"]
                .iter()
                .all(|name| model_dir.join(name).is_file()),
            "license, attribution, manifest, and checksums should remain beside the dictionary"
        );
        let dictionary = vibrato_rkyv::Dictionary::from_path(
            &dictionary_path,
            vibrato_rkyv::LoadMode::TrustCache,
        )
        .expect("installed dictionary should mmap-load through vibrato-rkyv");
        drop(dictionary);
        assert!(
            japanese_morph_model_installed(&model_dir),
            "the installed public release should satisfy production status checks"
        );
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
    fn explicit_japanese_morph_integrity_check_rejects_same_size_dictionary_corruption() {
        let root = unique_test_models_root("model-status-japanese-morph");
        let model_dir = root.join("unidic-cwj-3_1_1");
        let expected = write_installed_compact_raw_dictionary(&model_dir);
        assert!(
            japanese_morph_model_installed_for_integrity(&model_dir, expected),
            "exact dictionary and manifest should be installed"
        );

        let dictionary_path = model_dir.join("system.dic");
        let mut corrupted =
            fs::read(&dictionary_path).expect("failed to read dictionary for corruption test");
        let last = corrupted
            .last_mut()
            .expect("test dictionary should not be empty");
        *last ^= 0xff;
        fs::write(&dictionary_path, corrupted)
            .expect("failed to write same-size corrupted dictionary");

        assert!(
            !japanese_morph_model_installed_for_integrity(&model_dir, expected),
            "same-size corruption must not be accepted from manifest and magic alone"
        );
    }

    #[test]
    fn japanese_morph_status_uses_verified_manifest_and_file_size() {
        let root = unique_test_models_root("model-status-japanese-morph-fast-check");
        let model_dir = root.join("unidic-cwj-3_1_1");
        let expected = write_installed_compact_raw_dictionary(&model_dir);

        let dictionary_path = model_dir.join("system.dic");
        let mut changed =
            fs::read(&dictionary_path).expect("failed to read dictionary for status test");
        changed[0] ^= 0xff;
        fs::write(&dictionary_path, &changed)
            .expect("failed to write same-size dictionary for status test");

        assert!(
            japanese_morph_model_present_for_integrity(&model_dir, expected),
            "routine status checks should trust the verified manifest and exact file size"
        );
        assert!(
            !japanese_morph_model_installed_for_integrity(&model_dir, expected),
            "the explicit integrity check must still detect same-size corruption"
        );

        fs::write(&dictionary_path, &changed[..changed.len() - 1])
            .expect("failed to truncate dictionary for status test");
        assert!(
            !japanese_morph_model_present_for_integrity(&model_dir, expected),
            "routine status checks must reject a dictionary with the wrong size"
        );
    }

    #[test]
    fn magic_only_legacy_dictionary_without_compact_raw_marker_schedules_upgrade() {
        let root = unique_test_models_root("model-status-japanese-morph-legacy");
        let model_dir = root.join("unidic-cwj-3_1_1");
        let dictionary_path = model_dir.join("system.dic");
        fs::create_dir_all(&model_dir).expect("failed to create dictionary parent");
        fs::write(&dictionary_path, VIBRATO_MODEL_MAGIC)
            .expect("failed to write legacy dictionary marker");
        let config = parapper_config! {
            turn_detector: TurnDetector::Namo,
            ..ParapperConfig::default()
        };

        let status = model_status_from_root(&root, &config);
        assert_eq!(
            status
                .japanese_morph
                .as_ref()
                .map(|status| status.installed),
            Some(false)
        );

        let mut targets = Vec::new();
        push_japanese_morph_download_targets(&mut targets, &root, &config)
            .expect("legacy dictionary should schedule a compact Raw upgrade");
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].kind,
            DownloadTargetKind::TarZstdJapaneseMorphDirectory
        );
    }

    #[test]
    fn japanese_morph_status_rejects_compressed_dictionary_without_rkyv_system_dic() {
        let root = unique_test_models_root("model-status-japanese-morph-compressed-only");
        let config = parapper_config! {
            turn_detector: TurnDetector::Namo,
            ..ParapperConfig::default()
        };
        let dictionary_path = root.join("unidic-cwj-3_1_1").join("system.dic.zst");
        fs::create_dir_all(
            dictionary_path
                .parent()
                .expect("dictionary path should have parent"),
        )
        .expect("failed to create dictionary parent");
        write_zstd_vibrato_dictionary_marker(&dictionary_path);

        let status = model_status_from_root(&root, &config);
        assert_eq!(
            status
                .japanese_morph
                .as_ref()
                .map(|status| status.installed),
            Some(false),
            "runtime should not accept compressed dictionaries as installed"
        );
    }

    #[test]
    fn japanese_morph_dictionary_paths_only_include_rkyv_dictionary() {
        let model_dir = Path::new("unidic-cwj-3_1_1");
        let paths = japanese_morph_dictionary_paths_from_model_dir(model_dir);

        assert_eq!(paths, vec![model_dir.join("system.dic")]);
    }

    #[test]
    fn japanese_morph_download_materializes_rkyv_dictionary_from_zst() {
        let root = unique_test_models_root("model-status-japanese-morph-transcode");
        let model_dir = root.join("unidic-cwj-3_1_1");
        let compressed_path = model_dir.join("system.dic.zst");
        let rkyv_path = model_dir.join("system.dic");
        fs::create_dir_all(&model_dir).expect("failed to create dictionary parent");
        write_zstd_rkyv_vibrato_dictionary(&compressed_path);

        materialize_rkyv_japanese_morph_dictionary(&model_dir)
            .expect("zstd dictionary should materialize during model installation");

        assert!(
            compressed_path.is_file(),
            "release checksum inputs should remain beside the expanded dictionary"
        );
        assert!(
            japanese_morph_dictionary_compatible(&rkyv_path)
                .expect("rkyv dictionary should be readable"),
            "installed dictionary must be a compatible Vibrato rkyv dictionary"
        );
        let dictionary =
            vibrato_rkyv::Dictionary::from_path(&rkyv_path, vibrato_rkyv::LoadMode::Validate)
                .expect("installed dictionary should mmap-load through vibrato-rkyv");
        let tokenizer = vibrato_rkyv::Tokenizer::new(dictionary);
        let mut worker = tokenizer.new_worker();
        worker.reset_sentence("京都東京都");
        worker.tokenize();
        assert_eq!(worker.num_tokens(), 2);
    }

    #[test]
    fn japanese_morph_status_rejects_non_rkyv_system_dictionary() {
        let root = unique_test_models_root("model-status-japanese-morph-non-rkyv");
        let config = parapper_config! {
            turn_detector: TurnDetector::Namo,
            ..ParapperConfig::default()
        };
        let dictionary_path = root.join("unidic-cwj-3_1_1").join("system.dic");
        fs::create_dir_all(
            dictionary_path
                .parent()
                .expect("dictionary path should have parent"),
        )
        .expect("failed to create dictionary parent");
        fs::write(&dictionary_path, b"legacy vibrato dictionary")
            .expect("failed to write incompatible dictionary");

        let status = model_status_from_root(&root, &config);
        let morph = status
            .japanese_morph
            .as_ref()
            .expect("Japanese morph status should be present for Namo Japanese");
        assert!(
            !morph.installed,
            "non-rkyv dictionaries must be redownloaded or reinstalled as rkyv"
        );
        assert!(
            !morph.preparing,
            "an incompatible installed file should be treated as missing, not as an active download"
        );
    }

    #[test]
    fn japanese_morph_status_marks_partial_archive_as_preparing_not_installed() {
        for (case, marker_path) in [
            (
                "downloaded archive",
                Path::new("unidic-cwj-3_1_1.download").to_path_buf(),
            ),
            (
                "extracting directory",
                Path::new("unidic-cwj-3_1_1.extracting").to_path_buf(),
            ),
        ] {
            let root = unique_test_models_root(&format!("model-status-japanese-morph-{case}"));
            let config = parapper_config! {
                turn_detector: TurnDetector::Namo,
                ..ParapperConfig::default()
            };
            let marker_path = root.join(marker_path);
            if marker_path
                .extension()
                .is_some_and(|ext| ext == "extracting")
            {
                fs::create_dir_all(&marker_path).expect("failed to create extracting marker");
            } else {
                fs::create_dir_all(
                    marker_path
                        .parent()
                        .expect("marker path should have parent"),
                )
                .expect("failed to create marker parent");
                fs::write(&marker_path, b"partial archive")
                    .expect("failed to write download marker");
            }

            let status = model_status_from_root(&root, &config);
            let morph = status
                .japanese_morph
                .as_ref()
                .expect("Japanese morph status should be present for Namo Japanese");
            assert!(
                !morph.installed,
                "{case} must not allow recognition to start before the dictionary is available"
            );
            assert!(morph.preparing, "{case} should be shown as downloading");
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

    fn write_zstd_vibrato_dictionary_marker(path: &Path) {
        write_zstd_dictionary_marker(path, VIBRATO_MODEL_MAGIC);
    }

    fn write_zstd_rkyv_vibrato_dictionary(path: &Path) {
        let lexicon_csv = "京都,4,4,5,京都,名詞,固有名詞,地名,一般,*,*,キョウト,京都,*,A,*,*,*,1/5\n東京都,5,5,9,東京都,名詞,固有名詞,地名,一般,*,*,トウキョウト,東京都,*,B,5/9,*,5/9,*";
        let matrix_def = "10 10\n0 4 -5\n0 5 -9";
        let char_def = "DEFAULT 0 1 0";
        let unk_def = "DEFAULT,5,5,-1000,DEFAULT,名詞,普通名詞,*,*,*,*,*,*,*,*,*,*,*,*";
        let dictionary = vibrato_rkyv::SystemDictionaryBuilder::from_readers(
            lexicon_csv.as_bytes(),
            matrix_def.as_bytes(),
            char_def.as_bytes(),
            unk_def.as_bytes(),
        )
        .expect("failed to build rkyv test dictionary");
        let mut rkyv_bytes = Vec::new();
        dictionary
            .write(&mut rkyv_bytes)
            .expect("failed to write rkyv test dictionary");
        write_zstd_dictionary_marker(path, &rkyv_bytes);
    }

    fn write_zstd_dictionary_marker(path: &Path, bytes: &[u8]) {
        let mut encoder = zstd::Encoder::new(Vec::new(), 0).expect("failed to create zstd encoder");
        std::io::Write::write_all(&mut encoder, bytes)
            .expect("failed to write zstd dictionary marker");
        let compressed = encoder.finish().expect("failed to finish zstd marker");
        fs::write(path, compressed).expect("failed to write dictionary marker");
    }

    fn write_invalid_compact_raw_archive(path: &Path) {
        let archive_file = fs::File::create(path).expect("failed to create test archive");
        let encoder =
            zstd::Encoder::new(archive_file, 0).expect("failed to create test zstd encoder");
        let mut archive = tar::Builder::new(encoder);
        let invalid_dictionary = b"not the published compact Raw dictionary";
        let mut header = tar::Header::new_gnu();
        header.set_size(
            u64::try_from(invalid_dictionary.len()).expect("test dictionary size should fit u64"),
        );
        header.set_mode(0o644);
        header.set_cksum();
        archive
            .append_data(
                &mut header,
                "unidic-cwj-3_1_1/system.dic.zst",
                invalid_dictionary.as_slice(),
            )
            .expect("failed to add invalid dictionary to test archive");
        let encoder = archive
            .into_inner()
            .expect("failed to finish test tar archive");
        encoder
            .finish()
            .expect("failed to finish test zstd archive");
    }

    fn write_installed_compact_raw_dictionary(
        model_dir: &Path,
    ) -> crate::model::catalog::FileIntegrity {
        fs::create_dir_all(model_dir).expect("failed to create compact Raw dictionary dir");
        let dictionary_path = model_dir.join("system.dic");
        let mut dictionary = VIBRATO_MODEL_MAGIC.to_vec();
        dictionary.extend_from_slice(b"compact Raw integrity fixture");
        fs::write(&dictionary_path, &dictionary).expect("failed to write compact Raw dictionary");
        let sha256 = sha256_file(&dictionary_path, "test compact Raw dictionary")
            .expect("failed to hash compact Raw dictionary");
        let sha256 = Box::leak(sha256.into_boxed_str());
        let expected = crate::model::catalog::FileIntegrity {
            size: u64::try_from(dictionary.len()).expect("test dictionary size should fit u64"),
            sha256,
        };
        fs::write(
            model_dir.join("manifest.json"),
            format!(
                r#"{{
                "schema_version": 1,
                "dictionary_id": "unidic-cwj-3_1_1",
                "representation": "compact-raw",
                "feature_encoding": "[PP][S][F]",
                "expanded_dictionary": {{
                    "size": {},
                    "sha256": "{}"
                }}
            }}"#,
                expected.size, expected.sha256
            ),
        )
        .expect("failed to write compact Raw install manifest");
        expected
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
    fn nemotron_asr_status_uses_selected_archive_required_files() {
        for (model, archive_name) in [
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11.tar.bz2",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
                "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
            ),
        ] {
            let root = unique_test_models_root("model-status-nemotron-asr");
            let config = parapper_config! {
                asr_language: crate::config::AsrLanguage::Multilingual,
                asr_model: model,
                asr_precision: AsrPrecision::Int8,
                ..ParapperConfig::default()
            }
            .normalized();

            assert_eq!(asr_model_archive_name(model).as_deref(), Some(archive_name));

            let missing = model_status_from_root(&root, &config);
            assert!(!missing.asr.installed);

            write_required_asr_model_files(&root, model, AsrPrecision::Int8);
            write_required_asr_model_files(
                &root,
                config.completion_asr_model(),
                config.asr_precision_for(config.completion_asr_model()),
            );

            let installed = model_status_from_root(&root, &config);
            assert!(installed.asr.installed);
        }
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
        assert!(targets.iter().all(|target| target.url.starts_with(
            "https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX/resolve/main/"
        )));
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
