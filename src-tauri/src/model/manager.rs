use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::{Duration, SystemTime},
};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use serde::{Deserialize, Serialize};
use tar::Archive;
use tauri::{AppHandle, Emitter, Manager};
use tokio::{fs::File, io::AsyncWriteExt};
use vibrato_rkyv::Dictionary;
use xz2::read::XzDecoder;

use super::catalog::{
    ALL_ASR_MODELS, ALL_NAMO_TURN_DETECTOR_MODELS, ALL_NOISE_CANCELLATION_MODELS,
    NamoTurnDetectorModel, VAD_MODEL_URL, VIBRATO_MODEL_MAGIC, asr_model_base_url,
    asr_model_dir_name, asr_model_required_file_names, language_id_model_base_url,
    language_id_model_dir_name, language_id_model_files, local_tts_model_archive_name,
    local_tts_model_base_url, local_tts_model_required_dir_names,
    local_tts_model_required_file_names, namo_turn_detector_base_url, namo_turn_detector_dir_name,
    namo_turn_detector_files, noise_cancellation_model_base_url, noise_cancellation_model_dir_name,
    noise_cancellation_model_required_file_names, supertonic_tts_model_base_url,
    vibrato_unidic_archive_url, vibrato_unidic_dir_name,
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
    pub japanese_morph: Option<ModelAssetStatus>,
    pub language_id: Option<ModelAssetStatus>,
    pub turn_detectors: Vec<ModelAssetStatus>,
    pub tts: Vec<ModelAssetStatus>,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadTargetKind {
    File,
    TarBz2Directory,
    TarXzDirectory,
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
        file_name: format!("{}.tar.xz", vibrato_unidic_dir_name()),
        kind: DownloadTargetKind::TarXzDirectory,
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
    japanese_morph_dictionary_paths_from_model_dir(model_dir)
        .iter()
        .any(|path| japanese_morph_dictionary_compatible(path).unwrap_or(false))
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
    fs::remove_file(&compressed_path).ok();
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
    transcode_zstd_legacy_japanese_morph_dictionary_to_rkyv(compressed_path, output_path)
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

fn transcode_zstd_legacy_japanese_morph_dictionary_to_rkyv(
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
    let input = fs::File::open(compressed_path).with_context(|| {
        format!(
            "Failed to open compressed Japanese morph dictionary: {}",
            compressed_path.display()
        )
    })?;
    let decoder = zstd::Decoder::new(input).with_context(|| {
        format!(
            "Failed to decode Japanese morph dictionary: {}",
            compressed_path.display()
        )
    })?;
    let temporary_path = output_path.with_extension("dic.transcoding");
    let mut output = fs::File::create(&temporary_path).with_context(|| {
        format!(
            "Failed to create rkyv Japanese morph dictionary: {}",
            temporary_path.display()
        )
    })?;

    // The upstream archive is a legacy Vibrato dictionary; convert it once during installation.
    let dictionary = unsafe { Dictionary::from_legacy_reader(decoder) }.with_context(|| {
        format!(
            "Failed to read legacy Japanese morph dictionary: {}",
            compressed_path.display()
        )
    })?;
    dictionary.write(&mut output).with_context(|| {
        format!(
            "Failed to write rkyv Japanese morph dictionary: {}",
            temporary_path.display()
        )
    })?;
    output.flush().with_context(|| {
        format!(
            "Failed to flush rkyv Japanese morph dictionary: {}",
            temporary_path.display()
        )
    })?;
    drop(output);

    if !japanese_morph_dictionary_compatible(&temporary_path)? {
        fs::remove_file(&temporary_path).ok();
        anyhow::bail!(
            "Transcoded Japanese morph dictionary is not a Vibrato rkyv dictionary: {}",
            compressed_path.display()
        );
    }
    fs::rename(&temporary_path, output_path).with_context(|| {
        format!(
            "Failed to move rkyv Japanese morph dictionary from {} to {}",
            temporary_path.display(),
            output_path.display()
        )
    })?;
    Ok(())
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
    if target.kind != DownloadTargetKind::File && temporary_path.is_file() {
        match install_downloaded_archive(handle, target, &temporary_path, file_index, total_files) {
            Ok(()) => return Ok(()),
            Err(err) => {
                log::warn!(
                    "Failed to install existing model archive {}; downloading it again: {err}",
                    temporary_path.display()
                );
                fs::remove_file(&temporary_path).ok();
            }
        }
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
        DownloadTargetKind::TarXzDirectory => {
            extract_tar_xz_directory(&temporary_path, &target.output_path)?;
            materialize_rkyv_japanese_morph_dictionary(&target.output_path)?;
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

fn install_downloaded_archive(
    handle: &AppHandle,
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

    match target.kind {
        DownloadTargetKind::File => unreachable!("file downloads are not archive-installed"),
        DownloadTargetKind::TarBz2Directory => {
            extract_tar_bz2_directory(temporary_path, &target.output_path)?;
        }
        DownloadTargetKind::TarXzDirectory => {
            extract_tar_xz_directory(temporary_path, &target.output_path)?;
            materialize_rkyv_japanese_morph_dictionary(&target.output_path)?;
        }
    }

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

fn extract_tar_bz2_directory(archive_path: &Path, output_path: &Path) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("Failed to open TTS archive: {}", archive_path.display()))?;
    let decoder = BzDecoder::new(file);
    extract_tar_directory(decoder, archive_path, output_path, "TTS")
}

fn extract_tar_xz_directory(archive_path: &Path, output_path: &Path) -> Result<()> {
    match extract_tar_xz_directory_with_system_tar(archive_path, output_path) {
        Ok(()) => return Ok(()),
        Err(err) => {
            log::warn!(
                "Failed to extract Japanese morph dictionary with system tar; falling back to Rust extractor: {err}"
            );
        }
    }

    let file = fs::File::open(archive_path).with_context(|| {
        format!(
            "Failed to open Japanese morph dictionary archive: {}",
            archive_path.display()
        )
    })?;
    let decoder = XzDecoder::new(file);
    extract_tar_directory(decoder, archive_path, output_path, "model")
}

fn extract_tar_xz_directory_with_system_tar(archive_path: &Path, output_path: &Path) -> Result<()> {
    extract_archive_directory(archive_path, output_path, "model", |temp_dir| {
        let status = std::process::Command::new("tar")
            .arg("-xf")
            .arg(archive_path)
            .arg("-C")
            .arg(temp_dir)
            .status()
            .with_context(|| "Failed to start system tar for Japanese morph dictionary")?;
        if !status.success() {
            anyhow::bail!("system tar exited with status {status}");
        }
        Ok(())
    })
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
        NamoTurnDetectorModel, contained_tar_entry_path, default_asr_model_dir_from_root,
        japanese_morph_dictionary_compatible, japanese_morph_dictionary_paths_from_model_dir,
        local_tts_voices_for_config, materialize_rkyv_japanese_morph_dictionary,
        model_status_from_root, namo_turn_detector_models_for_config,
    };
    use crate::config::{
        AsrModel, AsrPrecision, LocalTtsVoice, NoiseCancellationModel, ParapperConfig,
        SpeechBackend, SpeechMapping, SpeechSourceKind, TurnDetector,
    };
    use crate::model::catalog::{VIBRATO_MODEL_MAGIC, asr_model_required_file_names};
    use std::{fs, path::Path, time::SystemTime};

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
    fn japanese_morph_status_is_installed_when_system_dictionary_is_vibrato_rkyv_model() {
        let root = unique_test_models_root("model-status-japanese-morph");
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

        let dictionary_path = root.join("unidic-cwj-3_1_1").join("system.dic");
        fs::create_dir_all(
            dictionary_path
                .parent()
                .expect("dictionary path should have parent"),
        )
        .expect("failed to create dictionary parent");
        fs::write(&dictionary_path, VIBRATO_MODEL_MAGIC)
            .expect("failed to write dictionary marker");

        let status = model_status_from_root(&root, &config);
        assert_eq!(
            status
                .japanese_morph
                .as_ref()
                .map(|status| status.installed),
            Some(true)
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
            !compressed_path.exists(),
            "compressed dictionary should be removed so startup cannot pick it first"
        );
        assert!(
            japanese_morph_dictionary_compatible(&rkyv_path)
                .expect("rkyv dictionary should be readable"),
            "installed dictionary must be a compatible Vibrato rkyv dictionary"
        );
        let dictionary =
            vibrato_rkyv::Dictionary::from_path(&rkyv_path, vibrato_rkyv::LoadMode::TrustCache)
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
}
