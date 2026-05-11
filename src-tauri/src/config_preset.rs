use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::config::{
    AsrLanguage, AsrModel, AsrPrecision, LocalTtsVoice, NoiseCancellationModel, ParapperConfig,
    SpeechBackend, SpeechMapping, SpeechSourceKind, TranslationMapping, TurnDetector,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPreset {
    pub name: String,
    pub built_in: bool,
    pub config: ParapperConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredConfigPreset {
    name: String,
    config: ParapperConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct StoredConfigPresets {
    presets: Vec<StoredConfigPreset>,
}

pub fn load_config_presets(path: &Path) -> Result<Vec<ConfigPreset>> {
    Ok(merged_config_presets(load_stored_config_presets(path)?))
}

pub fn save_config_preset(
    path: &Path,
    name: String,
    config: ParapperConfig,
) -> Result<Vec<ConfigPreset>> {
    let name = normalized_preset_name(name)?;
    let mut stored = load_stored_config_presets(path)?;
    let preset = StoredConfigPreset {
        name: name.clone(),
        config: config.normalized(),
    };

    if let Some(existing) = stored.presets.iter_mut().find(|preset| preset.name == name) {
        *existing = preset;
    } else {
        stored.presets.push(preset);
    }
    stored
        .presets
        .sort_by(|left, right| left.name.cmp(&right.name));
    save_stored_config_presets(path, &stored)?;

    Ok(merged_config_presets(stored))
}

pub fn delete_config_preset(path: &Path, name: String) -> Result<Vec<ConfigPreset>> {
    let name = normalized_preset_name(name)?;
    let mut stored = load_stored_config_presets(path)?;
    stored.presets.retain(|preset| preset.name != name);
    save_stored_config_presets(path, &stored)?;

    Ok(merged_config_presets(stored))
}

fn load_stored_config_presets(path: &Path) -> Result<StoredConfigPresets> {
    if !path.is_file() {
        return Ok(StoredConfigPresets::default());
    }
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config presets: {}", path.display()))?;
    serde_json::from_str::<StoredConfigPresets>(&content)
        .with_context(|| format!("Failed to parse config presets: {}", path.display()))
}

fn save_stored_config_presets(path: &Path, presets: &StoredConfigPresets) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create config preset dir: {}", parent.display()))?;
    }
    let content = serde_json::to_string_pretty(presets)?;
    fs::write(path, content)
        .with_context(|| format!("Failed to write config presets: {}", path.display()))
}

fn merged_config_presets(stored: StoredConfigPresets) -> Vec<ConfigPreset> {
    let mut presets = built_in_config_presets()
        .into_iter()
        .filter(|built_in| {
            !stored
                .presets
                .iter()
                .any(|stored| stored.name == built_in.name)
        })
        .collect::<Vec<_>>();
    presets.extend(stored.presets.into_iter().map(|preset| ConfigPreset {
        name: preset.name,
        built_in: false,
        config: preset.config.normalized(),
    }));
    presets
}

fn normalized_preset_name(name: String) -> Result<String> {
    let trimmed_len = name.trim().len();
    if trimmed_len == 0 {
        bail!("Config preset name is empty");
    }
    if trimmed_len == name.len() {
        Ok(name)
    } else {
        Ok(name.trim().to_string())
    }
}

fn built_in_config_presets() -> Vec<ConfigPreset> {
    vec![
        ConfigPreset {
            name: "日本語文字起こし シンプル".to_string(),
            built_in: true,
            config: japanese_transcription_simple_config(),
        },
        ConfigPreset {
            name: "日本語文字起こし".to_string(),
            built_in: true,
            config: japanese_transcription_rich_config(),
        },
        ConfigPreset {
            name: "日→英 翻訳".to_string(),
            built_in: true,
            config: japanese_to_english_translation_config(),
        },
        ConfigPreset {
            name: "日→英 翻訳読み上げ".to_string(),
            built_in: true,
            config: japanese_to_english_translation_speech_config(),
        },
        ConfigPreset {
            name: "日英⇔英日 翻訳読み上げ".to_string(),
            built_in: true,
            config: japanese_english_bidirectional_translation_speech_config(),
        },
    ]
}

fn japanese_transcription_simple_config() -> ParapperConfig {
    let mut config = base_japanese_config();
    config.noise_cancellation_enabled = false;
    config.turn_detector = TurnDetector::Simple;
    config.interim_result_enabled = false;
    config.turn_rerecognize_full_on_complete = false;
    config.translation_enabled = false;
    config.translation_mappings = Vec::new();
    config.speech_mappings = Vec::new();
    config.normalized()
}

fn japanese_transcription_rich_config() -> ParapperConfig {
    base_rich_japanese_config().normalized()
}

fn japanese_to_english_translation_config() -> ParapperConfig {
    let mut config = base_rich_japanese_config();
    config.translation_enabled = true;
    config.translation_mappings = vec![japanese_to_english_translation_mapping()];
    config.speech_mappings = Vec::new();
    config.normalized()
}

fn japanese_to_english_translation_speech_config() -> ParapperConfig {
    let mut config = base_rich_japanese_config();
    config.translation_enabled = true;
    config.translation_mappings = vec![japanese_to_english_translation_mapping()];
    config.speech_mappings = vec![supertonic_translation_speech_mapping(
        "speech-en",
        "en_US",
        LocalTtsVoice::Supertonic2Onnx,
        "en",
    )];
    config.normalized()
}

fn japanese_to_english_translation_mapping() -> TranslationMapping {
    TranslationMapping {
        id: "translate-ja-en".to_string(),
        source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
        target_lang: "en_US".to_string(),
    }
}

fn japanese_english_bidirectional_translation_speech_config() -> ParapperConfig {
    let mut config = base_rich_japanese_config();
    config.multilingual_asr_enabled = true;
    config.enabled_asr_models = vec![
        AsrModel::ReazonSpeechK2V2,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
    ];
    config.translation_enabled = true;
    config.translation_mappings = vec![
        TranslationMapping {
            id: "translate-ja-en".to_string(),
            source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
            target_lang: "en_US".to_string(),
        },
        TranslationMapping {
            id: "translate-en-ja".to_string(),
            source_asr_model: Some(AsrModel::NemoParakeetTdt0_6BV2Int8),
            target_lang: "ja_JP".to_string(),
        },
    ];
    config.speech_mappings = vec![
        supertonic_translation_speech_mapping(
            "speech-en",
            "en_US",
            LocalTtsVoice::Supertonic2Onnx,
            "en",
        ),
        supertonic_translation_speech_mapping(
            "speech-ja",
            "ja_JP",
            LocalTtsVoice::Supertonic3Onnx,
            "ja",
        ),
    ];
    config.normalized()
}

fn base_japanese_config() -> ParapperConfig {
    ParapperConfig {
        input_volume_db: 0.0,
        asr_language: AsrLanguage::Japanese,
        asr_model: AsrModel::ReazonSpeechK2V2,
        asr_precision: AsrPrecision::Int8Float32,
        asr_num_threads: 4,
        asr_normalize_input_audio: true,
        multilingual_asr_enabled: false,
        enabled_asr_models: vec![AsrModel::ReazonSpeechK2V2],
        turn_detector: TurnDetector::Simple,
        interim_result_enabled: false,
        interim_result_silence_ms: 96,
        turn_check_silence_ms: 320,
        segment_start_speech_ms: 96,
        turn_rerecognize_full_on_complete: false,
        noise_cancellation_enabled: false,
        noise_cancellation_model: NoiseCancellationModel::UlUnas,
        translation_enabled: false,
        translation_mappings: Vec::new(),
        speech_mappings: Vec::new(),
        ..ParapperConfig::default()
    }
}

fn base_rich_japanese_config() -> ParapperConfig {
    ParapperConfig {
        noise_cancellation_enabled: true,
        turn_detector: TurnDetector::Namo,
        interim_result_enabled: true,
        turn_check_silence_ms: 320,
        interim_result_silence_ms: 320,
        segment_start_speech_ms: 96,
        turn_rerecognize_full_on_complete: true,
        ..base_japanese_config()
    }
}

fn supertonic_translation_speech_mapping(
    id: &str,
    target_lang: &str,
    voice: LocalTtsVoice,
    local_tts_language: &str,
) -> SpeechMapping {
    SpeechMapping {
        id: id.to_string(),
        source_kind: SpeechSourceKind::Translation,
        source_asr_model: None,
        target_lang: Some(target_lang.to_string()),
        backend: SpeechBackend::LocalTts,
        talker: String::new(),
        local_tts_voice: Some(voice),
        local_tts_language: Some(local_tts_language.to_string()),
        local_tts_speaker_id: Some(2),
        output_device_id: None,
        output_device_host: None,
        output_device_name: None,
        muted: false,
        volume: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{japanese_english_bidirectional_translation_speech_config, load_config_presets};
    use crate::config::{AsrModel, LocalTtsVoice};

    #[test]
    fn bidirectional_translation_speech_preset_uses_enabled_asr_models() {
        let config = japanese_english_bidirectional_translation_speech_config();

        assert!(config.multilingual_asr_enabled);
        assert_eq!(
            config.enabled_asr_models,
            vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8
            ]
        );
        assert_eq!(config.translation_mappings.len(), 2);
        assert_eq!(config.speech_mappings.len(), 2);
        assert_eq!(
            config.speech_mappings[0].local_tts_voice,
            Some(LocalTtsVoice::Supertonic2Onnx)
        );
        assert_eq!(
            config.speech_mappings[1].local_tts_voice,
            Some(LocalTtsVoice::Supertonic3Onnx)
        );
        assert_eq!(config.speech_mappings[0].local_tts_speaker_id, Some(2));
        assert_eq!(config.speech_mappings[1].local_tts_speaker_id, Some(2));
    }

    #[test]
    fn built_in_presets_are_available_without_user_file() {
        let presets = load_config_presets(std::path::Path::new("missing-presets.json"))
            .expect("built-in presets should load");

        assert!(presets.iter().any(|preset| preset.built_in));
        assert!(
            presets
                .iter()
                .any(|preset| preset.name == "日本語文字起こし シンプル")
        );
    }

    #[test]
    fn built_in_presets_use_responsive_segment_start_default() {
        let presets = load_config_presets(std::path::Path::new("missing-presets.json"))
            .expect("built-in presets should load");

        for preset in presets.iter().filter(|preset| preset.built_in) {
            assert_eq!(
                preset.config.segment_start_speech_ms, 96,
                "preset={}",
                preset.name
            );
        }
    }
}
