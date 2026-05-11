pub(in crate::synthesis) use imp::SherpaOnnxTtsEngine;

#[cfg(test)]
mod imp {
    use std::path::Path;

    use anyhow::{Result, bail};

    use crate::config::LocalTtsVoice;

    pub(in crate::synthesis) struct SherpaOnnxTtsEngine {
        unavailable: bool,
    }

    pub(in crate::synthesis) struct SynthesizedTtsAudio {
        pub(in crate::synthesis) samples: Vec<f32>,
        pub(in crate::synthesis) sample_rate: i32,
    }

    impl SherpaOnnxTtsEngine {
        pub(in crate::synthesis) fn new(
            _model_dir: &Path,
            _voice: LocalTtsVoice,
            _num_threads: i32,
        ) -> Result<Self> {
            bail!("Sherpa ONNX TTS is unavailable in unit tests")
        }

        pub(in crate::synthesis) fn synthesize(
            &self,
            _text: &str,
            _language: Option<&str>,
        ) -> Result<SynthesizedTtsAudio> {
            bail!(self.unavailable_message())
        }

        fn unavailable_message(&self) -> &'static str {
            if self.unavailable {
                "Sherpa ONNX TTS is unavailable in unit tests"
            } else {
                "Sherpa ONNX TTS stub was not initialized"
            }
        }
    }
}

#[cfg(not(test))]
mod imp {
    use std::{collections::HashMap, path::Path};

    use anyhow::{Context, Result, anyhow, bail};
    use sherpa_onnx::{
        GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
        OfflineTtsVitsModelConfig,
    };

    use crate::config::{LocalTtsFamily, LocalTtsVoice};

    pub(in crate::synthesis) struct SherpaOnnxTtsEngine {
        tts: OfflineTts,
        voice: LocalTtsVoice,
    }

    unsafe impl Send for SherpaOnnxTtsEngine {}

    impl SherpaOnnxTtsEngine {
        pub(in crate::synthesis) fn new(
            model_dir: &Path,
            voice: LocalTtsVoice,
            num_threads: i32,
        ) -> Result<Self> {
            let files = VitsModelFiles::from_dir(model_dir, voice)?;
            let mut model_config = OfflineTtsModelConfig {
                provider: Some("cpu".to_string()),
                num_threads,
                ..Default::default()
            };
            model_config.vits = OfflineTtsVitsModelConfig {
                model: Some(files.model.display().to_string()),
                tokens: Some(files.tokens.display().to_string()),
                data_dir: Some(files.data_dir.display().to_string()),
                ..Default::default()
            };
            let config = OfflineTtsConfig {
                model: model_config,
                max_num_sentences: 1,
                ..Default::default()
            };

            let tts = OfflineTts::create(&config)
                .ok_or_else(|| anyhow!("Failed to create sherpa-onnx TTS engine"))?;
            Ok(Self { tts, voice })
        }

        pub(in crate::synthesis) fn synthesize(
            &self,
            text: &str,
            language: Option<&str>,
        ) -> Result<SynthesizedTtsAudio> {
            let generation_config = generation_config_for_voice(self.voice, language);
            let audio = self
                .tts
                .generate_with_config(
                    text,
                    &generation_config,
                    Option::<fn(&[f32], f32) -> bool>::None,
                )
                .ok_or_else(|| anyhow!("Failed to generate sherpa-onnx TTS audio"))?;
            Ok(SynthesizedTtsAudio {
                samples: audio.samples().to_vec(),
                sample_rate: audio.sample_rate(),
            })
        }
    }

    pub(in crate::synthesis) struct SynthesizedTtsAudio {
        pub(in crate::synthesis) samples: Vec<f32>,
        pub(in crate::synthesis) sample_rate: i32,
    }

    struct VitsModelFiles {
        model: std::path::PathBuf,
        tokens: std::path::PathBuf,
        data_dir: std::path::PathBuf,
    }

    impl VitsModelFiles {
        fn from_dir(model_dir: &Path, voice: LocalTtsVoice) -> Result<Self> {
            let files = match voice.family() {
                LocalTtsFamily::Vits => Self {
                    model: model_dir.join(voice.onnx_file_name()),
                    tokens: model_dir.join("tokens.txt"),
                    data_dir: model_dir.join("espeak-ng-data"),
                },
                LocalTtsFamily::Supertonic => {
                    bail!("Supertonic ONNX is handled by the ORT engine")
                }
            };
            files.validate().with_context(|| {
                format!(
                    "Sherpa ONNX TTS model is not installed: {}",
                    model_dir.display()
                )
            })?;
            Ok(files)
        }

        fn validate(&self) -> Result<()> {
            for path in [&self.model, &self.tokens] {
                if !path.is_file() {
                    return Err(anyhow!("TTS model file not found: {}", path.display()));
                }
            }
            if !self.data_dir.is_dir() {
                return Err(anyhow!(
                    "TTS espeak-ng data dir not found: {}",
                    self.data_dir.display()
                ));
            }
            Ok(())
        }
    }

    fn generation_config_for_voice(
        voice: LocalTtsVoice,
        language: Option<&str>,
    ) -> GenerationConfig {
        let mut config = GenerationConfig::default();
        if voice.family() == LocalTtsFamily::Supertonic {
            let mut extra = HashMap::new();
            let language = language
                .map(str::trim)
                .filter(|language| !language.is_empty())
                .unwrap_or("en")
                .to_ascii_lowercase();
            extra.insert(
                "lang".to_string(),
                serde_json::Value::String(match language.as_str() {
                    "en" | "ko" | "es" | "pt" | "fr" => language,
                    _ => "en".to_string(),
                }),
            );
            config.sid = 0;
            config.num_steps = 5;
            config.extra = Some(extra);
        }
        config
    }
}
