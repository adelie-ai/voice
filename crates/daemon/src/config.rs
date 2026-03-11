use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub wake_word: WakeWordConfig,
    pub vad: VadConfig,
    pub stt: SttConfig,
    pub tts: TtsConfig,
    pub assistant: AssistantConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub input_device: String,
    pub output_device: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct WakeWordConfig {
    pub model_path: PathBuf,
    pub sensitivity: f32,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    pub speech_threshold: f32,
    pub silence_duration_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub model_path: PathBuf,
    pub language: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub piper_binary: String,
    pub model_path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct AssistantConfig {
    pub conversation_title: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            wake_word: WakeWordConfig::default(),
            vad: VadConfig::default(),
            stt: SttConfig::default(),
            tts: TtsConfig::default(),
            assistant: AssistantConfig::default(),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_device: "default".into(),
            output_device: "default".into(),
        }
    }
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        let data_dir = dirs_path("adele-voice/models");
        Self {
            model_path: data_dir.join("hey-adele.rpw"),
            sensitivity: 0.5,
        }
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence_duration_ms: 800,
        }
    }
}

impl Default for SttConfig {
    fn default() -> Self {
        let data_dir = dirs_path("adele-voice/models");
        Self {
            model_path: data_dir.join("ggml-distil-large-v3.bin"),
            language: "en".into(),
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        let data_dir = dirs_path("adele-voice/models");
        Self {
            piper_binary: "piper".into(),
            model_path: data_dir.join("en_US-amy-medium.onnx"),
        }
    }
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            conversation_title: "Voice Conversation".into(),
        }
    }
}

fn dirs_path(suffix: &str) -> PathBuf {
    std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".local/share")
        })
        .join(suffix)
}

fn config_path() -> PathBuf {
    std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config")
        })
        .join("adele-voice/config.toml")
}

pub fn load() -> Result<Config> {
    let path = config_path();
    if !path.exists() {
        tracing::info!("no config file at {}, using defaults", path.display());
        return Ok(Config::default());
    }

    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read config from {}", path.display()))?;

    let config: Config = toml::from_str(&contents)
        .with_context(|| format!("failed to parse config from {}", path.display()))?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = Config::default();
        assert_eq!(config.audio.input_device, "default");
        assert_eq!(config.vad.silence_duration_ms, 800);
        assert_eq!(config.stt.language, "en");
        assert_eq!(config.assistant.conversation_title, "Voice Conversation");
    }

    #[test]
    fn parses_minimal_toml() {
        let toml_str = r#"
[audio]
input_device = "hw:1"

[vad]
speech_threshold = 0.7
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.audio.input_device, "hw:1");
        assert_eq!(config.vad.speech_threshold, 0.7);
        // Defaults for unspecified fields
        assert_eq!(config.audio.output_device, "default");
        assert_eq!(config.stt.language, "en");
    }
}
