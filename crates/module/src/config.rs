//! Configuration for the embeddable voice adapters — audio devices, VAD, STT,
//! and the TTS backend. These are the knobs an embedding client (or the daemon)
//! deserializes from its own config file; the daemon composes them with its
//! daemon-only sections (wake word, assistant, idle-exit).

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub input_device: String,
    pub output_device: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    pub model_path: PathBuf,
    pub speech_threshold: f32,
    pub silence_duration_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub model_path: PathBuf,
    pub language: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    /// "kokoro" (local, default), "piper" (local), or "polly" (AWS cloud).
    pub backend: String,
    pub piper_binary: String,
    pub model_path: PathBuf,
    /// Polly voice id (e.g. "Joanna", "Ruth", "Matthew") when backend = "polly".
    pub polly_voice: String,
    /// Polly engine: "neural" (default), "generative" (most natural), "long-form", or "standard".
    pub polly_engine: String,
    /// AWS region for Polly; falls back to the AWS credential chain when unset.
    pub polly_region: Option<String>,
    /// Kokoro ONNX model path (when backend = "kokoro").
    pub kokoro_model_path: PathBuf,
    /// Directory of Kokoro voice `.bin` files (one per voice).
    pub kokoro_voices_dir: PathBuf,
    /// Kokoro voice name — a `<name>.bin` in the voices dir, e.g. "af_heart".
    pub kokoro_voice: String,
    /// espeak-ng language for Kokoro phonemization, e.g. "en-us" or "en-gb".
    pub kokoro_lang: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_device: "default".into(),
            output_device: "default".into(),
        }
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: models_dir().join("silero_vad.onnx"),
            speech_threshold: 0.5,
            silence_duration_ms: 800,
        }
    }
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            model_path: models_dir().join("ggml-distil-large-v3.bin"),
            language: "en".into(),
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            backend: "kokoro".into(),
            piper_binary: "piper".into(),
            model_path: models_dir().join("en_US-amy-medium.onnx"),
            polly_voice: "Joanna".into(),
            polly_engine: "neural".into(),
            polly_region: None,
            kokoro_model_path: models_dir().join("kokoro.onnx"),
            kokoro_voices_dir: models_dir().join("kokoro-voices"),
            kokoro_voice: "af_heart".into(),
            kokoro_lang: "en-us".into(),
        }
    }
}

/// The shared `adele-voice/models` data directory the default model paths live
/// under (`$XDG_DATA_HOME/adele-voice/models`, falling back to the cwd).
fn models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("adele-voice/models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_provisioning_layout() {
        assert_eq!(AudioConfig::default().input_device, "default");
        assert_eq!(VadConfig::default().silence_duration_ms, 800);
        assert_eq!(SttConfig::default().language, "en");
        assert_eq!(TtsConfig::default().backend, "kokoro");
    }

    #[test]
    fn tts_config_parses_a_minimal_table() {
        let cfg: TtsConfig = toml::from_str(r#"backend = "piper""#).unwrap();
        assert_eq!(cfg.backend, "piper");
        // Unspecified fields fall back to defaults.
        assert_eq!(cfg.kokoro_voice, "af_heart");
    }
}
