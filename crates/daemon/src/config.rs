use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub wake_word: WakeWordConfig,
    pub vad: VadConfig,
    pub stt: SttConfig,
    pub tts: TtsConfig,
    pub assistant: AssistantConfig,
    /// Exit after this many ms idle (wake word off and nothing playing) so the
    /// daemon isn't resident when unused; D-Bus activation restarts it on
    /// demand. 0 (the default) keeps it always-on.
    pub idle_exit_timeout_ms: u64,
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
    pub model_path: PathBuf,
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
    /// When true, after replying the daemon re-opens the mic for a follow-up
    /// turn instead of returning to wake-word idle, until `followup_timeout_ms`
    /// of silence elapses.
    pub conversation_mode: bool,
    /// How long to wait (ms) for follow-up speech before ending a conversation.
    pub followup_timeout_ms: u64,
    /// Instruction prepended to each spoken prompt so the assistant keeps
    /// replies short, conversational, and read-aloud friendly. Empty disables.
    pub spoken_response_hint: String,
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
        let data_dir = dirs_path("adele-voice/models");
        Self {
            model_path: data_dir.join("silero_vad.onnx"),
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
            conversation_mode: false,
            followup_timeout_ms: 8000,
            spoken_response_hint: "You are Adele, responding by voice — the user's message was transcribed from speech, so expect occasional recognition errors — use your judgment, and if anything seems garbled or like a non-sequitur, briefly lead with how you understood it (e.g. open with 'it sounds like you asked about X') so the user can catch a mishearing, then answer; ask a short clarifying question only if you truly cannot tell. Your reply will be read aloud, so keep it brief, conversational, and relevant, never a monologue. Default to a few short sentences. If a full answer would be long, give only the most salient points and then ask whether they'd like the details. If they ask for more, you may expand but stay under about ten sentences (roughly a 30-second read). Let the user lead — invite follow-up questions and let them steer the conversation. Avoid markdown, lists, code blocks, and emoji.".into(),
        }
    }
}

fn dirs_path(suffix: &str) -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(suffix)
}

fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
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
