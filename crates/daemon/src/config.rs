use adele_voice_assistant_connector::{ConnectionConfig, TransportMode};
use adele_voice_module::config::{AudioConfig, SttConfig, TtsConfig, VadConfig};
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
pub struct WakeWordConfig {
    pub model_path: PathBuf,
    pub sensitivity: f32,
    /// Fire the wake word the moment enough partial frames clear the threshold
    /// (eager) rather than at the end of the utterance. Snappier trigger, at a
    /// higher false-trigger risk — tune alongside `sensitivity` (#50). Default
    /// on for the lower latency the issue asks for; flip off to be conservative.
    pub eager: bool,
    /// Audible cue played the instant the daemon enters Listening (and on each
    /// conversation-mode follow-up re-listen): `"ding"` (a short earcon, the
    /// default — instant), `"phrase"` (a spoken micro-phrase like "Yes?" —
    /// friendlier but adds ~1 s), or `"off"` (no cue) (#51).
    pub listening_cue: ListeningCue,
}

/// How the daemon announces that it has started Listening (#51).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ListeningCue {
    /// A short generated earcon (~120 ms tone). Instant and reliable — default.
    #[default]
    Ding,
    /// A spoken micro-phrase ("Yes?", "How can I help?", …) via the TTS path.
    /// Friendlier but adds ~1 s of synthesis/playback before the user can speak.
    Phrase,
    /// No cue at all.
    Off,
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
    /// Transport used to reach the orchestrator: `"uds"` (the default — the
    /// local Unix socket), `"ws"` (a possibly-remote WebSocket), or `"dbus"`
    /// (legacy local D-Bus). The voice service runs wherever the microphone is,
    /// which may not be where the orchestrator runs (voice#31).
    pub transport: String,
    /// WebSocket URL when `transport = "ws"`, e.g. `"wss://host:11339/ws"`.
    pub ws_url: Option<String>,
    /// Local socket path when `transport = "uds"`; unset resolves to
    /// `$XDG_RUNTIME_DIR/adelie/sock`.
    pub socket_path: Option<PathBuf>,
    /// Pre-minted bearer JWT for ws/uds; unset mints one via the local D-Bus
    /// token minter (the same path the chat clients use locally).
    pub ws_jwt: Option<String>,
    /// Username/password for the WebSocket `/login` token fallback.
    pub ws_login_username: Option<String>,
    pub ws_login_password: Option<String>,
    /// PEM CA certificate to trust for `wss://` (defaults to the daemon's CA).
    pub tls_ca_cert: Option<PathBuf>,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        let data_dir = dirs_path("adele-voice/models");
        Self {
            model_path: data_dir.join("hey-adele.rpw"),
            sensitivity: 0.5,
            eager: true,
            listening_cue: ListeningCue::Ding,
        }
    }
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            conversation_title: "Voice Conversation".into(),
            conversation_mode: false,
            followup_timeout_ms: 8000,
            spoken_response_hint: "You are Adele, responding by voice. The user's message was transcribed from speech, so expect occasional recognition errors and use your judgment. Answer directly and naturally — do not restate or paraphrase the question back before answering; just respond. Only if a message is clearly garbled or you truly cannot tell what was meant should you briefly check (e.g. 'did you mean X?') or ask one short clarifying question. Your reply will be read aloud, so keep it brief, conversational, and relevant — never a monologue. Default to a few short sentences. If a full answer would be long, give only the most salient points, then ask whether they'd like more. If they ask for more, you may expand but stay under about ten sentences. Let the user lead — invite follow-ups and let them steer. Avoid markdown, lists, code blocks, and emoji.".into(),
            transport: "uds".into(),
            ws_url: None,
            socket_path: None,
            ws_jwt: None,
            ws_login_username: None,
            ws_login_password: None,
            tls_ca_cert: None,
        }
    }
}

impl AssistantConfig {
    /// Build the client-common connection config from the `[assistant]`
    /// transport settings, defaulting to the local UDS transport. Unset fields
    /// fall back to client-common's defaults (e.g. the standard CA path).
    pub fn connection_config(&self) -> ConnectionConfig {
        let transport_mode = match self.transport.to_ascii_lowercase().as_str() {
            "ws" | "websocket" => TransportMode::Ws,
            "dbus" => TransportMode::Dbus,
            _ => TransportMode::Uds, // "uds" / "local" / anything else
        };
        let mut config = ConnectionConfig {
            transport_mode,
            ..ConnectionConfig::default()
        };
        if let Some(url) = &self.ws_url {
            config.ws_url = url.clone();
        }
        if self.socket_path.is_some() {
            config.socket_path = self.socket_path.clone();
        }
        if self.ws_jwt.is_some() {
            config.ws_jwt = self.ws_jwt.clone();
        }
        if self.ws_login_username.is_some() {
            config.ws_login_username = self.ws_login_username.clone();
        }
        if self.ws_login_password.is_some() {
            config.ws_login_password = self.ws_login_password.clone();
        }
        if self.tls_ca_cert.is_some() {
            config.tls_ca_cert = self.tls_ca_cert.clone();
        }
        config
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
    fn wake_word_defaults_to_eager_with_ding_cue() {
        // #50/#51: snappier eager trigger by default, and the ding earcon as the
        // default Listening cue.
        let config = WakeWordConfig::default();
        assert!(config.eager, "eager wake trigger is the default (#50)");
        assert_eq!(
            config.listening_cue,
            ListeningCue::Ding,
            "the ding earcon is the default Listening cue (#51)"
        );
    }

    #[test]
    fn parses_wake_word_eager_and_listening_cue() {
        // The new [wake_word] knobs round-trip from TOML, including the
        // lowercase cue variants.
        let toml_str = r#"
[wake_word]
eager = false
listening_cue = "phrase"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(!config.wake_word.eager);
        assert_eq!(config.wake_word.listening_cue, ListeningCue::Phrase);

        let off: Config = toml::from_str("[wake_word]\nlistening_cue = \"off\"\n").unwrap();
        assert_eq!(off.wake_word.listening_cue, ListeningCue::Off);
        // Unspecified fields keep their defaults.
        assert!(off.wake_word.eager, "unspecified eager stays the default");
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

    #[test]
    fn defaults_to_local_uds_transport() {
        let config = AssistantConfig::default();
        assert_eq!(config.transport, "uds");
        assert_eq!(
            config.connection_config().transport_mode,
            TransportMode::Uds
        );
    }

    #[test]
    fn transport_selection_maps_to_connection_config() {
        let ws = AssistantConfig {
            transport: "ws".into(),
            ws_url: Some("wss://host:11339/ws".into()),
            ..AssistantConfig::default()
        };
        let cfg = ws.connection_config();
        assert_eq!(cfg.transport_mode, TransportMode::Ws);
        assert_eq!(cfg.ws_url, "wss://host:11339/ws");

        let dbus = AssistantConfig {
            transport: "dbus".into(),
            ..AssistantConfig::default()
        };
        assert_eq!(dbus.connection_config().transport_mode, TransportMode::Dbus);
    }
}
