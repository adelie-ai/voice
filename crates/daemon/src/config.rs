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
    pub timeouts: TimeoutConfig,
    /// Exit after this many ms idle (wake word off and nothing playing) so the
    /// daemon isn't resident when unused; D-Bus activation restarts it on
    /// demand. 0 (the default) keeps it always-on.
    pub idle_exit_timeout_ms: u64,
}

/// Bounds on the otherwise-unbounded turn operations (#58). Every wait that can
/// stall — the orchestrator response, STT decode, TTS synth, and the
/// conversation create/subscribe/send round-trips — is capped so a wedged
/// dependency apologizes and returns to Idle instead of hanging the user in
/// Processing forever.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TimeoutConfig {
    /// Per-event stall deadline for the streaming response: if no progress
    /// event (a text chunk OR a status) arrives within this window, the turn is
    /// considered wedged. Resets on every event — the "if we aren't getting
    /// progress quickly, the user has given up" guard. 0 disables.
    pub response_stall_ms: u64,
    /// Overall ceiling on a single turn's streaming response, regardless of
    /// heartbeats — a backstop against an event source that dribbles just often
    /// enough to keep resetting the stall deadline. 0 disables.
    pub turn_budget_ms: u64,
    /// Ceiling on a single Whisper STT decode. 0 disables.
    pub stt_ms: u64,
    /// Ceiling on a single TTS synth (one sentence). 0 disables.
    pub tts_ms: u64,
    /// Ceiling on each conversation create / subscribe / send round-trip to the
    /// orchestrator. 0 disables.
    pub connect_ms: u64,
    /// Minimum gap between spoken status narrations within a turn, so a tool-
    /// heavy turn doesn't chatter one line per tool call. The first status of a
    /// turn always speaks; later ones are rate-limited to this interval. 0
    /// speaks every status (not recommended); a very large value speaks only
    /// the first.
    pub status_narration_min_gap_ms: u64,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            // Generous enough not to clip a normal tool turn's gap between
            // events, tight enough that a truly wedged orchestrator gives up
            // before the user does.
            response_stall_ms: 30_000,
            // A turn that runs past two minutes is almost certainly wedged.
            turn_budget_ms: 120_000,
            stt_ms: 20_000,
            tts_ms: 20_000,
            connect_ms: 10_000,
            // Speak the first status immediately, then at most one line every
            // ~15 s — enough to reassure on a long turn without narrating every
            // tool call.
            status_narration_min_gap_ms: 15_000,
        }
    }
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
    /// Reuse the most recent conversation on a fresh wake (not an in-turn
    /// follow-up) when the last activity was within this window (voice#53).
    /// Default 600000 = 10 min; `0` always starts a new conversation. A
    /// conversation ended explicitly via the `stop_listening` client tool is
    /// never reused regardless of this window (voice#59).
    pub conversation_reuse_window_ms: u64,
    /// Per-tool enable toggles for the LLM-driven session-control client tools
    /// (voice#61). All on by default; flip one off to withhold that tool from
    /// the orchestrator's tool list.
    pub client_tools: ClientToolsConfig,
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

/// Per-tool enable toggles for the LLM-driven session-control client tools
/// (voice#61). Each defaults to `true`; set one to `false` in
/// `[assistant.client_tools]` to withhold it from the orchestrator's tool list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ClientToolsConfig {
    /// `stop_listening` — end the session when the user signals they're done.
    pub stop_listening: bool,
    /// `listen_for_more` — keep listening when a reply expects a response.
    pub listen_for_more: bool,
    /// `say_this` — speak a specific line immediately (LLM-driven narration).
    pub say_this: bool,
}

impl Default for ClientToolsConfig {
    fn default() -> Self {
        Self {
            stop_listening: true,
            listen_for_more: true,
            say_this: true,
        }
    }
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
            conversation_reuse_window_ms: 600_000,
            client_tools: ClientToolsConfig::default(),
            spoken_response_hint: "You are Adele, responding by voice. The user's message was transcribed from speech, so expect occasional recognition errors and use your judgment. For a simple, quick question, just answer directly — do not preface it. But if the request will take real work (looking something up, using a tool, a multi-step task), open with a SHORT spoken acknowledgement that shows you understood — a handful of words, ending in a period so it can be read aloud immediately (e.g. 'Got it — checking that now.' or 'Sure, looking into your calendar.') — then do the work and answer. Keep that acknowledgement to one short clause; never restate the whole question. Only if a message is clearly garbled or you truly cannot tell what was meant should you briefly check (e.g. 'did you mean X?') or ask one short clarifying question. Your reply will be read aloud, so keep it brief, conversational, and relevant — never a monologue. Default to a few short sentences. If a full answer would be long, give only the most salient points, then ask whether they'd like more. If they ask for more, you may expand but stay under about ten sentences. Let the user lead — invite follow-ups and let them steer. Never use markdown or formatting of any kind — no asterisks, underscores, backticks, pound signs, bullet characters, or emoji. Speak in plain, natural prose. Write every reply as a script to be read aloud verbatim — exactly the words to be spoken. Spell out abbreviations and acronyms as full words (say 'for example' not 'e.g.', 'versus' not 'vs.', 'and so on' not 'etc.', 'approximately' not 'approx.'). Avoid symbols that do not read well aloud: write 'and' not '&', 'percent' not '%', 'number' not '#', 'dollars' not '$', and 'to' for a range rather than a dash. Never read out a URL, file path, or email address — describe it in words instead. Write numbers, dates, and times the way you would say them out loud. Punctuate for the ear, not the page: use commas and periods to control spoken pacing and where natural pauses fall. When spoken phrasing and strict grammar disagree, favor what sounds right read aloud — short clauses and sentences, with commas placed where a speaker would breathe.".into(),
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

pub fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("adele-voice/config.toml")
}

/// The subset of [`Config`] the daemon can apply to a running pipeline on a live
/// reload (config#52), captured as plain values so old/new snapshots can be
/// diffed cheaply. Anything not here (model paths, STT/TTS, transport) still
/// requires a restart — those rebuild expensive sessions or reconnect a socket.
///
/// New tunable fields added on `main` flow in automatically: derive `Clone` +
/// `PartialEq`, populate them in [`Config::tunables`], and apply them in the
/// pipeline's apply path. The watcher diffs whole [`Tunables`] values, so a
/// reload picks up any field present here without further plumbing.
#[derive(Debug, Clone, PartialEq)]
pub struct Tunables {
    /// Hot-swapped in place on the live `Endpointer`.
    pub speech_threshold: f32,
    /// Hot-swapped in place on the live `Endpointer`.
    pub silence_duration_ms: u64,
    /// Hot-swapped on the pipeline (affects the next turn).
    pub followup_timeout_ms: u64,
    /// Hot-swapped on the pipeline (affects the next fresh wake); 0 disables
    /// cross-wake conversation reuse (voice#53).
    pub conversation_reuse_window_ms: u64,
    /// Hot-swapped on the pipeline (affects the next turn boundary).
    pub conversation_mode: bool,
    /// Hot-swapped on the pipeline (next idle check); 0 disables idle-exit.
    pub idle_exit_timeout_ms: u64,
    /// Requires rebuilding the wake detector (rustpotter bakes the threshold in
    /// at construction).
    pub wake_sensitivity: f32,
    /// Per-event response stall deadline (#58). Hot-swapped on the pipeline
    /// (affects the next turn). 0 disables.
    pub response_stall_ms: u64,
    /// Overall per-turn response budget (#58). Hot-swapped on the pipeline
    /// (affects the next turn). 0 disables.
    pub turn_budget_ms: u64,
    /// Minimum gap between spoken status narrations (#58). Hot-swapped on the
    /// pipeline (affects the next turn).
    pub status_narration_min_gap_ms: u64,
    /// Changing either device requires restarting the capture/playback stream,
    /// which we do not do live — the daemon logs "restart required" instead.
    pub input_device: String,
    pub output_device: String,
}

impl Config {
    /// Snapshot the live-applicable knobs for diffing on reload.
    pub fn tunables(&self) -> Tunables {
        Tunables {
            speech_threshold: self.vad.speech_threshold,
            silence_duration_ms: self.vad.silence_duration_ms,
            followup_timeout_ms: self.assistant.followup_timeout_ms,
            conversation_reuse_window_ms: self.assistant.conversation_reuse_window_ms,
            conversation_mode: self.assistant.conversation_mode,
            idle_exit_timeout_ms: self.idle_exit_timeout_ms,
            wake_sensitivity: self.wake_word.sensitivity,
            response_stall_ms: self.timeouts.response_stall_ms,
            turn_budget_ms: self.timeouts.turn_budget_ms,
            status_narration_min_gap_ms: self.timeouts.status_narration_min_gap_ms,
            input_device: self.audio.input_device.clone(),
            output_device: self.audio.output_device.clone(),
        }
    }
}

/// The work a reload implies, derived purely from the old/new [`Tunables`]. Pure
/// and side-effect-free so the apply decision is unit-tested without audio,
/// file-watching, or a real pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReloadPlan {
    /// Apply `endpointer.set_speech_threshold`.
    pub set_speech_threshold: Option<f32>,
    /// Apply `endpointer.set_silence`.
    pub set_silence_ms: Option<u64>,
    /// Update the pipeline's follow-up timeout.
    pub set_followup_timeout_ms: Option<u64>,
    /// Update the pipeline's cross-wake conversation reuse window (voice#53; 0
    /// disables).
    pub set_conversation_reuse_window_ms: Option<u64>,
    /// Update the pipeline's conversation-mode flag.
    pub set_conversation_mode: Option<bool>,
    /// Update the pipeline's idle-exit timeout (0 disables).
    pub set_idle_exit_timeout_ms: Option<u64>,
    /// Update the pipeline's per-event response stall deadline (#58, 0 disables).
    pub set_response_stall_ms: Option<u64>,
    /// Update the pipeline's overall per-turn budget (#58, 0 disables).
    pub set_turn_budget_ms: Option<u64>,
    /// Update the pipeline's status-narration min gap (#58).
    pub set_status_narration_min_gap_ms: Option<u64>,
    /// Rebuild the wake detector at this sensitivity.
    pub rebuild_wake_sensitivity: Option<f32>,
    /// A device changed; the capture/playback stream would need a restart, which
    /// is not done live. Carries the human-readable change for the log.
    pub restart_required_for_device: Option<String>,
}

impl ReloadPlan {
    /// True when nothing changed — the watcher can skip a no-op reload.
    pub fn is_empty(&self) -> bool {
        *self == ReloadPlan::default()
    }
}

/// Diff two tunable snapshots into the concrete work a reload implies.
///
/// - `speech_threshold`, `silence_duration_ms`, `followup_timeout_ms`,
///   `conversation_mode`, `idle_exit_timeout_ms` hot-apply in place.
/// - `wake_sensitivity` requires rebuilding the wake detector (rustpotter bakes
///   the threshold in at construction).
/// - an `input_device`/`output_device` change can't be applied live (it would
///   need restarting the cpal stream); the plan flags a restart-required note
///   instead, and every other changed knob still applies.
pub fn plan_reload(old: &Tunables, new: &Tunables) -> ReloadPlan {
    let mut plan = ReloadPlan::default();
    if old.speech_threshold != new.speech_threshold {
        plan.set_speech_threshold = Some(new.speech_threshold);
    }
    if old.silence_duration_ms != new.silence_duration_ms {
        plan.set_silence_ms = Some(new.silence_duration_ms);
    }
    if old.followup_timeout_ms != new.followup_timeout_ms {
        plan.set_followup_timeout_ms = Some(new.followup_timeout_ms);
    }
    if old.conversation_reuse_window_ms != new.conversation_reuse_window_ms {
        plan.set_conversation_reuse_window_ms = Some(new.conversation_reuse_window_ms);
    }
    if old.conversation_mode != new.conversation_mode {
        plan.set_conversation_mode = Some(new.conversation_mode);
    }
    if old.idle_exit_timeout_ms != new.idle_exit_timeout_ms {
        plan.set_idle_exit_timeout_ms = Some(new.idle_exit_timeout_ms);
    }
    if old.response_stall_ms != new.response_stall_ms {
        plan.set_response_stall_ms = Some(new.response_stall_ms);
    }
    if old.turn_budget_ms != new.turn_budget_ms {
        plan.set_turn_budget_ms = Some(new.turn_budget_ms);
    }
    if old.status_narration_min_gap_ms != new.status_narration_min_gap_ms {
        plan.set_status_narration_min_gap_ms = Some(new.status_narration_min_gap_ms);
    }
    if old.wake_sensitivity != new.wake_sensitivity {
        plan.rebuild_wake_sensitivity = Some(new.wake_sensitivity);
    }
    if old.input_device != new.input_device || old.output_device != new.output_device {
        plan.restart_required_for_device = Some(format!(
            "input {:?}->{:?}, output {:?}->{:?}",
            old.input_device, new.input_device, old.output_device, new.output_device
        ));
    }
    plan
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
    fn no_change_yields_an_empty_plan() {
        let t = Config::default().tunables();
        let plan = plan_reload(&t, &t);
        assert!(
            plan.is_empty(),
            "an unchanged config must be a no-op reload"
        );
    }

    #[test]
    fn hot_knobs_apply_in_place() {
        let old = Config::default().tunables();
        let new = Tunables {
            speech_threshold: old.speech_threshold + 0.1,
            silence_duration_ms: old.silence_duration_ms + 100,
            followup_timeout_ms: old.followup_timeout_ms + 1000,
            conversation_mode: !old.conversation_mode,
            idle_exit_timeout_ms: old.idle_exit_timeout_ms + 5000,
            ..old.clone()
        };
        let plan = plan_reload(&old, &new);
        assert_eq!(plan.set_speech_threshold, Some(new.speech_threshold));
        assert_eq!(plan.set_silence_ms, Some(new.silence_duration_ms));
        assert_eq!(plan.set_followup_timeout_ms, Some(new.followup_timeout_ms));
        assert_eq!(plan.set_conversation_mode, Some(new.conversation_mode));
        assert_eq!(
            plan.set_idle_exit_timeout_ms,
            Some(new.idle_exit_timeout_ms)
        );
        // No rebuild / restart implied by the hot knobs alone.
        assert_eq!(plan.rebuild_wake_sensitivity, None);
        assert_eq!(plan.restart_required_for_device, None);
        assert!(!plan.is_empty());
    }

    #[test]
    fn wake_sensitivity_change_requires_rebuilding_the_detector() {
        let old = Config::default().tunables();
        let new = Tunables {
            wake_sensitivity: old.wake_sensitivity + 0.2,
            ..old.clone()
        };
        let plan = plan_reload(&old, &new);
        assert_eq!(plan.rebuild_wake_sensitivity, Some(new.wake_sensitivity));
        // Only the wake detector — nothing else changed.
        assert_eq!(plan.set_speech_threshold, None);
        assert_eq!(plan.restart_required_for_device, None);
    }

    #[test]
    fn device_change_flags_restart_required_and_does_not_block_other_knobs() {
        let old = Config::default().tunables();
        let new = Tunables {
            input_device: "hw:1".into(),
            speech_threshold: old.speech_threshold + 0.1,
            ..old.clone()
        };
        let plan = plan_reload(&old, &new);
        // A device change can't apply live → flagged for restart …
        assert!(plan.restart_required_for_device.is_some());
        // … but the hot knob in the same edit still applies.
        assert_eq!(plan.set_speech_threshold, Some(new.speech_threshold));
    }

    #[test]
    fn output_device_change_also_flags_restart() {
        let old = Config::default().tunables();
        let new = Tunables {
            output_device: "hw:2".into(),
            ..old.clone()
        };
        let plan = plan_reload(&old, &new);
        assert!(plan.restart_required_for_device.is_some());
        assert_eq!(plan.set_speech_threshold, None);
    }

    #[test]
    fn timeout_defaults_are_sane_and_parse_from_toml() {
        // #58: the new [timeouts] section defaults to bounded values and
        // round-trips from TOML, with unspecified knobs keeping their defaults.
        let d = TimeoutConfig::default();
        assert!(d.response_stall_ms > 0 && d.turn_budget_ms > d.response_stall_ms);
        assert!(d.stt_ms > 0 && d.tts_ms > 0 && d.connect_ms > 0);

        let cfg: Config = toml::from_str(
            r#"
[timeouts]
response_stall_ms = 5000
stt_ms = 0
"#,
        )
        .unwrap();
        assert_eq!(cfg.timeouts.response_stall_ms, 5000);
        assert_eq!(cfg.timeouts.stt_ms, 0, "0 disables the STT bound");
        // Unspecified knobs keep their defaults.
        assert_eq!(
            cfg.timeouts.turn_budget_ms,
            TimeoutConfig::default().turn_budget_ms
        );
    }

    #[test]
    fn timeout_knob_changes_hot_apply() {
        // #58: the hot-reloadable timeout knobs (stall / budget / narration gap)
        // produce an apply plan; the restart-only ones (stt/tts/connect) don't
        // need to and aren't part of Tunables.
        let old = Config::default().tunables();
        let new = Tunables {
            response_stall_ms: old.response_stall_ms + 1000,
            turn_budget_ms: old.turn_budget_ms + 10_000,
            status_narration_min_gap_ms: old.status_narration_min_gap_ms + 5000,
            ..old.clone()
        };
        let plan = plan_reload(&old, &new);
        assert_eq!(plan.set_response_stall_ms, Some(new.response_stall_ms));
        assert_eq!(plan.set_turn_budget_ms, Some(new.turn_budget_ms));
        assert_eq!(
            plan.set_status_narration_min_gap_ms,
            Some(new.status_narration_min_gap_ms)
        );
        assert_eq!(plan.rebuild_wake_sensitivity, None);
    }

    #[test]
    fn conversation_reuse_window_defaults_to_ten_minutes_and_parses() {
        // voice#53: the reuse window defaults to 10 min and round-trips from TOML.
        assert_eq!(
            AssistantConfig::default().conversation_reuse_window_ms,
            600_000
        );
        let cfg: Config =
            toml::from_str("[assistant]\nconversation_reuse_window_ms = 120000\n").unwrap();
        assert_eq!(cfg.assistant.conversation_reuse_window_ms, 120_000);
        // 0 disables reuse (always fresh).
        let off: Config =
            toml::from_str("[assistant]\nconversation_reuse_window_ms = 0\n").unwrap();
        assert_eq!(off.assistant.conversation_reuse_window_ms, 0);
    }

    #[test]
    fn client_tools_default_on_and_parse_per_tool_toggles() {
        // voice#61: each session-control tool defaults on and can be disabled
        // individually via [assistant.client_tools].
        let d = ClientToolsConfig::default();
        assert!(d.stop_listening && d.listen_for_more && d.say_this);

        let cfg: Config = toml::from_str("[assistant.client_tools]\nsay_this = false\n").unwrap();
        assert!(cfg.assistant.client_tools.stop_listening);
        assert!(cfg.assistant.client_tools.listen_for_more);
        assert!(
            !cfg.assistant.client_tools.say_this,
            "an explicitly-disabled tool must be off; unspecified ones stay on"
        );
    }

    #[test]
    fn conversation_reuse_window_change_hot_applies() {
        // voice#53: the reuse window is a hot-reloadable tunable.
        let old = Config::default().tunables();
        let new = Tunables {
            conversation_reuse_window_ms: old.conversation_reuse_window_ms + 60_000,
            ..old.clone()
        };
        let plan = plan_reload(&old, &new);
        assert_eq!(
            plan.set_conversation_reuse_window_ms,
            Some(new.conversation_reuse_window_ms)
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
