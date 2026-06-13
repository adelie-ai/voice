use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod config;
mod cue;
mod pipeline;
mod session;
mod tts_service;

use adele_voice_assistant_connector::ConnectorAssistantGateway;
use adele_voice_audio_cpal::{CpalAudioSink, CpalAudioSource};
use adele_voice_core::domain::State;
use adele_voice_core::ports::audio::AudioSink;
use adele_voice_dbus_interface::{CaptureState, DbusVoiceAdapter, StopRequest, TtsCommand};
use adele_voice_module::{Speaker, TtsBackend};
use adele_voice_stt_whisper::WhisperStt;
use adele_voice_vad_silero::SileroVad;
use adele_voice_wake_rustpotter::RustpotterWakeWordDetector;

const DBUS_VOICE_PATH: &str = "/org/desktopAssistant/Voice";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            // `ort` logs onnxruntime's allocator/arena activity at INFO (very
            // noisy for the Kokoro session); keep it at warn by default.
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,ort=warn".into()),
        )
        .init();

    let config = config::load()?;

    if std::env::args()
        .skip(1)
        .any(|a| a == "check-setup" || a == "--check")
    {
        check_setup(&config);
        return Ok(());
    }

    if std::env::args().skip(1).any(|a| a == "list-devices") {
        list_devices();
        return Ok(());
    }

    tracing::info!("adele-voice starting");

    // Snapshot the live-applicable knobs now, before fields move into the
    // pipeline, so a reload can diff the on-disk config against the running
    // values (config#52).
    let tunables = config.tunables();

    // Initialize components. The wake builder rebuilds the detector at a new
    // sensitivity on reload (rustpotter bakes the threshold in at construction).
    // `eager` (#50) is captured at startup — changing it needs a restart.
    let wake_model_path = config.wake_word.model_path.clone();
    let wake_eager = config.wake_word.eager;
    let wake_builder: pipeline::WakeBuilder<RustpotterWakeWordDetector> = {
        let model_path = wake_model_path.clone();
        Box::new(move |sensitivity| {
            RustpotterWakeWordDetector::new(&model_path, sensitivity, wake_eager)
        })
    };
    let wake = wake_builder(config.wake_word.sensitivity)?;
    let vad = SileroVad::new(&config.vad.model_path)?;
    // STT decode is bounded (#58): a wedged inference apologizes instead of
    // hanging the turn. 0 disables the bound.
    let stt = if config.timeouts.stt_ms == 0 {
        WhisperStt::new(&config.stt.model_path, &config.stt.language)?
    } else {
        WhisperStt::with_timeout(
            &config.stt.model_path,
            &config.stt.language,
            Duration::from_millis(config.timeouts.stt_ms),
        )?
    };
    // Select the TTS backend (local-first: Kokoro→Piper fallback, never auto
    // cloud). The conversation pipeline and the SayText service share it.
    let tts = TtsBackend::from_config(&config.tts).await;
    // Reach the orchestrator over the configured transport — local UDS by
    // default, or a remote WebSocket / legacy D-Bus (voice#31). An orchestrator
    // that isn't up yet must NOT kill the daemon (voice#86): we'd crash-loop
    // under systemd during a session-start race even though the gateway already
    // reconnects per-call. `connect_or_degrade` starts disconnected if needed
    // and connects lazily once the orchestrator appears, so wake word, D-Bus and
    // TTS keep serving meanwhile.
    let connection_config = config.assistant.connection_config();
    let assistant = ConnectorAssistantGateway::connect_or_degrade(&connection_config).await;

    let source = Arc::new(CpalAudioSource::new(&config.audio.input_device));
    let sink: Arc<dyn AudioSink> = Arc::new(CpalAudioSink::new(&config.audio.output_device));

    // State channels
    let (state_tx, state_rx) = tokio::sync::watch::channel(State::Idle);
    // Real capture (mic-open) state, surfaced over D-Bus so the KDE overlay /
    // health report can show whether the mic is actually open vs paused
    // (voice#103). The pipeline publishes; the D-Bus adapter reads.
    let (capture_state_tx, capture_state_rx) = tokio::sync::watch::channel(CaptureState::Capturing);
    let (enabled_tx, enabled_rx) = tokio::sync::watch::channel(true);
    // PTT payload: the target conversation id (None = the daemon's own session).
    let (ptt_tx, ptt_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);
    let (stop_tx, stop_rx) = tokio::sync::mpsc::channel::<StopRequest>(1);
    // Reload ping: the config-file watcher and the D-Bus `Reload` method both
    // ask the pipeline to re-read config.toml and apply changed tunables (#52).
    let (reload_tx, reload_rx) = tokio::sync::mpsc::channel::<()>(4);

    // Text-to-speech service backing SayText/SynthesizeText and voice
    // selection. Shares the backend instance (TTS clones share the active-voice
    // state, so SetVoice affects both) and, via a `Speaker`, the audio sink —
    // so spoken replies and SayText queue onto one playback stream.
    let (tts_tx, tts_rx) = tokio::sync::mpsc::channel::<TtsCommand>(16);
    let models_dir = config
        .tts
        .model_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let tts_handle = Arc::new(tts.clone());
    let mut say_speaker = Speaker::new(Arc::clone(&tts_handle), Arc::clone(&sink));
    // Bound on-demand SayText synth too, so a wedged backend can't hang the
    // SayText service (#58). 0 disables.
    if config.timeouts.tts_ms > 0 {
        say_speaker.set_synth_timeout(Duration::from_millis(config.timeouts.tts_ms));
    }
    tokio::spawn(tts_service::run_tts_service(
        say_speaker,
        tts_handle,
        models_dir,
        tts_rx,
    ));

    // Per-turn text-signal channel (voice#85): the pipeline reports the
    // transcript and the lines it speaks; a forwarder task emits them (and state
    // transitions) as the org.desktopAssistant.Voice D-Bus signals so clients
    // (the KDE widget) react without polling GetState. A small bound is fine —
    // emission is best-effort (`try_send`), and the consumer just relays.
    let (signal_tx, signal_rx) =
        tokio::sync::mpsc::channel::<adele_voice_dbus_interface::VoiceSignal>(32);

    // D-Bus server interface
    let dbus_adapter = DbusVoiceAdapter::new(
        state_rx.clone(),
        capture_state_rx,
        enabled_tx,
        enabled_rx.clone(),
        ptt_tx,
        stop_tx,
        tts_tx,
        reload_tx.clone(),
    );

    let connection = zbus::Connection::session().await?;
    connection
        .object_server()
        .at(DBUS_VOICE_PATH, dbus_adapter)
        .await?;
    connection
        .request_name("org.desktopAssistant.Voice")
        .await?;
    tracing::info!("D-Bus interface registered at {DBUS_VOICE_PATH}");

    // Emit StateChanged / TranscriptReady / SpeakingText from the pipeline's
    // state watch + the signal channel (voice#85).
    {
        let emitter = zbus::object_server::SignalEmitter::new(&connection, DBUS_VOICE_PATH)?;
        tokio::spawn(adele_voice_dbus_interface::run_signal_forwarder(
            emitter, state_rx, signal_rx,
        ));
    }

    // Watch the config file and ping the pipeline (debounced) on edits made
    // outside the KCM, e.g. a hand-edit, so live tuning works either way (#52).
    spawn_config_watcher(reload_tx);

    // Turn timeouts (#58): bound the synth, response heartbeat, overall budget,
    // connect round-trips, and status-narration cadence. 0 disables a bound.
    let turn_timeouts = pipeline::TurnTimeouts {
        response_stall: Duration::from_millis(config.timeouts.response_stall_ms),
        turn_budget: Duration::from_millis(config.timeouts.turn_budget_ms),
        synth: Duration::from_millis(config.timeouts.tts_ms),
        connect: Duration::from_millis(config.timeouts.connect_ms),
        status_narration_min_gap: Duration::from_millis(
            config.timeouts.status_narration_min_gap_ms,
        ),
        liveness_delay: Duration::from_millis(config.timeouts.narration_liveness_delay_ms),
        narration_flush: Duration::from_millis(config.timeouts.narration_flush_ms),
    };

    // logind session gating (voice#103): release the mic when the session goes
    // inactive (fast user switch) so the daemon doesn't keep recording the
    // foreground user. Capability-detected: absent logind => inert gate (capture
    // as before); see `session.rs` for the three-state model.
    let session_gate =
        session::spawn_session_gate(config.wake_word.pause_on_session_inactive).await;

    // Build and run pipeline
    let pipeline = pipeline::Pipeline::new(
        wake,
        vad,
        stt,
        tts,
        assistant,
        source,
        sink,
        state_tx,
        enabled_rx,
        ptt_rx,
        stop_rx,
        reload_rx,
        wake_builder,
        pipeline::PipelineConfig {
            tunables,
            conversation_title: config.assistant.conversation_title,
            silence_duration: Duration::from_millis(config.vad.silence_duration_ms),
            speech_threshold: config.vad.speech_threshold,
            conversation_mode: config.assistant.conversation_mode,
            conversation_reuse_window: Duration::from_millis(
                config.assistant.conversation_reuse_window_ms,
            ),
            followup_timeout: Duration::from_millis(config.assistant.followup_timeout_ms),
            idle_exit_timeout: (config.idle_exit_timeout_ms > 0)
                .then(|| Duration::from_millis(config.idle_exit_timeout_ms)),
            spoken_response_hint: config.assistant.spoken_response_hint,
            listening_cue: config.wake_word.listening_cue,
            timeouts: turn_timeouts,
            client_tools: config.assistant.client_tools,
        },
    )
    .with_signal_tx(signal_tx)
    .with_session_gate(session_gate)
    .with_capture_state(capture_state_tx);

    tokio::select! {
        result = pipeline.run() => {
            if let Err(e) = result {
                tracing::error!("pipeline error: {e}");
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("adele-voice shutting down");
        }
    }

    Ok(())
}

/// Watch the config file for edits and ping the pipeline to reload (#52),
/// debounced so a flurry of write events (editors often write+rename+chmod)
/// collapses into a single reload.
///
/// We watch the *parent directory* rather than the file: many editors replace
/// the config via a temp-file rename, which would break a watch bound to the
/// original inode. A blocking `notify` watcher feeds a std channel; a Tokio task
/// debounces and forwards a single `()` ping per quiet window. The KCM gets
/// instant reload via the D-Bus `Reload` method; this covers edits made any
/// other way (hand-edits, other tools).
fn spawn_config_watcher(reload_tx: tokio::sync::mpsc::Sender<()>) {
    use notify::{RecursiveMode, Watcher};

    let path = config::config_path();
    let dir = match path.parent() {
        Some(d) => d.to_path_buf(),
        None => {
            tracing::warn!("config watcher: config path has no parent dir, not watching");
            return;
        }
    };
    let file_name = path.file_name().map(std::ffi::OsString::from);

    // notify's callback runs on its own (non-async) thread; bridge to a std
    // mpsc, then debounce on a dedicated blocking thread that forwards into the
    // async reload channel. A blocking thread (not a Tokio task) is used because
    // the std `recv()` would otherwise park a runtime worker.
    let (raw_tx, raw_rx) = std::sync::mpsc::channel::<()>();
    let watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            // Only react to events touching our config file (the dir may hold
            // other files). Match by file name; rename targets count too.
            let touches_config = file_name.as_ref().is_none_or(|name| {
                event
                    .paths
                    .iter()
                    .any(|p| p.file_name() == Some(name.as_os_str()))
            });
            let relevant = matches!(
                event.kind,
                notify::EventKind::Modify(_)
                    | notify::EventKind::Create(_)
                    | notify::EventKind::Remove(_)
            );
            if touches_config && relevant {
                let _ = raw_tx.send(());
            }
        }
    });
    let watcher = match watcher {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(
                "config watcher: failed to create watcher, live reload on file edits disabled: {e}"
            );
            return;
        }
    };
    let mut watcher = watcher;
    if let Err(e) = watcher.watch(&dir, RecursiveMode::NonRecursive) {
        tracing::warn!(
            dir = %dir.display(),
            "config watcher: failed to watch dir, live reload on file edits disabled: {e}"
        );
        return;
    }

    std::thread::spawn(move || {
        // Keep the watcher alive for the life of the thread (dropping it stops
        // watching).
        let _watcher = watcher;
        // Block until the first raw event, then wait out a short quiet window
        // and drain any burst — collapsing an editor's write/rename/chmod storm
        // into one reload.
        while raw_rx.recv().is_ok() {
            std::thread::sleep(std::time::Duration::from_millis(300));
            while raw_rx.try_recv().is_ok() {}
            tracing::info!("config file changed, reloading");
            if reload_tx.blocking_send(()).is_err() {
                break; // pipeline gone
            }
        }
    });
}

/// True if `bin` resolves to an existing file — an explicit path, or a bare
/// name found on `PATH`. Used instead of `<bin> --version`, which some TTS
/// CLIs (notably piper-tts) don't implement, yielding a false "missing".
fn binary_resolves(bin: &str) -> bool {
    if bin.contains('/') {
        return std::path::Path::new(bin).is_file();
    }
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(bin).is_file()))
        .unwrap_or(false)
}

/// Print the input devices capture can actually open, as JSON, for the KCM
/// picker (adele-kde). Each entry is probed through the real capture negotiation
/// so `supported` accounts for the downmix/resample fallback. Run with
/// `adele-voice list-devices`.
fn list_devices() {
    use serde_json::json;

    // The synthetic "default" route is always valid: capture special-cases it to
    // the host default input device.
    let mut arr = vec![json!({
        "value": "default",
        "label": "Follow system default (recommended)",
        "is_default": true,
        "kind": "default",
        "supported": true,
        "rate": serde_json::Value::Null,
        "channels": serde_json::Value::Null,
        "reason": serde_json::Value::Null,
    })];

    match adele_voice_audio_cpal::probe_input_devices() {
        Ok(devices) => {
            for d in devices {
                arr.push(json!({
                    "value": d.value,
                    "label": d.label,
                    "is_default": d.is_default,
                    "kind": d.kind,
                    "supported": d.supported,
                    "rate": d.rate,
                    "channels": d.channels,
                    "reason": d.reason,
                }));
            }
        }
        Err(e) => {
            eprintln!("adele-voice: device probe failed: {e}");
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Array(arr))
            .unwrap_or_else(|_| "[]".into())
    );
}

/// Print which backends are provisioned/available, which one is configured, and
/// the local-first fallback policy. Run with `adele-voice check-setup`.
fn check_setup(config: &config::Config) {
    use std::path::Path;
    let mark = |b: bool| if b { "ok  " } else { "MISS" };
    let is_file = |p: &Path| p.is_file();
    let runnable = |bin: &str| {
        std::process::Command::new(bin)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    println!("adele-voice — setup check\n");
    println!("Base models:");
    println!(
        "  [{}] Silero VAD   {}",
        mark(is_file(&config.vad.model_path)),
        config.vad.model_path.display()
    );
    println!(
        "  [{}] Whisper STT  {}",
        mark(is_file(&config.stt.model_path)),
        config.stt.model_path.display()
    );
    println!(
        "  [{}] Wake word    {}",
        mark(is_file(&config.wake_word.model_path)),
        config.wake_word.model_path.display()
    );

    println!("\nTTS backends:");
    let kokoro_model = is_file(&config.tts.kokoro_model_path);
    let voices = std::fs::read_dir(&config.tts.kokoro_voices_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("bin"))
        .count();
    let espeak = runnable("espeak-ng");
    println!(
        "  [{}] kokoro (local, DEFAULT)   model:{} voices:{} espeak-ng:{}",
        mark(kokoro_model && voices > 0 && espeak),
        if kokoro_model { "yes" } else { "no" },
        voices,
        if espeak { "yes" } else { "NO" }
    );
    let piper = binary_resolves(&config.tts.piper_binary);
    println!(
        "  [{}] piper  (local)            binary:{} voice:{}",
        mark(piper && is_file(&config.tts.model_path)),
        if piper { "yes" } else { "no" },
        if is_file(&config.tts.model_path) {
            "yes"
        } else {
            "no"
        }
    );
    let aws = std::env::var_os("AWS_PROFILE").is_some()
        || std::env::var_os("AWS_ACCESS_KEY_ID").is_some()
        || dirs::home_dir()
            .map(|h| h.join(".aws/credentials").is_file())
            .unwrap_or(false);
    println!(
        "  [{}] polly  (CLOUD, BILLABLE)  AWS creds:{}  — opt-in only",
        if aws { "ok  " } else { "--  " },
        if aws { "present" } else { "none" }
    );

    let active_voice = match config.tts.backend.as_str() {
        "kokoro" => config.tts.kokoro_voice.clone(),
        "polly" => format!("{} ({})", config.tts.polly_voice, config.tts.polly_engine),
        _ => config
            .tts
            .model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .into(),
    };
    println!(
        "\nConfigured backend: {}  (voice: {active_voice})",
        config.tts.backend
    );
    println!(
        "Fallback policy: if the configured backend can't initialize it falls back to a LOCAL"
    );
    println!("backend (Piper) — never to a billable cloud backend automatically.");
}

#[cfg(test)]
mod tests {
    use super::binary_resolves;

    #[test]
    fn resolves_bare_name_on_path() {
        // `sh` is on PATH on any Linux/macOS host.
        assert!(binary_resolves("sh"));
    }

    #[test]
    fn resolves_explicit_existing_path() {
        let exe = std::env::current_exe().unwrap();
        assert!(binary_resolves(exe.to_str().unwrap()));
    }

    #[test]
    fn rejects_missing_binary() {
        assert!(!binary_resolves("definitely-not-a-real-binary-xyzzy"));
    }
}
