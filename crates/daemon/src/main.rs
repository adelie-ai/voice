use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod config;
mod pipeline;
mod tts_backend;
mod tts_service;

use tts_backend::TtsBackend;

use adele_voice_assistant_dbus::DbusAssistantGateway;
use adele_voice_audio_cpal::{CpalAudioSink, CpalAudioSource};
use adele_voice_core::domain::State;
use adele_voice_core::ports::audio::AudioSink;
use adele_voice_dbus_interface::{DbusVoiceAdapter, StopRequest, TtsCommand};
use adele_voice_stt_whisper::WhisperStt;
use adele_voice_tts_kokoro::KokoroTts;
use adele_voice_tts_piper::PiperTts;
use adele_voice_tts_polly::PollyTts;
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

    tracing::info!("adele-voice starting");

    // Initialize components
    let wake = RustpotterWakeWordDetector::new(
        &config.wake_word.model_path,
        config.wake_word.sensitivity,
    )?;
    let vad = SileroVad::new(&config.vad.model_path)?;
    let stt = WhisperStt::new(&config.stt.model_path, &config.stt.language)?;
    let tts = match config.tts.backend.as_str() {
        "polly" => {
            tracing::info!(
                voice = %config.tts.polly_voice,
                engine = %config.tts.polly_engine,
                "using AWS Polly TTS backend"
            );
            TtsBackend::Polly(
                PollyTts::new(
                    &config.tts.polly_voice,
                    &config.tts.polly_engine,
                    config.tts.polly_region.clone(),
                )
                .await,
            )
        }
        "kokoro" => match KokoroTts::new(
            &config.tts.kokoro_model_path,
            &config.tts.kokoro_voices_dir,
            &config.tts.kokoro_voice,
            &config.tts.kokoro_lang,
        ) {
            Ok(k) => {
                tracing::info!(voice = %config.tts.kokoro_voice, "using local Kokoro TTS backend");
                TtsBackend::Kokoro(k)
            }
            Err(e) => {
                tracing::warn!(
                    "Kokoro init failed ({e}); falling back to Piper. Run scripts/setup.sh to provision Kokoro."
                );
                TtsBackend::Piper(PiperTts::new(
                    &config.tts.piper_binary,
                    &config.tts.model_path,
                ))
            }
        },
        other => {
            if other != "piper" {
                tracing::warn!(backend = %other, "unknown tts.backend, falling back to piper");
            }
            TtsBackend::Piper(PiperTts::new(
                &config.tts.piper_binary,
                &config.tts.model_path,
            ))
        }
    };
    let assistant = DbusAssistantGateway::connect().await?;

    let source = Arc::new(CpalAudioSource::new(&config.audio.input_device));
    let sink: Arc<dyn AudioSink> = Arc::new(CpalAudioSink::new(&config.audio.output_device));

    // State channels
    let (state_tx, state_rx) = tokio::sync::watch::channel(State::Idle);
    let (enabled_tx, enabled_rx) = tokio::sync::watch::channel(true);
    // PTT payload: the target conversation id (None = the daemon's own session).
    let (ptt_tx, ptt_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);
    let (stop_tx, stop_rx) = tokio::sync::mpsc::channel::<StopRequest>(1);

    // Text-to-speech service backing SayText/SynthesizeText and voice
    // selection. Shares the pipeline's Piper instance (PiperTts clones share
    // the active-voice state, so SetVoice affects both) and the audio sink.
    let (tts_tx, tts_rx) = tokio::sync::mpsc::channel::<TtsCommand>(16);
    let models_dir = config
        .tts
        .model_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    tokio::spawn(tts_service::run_tts_service(
        Arc::new(tts.clone()),
        Arc::clone(&sink),
        models_dir,
        tts_rx,
    ));

    // D-Bus server interface
    let dbus_adapter = DbusVoiceAdapter::new(
        state_rx,
        enabled_tx,
        enabled_rx.clone(),
        ptt_tx,
        stop_tx,
        tts_tx,
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
        config.assistant.conversation_title,
        Duration::from_millis(config.vad.silence_duration_ms),
        config.vad.speech_threshold,
        config.assistant.conversation_mode,
        Duration::from_millis(config.assistant.followup_timeout_ms),
        (config.idle_exit_timeout_ms > 0)
            .then(|| Duration::from_millis(config.idle_exit_timeout_ms)),
        config.assistant.spoken_response_hint,
    );

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
