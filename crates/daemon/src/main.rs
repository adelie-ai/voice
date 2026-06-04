use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod config;
mod pipeline;
mod tts_service;

use adele_voice_assistant_dbus::DbusAssistantGateway;
use adele_voice_audio_cpal::{CpalAudioSink, CpalAudioSource};
use adele_voice_core::domain::State;
use adele_voice_core::ports::audio::AudioSink;
use adele_voice_dbus_interface::{DbusVoiceAdapter, TtsCommand};
use adele_voice_stt_whisper::WhisperStt;
use adele_voice_tts_piper::PiperTts;
use adele_voice_vad_silero::SileroVad;
use adele_voice_wake_rustpotter::RustpotterWakeWordDetector;

const DBUS_VOICE_PATH: &str = "/org/desktopAssistant/Voice";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let config = config::load()?;
    tracing::info!("adele-voice starting");

    // Initialize components
    let wake = RustpotterWakeWordDetector::new(
        &config.wake_word.model_path,
        config.wake_word.sensitivity,
    )?;
    let vad = SileroVad::new(&config.vad.model_path)?;
    let stt = WhisperStt::new(&config.stt.model_path, &config.stt.language)?;
    let tts = PiperTts::new(&config.tts.piper_binary, &config.tts.model_path);
    let assistant = DbusAssistantGateway::connect().await?;

    let source = Arc::new(CpalAudioSource::new(&config.audio.input_device));
    let sink: Arc<dyn AudioSink> = Arc::new(CpalAudioSink::new(&config.audio.output_device));

    // State channels
    let (state_tx, state_rx) = tokio::sync::watch::channel(State::Idle);
    let (enabled_tx, enabled_rx) = tokio::sync::watch::channel(true);
    let (ptt_tx, ptt_rx) = tokio::sync::mpsc::channel(1);
    let (stop_tx, stop_rx) = tokio::sync::mpsc::channel(1);

    // Text-to-speech service backing SayText/SynthesizeText. Uses its own
    // Piper instance (Piper is stateless) and shares the audio sink with the
    // conversation pipeline, so all playback serializes through the sink.
    let (tts_tx, tts_rx) = tokio::sync::mpsc::channel::<TtsCommand>(16);
    let tts_service = Arc::new(PiperTts::new(
        &config.tts.piper_binary,
        &config.tts.model_path,
    ));
    tokio::spawn(tts_service::run_tts_service(
        tts_service,
        Arc::clone(&sink),
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
