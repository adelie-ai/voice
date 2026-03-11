use std::sync::Arc;

use adele_voice_core::domain::State;
use tokio::sync::watch;
use zbus::object_server::SignalEmitter;
use zbus::{fdo, interface};

/// D-Bus adapter exposing org.desktopAssistant.Voice.
pub struct DbusVoiceAdapter {
    state_rx: watch::Receiver<State>,
    enabled_tx: Arc<watch::Sender<bool>>,
    enabled_rx: watch::Receiver<bool>,
    ptt_tx: tokio::sync::mpsc::Sender<()>,
    stop_tx: tokio::sync::mpsc::Sender<()>,
}

impl DbusVoiceAdapter {
    pub fn new(
        state_rx: watch::Receiver<State>,
        enabled_tx: watch::Sender<bool>,
        enabled_rx: watch::Receiver<bool>,
        ptt_tx: tokio::sync::mpsc::Sender<()>,
        stop_tx: tokio::sync::mpsc::Sender<()>,
    ) -> Self {
        Self {
            state_rx,
            enabled_tx: Arc::new(enabled_tx),
            enabled_rx,
            ptt_tx,
            stop_tx,
        }
    }
}

#[interface(name = "org.desktopAssistant.Voice")]
impl DbusVoiceAdapter {
    /// Get the current pipeline state: "Idle", "Listening", "Processing", or "Speaking".
    async fn get_state(&self) -> fdo::Result<String> {
        Ok(self.state_rx.borrow().to_string())
    }

    /// Enable or disable voice processing.
    async fn set_enabled(&self, enabled: bool) -> fdo::Result<()> {
        self.enabled_tx
            .send(enabled)
            .map_err(|e| fdo::Error::Failed(format!("failed to set enabled: {e}")))?;
        tracing::info!(enabled, "voice processing toggled");
        Ok(())
    }

    /// Get whether voice processing is enabled.
    async fn get_enabled(&self) -> fdo::Result<bool> {
        Ok(*self.enabled_rx.borrow())
    }

    /// Trigger push-to-talk (skip wake word, go directly to Listening).
    async fn push_to_talk(&self) -> fdo::Result<()> {
        self.ptt_tx
            .send(())
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to trigger PTT: {e}")))?;
        Ok(())
    }

    /// Stop any ongoing speech playback.
    async fn stop_speaking(&self) -> fdo::Result<()> {
        self.stop_tx
            .send(())
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to stop speaking: {e}")))?;
        Ok(())
    }

    /// Signal emitted when the pipeline state changes.
    #[zbus(signal)]
    pub async fn state_changed(emitter: &SignalEmitter<'_>, state: &str) -> zbus::Result<()>;

    /// Signal emitted when a transcript is ready.
    #[zbus(signal)]
    pub async fn transcript_ready(emitter: &SignalEmitter<'_>, text: &str) -> zbus::Result<()>;

    /// Signal emitted when Adele starts speaking a sentence.
    #[zbus(signal)]
    pub async fn speaking_text(emitter: &SignalEmitter<'_>, text: &str) -> zbus::Result<()>;
}
