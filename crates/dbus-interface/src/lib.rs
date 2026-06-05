use std::sync::Arc;

use adele_voice_core::domain::State;
use tokio::sync::{mpsc, oneshot, watch};
use zbus::object_server::SignalEmitter;
use zbus::{fdo, interface};

/// A request to the text-to-speech service backing `SayText` /
/// `SynthesizeText`. Processed on a single task so requests serialize rather
/// than collide.
pub enum TtsCommand {
    /// Synthesize and play the text through the daemon's audio sink.
    Say(String),
    /// Synthesize the text and return it as WAV (16-bit PCM mono) bytes
    /// without playing it.
    Synthesize {
        text: String,
        reply: oneshot::Sender<Result<Vec<u8>, String>>,
    },
    /// List installed voices as (id, display name, language, num_speakers).
    ListVoices {
        reply: oneshot::Sender<Vec<(String, String, String, u32)>>,
    },
    /// Get the current voice as (id, speaker_id); speaker_id is -1 if unset.
    GetVoice {
        reply: oneshot::Sender<(String, i32)>,
    },
    /// Set the active voice (and optional speaker id; -1 for default/single).
    SetVoice {
        voice_id: String,
        speaker: i32,
        reply: oneshot::Sender<Result<(), String>>,
    },
}

/// A push-to-talk trigger. The payload is the conversation the utterance
/// should be routed to: `None` uses the daemon's own session ("Voice
/// Conversation"); `Some(id)` targets the orchestrator conversation with that
/// id (the id returned by `org.desktopAssistant.Conversations.CreateConversation`),
/// so the mic button in a chat window dictates into that chat.
pub type PttRequest = Option<String>;

/// D-Bus adapter exposing org.desktopAssistant.Voice.
pub struct DbusVoiceAdapter {
    state_rx: watch::Receiver<State>,
    enabled_tx: Arc<watch::Sender<bool>>,
    enabled_rx: watch::Receiver<bool>,
    ptt_tx: mpsc::Sender<PttRequest>,
    stop_tx: mpsc::Sender<()>,
    tts_tx: mpsc::Sender<TtsCommand>,
}

impl DbusVoiceAdapter {
    pub fn new(
        state_rx: watch::Receiver<State>,
        enabled_tx: watch::Sender<bool>,
        enabled_rx: watch::Receiver<bool>,
        ptt_tx: mpsc::Sender<PttRequest>,
        stop_tx: mpsc::Sender<()>,
        tts_tx: mpsc::Sender<TtsCommand>,
    ) -> Self {
        Self {
            state_rx,
            enabled_tx: Arc::new(enabled_tx),
            enabled_rx,
            ptt_tx,
            stop_tx,
            tts_tx,
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

    /// Trigger push-to-talk (skip wake word, go directly to Listening). The
    /// utterance is routed to the daemon's own session ("Voice Conversation").
    async fn push_to_talk(&self) -> fdo::Result<()> {
        self.ptt_tx
            .send(None)
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to trigger PTT: {e}")))?;
        Ok(())
    }

    /// Trigger push-to-talk and route this utterance to a specific
    /// conversation instead of the daemon's own session. `conversation_id` is
    /// the orchestrator conversation id (as returned by
    /// `org.desktopAssistant.Conversations.CreateConversation` / `ListConversations`);
    /// an empty string falls back to the daemon's own session, matching
    /// `PushToTalk()`. Use this for the mic button inside a chat window so the
    /// spoken prompt and reply appear in the conversation the user is viewing.
    async fn push_to_talk_in_conversation(&self, conversation_id: &str) -> fdo::Result<()> {
        let target = Some(conversation_id.to_string()).filter(|id| !id.is_empty());
        self.ptt_tx
            .send(target)
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

    /// Speak the given text aloud with the on-device neural voice. Queues
    /// behind any in-progress speech; does NOT open the microphone.
    async fn say_text(&self, text: &str) -> fdo::Result<()> {
        self.tts_tx
            .send(TtsCommand::Say(text.to_string()))
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to queue SayText: {e}")))?;
        Ok(())
    }

    /// Synthesize the given text and return it as WAV (16-bit PCM mono) bytes
    /// without playing it — for callers (e.g. accessibility tools) that route
    /// their own audio.
    async fn synthesize_text(&self, text: &str) -> fdo::Result<Vec<u8>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tts_tx
            .send(TtsCommand::Synthesize {
                text: text.to_string(),
                reply: reply_tx,
            })
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to queue SynthesizeText: {e}")))?;
        reply_rx
            .await
            .map_err(|e| fdo::Error::Failed(format!("TTS service dropped the request: {e}")))?
            .map_err(fdo::Error::Failed)
    }

    /// List installed TTS voices as (id, display name, language, num_speakers).
    async fn list_voices(&self) -> fdo::Result<Vec<(String, String, String, u32)>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tts_tx
            .send(TtsCommand::ListVoices { reply: reply_tx })
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to queue ListVoices: {e}")))?;
        reply_rx
            .await
            .map_err(|e| fdo::Error::Failed(format!("TTS service dropped the request: {e}")))
    }

    /// Get the current voice as (id, speaker_id); speaker_id is -1 if unset.
    async fn get_voice(&self) -> fdo::Result<(String, i32)> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tts_tx
            .send(TtsCommand::GetVoice { reply: reply_tx })
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to queue GetVoice: {e}")))?;
        reply_rx
            .await
            .map_err(|e| fdo::Error::Failed(format!("TTS service dropped the request: {e}")))
    }

    /// Set the active voice by id (and optional multi-speaker id; -1 for the
    /// default). Affects both spoken responses and SayText.
    async fn set_voice(&self, voice_id: &str, speaker: i32) -> fdo::Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tts_tx
            .send(TtsCommand::SetVoice {
                voice_id: voice_id.to_string(),
                speaker,
                reply: reply_tx,
            })
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to queue SetVoice: {e}")))?;
        reply_rx
            .await
            .map_err(|e| fdo::Error::Failed(format!("TTS service dropped the request: {e}")))?
            .map_err(fdo::Error::Failed)
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
