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

/// A pipeline event that should be broadcast as a D-Bus signal so clients (the
/// KDE widget) can react without polling (voice#85). State transitions are
/// carried separately by the existing `State` watch channel; these are the
/// per-turn text events.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceSignal {
    /// A user utterance was transcribed (the clean transcript text).
    TranscriptReady(String),
    /// Adele is about to speak a sentence aloud (the sentence text).
    SpeakingText(String),
    /// Wake-word calibration progress (#121): `captured` utterances of `total`
    /// recorded so far, and the peak `score` of the most recent one. A negative
    /// `score` is a prompt rather than a measurement: `-1.0` = "speak the next
    /// utterance now", `-2.0` = "no clear wake word heard — try again", `-3.0` =
    /// "measuring background noise — stay quiet".
    CalibrationProgress {
        captured: u32,
        total: u32,
        score: f64,
    },
}

/// A request to run wake-word calibration on the pipeline (#121). The pipeline
/// takes over its mic capture briefly, measures `utterances` spoken wake-word
/// peaks (emitting [`VoiceSignal::CalibrationProgress`] as it goes), sets the
/// new cutoff live, persists it, and replies with the outcome.
pub struct CalibrationRequest {
    pub utterances: u32,
    pub reply: oneshot::Sender<Result<CalibrationOutcome, String>>,
}

/// The result of a calibration run: the cutoff AND wake mode that were applied
/// (and persisted), plus the measurements they were derived from and the
/// per-mode candidate cutoffs so a client can show both options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationOutcome {
    /// The wake sensitivity that was applied and persisted.
    pub sensitivity: f32,
    /// The wake mode that was applied and persisted (true = eager). Calibration
    /// picks the best available mode for the measured scores.
    pub eager: bool,
    /// How many utterance peaks the recommendation was based on.
    pub samples: u32,
    /// The mean observed utterance peak.
    pub mean_peak: f32,
    /// The measured background match level (ambient) — the floor a non-eager
    /// cutoff must sit above so the score can fall back below it.
    pub noise_floor: f32,
    /// The cutoff eager mode would use (a margin below the weakest peak).
    pub eager_cutoff: f32,
    /// The cutoff standard (non-eager) mode would use, or a negative value when
    /// standard mode isn't reliable for this mic (peaks too close to background).
    pub non_eager_cutoff: f32,
}

/// The real capture (microphone) state, separate from the pipeline `State`
/// (voice#103). The KDE mic overlay / health report needs the *truth* about
/// whether the mic is actually open: pre-#103, `State::Idle` could mean either
/// "listening for the wake word" (mic open) or "paused" (mic should be closed),
/// and the device stayed open even when "disabled" or the session was inactive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureState {
    /// The mic is open and the pipeline is processing audio.
    #[default]
    Capturing,
    /// Capture is released because the logind session is inactive (e.g. fast
    /// user switch) — privacy/cost gate (voice#103).
    PausedSessionInactive,
    /// Capture is released because voice processing was disabled
    /// (`SetEnabled(false)`) — the mic indicator must not stay lit.
    PausedDisabled,
    /// The input device couldn't be opened (degraded) — capture is unavailable
    /// until the device is fixed / a reload recovers it.
    Unavailable,
}

impl std::fmt::Display for CaptureState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Capturing => "Capturing",
            Self::PausedSessionInactive => "PausedSessionInactive",
            Self::PausedDisabled => "PausedDisabled",
            Self::Unavailable => "Unavailable",
        };
        f.write_str(s)
    }
}

/// A push-to-talk trigger. The payload is the conversation the utterance
/// should be routed to: `None` uses the daemon's own session ("Voice
/// Conversation"); `Some(id)` targets the orchestrator conversation with that
/// id (the id returned by `org.desktopAssistant.Conversations.CreateConversation`),
/// so the mic button in a chat window dictates into that chat.
pub type PttRequest = Option<String>;

/// How to interrupt the pipeline. `Speaking` cancels current playback
/// (barge-in) but leaves a conversation-mode session listening; `Conversation`
/// ends the whole interaction — stop any playback and return to wake-word idle,
/// clearing the session so a follow-up doesn't keep listening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopRequest {
    Speaking,
    Conversation,
}

/// D-Bus adapter exposing org.desktopAssistant.Voice.
pub struct DbusVoiceAdapter {
    state_rx: watch::Receiver<State>,
    /// The real capture (mic-open) state (voice#103), distinct from `state_rx`.
    capture_state_rx: watch::Receiver<CaptureState>,
    enabled_tx: Arc<watch::Sender<bool>>,
    enabled_rx: watch::Receiver<bool>,
    ptt_tx: mpsc::Sender<PttRequest>,
    stop_tx: mpsc::Sender<StopRequest>,
    tts_tx: mpsc::Sender<TtsCommand>,
    /// Pings the pipeline to re-read the config file and apply changed tunables
    /// (config#52). The KCM calls `Reload` after writing config.toml so live
    /// tuning takes effect without waiting for the file watcher.
    reload_tx: mpsc::Sender<()>,
    /// Hands a wake-word calibration request to the pipeline (#121) and awaits
    /// the outcome. The CLI and the KCM both drive calibration through this.
    calibrate_tx: mpsc::Sender<CalibrationRequest>,
}

impl DbusVoiceAdapter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state_rx: watch::Receiver<State>,
        capture_state_rx: watch::Receiver<CaptureState>,
        enabled_tx: watch::Sender<bool>,
        enabled_rx: watch::Receiver<bool>,
        ptt_tx: mpsc::Sender<PttRequest>,
        stop_tx: mpsc::Sender<StopRequest>,
        tts_tx: mpsc::Sender<TtsCommand>,
        reload_tx: mpsc::Sender<()>,
        calibrate_tx: mpsc::Sender<CalibrationRequest>,
    ) -> Self {
        Self {
            state_rx,
            capture_state_rx,
            enabled_tx: Arc::new(enabled_tx),
            enabled_rx,
            ptt_tx,
            stop_tx,
            tts_tx,
            reload_tx,
            calibrate_tx,
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

    /// Get the real capture (microphone) state (voice#103): "Capturing",
    /// "PausedSessionInactive", "PausedDisabled", or "Unavailable". Unlike
    /// `GetState` (the conversation state machine), this reflects whether the mic
    /// is actually open — so the KDE overlay / health report can show the truth
    /// (e.g. paused on fast-user-switch) instead of a stale "Idle/listening".
    async fn get_capture_state(&self) -> fdo::Result<String> {
        Ok(self.capture_state_rx.borrow().to_string())
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

    /// Stop any ongoing speech playback (barge-in). A conversation-mode session
    /// keeps listening afterward; use `StopListening` to end it.
    async fn stop_speaking(&self) -> fdo::Result<()> {
        self.stop_tx
            .send(StopRequest::Speaking)
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to stop speaking: {e}")))?;
        Ok(())
    }

    /// End the current conversation: stop any playback and return to wake-word
    /// idle, clearing the session so a conversation-mode follow-up stops
    /// listening. Lets a client "stop listening" without waiting out the silence
    /// timeout.
    async fn stop_listening(&self) -> fdo::Result<()> {
        self.stop_tx
            .send(StopRequest::Conversation)
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to stop listening: {e}")))?;
        Ok(())
    }

    /// Re-read `~/.config/adele-voice/config.toml` and apply any changed
    /// tunables to the running pipeline without a service restart (config#52).
    /// Hot-applies vad.speech_threshold, vad.silence_duration_ms,
    /// assistant.followup_timeout_ms, assistant.conversation_mode,
    /// idle_exit_timeout_ms, and wake_word.sensitivity (the last applied live to
    /// the running detector, no rebuild). An audio-device change is logged as
    /// needing a restart (it can't be hot-swapped). The KCM calls this after
    /// writing the config so live tuning is instant; a file watcher also picks
    /// up edits made any other way.
    async fn reload(&self) -> fdo::Result<()> {
        self.reload_tx
            .send(())
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to trigger reload: {e}")))?;
        tracing::info!("config reload requested over D-Bus");
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

    /// Calibrate the wake-word sensitivity to this user/mic (#121). The daemon
    /// briefly takes over the microphone and asks the user to say "Hey Adele"
    /// `utterances` times (0 selects a sensible default; the count is clamped to
    /// a reasonable range), measuring each one's peak match score. It then sets
    /// the sensitivity a margin below the worst score, applies it live, and
    /// persists it to the config file. Progress is reported via
    /// `CalibrationProgress` signals while this call is in flight.
    ///
    /// Returns `(sensitivity, eager, samples, mean_peak, noise_floor,
    /// eager_cutoff, non_eager_cutoff)`: the applied cutoff and wake mode, the
    /// measurements behind them, and the per-mode candidate cutoffs (with
    /// `non_eager_cutoff` negative when standard mode isn't reliable). Calibration
    /// switches to whichever mode is best for the measured scores. Fails if voice
    /// is busy (not idle) or no clear wake word could be heard.
    async fn calibrate_wake(
        &self,
        utterances: u32,
    ) -> fdo::Result<(f64, bool, u32, f64, f64, f64, f64)> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.calibrate_tx
            .send(CalibrationRequest {
                utterances,
                reply: reply_tx,
            })
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to start calibration: {e}")))?;
        let outcome = reply_rx
            .await
            .map_err(|e| fdo::Error::Failed(format!("calibration was dropped: {e}")))?
            .map_err(fdo::Error::Failed)?;
        Ok((
            outcome.sensitivity as f64,
            outcome.eager,
            outcome.samples,
            outcome.mean_peak as f64,
            outcome.noise_floor as f64,
            outcome.eager_cutoff as f64,
            outcome.non_eager_cutoff as f64,
        ))
    }

    /// Signal emitted when the pipeline state changes.
    #[zbus(signal)]
    pub async fn state_changed(emitter: &SignalEmitter<'_>, state: &str) -> zbus::Result<()>;

    /// Wake-word calibration progress (#121): `captured` of `total` utterances
    /// recorded, and the peak `score` of the most recent. A negative `score` is
    /// a prompt, not a measurement (`-1.0` = "say the next one now", `-2.0` =
    /// "didn't hear it, try again", `-3.0` = "measuring background — stay quiet").
    #[zbus(signal)]
    pub async fn calibration_progress(
        emitter: &SignalEmitter<'_>,
        captured: u32,
        total: u32,
        score: f64,
    ) -> zbus::Result<()>;

    /// Signal emitted when a transcript is ready.
    #[zbus(signal)]
    pub async fn transcript_ready(emitter: &SignalEmitter<'_>, text: &str) -> zbus::Result<()>;

    /// Signal emitted when Adele starts speaking a sentence.
    #[zbus(signal)]
    pub async fn speaking_text(emitter: &SignalEmitter<'_>, text: &str) -> zbus::Result<()>;
}

/// Drive the `org.desktopAssistant.Voice` signals (voice#85). The signals are
/// declared on the interface but were never emitted, so a client (the KDE
/// widget) had to poll `GetState`. This forwarder watches the pipeline's
/// existing `State` watch channel and a per-turn text-event channel, emitting
/// `StateChanged` / `TranscriptReady` / `SpeakingText` at the right moments.
///
/// Runs until both sources end (the pipeline is gone). `emitter` is bound to the
/// interface's object path on the daemon's session connection.
pub async fn run_signal_forwarder(
    emitter: SignalEmitter<'static>,
    mut state_rx: watch::Receiver<State>,
    mut signal_rx: mpsc::Receiver<VoiceSignal>,
) {
    // Emit the initial state so a client that connects late learns the current
    // state without a separate GetState round-trip. Copy the value out and drop
    // the borrow BEFORE awaiting (the watch guard isn't Send).
    let initial = *state_rx.borrow();
    emit_state(&emitter, initial).await;

    loop {
        tokio::select! {
            // State transitions: the watch coalesces, so we emit the latest
            // value after each change (intermediate values may be skipped — a
            // signal stream of states, not a guaranteed transition log).
            changed = state_rx.changed() => {
                if changed.is_err() {
                    // Sender dropped — the pipeline is gone. Stop once the other
                    // source is also done.
                    if signal_rx.is_closed() { break; }
                    continue;
                }
                let state = *state_rx.borrow_and_update();
                emit_state(&emitter, state).await;
            }
            event = signal_rx.recv() => {
                match event {
                    Some(VoiceSignal::TranscriptReady(text)) => {
                        if let Err(e) =
                            DbusVoiceAdapter::transcript_ready(&emitter, &text).await
                        {
                            tracing::debug!(error = %e, "failed to emit TranscriptReady");
                        }
                    }
                    Some(VoiceSignal::SpeakingText(text)) => {
                        if let Err(e) = DbusVoiceAdapter::speaking_text(&emitter, &text).await {
                            tracing::debug!(error = %e, "failed to emit SpeakingText");
                        }
                    }
                    Some(VoiceSignal::CalibrationProgress {
                        captured,
                        total,
                        score,
                    }) => {
                        if let Err(e) = DbusVoiceAdapter::calibration_progress(
                            &emitter, captured, total, score,
                        )
                        .await
                        {
                            tracing::debug!(error = %e, "failed to emit CalibrationProgress");
                        }
                    }
                    None => {
                        // Pipeline dropped the signal sender. Stop once state is
                        // also done; otherwise keep mirroring state.
                        if state_rx.has_changed().is_err() { break; }
                    }
                }
            }
        }
    }
}

async fn emit_state(emitter: &SignalEmitter<'static>, state: State) {
    if let Err(e) = DbusVoiceAdapter::state_changed(emitter, &state.to_string()).await {
        tracing::debug!(error = %e, "failed to emit StateChanged");
    }
}
