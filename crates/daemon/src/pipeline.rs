use std::sync::Arc;
use std::time::{Duration, Instant};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::{SAMPLE_RATE, State, StateEvent};
use adele_voice_core::ports::assistant::{AssistantEvent, AssistantGateway};
use adele_voice_core::ports::audio::{AudioSink, AudioSource};
use adele_voice_core::ports::stt::SpeechToText;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_core::ports::vad::VoiceActivityDetector;
use adele_voice_core::ports::wake::WakeWordDetector;
use adele_voice_core::sentence_buffer::SentenceBuffer;
use adele_voice_dbus_interface::StopRequest;
use adele_voice_module::{Endpoint, Endpointer, PreBuffer, Speaker, Transcriber};
use tokio::sync::{mpsc, watch};

use crate::config::{self, Tunables, plan_reload};
use crate::cue::{self, ListeningCue};

/// Builds a fresh wake detector at a given sensitivity. rustpotter bakes the
/// detection threshold in at construction, so changing it on reload (config#52)
/// means rebuilding the detector rather than poking a setter.
pub type WakeBuilder<W> = Box<dyn Fn(f32) -> Result<W, VoiceError> + Send>;

/// Spoken when the assistant turn fails — short and human, never the raw error.
const ERROR_APOLOGY: &str = "Sorry, I ran into an error and couldn't answer that.";

/// Buffered-sample floor below which a trailing silence won't close an
/// utterance — guards against a single stray blip (50 ms at 16 kHz).
const ENDPOINT_MIN_SAMPLES: usize = 800;

/// Rolling pre-buffer length kept while idle so the wake→listen handoff can seed
/// the utterance with the audio captured right around the trigger — the start of
/// a command spoken in the same breath ("hey adele what time is it") that the
/// Idle→Listening transition would otherwise drop (#50). 300 ms at 16 kHz.
const WAKE_PREBUFFER_SAMPLES: usize = (SAMPLE_RATE as usize * 300) / 1000;

/// Heuristic: does this look like an orchestrator error surfaced as reply text?
/// The orchestrator reports LLM failures as the assistant message (so they show
/// in the chat UI), e.g. "Details: LLM error: Bedrock …". Reading that aloud is
/// terrible, so we substitute a short apology. "llm error" is specific enough
/// that a genuine spoken reply won't contain it.
fn is_error_response(text: &str) -> bool {
    text.to_ascii_lowercase().contains("llm error")
}

/// Outcome of handling one captured utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UtteranceOutcome {
    /// Normal turn — the run loop decides whether to keep listening (in
    /// conversation mode) or return to wake-word idle.
    Continue,
    /// The user spoke a stop phrase — end the conversation now, whatever the mode.
    EndConversation,
}

/// Whole-utterance "stop listening" phrases, matched only against the entire
/// normalized transcript (so "stop" inside a sentence isn't a command). Lets the
/// user end a conversation hands-free.
fn is_stop_phrase(text: &str) -> bool {
    let normalized = text
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, '.' | ',' | '!' | '?'))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    matches!(
        normalized.as_str(),
        "stop"
            | "stop listening"
            | "stop listening adele"
            | "stop adele"
            | "never mind"
            | "nevermind"
            | "that's all"
            | "thats all"
            | "that is all"
            | "that'll be all"
            | "goodbye"
            | "good bye"
            | "cancel"
            | "we're done"
            | "were done"
            | "i'm done"
            | "im done"
    )
}

pub struct Pipeline<W, V, S, T, A>
where
    W: WakeWordDetector + 'static,
    V: VoiceActivityDetector + 'static,
    S: SpeechToText + 'static,
    T: TextToSpeech + 'static,
    A: AssistantGateway + 'static,
{
    wake: W,
    vad: V,
    transcriber: Transcriber<S>,
    speaker: Speaker<T>,
    assistant: Arc<A>,
    source: Arc<dyn AudioSource>,
    /// Direct sink handle for the raw earcon (the `ding` cue is generated PCM,
    /// not TTS, so it bypasses the `Speaker`). Shares the same playback stream.
    sink: Arc<dyn AudioSink>,
    endpointer: Endpointer,
    /// Rolling window of recent idle audio, used to seed the utterance with the
    /// post-wake speech so a command spoken in the same breath isn't dropped (#50).
    prebuffer: PreBuffer,
    /// Audible "Listening" cue mode (ding / phrase / off) (#51).
    listening_cue: ListeningCue,
    /// Free-running counter so the spoken-phrase cue rotates deterministically.
    cue_phrase_counter: u64,
    state_tx: watch::Sender<State>,
    enabled_rx: watch::Receiver<bool>,
    ptt_rx: mpsc::Receiver<Option<String>>,
    stop_rx: mpsc::Receiver<StopRequest>,
    /// A ping (from the config-file watcher or the D-Bus `Reload` method) asking
    /// the pipeline to re-read the config and apply any changed tunables live
    /// (config#52).
    reload_rx: mpsc::Receiver<()>,
    /// Rebuilds the wake detector when `wake_word.sensitivity` changes.
    wake_builder: WakeBuilder<W>,
    /// Snapshot of the live-applicable knobs, diffed against a freshly loaded
    /// config on each reload to decide what to apply.
    tunables: Tunables,
    conversation_id: Option<String>,
    /// When a push-to-talk specified a target conversation, its orchestrator
    /// id. Set on `PushToTalkInConversation`, used by `process_utterance` to
    /// route the turn (and any conversation-mode follow-ups) to that
    /// conversation instead of the daemon's own session; cleared when the
    /// conversation ends. `None` means "use the daemon's own session".
    ptt_conversation_override: Option<String>,
    conversation_title: String,
    speech_threshold: f32,
    conversation_mode: bool,
    followup_timeout: Duration,
    idle_exit_timeout: Option<Duration>,
    spoken_response_hint: String,
}

impl<W, V, S, T, A> Pipeline<W, V, S, T, A>
where
    W: WakeWordDetector + 'static,
    V: VoiceActivityDetector + 'static,
    S: SpeechToText + 'static,
    T: TextToSpeech + 'static,
    A: AssistantGateway + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wake: W,
        vad: V,
        stt: S,
        tts: T,
        assistant: A,
        source: Arc<dyn AudioSource>,
        sink: Arc<dyn AudioSink>,
        state_tx: watch::Sender<State>,
        enabled_rx: watch::Receiver<bool>,
        ptt_rx: mpsc::Receiver<Option<String>>,
        stop_rx: mpsc::Receiver<StopRequest>,
        reload_rx: mpsc::Receiver<()>,
        wake_builder: WakeBuilder<W>,
        tunables: Tunables,
        conversation_title: String,
        silence_duration: Duration,
        speech_threshold: f32,
        conversation_mode: bool,
        followup_timeout: Duration,
        idle_exit_timeout: Option<Duration>,
        spoken_response_hint: String,
        listening_cue: ListeningCue,
    ) -> Self {
        Self {
            wake,
            vad,
            transcriber: Transcriber::new(Arc::new(stt)),
            speaker: Speaker::new(Arc::new(tts), Arc::clone(&sink)),
            assistant: Arc::new(assistant),
            source,
            sink,
            endpointer: Endpointer::new(speech_threshold, silence_duration, ENDPOINT_MIN_SAMPLES),
            prebuffer: PreBuffer::new(WAKE_PREBUFFER_SAMPLES),
            listening_cue,
            cue_phrase_counter: 0,
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            wake_builder,
            tunables,
            conversation_id: None,
            ptt_conversation_override: None,
            conversation_title,
            speech_threshold,
            conversation_mode,
            followup_timeout,
            idle_exit_timeout,
            spoken_response_hint,
        }
    }

    fn set_state(&self, state: State) {
        let _ = self.state_tx.send(state);
        tracing::info!(state = %state, "state changed");
    }

    /// Play the audible "Listening" cue (#51) on entering the Listening state.
    ///
    /// - `Ding`: a short generated earcon, queued straight onto the sink — no
    ///   TTS, so it's instant and reliable.
    /// - `Phrase`: a rotating spoken micro-phrase via the TTS `Speaker`;
    ///   friendlier but adds the synthesis/playback latency of a short
    ///   utterance, so it isn't the default.
    /// - `Off`: nothing.
    ///
    /// A cue failure must never derail entering Listening — errors are logged
    /// and swallowed.
    async fn play_listening_cue(&mut self) {
        match self.listening_cue {
            ListeningCue::Off => {}
            ListeningCue::Ding => {
                if let Err(e) = self.sink.play(cue::ding_samples()) {
                    tracing::warn!("failed to play listening ding cue: {e}");
                }
            }
            ListeningCue::Phrase => {
                let phrase = cue::phrase(self.cue_phrase_counter);
                self.cue_phrase_counter = self.cue_phrase_counter.wrapping_add(1);
                if let Err(e) = self.speaker.say(phrase).await {
                    tracing::warn!("failed to speak listening phrase cue: {e}");
                }
            }
        }
    }

    /// Re-read the config file and apply any changed tunables to the running
    /// pipeline (config#52). A failed/missing read is logged and ignored — a
    /// momentary write-in-progress must never tear the daemon down.
    fn reload(&mut self) {
        let new_config = match config::load() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("config reload: failed to re-read config, keeping current: {e}");
                return;
            }
        };
        let new_tunables = new_config.tunables();
        let plan = plan_reload(&self.tunables, &new_tunables);
        if plan.is_empty() {
            tracing::info!("config reload: no tunable changes");
            return;
        }
        self.apply_plan(&plan);
        // Adopt the new snapshot whole — including any field a sibling added,
        // and the device fields we only logged about — so we diff against the
        // on-disk truth next time rather than re-flagging the same change.
        self.tunables = new_tunables;
    }

    /// Apply a [`ReloadPlan`] to the live pipeline. Pure decisions live in
    /// [`plan_reload`]; this carries them out.
    fn apply_plan(&mut self, plan: &config::ReloadPlan) {
        if let Some(t) = plan.set_speech_threshold {
            self.speech_threshold = t;
            self.endpointer.set_speech_threshold(t);
            tracing::info!(
                speech_threshold = t,
                "config reload: applied vad.speech_threshold"
            );
        }
        if let Some(ms) = plan.set_silence_ms {
            self.endpointer.set_silence(Duration::from_millis(ms));
            tracing::info!(
                silence_duration_ms = ms,
                "config reload: applied vad.silence_duration_ms"
            );
        }
        if let Some(ms) = plan.set_followup_timeout_ms {
            self.followup_timeout = Duration::from_millis(ms);
            tracing::info!(
                followup_timeout_ms = ms,
                "config reload: applied assistant.followup_timeout_ms"
            );
        }
        if let Some(mode) = plan.set_conversation_mode {
            self.conversation_mode = mode;
            tracing::info!(
                conversation_mode = mode,
                "config reload: applied assistant.conversation_mode"
            );
        }
        if let Some(ms) = plan.set_idle_exit_timeout_ms {
            self.idle_exit_timeout = (ms > 0).then(|| Duration::from_millis(ms));
            tracing::info!(
                idle_exit_timeout_ms = ms,
                "config reload: applied idle_exit_timeout_ms"
            );
        }
        if let Some(sensitivity) = plan.rebuild_wake_sensitivity {
            match (self.wake_builder)(sensitivity) {
                Ok(wake) => {
                    self.wake = wake;
                    tracing::info!(
                        sensitivity,
                        "config reload: rebuilt wake detector for wake_word.sensitivity"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        sensitivity,
                        "config reload: failed to rebuild wake detector, keeping current: {e}"
                    );
                }
            }
        }
        if let Some(change) = &plan.restart_required_for_device {
            tracing::warn!(
                "config reload: audio device change ({change}) needs a daemon restart to take \
                 effect — `systemctl --user restart adele-voice`. All other knobs were applied live."
            );
        }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        let mut audio_rx = self.source.start()?;

        let mut state = State::Idle;
        self.set_state(state);

        // For idle-exit (#5): time of the last activity other than idle-while-
        // wake-disabled. (Utterance accumulation and the follow-up/lead-in
        // deadline now live in the shared `Endpointer`.)
        let mut last_activity = Instant::now();

        loop {
            tokio::select! {
                // Push-to-talk: skip wake word, go to Listening. The payload is
                // the target conversation: `None` uses the daemon's own
                // session, `Some(id)` routes the utterance to that orchestrator
                // conversation (the in-chat mic button).
                Some(target) = self.ptt_rx.recv() => {
                    if state == State::Idle || state == State::Speaking {
                        if state == State::Speaking {
                            self.speaker.stop()?;
                        }
                        // Route this PTT session: `Some(id)` dictates into that
                        // conversation; `None` (plain PushToTalk) falls back to
                        // the daemon's own session, which — like the wake word —
                        // persists across presses for continuity. (A stale
                        // override can't leak in: every press overwrites it, and
                        // the wake-word entry resets it to None.)
                        self.ptt_conversation_override = target.clone();
                        state = State::Listening;
                        self.set_state(state);
                        // Wait (lead-in) for speech to start rather than cutting
                        // on the silence timer from the moment of the press; only
                        // cut after speech-then-silence, or if the lead-in elapses.
                        self.endpointer.arm(Some(self.followup_timeout));
                        self.vad.reset();
                        tracing::info!(
                            target_conversation = target.as_deref().unwrap_or("<own session>"),
                            "push-to-talk activated, waiting for speech"
                        );
                    }
                }

                // Stop: cancel current playback (Speaking) or end the whole
                // conversation and return to wake-word idle.
                Some(req) = self.stop_rx.recv() => {
                    match req {
                        StopRequest::Speaking => {
                            if state == State::Speaking {
                                self.speaker.stop()?;
                                state = State::Idle;
                                self.set_state(state);
                            }
                        }
                        StopRequest::Conversation => {
                            // "Stop listening": end the session now without
                            // waiting out the silence timeout.
                            if state != State::Idle {
                                let _ = self.speaker.stop();
                                state = State::Idle;
                                self.set_state(state);
                            }
                            self.conversation_id = None;
                            self.ptt_conversation_override = None;
                            self.endpointer.reset();
                        }
                    }
                }

                // Reload: re-read the config file and apply any changed
                // tunables to the running pipeline (config#52). Triggered by the
                // file watcher (debounced) or the D-Bus `Reload` method.
                Some(()) = self.reload_rx.recv() => {
                    self.reload();
                }

                // Process audio chunks
                Some(chunk) = audio_rx.recv() => {
                    // `enabled` governs only always-on wake-word listening:
                    // push-to-talk (and SayText) must work even when "Hey
                    // Adele" is off, so the gate is scoped to the Idle state
                    // rather than the whole handler (#3).
                    if state == State::Idle && !*self.enabled_rx.borrow() {
                        // Idle-exit (#5): with wake listening off and nothing
                        // playing, exit after the configured idle window so
                        // D-Bus activation can restart the daemon on demand.
                        if let Some(timeout) = self.idle_exit_timeout
                            && last_activity.elapsed() >= timeout
                            && !self.speaker.is_playing()
                        {
                            tracing::info!(
                                "idle-exit: wake word disabled and idle, exiting for on-demand activation"
                            );
                            break;
                        }
                        continue;
                    }
                    last_activity = Instant::now();
                    match state {
                        State::Idle => {
                            // Keep a rolling pre-buffer of recent idle audio so a
                            // command spoken in the same breath as the wake word
                            // isn't dropped during the handoff (#50).
                            self.prebuffer.push(&chunk);
                            // Feed to wake word detector
                            if self.wake.detect(&chunk)? {
                                tracing::info!("wake word detected");
                                if let Some(new_state) = state.transition(&StateEvent::WakeWordDetected) {
                                    state = new_state;
                                    self.set_state(state);
                                    // Wake word always uses the daemon's own
                                    // session; clear any push-to-talk target
                                    // left over from a session ended via
                                    // StopSpeaking so this utterance can't leak
                                    // into a previously dictated conversation.
                                    self.ptt_conversation_override = None;
                                    // Seed the utterance with the post-wake audio
                                    // so "hey adele <command>" said in one breath
                                    // captures the command (#50). The lead-in still
                                    // applies and the VAD must still confirm speech.
                                    let preroll = self.prebuffer.take();
                                    self.endpointer
                                        .arm_with_preroll(Some(self.followup_timeout), &preroll);
                                    self.vad.reset();
                                    // Audible "Listening" cue (#51) — instant ding
                                    // by default, optional spoken phrase, or off.
                                    self.play_listening_cue().await;
                                }
                            }
                        }

                        State::Listening => {
                            // Feed to VAD; the endpointer accumulates and decides
                            // when the utterance ends (or the lead-in elapses).
                            let prob = self.vad.speech_probability(&chunk)?;
                            match self.endpointer.push(&chunk, prob) {
                                Endpoint::SpeechStarted => {
                                    tracing::info!(prob, "speech detected, recording");
                                }
                                Endpoint::Accumulating => {}
                                Endpoint::Complete(samples) => {
                                    tracing::info!(
                                        samples = samples.len(),
                                        "silence detected, transitioning to processing"
                                    );
                                    if let Some(new_state) =
                                        state.transition(&StateEvent::SilenceDetected)
                                    {
                                        state = new_state;
                                        self.set_state(state);

                                        // A failed turn must NOT crash the
                                        // daemon. The orchestrator may have
                                        // restarted and dropped the connection;
                                        // log it, apologize, and end the turn —
                                        // the gateway reconnects so the next
                                        // turn works.
                                        let outcome = match self.process_utterance(samples).await {
                                            Ok(outcome) => outcome,
                                            Err(e) => {
                                                tracing::error!("voice turn failed: {e}");
                                                self.set_state(State::Speaking);
                                                let _ = self.speaker.say(ERROR_APOLOGY).await;
                                                UtteranceOutcome::EndConversation
                                            }
                                        };

                                        if outcome == UtteranceOutcome::EndConversation {
                                            // A voice "stop" command ends the
                                            // conversation regardless of mode.
                                            state = State::Idle;
                                            self.set_state(state);
                                            self.conversation_id = None;
                                            self.ptt_conversation_override = None;
                                            self.endpointer.reset();
                                        } else if self.conversation_mode {
                                            // Re-open the mic for a follow-up turn:
                                            // wait for the reply to finish playing,
                                            // then drop any audio captured during
                                            // playback (echo) before listening again.
                                            while self.speaker.is_playing() {
                                                tokio::time::sleep(Duration::from_millis(50)).await;
                                            }
                                            while audio_rx.try_recv().is_ok() {}
                                            state = State::Listening;
                                            self.set_state(state);
                                            // Cue the follow-up re-listen too (#51),
                                            // then wait for the cue to finish and
                                            // drop the echo it queued into the mic
                                            // before arming, so it isn't captured as
                                            // the follow-up.
                                            self.play_listening_cue().await;
                                            while self.speaker.is_playing() {
                                                tokio::time::sleep(Duration::from_millis(50)).await;
                                            }
                                            while audio_rx.try_recv().is_ok() {}
                                            self.endpointer.arm(Some(self.followup_timeout));
                                            self.vad.reset();
                                        } else {
                                            // Single-shot: back to wake-word idle.
                                            // Drop any PTT-into-conversation target
                                            // so the next own-session turn doesn't
                                            // inherit it.
                                            state = State::Idle;
                                            self.set_state(state);
                                            self.ptt_conversation_override = None;
                                        }
                                    }
                                }
                                Endpoint::Timeout => {
                                    // No follow-up speech within the timeout: end
                                    // the conversation, return to wake-word idle.
                                    tracing::info!("conversation follow-up timed out");
                                    state = State::Idle;
                                    self.set_state(state);
                                    self.conversation_id = None;
                                    self.ptt_conversation_override = None;
                                    self.endpointer.reset();
                                }
                            }
                        }

                        State::Speaking => {
                            // Check for barge-in
                            let prob = self.vad.speech_probability(&chunk)?;
                            if prob >= self.speech_threshold {
                                tracing::info!("barge-in detected");
                                self.speaker.stop()?;
                                if let Some(new_state) = state.transition(&StateEvent::BargeIn) {
                                    state = new_state;
                                    self.set_state(state);
                                    // Seed the endpointer mid-speech so the next
                                    // silence closes this barge-in utterance.
                                    self.endpointer.arm_speaking(&chunk);
                                    self.vad.reset();
                                }
                            } else if !self.speaker.is_playing()
                                && let Some(new_state) =
                                    state.transition(&StateEvent::PlaybackComplete)
                            {
                                // Playback finished naturally
                                state = new_state;
                                self.set_state(state);
                            }
                        }

                        State::Processing => {
                            // Ignore audio while processing
                        }
                    }
                }

                else => break,
            }
        }

        self.source.stop()?;
        Ok(())
    }

    async fn process_utterance(&mut self, samples: Vec<f32>) -> anyhow::Result<UtteranceOutcome> {
        // Energy-gate + transcribe (in the module's `Transcriber`). The gate
        // discards near-silent captures — ambient noise or the tail of our own
        // playback can trip the VAD without real speech, and Whisper then
        // hallucinates filler ("Thank you.") that loops every follow-up window —
        // and an empty transcript is likewise nothing to act on; both yield
        // `None`. We transcribe before touching the orchestrator so a "stop"
        // command needn't create or poke the conversation.
        let transcript = match self.transcriber.transcribe(&samples).await? {
            Some(t) => t,
            None => return Ok(UtteranceOutcome::Continue),
        };
        tracing::info!(text = %transcript.text, "transcribed");

        // A whole-utterance stop phrase ("stop", "never mind", "that's all", …)
        // ends the conversation hands-free: acknowledge briefly and return to
        // wake-word idle instead of sending it to the assistant.
        if is_stop_phrase(&transcript.text) {
            tracing::info!(text = %transcript.text, "stop phrase — ending conversation");
            self.set_state(State::Speaking);
            self.speaker.say("Okay.").await?;
            return Ok(UtteranceOutcome::EndConversation);
        }

        // Resolve the target conversation. A push-to-talk into a specific
        // conversation (the in-chat mic button) routes this turn — and any
        // conversation-mode follow-ups — to that existing orchestrator
        // conversation; we never create it (the client owns its lifecycle).
        // Otherwise fall back to the daemon's own session, creating it lazily
        // and reusing it across turns.
        let conversation_id = if let Some(target) = self.ptt_conversation_override.clone() {
            target
        } else {
            if self.conversation_id.is_none() {
                let id = self
                    .assistant
                    .create_conversation(&self.conversation_title)
                    .await?;
                tracing::info!(conversation_id = %id, "created voice conversation");
                self.conversation_id = Some(id);
            }
            self.conversation_id.as_ref().unwrap().clone()
        };

        // Subscribe to response signals
        let mut signal_rx = self.assistant.subscribe().await?;

        // Send the CLEAN transcript as the user message and pass the
        // spoken-response hint as a per-request system_refinement, so the reply
        // stays short and conversational for read-aloud WITHOUT the blurb
        // polluting the visible chat transcript. The gateway falls back to
        // prepending the hint (the pre-#200 behaviour) when the orchestrator
        // lacks the refinement-aware method.
        let request_id = self
            .assistant
            .send_prompt_with_system_refinement(
                &conversation_id,
                &transcript.text,
                &self.spoken_response_hint,
            )
            .await?;

        // Stream response through sentence buffer → TTS → speaker
        let mut sentence_buf = SentenceBuffer::new(Duration::from_millis(500));
        let mut first_chunk = true;

        loop {
            tokio::select! {
                event = signal_rx.recv() => {
                    match event {
                        Some(AssistantEvent::Chunk { request_id: rid, text }) if rid == request_id => {
                            if first_chunk && is_error_response(&text) {
                                tracing::error!(chunk = %text, "assistant streamed an error message; speaking a short apology instead");
                                self.set_state(State::Speaking);
                                self.speaker.say(ERROR_APOLOGY).await?;
                                break;
                            }
                            if first_chunk {
                                first_chunk = false;
                                self.set_state(State::Speaking);
                            }

                            let sentences = sentence_buf.push(&text);
                            for sentence in sentences {
                                self.speaker.say(&sentence).await?;
                            }
                        }
                        Some(AssistantEvent::Complete { request_id: rid, full_response }) if rid == request_id => {
                            if sentence_buf.has_content() {
                                let remaining = sentence_buf.flush();
                                if !remaining.is_empty() {
                                    self.speaker.say(&remaining).await?;
                                }
                            } else if first_chunk && !full_response.trim().is_empty() {
                                // Nothing was streamed as chunks — e.g. a
                                // tool-using reply delivered as one final block.
                                self.set_state(State::Speaking);
                                if is_error_response(&full_response) {
                                    // The orchestrator surfaces LLM failures as
                                    // the reply text (so they show in chat);
                                    // don't read the raw error aloud.
                                    tracing::error!(response = %full_response, "assistant returned an error message; speaking a short apology instead");
                                    self.speaker.say(ERROR_APOLOGY).await?;
                                } else {
                                    // Speak the full response instead of dropping it.
                                    let sentences = sentence_buf.push(&full_response);
                                    for sentence in sentences {
                                        self.speaker.say(&sentence).await?;
                                    }
                                    let remaining = sentence_buf.flush();
                                    if !remaining.is_empty() {
                                        self.speaker.say(&remaining).await?;
                                    }
                                }
                            }
                            tracing::info!(streamed = !first_chunk, "assistant response complete");
                            break;
                        }
                        Some(AssistantEvent::Error { request_id: rid, error }) if rid == request_id => {
                            tracing::error!(error = %error, "assistant response error; speaking a short apology");
                            self.set_state(State::Speaking);
                            self.speaker.say(ERROR_APOLOGY).await?;
                            break;
                        }
                        None => {
                            tracing::warn!("assistant signal stream closed before completion");
                            if first_chunk {
                                // The reply stream dropped before any content
                                // arrived (e.g. the orchestrator restarted
                                // mid-turn) — don't leave the user in silence.
                                self.set_state(State::Speaking);
                                self.speaker.say(ERROR_APOLOGY).await?;
                            }
                            break;
                        }
                        _ => {} // Ignore events for other requests
                    }
                }
                // Check for timeout flush while waiting for chunks
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    if let Some(sentence) = sentence_buf.flush_if_timeout() {
                        self.speaker.say(&sentence).await?;
                    }
                }
            }
        }

        Ok(UtteranceOutcome::Continue)
    }
}

#[cfg(test)]
mod tests {
    //! Pipeline tests with fake adapters. Focus: #3 — the `enabled` flag must
    //! gate ONLY always-on wake-word detection, so an explicit push-to-talk
    //! still captures and transcribes an utterance while "Hey Adele" is off.
    use super::*;
    use adele_voice_core::domain::Transcript;
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn detects_orchestrator_error_responses() {
        // An LLM failure surfaced as reply text must be recognized so the
        // daemon apologizes instead of reading the raw error aloud.
        assert!(is_error_response(
            "Details: LLM error: Bedrock converse_stream request failed: validation error"
        ));
        assert!(is_error_response("LLM error: provider unavailable"));
    }

    #[test]
    fn normal_replies_are_not_errors() {
        assert!(!is_error_response("It's sunny and about 72 degrees today."));
        assert!(!is_error_response(
            "The forecast calls for rain this afternoon, clearing by evening."
        ));
    }

    #[test]
    fn stop_phrases_match_whole_utterance_only() {
        assert!(is_stop_phrase("stop"));
        assert!(is_stop_phrase("Stop listening."));
        assert!(is_stop_phrase("never mind"));
        assert!(is_stop_phrase("That's all!"));
        assert!(is_stop_phrase("goodbye"));
        // Not a command when it's only part of a real request.
        assert!(!is_stop_phrase("stop the timer"));
        assert!(!is_stop_phrase("what should I never mind about?"));
        assert!(!is_stop_phrase("tell me a story"));
    }

    #[tokio::test]
    async fn stop_phrase_ends_conversation_without_prompting() {
        // A whole-utterance "stop" must end the conversation — even in
        // conversation mode — and must NOT be sent to the assistant.
        let mut h = spawn_pipeline(Cfg {
            stt_text: "stop".to_string(),
            conversation_mode: true,
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        // Conversation mode would normally re-listen; a stop phrase returns to Idle.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("stop phrase returns to Idle")
        .unwrap();
        h.handle.abort();
        assert!(
            h.prompt_rx.try_recv().is_err(),
            "a stop phrase must not be sent to the assistant"
        );
    }

    #[tokio::test]
    async fn stop_listening_ends_an_active_conversation() {
        // StopListening (StopRequest::Conversation) ends a live conversation-mode
        // follow-up immediately, returning to wake-word idle.
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: true,
            followup_timeout: Duration::from_secs(30),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();
        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> re-listen
        // Wait until the first turn actually reached the assistant, so the
        // conversation is genuinely active before we stop it.
        tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("first turn prompted")
            .expect("prompt sender open");

        h.stop_tx.send(StopRequest::Conversation).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("StopListening -> Idle")
        .unwrap();
        h.handle.abort();
    }

    struct FakeWake {
        detects: bool,
    }
    impl WakeWordDetector for FakeWake {
        fn detect(&mut self, _samples: &[f32]) -> Result<bool, adele_voice_core::VoiceError> {
            Ok(self.detects)
        }
    }

    /// VAD that returns scripted probabilities, then 0.0 once the script is
    /// exhausted — so one "speech" chunk followed by anything reads as
    /// speech-then-silence.
    struct FakeVad {
        probs: StdMutex<VecDeque<f32>>,
    }
    impl VoiceActivityDetector for FakeVad {
        fn speech_probability(
            &mut self,
            _samples: &[f32],
        ) -> Result<f32, adele_voice_core::VoiceError> {
            Ok(self.probs.lock().unwrap().pop_front().unwrap_or(0.0))
        }
        fn reset(&mut self) {}
    }

    /// STT that signals when it runs (proving audio reached transcription) and
    /// returns a non-empty transcript so the response cycle proceeds.
    struct FakeStt {
        hit: mpsc::UnboundedSender<()>,
        text: String,
    }
    impl SpeechToText for FakeStt {
        async fn transcribe(
            &self,
            _samples: &[f32],
        ) -> Result<Transcript, adele_voice_core::VoiceError> {
            let _ = self.hit.send(());
            Ok(Transcript {
                text: self.text.clone(),
            })
        }
    }

    struct FakeTts;
    impl TextToSpeech for FakeTts {
        async fn synthesize(&self, _text: &str) -> Result<Vec<f32>, adele_voice_core::VoiceError> {
            Ok(Vec::new())
        }
    }

    /// What the pipeline handed to the assistant gateway for one turn.
    /// Captures the target conversation plus the split between the
    /// user-visible `prompt` and the per-request `system_refinement`, so a
    /// test can assert the clean transcript is the message and the hint
    /// rides as the refinement.
    #[derive(Debug, Clone)]
    struct SentPrompt {
        conversation_id: String,
        prompt: String,
        system_refinement: String,
    }

    /// Assistant that completes immediately: `subscribe` hands back a receiver
    /// and each send method pushes a matching `Complete` so
    /// `process_utterance` returns without hanging. It records every send
    /// (via `prompt_tx`) so tests can assert PTT routing and the
    /// prompt/refinement split, and reports the title of any conversation it
    /// created (via `created_tx`).
    struct FakeAssistant {
        tx: StdMutex<Option<mpsc::UnboundedSender<AssistantEvent>>>,
        prompt_tx: mpsc::UnboundedSender<SentPrompt>,
        created_tx: mpsc::UnboundedSender<String>,
        /// When set, `create_conversation` errors — simulating a dropped
        /// orchestrator connection so the turn fails mid-flight.
        fail: bool,
    }
    impl FakeAssistant {
        /// Shared recording + immediate-completion path for both send
        /// methods. Records exactly what reached the gateway (target
        /// conversation, the user-visible `prompt`, and the per-request
        /// `system_refinement`) and pushes a matching `Complete`.
        fn record_and_complete(
            &self,
            conversation_id: &str,
            prompt: &str,
            system_refinement: &str,
        ) -> String {
            let _ = self.prompt_tx.send(SentPrompt {
                conversation_id: conversation_id.to_string(),
                prompt: prompt.to_string(),
                system_refinement: system_refinement.to_string(),
            });
            let request_id = "req".to_string();
            if let Some(tx) = self.tx.lock().unwrap().as_ref() {
                let _ = tx.send(AssistantEvent::Complete {
                    request_id: request_id.clone(),
                    full_response: "hello".to_string(),
                });
            }
            request_id
        }
    }
    impl AssistantGateway for FakeAssistant {
        async fn create_conversation(
            &self,
            title: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            if self.fail {
                return Err(adele_voice_core::VoiceError::Assistant(
                    "uds connection closed".to_string(),
                ));
            }
            let _ = self.created_tx.send(title.to_string());
            Ok("own-session".to_string())
        }
        async fn send_prompt(
            &self,
            conversation_id: &str,
            prompt: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            Ok(self.record_and_complete(conversation_id, prompt, ""))
        }
        async fn send_prompt_with_system_refinement(
            &self,
            conversation_id: &str,
            prompt: &str,
            system_refinement: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            Ok(self.record_and_complete(conversation_id, prompt, system_refinement))
        }
        async fn subscribe(
            &self,
        ) -> Result<mpsc::UnboundedReceiver<AssistantEvent>, adele_voice_core::VoiceError> {
            let (tx, rx) = mpsc::unbounded_channel();
            *self.tx.lock().unwrap() = Some(tx);
            Ok(rx)
        }
    }

    /// Audio source whose receiver is driven by the test via `audio_tx`.
    struct FakeSource {
        rx: StdMutex<Option<mpsc::Receiver<Vec<f32>>>>,
    }
    impl AudioSource for FakeSource {
        fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, adele_voice_core::VoiceError> {
            self.rx
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| adele_voice_core::VoiceError::Audio("already started".to_string()))
        }
        fn stop(&self) -> Result<(), adele_voice_core::VoiceError> {
            Ok(())
        }
    }

    /// Records the length of every buffer it was asked to play, so a test can
    /// assert the listening cue (the ding earcon) was/wasn't queued.
    #[derive(Default, Clone)]
    struct FakeSink {
        played: Arc<StdMutex<Vec<usize>>>,
    }
    impl AudioSink for FakeSink {
        fn play(&self, samples: Vec<f32>) -> Result<(), adele_voice_core::VoiceError> {
            self.played.lock().unwrap().push(samples.len());
            Ok(())
        }
        fn stop(&self) -> Result<(), adele_voice_core::VoiceError> {
            Ok(())
        }
        fn is_playing(&self) -> bool {
            false
        }
    }

    struct Harness {
        audio_tx: mpsc::Sender<Vec<f32>>,
        ptt_tx: mpsc::Sender<Option<String>>,
        _enabled_tx: watch::Sender<bool>,
        stop_tx: mpsc::Sender<StopRequest>,
        state_rx: watch::Receiver<State>,
        transcribe_rx: mpsc::UnboundedReceiver<()>,
        /// Every send the pipeline made (target conversation + the
        /// prompt/refinement split).
        prompt_rx: mpsc::UnboundedReceiver<SentPrompt>,
        /// Title of each conversation the daemon asked to create.
        created_rx: mpsc::UnboundedReceiver<String>,
        /// Lengths of every buffer queued on the sink — the listening cue (the
        /// ding earcon) shows up here.
        sink_played: Arc<StdMutex<Vec<usize>>>,
        handle: tokio::task::JoinHandle<()>,
    }

    struct Cfg {
        enabled: bool,
        wake_detects: bool,
        conversation_mode: bool,
        followup_timeout: Duration,
        idle_exit_timeout: Option<Duration>,
        spoken_response_hint: String,
        vad: Vec<f32>,
        stt_text: String,
        assistant_fails: bool,
        listening_cue: ListeningCue,
    }
    impl Default for Cfg {
        fn default() -> Self {
            Self {
                enabled: true,
                wake_detects: false,
                conversation_mode: false,
                followup_timeout: Duration::from_millis(50),
                idle_exit_timeout: None,
                spoken_response_hint: String::new(),
                vad: vec![0.9],
                stt_text: "hello".to_string(),
                assistant_fails: false,
                // Default the cue off in tests so most cases don't queue cue
                // audio onto the recording sink; cue-specific tests opt in.
                listening_cue: ListeningCue::Off,
            }
        }
    }

    /// A neutral tunables snapshot for the fake pipeline. Matches the fakes'
    /// constructor args (0.5 threshold, 0 ms silence) so an initial reload that
    /// re-reads those same values is a no-op.
    fn test_tunables() -> Tunables {
        Tunables {
            speech_threshold: 0.5,
            silence_duration_ms: 0,
            followup_timeout_ms: 50,
            conversation_mode: false,
            idle_exit_timeout_ms: 0,
            wake_sensitivity: 0.5,
            input_device: "default".into(),
            output_device: "default".into(),
        }
    }

    /// Build a non-running pipeline with fakes so `apply_plan` can be exercised
    /// directly (no audio, no file watch) — the apply side of the reload, while
    /// `plan_reload`'s decision logic is unit-tested in `config`.
    fn build_pipeline() -> Pipeline<FakeWake, FakeVad, FakeStt, FakeTts, FakeAssistant> {
        let (_audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(1);
        let (_enabled_tx, enabled_rx) = watch::channel(true);
        let (_ptt_tx, ptt_rx) = mpsc::channel(1);
        let (_stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, _state_rx) = watch::channel(State::Idle);
        let (hit_tx, _transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, _prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, _created_rx) = mpsc::unbounded_channel();
        let (_reload_tx, reload_rx) = mpsc::channel(4);
        let wake_builder: WakeBuilder<FakeWake> =
            Box::new(|_sensitivity| Ok(FakeWake { detects: false }));
        Pipeline::new(
            FakeWake { detects: false },
            FakeVad {
                probs: StdMutex::new(VecDeque::new()),
            },
            FakeStt {
                hit: hit_tx,
                text: "hello".to_string(),
            },
            FakeTts,
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                fail: false,
            },
            Arc::new(FakeSource {
                rx: StdMutex::new(Some(audio_rx)),
            }),
            Arc::new(FakeSink::default()),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            wake_builder,
            test_tunables(),
            "test".to_string(),
            Duration::from_millis(0),
            0.5,
            false,
            Duration::from_millis(50),
            None,
            String::new(),
            ListeningCue::Off,
        )
    }

    #[test]
    fn apply_plan_updates_live_tunable_state() {
        // The apply side of reload: a plan's hot knobs must mutate the running
        // pipeline's fields (and the shared endpointer threshold) in place.
        let mut p = build_pipeline();
        let plan = config::ReloadPlan {
            set_speech_threshold: Some(0.8),
            set_silence_ms: Some(1200),
            set_followup_timeout_ms: Some(9000),
            set_conversation_mode: Some(true),
            set_idle_exit_timeout_ms: Some(60_000),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert_eq!(p.speech_threshold, 0.8);
        assert_eq!(p.followup_timeout, Duration::from_millis(9000));
        assert!(p.conversation_mode);
        assert_eq!(p.idle_exit_timeout, Some(Duration::from_millis(60_000)));
    }

    #[test]
    fn apply_plan_idle_exit_zero_disables() {
        // idle_exit_timeout_ms = 0 means "always-on" → the Option clears to None.
        let mut p = build_pipeline();
        p.idle_exit_timeout = Some(Duration::from_millis(1000));
        let plan = config::ReloadPlan {
            set_idle_exit_timeout_ms: Some(0),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert_eq!(p.idle_exit_timeout, None);
    }

    #[test]
    fn apply_plan_rebuilds_wake_detector_on_sensitivity_change() {
        // The wake-sensitivity branch must invoke the builder; the builder here
        // flips `detects` to true so we can observe the swap took effect.
        let mut p = build_pipeline();
        // Replace the builder with one that yields a detector that always fires.
        p.wake_builder = Box::new(|_s| Ok(FakeWake { detects: true }));
        assert!(!p.wake.detect(&[0.0; 10]).unwrap());
        let plan = config::ReloadPlan {
            rebuild_wake_sensitivity: Some(0.9),
            ..config::ReloadPlan::default()
        };
        p.apply_plan(&plan);
        assert!(
            p.wake.detect(&[0.0; 10]).unwrap(),
            "the wake detector must be rebuilt by the builder on a sensitivity change"
        );
    }

    fn spawn_pipeline(cfg: Cfg) -> Harness {
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(64);
        let (enabled_tx, enabled_rx) = watch::channel(cfg.enabled);
        let (ptt_tx, ptt_rx) = mpsc::channel(1);
        let (stop_tx, stop_rx) = mpsc::channel(1);
        let (state_tx, state_rx) = watch::channel(State::Idle);
        let (hit_tx, transcribe_rx) = mpsc::unbounded_channel();
        let (prompt_tx, prompt_rx) = mpsc::unbounded_channel();
        let (created_tx, created_rx) = mpsc::unbounded_channel();
        let (_reload_tx, reload_rx) = mpsc::channel(4);

        let wake_detects = cfg.wake_detects;
        let wake_builder: WakeBuilder<FakeWake> = Box::new(move |_sensitivity| {
            Ok(FakeWake {
                detects: wake_detects,
            })
        });

        let sink = FakeSink::default();
        let sink_played = Arc::clone(&sink.played);

        let pipeline = Pipeline::new(
            FakeWake {
                detects: cfg.wake_detects,
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::from(cfg.vad)),
            },
            FakeStt {
                hit: hit_tx,
                text: cfg.stt_text,
            },
            FakeTts,
            FakeAssistant {
                tx: StdMutex::new(None),
                prompt_tx,
                created_tx,
                fail: cfg.assistant_fails,
            },
            Arc::new(FakeSource {
                rx: StdMutex::new(Some(audio_rx)),
            }),
            Arc::new(sink),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            reload_rx,
            wake_builder,
            test_tunables(),
            "test".to_string(),
            Duration::from_millis(0),
            0.5,
            cfg.conversation_mode,
            cfg.followup_timeout,
            cfg.idle_exit_timeout,
            cfg.spoken_response_hint,
            cfg.listening_cue,
        );
        let handle = tokio::spawn(async move {
            let _ = pipeline.run().await;
        });
        Harness {
            audio_tx,
            ptt_tx,
            _enabled_tx: enabled_tx,
            stop_tx,
            state_rx,
            transcribe_rx,
            prompt_rx,
            created_rx,
            sink_played,
            handle,
        }
    }

    /// Each chunk is 1000 samples (> the 800-sample floor for closing an
    /// utterance) at a non-silent amplitude so the captured buffer clears the
    /// `process_utterance` energy gate. With a zero silence-duration, one speech
    /// chunk (VAD 0.9) then one silence chunk (VAD 0.0) closes the utterance.
    async fn send_chunk(h: &Harness) {
        h.audio_tx.send(vec![0.1f32; 1000]).await.unwrap();
    }

    #[tokio::test]
    async fn failed_turn_does_not_crash_the_daemon() {
        // A dropped orchestrator connection (create_conversation errors) must
        // not crash the pipeline: it apologizes, returns to Idle, and keeps
        // running so the next turn — after the gateway reconnects — works.
        let mut h = spawn_pipeline(Cfg {
            assistant_fails: true,
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process -> create_conversation fails

        // The failed turn must recover to Idle rather than crashing.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("a failed turn must recover to Idle")
        .unwrap();
        assert!(
            !h.handle.is_finished(),
            "a failed turn must not crash the daemon"
        );
        h.handle.abort();
    }

    #[tokio::test]
    async fn push_to_talk_transcribes_even_when_wake_disabled() {
        // #3: wake word OFF, but an explicit push-to-talk must still capture
        // and transcribe. Pre-fix this times out — chunks were dropped by the
        // top-level enable gate before reaching the Listening state.
        let mut h = spawn_pipeline(Cfg {
            enabled: false,
            ..Default::default()
        });

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("push-to-talk should enter Listening even when disabled")
        .unwrap();

        send_chunk(&h).await; // VAD 0.9 -> speech
        send_chunk(&h).await; // VAD 0.0 -> silence -> transcription

        let got = tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv()).await;
        h.handle.abort();
        assert!(
            matches!(got, Ok(Some(()))),
            "transcription must run for a push-to-talk utterance while wake word is disabled"
        );
    }

    #[tokio::test]
    async fn ptt_with_conversation_id_routes_to_that_conversation() {
        // #24 (core acceptance): a push-to-talk carrying a conversation id
        // routes the utterance to THAT orchestrator conversation — it must
        // send the prompt to the supplied id and must NOT create the daemon's
        // own session.
        let mut h = spawn_pipeline(Cfg::default());

        h.ptt_tx
            .send(Some("chat-window-42".to_string()))
            .await
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        let routed = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("a prompt should be sent")
            .expect("prompt sender open");
        h.handle.abort();
        assert_eq!(
            routed.conversation_id, "chat-window-42",
            "the utterance must be routed to the conversation id the PTT supplied"
        );
        assert!(
            h.created_rx.try_recv().is_err(),
            "PTT-into-conversation must not create the daemon's own session"
        );
    }

    #[tokio::test]
    async fn turn_sends_clean_transcript_with_hint_as_system_refinement() {
        // Core of the #200 voice follow-up: the pipeline must send the CLEAN
        // transcript as the user-visible prompt and pass the configured
        // spoken-response hint as the per-request system_refinement — NOT
        // prepend the hint blurb into the message. That keeps the visible
        // chat transcript free of the "respond briefly, by voice" boilerplate.
        let mut h = spawn_pipeline(Cfg {
            spoken_response_hint: "Respond briefly, by voice.".to_string(),
            stt_text: "what's the weather?".to_string(),
            ..Default::default()
        });

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        let sent = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("a prompt should be sent")
            .expect("prompt sender open");
        h.handle.abort();

        // The user message is the clean transcript — no hint blurb folded in.
        assert_eq!(
            sent.prompt, "what's the weather?",
            "the user-visible message must be the clean transcript"
        );
        assert!(
            !sent.prompt.contains("Respond briefly"),
            "the spoken-response hint must NOT be prepended to the prompt"
        );
        // The hint rides as the per-request system_refinement.
        assert_eq!(
            sent.system_refinement, "Respond briefly, by voice.",
            "the configured hint must be passed as the per-request system_refinement"
        );
    }

    #[tokio::test]
    async fn plain_ptt_uses_daemon_own_session() {
        // #24 (back-compat): a plain PushToTalk() (no id) must keep creating
        // and using the daemon's own session ("test" title here) rather than
        // any caller conversation.
        let mut h = spawn_pipeline(Cfg::default());

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process

        let created = tokio::time::timeout(Duration::from_secs(2), h.created_rx.recv())
            .await
            .expect("the daemon's own session should be created")
            .expect("created sender open");
        assert_eq!(
            created, "test",
            "a plain PTT must create the daemon's own session by its configured title"
        );
        let routed = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("a prompt should be sent")
            .expect("prompt sender open");
        h.handle.abort();
        assert_eq!(
            routed.conversation_id, "own-session",
            "a plain PTT must route to the daemon's own session id, not a caller conversation"
        );
    }

    #[tokio::test]
    async fn plain_ptt_reuses_own_session_across_presses() {
        // Regression guard (#24): a plain PushToTalk() continues the daemon's
        // own session across presses — like the wake word — instead of spawning
        // a fresh "Voice Conversation" each press.
        // VAD script drives two utterances: speech/silence, then speech/(exhausted)silence.
        let mut h = spawn_pipeline(Cfg {
            vad: vec![0.9, 0.0, 0.9],
            ..Default::default()
        });

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("first ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        let first = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("first prompt")
            .expect("prompt sender open");
        assert_eq!(first.conversation_id, "own-session");

        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await
        .expect("back to Idle after the first turn")
        .unwrap();

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("second ptt -> Listening")
        .unwrap();
        send_chunk(&h).await;
        send_chunk(&h).await;
        let second = tokio::time::timeout(Duration::from_secs(2), h.prompt_rx.recv())
            .await
            .expect("second prompt")
            .expect("prompt sender open");
        h.handle.abort();

        assert_eq!(
            second.conversation_id, "own-session",
            "the second plain PTT must reuse the own session id"
        );
        let created = h
            .created_rx
            .try_recv()
            .expect("the own session must have been created");
        assert_eq!(created, "test");
        assert!(
            h.created_rx.try_recv().is_err(),
            "a second plain PTT must NOT create a new session — it reuses the own session"
        );
    }

    #[tokio::test]
    async fn near_silent_capture_is_discarded() {
        // The energy gate must drop a near-silent buffer (noise/echo that
        // tripped the VAD) before STT, so Whisper can't hallucinate filler
        // that would loop in conversation mode.
        let mut h = spawn_pipeline(Cfg::default());

        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("push-to-talk should enter Listening")
        .unwrap();

        // VAD scripts this as speech-then-silence, but the samples are ~silent.
        h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();
        h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();

        let got = tokio::time::timeout(Duration::from_millis(500), h.transcribe_rx.recv()).await;
        h.handle.abort();
        assert!(
            got.is_err(),
            "a near-silent capture must be discarded before transcription"
        );
    }

    #[tokio::test]
    async fn wake_word_ignored_when_disabled() {
        // Regression guard: an always-firing detector must not trigger
        // Listening while wake-word listening is disabled.
        let h = spawn_pipeline(Cfg {
            enabled: false,
            wake_detects: true,
            ..Default::default()
        });
        send_chunk(&h).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let state = *h.state_rx.borrow();
        h.handle.abort();
        assert_eq!(
            state,
            State::Idle,
            "wake word must be ignored while disabled"
        );
    }

    #[tokio::test]
    async fn wake_word_triggers_listening_when_enabled() {
        // Regression guard: with wake enabled, detection moves to Listening.
        let mut h = spawn_pipeline(Cfg {
            wake_detects: true,
            ..Default::default()
        });
        send_chunk(&h).await;
        let reached = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await;
        h.handle.abort();
        assert!(
            reached.is_ok(),
            "wake word must enter Listening when enabled"
        );
    }

    #[tokio::test]
    async fn ding_cue_plays_on_wake_word_entry() {
        // #51: with the ding cue, entering Listening queues the generated earcon
        // (the only buffer played here, since the FakeTts produces no audio).
        let mut h = spawn_pipeline(Cfg {
            wake_detects: true,
            listening_cue: ListeningCue::Ding,
            ..Default::default()
        });
        send_chunk(&h).await;
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("wake -> Listening")
        .unwrap();
        // Give the cue a beat to be queued after the state change.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let played = h.sink_played.lock().unwrap().clone();
        h.handle.abort();
        assert_eq!(
            played,
            vec![cue::ding_samples().len()],
            "the ding earcon must be queued on entering Listening"
        );
    }

    #[tokio::test]
    async fn no_cue_plays_when_listening_cue_off() {
        // #51: with the cue off, entering Listening must NOT queue any audio.
        let mut h = spawn_pipeline(Cfg {
            wake_detects: true,
            listening_cue: ListeningCue::Off,
            ..Default::default()
        });
        send_chunk(&h).await;
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("wake -> Listening")
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let played = h.sink_played.lock().unwrap().clone();
        h.handle.abort();
        assert!(played.is_empty(), "no cue must be queued when set to off");
    }

    #[tokio::test]
    async fn conversation_mode_relistens_after_response() {
        // #6: in conversation mode, after replying the pipeline re-opens the
        // mic for a follow-up turn instead of returning to wake-word idle.
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: true,
            followup_timeout: Duration::from_secs(5),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process turn 1
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("turn 1 should transcribe")
            .unwrap();

        // After the reply, conversation mode returns to Listening (not Idle).
        let relisten = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await;
        h.handle.abort();
        assert!(
            relisten.is_ok(),
            "conversation mode must re-open the mic for a follow-up turn"
        );
    }

    #[tokio::test]
    async fn conversation_mode_times_out_to_idle() {
        // #6: with no follow-up speech, the conversation ends after the
        // follow-up timeout and the pipeline returns to wake-word idle.
        let mut h = spawn_pipeline(Cfg {
            conversation_mode: true,
            followup_timeout: Duration::from_millis(100),
            ..Default::default()
        });
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process turn 1
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("turn 1 should transcribe")
            .unwrap();
        // Wait for the follow-up re-listen.
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("re-listen")
        .unwrap();

        // No follow-up speech: wait past the timeout, then one silence chunk
        // trips the deadline check.
        tokio::time::sleep(Duration::from_millis(160)).await;
        send_chunk(&h).await; // VAD script exhausted -> 0.0 (silence)

        let idle = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await;
        h.handle.abort();
        assert!(
            idle.is_ok(),
            "conversation must return to Idle after the follow-up timeout"
        );
    }

    #[tokio::test]
    async fn non_conversation_mode_returns_to_idle() {
        // Regression guard: without conversation mode, a reply returns to Idle.
        let mut h = spawn_pipeline(Cfg::default());
        h.ptt_tx.send(None).await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Listening),
        )
        .await
        .expect("ptt -> Listening")
        .unwrap();

        send_chunk(&h).await; // speech
        send_chunk(&h).await; // silence -> process
        tokio::time::timeout(Duration::from_secs(2), h.transcribe_rx.recv())
            .await
            .expect("should transcribe")
            .unwrap();

        let idle = tokio::time::timeout(
            Duration::from_secs(2),
            h.state_rx.wait_for(|s| *s == State::Idle),
        )
        .await;
        h.handle.abort();
        assert!(idle.is_ok(), "non-conversation mode must return to Idle");
    }

    #[tokio::test]
    async fn idle_exits_when_wake_disabled_and_idle() {
        // #5: with wake listening off and an idle-exit timeout configured, the
        // daemon exits after the idle window so D-Bus activation can restart it.
        let h = spawn_pipeline(Cfg {
            enabled: false,
            idle_exit_timeout: Some(Duration::from_millis(80)),
            ..Default::default()
        });
        // Stay idle past the window, then one chunk trips the idle-exit check.
        tokio::time::sleep(Duration::from_millis(120)).await;
        h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();
        let exited = tokio::time::timeout(Duration::from_secs(2), h.handle).await;
        assert!(
            exited.is_ok(),
            "daemon should idle-exit when wake disabled and idle past the timeout"
        );
    }

    #[tokio::test]
    async fn does_not_idle_exit_while_wake_enabled() {
        // Guard: wake listening on means always-on — never idle-exit.
        let h = spawn_pipeline(Cfg {
            enabled: true,
            idle_exit_timeout: Some(Duration::from_millis(40)),
            ..Default::default()
        });
        for _ in 0..5 {
            h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let exited = tokio::time::timeout(Duration::from_millis(100), h.handle).await;
        assert!(
            exited.is_err(),
            "must not idle-exit while wake word is enabled"
        );
    }
}
