use std::sync::Arc;
use std::time::{Duration, Instant};

use adele_voice_core::domain::{State, StateEvent};
use adele_voice_core::ports::assistant::{AssistantEvent, AssistantGateway};
use adele_voice_core::ports::audio::{AudioSink, AudioSource};
use adele_voice_core::ports::stt::SpeechToText;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_core::ports::vad::VoiceActivityDetector;
use adele_voice_core::ports::wake::WakeWordDetector;
use adele_voice_core::sentence_buffer::SentenceBuffer;
use adele_voice_dbus_interface::StopRequest;
use tokio::sync::{mpsc, watch};

/// Compose the prompt sent to the assistant: when a spoken-response hint is
/// configured, prepend it so the reply stays short and conversational.
fn compose_prompt(hint: &str, transcript: &str) -> String {
    if hint.trim().is_empty() {
        transcript.to_string()
    } else {
        format!("{hint}\n\n{transcript}")
    }
}

/// Spoken when the assistant turn fails — short and human, never the raw error.
const ERROR_APOLOGY: &str = "Sorry, I ran into an error and couldn't answer that.";

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
    stt: Arc<S>,
    tts: Arc<T>,
    assistant: Arc<A>,
    source: Arc<dyn AudioSource>,
    sink: Arc<dyn AudioSink>,
    state_tx: watch::Sender<State>,
    enabled_rx: watch::Receiver<bool>,
    ptt_rx: mpsc::Receiver<Option<String>>,
    stop_rx: mpsc::Receiver<StopRequest>,
    conversation_id: Option<String>,
    /// When a push-to-talk specified a target conversation, its orchestrator
    /// id. Set on `PushToTalkInConversation`, used by `process_utterance` to
    /// route the turn (and any conversation-mode follow-ups) to that
    /// conversation instead of the daemon's own session; cleared when the
    /// conversation ends. `None` means "use the daemon's own session".
    ptt_conversation_override: Option<String>,
    conversation_title: String,
    silence_duration: Duration,
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
        conversation_title: String,
        silence_duration: Duration,
        speech_threshold: f32,
        conversation_mode: bool,
        followup_timeout: Duration,
        idle_exit_timeout: Option<Duration>,
        spoken_response_hint: String,
    ) -> Self {
        Self {
            wake,
            vad,
            stt: Arc::new(stt),
            tts: Arc::new(tts),
            assistant: Arc::new(assistant),
            source,
            sink,
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            conversation_id: None,
            ptt_conversation_override: None,
            conversation_title,
            silence_duration,
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

    pub async fn run(mut self) -> anyhow::Result<()> {
        let mut audio_rx = self.source.start()?;

        let mut state = State::Idle;
        self.set_state(state);

        let mut speech_buffer: Vec<f32> = Vec::new();
        let mut last_speech_at: Option<Instant> = None;
        // In conversation mode, set after a reply while awaiting a follow-up
        // turn; elapsing it with no speech ends the conversation.
        let mut followup_deadline: Option<Instant> = None;
        // For idle-exit (#5): time of the last activity other than idle-while-
        // wake-disabled.
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
                            self.sink.stop()?;
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
                        speech_buffer.clear();
                        // Wait (lead-in) for speech to start rather than cutting
                        // on the silence timer from the moment of the press; only
                        // cut after speech-then-silence, or if the lead-in elapses.
                        last_speech_at = None;
                        followup_deadline = Some(Instant::now() + self.followup_timeout);
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
                                self.sink.stop()?;
                                state = State::Idle;
                                self.set_state(state);
                            }
                        }
                        StopRequest::Conversation => {
                            // "Stop listening": end the session now without
                            // waiting out the silence timeout.
                            if state != State::Idle {
                                let _ = self.sink.stop();
                                state = State::Idle;
                                self.set_state(state);
                            }
                            self.conversation_id = None;
                            self.ptt_conversation_override = None;
                            speech_buffer.clear();
                            last_speech_at = None;
                            followup_deadline = None;
                        }
                    }
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
                            && !self.sink.is_playing()
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
                                    speech_buffer.clear();
                                    last_speech_at = None;
                                    followup_deadline = Some(Instant::now() + self.followup_timeout);
                                    self.vad.reset();
                                }
                            }
                        }

                        State::Listening => {
                            // Feed to VAD and accumulate speech
                            let prob = self.vad.speech_probability(&chunk)?;
                            speech_buffer.extend_from_slice(&chunk);

                            if prob >= self.speech_threshold {
                                if last_speech_at.is_none() {
                                    tracing::info!(prob, "speech detected, recording");
                                }
                                last_speech_at = Some(Instant::now());
                                followup_deadline = None; // speech began; cancel follow-up timeout
                            } else if let Some(last) = last_speech_at
                                && last.elapsed() >= self.silence_duration
                                && speech_buffer.len() > 800
                            {
                                // Silence detected after speech
                                tracing::info!(
                                    samples = speech_buffer.len(),
                                    "silence detected, transitioning to processing"
                                );
                                if let Some(new_state) = state.transition(&StateEvent::SilenceDetected) {
                                    state = new_state;
                                    self.set_state(state);

                                    // Spawn transcription + response pipeline
                                    let samples = std::mem::take(&mut speech_buffer);
                                    let outcome = self.process_utterance(samples).await?;

                                    if outcome == UtteranceOutcome::EndConversation {
                                        // A voice "stop" command ends the
                                        // conversation regardless of mode.
                                        state = State::Idle;
                                        self.set_state(state);
                                        self.conversation_id = None;
                                        self.ptt_conversation_override = None;
                                        speech_buffer.clear();
                                        last_speech_at = None;
                                        followup_deadline = None;
                                    } else if self.conversation_mode {
                                        // Re-open the mic for a follow-up turn:
                                        // wait for the reply to finish playing,
                                        // then drop any audio captured during
                                        // playback (echo) before listening again.
                                        while self.sink.is_playing() {
                                            tokio::time::sleep(Duration::from_millis(50)).await;
                                        }
                                        while audio_rx.try_recv().is_ok() {}
                                        state = State::Listening;
                                        self.set_state(state);
                                        speech_buffer.clear();
                                        last_speech_at = None;
                                        followup_deadline =
                                            Some(Instant::now() + self.followup_timeout);
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
                            } else if let Some(deadline) = followup_deadline
                                && Instant::now() >= deadline
                            {
                                // No follow-up speech within the timeout: end
                                // the conversation, return to wake-word idle.
                                tracing::info!("conversation follow-up timed out");
                                state = State::Idle;
                                self.set_state(state);
                                self.conversation_id = None;
                                self.ptt_conversation_override = None;
                                speech_buffer.clear();
                                last_speech_at = None;
                                followup_deadline = None;
                            }
                        }

                        State::Speaking => {
                            // Check for barge-in
                            let prob = self.vad.speech_probability(&chunk)?;
                            if prob >= self.speech_threshold {
                                tracing::info!("barge-in detected");
                                self.sink.stop()?;
                                if let Some(new_state) = state.transition(&StateEvent::BargeIn) {
                                    state = new_state;
                                    self.set_state(state);
                                    speech_buffer.clear();
                                    speech_buffer.extend_from_slice(&chunk);
                                    last_speech_at = Some(Instant::now());
                                    self.vad.reset();
                                }
                            } else if !self.sink.is_playing()
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
        // Discard near-silent captures before they reach STT. Ambient noise or
        // the tail of our own playback can trip the VAD without containing real
        // speech, and Whisper then hallucinates filler ("Thank you.") — which in
        // conversation mode loops every follow-up window. Real speech sits well
        // above this RMS floor (a buffer with even a brief utterance is ~0.02+,
        // noise/echo is ~0.003-0.008).
        const MIN_SPEECH_RMS: f32 = 0.01;
        let rms = if samples.is_empty() {
            0.0
        } else {
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
        };
        tracing::info!(rms, samples = samples.len(), "utterance captured");
        if rms < MIN_SPEECH_RMS {
            tracing::info!(rms, "discarding near-silent capture (no speech)");
            return Ok(UtteranceOutcome::Continue);
        }

        // Transcribe first, so a "stop" command ends the conversation without
        // creating or poking the orchestrator conversation.
        let transcript = self.stt.transcribe(&samples).await?;
        if transcript.text.is_empty() {
            tracing::debug!("empty transcript, skipping");
            return Ok(UtteranceOutcome::Continue);
        }
        tracing::info!(text = %transcript.text, "transcribed");

        // A whole-utterance stop phrase ("stop", "never mind", "that's all", …)
        // ends the conversation hands-free: acknowledge briefly and return to
        // wake-word idle instead of sending it to the assistant.
        if is_stop_phrase(&transcript.text) {
            tracing::info!(text = %transcript.text, "stop phrase — ending conversation");
            self.set_state(State::Speaking);
            self.speak_sentence("Okay.").await?;
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

        // Send prompt, prefixed with the spoken-response hint so the reply
        // stays short and conversational for read-aloud.
        let prompt = compose_prompt(&self.spoken_response_hint, &transcript.text);
        let request_id = self
            .assistant
            .send_prompt(&conversation_id, &prompt)
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
                                self.speak_sentence(ERROR_APOLOGY).await?;
                                break;
                            }
                            if first_chunk {
                                first_chunk = false;
                                self.set_state(State::Speaking);
                            }

                            let sentences = sentence_buf.push(&text);
                            for sentence in sentences {
                                self.speak_sentence(&sentence).await?;
                            }
                        }
                        Some(AssistantEvent::Complete { request_id: rid, full_response }) if rid == request_id => {
                            if sentence_buf.has_content() {
                                let remaining = sentence_buf.flush();
                                if !remaining.is_empty() {
                                    self.speak_sentence(&remaining).await?;
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
                                    self.speak_sentence(ERROR_APOLOGY).await?;
                                } else {
                                    // Speak the full response instead of dropping it.
                                    let sentences = sentence_buf.push(&full_response);
                                    for sentence in sentences {
                                        self.speak_sentence(&sentence).await?;
                                    }
                                    let remaining = sentence_buf.flush();
                                    if !remaining.is_empty() {
                                        self.speak_sentence(&remaining).await?;
                                    }
                                }
                            }
                            tracing::info!(streamed = !first_chunk, "assistant response complete");
                            break;
                        }
                        Some(AssistantEvent::Error { request_id: rid, error }) if rid == request_id => {
                            tracing::error!(error = %error, "assistant response error; speaking a short apology");
                            self.set_state(State::Speaking);
                            self.speak_sentence(ERROR_APOLOGY).await?;
                            break;
                        }
                        None => {
                            tracing::warn!("assistant signal stream closed before completion");
                            break;
                        }
                        _ => {} // Ignore events for other requests
                    }
                }
                // Check for timeout flush while waiting for chunks
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    if let Some(sentence) = sentence_buf.flush_if_timeout() {
                        self.speak_sentence(&sentence).await?;
                    }
                }
            }
        }

        Ok(UtteranceOutcome::Continue)
    }

    async fn speak_sentence(&self, text: &str) -> anyhow::Result<()> {
        tracing::info!(text = %text, "speaking");
        let samples = self.tts.synthesize(text).await?;
        if !samples.is_empty() {
            self.sink.play(samples)?;
        }
        Ok(())
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
    fn compose_prompt_prepends_hint() {
        assert_eq!(
            compose_prompt("Be brief.", "what's the weather?"),
            "Be brief.\n\nwhat's the weather?"
        );
    }

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

    #[test]
    fn compose_prompt_empty_hint_is_bare_transcript() {
        assert_eq!(compose_prompt("", "hello"), "hello");
        assert_eq!(compose_prompt("   ", "hello"), "hello");
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

    /// Assistant that completes immediately: `subscribe` hands back a receiver
    /// and `send_prompt` pushes a matching `Complete` so `process_utterance`
    /// returns without hanging. It records the conversation id each prompt was
    /// sent to (via `prompt_tx`) so tests can assert PTT routing, and reports
    /// the title of any conversation it created (via `created_tx`).
    struct FakeAssistant {
        tx: StdMutex<Option<mpsc::UnboundedSender<AssistantEvent>>>,
        prompt_tx: mpsc::UnboundedSender<String>,
        created_tx: mpsc::UnboundedSender<String>,
    }
    impl AssistantGateway for FakeAssistant {
        async fn create_conversation(
            &self,
            title: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            let _ = self.created_tx.send(title.to_string());
            Ok("own-session".to_string())
        }
        async fn send_prompt(
            &self,
            conversation_id: &str,
            _prompt: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            let _ = self.prompt_tx.send(conversation_id.to_string());
            let request_id = "req".to_string();
            if let Some(tx) = self.tx.lock().unwrap().as_ref() {
                let _ = tx.send(AssistantEvent::Complete {
                    request_id: request_id.clone(),
                    full_response: "hello".to_string(),
                });
            }
            Ok(request_id)
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

    struct FakeSink;
    impl AudioSink for FakeSink {
        fn play(&self, _samples: Vec<f32>) -> Result<(), adele_voice_core::VoiceError> {
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
        /// Conversation id each prompt was routed to.
        prompt_rx: mpsc::UnboundedReceiver<String>,
        /// Title of each conversation the daemon asked to create.
        created_rx: mpsc::UnboundedReceiver<String>,
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
            }
        }
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
            },
            Arc::new(FakeSource {
                rx: StdMutex::new(Some(audio_rx)),
            }),
            Arc::new(FakeSink),
            state_tx,
            enabled_rx,
            ptt_rx,
            stop_rx,
            "test".to_string(),
            Duration::from_millis(0),
            0.5,
            cfg.conversation_mode,
            cfg.followup_timeout,
            cfg.idle_exit_timeout,
            cfg.spoken_response_hint,
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
            routed, "chat-window-42",
            "the utterance must be routed to the conversation id the PTT supplied"
        );
        assert!(
            h.created_rx.try_recv().is_err(),
            "PTT-into-conversation must not create the daemon's own session"
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
            routed, "own-session",
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
        assert_eq!(first, "own-session");

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
            second, "own-session",
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
