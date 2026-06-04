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
    ptt_rx: mpsc::Receiver<()>,
    stop_rx: mpsc::Receiver<()>,
    conversation_id: Option<String>,
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
        ptt_rx: mpsc::Receiver<()>,
        stop_rx: mpsc::Receiver<()>,
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
                // Push-to-talk: skip wake word, go to Listening
                Some(()) = self.ptt_rx.recv() => {
                    if state == State::Idle || state == State::Speaking {
                        if state == State::Speaking {
                            self.sink.stop()?;
                        }
                        state = State::Listening;
                        self.set_state(state);
                        speech_buffer.clear();
                        last_speech_at = Some(Instant::now());
                        followup_deadline = None;
                        self.vad.reset();
                        tracing::info!("push-to-talk activated");
                    }
                }

                // Stop speaking
                Some(()) = self.stop_rx.recv() => {
                    if state == State::Speaking {
                        self.sink.stop()?;
                        state = State::Idle;
                        self.set_state(state);
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
                                    speech_buffer.clear();
                                    last_speech_at = Some(Instant::now());
                                    followup_deadline = None;
                                    self.vad.reset();
                                }
                            }
                        }

                        State::Listening => {
                            // Feed to VAD and accumulate speech
                            let prob = self.vad.speech_probability(&chunk)?;
                            speech_buffer.extend_from_slice(&chunk);

                            if prob >= self.speech_threshold {
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
                                    self.process_utterance(samples).await?;

                                    if self.conversation_mode {
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
                                        state = State::Idle;
                                        self.set_state(state);
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

    async fn process_utterance(&mut self, samples: Vec<f32>) -> anyhow::Result<()> {
        // Ensure we have a conversation
        if self.conversation_id.is_none() {
            let id = self
                .assistant
                .create_conversation(&self.conversation_title)
                .await?;
            tracing::info!(conversation_id = %id, "created voice conversation");
            self.conversation_id = Some(id);
        }
        let conversation_id = self.conversation_id.as_ref().unwrap().clone();

        // Transcribe
        let transcript = self.stt.transcribe(&samples).await?;
        if transcript.text.is_empty() {
            tracing::debug!("empty transcript, skipping");
            return Ok(());
        }
        tracing::info!(text = %transcript.text, "transcribed");

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
                            if first_chunk {
                                first_chunk = false;
                                self.set_state(State::Speaking);
                            }

                            let sentences = sentence_buf.push(&text);
                            for sentence in sentences {
                                self.speak_sentence(&sentence).await?;
                            }
                        }
                        Some(AssistantEvent::Complete { request_id: rid, .. }) if rid == request_id => {
                            // Flush remaining text
                            if sentence_buf.has_content() {
                                let remaining = sentence_buf.flush();
                                if !remaining.is_empty() {
                                    self.speak_sentence(&remaining).await?;
                                }
                            }
                            break;
                        }
                        Some(AssistantEvent::Error { request_id: rid, error }) if rid == request_id => {
                            tracing::error!(error = %error, "assistant response error");
                            break;
                        }
                        None => break,
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

        Ok(())
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
    }
    impl SpeechToText for FakeStt {
        async fn transcribe(
            &self,
            _samples: &[f32],
        ) -> Result<Transcript, adele_voice_core::VoiceError> {
            let _ = self.hit.send(());
            Ok(Transcript {
                text: "hello".to_string(),
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
    /// returns without hanging.
    struct FakeAssistant {
        tx: StdMutex<Option<mpsc::UnboundedSender<AssistantEvent>>>,
    }
    impl AssistantGateway for FakeAssistant {
        async fn create_conversation(
            &self,
            _title: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
            Ok("conv".to_string())
        }
        async fn send_prompt(
            &self,
            _conversation_id: &str,
            _prompt: &str,
        ) -> Result<String, adele_voice_core::VoiceError> {
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
        ptt_tx: mpsc::Sender<()>,
        _enabled_tx: watch::Sender<bool>,
        _stop_tx: mpsc::Sender<()>,
        state_rx: watch::Receiver<State>,
        transcribe_rx: mpsc::UnboundedReceiver<()>,
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

        let pipeline = Pipeline::new(
            FakeWake {
                detects: cfg.wake_detects,
            },
            FakeVad {
                probs: StdMutex::new(VecDeque::from(cfg.vad)),
            },
            FakeStt { hit: hit_tx },
            FakeTts,
            FakeAssistant {
                tx: StdMutex::new(None),
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
            _stop_tx: stop_tx,
            state_rx,
            transcribe_rx,
            handle,
        }
    }

    /// Each chunk is 1000 samples (> the 800-sample floor for closing an
    /// utterance). With a zero silence-duration, one speech chunk (VAD 0.9)
    /// then one silence chunk (VAD 0.0) closes the utterance.
    async fn send_chunk(h: &Harness) {
        h.audio_tx.send(vec![0.0f32; 1000]).await.unwrap();
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

        h.ptt_tx.send(()).await.unwrap();
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
        h.ptt_tx.send(()).await.unwrap();
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
        h.ptt_tx.send(()).await.unwrap();
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
        h.ptt_tx.send(()).await.unwrap();
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
