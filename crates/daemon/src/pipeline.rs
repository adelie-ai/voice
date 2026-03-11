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
                    if !*self.enabled_rx.borrow() {
                        continue;
                    }

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
                            } else if let Some(last) = last_speech_at {
                                if last.elapsed() >= self.silence_duration && speech_buffer.len() > 800 {
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

                                        // After processing completes, return to idle
                                        state = State::Idle;
                                        self.set_state(state);
                                    }
                                }
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
                            } else if !self.sink.is_playing() {
                                // Playback finished naturally
                                if let Some(new_state) = state.transition(&StateEvent::PlaybackComplete) {
                                    state = new_state;
                                    self.set_state(state);
                                }
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

        // Send prompt
        let request_id = self
            .assistant
            .send_prompt(&conversation_id, &transcript.text)
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
