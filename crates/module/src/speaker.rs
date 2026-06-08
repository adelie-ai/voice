//! Text-to-speech playback: synthesize with the configured backend, then queue
//! the audio on a sink. Independent of the microphone — speaking never listens.

use std::sync::Arc;
use std::time::Duration;

use adele_voice_core::ports::audio::AudioSink;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_core::{VoiceError, strip_markdown_for_speech};

/// Default ceiling on a single synth call (#58). Local Kokoro/Piper synth a
/// short sentence in well under a second and even a cloud Polly round-trip is a
/// second or two, so this only fires when the backend is genuinely wedged.
pub const DEFAULT_SYNTH_TIMEOUT: Duration = Duration::from_secs(20);

/// Synthesizes text and plays it through an [`AudioSink`].
///
/// Holds shared handles (the TTS backend and the sink), so multiple `Speaker`s
/// over the same sink queue onto one playback stream — that is how the daemon
/// lets spoken replies and on-demand `SayText` share a single output.
pub struct Speaker<T> {
    tts: Arc<T>,
    sink: Arc<dyn AudioSink>,
    /// Per-synth timeout; a synth that exceeds it errors so the caller can
    /// apologize and move on rather than hang (#58). 0 disables.
    synth_timeout: Duration,
}

// Manual `Clone` clones the two handles without requiring `T: Clone`.
impl<T> Clone for Speaker<T> {
    fn clone(&self) -> Self {
        Self {
            tts: Arc::clone(&self.tts),
            sink: Arc::clone(&self.sink),
            synth_timeout: self.synth_timeout,
        }
    }
}

impl<T: TextToSpeech> Speaker<T> {
    pub fn new(tts: Arc<T>, sink: Arc<dyn AudioSink>) -> Self {
        Self {
            tts,
            sink,
            synth_timeout: DEFAULT_SYNTH_TIMEOUT,
        }
    }

    /// Override the per-synth timeout (config knob, #58). `Duration::ZERO`
    /// disables the bound.
    pub fn set_synth_timeout(&mut self, timeout: Duration) {
        self.synth_timeout = timeout;
    }

    /// Synthesize `text` and queue it for playback. Empty synthesis (e.g. a
    /// backend that produced no audio) is a no-op rather than an error. The
    /// synth is bounded by [`set_synth_timeout`](Self::set_synth_timeout): a
    /// wedged backend errors instead of hanging the turn (#58).
    ///
    /// `text` is sanitized through [`strip_markdown_for_speech`] first, so any
    /// markdown the LLM emits (bold/headers/lists/code fences/links) is spoken
    /// as clean prose rather than read aloud as "asterisk asterisk", pound
    /// signs, or backticks (voice#63). This is the single chokepoint for *all*
    /// spoken text — streamed sentences, the leading ack, the `say_this` client
    /// tool, error apologies, and the D-Bus `SayText` method all flow through
    /// here. If sanitizing leaves nothing speakable (e.g. a "sentence" that was
    /// just `---` or a stray `**` from a SentenceBuffer split), synthesis is
    /// skipped entirely so the engine is never fed empty text.
    pub async fn say(&self, text: &str) -> Result<(), VoiceError> {
        let spoken = strip_markdown_for_speech(text);
        if spoken.is_empty() {
            tracing::debug!(original = %text, "nothing to speak after markdown strip; skipping synthesis");
            return Ok(());
        }
        tracing::info!(text = %spoken, "speaking");
        let synth = self.tts.synthesize(&spoken);
        let samples = if self.synth_timeout.is_zero() {
            synth.await?
        } else {
            match tokio::time::timeout(self.synth_timeout, synth).await {
                Ok(result) => result?,
                Err(_elapsed) => {
                    return Err(VoiceError::Tts(format!(
                        "synthesis timed out after {} ms",
                        self.synth_timeout.as_millis()
                    )));
                }
            }
        };
        if !samples.is_empty() {
            self.sink.play(samples)?;
        }
        Ok(())
    }

    /// Stop any ongoing playback and clear the queue.
    pub fn stop(&self) -> Result<(), VoiceError> {
        self.sink.stop()
    }

    /// Whether audio is currently playing.
    pub fn is_playing(&self) -> bool {
        self.sink.is_playing()
    }

    /// Whether playback is in its tail pad (audio done, latency cushion still
    /// running). See `AudioSink::in_tail_pad` (#70).
    pub fn in_tail_pad(&self) -> bool {
        self.sink.in_tail_pad()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// TTS that returns a fixed number of samples per call (0 = "no audio")
    /// and records the (sanitized) text each call received.
    #[derive(Default)]
    struct FakeTts {
        samples_per_call: usize,
        synthesized: StdMutex<Vec<String>>,
    }
    impl FakeTts {
        fn with_samples(n: usize) -> Self {
            Self {
                samples_per_call: n,
                synthesized: StdMutex::new(Vec::new()),
            }
        }
    }
    impl TextToSpeech for FakeTts {
        async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
            self.synthesized.lock().unwrap().push(text.to_string());
            Ok(vec![0.1; self.samples_per_call])
        }
    }

    /// Sink that records what it was asked to play.
    #[derive(Default)]
    struct FakeSink {
        played: StdMutex<Vec<usize>>,
        stopped: StdMutex<bool>,
    }
    impl AudioSink for FakeSink {
        fn play(&self, samples: Vec<f32>) -> Result<(), VoiceError> {
            self.played.lock().unwrap().push(samples.len());
            Ok(())
        }
        fn stop(&self) -> Result<(), VoiceError> {
            *self.stopped.lock().unwrap() = true;
            Ok(())
        }
        fn is_playing(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn say_synthesizes_and_plays() {
        let sink = Arc::new(FakeSink::default());
        let speaker = Speaker::new(Arc::new(FakeTts::with_samples(320)), sink.clone());
        speaker.say("hello").await.unwrap();
        assert_eq!(
            *sink.played.lock().unwrap(),
            vec![320],
            "synthesized audio must be queued on the sink"
        );
    }

    #[tokio::test]
    async fn empty_synthesis_does_not_play() {
        let sink = Arc::new(FakeSink::default());
        let speaker = Speaker::new(Arc::new(FakeTts::with_samples(0)), sink.clone());
        speaker.say("…").await.unwrap();
        assert!(
            sink.played.lock().unwrap().is_empty(),
            "empty synthesis must not queue a play"
        );
    }

    #[tokio::test]
    async fn stop_forwards_to_sink() {
        let sink = Arc::new(FakeSink::default());
        let speaker = Speaker::new(Arc::new(FakeTts::with_samples(0)), sink.clone());
        speaker.stop().unwrap();
        assert!(*sink.stopped.lock().unwrap(), "stop must reach the sink");
    }

    #[tokio::test]
    async fn say_strips_markdown_before_synthesis() {
        // voice#63: a streamed sentence with bold must reach the backend (and be
        // spoken) without asterisks.
        let tts = Arc::new(FakeTts::with_samples(64));
        let sink = Arc::new(FakeSink::default());
        let speaker = Speaker::new(tts.clone(), sink.clone());
        speaker.say("**Here** are the steps:").await.unwrap();
        assert_eq!(
            *tts.synthesized.lock().unwrap(),
            vec!["Here are the steps:".to_string()],
            "the backend must receive sanitized prose, not markdown"
        );
        assert_eq!(
            *sink.played.lock().unwrap(),
            vec![64],
            "the sanitized sentence is still spoken"
        );
    }

    #[tokio::test]
    async fn say_skips_synthesis_for_all_markdown_input() {
        // An all-markdown "sentence" (e.g. a horizontal rule the SentenceBuffer
        // emitted on its own) sanitizes to empty — never feed empty text to the
        // engine, and never queue a play.
        let tts = Arc::new(FakeTts::with_samples(64));
        let sink = Arc::new(FakeSink::default());
        let speaker = Speaker::new(tts.clone(), sink.clone());
        speaker.say("---").await.unwrap();
        speaker.say("**").await.unwrap();
        assert!(
            tts.synthesized.lock().unwrap().is_empty(),
            "no synth call for all-markdown input"
        );
        assert!(
            sink.played.lock().unwrap().is_empty(),
            "no playback for all-markdown input"
        );
    }
}
