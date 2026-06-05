//! Text-to-speech playback: synthesize with the configured backend, then queue
//! the audio on a sink. Independent of the microphone — speaking never listens.

use std::sync::Arc;

use adele_voice_core::VoiceError;
use adele_voice_core::ports::audio::AudioSink;
use adele_voice_core::ports::tts::TextToSpeech;

/// Synthesizes text and plays it through an [`AudioSink`].
///
/// Holds shared handles (the TTS backend and the sink), so multiple `Speaker`s
/// over the same sink queue onto one playback stream — that is how the daemon
/// lets spoken replies and on-demand `SayText` share a single output.
pub struct Speaker<T> {
    tts: Arc<T>,
    sink: Arc<dyn AudioSink>,
}

// Manual `Clone` clones the two handles without requiring `T: Clone`.
impl<T> Clone for Speaker<T> {
    fn clone(&self) -> Self {
        Self {
            tts: Arc::clone(&self.tts),
            sink: Arc::clone(&self.sink),
        }
    }
}

impl<T: TextToSpeech> Speaker<T> {
    pub fn new(tts: Arc<T>, sink: Arc<dyn AudioSink>) -> Self {
        Self { tts, sink }
    }

    /// Synthesize `text` and queue it for playback. Empty synthesis (e.g. a
    /// backend that produced no audio) is a no-op rather than an error.
    pub async fn say(&self, text: &str) -> Result<(), VoiceError> {
        tracing::info!(text = %text, "speaking");
        let samples = self.tts.synthesize(text).await?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// TTS that returns a fixed number of samples per call (0 = "no audio").
    struct FakeTts {
        samples_per_call: usize,
    }
    impl TextToSpeech for FakeTts {
        async fn synthesize(&self, _text: &str) -> Result<Vec<f32>, VoiceError> {
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
        let speaker = Speaker::new(
            Arc::new(FakeTts {
                samples_per_call: 320,
            }),
            sink.clone(),
        );
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
        let speaker = Speaker::new(
            Arc::new(FakeTts {
                samples_per_call: 0,
            }),
            sink.clone(),
        );
        speaker.say("…").await.unwrap();
        assert!(
            sink.played.lock().unwrap().is_empty(),
            "empty synthesis must not queue a play"
        );
    }

    #[tokio::test]
    async fn stop_forwards_to_sink() {
        let sink = Arc::new(FakeSink::default());
        let speaker = Speaker::new(
            Arc::new(FakeTts {
                samples_per_call: 0,
            }),
            sink.clone(),
        );
        speaker.stop().unwrap();
        assert!(*sink.stopped.lock().unwrap(), "stop must reach the sink");
    }
}
