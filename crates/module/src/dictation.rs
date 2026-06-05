//! One-shot dictation: open the mic, endpoint a single utterance, transcribe.
//!
//! This is "push-to-talk minus everything else" — no wake word, no continuous
//! loop, no assistant. A client taps a mic button, [`Dictation::dictate`] waits
//! for the user to speak and stop, and returns the transcript (or `None`).

use std::sync::Arc;
use std::time::Duration;

use adele_voice_core::VoiceError;
use adele_voice_core::ports::audio::AudioSource;
use adele_voice_core::ports::stt::SpeechToText;
use adele_voice_core::ports::vad::VoiceActivityDetector;

use crate::endpointer::{Endpoint, Endpointer};
use crate::transcriber::Transcriber;

/// Tuning for [`Dictation`].
#[derive(Debug, Clone)]
pub struct DictationOptions {
    /// VAD probability at/above which a chunk counts as speech.
    pub speech_threshold: f32,
    /// Trailing silence that ends an utterance once speech has been heard.
    pub silence: Duration,
    /// Minimum buffered samples before a silence may close an utterance.
    pub min_samples: usize,
    /// Give up and return `None` if no speech starts within this lead-in.
    pub lead_in: Duration,
}

impl Default for DictationOptions {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence: Duration::from_millis(800),
            min_samples: 800,
            lead_in: Duration::from_secs(10),
        }
    }
}

/// Captures and transcribes one utterance on demand.
pub struct Dictation<V, S> {
    source: Arc<dyn AudioSource>,
    vad: V,
    transcriber: Transcriber<S>,
    endpointer: Endpointer,
    lead_in: Duration,
}

impl<V: VoiceActivityDetector, S: SpeechToText> Dictation<V, S> {
    pub fn new(source: Arc<dyn AudioSource>, vad: V, stt: S, opts: DictationOptions) -> Self {
        Self {
            source,
            vad,
            transcriber: Transcriber::new(Arc::new(stt)),
            endpointer: Endpointer::new(opts.speech_threshold, opts.silence, opts.min_samples),
            lead_in: opts.lead_in,
        }
    }

    /// Capture one utterance and transcribe it.
    ///
    /// Opens the mic, waits (up to the lead-in) for speech, endpoints on the
    /// trailing silence, then transcribes. Returns `None` when no speech arrived
    /// within the lead-in, the capture was near-silent (noise/echo), or the
    /// transcript came back empty. Always stops the source before returning.
    pub async fn dictate(&mut self) -> Result<Option<String>, VoiceError> {
        let mut rx = self.source.start()?;
        self.vad.reset();
        self.endpointer.arm(Some(self.lead_in));

        let captured = loop {
            match rx.recv().await {
                Some(chunk) => {
                    let prob = self.vad.speech_probability(&chunk)?;
                    match self.endpointer.push(&chunk, prob) {
                        Endpoint::Complete(samples) => break Some(samples),
                        Endpoint::Timeout => break None,
                        Endpoint::SpeechStarted | Endpoint::Accumulating => {}
                    }
                }
                None => break None, // source closed before an utterance completed
            }
        };

        self.source.stop()?;

        match captured {
            Some(samples) => Ok(self.transcriber.transcribe(&samples).await?.map(|t| t.text)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adele_voice_core::domain::Transcript;
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;
    use tokio::sync::mpsc;

    /// Source that hands out a pre-loaded receiver once; the test drives it.
    struct FakeSource {
        rx: StdMutex<Option<mpsc::Receiver<Vec<f32>>>>,
        stopped: StdMutex<bool>,
    }
    impl AudioSource for FakeSource {
        fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, VoiceError> {
            self.rx
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| VoiceError::Audio("already started".into()))
        }
        fn stop(&self) -> Result<(), VoiceError> {
            *self.stopped.lock().unwrap() = true;
            Ok(())
        }
    }

    /// VAD that replays scripted probabilities, then 0.0 once exhausted.
    struct FakeVad {
        probs: VecDeque<f32>,
    }
    impl VoiceActivityDetector for FakeVad {
        fn speech_probability(&mut self, _samples: &[f32]) -> Result<f32, VoiceError> {
            Ok(self.probs.pop_front().unwrap_or(0.0))
        }
        fn reset(&mut self) {}
    }

    struct FakeStt {
        text: String,
    }
    impl SpeechToText for FakeStt {
        async fn transcribe(&self, _samples: &[f32]) -> Result<Transcript, VoiceError> {
            Ok(Transcript {
                text: self.text.clone(),
            })
        }
    }

    fn opts() -> DictationOptions {
        DictationOptions {
            speech_threshold: 0.5,
            silence: Duration::from_millis(0),
            min_samples: 800,
            lead_in: Duration::from_secs(5),
        }
    }

    fn dictation(
        chunks: Vec<Vec<f32>>,
        vad: Vec<f32>,
        stt_text: &str,
        opts: DictationOptions,
    ) -> Dictation<FakeVad, FakeStt> {
        let (tx, rx) = mpsc::channel(64);
        for c in chunks {
            tx.try_send(c).unwrap();
        }
        drop(tx); // closing the channel ends the loop if no utterance completes
        let source = Arc::new(FakeSource {
            rx: StdMutex::new(Some(rx)),
            stopped: StdMutex::new(false),
        });
        Dictation::new(
            source,
            FakeVad {
                probs: VecDeque::from(vad),
            },
            FakeStt {
                text: stt_text.to_string(),
            },
            opts,
        )
    }

    #[tokio::test]
    async fn dictate_returns_transcript_for_speech_then_silence() {
        // Speech chunk (VAD 0.9) then silence (VAD 0.0) closes the utterance;
        // the buffer is loud enough to clear the energy gate.
        let mut d = dictation(
            vec![vec![0.1; 1000], vec![0.1; 1000]],
            vec![0.9],
            "what's the weather",
            opts(),
        );
        let out = d.dictate().await.unwrap();
        assert_eq!(out.as_deref(), Some("what's the weather"));
    }

    #[tokio::test]
    async fn dictate_times_out_with_no_speech() {
        // Lead-in of 0 ms and a sub-threshold chunk → Timeout → None, and STT
        // never runs (no transcript to gate).
        let mut d = dictation(
            vec![vec![0.0; 1000]],
            vec![0.0],
            "unreachable",
            DictationOptions {
                lead_in: Duration::from_millis(0),
                ..opts()
            },
        );
        assert!(d.dictate().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn dictate_gates_near_silent_capture() {
        // VAD reports speech-then-silence (so the utterance completes), but the
        // samples are ~silent — the energy gate drops it before returning text.
        let mut d = dictation(
            vec![vec![0.0; 1000], vec![0.0; 1000]],
            vec![0.9],
            "hallucinated filler",
            opts(),
        );
        assert!(d.dictate().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn dictate_stops_the_source() {
        let (tx, rx) = mpsc::channel(4);
        drop(tx); // empty + closed → loop ends immediately with None
        let source = Arc::new(FakeSource {
            rx: StdMutex::new(Some(rx)),
            stopped: StdMutex::new(false),
        });
        let stopped_handle = Arc::clone(&source);
        let mut d = Dictation::new(
            source,
            FakeVad {
                probs: VecDeque::new(),
            },
            FakeStt { text: "x".into() },
            opts(),
        );
        let _ = d.dictate().await.unwrap();
        assert!(
            *stopped_handle.stopped.lock().unwrap(),
            "dictate must stop the source even when nothing was captured"
        );
    }
}
