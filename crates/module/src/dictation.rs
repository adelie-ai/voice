//! One-shot dictation: open the mic, endpoint a single utterance, transcribe.
//!
//! This is "push-to-talk minus everything else" — no wake word, no continuous
//! loop, no assistant. A client taps a mic button, [`Dictation::dictate`] waits
//! for the user to speak and stop, and returns the transcript (or `None`).

use std::sync::Arc;
use std::time::Duration;

use adele_voice_core::VoiceError;
use adele_voice_core::ports::audio::{AudioSink, AudioSource};
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
    /// Settle time after the echo guard reports playback stopped before the mic
    /// is trusted again — clears the acoustic echo tail so the start of the
    /// captured utterance isn't the daemon's own dying TTS. Only consulted when
    /// an echo guard is set (see [`Dictation::with_echo_guard`]).
    pub echo_hangover: Duration,
}

impl Default for DictationOptions {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence: Duration::from_millis(800),
            min_samples: 800,
            lead_in: Duration::from_secs(10),
            echo_hangover: Duration::from_millis(200),
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
    /// Optional half-duplex echo guard: the output sink the client's [`Speaker`]
    /// plays through. When set, `dictate` waits out playback (and drops any
    /// chunk captured while playback is live) so Adele's own TTS isn't captured
    /// and transcribed back as a new user message. `None` = the original
    /// playback-unaware behavior.
    ///
    /// [`Speaker`]: crate::speaker::Speaker
    echo_guard: Option<Arc<dyn AudioSink>>,
    /// Settle time after `echo_guard` reports playback stopped before the mic is
    /// trusted; see [`DictationOptions::echo_hangover`].
    echo_hangover: Duration,
}

impl<V: VoiceActivityDetector, S: SpeechToText> Dictation<V, S> {
    pub fn new(source: Arc<dyn AudioSource>, vad: V, stt: S, opts: DictationOptions) -> Self {
        Self {
            source,
            vad,
            transcriber: Transcriber::new(Arc::new(stt)),
            endpointer: Endpointer::new(opts.speech_threshold, opts.silence, opts.min_samples),
            lead_in: opts.lead_in,
            echo_guard: None,
            echo_hangover: opts.echo_hangover,
        }
    }

    /// Attach a half-duplex echo guard: the same output sink the
    /// client's [`Speaker`] plays through. With it set, [`dictate`] consults
    /// [`AudioSink::is_playing`] to avoid capturing and transcribing Adele's own
    /// TTS — it waits out playback before arming, drains the buffered echo tail,
    /// and drops any chunk captured if playback restarts mid-listen. This is the
    /// cross-platform "floor" mitigation (no PipeWire / no AEC). Builder-style so
    /// callers can opt in without changing [`Dictation::new`].
    ///
    /// [`dictate`]: Self::dictate
    /// [`Speaker`]: crate::speaker::Speaker
    #[must_use]
    pub fn with_echo_guard(mut self, sink: Arc<dyn AudioSink>) -> Self {
        self.echo_guard = Some(sink);
        self
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

        // Half-duplex echo guard: if the output sink is still
        // sounding Adele's TTS, wait it out, let the acoustic tail settle, then
        // drop the echo frames already buffered on the source — mirrors the
        // daemon's `drain_playback_echo`. Skipped entirely when no guard is set.
        if let Some(guard) = &self.echo_guard {
            while guard.is_playing() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            tokio::time::sleep(self.echo_hangover).await;
            while rx.try_recv().is_ok() {}
        }

        self.endpointer.arm(Some(self.lead_in));

        let captured = loop {
            match rx.recv().await {
                Some(chunk) => {
                    // Defensive: if playback restarts mid-listen, drop the chunk
                    // rather than feed the echo to the VAD/endpointer.
                    if let Some(guard) = &self.echo_guard
                        && guard.is_playing()
                    {
                        continue;
                    }
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
    use std::sync::atomic::{AtomicBool, Ordering};
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

    /// Sink that only models `is_playing()`, flipped by the test. The echo guard
    /// never plays through it (dictation only reads `is_playing`), so `play` /
    /// `stop` are no-ops here.
    #[derive(Default)]
    struct FakeSink {
        playing: AtomicBool,
    }
    impl FakeSink {
        fn set_playing(&self, playing: bool) {
            self.playing.store(playing, Ordering::SeqCst);
        }
    }
    impl AudioSink for FakeSink {
        fn play(&self, _samples: Vec<f32>) -> Result<(), VoiceError> {
            Ok(())
        }
        fn stop(&self) -> Result<(), VoiceError> {
            Ok(())
        }
        fn is_playing(&self) -> bool {
            self.playing.load(Ordering::SeqCst)
        }
    }

    fn opts() -> DictationOptions {
        DictationOptions {
            speech_threshold: 0.5,
            silence: Duration::from_millis(0),
            min_samples: 800,
            lead_in: Duration::from_secs(5),
            echo_hangover: Duration::from_millis(200),
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

    /// Build a guarded dictation over an open (not pre-closed) channel so a test
    /// can deliver chunks while `dictate` runs. Returns the dictation, the sender
    /// (drop it to end the loop), and the sink whose `is_playing` the test flips.
    fn guarded_dictation(
        vad: Vec<f32>,
        stt_text: &str,
        opts: DictationOptions,
    ) -> (
        Dictation<FakeVad, FakeStt>,
        mpsc::Sender<Vec<f32>>,
        Arc<FakeSink>,
    ) {
        let (tx, rx) = mpsc::channel(64);
        let source = Arc::new(FakeSource {
            rx: StdMutex::new(Some(rx)),
            stopped: StdMutex::new(false),
        });
        let sink = Arc::new(FakeSink::default());
        let d = Dictation::new(
            source,
            FakeVad {
                probs: VecDeque::from(vad),
            },
            FakeStt {
                text: stt_text.to_string(),
            },
            opts,
        )
        .with_echo_guard(Arc::clone(&sink) as Arc<dyn AudioSink>);
        (d, tx, sink)
    }

    #[tokio::test(start_paused = true)]
    async fn dictate_drops_chunk_that_arrives_while_guard_is_playing() {
        // The mid-listen guard: dictate arms while the guard is IDLE (so the
        // pre-arm wait is skipped), then playback starts during capture. The
        // chunk that arrives while playing must be dropped (echo), not fed to the
        // endpointer; once playback stops, the genuine utterance transcribes.
        //
        // NB: must flip the guard back to idle — a guard left permanently playing
        // would wedge dictate (it waits out playback), which is the wait-path
        // case covered by `dictate_waits_for_playback_then_captures_utterance`,
        // not this drop-path case.
        let (mut d, tx, sink) = guarded_dictation(vec![0.9], "what's the weather", opts());
        let task = tokio::spawn(async move { d.dictate().await });
        // Advance past the pre-arm hangover (200ms) + its (empty) drain so dictate
        // is parked IN the capture loop before any chunk arrives — otherwise the
        // pre-arm drain, not the mid-loop guard, would eat the chunks. Idle guard
        // at arm means the pre-arm wait loop is skipped; only the hangover runs.
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Playback live → this echo chunk must be dropped (mid-loop `continue`),
        // never reaching the VAD/endpointer.
        sink.set_playing(true);
        tx.send(vec![0.1; 1000]).await.unwrap();
        // Give dictate a beat to receive + drop the echo while still "playing".
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Playback stops; the real utterance (speech then silence) transcribes.
        sink.set_playing(false);
        tx.send(vec![0.1; 1000]).await.unwrap();
        tx.send(vec![0.1; 1000]).await.unwrap();
        drop(tx);

        let out = task.await.unwrap().unwrap();
        assert_eq!(
            out.as_deref(),
            Some("what's the weather"),
            "the echo captured while playing must be dropped; the later real utterance transcribes"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn dictate_waits_for_playback_then_captures_utterance() {
        // Guard starts playing; dictate must poll-wait until it stops, settle the
        // hangover, drain the buffered echo, then capture the real utterance that
        // arrives afterwards. Paused time lets us advance the hangover
        // deterministically.
        let (mut d, tx, sink) = guarded_dictation(vec![0.9], "what's the weather", opts());
        sink.set_playing(true);
        // Echo buffered during playback — must be drained, never transcribed.
        tx.try_send(vec![0.1; 1000]).unwrap();

        let task = tokio::spawn(async move { d.dictate().await });

        // Let dictate reach its first is_playing() poll, then stop playback.
        tokio::task::yield_now().await;
        sink.set_playing(false);
        // Advance past the 50ms poll and the 200ms hangover so the drain runs.
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Now deliver the genuine utterance: speech chunk then a silence chunk to
        // close it (silence Duration is 0 in opts()).
        tx.try_send(vec![0.1; 1000]).unwrap();
        tx.try_send(vec![0.1; 1000]).unwrap();
        drop(tx);

        let out = task.await.unwrap().unwrap();
        assert_eq!(
            out.as_deref(),
            Some("what's the weather"),
            "after waiting out playback, the real utterance must transcribe; the echo must not"
        );
    }

    #[tokio::test]
    async fn dictate_without_guard_is_unchanged() {
        // No echo guard → identical to the playback-unaware path: speech then
        // silence transcribes normally.
        let mut d = dictation(
            vec![vec![0.1; 1000], vec![0.1; 1000]],
            vec![0.9],
            "no guard here",
            opts(),
        );
        let out = d.dictate().await.unwrap();
        assert_eq!(out.as_deref(), Some("no guard here"));
    }
}
