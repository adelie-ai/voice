//! Single-utterance VAD endpointing — pure, chunk-driven, no I/O.
//!
//! Given a stream of captured audio chunks plus each chunk's speech probability
//! (from a [`VoiceActivityDetector`](adele_voice_core::ports::vad::VoiceActivityDetector)),
//! decide when one utterance has ended. This is the heart of the daemon's
//! `Listening` state and of [`Dictation`](crate::Dictation), shared so both
//! endpoint identically.

use std::time::{Duration, Instant};

/// The decision after feeding one chunk to an [`Endpointer`].
#[derive(Debug, Clone, PartialEq)]
pub enum Endpoint {
    /// Still accumulating — keep feeding chunks.
    Accumulating,
    /// Speech just began (the first chunk over the threshold this utterance).
    /// Distinguished only so callers can log/announce the transition.
    SpeechStarted,
    /// Silence after speech: the utterance is complete. Carries the captured
    /// PCM (the accumulated buffer, including the closing chunk).
    Complete(Vec<f32>),
    /// The lead-in elapsed with no speech at all — give up on this utterance.
    Timeout,
}

/// Accumulates captured audio and reports when an utterance has ended.
///
/// Endpointing rule (matching the daemon's long-standing behaviour):
/// - a chunk whose speech probability is `>= speech_threshold` (re)starts the
///   silence timer and cancels any lead-in deadline;
/// - once speech has been seen, `silence` of sub-threshold audio **and** at
///   least `min_samples` buffered closes the utterance ([`Endpoint::Complete`]);
/// - before any speech, an optional lead-in deadline elapsing yields
///   [`Endpoint::Timeout`].
pub struct Endpointer {
    speech_threshold: f32,
    silence: Duration,
    min_samples: usize,
    buffer: Vec<f32>,
    last_speech_at: Option<Instant>,
    deadline: Option<Instant>,
}

impl Endpointer {
    /// `min_samples` is the floor of buffered samples below which a silence will
    /// not close an utterance (guards against a single stray blip); the daemon
    /// uses 800 (50 ms at 16 kHz).
    pub fn new(speech_threshold: f32, silence: Duration, min_samples: usize) -> Self {
        Self {
            speech_threshold,
            silence,
            min_samples,
            buffer: Vec::new(),
            last_speech_at: None,
            deadline: None,
        }
    }

    /// Hot-swap the speech-probability threshold in place. The current
    /// utterance buffer and timers are left untouched, so a live reload
    /// (config#52) takes effect on the next pushed chunk without disturbing an
    /// in-flight capture.
    pub fn set_speech_threshold(&mut self, speech_threshold: f32) {
        self.speech_threshold = speech_threshold;
    }

    /// Hot-swap the post-speech silence duration that closes an utterance, in
    /// place (config#52). Like [`set_speech_threshold`](Self::set_speech_threshold),
    /// the in-flight buffer and timers are preserved.
    pub fn set_silence(&mut self, silence: Duration) {
        self.silence = silence;
    }

    /// Clear all state without arming a deadline (used when abandoning a turn).
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.last_speech_at = None;
        self.deadline = None;
    }

    /// Begin a fresh utterance from silence, with an optional lead-in: if no
    /// speech arrives before it elapses, [`push`](Self::push) returns
    /// [`Endpoint::Timeout`].
    pub fn arm(&mut self, lead_in: Option<Duration>) {
        self.reset();
        self.deadline = lead_in.map(|d| Instant::now() + d);
    }

    /// Begin an utterance already mid-speech (barge-in): seed the buffer with
    /// `chunk` and mark speech as having started now, so the next silence closes
    /// it normally.
    pub fn arm_speaking(&mut self, chunk: &[f32]) {
        self.buffer.clear();
        self.buffer.extend_from_slice(chunk);
        self.last_speech_at = Some(Instant::now());
        self.deadline = None;
    }

    /// Feed one captured chunk and its VAD speech probability.
    pub fn push(&mut self, chunk: &[f32], prob: f32) -> Endpoint {
        self.buffer.extend_from_slice(chunk);

        if prob >= self.speech_threshold {
            let started = self.last_speech_at.is_none();
            self.last_speech_at = Some(Instant::now());
            self.deadline = None; // speech began; cancel the lead-in
            if started {
                Endpoint::SpeechStarted
            } else {
                Endpoint::Accumulating
            }
        } else if let Some(last) = self.last_speech_at
            && last.elapsed() >= self.silence
            && self.buffer.len() > self.min_samples
        {
            Endpoint::Complete(std::mem::take(&mut self.buffer))
        } else if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            Endpoint::Timeout
        } else {
            Endpoint::Accumulating
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0 ms silence: one speech chunk then one sub-threshold chunk closes the
    /// utterance — provided the buffer cleared the `min_samples` floor.
    fn endpointer() -> Endpointer {
        Endpointer::new(0.5, Duration::from_millis(0), 800)
    }

    #[test]
    fn speech_then_silence_completes_with_buffer() {
        let mut ep = endpointer();
        ep.arm(None);
        assert_eq!(ep.push(&vec![0.1; 1000], 0.9), Endpoint::SpeechStarted);
        match ep.push(&vec![0.1; 1000], 0.0) {
            Endpoint::Complete(samples) => assert_eq!(samples.len(), 2000),
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn first_speech_chunk_is_speech_started_then_accumulating() {
        let mut ep = endpointer();
        ep.arm(None);
        assert_eq!(ep.push(&vec![0.1; 100], 0.9), Endpoint::SpeechStarted);
        // A second speech chunk is just accumulation, not another "started".
        assert_eq!(ep.push(&vec![0.1; 100], 0.9), Endpoint::Accumulating);
    }

    #[test]
    fn silence_below_min_samples_does_not_complete() {
        // Floor guards a stray blip: a tiny buffer must not close even on silence.
        let mut ep = Endpointer::new(0.5, Duration::from_millis(0), 800);
        ep.arm(None);
        assert_eq!(ep.push(&vec![0.1; 100], 0.9), Endpoint::SpeechStarted);
        // 100 + 100 = 200 samples, below the 800 floor → keep accumulating.
        assert_eq!(ep.push(&vec![0.1; 100], 0.0), Endpoint::Accumulating);
    }

    #[test]
    fn lead_in_times_out_with_no_speech() {
        let mut ep = Endpointer::new(0.5, Duration::from_millis(0), 800);
        ep.arm(Some(Duration::from_millis(0)));
        // Sub-threshold chunk after a 0 ms lead-in → Timeout (never any speech).
        assert_eq!(ep.push(&vec![0.0; 1000], 0.0), Endpoint::Timeout);
    }

    #[test]
    fn speech_cancels_the_lead_in() {
        let mut ep = Endpointer::new(0.5, Duration::from_millis(10_000), 800);
        ep.arm(Some(Duration::from_millis(0)));
        // Speech arrives — even though the lead-in already elapsed, speech wins
        // (checked first) and cancels the deadline, so no Timeout follows.
        assert_eq!(ep.push(&vec![0.1; 1000], 0.9), Endpoint::SpeechStarted);
        assert_eq!(ep.push(&vec![0.1; 1000], 0.9), Endpoint::Accumulating);
    }

    #[test]
    fn arm_speaking_seeds_buffer_for_barge_in() {
        let mut ep = Endpointer::new(0.5, Duration::from_millis(0), 800);
        ep.arm_speaking(&vec![0.1; 500]);
        // Already mid-speech: the next sub-threshold chunk closes immediately,
        // and the seeded 500 + this 500 clear the 800 floor.
        match ep.push(&vec![0.1; 500], 0.0) {
            Endpoint::Complete(samples) => assert_eq!(samples.len(), 1000),
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn set_speech_threshold_takes_effect_on_the_next_chunk() {
        // Live reload (config#52): raising the threshold mid-stream means a chunk
        // that previously read as speech no longer does.
        let mut ep = endpointer();
        ep.arm(None);
        // prob 0.6 ≥ default 0.5 → speech.
        assert_eq!(ep.push(&vec![0.1; 100], 0.6), Endpoint::SpeechStarted);
        ep.reset();
        ep.arm(None);
        ep.set_speech_threshold(0.8);
        // Same 0.6 now < 0.8 → no speech, just accumulation (no prior speech to close).
        assert_eq!(ep.push(&vec![0.1; 100], 0.6), Endpoint::Accumulating);
    }

    #[test]
    fn set_silence_changes_when_an_utterance_closes() {
        // With a long silence window, one sub-threshold chunk after speech keeps
        // accumulating; shortening it to 0 closes on the next silence chunk.
        let mut ep = Endpointer::new(0.5, Duration::from_secs(3600), 800);
        ep.arm(None);
        assert_eq!(ep.push(&vec![0.1; 1000], 0.9), Endpoint::SpeechStarted);
        assert_eq!(ep.push(&vec![0.1; 1000], 0.0), Endpoint::Accumulating);
        ep.set_silence(Duration::from_millis(0));
        match ep.push(&vec![0.1; 1000], 0.0) {
            Endpoint::Complete(samples) => assert_eq!(samples.len(), 3000),
            other => panic!("expected Complete after shortening silence, got {other:?}"),
        }
    }

    #[test]
    fn reset_clears_pending_state() {
        let mut ep = endpointer();
        ep.arm(None);
        ep.push(&vec![0.1; 1000], 0.9); // speech buffered
        ep.reset();
        // After reset there's no speech history, so a lone silence chunk can't
        // complete (would need prior speech) — it just accumulates.
        assert_eq!(ep.push(&vec![0.0; 1000], 0.0), Endpoint::Accumulating);
    }
}
