//! The audible "Listening" cue (#51).
//!
//! When the daemon enters the `Listening` state — on a wake-word fire and on
//! each conversation-mode follow-up re-listen — it gives the user instant,
//! reliable feedback that the mic is open. Two flavours:
//!
//! - [`ListeningCue::Ding`] — a short generated earcon (no shipped asset; the
//!   tone is synthesized in [`ding_samples`]). Instant.
//! - [`ListeningCue::Phrase`] — a spoken micro-phrase ("Yes?", "How can I
//!   help?", …) rotated deterministically through [`phrase`]. Friendlier, but
//!   adds the synthesis/playback latency of a short TTS utterance.
//!
//! The pure pieces (tone generation, phrase rotation) live here so they unit-
//! test without any audio hardware.

use adele_voice_core::domain::SAMPLE_RATE;

pub use crate::config::ListeningCue;

/// Earcon frequency (A5). High enough to be crisp over speech, low enough to be
/// pleasant.
const DING_FREQ_HZ: f32 = 880.0;

/// Earcon duration. Long enough to register, short enough to feel instant.
const DING_DURATION_MS: u64 = 120;

/// Linear fade applied to each end of the tone (ms) to avoid the click an
/// abrupt start/stop produces.
const DING_FADE_MS: u64 = 10;

/// Peak amplitude of the earcon (well below clipping, comfortable level).
const DING_AMPLITUDE: f32 = 0.25;

/// Rotating micro-phrases for [`ListeningCue::Phrase`]. Kept short so the added
/// latency stays close to ~1 s.
const PHRASES: &[&str] = &[
    "Yes?",
    "How can I help?",
    "I'm listening.",
    "Go ahead.",
    "Mm-hmm?",
];

/// Generate the ding earcon as mono f32 PCM at [`SAMPLE_RATE`]: a `DING_FREQ_HZ`
/// sine of `DING_DURATION_MS`, with a short linear fade in/out so it doesn't
/// click. Pure — no device, so it is unit-tested directly.
pub fn ding_samples() -> Vec<f32> {
    let sample_rate = SAMPLE_RATE as f32;
    let total = (sample_rate * DING_DURATION_MS as f32 / 1000.0) as usize;
    let fade = (sample_rate * DING_FADE_MS as f32 / 1000.0).max(1.0) as usize;
    // Clamp the fade so a fade-in plus fade-out never overruns a short tone.
    let fade = fade.min(total / 2);

    (0..total)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let sample = (2.0 * std::f32::consts::PI * DING_FREQ_HZ * t).sin() * DING_AMPLITUDE;
            // Linear fade in over the first `fade` samples, out over the last.
            let gain = if i < fade {
                i as f32 / fade as f32
            } else if i >= total - fade {
                (total - 1 - i) as f32 / fade as f32
            } else {
                1.0
            };
            sample * gain
        })
        .collect()
}

/// Pick the cue phrase for the `n`-th Listening entry, rotating deterministically
/// through [`PHRASES`] so repeated cues don't feel robotic. `n` is a free-running
/// counter the caller bumps each time it plays a phrase cue.
pub fn phrase(n: u64) -> &'static str {
    PHRASES[(n as usize) % PHRASES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ding_has_expected_length_and_is_bounded() {
        let samples = ding_samples();
        // 120 ms at 16 kHz = 1920 samples.
        assert_eq!(samples.len(), 1920);
        assert!(
            samples.iter().all(|s| s.abs() <= DING_AMPLITUDE + 1e-6),
            "the tone must never exceed its amplitude (no clipping)"
        );
    }

    #[test]
    fn ding_fades_in_and_out_to_avoid_clicks() {
        let samples = ding_samples();
        // First and last samples sit at (or essentially at) zero from the fade,
        // so playback starts/ends without an audible click.
        assert!(
            samples.first().unwrap().abs() < 1e-3,
            "must fade in from ~0"
        );
        assert!(samples.last().unwrap().abs() < 1e-3, "must fade out to ~0");
        // A sample in the sustained middle should carry real energy.
        let mid = samples[samples.len() / 2];
        assert!(mid.abs() > 0.01, "the middle of the tone must be audible");
    }

    #[test]
    fn phrase_rotates_deterministically_and_wraps() {
        let len = PHRASES.len() as u64;
        // Consecutive counters give consecutive phrases.
        assert_eq!(phrase(0), PHRASES[0]);
        assert_eq!(phrase(1), PHRASES[1]);
        // Wraps cleanly at the end of the list.
        assert_eq!(phrase(len), PHRASES[0]);
        assert_eq!(phrase(len + 2), PHRASES[2]);
    }

    #[test]
    fn all_phrases_are_non_empty() {
        assert!(!PHRASES.is_empty(), "there must be at least one phrase");
        assert!(
            PHRASES.iter().all(|p| !p.trim().is_empty()),
            "every rotating phrase must be speakable"
        );
    }
}
