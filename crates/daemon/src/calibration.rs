//! Wake-word calibration policy (#121): turn a handful of measured wake-word
//! peak scores into a recommended sensitivity cutoff. This module is pure and
//! unit-tested; the microphone capture loop that *produces* the peaks lives in
//! the pipeline (it needs the running detector and audio source).

use std::time::Duration;

use adele_voice_dbus_interface::CalibrationOutcome;
use adele_voice_wake_rustpotter::{MAX_SENSITIVITY, MIN_SENSITIVITY};

/// Number of utterances to ask for when the caller passes 0.
pub const DEFAULT_UTTERANCES: u32 = 5;
/// Lower clamp for a requested utterance count — fewer than this isn't
/// statistically meaningful.
pub const MIN_UTTERANCES: u32 = 3;
/// Upper clamp for a requested utterance count — more than this and the user
/// gives up before finishing.
pub const MAX_UTTERANCES: u32 = 10;
/// Minimum successful captures before a recommendation is trustworthy.
pub const MIN_SAMPLES: usize = 3;
/// How long to wait for each spoken wake word before prompting a retry.
pub const UTTERANCE_TIMEOUT: Duration = Duration::from_secs(12);
/// A measured peak below this is treated as "no clear wake word heard" (only
/// ambient/noise weakly matched the template) rather than a real utterance, so
/// it's retried instead of recorded. Set above typical ambient template-match
/// noise and well below a genuine "Hey Adele" (~0.4+), so noise can't latch the
/// settle timer before the user actually speaks.
pub const MIN_PEAK: f32 = 0.20;
/// Once the running peak has reached [`MIN_PEAK`] and then stopped rising for
/// this long, the utterance is considered finished and its peak is recorded —
/// no dependence on the detector "firing".
pub const PEAK_SETTLE: Duration = Duration::from_millis(800);

/// Smallest margin below the worst observed peak (in score units), so the cutoff
/// never sits exactly on a score the user actually produced.
const MARGIN_MIN: f32 = 0.04;
/// How much the margin widens per unit of score spread (standard deviation):
/// inconsistent scores get a wider safety margin so a slightly weaker future
/// utterance still fires.
const MARGIN_STDDEV_K: f32 = 1.0;

/// Clamp a requested utterance count into the supported range, mapping 0 to the
/// default.
pub fn clamp_utterances(requested: u32) -> u32 {
    if requested == 0 {
        DEFAULT_UTTERANCES
    } else {
        requested.clamp(MIN_UTTERANCES, MAX_UTTERANCES)
    }
}

/// Total listen attempts (captures + retries) allowed for `utterances`, so a
/// silent room or a flaky mic can't hang calibration forever.
pub fn max_attempts(utterances: u32) -> u32 {
    utterances.saturating_mul(3)
}

/// Turn measured per-utterance peak scores into a recommended cutoff plus the
/// stats it came from, or `None` if there aren't enough samples.
///
/// The cutoff sits a margin below the WORST (minimum) observed peak, so every
/// measured utterance would still have fired; the margin widens with the score
/// spread (stddev) so an inconsistent speaker/mic gets more headroom. The result
/// is clamped to the detector's valid sensitivity range.
pub fn recommend(peaks: &[f32]) -> Option<CalibrationOutcome> {
    if peaks.len() < MIN_SAMPLES {
        return None;
    }
    let n = peaks.len() as f32;
    let min = peaks.iter().copied().fold(f32::INFINITY, f32::min);
    let max = peaks.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = peaks.iter().sum::<f32>() / n;
    let variance = peaks.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / n;
    let stddev = variance.sqrt();
    let margin = MARGIN_MIN.max(MARGIN_STDDEV_K * stddev);
    let sensitivity = (min - margin).clamp(MIN_SENSITIVITY, MAX_SENSITIVITY);
    Some(CalibrationOutcome {
        sensitivity,
        samples: peaks.len() as u32,
        min_peak: min,
        max_peak: max,
        mean_peak: mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_samples_yields_no_recommendation() {
        assert!(recommend(&[]).is_none());
        assert!(recommend(&[0.5, 0.5]).is_none(), "below MIN_SAMPLES");
    }

    #[test]
    fn consistent_peaks_sit_a_small_margin_below_the_worst() {
        // Tight cluster around 0.5 → stddev ~0, so the margin is the floor and
        // the cutoff is just below the minimum observed score.
        let out = recommend(&[0.50, 0.51, 0.49, 0.50]).unwrap();
        assert_eq!(out.samples, 4);
        assert!((out.min_peak - 0.49).abs() < 1e-6);
        assert!((out.max_peak - 0.51).abs() < 1e-6);
        // cutoff = min - MARGIN_MIN (stddev negligible) = 0.49 - 0.04 = 0.45.
        assert!(
            (out.sensitivity - 0.45).abs() < 1e-3,
            "got {}",
            out.sensitivity
        );
        assert!(
            out.sensitivity < out.min_peak,
            "cutoff must sit below every measured utterance"
        );
    }

    #[test]
    fn noisy_peaks_get_a_wider_margin() {
        // Same minimum (0.40) but a wide spread → margin grows beyond the floor,
        // so the cutoff sits further below the minimum than in the tight case.
        let tight = recommend(&[0.40, 0.41, 0.40, 0.41]).unwrap();
        let noisy = recommend(&[0.40, 0.70, 0.55, 0.85]).unwrap();
        assert!(
            noisy.sensitivity < tight.sensitivity,
            "noisy {} should be lower than tight {}",
            noisy.sensitivity,
            tight.sensitivity
        );
    }

    #[test]
    fn cutoff_is_clamped_to_the_detector_range() {
        // Very low scores would push the cutoff below the hard minimum.
        let out = recommend(&[0.11, 0.12, 0.11]).unwrap();
        assert!(
            out.sensitivity >= MIN_SENSITIVITY,
            "must not recommend below the hard minimum"
        );
    }

    #[test]
    fn clamp_utterances_maps_zero_to_default_and_bounds_the_rest() {
        assert_eq!(clamp_utterances(0), DEFAULT_UTTERANCES);
        assert_eq!(clamp_utterances(1), MIN_UTTERANCES);
        assert_eq!(clamp_utterances(100), MAX_UTTERANCES);
        assert_eq!(clamp_utterances(5), 5);
    }
}
