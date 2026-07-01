//! Wake-word calibration policy (#121): turn measured wake-word peak scores and
//! the room's background match level into a recommended cutoff AND the best wake
//! mode for this voice/mic. This module is pure and unit-tested; the microphone
//! capture loop that *produces* the measurements lives in the pipeline (it needs
//! the running detector and audio source).
//!
//! The two modes key off different things, so one calibration serves both:
//! - **Eager** fires on the rising edge, so its cutoff only needs to sit below
//!   the weakest peak.
//! - **Standard (non-eager)** fires when the score climbs past the cutoff and
//!   then falls back below it, so its cutoff must sit in the *gap* between the
//!   background match level and the weakest peak. With no usable gap it can't
//!   reliably finalize a wake — that's the case eager exists for.

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
/// How long to sample the room in silence to measure the background match level
/// before asking for utterances.
pub const AMBIENT_MEASURE: Duration = Duration::from_millis(2500);
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

/// Margin below the weakest peak for the EAGER cutoff — eager fires on the way
/// up, so it just needs to sit a little under the worst utterance.
const EAGER_MARGIN: f32 = 0.05;
/// Minimum separation between the background match level and the weakest peak
/// for STANDARD (non-eager) mode to be chosen. Non-eager only fires when the
/// score falls *back below* the cutoff, so the cutoff must sit both well above
/// the background (to fall back at all) AND below the peaks (to clear them) —
/// and it must do so before the energy gate resets the detector after speech.
/// A *generous* gap is required; anything tighter is handed to eager, which only
/// needs to sit below the peaks and is unaffected by the fall-back timing.
const MIN_GAP: f32 = 0.18;
/// Where in the floor→peak gap the non-eager cutoff sits, as a fraction up from
/// the floor. Biased toward the middle-high (0.55) so there's real headroom
/// above the background — the score reliably drops below the cutoff at the end
/// of the phrase — while still clearing below the peaks.
const NON_EAGER_FLOOR_FRACTION: f32 = 0.55;

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

/// Turn the measured utterance peaks and the background match level (`noise_floor`)
/// into a recommendation: a cutoff for each mode, and the best mode + cutoff to
/// apply. `None` if there aren't enough samples.
///
/// Best mode = **non-eager when the mic supports it** (a real gap between the
/// background level and the weakest peak) — it confirms the full match and false-
/// wakes less — otherwise **eager**, which only needs to sit below the peaks.
pub fn recommend(peaks: &[f32], noise_floor: f32) -> Option<CalibrationOutcome> {
    if peaks.len() < MIN_SAMPLES {
        return None;
    }
    let n = peaks.len() as f32;
    let min_peak = peaks.iter().copied().fold(f32::INFINITY, f32::min);
    let mean_peak = peaks.iter().sum::<f32>() / n;

    // Eager: sit a margin below the weakest peak so the rising edge always crosses.
    let eager_cutoff = (min_peak - EAGER_MARGIN).clamp(MIN_SENSITIVITY, MAX_SENSITIVITY);

    // Non-eager: place the cutoff inside the background→peak gap, if there is one.
    let gap = min_peak - noise_floor;
    let non_eager_cutoff = (gap >= MIN_GAP).then(|| {
        (noise_floor + NON_EAGER_FLOOR_FRACTION * gap).clamp(MIN_SENSITIVITY, MAX_SENSITIVITY)
    });

    let (eager, sensitivity) = match non_eager_cutoff {
        Some(c) => (false, c),
        None => (true, eager_cutoff),
    };

    Some(CalibrationOutcome {
        sensitivity,
        eager,
        samples: peaks.len() as u32,
        mean_peak,
        noise_floor,
        eager_cutoff,
        non_eager_cutoff: non_eager_cutoff.unwrap_or(-1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_samples_yields_no_recommendation() {
        assert!(recommend(&[], 0.1).is_none());
        assert!(recommend(&[0.5, 0.5], 0.1).is_none(), "below MIN_SAMPLES");
    }

    #[test]
    fn clear_gap_picks_non_eager_inside_the_gap() {
        // Strong, consistent peaks well above a low background → standard mode is
        // reliable and preferred; its cutoff sits between the floor and the peaks.
        let out = recommend(&[0.44, 0.45, 0.42, 0.42, 0.43], 0.15).unwrap();
        assert!(!out.eager, "a comfortable gap should choose non-eager");
        assert!(
            out.sensitivity > out.noise_floor && out.sensitivity < 0.42,
            "non-eager cutoff must sit inside the floor→peak gap, got {}",
            out.sensitivity
        );
        assert!(
            out.non_eager_cutoff > 0.0,
            "standard mode should be offered"
        );
        // eager_cutoff is still reported (just below the weakest peak = 0.42).
        assert!(
            (out.eager_cutoff - 0.37).abs() < 1e-3,
            "got {}",
            out.eager_cutoff
        );
    }

    #[test]
    fn no_gap_falls_back_to_eager() {
        // Weak peaks close to the background → no usable non-eager window, so
        // eager is chosen and standard mode is flagged unavailable (negative).
        let out = recommend(&[0.28, 0.30, 0.29], 0.22).unwrap();
        assert!(out.eager, "peaks near the floor must choose eager");
        assert!(
            out.non_eager_cutoff < 0.0,
            "standard mode must be flagged unavailable"
        );
        // eager cutoff = weakest (0.28) - EAGER_MARGIN = 0.23.
        assert!(
            (out.sensitivity - 0.23).abs() < 1e-3,
            "got {}",
            out.sensitivity
        );
    }

    #[test]
    fn marginal_gap_falls_back_to_eager() {
        // The real regression: strong, consistent peaks (~0.42) but a HIGH
        // background (~0.29), so the gap (~0.13) is too small for a dependable
        // non-eager fall-back → eager, not a cutoff that sits right on the noise.
        let out = recommend(&[0.42, 0.43, 0.42, 0.43, 0.42], 0.29).unwrap();
        assert!(out.eager, "a marginal gap must not choose non-eager");
        assert!(
            out.non_eager_cutoff < 0.0,
            "standard mode flagged unavailable"
        );
        assert!(
            out.sensitivity > 0.29,
            "the eager cutoff must sit above the background"
        );
    }

    #[test]
    fn applied_cutoff_matches_the_chosen_mode() {
        let non_eager = recommend(&[0.44, 0.45, 0.43], 0.15).unwrap();
        assert_eq!(non_eager.sensitivity, non_eager.non_eager_cutoff);
        let eager = recommend(&[0.28, 0.30, 0.29], 0.22).unwrap();
        assert_eq!(eager.sensitivity, eager.eager_cutoff);
    }

    #[test]
    fn cutoffs_are_clamped_to_the_detector_range() {
        // Very low scores would push a cutoff below the hard minimum.
        let out = recommend(&[0.11, 0.12, 0.11], 0.02).unwrap();
        assert!(out.sensitivity >= MIN_SENSITIVITY);
        assert!(out.eager_cutoff >= MIN_SENSITIVITY);
    }

    #[test]
    fn clamp_utterances_maps_zero_to_default_and_bounds_the_rest() {
        assert_eq!(clamp_utterances(0), DEFAULT_UTTERANCES);
        assert_eq!(clamp_utterances(1), MIN_UTTERANCES);
        assert_eq!(clamp_utterances(100), MAX_UTTERANCES);
        assert_eq!(clamp_utterances(5), 5);
    }
}
