//! Online wake-sensitivity adaptation (#121, Phase 2). Passive, bounded, slow
//! self-tuning of the wake cutoff from signals we already have per turn. This
//! module is the PURE policy — targeting + guardrails; the pipeline feeds it a
//! labeled observation after each wake-initiated turn and logs (and, when
//! enabled, applies) the moves it proposes.
//!
//! Phase 1 (the calibration wizard) sets the cutoff from a one-time guided
//! measurement. This keeps it tuned as the room / mic / voice drift, without
//! another wizard, from two labels we can read for free:
//! - a wake that led to a real transcribed command is a **true positive** at
//!   that score — the cutoff belongs at/below it (mirrors calibration, which
//!   sits a margin below the weakest measured peak);
//! - a wake that produced no usable speech at all (an empty / near-silent
//!   capture) is a **false positive** — the cutoff belongs above it.
//!
//! Near-misses — real wakes that scored just UNDER the cutoff and so never fired
//! — are invisible here (the detector reports nothing below the cutoff in normal
//! operation) and are handled by a follow-up; see #121. Consequently this first
//! increment can tighten toward where real wakes land and reject observed false
//! fires, but cannot by itself discover a real wake that is silently missed.
//!
//! ## Guardrails (so it is safe to run unattended)
//! - **Anchor band** — never strays more than [`ANCHOR_BAND`] from the last
//!   *calibrated* (or hand-set) cutoff, the known-good center.
//! - **Slew** — moves at most [`MAX_STEP`] per adjustment (slow adaptation).
//! - **Deadband** — holds for proposed moves smaller than [`DEADBAND`]
//!   (hysteresis; also keeps the detector-rebuild cost of an apply rare).
//! - **Cadence** — re-evaluates only every [`MIN_OBSERVATIONS`] observations, so
//!   a single odd turn can't move the cutoff.
//! - **Hard clamp** to the detector's `[MIN_SENSITIVITY, MAX_SENSITIVITY]`.
//! - **Mode-aware band** — symmetric in eager mode; **raise-only** in non-eager
//!   mode (a non-eager cutoff lowered toward the noise floor stops firing — the
//!   Phase 1 "wake went dead" trap), so it never drops below the anchor there.

use std::collections::VecDeque;

use adele_voice_wake_rustpotter::{MAX_SENSITIVITY, MIN_SENSITIVITY};

/// How a fired wake turned out, judged by what followed it in the same turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeLabel {
    /// The wake led to a real transcribed command (sent to the assistant).
    TruePositive,
    /// The wake produced no usable speech — an empty / near-silent capture.
    FalsePositive,
}

/// One labeled wake observation: the score the detector fired at and the label.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Observation {
    score: f32,
    label: WakeLabel,
}

/// A proposed cutoff move, surfaced to the pipeline. The pipeline always LOGS it;
/// it APPLIES it (rebuilds the detector at `to`) only when [`Adjustment::apply`]
/// is set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adjustment {
    /// The current live cutoff (where the move starts).
    pub from: f32,
    /// The next cutoff if applied — `from` slewed at most [`MAX_STEP`] toward
    /// `target`.
    pub to: f32,
    /// The banded destination the accumulated evidence argues for (the cutoff
    /// converges here over successive steps). Logged so the intent is visible
    /// even while log-only.
    pub target: f32,
    /// True-positive and false-positive scores overlap — no cutoff cleanly
    /// separates them, so `target` is a compromise. Logged so a persistently
    /// ambiguous mic/room is visible.
    pub ambiguous: bool,
    /// Whether the tuner is enabled to APPLY this move. `false` = log-only (the
    /// default while we live with Phase 1): the pipeline logs the proposal but
    /// leaves the live cutoff untouched.
    pub apply: bool,
}

/// Never let the online cutoff stray more than this from the last CALIBRATED
/// value (the known-good anchor). Bounds drift within a session.
const ANCHOR_BAND: f32 = 0.10;
/// Largest single cutoff move — adaptation is deliberately slow.
const MAX_STEP: f32 = 0.02;
/// Hold for proposed moves smaller than this (hysteresis).
const DEADBAND: f32 = 0.01;
/// Re-evaluate (and possibly step) only once per this many observations, so a
/// single odd turn can't move the cutoff and applies (detector rebuilds) stay
/// infrequent.
pub const MIN_OBSERVATIONS: usize = 5;
/// Bound on retained observations — recent behaviour only.
const HISTORY_CAP: usize = 64;
/// Margin below the weakest true-positive score for the target — a real wake a
/// little weaker than any yet seen still clears the cutoff. Mirrors
/// calibration's eager margin.
const TP_MARGIN: f32 = 0.05;
/// Margin above the strongest false-positive score for the target.
const FP_MARGIN: f32 = 0.02;

/// The raw cutoff the observations argue for (before banding / slew), plus
/// whether TP/FP overlap forced a compromise. `None` when there is no evidence
/// of either kind yet.
///
/// - only true positives → a margin below the weakest (calibration-style);
/// - only false positives → a margin above the strongest;
/// - both, separable → below the weakest true positive (it already clears the
///   false fires);
/// - both, overlapping (a false fire scored as high as a real wake) → the
///   midpoint, flagged ambiguous.
fn target_cutoff(history: &[Observation]) -> Option<(f32, bool)> {
    let min_tp = history
        .iter()
        .filter(|o| o.label == WakeLabel::TruePositive)
        .map(|o| o.score)
        .fold(f32::INFINITY, f32::min);
    let max_fp = history
        .iter()
        .filter(|o| o.label == WakeLabel::FalsePositive)
        .map(|o| o.score)
        .fold(f32::NEG_INFINITY, f32::max);

    match (min_tp.is_finite(), max_fp.is_finite()) {
        (true, false) => Some((min_tp - TP_MARGIN, false)),
        (false, true) => Some((max_fp + FP_MARGIN, false)),
        (true, true) => {
            // Sit below the weakest real wake (keep catching it) yet above the
            // strongest false fire (reject it).
            let upper = min_tp - TP_MARGIN;
            let lower = max_fp + FP_MARGIN;
            if lower <= upper {
                // A comfortable gap: below the weakest real already clears the
                // false fires. Mirrors the calibration cutoff.
                Some((upper, false))
            } else {
                // Overlap — a false fire scored as high as a real wake, so no
                // cutoff separates them. Split the difference and flag it.
                Some(((min_tp + max_fp) / 2.0, true))
            }
        }
        (false, false) => None,
    }
}

/// Online wake-cutoff tuner (#121, Phase 2). Owns the observation history and a
/// mirror of the live cutoff; the pipeline feeds it labeled fires and logs /
/// applies its proposals. Free of I/O — all timing and persistence live in the
/// pipeline.
pub struct WakeTuner {
    /// Whether proposals may be APPLIED (vs log-only). Hot-toggled from config.
    enabled: bool,
    /// Current wake mode — controls the band direction (see module docs).
    eager: bool,
    /// Last calibrated (or hand-set) cutoff: the fixed center of the drift band.
    anchor: f32,
    /// The live cutoff, kept in step with the detector via [`Self::commit`].
    current: f32,
    history: VecDeque<Observation>,
    /// Observations since the last evaluation (the cadence counter).
    since_eval: usize,
}

impl WakeTuner {
    /// Create a tuner anchored at the current cutoff and mode. `enabled` is the
    /// `wake_word.auto_adapt` toggle (default off — log-only).
    pub fn new(cutoff: f32, eager: bool, enabled: bool) -> Self {
        Self {
            enabled,
            eager,
            anchor: cutoff,
            current: cutoff,
            history: VecDeque::new(),
            since_eval: 0,
        }
    }

    /// Hot-toggle whether proposals may be applied (config reload).
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// The live cutoff the tuner believes is in effect.
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Re-anchor to a freshly calibrated cutoff + mode and forget history — start
    /// adapting again from the new known-good baseline.
    pub fn recalibrated(&mut self, cutoff: f32, eager: bool) {
        self.anchor = cutoff;
        self.current = cutoff;
        self.eager = eager;
        self.history.clear();
        self.since_eval = 0;
    }

    /// Re-anchor to a new cutoff (a manual `wake_word.sensitivity` edit) without
    /// changing the mode, forgetting history.
    pub fn reanchor(&mut self, cutoff: f32) {
        self.anchor = cutoff;
        self.current = cutoff;
        self.history.clear();
        self.since_eval = 0;
    }

    /// Confirm an applied move — the pipeline calls this once the detector has
    /// taken the new cutoff, so the tuner's mirror stays in step.
    pub fn commit(&mut self, cutoff: f32) {
        self.current = cutoff;
    }

    /// Record one labeled wake and, on the evaluation cadence, return a proposed
    /// move (banded, slewed, deadbanded). `None` when it isn't time to evaluate
    /// or nothing moves. Recording happens regardless of `enabled`, so a log-only
    /// tuner still surfaces what it WOULD do.
    pub fn observe(&mut self, score: f32, label: WakeLabel) -> Option<Adjustment> {
        self.history.push_back(Observation { score, label });
        while self.history.len() > HISTORY_CAP {
            self.history.pop_front();
        }
        self.since_eval += 1;
        if self.since_eval < MIN_OBSERVATIONS {
            return None;
        }
        self.since_eval = 0;

        let (raw, ambiguous) = target_cutoff(self.history.make_contiguous())?;

        // Band to the anchor: symmetric in eager, raise-only in non-eager (a
        // non-eager cutoff dropped toward the noise floor stops firing).
        let (lo, hi) = if self.eager {
            (self.anchor - ANCHOR_BAND, self.anchor + ANCHOR_BAND)
        } else {
            (self.anchor, self.anchor + ANCHOR_BAND)
        };
        let target = raw.clamp(lo, hi).clamp(MIN_SENSITIVITY, MAX_SENSITIVITY);

        // Slew toward the banded target, then apply the deadband (hysteresis).
        let delta = (target - self.current).clamp(-MAX_STEP, MAX_STEP);
        if delta.abs() < DEADBAND {
            return None;
        }

        Some(Adjustment {
            from: self.current,
            to: self.current + delta,
            target,
            ambiguous,
            apply: self.enabled,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tp(score: f32) -> Observation {
        Observation {
            score,
            label: WakeLabel::TruePositive,
        }
    }
    fn fp(score: f32) -> Observation {
        Observation {
            score,
            label: WakeLabel::FalsePositive,
        }
    }

    // Drive a full evaluation cycle: feed `n` copies of one observation, applying
    // any proposal (as the pipeline would when enabled). Returns the last
    // proposal seen. Feeds a multiple of MIN_OBSERVATIONS so an eval fires.
    fn feed(t: &mut WakeTuner, obs: Observation, n: usize) -> Option<Adjustment> {
        let mut last = None;
        for _ in 0..n {
            if let Some(adj) = t.observe(obs.score, obs.label) {
                if adj.apply {
                    t.commit(adj.to);
                }
                last = Some(adj);
            }
        }
        last
    }

    // --- pure target_cutoff ---

    #[test]
    fn no_observations_yields_no_target() {
        assert_eq!(target_cutoff(&[]), None);
    }

    #[test]
    fn only_true_positives_targets_margin_below_weakest() {
        let (target, ambiguous) = target_cutoff(&[tp(0.42), tp(0.45), tp(0.40)]).unwrap();
        assert!(!ambiguous);
        assert!((target - (0.40 - TP_MARGIN)).abs() < 1e-6, "got {target}");
    }

    #[test]
    fn only_false_positives_targets_margin_above_strongest() {
        let (target, ambiguous) = target_cutoff(&[fp(0.18), fp(0.22), fp(0.20)]).unwrap();
        assert!(!ambiguous);
        assert!((target - (0.22 + FP_MARGIN)).abs() < 1e-6, "got {target}");
    }

    #[test]
    fn separable_tp_fp_sits_below_weakest_true_positive() {
        // Weakest real 0.42, strongest false 0.20 → comfortable gap → sit a
        // margin below the real (which already clears the false fire).
        let (target, ambiguous) = target_cutoff(&[tp(0.42), fp(0.20)]).unwrap();
        assert!(!ambiguous);
        assert!((target - (0.42 - TP_MARGIN)).abs() < 1e-6, "got {target}");
    }

    #[test]
    fn overlapping_tp_fp_splits_and_flags_ambiguous() {
        // A false fire (0.32) scored HIGHER than a real wake (0.30): no cutoff
        // separates them → midpoint, flagged.
        let (target, ambiguous) = target_cutoff(&[tp(0.30), fp(0.32)]).unwrap();
        assert!(ambiguous);
        assert!((target - 0.31).abs() < 1e-6, "got {target}");
    }

    // --- stateful WakeTuner: cadence & guardrails ---

    #[test]
    fn holds_until_min_observations() {
        let mut t = WakeTuner::new(0.30, true, true);
        for _ in 0..(MIN_OBSERVATIONS - 1) {
            assert_eq!(t.observe(0.20, WakeLabel::FalsePositive), None);
        }
    }

    #[test]
    fn proposes_after_min_observations() {
        // False fires below an over-loose cutoff should propose raising it.
        let mut t = WakeTuner::new(0.20, true, true);
        let adj = feed(&mut t, fp(0.24), MIN_OBSERVATIONS);
        let adj = adj.expect("an eval should fire at the cadence");
        assert!(adj.to > adj.from, "false fires should raise the cutoff");
    }

    #[test]
    fn slew_limits_step_to_max_step() {
        // A far target must still move at most MAX_STEP in one step.
        let mut t = WakeTuner::new(0.20, true, true);
        let adj = feed(&mut t, tp(0.42), MIN_OBSERVATIONS).unwrap();
        assert!((adj.to - adj.from).abs() <= MAX_STEP + 1e-6, "step {adj:?}");
    }

    #[test]
    fn deadband_holds_small_moves() {
        // Target essentially equals current → no move.
        let mut t = WakeTuner::new(0.37, true, true);
        // Weakest TP 0.42 → target 0.37 == current.
        assert_eq!(feed(&mut t, tp(0.42), MIN_OBSERVATIONS), None);
    }

    #[test]
    fn respects_anchor_band_eager_symmetric() {
        // Target 0.90 is far above the band; the cutoff may never exceed
        // anchor + ANCHOR_BAND however long it runs.
        let mut t = WakeTuner::new(0.30, true, true);
        feed(&mut t, tp(0.95), 200);
        assert!(t.current() <= 0.30 + ANCHOR_BAND + 1e-6, "got {}", t.current());
    }

    #[test]
    fn non_eager_band_is_raise_only() {
        // Non-eager anchored at 0.40; TPs at 0.42 give an eager target of 0.37
        // (a LOWER move) — non-eager must refuse to drop below the anchor.
        let mut t = WakeTuner::new(0.40, false, true);
        feed(&mut t, tp(0.42), 200);
        assert!(t.current() >= 0.40 - 1e-6, "non-eager lowered below anchor: {}", t.current());
    }

    #[test]
    fn clamps_to_detector_max() {
        // Anchor near the ceiling; near-1.0 false fires push the target over
        // MAX_SENSITIVITY, so the hard clamp must cap the live cutoff.
        let mut t = WakeTuner::new(MAX_SENSITIVITY - 0.03, true, true);
        feed(&mut t, fp(0.99), 200);
        assert!(t.current() <= MAX_SENSITIVITY + 1e-6, "got {}", t.current());
    }

    #[test]
    fn log_only_marks_apply_false() {
        let mut t = WakeTuner::new(0.20, true, false);
        let adj = feed(&mut t, fp(0.24), MIN_OBSERVATIONS).unwrap();
        assert!(!adj.apply, "a disabled tuner must not mark proposals applyable");
    }

    #[test]
    fn enabled_marks_apply_true() {
        let mut t = WakeTuner::new(0.20, true, true);
        let adj = feed(&mut t, fp(0.24), MIN_OBSERVATIONS).unwrap();
        assert!(adj.apply);
    }

    #[test]
    fn history_capped_evicts_oldest() {
        // Flood with FPs, then flood with TPs; once the FPs age out past the cap
        // the target should reflect only the recent TPs (no lingering FP floor).
        let mut t = WakeTuner::new(0.30, true, false);
        for _ in 0..HISTORY_CAP {
            let _ = t.observe(0.10, WakeLabel::FalsePositive);
        }
        for _ in 0..HISTORY_CAP {
            let _ = t.observe(0.42, WakeLabel::TruePositive);
        }
        // After a further eval the proposal must reflect TP-only targeting.
        let adj = feed(&mut t, tp(0.42), MIN_OBSERVATIONS).unwrap();
        assert!(!adj.ambiguous, "aged-out FPs should no longer force overlap");
        assert!((adj.target - (0.42 - TP_MARGIN)).abs() < 1e-6, "got {}", adj.target);
    }

    #[test]
    fn set_enabled_toggles_apply() {
        let mut t = WakeTuner::new(0.20, true, false);
        assert!(!feed(&mut t, fp(0.24), MIN_OBSERVATIONS).unwrap().apply);
        t.set_enabled(true);
        assert!(feed(&mut t, fp(0.24), MIN_OBSERVATIONS).unwrap().apply);
    }

    #[test]
    fn reanchor_resets_without_changing_mode() {
        // Non-eager tuner: reanchor to 0.50 and confirm the mode is preserved by
        // checking it stays raise-only (an eager target of 0.47 is refused).
        let mut t = WakeTuner::new(0.30, false, true);
        let _ = feed(&mut t, fp(0.28), MIN_OBSERVATIONS * 2);
        t.reanchor(0.50);
        assert!((t.current() - 0.50).abs() < 1e-6);
        feed(&mut t, tp(0.52), 50);
        assert!(t.current() >= 0.50 - 1e-6, "reanchor must preserve non-eager raise-only");
    }

    #[test]
    fn recalibrated_resets_anchor_mode_and_clears_history() {
        let mut t = WakeTuner::new(0.30, true, true);
        let _ = feed(&mut t, fp(0.25), MIN_OBSERVATIONS * 2);
        t.recalibrated(0.45, false);
        assert!((t.current() - 0.45).abs() < 1e-6);
        // History cleared → the next partial batch produces no proposal yet.
        for _ in 0..(MIN_OBSERVATIONS - 1) {
            assert_eq!(t.observe(0.50, WakeLabel::TruePositive), None);
        }
    }

    // --- ACCEPTANCE: converges on synthetic TP/FP streams (#121) ---

    #[test]
    fn converges_on_synthetic_tp_fp_stream() {
        // Real wakes cluster ~0.42, false fires ~0.20, anchor an over-loose 0.30.
        // The cutoff should climb to ~0.37 (a margin below the weakest real, well
        // above the false fires) and then hold, staying inside every bound.
        let mut t = WakeTuner::new(0.30, true, true);
        for i in 0..200 {
            let obs = if i % 2 == 0 { tp(0.42) } else { fp(0.20) };
            if let Some(adj) = t.observe(obs.score, obs.label) {
                if adj.apply {
                    t.commit(adj.to);
                }
                assert!(adj.to >= 0.30 - ANCHOR_BAND && adj.to <= 0.30 + ANCHOR_BAND);
                assert!(adj.to >= MIN_SENSITIVITY && adj.to <= MAX_SENSITIVITY);
            }
        }
        assert!((t.current() - 0.37).abs() < 0.02, "did not converge: {}", t.current());
    }

    #[test]
    fn converges_up_on_false_positive_stream() {
        // Only false fires at 0.25 with an over-loose anchor 0.20 → raise to just
        // above them (~0.27), bounded by the anchor band.
        let mut t = WakeTuner::new(0.20, true, true);
        feed(&mut t, fp(0.25), 200);
        assert!((t.current() - (0.25 + FP_MARGIN)).abs() < 0.02, "got {}", t.current());
    }
}
