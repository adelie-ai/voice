//! Rustpotter wake-word detector, with an optional energy gate in front of it.
//!
//! # Why the gate exists
//!
//! Wake-word detection is always-on: every captured frame is fed to rustpotter,
//! which runs a full MFCC / FFT feature extraction plus a template comparison
//! per frame — even in a silent room, which is most of the idle time. Profiling
//! the live daemon showed this single path accounts for ~22% of a CPU core at
//! idle (it is the dominant cost; the ONNX models and the resampler are
//! near-zero while idle).
//!
//! The [`EnergyGate`] is the classic two-stage keyword-spotting design: a cheap
//! short-term-energy (RMS) test decides whether a frame is even *plausibly*
//! speech before paying for rustpotter. During silence the expensive path is
//! skipped, so idle CPU drops roughly in proportion to the fraction of time the
//! room is quiet. When there is speech-level energy the gate is fully open and
//! rustpotter sees an unbroken stream, so detection accuracy is unchanged.
//!
//! # Requirements the gate must satisfy
//!
//! 1. **Never miss a wake word.** A quiet onset must still reach rustpotter, and
//!    rustpotter must see a *contiguous* window over the whole utterance. This
//!    is met by (a) a pre-roll: the gate keeps the last [`PREROLL_SECS`] of
//!    frames while closed and flushes them into rustpotter the instant it opens,
//!    so the leading phoneme is never clipped; and (b) a hangover: once energy
//!    drops the gate keeps feeding for [`HANGOVER_SECS`] before closing, which
//!    covers the trailing silence a non-eager rustpotter needs to confirm a
//!    score peak (it fires *after* the word, on the fall-back).
//! 2. **Self-tuning across rooms.** A fixed threshold is wrong for every mic. An
//!    adaptive noise floor (see [`EnergyGate::update_floor`]) tracks the ambient
//!    level so the open/close decision is relative to the room, not absolute.
//! 3. **Fail open, never closed.** Every uncertain case errs toward feeding
//!    rustpotter: the floor starts at its minimum (so the gate is wide open at
//!    startup until it has learned the room), and the whole gate is opt-out via
//!    config (`wake_word.energy_gate = false`) if it ever misbehaves.
//! 4. **Negligible cost.** The gate is one pass of multiply-accumulate plus a
//!    sqrt per frame — trivial next to rustpotter's FFT — so a *closed* frame is
//!    effectively free.
//!
//! # The math
//!
//! For a frame of `N` samples `x[0..N]` (f32, roughly in `[-1, 1]`), the
//! short-term energy is the root-mean-square amplitude:
//!
//! ```text
//!     rms = sqrt( (1/N) * Σ x[i]² )
//! ```
//!
//! Thresholds are expressed as multiples of the adaptive floor, which is the
//! natural way to talk about a signal-to-noise margin (a ratio in linear
//! amplitude is a fixed dB offset: `dB = 20·log10(ratio)`):
//!
//! - open when `rms ≥ floor · OPEN_RATIO`   (`OPEN_RATIO = 4` ⇒ +12 dB over floor)
//! - close when `rms < floor · CLOSE_RATIO` (`CLOSE_RATIO = 2` ⇒ +6 dB over floor)
//!
//! The gap between the two (12 dB open vs 6 dB close) is hysteresis: it stops the
//! gate chattering on energy that hovers right at the threshold.
//!
//! The floor is an exponential moving average with asymmetric time constants:
//!
//! ```text
//!     floor ← floor + α·(rms − floor),   α = 1 − exp(−Δt / τ)
//! ```
//!
//! where `Δt` is the frame duration (`samples_per_frame / SAMPLE_RATE`). A slow
//! attack (`τ = FLOOR_UP_TAU_SECS`, ~10 s) means a few seconds of speech barely
//! moves the floor, so the floor learns *ambient*, not speech; a fast release
//! (`τ = FLOOR_DOWN_TAU_SECS`, ~0.5 s) means any inflation from a long utterance
//! is undone in the inter-utterance gap. The floor is clamped to
//! `[FLOOR_MIN, FLOOR_MAX]` so it can neither collapse to zero in perfect
//! digital silence (which would make any sample "loud") nor run away so high in
//! sustained noise that genuine speech can't clear it.

use std::collections::VecDeque;

use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::wake::WakeWordDetector;
use rustpotter::{Rustpotter, RustpotterConfig, SampleFormat};
use std::path::Path;

/// Hard lower bound for the live wake sensitivity (rustpotter's score
/// threshold). Below this even ambient noise clears the bar (constant false
/// wakes), so a calibrated or hand-set cutoff is clamped here.
pub const MIN_SENSITIVITY: f32 = 0.10;
/// Hard upper bound for the live wake sensitivity. At/above this essentially no
/// real utterance scores high enough to fire, so the cutoff is clamped here.
pub const MAX_SENSITIVITY: f32 = 0.95;
/// Threshold used *only while calibrating*. Very low, so rustpotter forms a
/// scoring partial for any speech (even a weak match) and its running peak can
/// be read via `get_partial_detection` — calibration measures that peak directly
/// rather than waiting for a non-eager "fire", which never happens when the
/// threshold is below the score's noise floor.
const CALIBRATION_THRESHOLD: f32 = 0.01;

/// Open the gate at +12 dB over the noise floor (`20·log10(4) ≈ 12.0 dB`).
const OPEN_RATIO: f32 = 4.0;
/// Close the gate below +6 dB over the noise floor (`20·log10(2) ≈ 6.0 dB`); the
/// 6 dB gap to [`OPEN_RATIO`] is the hysteresis that prevents chatter.
const CLOSE_RATIO: f32 = 2.0;
/// Floor attack time constant: how slowly the floor rises toward a louder level.
/// Long, so a multi-second utterance barely lifts it (the floor tracks ambient,
/// not speech) and so startup converges to the room over a few seconds.
const FLOOR_UP_TAU_SECS: f32 = 10.0;
/// Floor release time constant: how quickly the floor falls toward a quieter
/// level. Short, so any inflation a long utterance caused is recovered within
/// the gap before the next utterance.
const FLOOR_DOWN_TAU_SECS: f32 = 0.5;
/// Floor lower clamp ≈ −60 dBFS (`20·log10(1e-3)`). Stops the floor collapsing
/// to zero in pure digital silence, which would make the gate open on a single
/// stray sample. With this floor the gate opens no lower than ~−48 dBFS.
const FLOOR_MIN: f32 = 1e-3;
/// Floor upper clamp ≈ −30 dBFS (`20·log10(3e-2)`). Stops a runaway floor in
/// sustained loud noise from rising so high that real speech can't clear the
/// open ratio. In an environment loud enough to hit this, disable the gate.
const FLOOR_MAX: f32 = 3e-2;
/// Pre-roll kept while the gate is closed and flushed to rustpotter on open, so
/// a quiet word onset just before the energy crosses the open threshold is not
/// clipped. ~300 ms mirrors the pipeline's same-breath wake pre-buffer.
const PREROLL_SECS: f32 = 0.30;
/// Hangover: keep feeding rustpotter this long after energy drops below the
/// close threshold before actually closing. Must comfortably exceed the trailing
/// silence a non-eager rustpotter needs to confirm a score peak (it fires on the
/// fall-back *after* the word), and bridge brief intra-phrase dips.
const HANGOVER_SECS: f32 = 0.60;

/// Root-mean-square amplitude of one frame — the cheap short-term-energy measure
/// the gate decides on. Returns 0 for an empty frame. See the module math note.
fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

/// Energy gate that sits in front of rustpotter and decides, per frame, whether
/// to run the expensive detector. Holds the adaptive noise floor, the open/hangover
/// state, and the pre-roll ring. Pure and rustpotter-free so the decision logic
/// is unit-tested directly (the risky part of the change).
struct EnergyGate {
    /// True while the gate is open and frames flow straight to rustpotter.
    open: bool,
    /// Adaptive ambient-noise estimate in RMS amplitude (see [`Self::update_floor`]).
    floor: f32,
    /// Frames left to keep feeding after energy dropped, before closing.
    hangover_left: usize,
    /// Recent frames retained while closed, flushed on open as the onset pre-roll.
    preroll: VecDeque<Vec<f32>>,
    /// Per-frame EMA coefficient for a rising floor (derived from [`FLOOR_UP_TAU_SECS`]).
    up_alpha: f32,
    /// Per-frame EMA coefficient for a falling floor (derived from [`FLOOR_DOWN_TAU_SECS`]).
    down_alpha: f32,
    /// Hangover length in frames (derived from [`HANGOVER_SECS`]).
    hangover_frames: usize,
    /// Pre-roll capacity in frames (derived from [`PREROLL_SECS`]).
    preroll_frames: usize,
}

impl EnergyGate {
    /// Build a gate for rustpotter's frame size. The duration-based constants are
    /// converted to per-frame quantities here, where the frame duration
    /// (`samples_per_frame / SAMPLE_RATE`) is known, so the tuning stays
    /// expressed in seconds/dB regardless of rustpotter's internal frame length.
    fn new(samples_per_frame: usize) -> Self {
        let frame_secs = samples_per_frame as f32 / SAMPLE_RATE as f32;
        // α = 1 − exp(−Δt/τ): the standard one-pole EMA coefficient for a given
        // time constant at this frame rate.
        let up_alpha = 1.0 - (-frame_secs / FLOOR_UP_TAU_SECS).exp();
        let down_alpha = 1.0 - (-frame_secs / FLOOR_DOWN_TAU_SECS).exp();
        // Convert the duration windows to a frame count via integer ceil-division
        // in the sample domain, so the windows are never shorter than the spec'd
        // duration and an exact ratio (e.g. 300 ms / 30 ms) isn't bumped to the
        // next frame by float-`ceil` rounding error.
        let hangover_frames =
            ((HANGOVER_SECS * SAMPLE_RATE as f32) as usize).div_ceil(samples_per_frame);
        let preroll_frames =
            ((PREROLL_SECS * SAMPLE_RATE as f32) as usize).div_ceil(samples_per_frame);
        Self {
            open: false,
            // Start at the minimum so the gate is wide open until it has learned
            // the room — fail-open, never clipping a wake word at startup.
            floor: FLOOR_MIN,
            hangover_left: 0,
            preroll: VecDeque::with_capacity(preroll_frames + 1),
            up_alpha,
            down_alpha,
            hangover_frames,
            preroll_frames,
        }
    }

    /// Update the adaptive noise floor toward this frame's RMS with an asymmetric
    /// one-pole EMA — slow when rising (so speech doesn't inflate it), fast when
    /// falling (so it recovers quickly) — then clamp to `[FLOOR_MIN, FLOOR_MAX]`.
    fn update_floor(&mut self, rms: f32) {
        let alpha = if rms > self.floor {
            self.up_alpha
        } else {
            self.down_alpha
        };
        self.floor += alpha * (rms - self.floor);
        self.floor = self.floor.clamp(FLOOR_MIN, FLOOR_MAX);
    }

    /// Drop the oldest pre-roll frames until the ring is within capacity.
    fn trim_preroll(&mut self) {
        while self.preroll.len() > self.preroll_frames {
            self.preroll.pop_front();
        }
    }

    /// Admit one rustpotter-sized frame. Returns `(reset, frames)` where `frames`
    /// are the frames to run rustpotter on, oldest-first (empty = skip this
    /// frame, the CPU saving), and `reset` asks the caller to clear rustpotter's
    /// window because the gate just closed (so the next utterance starts from a
    /// clean window seeded only by its own pre-roll).
    ///
    /// Four cases:
    /// - **closed, quiet** → buffer the frame as pre-roll, skip rustpotter.
    /// - **closed, loud**  → open; flush the pre-roll (onset context) plus this
    ///   frame to rustpotter.
    /// - **open**          → feed this frame; re-arm the hangover while energy
    ///   stays up, count it down once energy drops.
    /// - **open, hangover elapsed** → close; ask for a rustpotter reset.
    fn admit(&mut self, frame: Vec<f32>) -> (bool, Vec<Vec<f32>>) {
        let rms = frame_rms(&frame);
        self.update_floor(rms);

        if self.open {
            if rms >= self.floor * CLOSE_RATIO {
                // Still speech-level: keep open and re-arm the hangover.
                self.hangover_left = self.hangover_frames;
                (false, vec![frame])
            } else if self.hangover_left > 0 {
                // Quiet, but inside the hangover — keep feeding the tail so a
                // non-eager detector can still confirm its peak.
                self.hangover_left -= 1;
                (false, vec![frame])
            } else {
                // Hangover elapsed: close, reset rustpotter, and start a fresh
                // pre-roll with this frame as its most recent context.
                self.open = false;
                self.preroll.clear();
                self.preroll.push_back(frame);
                self.trim_preroll();
                (true, Vec::new())
            }
        } else if rms >= self.floor * OPEN_RATIO {
            // Onset: open and flush the buffered pre-roll plus this frame so the
            // leading phonemes reach rustpotter as a contiguous block. The history
            // was already capped to `preroll_frames` while closed; the onset frame
            // is flushed in addition to it (so up to `preroll_frames + 1` frames),
            // hence no trim here.
            self.open = true;
            self.hangover_left = self.hangover_frames;
            self.preroll.push_back(frame);
            (false, self.preroll.drain(..).collect())
        } else {
            // Quiet: retain as pre-roll and skip the expensive detector.
            self.preroll.push_back(frame);
            self.trim_preroll();
            (false, Vec::new())
        }
    }
}

pub struct RustpotterWakeWordDetector {
    rustpotter: Rustpotter,
    /// The detector config we built rustpotter from, kept so the sensitivity
    /// (`detector.threshold`) can be poked at runtime via
    /// [`Rustpotter::update_detector_config`] instead of rebuilding the whole
    /// detector. Always holds the *normal-operation* settings (notably the
    /// user's `eager`); calibration drives rustpotter from a temporary config
    /// and restores from this one.
    config: RustpotterConfig,
    /// Exact number of samples rustpotter consumes per `process_samples` call.
    /// `process_samples` returns `None` (a silent no-op) for ANY other length and
    /// does not buffer across calls, so we must hand it exactly one frame.
    samples_per_frame: usize,
    /// Carries leftover samples between `detect()` calls so arbitrary capture
    /// chunk sizes (the daemon sends 20 ms / 320-sample chunks) are re-framed
    /// into rustpotter's frame size instead of being dropped on the floor (#44).
    buf: Vec<f32>,
    /// Optional energy gate (`wake_word.energy_gate`). `None` restores the
    /// original always-run behaviour; `Some` skips rustpotter on silent frames.
    gate: Option<EnergyGate>,
    /// True while a calibration session is in progress. Used to guard
    /// `set_sensitivity` so a config reload can't clobber the calibration
    /// threshold mid-session.
    calibrating: bool,
    /// Running peak match score for the current calibration utterance — the max
    /// of every fire / partial score seen since the last [`Self::clear_peak`].
    /// Read by [`Self::peak_score`].
    calib_peak: f32,
    /// The `eager` + `threshold` settings saved at [`Self::begin_calibration`],
    /// so [`Self::cancel_calibration`] can restore them if calibration is aborted
    /// (calibration mutates the stored `DetectorConfig` in place — it isn't
    /// `Clone` — driving a low, non-eager measurement config meanwhile).
    saved_eager: bool,
    saved_threshold: f32,
}

/// Pop one complete `frame`-sized chunk from the front of `buf`, retaining the
/// sub-frame remainder for the next call; returns `None` when there isn't yet a
/// full frame. Pure framing logic, kept free of rustpotter so it is unit-tested
/// directly — this is the fix for #44 (rustpotter silently drops any input that
/// isn't exactly one frame).
fn take_frame(buf: &mut Vec<f32>, frame: usize) -> Option<Vec<f32>> {
    if frame == 0 || buf.len() < frame {
        return None;
    }
    let out: Vec<f32> = buf[..frame].to_vec();
    buf.drain(..frame);
    Some(out)
}

impl RustpotterWakeWordDetector {
    /// `eager`: when true, fire the moment `min_scores` partial frames clear the
    /// threshold instead of waiting for the score to peak and fall back below it.
    /// This trims the ~2 s tail latency of the default (non-eager) detector at
    /// the cost of a higher false-trigger risk — tune with `sensitivity` (#50).
    ///
    /// `energy_gate`: when true, run frames through an [`EnergyGate`] so the
    /// expensive rustpotter path is skipped during silence (lower idle CPU). The
    /// gate is fail-open and accuracy-neutral when there is speech; set false to
    /// restore the original always-run behaviour.
    pub fn new(
        model_path: &Path,
        sensitivity: f32,
        eager: bool,
        energy_gate: bool,
    ) -> Result<Self, VoiceError> {
        let mut config = RustpotterConfig::default();
        config.fmt.sample_rate = SAMPLE_RATE as usize;
        config.fmt.sample_format = SampleFormat::F32;
        config.fmt.channels = 1;
        config.detector.threshold = sensitivity.clamp(MIN_SENSITIVITY, MAX_SENSITIVITY);
        // Fire as soon as `min_scores` partials clear the threshold rather than at
        // the END of the utterance (score peak → fall-back), shaving the wake→
        // listen latency. Off by default in rustpotter; we make it a config knob
        // because eager trades latency for a higher false-trigger risk (#50).
        config.detector.eager = eager;
        // Disable the averaged-score pre-gate (rustpotter default 0.2). It sits IN
        // FRONT of the per-frame `threshold`, so whenever the windowed average dips
        // below it a real wake word is silently dropped even though individual
        // frames clear `threshold`. We gate purely on `threshold` + `min_scores`.
        config.detector.avg_threshold = 0.0;
        // Deliberately leave the gain-normalizer DISABLED. rustpotter's MFCC is
        // already level-tolerant, and enabling the normalizer *destroys* live
        // detection: on quieter-than-training speech it amplifies the input and the
        // match collapses to ~0 (measured: same utterance scores 0.46 with the
        // normalizer off, 0.0 with it on at any max_gain). See #44.

        let mut rustpotter = Rustpotter::new(&config)
            .map_err(|e| VoiceError::WakeWord(format!("failed to create rustpotter: {e}")))?;

        let model_str = model_path.to_string_lossy();
        rustpotter
            .add_wakeword_from_file("hey-adele", &model_str)
            .map_err(|e| VoiceError::WakeWord(format!("failed to load wake word model: {e}")))?;

        let samples_per_frame = rustpotter.get_samples_per_frame();

        tracing::info!(
            model = %model_path.display(),
            sensitivity,
            eager,
            energy_gate,
            samples_per_frame,
            "wake word detector initialized"
        );

        Ok(Self {
            rustpotter,
            config,
            samples_per_frame,
            buf: Vec::new(),
            gate: energy_gate.then(|| EnergyGate::new(samples_per_frame)),
            calibrating: false,
            calib_peak: 0.0,
            saved_eager: eager,
            saved_threshold: sensitivity.clamp(MIN_SENSITIVITY, MAX_SENSITIVITY),
        })
    }

    /// Run rustpotter on exactly one frame, logging and reporting a detection.
    /// Takes the frame by value and hands it straight to `process_samples`, which
    /// accepts `&[f32]`-equivalent owned samples directly — no LE-byte round-trip
    /// (the old `process_bytes` path serialized each f32 to bytes only for
    /// rustpotter to decode them straight back).
    fn run_frame(&mut self, frame: Vec<f32>) -> bool {
        if let Some(detection) = self.rustpotter.process_samples::<f32>(frame) {
            // Log the score so the threshold can be tuned from real fires.
            tracing::info!(
                score = detection.score,
                avg_score = detection.avg_score,
                gain = detection.gain,
                "wake word detected"
            );
            true
        } else {
            false
        }
    }

    /// The currently-configured sensitivity (rustpotter's live score threshold).
    /// Mainly for tests and diagnostics.
    pub fn sensitivity(&self) -> f32 {
        self.config.detector.threshold
    }
}

impl WakeWordDetector for RustpotterWakeWordDetector {
    fn detect(&mut self, samples: &[f32]) -> Result<bool, VoiceError> {
        // Re-frame: rustpotter consumes EXACTLY `samples_per_frame` samples per
        // call and silently no-ops otherwise (#44), so accumulate and feed it one
        // frame at a time, keeping the sub-frame remainder for the next call.
        self.buf.extend_from_slice(samples);

        let mut detected = false;
        let spf = self.samples_per_frame;
        while let Some(frame) = take_frame(&mut self.buf, spf) {
            if self.gate.is_some() {
                // The gate decides which frames are worth rustpotter's cost. Its
                // borrow ends with `admit` (it returns owned data), freeing the
                // subsequent `self.rustpotter` / `self.run_frame` borrows.
                let (reset, frames) = self.gate.as_mut().unwrap().admit(frame);
                if reset {
                    self.rustpotter.reset();
                }
                for f in frames {
                    if self.run_frame(f) {
                        detected = true;
                    }
                }
            } else if self.run_frame(frame) {
                detected = true;
            }
        }
        Ok(detected)
    }

    /// Apply a new sensitivity to the *running* detector with no rebuild, by
    /// mutating rustpotter's score threshold via
    /// [`Rustpotter::update_detector_config`].
    ///
    /// This keeps rustpotter's own threshold equal to our effective cutoff, so
    /// firing semantics are unchanged in BOTH modes — including eager, where the
    /// detector fires the instant `min_scores` partials clear the threshold and
    /// then `reset()`s. (A "low catch-all threshold + post-filter on
    /// `detection.score`" design would break eager: it would fire-and-reset on an
    /// early sub-cutoff partial and could miss the true peak in the post-reset
    /// re-buffering window.) `update_detector_config` clears rustpotter's window,
    /// which is harmless between utterances; the energy gate's learned floor is
    /// left intact.
    fn set_sensitivity(&mut self, sensitivity: f32) -> Result<(), VoiceError> {
        let clamped = sensitivity.clamp(MIN_SENSITIVITY, MAX_SENSITIVITY);
        self.config.detector.threshold = clamped;
        if !self.calibrating {
            self.rustpotter
                .update_detector_config(&self.config.detector);
        }
        tracing::info!(sensitivity = clamped, "wake sensitivity applied live");
        Ok(())
    }

    /// Enter calibration: drive rustpotter at a very low threshold so a scoring
    /// partial forms for any speech and its running peak can be read via
    /// [`Self::peak_score`]. (We don't rely on the detector "firing": at a
    /// threshold below the score noise floor a non-eager detector never falls
    /// back, so we read the partial's peak directly instead.)
    fn begin_calibration(&mut self) {
        self.calibrating = true;
        self.saved_eager = self.config.detector.eager;
        self.saved_threshold = self.config.detector.threshold;
        // Drive rustpotter at the low, non-eager calibration settings. We mutate
        // the stored detector config in place (it isn't `Clone`); `end_calibration`
        // overwrites `eager`/`threshold` with the chosen result, and
        // `cancel_calibration` restores the saved values.
        self.config.detector.eager = false;
        self.config.detector.threshold = CALIBRATION_THRESHOLD;
        self.rustpotter
            .update_detector_config(&self.config.detector);
        self.clear_peak();
    }

    /// Feed audio and return the running peak match score for the current
    /// utterance (the max of any fire score and the live partial score seen
    /// since the last [`Self::clear_peak`]). Returns `None` until something has
    /// scored. The energy gate is not consulted — calibration always wants every
    /// frame scored.
    fn peak_score(&mut self, samples: &[f32]) -> Option<f32> {
        self.buf.extend_from_slice(samples);
        let spf = self.samples_per_frame;
        while let Some(frame) = take_frame(&mut self.buf, spf) {
            // A fire returns the peak and resets the window; otherwise the live
            // partial holds the running max. Track the max of both so the peak
            // survives even if the detector happens to fire mid-utterance.
            if let Some(d) = self.rustpotter.process_samples::<f32>(frame) {
                self.calib_peak = self.calib_peak.max(d.score);
            } else if let Some(d) = self.rustpotter.get_partial_detection() {
                self.calib_peak = self.calib_peak.max(d.score);
            }
        }
        (self.calib_peak > 0.0).then_some(self.calib_peak)
    }

    /// Reset the running peak and rustpotter's window before the next utterance,
    /// and drop any buffered sub-frame audio so a new utterance starts clean.
    fn clear_peak(&mut self) {
        self.calib_peak = 0.0;
        self.buf.clear();
        self.rustpotter.reset();
    }

    /// Leave calibration mode and apply the calibrated result: set the wake mode
    /// to `eager` and the cutoff to `sensitivity`, pushed to rustpotter in one
    /// `update_detector_config` (via `set_sensitivity`, now that `calibrating` is
    /// false again).
    fn end_calibration(&mut self, sensitivity: f32, eager: bool) -> Result<(), VoiceError> {
        self.calibrating = false;
        self.calib_peak = 0.0;
        self.config.detector.eager = eager;
        self.set_sensitivity(sensitivity)
    }

    /// Abort calibration: restore the `eager`/`threshold` saved at
    /// `begin_calibration` so the detector is exactly as it was before.
    fn cancel_calibration(&mut self) {
        self.calibrating = false;
        self.calib_peak = 0.0;
        self.config.detector.eager = self.saved_eager;
        // set_sensitivity re-clamps + pushes the restored config to rustpotter.
        let _ = self.set_sensitivity(self.saved_threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- #44: re-framing arbitrary capture chunks into rustpotter frames ---

    #[test]
    fn sub_frame_chunk_yields_no_frame_but_is_retained() {
        let frame = 480;
        let mut buf = vec![0.0f32; 320]; // one 20 ms capture chunk
        assert!(take_frame(&mut buf, frame).is_none());
        assert_eq!(
            buf.len(),
            320,
            "sub-frame input must be retained, not dropped"
        );
    }

    #[test]
    fn frames_assemble_across_multiple_sub_frame_chunks() {
        let frame = 480;
        let mut buf: Vec<f32> = Vec::new();

        // Two 320-sample chunks = 640 samples => exactly one 480 frame + 160 left.
        buf.extend_from_slice(&[1.0; 320]);
        assert!(take_frame(&mut buf, frame).is_none());
        buf.extend_from_slice(&[1.0; 320]);

        let f = take_frame(&mut buf, frame).expect("a full frame should be available");
        assert_eq!(f.len(), frame);
        assert_eq!(buf.len(), 160, "remainder is kept for the next call");
        assert!(take_frame(&mut buf, frame).is_none());
    }

    #[test]
    fn multiple_whole_frames_drain_in_order() {
        let frame = 4;
        let mut buf: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 2 frames + 2 left

        assert_eq!(take_frame(&mut buf, frame), Some(vec![0.0, 1.0, 2.0, 3.0]));
        assert_eq!(take_frame(&mut buf, frame), Some(vec![4.0, 5.0, 6.0, 7.0]));
        assert!(take_frame(&mut buf, frame).is_none());
        assert_eq!(buf, vec![8.0, 9.0]);
    }

    #[test]
    fn zero_frame_size_never_loops() {
        let mut buf = vec![0.0f32; 10];
        assert!(take_frame(&mut buf, 0).is_none());
        assert_eq!(buf.len(), 10);
    }

    // --- energy gate ---

    /// rustpotter's real frame size at 16 kHz, so the derived per-frame windows
    /// (pre-roll/hangover) in these tests match production.
    const SPF: usize = 480; // 30 ms at 16 kHz
    /// A frame whose every sample equals `amp`, so its RMS is exactly `amp`.
    fn frame_at(amp: f32) -> Vec<f32> {
        vec![amp; SPF]
    }

    #[test]
    fn rms_of_constant_frame_is_its_amplitude() {
        // sqrt(mean(a²)) == |a| for a constant frame — the property the gate tests rely on.
        assert!((frame_rms(&frame_at(0.1)) - 0.1).abs() < 1e-6);
        assert!((frame_rms(&frame_at(0.0)) - 0.0).abs() < 1e-6);
        assert_eq!(frame_rms(&[]), 0.0);
    }

    #[test]
    fn gate_derives_sane_windows_from_frame_size() {
        let g = EnergyGate::new(SPF); // 30 ms frames
        // 0.30 s / 0.03 s = 10 frames; 0.60 s / 0.03 s = 20 frames.
        assert_eq!(g.preroll_frames, 10, "~300 ms pre-roll");
        assert_eq!(g.hangover_frames, 20, "~600 ms hangover");
        // EMA coefficients are in (0,1) and attack is far slower than release.
        assert!(g.up_alpha > 0.0 && g.up_alpha < g.down_alpha && g.down_alpha < 1.0);
    }

    #[test]
    fn silence_keeps_gate_closed_and_skips_rustpotter() {
        // Frames at the floor minimum never clear the open ratio, so every frame
        // is skipped — this is the idle-CPU saving.
        let mut g = EnergyGate::new(SPF);
        for _ in 0..200 {
            let (reset, frames) = g.admit(frame_at(FLOOR_MIN));
            assert!(!reset);
            assert!(frames.is_empty(), "silent frames must skip rustpotter");
        }
        assert!(!g.open);
    }

    #[test]
    fn loud_onset_opens_and_flushes_preroll_oldest_first() {
        let mut g = EnergyGate::new(SPF);
        // Five quiet frames buffer as pre-roll without running rustpotter.
        for _ in 0..5 {
            let (_, frames) = g.admit(frame_at(FLOOR_MIN));
            assert!(frames.is_empty());
        }
        // A clearly loud frame (well over +12 dB) opens the gate.
        let (reset, frames) = g.admit(frame_at(0.2));
        assert!(!reset);
        assert!(g.open, "loud onset must open the gate");
        // The 5 buffered quiet frames PLUS the loud onset frame are flushed, so
        // rustpotter sees the full onset as a contiguous block.
        assert_eq!(frames.len(), 6, "pre-roll + onset flushed together");
        assert!(
            (frame_rms(frames.last().unwrap()) - 0.2).abs() < 1e-6,
            "the onset frame is flushed last (most recent)"
        );
        assert!(
            frames[..5].iter().all(|f| frame_rms(f) <= FLOOR_MIN + 1e-6),
            "the quiet pre-roll precedes the onset"
        );
    }

    #[test]
    fn preroll_is_capped_to_its_window() {
        // Far more quiet frames than the pre-roll holds: only the most recent
        // `preroll_frames` (+ the onset) survive to the flush.
        let mut g = EnergyGate::new(SPF);
        for _ in 0..100 {
            g.admit(frame_at(FLOOR_MIN));
        }
        let (_, frames) = g.admit(frame_at(0.2));
        assert_eq!(
            frames.len(),
            g.preroll_frames + 1,
            "pre-roll is bounded; only recent context plus the onset is flushed"
        );
    }

    #[test]
    fn open_gate_feeds_then_closes_after_hangover_and_requests_reset() {
        let mut g = EnergyGate::new(SPF);
        g.admit(frame_at(0.2)); // open
        assert!(g.open);
        // While quiet but within the hangover, each frame is still fed (so a
        // non-eager detector can confirm its peak on the trailing silence).
        for i in 0..g.hangover_frames {
            let (reset, frames) = g.admit(frame_at(FLOOR_MIN));
            assert!(!reset, "still feeding during hangover (frame {i})");
            assert_eq!(frames.len(), 1, "the tail frame is fed (frame {i})");
        }
        // The frame after the hangover elapses closes the gate and asks for a reset.
        let (reset, frames) = g.admit(frame_at(FLOOR_MIN));
        assert!(reset, "gate close must reset rustpotter's window");
        assert!(frames.is_empty(), "nothing is run on the closing frame");
        assert!(!g.open);
    }

    #[test]
    fn ongoing_speech_re_arms_the_hangover_so_it_never_closes_mid_word() {
        let mut g = EnergyGate::new(SPF);
        g.admit(frame_at(0.2)); // open
        // Alternate quiet/loud well within the hangover: the loud frames keep
        // re-arming it, so the gate must stay open throughout (no reset).
        for _ in 0..(g.hangover_frames * 3) {
            let (reset_q, _) = g.admit(frame_at(FLOOR_MIN));
            assert!(!reset_q);
            let (reset_l, frames_l) = g.admit(frame_at(0.2));
            assert!(!reset_l);
            assert_eq!(frames_l.len(), 1);
        }
        assert!(
            g.open,
            "speech that keeps returning must hold the gate open"
        );
    }

    #[test]
    fn floor_adapts_so_steady_ambient_noise_eventually_gates() {
        // Steady moderate noise opens the gate at first (floor starts at the
        // minimum) but, as the floor climbs to the ambient level, the same noise
        // no longer clears the open ratio and the gate closes — the self-tuning
        // requirement.
        let mut g = EnergyGate::new(SPF);
        let ambient = 8e-3; // ~−42 dBFS, between FLOOR_MIN and FLOOR_MAX
        let (_, first) = g.admit(frame_at(ambient));
        assert!(
            !first.is_empty(),
            "before the floor has learned the room, ambient must fail OPEN (fed)"
        );
        // Let the floor converge (well past the ~10 s attack constant).
        for _ in 0..600 {
            g.admit(frame_at(ambient));
        }
        let (_, settled) = g.admit(frame_at(ambient));
        assert!(
            settled.is_empty() && !g.open,
            "once the floor tracks ambient, steady noise is gated out (CPU saved)"
        );
        assert!(g.floor > ambient / OPEN_RATIO, "floor rose toward ambient");
    }

    #[test]
    fn floor_never_collapses_below_the_minimum() {
        // Pure digital silence must not drive the floor to zero (which would make
        // the gate open on any nonzero sample).
        let mut g = EnergyGate::new(SPF);
        for _ in 0..1000 {
            g.admit(frame_at(0.0));
        }
        assert!(g.floor >= FLOOR_MIN, "floor is clamped at its minimum");
    }
}
