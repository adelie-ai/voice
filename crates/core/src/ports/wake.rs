use crate::VoiceError;

/// Outbound port for wake word detection.
pub trait WakeWordDetector: Send {
    /// Feed audio samples and return whether the wake word was detected.
    fn detect(&mut self, samples: &[f32]) -> Result<bool, VoiceError>;

    /// Apply a new detection sensitivity (the score threshold) to the running
    /// detector, without rebuilding it. The default is a no-op so detectors that
    /// can't retune live simply ignore live sensitivity changes.
    ///
    /// Implementations should treat this as the live runtime knob the config
    /// reload (and calibration) drives — it must be safe in both eager and
    /// non-eager modes (see the rustpotter adapter, which keeps the detector's
    /// own threshold equal to the cutoff so eager firing stays correct).
    fn set_sensitivity(&mut self, _sensitivity: f32) -> Result<(), VoiceError> {
        Ok(())
    }

    /// After a [`Self::detect`] call that returned `true`, the match score of
    /// that fire — consumed on read (returns `None` until the next fire). Lets
    /// the caller label a wake by how strongly it matched, for online
    /// sensitivity adaptation (#121). `detect` reports only a bool, so the score
    /// rustpotter computes at the fire would otherwise be lost. Default: not
    /// tracked (`None`), so detectors that can't report a score simply opt out.
    fn take_last_fire_score(&mut self) -> Option<f32> {
        None
    }

    /// Enter a calibration session: the detector should run so that the *running
    /// peak match score* of an utterance can be read out via [`Self::peak_score`]
    /// (e.g. at a very low threshold so a partial always forms). Default no-op.
    fn begin_calibration(&mut self) {}

    /// During calibration, feed audio and return the running peak match score for
    /// the CURRENT utterance so far (the maximum score seen since the last
    /// [`Self::clear_peak`]), or `None` if nothing has scored yet. Unlike
    /// `detect`, this never depends on the detector "firing" — the caller decides
    /// when the utterance ends — so it works regardless of the score noise floor.
    /// Default: nothing to report.
    fn peak_score(&mut self, _samples: &[f32]) -> Option<f32> {
        None
    }

    /// Reset the running peak (and the detector's window) before the next
    /// calibration utterance. Default no-op.
    fn clear_peak(&mut self) {}

    /// Leave calibration mode and apply the calibrated result: `sensitivity` as
    /// the live cutoff and `eager` as the wake mode (calibration picks the best
    /// available mode). The default applies the sensitivity; detectors that
    /// support an eager mode also switch it.
    fn end_calibration(&mut self, sensitivity: f32, _eager: bool) -> Result<(), VoiceError> {
        self.set_sensitivity(sensitivity)
    }

    /// Abort calibration without applying a result — restore the detector to the
    /// settings it had before [`Self::begin_calibration`]. Default no-op.
    fn cancel_calibration(&mut self) {}
}
