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

    /// Enter a calibration session: the detector should report the *true peak
    /// score* of each spoken wake word (e.g. by running at a low threshold,
    /// non-eager) so a caller can measure the user's real scores. Default no-op.
    fn begin_calibration(&mut self) {}

    /// During calibration, return the peak score of the most recent detection
    /// (one utterance → one peak), consuming it so the next call reflects the
    /// next utterance. Default: nothing to report.
    fn take_last_score(&mut self) -> Option<f32> {
        None
    }

    /// Leave calibration mode and apply `sensitivity` as the new live cutoff.
    /// The default just applies the sensitivity (calibration was a no-op).
    fn end_calibration(&mut self, sensitivity: f32) -> Result<(), VoiceError> {
        self.set_sensitivity(sensitivity)
    }
}
