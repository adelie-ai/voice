use crate::VoiceError;

/// Outbound port for voice activity detection.
pub trait VoiceActivityDetector: Send + Sync {
    /// Feed audio samples and return the speech probability (0.0 to 1.0).
    fn speech_probability(&mut self, samples: &[f32]) -> Result<f32, VoiceError>;

    /// Reset internal state (call between utterances).
    fn reset(&mut self);
}
