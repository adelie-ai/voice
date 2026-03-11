use crate::VoiceError;

/// Outbound port for wake word detection.
pub trait WakeWordDetector: Send + Sync {
    /// Feed audio samples and return whether the wake word was detected.
    fn detect(&mut self, samples: &[f32]) -> Result<bool, VoiceError>;
}
