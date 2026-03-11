use crate::VoiceError;

/// Outbound port for text-to-speech synthesis.
pub trait TextToSpeech: Send + Sync {
    /// Synthesize text into PCM f32 audio samples.
    fn synthesize(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, VoiceError>> + Send;
}
