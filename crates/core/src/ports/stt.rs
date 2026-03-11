use crate::VoiceError;
use crate::domain::Transcript;

/// Outbound port for speech-to-text transcription.
pub trait SpeechToText: Send + Sync {
    /// Transcribe PCM audio samples into text.
    fn transcribe(
        &self,
        samples: &[f32],
    ) -> impl std::future::Future<Output = Result<Transcript, VoiceError>> + Send;
}
