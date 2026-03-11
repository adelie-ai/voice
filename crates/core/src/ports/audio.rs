use crate::VoiceError;
use tokio::sync::mpsc;

/// Outbound port for capturing audio from a microphone.
pub trait AudioSource: Send + Sync {
    /// Start capturing audio. Returns a receiver that yields PCM f32 chunks.
    fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, VoiceError>;

    /// Stop capturing audio.
    fn stop(&self) -> Result<(), VoiceError>;
}

/// Outbound port for playing audio through speakers.
pub trait AudioSink: Send + Sync {
    /// Queue PCM f32 samples for playback.
    fn play(&self, samples: Vec<f32>) -> Result<(), VoiceError>;

    /// Stop any ongoing playback and clear the queue.
    fn stop(&self) -> Result<(), VoiceError>;

    /// Returns true if audio is currently being played.
    fn is_playing(&self) -> bool;
}
