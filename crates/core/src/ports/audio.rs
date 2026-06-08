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

    /// Whether playback is in its *tail pad* — the queued audio's real-time
    /// duration has elapsed (the audio deadline has passed) but we're still
    /// inside the latency cushion that keeps `is_playing` true. During this
    /// window nothing fresh is sounding, so same-breath mic audio can be
    /// pre-buffered (without running wake detect) instead of dropped (#70).
    /// Defaults to `false` so sinks that don't model a pad opt out cleanly.
    fn in_tail_pad(&self) -> bool {
        false
    }
}
