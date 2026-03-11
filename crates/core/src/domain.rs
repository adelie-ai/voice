mod state;

pub use state::{State, StateEvent};

/// Audio sample rate used throughout the pipeline (16kHz mono).
pub const SAMPLE_RATE: u32 = 16_000;

/// Number of audio channels (mono).
pub const CHANNELS: u16 = 1;

/// A chunk of PCM audio data (f32 mono samples at 16kHz).
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
}

impl AudioChunk {
    pub fn new(samples: Vec<f32>) -> Self {
        Self { samples }
    }

    /// Duration of this chunk in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE as f32
    }
}

/// Result of a speech-to-text transcription.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
}
