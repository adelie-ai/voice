//! Embeddable voice module: on-demand dictation and speech playback.
//!
//! Factored out of the voice daemon so any client can do speech-to-text and
//! text-to-speech **in-process without the daemon** — no wake word, no D-Bus,
//! no orchestrator/assistant coupling. The daemon is itself a consumer of these
//! primitives; it layers the wake word, the continuous-capture loop, and the
//! `org.desktopAssistant.Voice` D-Bus surface on top (those stay daemon-only).
//!
//! - [`Dictation`] — open the mic, endpoint one utterance (Silero VAD), and
//!   transcribe it (Whisper). "Push-to-talk minus everything else."
//! - [`Speaker`] — synthesize text with the configured backend
//!   (Kokoro/Piper/Polly) and play it through an audio sink.
//!
//! The lower-level primitives the daemon shares are also public: [`Endpointer`]
//! (VAD endpointing) and [`Transcriber`] (energy-gate + STT). [`build_dictation`]
//! and [`build_speaker`] wire the concrete adapter crates from [`config`] for
//! embedding clients that don't need to share an audio device.

pub mod builders;
pub mod config;
mod dictation;
mod endpointer;
mod speaker;
mod transcriber;
mod tts_backend;

pub use builders::{build_dictation, build_speaker};
pub use config::{AudioConfig, SttConfig, TtsConfig, VadConfig};
pub use dictation::{Dictation, DictationOptions};
pub use endpointer::{Endpoint, Endpointer, PreBuffer};
pub use speaker::Speaker;
pub use transcriber::{MIN_SPEECH_RMS, Transcriber, rms};
pub use tts_backend::TtsBackend;
