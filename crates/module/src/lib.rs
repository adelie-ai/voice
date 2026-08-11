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
//!
//! # Selecting backends
//!
//! Which adapter crates are compiled in is a build-time choice. The `default`
//! feature set is every backend, so a consumer that says nothing gets the whole
//! set. A consumer opts out to keep a native dependency out of its graph:
//! `vad-silero` and `tts-kokoro` bring ONNX Runtime (`ort`), which has no
//! prebuilt binary for every target the clients run on.
//!
//! Opting out narrows the API rather than changing it. [`build_dictation`]
//! needs `vad-silero` and `stt-whisper`, because it names both adapters in its
//! return type; without them a consumer wires [`Dictation`] from its own
//! implementations of the `core` ports. [`build_speaker`] is always available:
//! the Piper backend is unconditional, so the TTS selection always has a local
//! backend to land on.

pub mod builders;
pub mod config;
mod dictation;
mod endpointer;
mod speaker;
mod transcriber;
mod tts_backend;

#[cfg(all(feature = "vad-silero", feature = "stt-whisper"))]
pub use builders::build_dictation;
pub use builders::build_speaker;
pub use config::{AudioConfig, SttConfig, TtsConfig, VadConfig};
pub use dictation::{Dictation, DictationOptions};
pub use endpointer::{Endpoint, Endpointer, PreBuffer};
pub use speaker::Speaker;
pub use transcriber::{MIN_SPEECH_RMS, Transcriber, rms};
pub use tts_backend::{COMPILED_IN_BACKENDS, TtsBackend};
