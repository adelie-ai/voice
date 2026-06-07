pub mod domain;
pub mod error;
pub mod ports;
pub mod sentence_buffer;
pub mod speech_text;

pub use error::VoiceError;
pub use speech_text::strip_markdown_for_speech;
