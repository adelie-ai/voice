use thiserror::Error;

#[derive(Debug, Error)]
pub enum VoiceError {
    #[error("audio error: {0}")]
    Audio(String),

    #[error("wake word detection error: {0}")]
    WakeWord(String),

    #[error("VAD error: {0}")]
    Vad(String),

    #[error("speech-to-text error: {0}")]
    Stt(String),

    #[error("text-to-speech error: {0}")]
    Tts(String),

    #[error("assistant communication error: {0}")]
    Assistant(String),

    #[error("configuration error: {0}")]
    Config(String),
}
