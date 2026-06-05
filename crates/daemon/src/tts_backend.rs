//! Runtime-selectable TTS backend. `tts.backend` in config picks one at
//! startup. Cloning shares each backend's voice state (an `Arc`), so a
//! `SetVoice` reaches both the conversation pipeline and the SayText service.
//! Piper is local and the default; Polly is cloud (neural / generative AI
//! voices). The enum keeps the pipeline monomorphic over a single `T` while
//! still letting the backend be chosen at runtime.

use adele_voice_core::VoiceError;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_tts_piper::PiperTts;
use adele_voice_tts_polly::PollyTts;

#[derive(Clone)]
pub enum TtsBackend {
    Piper(PiperTts),
    Polly(PollyTts),
}

impl TextToSpeech for TtsBackend {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
        match self {
            TtsBackend::Piper(t) => t.synthesize(text).await,
            TtsBackend::Polly(t) => t.synthesize(text).await,
        }
    }
}
