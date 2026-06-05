//! Runtime-selectable TTS backend. `tts.backend` in config picks one at
//! startup. Cloning shares each backend's voice state (an `Arc`), so a
//! `SetVoice` reaches every consumer (the conversation pipeline and the
//! on-demand SayText service alike). Piper and Kokoro are local; Polly is cloud
//! (neural / generative voices). The enum keeps a [`Speaker`](crate::Speaker)
//! monomorphic over a single `T` while still letting the backend be chosen at
//! runtime.

use adele_voice_core::VoiceError;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_tts_kokoro::KokoroTts;
use adele_voice_tts_piper::PiperTts;
use adele_voice_tts_polly::PollyTts;

use crate::config::TtsConfig;

#[derive(Clone)]
pub enum TtsBackend {
    Piper(PiperTts),
    Polly(PollyTts),
    Kokoro(KokoroTts),
}

impl TtsBackend {
    /// Build the configured backend. Local-first: an unknown backend, or a
    /// Kokoro that can't initialize (missing model/voices), falls back to the
    /// local Piper backend — **never** to a billable cloud backend.
    pub async fn from_config(tts: &TtsConfig) -> TtsBackend {
        match tts.backend.as_str() {
            "polly" => {
                tracing::info!(
                    voice = %tts.polly_voice,
                    engine = %tts.polly_engine,
                    profile = tts.polly_profile.as_deref().unwrap_or(""),
                    "using AWS Polly TTS backend"
                );
                TtsBackend::Polly(
                    PollyTts::new(
                        &tts.polly_voice,
                        &tts.polly_engine,
                        tts.polly_region.clone(),
                        tts.polly_profile.clone(),
                    )
                    .await,
                )
            }
            "kokoro" => match KokoroTts::new(
                &tts.kokoro_model_path,
                &tts.kokoro_voices_dir,
                &tts.kokoro_voice,
                &tts.kokoro_lang,
            ) {
                Ok(k) => {
                    tracing::info!(voice = %tts.kokoro_voice, "using local Kokoro TTS backend");
                    TtsBackend::Kokoro(k)
                }
                Err(e) => {
                    tracing::warn!(
                        "Kokoro init failed ({e}); falling back to Piper. Run scripts/setup.sh to provision Kokoro."
                    );
                    TtsBackend::Piper(PiperTts::new(&tts.piper_binary, &tts.model_path))
                }
            },
            other => {
                if other != "piper" {
                    tracing::warn!(backend = %other, "unknown tts.backend, falling back to piper");
                }
                TtsBackend::Piper(PiperTts::new(&tts.piper_binary, &tts.model_path))
            }
        }
    }
}

impl TextToSpeech for TtsBackend {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
        match self {
            TtsBackend::Piper(t) => t.synthesize(text).await,
            TtsBackend::Polly(t) => t.synthesize(text).await,
            TtsBackend::Kokoro(t) => t.synthesize(text).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn piper_backend_is_selected() {
        let cfg = TtsConfig {
            backend: "piper".into(),
            ..TtsConfig::default()
        };
        assert!(matches!(
            TtsBackend::from_config(&cfg).await,
            TtsBackend::Piper(_)
        ));
    }

    #[tokio::test]
    async fn unknown_backend_falls_back_to_piper() {
        let cfg = TtsConfig {
            backend: "whisper-in-the-wind".into(),
            ..TtsConfig::default()
        };
        assert!(matches!(
            TtsBackend::from_config(&cfg).await,
            TtsBackend::Piper(_)
        ));
    }

    #[tokio::test]
    async fn kokoro_without_models_falls_back_to_piper_not_cloud() {
        // Local-first policy: a Kokoro that can't initialize must fall back to
        // the LOCAL Piper, never to a billable cloud backend.
        let cfg = TtsConfig {
            backend: "kokoro".into(),
            kokoro_model_path: "/nonexistent/kokoro.onnx".into(),
            kokoro_voices_dir: "/nonexistent/voices".into(),
            ..TtsConfig::default()
        };
        assert!(matches!(
            TtsBackend::from_config(&cfg).await,
            TtsBackend::Piper(_)
        ));
    }
}
