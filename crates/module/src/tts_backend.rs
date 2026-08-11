//! Runtime-selectable TTS backend. `tts.backend` in config picks one at
//! startup. Cloning shares each backend's voice state (an `Arc`), so a
//! `SetVoice` reaches every consumer (the conversation pipeline and the
//! on-demand SayText service alike). Piper and Kokoro are local; Polly is cloud
//! (neural / generative voices). The enum keeps a [`Speaker`](crate::Speaker)
//! monomorphic over a single `T` while still letting the backend be chosen at
//! runtime.
//!
//! Polly and Kokoro are also selectable at *build* time, by the `tts-polly` and
//! `tts-kokoro` features. Piper is always compiled in, because it is what the
//! selection falls back to. Asking for a backend this build does not contain is
//! a configuration mistake the operator can fix, so it is reported by name
//! rather than absorbed silently.

use adele_voice_core::VoiceError;
use adele_voice_core::ports::tts::TextToSpeech;
#[cfg(feature = "tts-kokoro")]
use adele_voice_tts_kokoro::KokoroTts;
use adele_voice_tts_piper::PiperTts;
#[cfg(feature = "tts-polly")]
use adele_voice_tts_polly::PollyTts;

use crate::config::TtsConfig;

/// The backends this build contains, in config-name form.
///
/// Used to say what *is* available when a configured backend is not, because
/// "polly is not compiled in" alone does not tell an operator what to set
/// instead.
pub const COMPILED_IN_BACKENDS: &[&str] = &[
    "piper",
    #[cfg(feature = "tts-polly")]
    "polly",
    #[cfg(feature = "tts-kokoro")]
    "kokoro",
];

#[derive(Clone)]
pub enum TtsBackend {
    Piper(PiperTts),
    #[cfg(feature = "tts-polly")]
    Polly(PollyTts),
    #[cfg(feature = "tts-kokoro")]
    Kokoro(KokoroTts),
}

impl TtsBackend {
    /// Build the configured backend. Local-first: an unknown backend, a backend
    /// this build does not contain, or a Kokoro that can't initialize (missing
    /// model/voices), falls back to the local Piper backend — **never** to a
    /// billable cloud backend.
    pub async fn from_config(tts: &TtsConfig) -> TtsBackend {
        match tts.backend.as_str() {
            "polly" => Self::polly(tts).await,
            "kokoro" => Self::kokoro(tts),
            other => {
                if other != "piper" {
                    tracing::warn!(backend = %other, "unknown tts.backend, falling back to piper");
                }
                Self::piper(tts)
            }
        }
    }

    fn piper(tts: &TtsConfig) -> TtsBackend {
        TtsBackend::Piper(PiperTts::new(&tts.piper_binary, &tts.model_path))
    }

    #[cfg(feature = "tts-polly")]
    async fn polly(tts: &TtsConfig) -> TtsBackend {
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

    #[cfg(not(feature = "tts-polly"))]
    async fn polly(tts: &TtsConfig) -> TtsBackend {
        report_not_compiled_in("polly", "tts-polly");
        Self::piper(tts)
    }

    #[cfg(feature = "tts-kokoro")]
    fn kokoro(tts: &TtsConfig) -> TtsBackend {
        match KokoroTts::new(
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
                Self::piper(tts)
            }
        }
    }

    #[cfg(not(feature = "tts-kokoro"))]
    fn kokoro(tts: &TtsConfig) -> TtsBackend {
        report_not_compiled_in("kokoro", "tts-kokoro");
        Self::piper(tts)
    }
}

/// Report a configured backend that this build does not contain.
///
/// At error level, and naming the feature that would add it, because nothing
/// the operator can see at runtime explains the difference otherwise: the
/// config says `kokoro`, the voice is Piper's, and no other line connects them.
///
/// A build with every backend selected has no caller for this, which is the
/// point: there is then no configured backend it can fail to contain.
#[cfg(any(not(feature = "tts-polly"), not(feature = "tts-kokoro")))]
fn report_not_compiled_in(backend: &str, feature: &str) {
    tracing::error!(
        backend,
        feature,
        available = ?COMPILED_IN_BACKENDS,
        "tts.backend is not compiled into this build; falling back to piper. \
         Rebuild with the named cargo feature, or configure an available backend."
    );
}

impl TextToSpeech for TtsBackend {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
        match self {
            TtsBackend::Piper(t) => t.synthesize(text).await,
            #[cfg(feature = "tts-polly")]
            TtsBackend::Polly(t) => t.synthesize(text).await,
            #[cfg(feature = "tts-kokoro")]
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

    #[cfg(feature = "tts-kokoro")]
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

    #[cfg(not(feature = "tts-kokoro"))]
    #[tokio::test]
    async fn kokoro_not_compiled_in_falls_back_to_piper() {
        let cfg = TtsConfig {
            backend: "kokoro".into(),
            ..TtsConfig::default()
        };
        assert!(matches!(
            TtsBackend::from_config(&cfg).await,
            TtsBackend::Piper(_)
        ));
    }

    #[cfg(not(feature = "tts-polly"))]
    #[tokio::test]
    async fn polly_not_compiled_in_falls_back_to_piper_not_a_panic() {
        let cfg = TtsConfig {
            backend: "polly".into(),
            ..TtsConfig::default()
        };
        assert!(matches!(
            TtsBackend::from_config(&cfg).await,
            TtsBackend::Piper(_)
        ));
    }

    #[test]
    fn piper_is_always_available_as_the_fallback() {
        // The fallback target must exist in every build, or `from_config` has
        // nowhere to land. This is what keeps `tts-piper` unconditional.
        assert!(COMPILED_IN_BACKENDS.contains(&"piper"));
    }

    #[cfg(feature = "tts-polly")]
    #[test]
    fn polly_is_listed_when_compiled_in() {
        assert!(COMPILED_IN_BACKENDS.contains(&"polly"));
    }

    #[cfg(not(feature = "tts-polly"))]
    #[test]
    fn polly_is_not_listed_when_absent() {
        assert!(!COMPILED_IN_BACKENDS.contains(&"polly"));
    }
}
