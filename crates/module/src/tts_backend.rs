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

/// The **TTS** backends this build contains, in `tts.backend` config-name form.
///
/// Used to say what *is* available when a configured backend is not, because
/// "polly is not compiled in" alone does not tell an operator what to set
/// instead. It describes the TTS selection only; VAD and STT have no
/// equivalent, because neither is chosen by config.
pub const COMPILED_IN_TTS_BACKENDS: &[&str] = &[
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
/// point: there is then no configured backend it can fail to contain. Hence
/// `allow(dead_code)` rather than a `cfg` naming every optional backend, which
/// would stop compiling the function while still compiling its callers the
/// next time a backend is added.
///
/// The `allow` costs the accidental signal that deleting a call site used to
/// give, so the report has a test of its own instead - see
/// `an_absent_backend_is_reported_at_error_level_with_backend_feature_and_available`.
#[allow(dead_code)]
fn report_not_compiled_in(backend: &str, feature: &str) {
    tracing::error!(
        backend,
        feature,
        available = ?COMPILED_IN_TTS_BACKENDS,
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
        assert!(COMPILED_IN_TTS_BACKENDS.contains(&"piper"));
    }

    /// A name on the listing must be one `from_config` recognizes. It cannot
    /// assert which backend comes back: a compiled-in backend still falls back
    /// to Piper when its models are absent, which is a provisioning state, not
    /// a build one. What it can assert is that no listed local name takes the
    /// unknown-backend path, which is the way a listing and the selection drift
    /// apart in practice.
    #[test]
    fn no_listed_local_backend_is_treated_as_unknown() {
        let unknown_warning = "unknown tts.backend";
        for backend in COMPILED_IN_TTS_BACKENDS {
            // Polly is excluded, not overlooked. Selecting it constructs a real
            // AWS client, which walks the credential chain and reaches IMDS on
            // a machine with no AWS config - a live network call in a unit test.
            // Its entry on the listing is covered by the two `polly_is_*` tests.
            if *backend == "polly" {
                continue;
            }
            let cfg = TtsConfig {
                backend: (*backend).into(),
                ..TtsConfig::default()
            };
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("current-thread runtime");
            let events =
                capture::while_capturing(|| _ = runtime.block_on(TtsBackend::from_config(&cfg)));
            assert!(
                !events
                    .iter()
                    .any(|(_, fields)| fields.iter().any(|(_, v)| v.contains(unknown_warning))),
                "{backend} is on COMPILED_IN_TTS_BACKENDS but from_config does not recognize it"
            );
        }
    }

    #[cfg(feature = "tts-polly")]
    #[test]
    fn polly_is_listed_when_compiled_in() {
        assert!(COMPILED_IN_TTS_BACKENDS.contains(&"polly"));
    }

    #[cfg(not(feature = "tts-polly"))]
    #[test]
    fn polly_is_not_listed_when_absent() {
        assert!(!COMPILED_IN_TTS_BACKENDS.contains(&"polly"));
    }

    #[cfg(feature = "tts-kokoro")]
    #[test]
    fn kokoro_is_listed_when_compiled_in() {
        assert!(COMPILED_IN_TTS_BACKENDS.contains(&"kokoro"));
    }

    #[cfg(not(feature = "tts-kokoro"))]
    #[test]
    fn kokoro_is_not_listed_when_absent() {
        assert!(!COMPILED_IN_TTS_BACKENDS.contains(&"kokoro"));
    }

    /// Captures tracing events so a test can assert on the report an operator
    /// reads, rather than only on the backend the call returns.
    ///
    /// One global subscriber for the whole test binary, with a per-thread slot
    /// that decides whether the current test is recording. A subscriber
    /// installed and dropped per test instead (`with_default`) is flaky under
    /// the default multi-threaded test runner: `tracing` caches each callsite's
    /// interest process-wide and rebuilds it whenever the subscriber set
    /// changes, so a concurrent test swapping its own subscriber in and out can
    /// make this one's event vanish. Installing once means the cache is built
    /// once and never invalidated.
    mod capture {
        use std::cell::RefCell;
        use std::sync::{Mutex, OnceLock, PoisonError};

        pub type Captured = Vec<(tracing::Level, Vec<(String, String)>)>;

        thread_local! {
            /// `Some` while this thread is inside `while_capturing`.
            static SLOT: RefCell<Option<Captured>> = const { RefCell::new(None) };
        }

        struct Layer;

        struct Recorder<'a>(&'a mut Vec<(String, String)>);

        impl tracing::field::Visit for Recorder<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0
                    .push((field.name().to_string(), format!("{value:?}")));
            }

            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.push((field.name().to_string(), value.to_string()));
            }
        }

        impl<S> tracing_subscriber::Layer<S> for Layer
        where
            S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
        {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                SLOT.with_borrow_mut(|slot| {
                    let Some(events) = slot.as_mut() else {
                        return; // this thread is not recording
                    };
                    let mut fields = Vec::new();
                    event.record(&mut Recorder(&mut fields));
                    events.push((*event.metadata().level(), fields));
                });
            }
        }

        /// Run `body` with this thread recording, and return what it logged.
        ///
        /// Serialized process-wide, and the interest cache is rebuilt on the
        /// way in. Two guards for two races that both showed up as the event
        /// simply going missing under the parallel test runner: another test
        /// emitting while this thread's slot is open, and a callsite first
        /// reached before any subscriber existed, which caches "nobody is
        /// interested" and keeps suppressing the event afterwards.
        pub fn while_capturing(body: impl FnOnce()) -> Captured {
            static INSTALLED: OnceLock<()> = OnceLock::new();
            static SERIALIZE: Mutex<()> = Mutex::new(());

            let _guard = SERIALIZE.lock().unwrap_or_else(PoisonError::into_inner);

            INSTALLED.get_or_init(|| {
                use tracing_subscriber::layer::SubscriberExt as _;
                tracing::subscriber::set_global_default(tracing_subscriber::registry().with(Layer))
                    .expect("no other global subscriber in this test binary");
            });
            tracing::callsite::rebuild_interest_cache();

            SLOT.with_borrow_mut(|slot| *slot = Some(Vec::new()));
            body();
            SLOT.with_borrow_mut(|slot| slot.take().unwrap_or_default())
        }
    }

    /// A configured backend this build does not contain must be reported at
    /// error level, naming the backend, the cargo feature that would add it,
    /// and what is available. That report is the only thing connecting the
    /// operator's config to the voice they actually hear.
    #[cfg(any(not(feature = "tts-polly"), not(feature = "tts-kokoro")))]
    #[test]
    fn an_absent_backend_is_reported_at_error_level_with_backend_feature_and_available() {
        let absent: Vec<(&str, &str)> = ["polly", "kokoro"]
            .into_iter()
            .filter(|b| !COMPILED_IN_TTS_BACKENDS.contains(b))
            .map(|b| match b {
                "polly" => ("polly", "tts-polly"),
                _ => ("kokoro", "tts-kokoro"),
            })
            .collect();
        assert!(
            !absent.is_empty(),
            "this configuration compiles the test but has no absent backend to report"
        );

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");

        for (backend, feature) in absent {
            let cfg = TtsConfig {
                backend: backend.into(),
                ..TtsConfig::default()
            };
            let events =
                capture::while_capturing(|| _ = runtime.block_on(TtsBackend::from_config(&cfg)));

            let report = events
                .iter()
                .find(|(level, _)| *level == tracing::Level::ERROR)
                .unwrap_or_else(|| {
                    panic!("no ERROR event when {backend} is configured but not compiled in")
                });

            let field = |name: &str| {
                report
                    .1
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(|| panic!("the {backend} report carries no `{name}` field"))
            };
            assert_eq!(field("backend"), backend);
            assert_eq!(field("feature"), feature);
            assert!(
                field("available").contains("piper"),
                "the {backend} report must say what IS available, got {:?}",
                field("available")
            );
        }
    }
}
