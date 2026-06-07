use adele_voice_core::VoiceError;
use adele_voice_core::domain::Transcript;
use adele_voice_core::ports::stt::SpeechToText;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Whisper inference is CPU-heavy and synchronous. We run it on a blocking
/// thread and cap it with a timeout so a wedged decode can't hang the voice
/// turn forever (#58) — on timeout the turn apologizes and returns to Idle
/// rather than leaving the user stuck in Processing.
pub struct WhisperStt {
    ctx: Arc<Mutex<WhisperContext>>,
    language: String,
    timeout: Duration,
}

/// Default ceiling on a single Whisper decode. Generous — a normal utterance
/// decodes in well under a second on the distil model, so this only fires when
/// inference is genuinely wedged, not on a slow-but-progressing decode.
pub const DEFAULT_STT_TIMEOUT: Duration = Duration::from_secs(20);

impl WhisperStt {
    pub fn new(model_path: &Path, language: &str) -> Result<Self, VoiceError> {
        Self::with_timeout(model_path, language, DEFAULT_STT_TIMEOUT)
    }

    /// Build with an explicit per-decode timeout (config knob, #58).
    pub fn with_timeout(
        model_path: &Path,
        language: &str,
        timeout: Duration,
    ) -> Result<Self, VoiceError> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .map_err(|e| VoiceError::Stt(format!("failed to load whisper model: {e}")))?;

        tracing::info!(
            model = %model_path.display(),
            language,
            timeout_ms = timeout.as_millis(),
            "whisper STT initialized"
        );

        Ok(Self {
            ctx: Arc::new(Mutex::new(ctx)),
            language: language.to_string(),
            timeout,
        })
    }

    /// Run one synchronous decode. Pure CPU work — invoked on a blocking thread.
    fn decode(
        ctx: &Mutex<WhisperContext>,
        language: &str,
        samples: &[f32],
    ) -> Result<String, VoiceError> {
        let ctx = ctx
            .lock()
            .map_err(|e| VoiceError::Stt(format!("lock poisoned: {e}")))?;

        let mut state = ctx
            .create_state()
            .map_err(|e| VoiceError::Stt(format!("failed to create whisper state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(language));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_single_segment(true);
        params.set_no_context(true);

        state
            .full(params, samples)
            .map_err(|e| VoiceError::Stt(format!("whisper inference failed: {e}")))?;

        let num_segments = state.full_n_segments();

        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i)
                && let Ok(s) = segment.to_str()
            {
                text.push_str(s);
            }
        }

        Ok(text.trim().to_string())
    }
}

impl SpeechToText for WhisperStt {
    async fn transcribe(&self, samples: &[f32]) -> Result<Transcript, VoiceError> {
        let ctx = Arc::clone(&self.ctx);
        let language = self.language.clone();
        let samples = samples.to_vec();

        // Inference is synchronous CPU work: run it on a blocking thread so it
        // doesn't stall the async runtime, and cap it with a timeout so a
        // wedged decode can't hang the turn forever (#58).
        let decode = tokio::task::spawn_blocking(move || Self::decode(&ctx, &language, &samples));

        let text = match tokio::time::timeout(self.timeout, decode).await {
            Ok(Ok(result)) => result?,
            Ok(Err(join_err)) => {
                return Err(VoiceError::Stt(format!(
                    "whisper inference task failed: {join_err}"
                )));
            }
            Err(_elapsed) => {
                return Err(VoiceError::Stt(format!(
                    "whisper inference timed out after {} ms",
                    self.timeout.as_millis()
                )));
            }
        };

        tracing::debug!(text = %text, "transcribed");

        Ok(Transcript { text })
    }
}
