use adele_voice_core::VoiceError;
use adele_voice_core::domain::Transcript;
use adele_voice_core::ports::stt::SpeechToText;
use std::path::Path;
use std::sync::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct WhisperStt {
    ctx: Mutex<WhisperContext>,
    language: String,
}

impl WhisperStt {
    pub fn new(model_path: &Path, language: &str) -> Result<Self, VoiceError> {
        let ctx = WhisperContext::new_with_params(
            model_path,
            WhisperContextParameters::default(),
        )
        .map_err(|e| VoiceError::Stt(format!("failed to load whisper model: {e}")))?;

        tracing::info!(
            model = %model_path.display(),
            language,
            "whisper STT initialized"
        );

        Ok(Self {
            ctx: Mutex::new(ctx),
            language: language.to_string(),
        })
    }
}

impl SpeechToText for WhisperStt {
    async fn transcribe(&self, samples: &[f32]) -> Result<Transcript, VoiceError> {
        let ctx = self
            .ctx
            .lock()
            .map_err(|e| VoiceError::Stt(format!("lock poisoned: {e}")))?;

        let mut state = ctx
            .create_state()
            .map_err(|e| VoiceError::Stt(format!("failed to create whisper state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(&self.language));
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
            if let Some(segment) = state.get_segment(i) {
                if let Ok(s) = segment.to_str() {
                    text.push_str(s);
                }
            }
        }

        let text = text.trim().to_string();
        tracing::debug!(text = %text, "transcribed");

        Ok(Transcript { text })
    }
}
