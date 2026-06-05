use std::sync::{Arc, RwLock};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::tts::TextToSpeech;
use aws_sdk_polly::types::{Engine, OutputFormat, VoiceId};

/// The active Polly voice and engine ("neural", "generative", "long-form", or
/// "standard").
#[derive(Clone)]
struct PollyVoice {
    voice_id: String,
    engine: String,
}

/// Text-to-speech via AWS Polly (neural / generative AI voices). Cloning shares
/// the active-voice state, mirroring `PiperTts`, so the conversation pipeline
/// and the SayText service stay on one voice. Synthesis is a cloud round-trip —
/// audio leaves the device — and credentials/region come from the standard AWS
/// chain (env, profile, IMDS), the same one the orchestrator uses for Bedrock.
#[derive(Clone)]
pub struct PollyTts {
    client: aws_sdk_polly::Client,
    voice: Arc<RwLock<PollyVoice>>,
}

impl PollyTts {
    /// Build a Polly client. `region` overrides the AWS-chain default when set;
    /// `profile` selects a named AWS credentials profile (an empty or `None`
    /// profile falls back to the ambient env/IMDS chain).
    pub async fn new(
        voice_id: &str,
        engine: &str,
        region: Option<String>,
        profile: Option<String>,
    ) -> Self {
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(r) = region {
            loader = loader.region(aws_config::Region::new(r));
        }
        if let Some(p) = profile.filter(|p| !p.is_empty()) {
            loader = loader.profile_name(p);
        }
        let shared = loader.load().await;
        let client = aws_sdk_polly::Client::new(&shared);
        Self {
            client,
            voice: Arc::new(RwLock::new(PollyVoice {
                voice_id: voice_id.to_string(),
                engine: engine.to_string(),
            })),
        }
    }

    /// Hot-swap the active voice/engine; subsequent synthesis uses them.
    pub fn set_voice(&self, voice_id: &str, engine: &str) {
        let mut v = self.voice.write().expect("polly voice lock poisoned");
        v.voice_id = voice_id.to_string();
        v.engine = engine.to_string();
    }

    /// The current (voice_id, engine).
    pub fn current_voice(&self) -> (String, String) {
        let v = self.voice.read().expect("polly voice lock poisoned");
        (v.voice_id.clone(), v.engine.clone())
    }
}

impl TextToSpeech for PollyTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let (voice_id, engine) = {
            let v = self.voice.read().expect("polly voice lock poisoned");
            (v.voice_id.clone(), v.engine.clone())
        };

        // Request 16-bit PCM at the pipeline's sample rate so no resampling is
        // needed. Polly PCM is signed 16-bit, mono, little-endian.
        let resp = self
            .client
            .synthesize_speech()
            .text(text)
            .voice_id(VoiceId::from(voice_id.as_str()))
            .engine(Engine::from(engine.as_str()))
            .output_format(OutputFormat::Pcm)
            .sample_rate(SAMPLE_RATE.to_string())
            .send()
            .await
            .map_err(|e| VoiceError::Tts(format!("polly synthesize: {e}")))?;

        let bytes = resp
            .audio_stream
            .collect()
            .await
            .map_err(|e| VoiceError::Tts(format!("polly stream read: {e}")))?
            .into_bytes();

        if bytes.len() % 2 != 0 {
            return Err(VoiceError::Tts("polly PCM has odd byte count".into()));
        }
        let samples: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
            .collect();

        tracing::debug!(
            text_len = text.len(),
            samples = samples.len(),
            "Polly synthesis complete"
        );
        Ok(samples)
    }
}
