//! Energy-gated speech-to-text.
//!
//! Wraps a [`SpeechToText`] adapter with the daemon's near-silence guard so
//! ambient noise or the tail of our own playback — which can trip the VAD
//! without containing real speech — never reaches Whisper (which would then
//! hallucinate filler like "Thank you.").

use std::sync::Arc;

use adele_voice_core::VoiceError;
use adele_voice_core::domain::Transcript;
use adele_voice_core::ports::stt::SpeechToText;

/// RMS floor below which a capture is treated as silence and dropped before
/// STT. Real speech sits well above this (a brief utterance is ~0.02+, while
/// noise/echo is ~0.003–0.008).
pub const MIN_SPEECH_RMS: f32 = 0.01;

/// Root-mean-square amplitude of a PCM buffer (0.0 for an empty buffer).
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        0.0
    } else {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }
}

/// Transcribes captured audio, gating out near-silent buffers and empty results.
pub struct Transcriber<S> {
    stt: Arc<S>,
    min_rms: f32,
}

impl<S: SpeechToText> Transcriber<S> {
    /// Use the default [`MIN_SPEECH_RMS`] energy floor.
    pub fn new(stt: Arc<S>) -> Self {
        Self {
            stt,
            min_rms: MIN_SPEECH_RMS,
        }
    }

    /// Override the energy floor (mainly for tuning/tests).
    pub fn with_min_rms(stt: Arc<S>, min_rms: f32) -> Self {
        Self { stt, min_rms }
    }

    /// Energy-gate then transcribe. Returns `None` when the capture is below the
    /// speech floor (noise/echo) or the transcript comes back empty — i.e.
    /// "nothing worth sending on."
    pub async fn transcribe(&self, samples: &[f32]) -> Result<Option<Transcript>, VoiceError> {
        let level = rms(samples);
        tracing::info!(rms = level, samples = samples.len(), "utterance captured");
        if level < self.min_rms {
            tracing::info!(rms = level, "discarding near-silent capture (no speech)");
            return Ok(None);
        }

        let transcript = self.stt.transcribe(samples).await?;
        if transcript.text.is_empty() {
            tracing::debug!("empty transcript, skipping");
            return Ok(None);
        }
        Ok(Some(transcript))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn rms_of_empty_is_zero() {
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn rms_of_constant_amplitude() {
        // RMS of a constant 0.1 signal is 0.1.
        assert!((rms(&[0.1; 100]) - 0.1).abs() < 1e-6);
    }

    /// STT that records whether it ran and returns a scripted transcript.
    struct FakeStt {
        ran: StdMutex<bool>,
        text: String,
    }
    impl FakeStt {
        fn new(text: &str) -> Arc<Self> {
            Arc::new(Self {
                ran: StdMutex::new(false),
                text: text.to_string(),
            })
        }
    }
    impl SpeechToText for FakeStt {
        async fn transcribe(&self, _samples: &[f32]) -> Result<Transcript, VoiceError> {
            *self.ran.lock().unwrap() = true;
            Ok(Transcript {
                text: self.text.clone(),
            })
        }
    }

    #[tokio::test]
    async fn near_silent_capture_is_gated_before_stt() {
        let stt = FakeStt::new("should not be reached");
        let t = Transcriber::new(Arc::clone(&stt));
        let out = t.transcribe(&[0.0; 1000]).await.unwrap();
        assert!(out.is_none(), "near-silent capture must be gated out");
        assert!(
            !*stt.ran.lock().unwrap(),
            "STT must not run on a gated capture"
        );
    }

    #[tokio::test]
    async fn empty_transcript_is_none() {
        let stt = FakeStt::new("");
        let t = Transcriber::new(stt);
        let out = t.transcribe(&[0.2; 1000]).await.unwrap();
        assert!(out.is_none(), "an empty transcript yields None");
    }

    #[tokio::test]
    async fn real_speech_transcribes() {
        let stt = FakeStt::new("hello there");
        let t = Transcriber::new(Arc::clone(&stt));
        let out = t.transcribe(&[0.2; 1000]).await.unwrap();
        assert_eq!(out.unwrap().text, "hello there");
        assert!(*stt.ran.lock().unwrap(), "STT must run for real speech");
    }
}
