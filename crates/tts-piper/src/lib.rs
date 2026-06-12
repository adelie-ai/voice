use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_core::resample;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Default Piper sample rate (en_US-amy-medium and most medium/high voices).
/// A per-voice rate from the `.onnx.json` overrides this when a voice is set.
pub const DEFAULT_PIPER_SAMPLE_RATE: u32 = 22050;

/// The active voice: which model to run, an optional multi-speaker id, and the
/// model's native sample rate (for resampling to the pipeline rate).
#[derive(Clone)]
struct Voice {
    model_path: PathBuf,
    speaker: Option<i64>,
    sample_rate: u32,
}

/// Text-to-speech via the Piper CLI. Cloning shares the active-voice state, so
/// a `set_voice` on any clone is seen by all — the conversation pipeline and
/// the SayText service share a single voice.
#[derive(Clone)]
pub struct PiperTts {
    piper_binary: String,
    voice: Arc<RwLock<Voice>>,
}

impl PiperTts {
    pub fn new(piper_binary: &str, model_path: &Path) -> Self {
        Self {
            piper_binary: piper_binary.to_string(),
            voice: Arc::new(RwLock::new(Voice {
                model_path: model_path.to_owned(),
                speaker: None,
                sample_rate: DEFAULT_PIPER_SAMPLE_RATE,
            })),
        }
    }

    /// Hot-swap the active voice. Subsequent synthesis uses the new model,
    /// speaker, and sample rate.
    pub fn set_voice(&self, model_path: PathBuf, speaker: Option<i64>, sample_rate: u32) {
        let mut v = self.voice.write().expect("piper voice lock poisoned");
        v.model_path = model_path;
        v.speaker = speaker;
        v.sample_rate = sample_rate;
    }

    /// The current voice's model path and (optional) speaker id.
    pub fn current_voice(&self) -> (PathBuf, Option<i64>) {
        let v = self.voice.read().expect("piper voice lock poisoned");
        (v.model_path.clone(), v.speaker)
    }
}

impl TextToSpeech for PiperTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let (model_path, speaker, piper_rate) = {
            let v = self.voice.read().expect("piper voice lock poisoned");
            (v.model_path.clone(), v.speaker, v.sample_rate)
        };

        // Piper reads text from stdin, writes raw PCM to stdout
        // Output format: 16-bit signed integer PCM, mono, at the model's sample rate
        let mut command = Command::new(&self.piper_binary);
        command.arg("--model").arg(&model_path).arg("--output_raw");
        if let Some(id) = speaker {
            command.arg("--speaker").arg(id.to_string());
        }
        let mut child = command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| VoiceError::Tts(format!("failed to spawn piper: {e}")))?;

        // Write text to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| VoiceError::Tts(format!("failed to write to piper stdin: {e}")))?;
            // Close stdin so piper starts processing
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| VoiceError::Tts(format!("piper process failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VoiceError::Tts(format!("piper failed: {stderr}")));
        }

        // Convert 16-bit PCM to f32 samples
        let pcm_bytes = &output.stdout;
        if pcm_bytes.len() % 2 != 0 {
            return Err(VoiceError::Tts("piper output has odd byte count".into()));
        }

        let i16_samples: Vec<i16> = pcm_bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let f32_samples: Vec<f32> = i16_samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();

        let input_samples = f32_samples.len();
        let resampled = if piper_rate == SAMPLE_RATE {
            f32_samples
        } else {
            resample(&f32_samples, piper_rate, SAMPLE_RATE)?
        };

        tracing::debug!(
            text_len = text.len(),
            input_samples,
            output_samples = resampled.len(),
            "TTS synthesis complete"
        );

        Ok(resampled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_voice_is_shared_across_clones() {
        // The pipeline and the SayText service hold clones; a SetVoice on one
        // must be seen by the other (shared active-voice state).
        let a = PiperTts::new("piper", Path::new("/m/en_US-amy-medium.onnx"));
        let b = a.clone();
        assert_eq!(
            a.current_voice().0,
            PathBuf::from("/m/en_US-amy-medium.onnx")
        );

        b.set_voice(PathBuf::from("/m/en_US-lessac-high.onnx"), Some(3), 22050);

        let (path, speaker) = a.current_voice();
        assert_eq!(path, PathBuf::from("/m/en_US-lessac-high.onnx"));
        assert_eq!(speaker, Some(3));
    }
}
