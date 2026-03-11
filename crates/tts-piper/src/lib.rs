use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::tts::TextToSpeech;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

pub struct PiperTts {
    piper_binary: String,
    model_path: PathBuf,
}

impl PiperTts {
    pub fn new(piper_binary: &str, model_path: &Path) -> Self {
        Self {
            piper_binary: piper_binary.to_string(),
            model_path: model_path.to_owned(),
        }
    }
}

impl TextToSpeech for PiperTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Piper reads text from stdin, writes raw PCM to stdout
        // Output format: 16-bit signed integer PCM, mono, at the model's sample rate
        let mut child = Command::new(&self.piper_binary)
            .arg("--model")
            .arg(&self.model_path)
            .arg("--output_raw")
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

        // Piper's default output rate is 22050Hz for most models.
        // Resample to our pipeline rate (16kHz) using simple linear interpolation.
        let piper_sample_rate = 22050u32;
        let f32_samples: Vec<f32> = i16_samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();

        if piper_sample_rate == SAMPLE_RATE {
            return Ok(f32_samples);
        }

        let ratio = piper_sample_rate as f64 / SAMPLE_RATE as f64;
        let output_len = (f32_samples.len() as f64 / ratio).ceil() as usize;
        let mut resampled = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;

            let sample = if idx + 1 < f32_samples.len() {
                f32_samples[idx] * (1.0 - frac as f32) + f32_samples[idx + 1] * frac as f32
            } else if idx < f32_samples.len() {
                f32_samples[idx]
            } else {
                0.0
            };

            resampled.push(sample);
        }

        tracing::debug!(
            text_len = text.len(),
            input_samples = i16_samples.len(),
            output_samples = resampled.len(),
            "TTS synthesis complete"
        );

        Ok(resampled)
    }
}
