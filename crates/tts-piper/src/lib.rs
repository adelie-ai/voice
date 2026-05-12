use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::tts::TextToSpeech;
use rubato::Resampler;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Piper's default sample rate for the en_US-amy and similar models.
const PIPER_SAMPLE_RATE: u32 = 22050;

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

        let f32_samples: Vec<f32> = i16_samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();

        let input_samples = f32_samples.len();
        let resampled = if PIPER_SAMPLE_RATE == SAMPLE_RATE {
            f32_samples
        } else {
            resample(&f32_samples, PIPER_SAMPLE_RATE, SAMPLE_RATE)?
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

/// Resample mono f32 audio between integer sample rates using rubato's
/// FFT-based synchronous resampler. Anti-aliased, suitable for batch
/// (one-shot) conversion of TTS output.
fn resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>, VoiceError> {
    // 1024-frame chunks balance memory and FFT cost; process_all_into_buffer
    // loops over the input internally.
    let chunk_size = 1024;
    let mut resampler = rubato::Fft::<f32>::new(
        src_rate as usize,
        dst_rate as usize,
        chunk_size,
        1,
        1,
        rubato::FixedSync::Input,
    )
    .map_err(|e| VoiceError::Tts(format!("resampler init: {e}")))?;

    let input_len = input.len();
    let output_len = resampler.process_all_needed_output_len(input_len);

    let input_data = vec![input.to_vec()];
    let mut output_data = vec![vec![0.0f32; output_len]];

    let in_adapter = SequentialSliceOfVecs::new(&input_data, 1, input_len)
        .map_err(|e| VoiceError::Tts(format!("resampler input adapter: {e}")))?;
    let mut out_adapter = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)
        .map_err(|e| VoiceError::Tts(format!("resampler output adapter: {e}")))?;

    let (_, nbr_out) = resampler
        .process_all_into_buffer(&in_adapter, &mut out_adapter, input_len, None)
        .map_err(|e| VoiceError::Tts(format!("resampler process: {e}")))?;

    let mut out = output_data.into_iter().next().unwrap();
    out.truncate(nbr_out);
    Ok(out)
}
