use adele_voice_core::VoiceError;
use adele_voice_core::ports::vad::VoiceActivityDetector;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// Silero VAD expects 512 samples at 16kHz (32ms chunks).
pub const VAD_CHUNK_SIZE: usize = 512;

pub struct SileroVad {
    session: Session,
    /// Hidden state for the LSTM — shape [2, 1, 64], flattened.
    h: Vec<f32>,
    /// Cell state for the LSTM — shape [2, 1, 64], flattened.
    c: Vec<f32>,
}

const STATE_SHAPE: [i64; 3] = [2, 1, 64];
const STATE_SIZE: usize = (STATE_SHAPE[0] * STATE_SHAPE[1] * STATE_SHAPE[2]) as usize;

impl SileroVad {
    pub fn new(model_path: &Path) -> Result<Self, VoiceError> {
        let session = Session::builder()
            .map_err(|e| VoiceError::Vad(format!("failed to create session builder: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| VoiceError::Vad(format!("failed to set threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| VoiceError::Vad(format!("failed to load model: {e}")))?;

        tracing::info!(model = %model_path.display(), "Silero VAD initialized");

        Ok(Self {
            session,
            h: vec![0.0; STATE_SIZE],
            c: vec![0.0; STATE_SIZE],
        })
    }
}

impl VoiceActivityDetector for SileroVad {
    fn speech_probability(&mut self, samples: &[f32]) -> Result<f32, VoiceError> {
        let map_err = |e| VoiceError::Vad(format!("VAD inference failed: {e}"));

        let input = Tensor::from_array(([1i64, samples.len() as i64], samples.to_vec()))
            .map_err(map_err)?;
        let sr = Tensor::from_array(([1i64], vec![16000i64])).map_err(map_err)?;
        let h = Tensor::from_array((STATE_SHAPE.to_vec(), self.h.clone())).map_err(map_err)?;
        let c = Tensor::from_array((STATE_SHAPE.to_vec(), self.c.clone())).map_err(map_err)?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input.upcast(),
                "sr" => sr.upcast(),
                "h" => h.upcast(),
                "c" => c.upcast(),
            ])
            .map_err(map_err)?;

        // Output "output": probability [1, 1]
        let prob = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(map_err)
            .map(|(_, data)| data.first().copied().unwrap_or(0.0))?;

        // Output "hn": new hidden state [2, 1, 64]
        if let Ok((_, data)) = outputs["hn"].try_extract_tensor::<f32>()
            && data.len() == STATE_SIZE
        {
            self.h.copy_from_slice(data);
        }

        // Output "cn": new cell state [2, 1, 64]
        if let Ok((_, data)) = outputs["cn"].try_extract_tensor::<f32>()
            && data.len() == STATE_SIZE
        {
            self.c.copy_from_slice(data);
        }

        Ok(prob)
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}
