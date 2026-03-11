use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::wake::WakeWordDetector;
use rustpotter::{Rustpotter, RustpotterConfig, SampleFormat};
use std::path::Path;

pub struct RustpotterWakeWordDetector {
    rustpotter: Rustpotter,
}

impl RustpotterWakeWordDetector {
    pub fn new(model_path: &Path, sensitivity: f32) -> Result<Self, VoiceError> {
        let mut config = RustpotterConfig::default();
        config.fmt.sample_rate = SAMPLE_RATE as usize;
        config.fmt.sample_format = SampleFormat::F32;
        config.fmt.channels = 1;
        config.detector.threshold = sensitivity;

        let mut rustpotter = Rustpotter::new(&config)
            .map_err(|e| VoiceError::WakeWord(format!("failed to create rustpotter: {e}")))?;

        let model_str = model_path.to_string_lossy();
        rustpotter
            .add_wakeword_from_file("hey-adele", &model_str)
            .map_err(|e| VoiceError::WakeWord(format!("failed to load wake word model: {e}")))?;

        tracing::info!(
            model = %model_path.display(),
            sensitivity,
            "wake word detector initialized"
        );

        Ok(Self { rustpotter })
    }
}

impl WakeWordDetector for RustpotterWakeWordDetector {
    fn detect(&mut self, samples: &[f32]) -> Result<bool, VoiceError> {
        // Convert f32 samples to bytes (little-endian) as rustpotter expects raw bytes
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let detection = self.rustpotter.process_bytes(&bytes);
        Ok(detection.is_some())
    }
}
