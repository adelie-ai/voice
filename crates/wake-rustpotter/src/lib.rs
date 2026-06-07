use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::wake::WakeWordDetector;
use rustpotter::{Rustpotter, RustpotterConfig, SampleFormat};
use std::path::Path;

pub struct RustpotterWakeWordDetector {
    rustpotter: Rustpotter,
    /// Exact number of samples rustpotter consumes per `process_bytes` call.
    /// `process_bytes` returns `None` (a silent no-op) for ANY other length and
    /// does not buffer across calls, so we must hand it exactly one frame.
    samples_per_frame: usize,
    /// Carries leftover samples between `detect()` calls so arbitrary capture
    /// chunk sizes (the daemon sends 20 ms / 320-sample chunks) are re-framed
    /// into rustpotter's frame size instead of being dropped on the floor (#44).
    buf: Vec<f32>,
}

/// Pop one complete `frame`-sized chunk from the front of `buf`, retaining the
/// sub-frame remainder for the next call; returns `None` when there isn't yet a
/// full frame. Pure framing logic, kept free of rustpotter so it is unit-tested
/// directly — this is the fix for #44 (rustpotter silently drops any input that
/// isn't exactly one frame).
fn take_frame(buf: &mut Vec<f32>, frame: usize) -> Option<Vec<f32>> {
    if frame == 0 || buf.len() < frame {
        return None;
    }
    let out: Vec<f32> = buf[..frame].to_vec();
    buf.drain(..frame);
    Some(out)
}

impl RustpotterWakeWordDetector {
    pub fn new(model_path: &Path, sensitivity: f32) -> Result<Self, VoiceError> {
        let mut config = RustpotterConfig::default();
        config.fmt.sample_rate = SAMPLE_RATE as usize;
        config.fmt.sample_format = SampleFormat::F32;
        config.fmt.channels = 1;
        config.detector.threshold = sensitivity;
        // Disable the averaged-score pre-gate (rustpotter default 0.2). It sits IN
        // FRONT of the per-frame `threshold`, so whenever the windowed average dips
        // below it a real wake word is silently dropped even though individual
        // frames clear `threshold`. We gate purely on `threshold` + `min_scores`.
        config.detector.avg_threshold = 0.0;
        // Deliberately leave the gain-normalizer DISABLED. rustpotter's MFCC is
        // already level-tolerant, and enabling the normalizer *destroys* live
        // detection: on quieter-than-training speech it amplifies the input and the
        // match collapses to ~0 (measured: same utterance scores 0.46 with the
        // normalizer off, 0.0 with it on at any max_gain). See #44.

        let mut rustpotter = Rustpotter::new(&config)
            .map_err(|e| VoiceError::WakeWord(format!("failed to create rustpotter: {e}")))?;

        let model_str = model_path.to_string_lossy();
        rustpotter
            .add_wakeword_from_file("hey-adele", &model_str)
            .map_err(|e| VoiceError::WakeWord(format!("failed to load wake word model: {e}")))?;

        let samples_per_frame = rustpotter.get_samples_per_frame();

        tracing::info!(
            model = %model_path.display(),
            sensitivity,
            samples_per_frame,
            "wake word detector initialized"
        );

        Ok(Self {
            rustpotter,
            samples_per_frame,
            buf: Vec::new(),
        })
    }
}

impl WakeWordDetector for RustpotterWakeWordDetector {
    fn detect(&mut self, samples: &[f32]) -> Result<bool, VoiceError> {
        // Re-frame: rustpotter consumes EXACTLY `samples_per_frame` samples per
        // call and silently no-ops otherwise (#44), so accumulate and feed it one
        // frame at a time, keeping the sub-frame remainder for the next call.
        self.buf.extend_from_slice(samples);

        let mut detected = false;
        while let Some(frame) = take_frame(&mut self.buf, self.samples_per_frame) {
            // f32 little-endian bytes, as rustpotter's F32 format expects.
            let bytes: Vec<u8> = frame.iter().flat_map(|s| s.to_le_bytes()).collect();
            if let Some(detection) = self.rustpotter.process_bytes(&bytes) {
                // Log the score so the threshold can be tuned from real fires.
                tracing::info!(
                    score = detection.score,
                    avg_score = detection.avg_score,
                    gain = detection.gain,
                    "wake word detected"
                );
                detected = true;
            }
        }
        Ok(detected)
    }
}

#[cfg(test)]
mod tests {
    use super::take_frame;

    #[test]
    fn sub_frame_chunk_yields_no_frame_but_is_retained() {
        let frame = 480;
        let mut buf = vec![0.0f32; 320]; // one 20 ms capture chunk
        assert!(take_frame(&mut buf, frame).is_none());
        assert_eq!(
            buf.len(),
            320,
            "sub-frame input must be retained, not dropped"
        );
    }

    #[test]
    fn frames_assemble_across_multiple_sub_frame_chunks() {
        let frame = 480;
        let mut buf: Vec<f32> = Vec::new();

        // Two 320-sample chunks = 640 samples => exactly one 480 frame + 160 left.
        buf.extend_from_slice(&[1.0; 320]);
        assert!(take_frame(&mut buf, frame).is_none());
        buf.extend_from_slice(&[1.0; 320]);

        let f = take_frame(&mut buf, frame).expect("a full frame should be available");
        assert_eq!(f.len(), frame);
        assert_eq!(buf.len(), 160, "remainder is kept for the next call");
        assert!(take_frame(&mut buf, frame).is_none());
    }

    #[test]
    fn multiple_whole_frames_drain_in_order() {
        let frame = 4;
        let mut buf: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 2 frames + 2 left

        assert_eq!(take_frame(&mut buf, frame), Some(vec![0.0, 1.0, 2.0, 3.0]));
        assert_eq!(take_frame(&mut buf, frame), Some(vec![4.0, 5.0, 6.0, 7.0]));
        assert!(take_frame(&mut buf, frame).is_none());
        assert_eq!(buf, vec![8.0, 9.0]);
    }

    #[test]
    fn zero_frame_size_never_loops() {
        let mut buf = vec![0.0f32; 10];
        assert!(take_frame(&mut buf, 0).is_none());
        assert_eq!(buf.len(), 10);
    }
}
