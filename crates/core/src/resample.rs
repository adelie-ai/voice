//! Shared audio resampling helper (voice#89).
//!
//! The TTS backends synthesize at their model's native rate (piper voices vary;
//! Kokoro is 24 kHz) and must convert to the pipeline's playback rate. They had
//! identical, line-for-line copies of this rubato wrapper; it lives here so they
//! share one implementation.
//!
//! This is the *batch* (one-shot) resampler — it converts a whole buffer at
//! once. audio-cpal's `StreamingResampler` is a deliberately separate wrapper:
//! it resamples the live capture stream chunk-by-chunk with carried-over state,
//! which is a different problem than converting a finished synthesis buffer.

use crate::VoiceError;
use rubato::Resampler;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;

/// Resample mono f32 audio between integer sample rates using rubato's FFT-based
/// synchronous resampler. Anti-aliased, suitable for batch (one-shot) conversion
/// of TTS output. An empty input yields an empty output.
pub fn resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>, VoiceError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
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
    .map_err(|e| VoiceError::Audio(format!("resampler init: {e}")))?;

    let input_len = input.len();
    let output_len = resampler.process_all_needed_output_len(input_len);

    let input_data = vec![input.to_vec()];
    let mut output_data = vec![vec![0.0f32; output_len]];

    let in_adapter = SequentialSliceOfVecs::new(&input_data, 1, input_len)
        .map_err(|e| VoiceError::Audio(format!("resampler input adapter: {e}")))?;
    let mut out_adapter = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)
        .map_err(|e| VoiceError::Audio(format!("resampler output adapter: {e}")))?;

    let (_, nbr_out) = resampler
        .process_all_into_buffer(&in_adapter, &mut out_adapter, input_len, None)
        .map_err(|e| VoiceError::Audio(format!("resampler process: {e}")))?;

    let mut out = output_data.into_iter().next().unwrap();
    out.truncate(nbr_out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(resample(&[], 24_000, 16_000).unwrap().is_empty());
    }

    #[test]
    fn downsampling_shrinks_the_buffer_proportionally() {
        // 24 kHz -> 16 kHz is a 2:3 ratio, so the output is roughly 2/3 the
        // length. Allow slack for the resampler's edge framing.
        let input = vec![0.0f32; 24_000];
        let out = resample(&input, 24_000, 16_000).unwrap();
        let expected = 16_000;
        let slack = 2_048;
        assert!(
            out.len().abs_diff(expected) < slack,
            "downsampled length {} should be near {expected}",
            out.len()
        );
    }

    #[test]
    fn same_rate_round_trips_close_to_input_length() {
        let input = vec![0.0f32; 8_000];
        let out = resample(&input, 16_000, 16_000).unwrap();
        assert!(
            out.len().abs_diff(input.len()) < 2_048,
            "same-rate length {} should be near {}",
            out.len(),
            input.len()
        );
    }
}
