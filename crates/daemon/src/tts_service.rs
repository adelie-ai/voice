//! Text-to-speech service backing the `SayText` / `SynthesizeText` D-Bus
//! methods. A single task processes [`TtsCommand`]s so concurrent requests
//! serialize instead of colliding, and SayText shares the audio sink (and thus
//! the playback queue) with the conversation pipeline. It never touches the
//! microphone — speaking is independent of listening.

use std::sync::Arc;

use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::audio::AudioSink;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_dbus_interface::TtsCommand;
use tokio::sync::mpsc;

/// Drive the TTS service until the command channel closes. `Say` synthesizes
/// and plays through `sink`; `Synthesize` synthesizes and returns WAV bytes via
/// the request's reply channel.
pub async fn run_tts_service<T: TextToSpeech>(
    tts: Arc<T>,
    sink: Arc<dyn AudioSink>,
    mut rx: mpsc::Receiver<TtsCommand>,
) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            TtsCommand::Say(text) => match tts.synthesize(&text).await {
                Ok(samples) if !samples.is_empty() => {
                    if let Err(e) = sink.play(samples) {
                        tracing::error!("SayText playback failed: {e}");
                    }
                }
                Ok(_) => {}
                Err(e) => tracing::error!("SayText synthesis failed: {e}"),
            },
            TtsCommand::Synthesize { text, reply } => {
                let result = tts
                    .synthesize(&text)
                    .await
                    .map(|samples| encode_wav(&samples, SAMPLE_RATE))
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
        }
    }
}

/// Encode mono f32 samples (range -1.0..=1.0) as a 16-bit PCM WAV buffer.
fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() as u32) * 2; // 16-bit mono
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::encode_wav;

    #[test]
    fn wav_header_and_size_are_correct() {
        let wav = encode_wav(&[0.0, 0.5, -0.5, 1.0], 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(wav.len(), 44 + 4 * 2);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16000
        );
        // RIFF chunk size = 36 + data length
        assert_eq!(u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]), 36 + 8);
    }

    #[test]
    fn empty_input_is_a_valid_header_only_wav() {
        let wav = encode_wav(&[], 16000);
        assert_eq!(wav.len(), 44);
        assert_eq!(&wav[0..4], b"RIFF");
    }

    #[test]
    fn out_of_range_samples_clamp_to_i16_bounds() {
        let wav = encode_wav(&[1.5, -1.5], 16000);
        let s0 = i16::from_le_bytes([wav[44], wav[45]]);
        let s1 = i16::from_le_bytes([wav[46], wav[47]]);
        assert_eq!(s0, i16::MAX);
        assert_eq!(s1, -i16::MAX);
    }
}
