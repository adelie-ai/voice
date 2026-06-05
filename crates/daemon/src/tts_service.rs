//! Text-to-speech service backing the `SayText` / `SynthesizeText` D-Bus
//! methods and voice selection (`ListVoices` / `GetVoice` / `SetVoice`). A
//! single task processes [`TtsCommand`]s so requests serialize, shares the
//! audio sink (and thus the playback queue) with the conversation pipeline, and
//! shares the `PiperTts` instance so a voice change is seen by both. It never
//! touches the microphone — speaking is independent of listening.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::audio::AudioSink;
use adele_voice_core::ports::tts::TextToSpeech;
use adele_voice_dbus_interface::TtsCommand;
use adele_voice_tts_kokoro::KokoroTts;
use adele_voice_tts_piper::{DEFAULT_PIPER_SAMPLE_RATE, PiperTts};
use tokio::sync::mpsc;

use crate::tts_backend::TtsBackend;

/// Drive the TTS service until the command channel closes.
pub async fn run_tts_service(
    tts: Arc<TtsBackend>,
    sink: Arc<dyn AudioSink>,
    models_dir: PathBuf,
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
            TtsCommand::ListVoices { reply } => {
                let voices = match &*tts {
                    TtsBackend::Piper(_) => list_voices(&models_dir),
                    TtsBackend::Polly(_) => polly_voices(),
                    TtsBackend::Kokoro(k) => kokoro_voices(k),
                };
                let _ = reply.send(voices);
            }
            TtsCommand::GetVoice { reply } => {
                let id_speaker = match &*tts {
                    TtsBackend::Piper(p) => {
                        let (path, speaker) = p.current_voice();
                        let id = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();
                        (id, speaker.map(|s| s as i32).unwrap_or(-1))
                    }
                    TtsBackend::Polly(p) => (p.current_voice().0, -1),
                    TtsBackend::Kokoro(k) => (k.current_voice(), -1),
                };
                let _ = reply.send(id_speaker);
            }
            TtsCommand::SetVoice {
                voice_id,
                speaker,
                reply,
            } => {
                let result = match &*tts {
                    TtsBackend::Piper(p) => set_voice(p, &models_dir, &voice_id, speaker),
                    TtsBackend::Polly(p) => {
                        let (_, engine) = p.current_voice();
                        p.set_voice(&voice_id, &engine);
                        Ok(())
                    }
                    TtsBackend::Kokoro(k) => k.set_voice(&voice_id).map_err(|e| e.to_string()),
                };
                let _ = reply.send(result);
            }
        }
    }
}

/// Resolve and apply a voice by id, validating that the model exists and the
/// speaker (if any) is in range. Affects both spoken responses and SayText.
fn set_voice(
    tts: &PiperTts,
    models_dir: &Path,
    voice_id: &str,
    speaker: i32,
) -> Result<(), String> {
    let model = models_dir.join(format!("{voice_id}.onnx"));
    if !model.exists() {
        return Err(format!(
            "voice '{voice_id}' not found in {}",
            models_dir.display()
        ));
    }
    let (_, _, num_speakers, sample_rate) =
        read_voice_metadata(&models_dir.join(format!("{voice_id}.onnx.json")));
    let speaker = if speaker >= 0 {
        if speaker as u32 >= num_speakers.max(1) {
            return Err(format!(
                "speaker {speaker} out of range (voice '{voice_id}' has {num_speakers})"
            ));
        }
        Some(speaker as i64)
    } else {
        None
    };
    tts.set_voice(model, speaker, sample_rate);
    Ok(())
}

/// Scan `models_dir` for Piper voices (`*.onnx` with a `.onnx.json` sidecar),
/// returning (id, display name, language, num_speakers) sorted by id.
fn list_voices(models_dir: &Path) -> Vec<(String, String, String, u32)> {
    let mut voices = Vec::new();
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return voices;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
            continue;
        }
        let json = path.with_extension("onnx.json");
        if !json.exists() {
            continue; // not a Piper voice (e.g. the VAD model)
        }
        let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let (name, language, num_speakers, _rate) = read_voice_metadata(&json);
        let display = if name.is_empty() {
            id.to_string()
        } else {
            name
        };
        voices.push((id.to_string(), display, language, num_speakers));
    }
    voices.sort();
    voices
}

/// A curated set of high-quality Polly English voices for the voice switcher.
/// Selecting one keeps the configured engine (neural/generative); it only
/// changes which voice speaks.
fn polly_voices() -> Vec<(String, String, String, u32)> {
    [
        ("Joanna", "Joanna (female, US)", "en_US"),
        ("Matthew", "Matthew (male, US)", "en_US"),
        ("Ruth", "Ruth (female, US)", "en_US"),
        ("Stephen", "Stephen (male, US)", "en_US"),
        ("Danielle", "Danielle (female, US)", "en_US"),
        ("Gregory", "Gregory (male, US)", "en_US"),
        ("Amy", "Amy (female, GB)", "en_GB"),
        ("Arthur", "Arthur (male, GB)", "en_GB"),
    ]
    .iter()
    .map(|(id, name, lang)| {
        (
            (*id).to_string(),
            (*name).to_string(),
            (*lang).to_string(),
            1u32,
        )
    })
    .collect()
}

/// List Kokoro voices (the `*.bin` files in the voices dir) for the switcher.
fn kokoro_voices(tts: &KokoroTts) -> Vec<(String, String, String, u32)> {
    tts.list_voices()
        .into_iter()
        .map(|name| {
            let lang = if name.starts_with('b') {
                "en_GB"
            } else {
                "en_US"
            };
            (name.clone(), name, lang.to_string(), 1u32)
        })
        .collect()
}

/// Read a voice's `.onnx.json`, returning (name, language, num_speakers,
/// sample_rate) with sensible defaults on any missing field or read error.
fn read_voice_metadata(json_path: &Path) -> (String, String, u32, u32) {
    std::fs::read_to_string(json_path)
        .ok()
        .map(|t| parse_voice_metadata(&t))
        .unwrap_or_else(|| (String::new(), String::new(), 1, DEFAULT_PIPER_SAMPLE_RATE))
}

fn parse_voice_metadata(json: &str) -> (String, String, u32, u32) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return (String::new(), String::new(), 1, DEFAULT_PIPER_SAMPLE_RATE);
    };
    let name = v
        .get("dataset")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let language = v
        .get("language")
        .and_then(|l| l.get("code").or_else(|| l.get("family")))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let num_speakers = v.get("num_speakers").and_then(|x| x.as_u64()).unwrap_or(1) as u32;
    let sample_rate = v
        .get("audio")
        .and_then(|a| a.get("sample_rate"))
        .and_then(|x| x.as_u64())
        .unwrap_or(DEFAULT_PIPER_SAMPLE_RATE as u64) as u32;
    (name, language, num_speakers, sample_rate)
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
    use super::*;

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
        assert_eq!(i16::from_le_bytes([wav[44], wav[45]]), i16::MAX);
        assert_eq!(i16::from_le_bytes([wav[46], wav[47]]), -i16::MAX);
    }

    #[test]
    fn parses_piper_voice_metadata() {
        let json = r#"{
            "dataset": "amy",
            "language": { "code": "en_US", "family": "en" },
            "audio": { "sample_rate": 22050 },
            "num_speakers": 1
        }"#;
        let (name, language, num_speakers, rate) = parse_voice_metadata(json);
        assert_eq!(name, "amy");
        assert_eq!(language, "en_US");
        assert_eq!(num_speakers, 1);
        assert_eq!(rate, 22050);
    }

    #[test]
    fn malformed_metadata_falls_back_to_defaults() {
        let (name, language, num_speakers, rate) = parse_voice_metadata("not json");
        assert!(name.is_empty());
        assert!(language.is_empty());
        assert_eq!(num_speakers, 1);
        assert_eq!(rate, DEFAULT_PIPER_SAMPLE_RATE);
    }
}
