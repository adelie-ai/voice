//! Concrete wiring for embedding clients: build a [`Dictation`] or [`Speaker`]
//! from [`config`](crate::config) using the local adapter crates (cpal mic +
//! Silero VAD + Whisper STT; the configured TTS backend → cpal sink).
//!
//! These are for clients that own their own audio devices (a chat app embedding
//! in-app dictation/playback). The daemon wires the primitives by hand instead,
//! because it shares one sink between spoken replies and on-demand SayText.

use std::sync::Arc;
use std::time::Duration;

use adele_voice_audio_cpal::{CpalAudioSink, CpalAudioSource};
use adele_voice_core::VoiceError;
use adele_voice_core::ports::audio::{AudioSink, AudioSource};
use adele_voice_stt_whisper::WhisperStt;
use adele_voice_vad_silero::SileroVad;

use crate::config::{AudioConfig, SttConfig, TtsConfig, VadConfig};
use crate::dictation::{Dictation, DictationOptions};
use crate::speaker::Speaker;
use crate::tts_backend::TtsBackend;

/// Wire a [`Dictation`] from config: the cpal microphone, Silero VAD, and
/// Whisper STT, with endpointing tuned from the `[vad]` section.
///
/// To enable the half-duplex echo guard so the mic doesn't capture
/// and transcribe Adele's own TTS, chain the speaker's sink onto the result:
///
/// ```ignore
/// let speaker = build_speaker(&cfg.tts, &cfg.audio).await;
/// let dictation =
///     build_dictation(&cfg.audio, &cfg.vad, &cfg.stt)?.with_echo_guard(speaker.sink());
/// ```
///
/// Both must play through the same output device for the guard to be accurate;
/// sharing one [`Speaker`]'s [`sink`](crate::speaker::Speaker::sink) guarantees
/// that. Without the chained guard, dictation is playback-unaware as before.
pub fn build_dictation(
    audio: &AudioConfig,
    vad: &VadConfig,
    stt: &SttConfig,
) -> Result<Dictation<SileroVad, WhisperStt>, VoiceError> {
    let source: Arc<dyn AudioSource> = Arc::new(CpalAudioSource::new(&audio.input_device));
    let vad_adapter = SileroVad::new(&vad.model_path)?;
    let stt_adapter = WhisperStt::new(&stt.model_path, &stt.language)?;
    let opts = DictationOptions {
        speech_threshold: vad.speech_threshold,
        silence: Duration::from_millis(vad.silence_duration_ms),
        ..DictationOptions::default()
    };
    Ok(Dictation::new(source, vad_adapter, stt_adapter, opts))
}

/// Wire a [`Speaker`] from config: the configured TTS backend (with the
/// local-first Kokoro→Piper fallback) playing to the cpal output device.
pub async fn build_speaker(tts: &TtsConfig, audio: &AudioConfig) -> Speaker<TtsBackend> {
    let backend = TtsBackend::from_config(tts).await;
    let sink: Arc<dyn AudioSink> = Arc::new(CpalAudioSink::new(&audio.output_device));
    Speaker::new(Arc::new(backend), sink)
}
