use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::audio::AudioSource;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample as _, SampleFormat, StreamConfig};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Producer, Split};
use rubato::Resampler;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use tokio::sync::mpsc;

/// Chunk size sent through the channel: 20ms of audio at 16kHz.
const CHUNK_FRAMES: usize = SAMPLE_RATE as usize / 50;

pub struct CpalAudioSource {
    device_name: String,
    running: Arc<AtomicBool>,
}

/// The capture format negotiated against a device. The pipeline always receives
/// 16 kHz mono `f32`; `rate`/`channels` describe what the *device* delivers,
/// which we downmix + resample to reach that contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChosenFormat {
    pub rate: u32,
    pub channels: u16,
    pub format: SampleFormat,
}

impl CpalAudioSource {
    pub fn new(device_name: &str) -> Self {
        Self {
            device_name: device_name.to_string(),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn find_input_device(name: &str) -> Result<cpal::Device, VoiceError> {
        let host = cpal::default_host();

        if name == "default" {
            return host
                .default_input_device()
                .ok_or_else(|| VoiceError::Audio("no default input device".into()));
        }

        host.input_devices()
            .map_err(|e| VoiceError::Audio(format!("failed to enumerate input devices: {e}")))?
            .find(|d| {
                d.description()
                    .map(|desc| desc.name().contains(name))
                    .unwrap_or(false)
            })
            .ok_or_else(|| VoiceError::Audio(format!("input device '{name}' not found")))
    }
}

/// Whether an ALSA pcm id is a sound-server route (shared with other apps)
/// rather than a raw hardware PCM. Raw cards take *exclusive* access, so an
/// always-listening capture on one can lock the mic away from other apps — and
/// from another logged-in user's session.
fn is_shared_route(pcm_id: &str) -> bool {
    matches!(pcm_id, "default" | "pipewire" | "pulse" | "jack")
}

/// The pcm id (ALSA "driver") behind a cpal device, if any.
fn device_pcm_id(device: &cpal::Device) -> Option<String> {
    device
        .description()
        .ok()
        .and_then(|d| d.driver().map(str::to_string))
}

/// Find a sound-server input route (PipeWire preferred, then PulseAudio).
fn find_shared_route() -> Option<cpal::Device> {
    let host = cpal::default_host();
    for want in ["pipewire", "pulse"] {
        // Re-enumerate per preference — the iterator is consumed by `find`.
        if let Some(d) = host
            .input_devices()
            .ok()?
            .find(|d| device_pcm_id(d).as_deref() == Some(want))
        {
            return Some(d);
        }
    }
    None
}

/// Resolve the configured input into a device to actually open, preferring a
/// shared sound-server route when the configured one is a raw ALSA card and a
/// server is available (the user opted into this). The raw card would take the
/// mic exclusively; routing through PipeWire/Pulse lets the assistant listen
/// without blocking other apps. `default` is left untouched — it's already the
/// system's chosen (shared) input.
fn resolve_input_device(name: &str) -> Result<cpal::Device, VoiceError> {
    let device = CpalAudioSource::find_input_device(name)?;
    if name == "default" {
        return Ok(device);
    }
    let pcm_id = device_pcm_id(&device).unwrap_or_default();
    if is_shared_route(&pcm_id) {
        return Ok(device);
    }
    // Configured device is a raw card — prefer a shared route if one exists.
    match find_shared_route() {
        Some(shared) => {
            let shared_id = device_pcm_id(&shared).unwrap_or_default();
            tracing::warn!(
                configured = %name,
                raw_pcm = %pcm_id,
                routing_via = %shared_id,
                "input_device is a raw ALSA card (exclusive access — can block \
                 other apps and other user sessions); routing via the shared \
                 sound server instead. It now follows the system default source. \
                 Set input_device = \"{shared_id}\" (or pick a shared device in \
                 settings) to silence this, or set the default source to the mic \
                 you want.",
            );
            Ok(shared)
        }
        None => Ok(device),
    }
}

/// One supported configuration range as reported by cpal, flattened to plain
/// values so the selection logic is pure and unit-testable.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ConfigRange {
    channels: u16,
    min_rate: u32,
    max_rate: u32,
    format: SampleFormat,
}

/// Sample formats we know how to convert to `f32` in the capture callback.
fn is_convertible(fmt: SampleFormat) -> bool {
    matches!(
        fmt,
        SampleFormat::F32
            | SampleFormat::F64
            | SampleFormat::I16
            | SampleFormat::U16
            | SampleFormat::I32
    )
}

/// Pick a capture format. Prefer one that yields 16 kHz directly (and, among
/// those, mono + `f32` so no conversion is needed) so the common PipeWire/PulseAudio
/// path stays a straight passthrough. When no supported range covers 16 kHz —
/// e.g. a raw 48 kHz USB mic — fall back to the device's default config and let
/// the pipeline downmix + resample. This is what stops the old hardcoded
/// `16 kHz mono` from tripping `snd_pcm_hw_params` (EINVAL) on such devices.
fn pick_capture_format(ranges: &[ConfigRange], default: ConfigRange) -> ChosenFormat {
    const TARGET: u32 = SAMPLE_RATE;
    let mut best: Option<(&ConfigRange, i32)> = None;
    for r in ranges
        .iter()
        .filter(|r| r.min_rate <= TARGET && TARGET <= r.max_rate && is_convertible(r.format))
    {
        // Lower score wins: mono beats multichannel, f32 beats a format we'd
        // have to convert.
        let score = (if r.channels == 1 { 0 } else { 1 })
            + (if r.format == SampleFormat::F32 { 0 } else { 2 });
        if best.is_none_or(|(_, bs)| score < bs) {
            best = Some((r, score));
        }
    }
    if let Some((r, _)) = best {
        return ChosenFormat {
            rate: TARGET,
            channels: r.channels,
            format: r.format,
        };
    }
    ChosenFormat {
        rate: default.min_rate,
        channels: default.channels,
        format: default.format,
    }
}

/// Negotiate a workable capture format for `device` (see [`pick_capture_format`]).
pub fn negotiate_input_config(device: &cpal::Device) -> Result<ChosenFormat, VoiceError> {
    let default = device
        .default_input_config()
        .map_err(|e| VoiceError::Audio(format!("no default input config: {e}")))?;
    let default_range = ConfigRange {
        channels: default.channels(),
        min_rate: default.sample_rate(),
        max_rate: default.sample_rate(),
        format: default.sample_format(),
    };
    let ranges: Vec<ConfigRange> = device
        .supported_input_configs()
        .map(|it| {
            it.map(|r| ConfigRange {
                channels: r.channels(),
                min_rate: r.min_sample_rate(),
                max_rate: r.max_sample_rate(),
                format: r.sample_format(),
            })
            .collect()
        })
        .unwrap_or_default();

    let chosen = pick_capture_format(&ranges, default_range);
    if !is_convertible(chosen.format) {
        return Err(VoiceError::Audio(format!(
            "device offers no usable sample format (best was {:?})",
            chosen.format
        )));
    }
    Ok(chosen)
}

/// Convert an interleaved slice of device samples to mono `f32`, averaging
/// channels. Reuses `out`'s capacity so the realtime callback doesn't allocate
/// after warm-up.
fn to_mono_f32<T>(data: &[T], channels: usize, out: &mut Vec<f32>)
where
    T: cpal::Sample,
    f32: cpal::FromSample<T>,
{
    out.clear();
    if channels <= 1 {
        out.extend(data.iter().map(|&s| f32::from_sample(s)));
        return;
    }
    for frame in data.chunks_exact(channels) {
        let sum: f32 = frame.iter().map(|&s| f32::from_sample(s)).sum();
        out.push(sum / channels as f32);
    }
}

/// Streaming integer/fractional resampler: feed src-rate mono `f32`, get 16 kHz
/// mono `f32` back as it becomes available. Holds rubato's overlap state across
/// calls so block boundaries don't introduce discontinuities.
struct StreamingResampler {
    inner: rubato::Fft<f32>,
    chunk: usize,
    pending: Vec<f32>,
    in_buf: Vec<Vec<f32>>,
    out_buf: Vec<Vec<f32>>,
}

impl StreamingResampler {
    fn new(src_rate: u32, dst_rate: u32) -> Result<Self, VoiceError> {
        let inner = rubato::Fft::<f32>::new(
            src_rate as usize,
            dst_rate as usize,
            1024,
            1,
            1,
            rubato::FixedSync::Input,
        )
        .map_err(|e| VoiceError::Audio(format!("resampler init: {e}")))?;
        let chunk = inner.input_frames_next();
        let out_max = inner.output_frames_max();
        Ok(Self {
            inner,
            chunk,
            pending: Vec::with_capacity(chunk * 2),
            in_buf: vec![vec![0.0f32; chunk]],
            out_buf: vec![vec![0.0f32; out_max]],
        })
    }

    fn push(&mut self, input: &[f32]) -> Result<Vec<f32>, VoiceError> {
        self.pending.extend_from_slice(input);
        let mut out = Vec::new();
        while self.pending.len() >= self.chunk {
            self.in_buf[0][..self.chunk].copy_from_slice(&self.pending[..self.chunk]);
            let in_adapter = SequentialSliceOfVecs::new(&self.in_buf, 1, self.chunk)
                .map_err(|e| VoiceError::Audio(format!("resampler input adapter: {e}")))?;
            let out_len = self.out_buf[0].len();
            let mut out_adapter = SequentialSliceOfVecs::new_mut(&mut self.out_buf, 1, out_len)
                .map_err(|e| VoiceError::Audio(format!("resampler output adapter: {e}")))?;
            let (_, nbr_out) = self
                .inner
                .process_into_buffer(&in_adapter, &mut out_adapter, None)
                .map_err(|e| VoiceError::Audio(format!("resampler process: {e}")))?;
            out.extend_from_slice(&self.out_buf[0][..nbr_out]);
            self.pending.drain(..self.chunk);
        }
        Ok(out)
    }
}

impl AudioSource for CpalAudioSource {
    fn start(&self) -> Result<mpsc::Receiver<Vec<f32>>, VoiceError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(VoiceError::Audio("capture already running".into()));
        }

        let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
        let device_name = self.device_name.clone();
        let running = Arc::clone(&self.running);

        // Build the stream on a dedicated thread since cpal::Stream is !Send
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        let running_clone = Arc::clone(&running);
        std::thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || {
                let device = match resolve_input_device(&device_name) {
                    Ok(d) => d,
                    Err(e) => {
                        let _ = result_tx.send(Err(e));
                        return;
                    }
                };

                let dev_name = device
                    .description()
                    .map(|desc| desc.name().to_string())
                    .unwrap_or_else(|_| "unknown".into());

                let fmt = match negotiate_input_config(&device) {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = result_tx.send(Err(e));
                        return;
                    }
                };
                tracing::info!(
                    device = %dev_name,
                    rate = fmt.rate,
                    channels = fmt.channels,
                    sample_format = ?fmt.format,
                    "opening input device"
                );

                let channels = fmt.channels as usize;
                let stream_config = StreamConfig {
                    channels: fmt.channels,
                    sample_rate: fmt.rate,
                    buffer_size: cpal::BufferSize::Default,
                };

                // Ring buffer holds ~1s of src-rate mono samples.
                let rb_cap = (fmt.rate as usize).max(SAMPLE_RATE as usize);
                let rb = HeapRb::<f32>::new(rb_cap);
                let (mut producer, mut consumer) = rb.split();

                let mut scratch: Vec<f32> = Vec::with_capacity(2048);
                let stream = match device.build_input_stream_raw(
                    &stream_config,
                    fmt.format,
                    move |data: &cpal::Data, _: &cpal::InputCallbackInfo| {
                        match fmt.format {
                            SampleFormat::F32 => {
                                if let Some(s) = data.as_slice::<f32>() {
                                    to_mono_f32(s, channels, &mut scratch);
                                }
                            }
                            SampleFormat::F64 => {
                                if let Some(s) = data.as_slice::<f64>() {
                                    to_mono_f32(s, channels, &mut scratch);
                                }
                            }
                            SampleFormat::I16 => {
                                if let Some(s) = data.as_slice::<i16>() {
                                    to_mono_f32(s, channels, &mut scratch);
                                }
                            }
                            SampleFormat::U16 => {
                                if let Some(s) = data.as_slice::<u16>() {
                                    to_mono_f32(s, channels, &mut scratch);
                                }
                            }
                            SampleFormat::I32 => {
                                if let Some(s) = data.as_slice::<i32>() {
                                    to_mono_f32(s, channels, &mut scratch);
                                }
                            }
                            _ => return,
                        }
                        let written = producer.push_slice(&scratch);
                        if written < scratch.len() {
                            tracing::debug!(
                                dropped = scratch.len() - written,
                                "input ring buffer overflow"
                            );
                        }
                    },
                    move |err| {
                        tracing::error!("input stream error: {err}");
                    },
                    None,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = result_tx.send(Err(VoiceError::Audio(format!(
                            "failed to build input stream: {e}"
                        ))));
                        return;
                    }
                };

                // Resample to 16 kHz only when the device isn't already there.
                let mut resampler = if fmt.rate != SAMPLE_RATE {
                    match StreamingResampler::new(fmt.rate, SAMPLE_RATE) {
                        Ok(r) => Some(r),
                        Err(e) => {
                            let _ = result_tx.send(Err(e));
                            return;
                        }
                    }
                } else {
                    None
                };

                if let Err(e) = stream.play() {
                    let _ = result_tx.send(Err(VoiceError::Audio(format!(
                        "failed to start input stream: {e}"
                    ))));
                    return;
                }

                running_clone.store(true, Ordering::SeqCst);
                let _ = result_tx.send(Ok(()));

                // Drain loop: pull src-rate mono from the ring buffer, resample
                // to 16 kHz, and forward. Sized so a 48 kHz device's 20ms worth
                // of audio is drained each tick.
                let read_cap = (fmt.rate as usize / 25).max(CHUNK_FRAMES);
                let mut read_buf = vec![0.0f32; read_cap];
                'drain: while running_clone.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(20));

                    loop {
                        let popped = consumer.pop_slice(&mut read_buf);
                        if popped == 0 {
                            break;
                        }
                        let out = match resampler.as_mut() {
                            Some(r) => match r.push(&read_buf[..popped]) {
                                Ok(o) => o,
                                Err(e) => {
                                    tracing::error!("resample failed: {e}");
                                    break 'drain;
                                }
                            },
                            None => read_buf[..popped].to_vec(),
                        };
                        if !out.is_empty() && tx.blocking_send(out).is_err() {
                            break 'drain; // receiver dropped
                        }
                        if popped < read_buf.len() {
                            break; // ring buffer drained for now
                        }
                    }
                }

                // Stream is dropped here, stopping capture
                drop(stream);
                tracing::info!("input capture thread exiting");
            })
            .map_err(|e| VoiceError::Audio(format!("failed to spawn capture thread: {e}")))?;

        result_rx
            .recv()
            .map_err(|_| VoiceError::Audio("capture thread exited unexpectedly".into()))??;

        Ok(rx)
    }

    fn stop(&self) -> Result<(), VoiceError> {
        self.running.store(false, Ordering::SeqCst);
        tracing::info!("input capture stop requested");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_f32_averages_channels() {
        let mut out = Vec::new();
        // L/R interleaved: (1.0,-1.0) -> 0.0, (0.5,0.5) -> 0.5
        to_mono_f32(&[1.0f32, -1.0, 0.5, 0.5], 2, &mut out);
        assert_eq!(out, vec![0.0, 0.5]);
    }

    #[test]
    fn mono_f32_passthrough() {
        let mut out = Vec::new();
        to_mono_f32(&[0.1f32, 0.2, 0.3], 1, &mut out);
        assert_eq!(out, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn downmix_converts_i16_to_f32() {
        let mut out = Vec::new();
        // Full-scale L/R both at max -> ~+1.0 after averaging.
        to_mono_f32(&[i16::MAX, i16::MAX], 2, &mut out);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-3, "got {}", out[0]);
        // Silence stays at 0.
        out.clear();
        to_mono_f32(&[0i16, 0], 2, &mut out);
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn picks_native_16k_mono_when_supported() {
        let ranges = [
            ConfigRange {
                channels: 2,
                min_rate: 44_100,
                max_rate: 48_000,
                format: SampleFormat::F32,
            },
            ConfigRange {
                channels: 1,
                min_rate: 8_000,
                max_rate: 48_000,
                format: SampleFormat::F32,
            },
        ];
        let default = ConfigRange {
            channels: 2,
            min_rate: 48_000,
            max_rate: 48_000,
            format: SampleFormat::F32,
        };
        let chosen = pick_capture_format(&ranges, default);
        assert_eq!(
            chosen,
            ChosenFormat {
                rate: 16_000,
                channels: 1,
                format: SampleFormat::F32
            }
        );
    }

    #[test]
    fn falls_back_to_default_when_16k_unsupported() {
        // A raw USB mic that only does 48 kHz stereo — the case that used to
        // crash with snd_pcm_hw_params EINVAL.
        let ranges = [ConfigRange {
            channels: 2,
            min_rate: 48_000,
            max_rate: 48_000,
            format: SampleFormat::F32,
        }];
        let default = ranges[0];
        let chosen = pick_capture_format(&ranges, default);
        assert_eq!(
            chosen,
            ChosenFormat {
                rate: 48_000,
                channels: 2,
                format: SampleFormat::F32
            }
        );
    }

    #[test]
    fn prefers_f32_over_integer_at_16k() {
        let ranges = [
            ConfigRange {
                channels: 1,
                min_rate: 16_000,
                max_rate: 16_000,
                format: SampleFormat::I16,
            },
            ConfigRange {
                channels: 2,
                min_rate: 8_000,
                max_rate: 48_000,
                format: SampleFormat::F32,
            },
        ];
        let default = ranges[0];
        let chosen = pick_capture_format(&ranges, default);
        // mono-i16 scores 0+2=2; stereo-f32 scores 1+0=1 -> f32 wins.
        assert_eq!(chosen.format, SampleFormat::F32);
        assert_eq!(chosen.rate, 16_000);
    }

    #[test]
    fn shared_routes_vs_raw_cards() {
        // Sound-server routes share the device with other apps.
        assert!(is_shared_route("default"));
        assert!(is_shared_route("pipewire"));
        assert!(is_shared_route("pulse"));
        assert!(is_shared_route("jack"));
        // Raw hardware PCMs take exclusive access.
        assert!(!is_shared_route("sysdefault:CARD=Mini"));
        assert!(!is_shared_route("front:CARD=Mini,DEV=0"));
        assert!(!is_shared_route("hw:CARD=PCH,DEV=0"));
        assert!(!is_shared_route("plughw:CARD=Mini"));
    }

    #[test]
    fn resampler_48k_to_16k_thirds_the_length() {
        let mut r = StreamingResampler::new(48_000, 16_000).unwrap();
        // 1s of 440 Hz sine at 48 kHz.
        let n = 48_000usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin())
            .collect();
        let out = r.push(&input).unwrap();
        // ~16k output (minus the final sub-chunk still buffered). Within 5%.
        let expected = n / 3;
        let diff = (out.len() as i64 - expected as i64).unsigned_abs() as usize;
        assert!(
            diff < expected / 20,
            "len {} vs expected ~{}",
            out.len(),
            expected
        );
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(out.iter().any(|&s| s.abs() > 0.1), "output is silent");
    }
}
