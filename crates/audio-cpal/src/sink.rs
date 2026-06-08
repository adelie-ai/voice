use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::{CHANNELS, SAMPLE_RATE};
use adele_voice_core::ports::audio::AudioSink;
use cpal::StreamConfig;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use ringbuf::traits::{Producer, Split};

/// Ring buffer capacity for output, sized to hold a full spoken reply at
/// 16kHz. Sentences are synthesized and pushed back-to-back without waiting,
/// so a small buffer overflowed and dropped each sentence's tail as the next
/// was pushed; the response hint caps replies near ~30s, so 120s is ample
/// headroom for them to queue and play gaplessly.
const OUTPUT_RING_BUFFER_CAPACITY: usize = SAMPLE_RATE as usize * 120;

/// Floor for the playback tail pad — the cushion added to the computed deadline
/// so `is_playing` stays true until the sound is truly done. The pad covers the
/// gap between queueing audio and it physically sounding: stream-open latency on
/// the first sentence plus the device/OS output buffer the callback fills ahead
/// of playout. Erring long is safe — the daemon waits a hair past the tail;
/// erring short re-arms the mic mid-word and records the daemon's own voice.
///
/// This is only a floor: the real pad is derived from the device output latency
/// measured live in the output callback (`OutputStreamTimestamp`), so
/// high-latency sinks (Bluetooth, large quantum) get a proportionally larger
/// cushion instead of being clipped at a fixed guess (#69). Until the first
/// callback measurement lands we fall back to this floor.
const PLAYBACK_TAIL_PAD_FLOOR: Duration = Duration::from_millis(250);

/// Safety margin added on top of the measured output latency before it is used
/// as the pad — small jitter cushion so a measurement at the exact buffer
/// boundary still over-covers rather than re-arming the mic a hair early (#69).
const PLAYBACK_LATENCY_MARGIN: Duration = Duration::from_millis(50);

/// Sentinel meaning "no latency measured yet" for the shared atomic. Real
/// measured latencies are small (single- to low-double-digit ms); `u64::MAX`
/// micros can never be a genuine reading, so it unambiguously selects the floor.
const LATENCY_UNMEASURED: u64 = u64::MAX;

pub struct CpalAudioSink {
    device_name: String,
    producer: Mutex<Option<ringbuf::HeapProd<f32>>>,
    /// Wall-clock instant at which all queued audio will have finished playing,
    /// or `None` when nothing is queued. Each `play` extends it by the real-time
    /// duration of the samples it adds; `is_playing` is true until it passes.
    ///
    /// This tracks *time*, not ring-buffer occupancy, on purpose. The device
    /// pulls samples out of the ring buffer into its own hardware/OS buffer well
    /// ahead of when they actually sound, so any occupancy- or callback-driven
    /// signal goes false long before playback is audibly done — which let the
    /// daemon re-arm the mic mid-reply and record its own speech. The queued
    /// samples have a known duration (count / sample_rate), so the honest answer
    /// to "is it still playing?" is "has that much wall-clock time elapsed yet?"
    playback_end: Mutex<Option<Instant>>,
    stream_running: Arc<AtomicBool>,
    /// Device output latency in microseconds, measured live in the output
    /// callback from `OutputStreamTimestamp` (the delta between when a buffer is
    /// handed to the device and when it will actually sound). Drives the tail
    /// pad in `is_playing` so high-latency sinks get a larger cushion than the
    /// fixed floor (#69). `LATENCY_UNMEASURED` until the first callback fires.
    measured_latency_micros: Arc<AtomicU64>,
}

impl CpalAudioSink {
    pub fn new(device_name: &str) -> Self {
        Self {
            device_name: device_name.to_string(),
            producer: Mutex::new(None),
            playback_end: Mutex::new(None),
            stream_running: Arc::new(AtomicBool::new(false)),
            measured_latency_micros: Arc::new(AtomicU64::new(LATENCY_UNMEASURED)),
        }
    }

    /// The tail pad to add to the playback deadline, derived from the most
    /// recent measured output latency: `max(latency + margin, floor)`. Before
    /// any measurement exists, returns the floor. Pure function of its inputs so
    /// the latency/pad math is unit-testable without a live device.
    fn tail_pad_from_latency(latency_micros: u64) -> Duration {
        if latency_micros == LATENCY_UNMEASURED {
            return PLAYBACK_TAIL_PAD_FLOOR;
        }
        let measured = Duration::from_micros(latency_micros) + PLAYBACK_LATENCY_MARGIN;
        measured.max(PLAYBACK_TAIL_PAD_FLOOR)
    }

    /// Current tail pad, reading the live measured latency.
    fn tail_pad(&self) -> Duration {
        Self::tail_pad_from_latency(self.measured_latency_micros.load(Ordering::Relaxed))
    }

    fn find_output_device(name: &str) -> Result<cpal::Device, VoiceError> {
        let host = cpal::default_host();

        if name == "default" {
            return host
                .default_output_device()
                .ok_or_else(|| VoiceError::Audio("no default output device".into()));
        }

        host.output_devices()
            .map_err(|e| VoiceError::Audio(format!("failed to enumerate output devices: {e}")))?
            .find(|d| {
                d.description()
                    .map(|desc| desc.name().contains(name))
                    .unwrap_or(false)
            })
            .ok_or_else(|| VoiceError::Audio(format!("output device '{name}' not found")))
    }

    /// Open the output stream if not already open.
    pub fn open(&self) -> Result<(), VoiceError> {
        let mut prod_guard = self
            .producer
            .lock()
            .map_err(|e| VoiceError::Audio(format!("lock poisoned: {e}")))?;

        if prod_guard.is_some() {
            return Ok(());
        }

        let device_name = self.device_name.clone();
        let stream_running = Arc::clone(&self.stream_running);
        let measured_latency = Arc::clone(&self.measured_latency_micros);

        let rb = HeapRb::<f32>::new(OUTPUT_RING_BUFFER_CAPACITY);
        let (producer, consumer) = rb.split();

        // Consumer goes to the stream thread; producer stays here
        let consumer = Arc::new(Mutex::new(consumer));

        let (result_tx, result_rx) = std::sync::mpsc::channel();

        // cpal::Stream is !Send, so manage it on a dedicated thread
        let consumer_clone = Arc::clone(&consumer);
        std::thread::Builder::new()
            .name("audio-playback".into())
            .spawn(move || {
                let device = match Self::find_output_device(&device_name) {
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
                tracing::info!(device = %dev_name, "opening output device");

                let config = StreamConfig {
                    channels: CHANNELS,
                    sample_rate: SAMPLE_RATE,
                    buffer_size: cpal::BufferSize::Default,
                };

                let consumer_cb = consumer_clone;
                let latency_cb = measured_latency;

                let stream = match device.build_output_stream(
                    &config,
                    move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                        // Measure real output latency: the device tells us when
                        // this buffer will physically sound (`playback`) vs when
                        // the callback ran (`callback`); their delta is the
                        // output latency the tail pad must cover (#69). Some
                        // hosts can report playback < callback transiently;
                        // `duration_since` returns None there, so we just skip
                        // the update and keep the last good reading.
                        let ts = info.timestamp();
                        if let Some(latency) = ts.playback.duration_since(&ts.callback) {
                            latency_cb.store(latency.as_micros() as u64, Ordering::Relaxed);
                        }

                        let mut cons = match consumer_cb.lock() {
                            Ok(c) => c,
                            Err(_) => {
                                data.fill(0.0);
                                return;
                            }
                        };
                        use ringbuf::traits::Consumer;
                        let filled = cons.pop_slice(data);
                        for sample in &mut data[filled..] {
                            *sample = 0.0;
                        }
                    },
                    move |err| {
                        tracing::error!("output stream error: {err}");
                    },
                    None,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = result_tx.send(Err(VoiceError::Audio(format!(
                            "failed to build output stream: {e}"
                        ))));
                        return;
                    }
                };

                if let Err(e) = stream.play() {
                    let _ = result_tx.send(Err(VoiceError::Audio(format!(
                        "failed to start output stream: {e}"
                    ))));
                    return;
                }

                stream_running.store(true, Ordering::SeqCst);
                let _ = result_tx.send(Ok(()));

                // Keep the stream alive until stopped
                while stream_running.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }

                drop(stream);
                tracing::info!("output playback thread exiting");
            })
            .map_err(|e| VoiceError::Audio(format!("failed to spawn playback thread: {e}")))?;

        result_rx
            .recv()
            .map_err(|_| VoiceError::Audio("playback thread exited unexpectedly".into()))??;

        *prod_guard = Some(producer);
        Ok(())
    }

    /// Push the playback deadline out by the real-time duration of the freshly
    /// queued audio. `samples` is the count of *interleaved* f32 samples (one
    /// per channel per frame), exactly as handed to `play`/`push_slice`. The
    /// real-time duration is `frames / SAMPLE_RATE`, and `frames = samples /
    /// CHANNELS`, hence the divide by `SAMPLE_RATE * CHANNELS`. At CHANNELS==1
    /// (the current config) samples == frames, but spelling out CHANNELS keeps
    /// the math correct if the output ever goes stereo — otherwise a channel
    /// bump would silently halve the deadline and re-arm the mic mid-tail.
    ///
    /// If audio is still playing, stack onto the current deadline so back-to-back
    /// sentences accumulate; otherwise start the clock from now.
    fn extend_playback_deadline(&self, samples: usize) {
        let added =
            Duration::from_secs_f64(samples as f64 / (SAMPLE_RATE as f64 * CHANNELS as f64));
        if let Ok(mut end) = self.playback_end.lock() {
            let now = Instant::now();
            let base = match *end {
                Some(prev) if prev > now => prev,
                _ => now,
            };
            *end = Some(base + added);
        }
    }
}

impl AudioSink for CpalAudioSink {
    fn play(&self, samples: Vec<f32>) -> Result<(), VoiceError> {
        self.open()?;

        let mut prod_guard = self
            .producer
            .lock()
            .map_err(|e| VoiceError::Audio(format!("lock poisoned: {e}")))?;

        if let Some(ref mut producer) = *prod_guard {
            let written = producer.push_slice(&samples);
            if written < samples.len() {
                tracing::debug!(
                    dropped = samples.len() - written,
                    "output ring buffer overflow"
                );
            }
            // Only extend the deadline when something was actually queued. A
            // zero-write (empty buffer, or a full ring that dropped everything)
            // would otherwise set playback_end = now and keep `is_playing` true
            // for the whole tail pad with nothing playing (#71).
            if written > 0 {
                self.extend_playback_deadline(written);
            }
        }

        Ok(())
    }

    fn stop(&self) -> Result<(), VoiceError> {
        self.stream_running.store(false, Ordering::SeqCst);
        // Barge-in/stop discards the queue, so nothing is outstanding.
        if let Ok(mut end) = self.playback_end.lock() {
            *end = None;
        }

        let mut prod_guard = self
            .producer
            .lock()
            .map_err(|e| VoiceError::Audio(format!("lock poisoned: {e}")))?;
        *prod_guard = None;

        tracing::info!("audio playback stop requested");
        Ok(())
    }

    fn is_playing(&self) -> bool {
        // True until the queued audio's real-time duration (plus a tail pad for
        // output latency) has elapsed — i.e. until it has actually finished
        // sounding, not merely been handed to the device. The pad tracks the
        // measured device latency so high-latency sinks don't re-arm the mic
        // mid-tail (#69).
        let pad = self.tail_pad();
        match self.playback_end.lock() {
            Ok(end) => match *end {
                Some(deadline) => Instant::now() < deadline + pad,
                None => false,
            },
            Err(_) => false,
        }
    }

    fn in_tail_pad(&self) -> bool {
        // The audio deadline has passed (nothing fresh is sounding) but we're
        // still inside the latency cushion that keeps `is_playing` true (#70).
        let pad = self.tail_pad();
        match self.playback_end.lock() {
            Ok(end) => match *end {
                Some(deadline) => {
                    let now = Instant::now();
                    now >= deadline && now < deadline + pad
                }
                None => false,
            },
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Remaining time on the playback deadline, for asserting accumulation
    /// without sleeping through full reply durations.
    fn remaining(sink: &CpalAudioSink) -> Duration {
        sink.playback_end
            .lock()
            .unwrap()
            .map(|end| end.saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::ZERO)
    }

    #[test]
    fn idle_sink_is_not_playing() {
        let sink = CpalAudioSink::new("default");
        assert!(!sink.is_playing(), "nothing queued ⇒ not playing");
    }

    #[test]
    fn queueing_marks_playing_immediately() {
        // The regression: `is_playing` must read true the instant audio is
        // queued, without waiting for a hardware callback to fire. One second
        // of audio (SAMPLE_RATE mono frames) is plainly still playing.
        let sink = CpalAudioSink::new("default");
        sink.extend_playback_deadline(SAMPLE_RATE as usize);
        assert!(sink.is_playing(), "freshly queued audio reads as playing");
        assert!(
            remaining(&sink) > Duration::from_millis(900),
            "≈1s of audio should leave ≈1s on the clock, got {:?}",
            remaining(&sink)
        );
    }

    #[test]
    fn back_to_back_sentences_accumulate() {
        // Two half-second batches queued back-to-back must stack to ≈1s, not
        // collapse to the last batch — otherwise the wait ends a sentence early.
        let sink = CpalAudioSink::new("default");
        let half_second = (SAMPLE_RATE / 2) as usize;
        sink.extend_playback_deadline(half_second);
        sink.extend_playback_deadline(half_second);
        assert!(
            remaining(&sink) > Duration::from_millis(900),
            "two 0.5s batches should accumulate to ≈1s, got {:?}",
            remaining(&sink)
        );
    }

    #[test]
    fn deadline_elapses_after_the_audio_duration() {
        // A short clip stops reading as playing once its duration + tail pad
        // passes. 10ms of audio + 250ms pad ⇒ done well before 400ms.
        let sink = CpalAudioSink::new("default");
        sink.extend_playback_deadline((SAMPLE_RATE / 100) as usize); // 10ms
        assert!(sink.is_playing(), "still playing right after queueing");
        std::thread::sleep(Duration::from_millis(400));
        assert!(!sink.is_playing(), "done after duration + tail pad elapses");
    }

    #[test]
    fn stop_clears_the_deadline() {
        // Barge-in: stop must immediately report not-playing even with audio
        // still on the clock.
        let sink = CpalAudioSink::new("default");
        sink.extend_playback_deadline(SAMPLE_RATE as usize * 10); // 10s queued
        assert!(sink.is_playing());
        sink.stop().unwrap();
        assert!(!sink.is_playing(), "stop discards the queue ⇒ not playing");
    }

    // ----- #71: a zero-write must not arm a phantom busy window. -----

    #[test]
    fn zero_write_does_not_arm_a_busy_window() {
        // play() must not extend the deadline when nothing was queued, else
        // `is_playing` stays true for the whole tail pad with no audio. We test
        // the guard at its source: extending by zero samples sets the deadline
        // to `now`, which then keeps is_playing true for the pad — so play()
        // skips it. Here we assert the sink stays idle when no real audio lands.
        let sink = CpalAudioSink::new("default");
        // Simulate the play() guard: written == 0 ⇒ no extend call.
        assert!(
            !sink.is_playing(),
            "no audio queued ⇒ not playing, even with a fresh sink"
        );
    }

    #[test]
    fn extend_by_zero_would_arm_the_pad_so_play_must_skip_it() {
        // Documents *why* the `written > 0` guard exists: extending by zero
        // pins the deadline at now, which is still < now + pad, so is_playing
        // would wrongly read true. This is the bug the guard in play() prevents.
        let sink = CpalAudioSink::new("default");
        sink.extend_playback_deadline(0);
        assert!(
            sink.is_playing(),
            "extending by zero arms the pad — hence play() guards on written > 0"
        );
    }

    // ----- #72: pin the interleaved-sample duration math. -----

    #[test]
    fn deadline_math_treats_count_as_interleaved_samples() {
        // The deadline is frames/SAMPLE_RATE with frames = samples / CHANNELS.
        // Pin it so a CHANNELS change can't silently halve (or double) the
        // deadline. At the current CHANNELS==1, SAMPLE_RATE samples == 1s.
        let sink = CpalAudioSink::new("default");
        let one_second_of_samples = SAMPLE_RATE as usize * CHANNELS as usize;
        sink.extend_playback_deadline(one_second_of_samples);
        let left = remaining(&sink);
        assert!(
            left > Duration::from_millis(950) && left < Duration::from_millis(1050),
            "SAMPLE_RATE*CHANNELS interleaved samples must be ~1s, got {left:?}"
        );
    }

    #[test]
    fn deadline_math_scales_with_sample_count() {
        // Half as many samples ⇒ half the duration. Guards the divisor: if it
        // dropped CHANNELS the ratio would still hold at CHANNELS==1, so we also
        // pin the absolute value above; together they fix both factors.
        let sink = CpalAudioSink::new("default");
        let half = (SAMPLE_RATE as usize * CHANNELS as usize) / 2;
        sink.extend_playback_deadline(half);
        let left = remaining(&sink);
        assert!(
            left > Duration::from_millis(450) && left < Duration::from_millis(550),
            "half a second of samples must be ~0.5s, got {left:?}"
        );
    }

    // ----- #69: tail pad derived from measured output latency. -----

    #[test]
    fn pad_falls_back_to_floor_before_any_measurement() {
        assert_eq!(
            CpalAudioSink::tail_pad_from_latency(LATENCY_UNMEASURED),
            PLAYBACK_TAIL_PAD_FLOOR,
            "with no measurement yet the pad is the floor"
        );
    }

    #[test]
    fn low_latency_does_not_over_deafen() {
        // A fast sink (2ms latency) must not balloon the pad — the floor caps
        // it so we don't keep the mic deaf longer than necessary.
        let pad = CpalAudioSink::tail_pad_from_latency(2_000); // 2ms
        assert_eq!(
            pad, PLAYBACK_TAIL_PAD_FLOOR,
            "low latency stays at the floor, got {pad:?}"
        );
    }

    #[test]
    fn high_latency_extends_the_pad_to_cover_it() {
        // A high-latency sink (e.g. Bluetooth, ~300ms) must get a pad that
        // covers its latency plus the margin — well past the 250ms floor — or
        // the mic re-arms mid-tail and records the daemon's own voice.
        let latency = Duration::from_millis(300);
        let pad = CpalAudioSink::tail_pad_from_latency(latency.as_micros() as u64);
        assert_eq!(
            pad,
            latency + PLAYBACK_LATENCY_MARGIN,
            "high latency ⇒ pad = latency + margin"
        );
        assert!(
            pad > PLAYBACK_TAIL_PAD_FLOOR,
            "a 300ms-latency sink must exceed the 250ms floor, got {pad:?}"
        );
    }

    #[test]
    fn pad_is_monotonic_in_latency() {
        // More device latency never yields a smaller pad.
        let lo = CpalAudioSink::tail_pad_from_latency(10_000);
        let hi = CpalAudioSink::tail_pad_from_latency(400_000);
        assert!(hi >= lo, "pad must not shrink as latency grows");
    }

    // ----- #70: distinguish real audio from the tail pad. -----

    #[test]
    fn in_tail_pad_is_false_during_real_audio() {
        // 1s of audio queued: the deadline is in the future, so we're playing
        // real audio, not in the pad.
        let sink = CpalAudioSink::new("default");
        sink.extend_playback_deadline(SAMPLE_RATE as usize);
        assert!(sink.is_playing());
        assert!(
            !sink.in_tail_pad(),
            "while real audio is sounding we are not in the tail pad"
        );
    }

    #[test]
    fn in_tail_pad_is_true_after_deadline_but_within_pad() {
        // A 10ms clip: after ~150ms the audio deadline has passed but the
        // 250ms floor pad still keeps is_playing true — that is the tail pad.
        let sink = CpalAudioSink::new("default");
        sink.extend_playback_deadline((SAMPLE_RATE / 100) as usize); // 10ms
        std::thread::sleep(Duration::from_millis(150));
        assert!(sink.is_playing(), "still inside the pad");
        assert!(
            sink.in_tail_pad(),
            "deadline passed but within the pad ⇒ in the tail pad"
        );
    }

    #[test]
    fn in_tail_pad_is_false_when_idle() {
        let sink = CpalAudioSink::new("default");
        assert!(!sink.in_tail_pad(), "nothing queued ⇒ not in a pad");
    }

    #[test]
    fn measured_latency_drives_is_playing_pad() {
        // End-to-end through the live atomic: queue a tiny clip, then inject a
        // high measured latency and confirm the clip is still "playing" past
        // the floor but inside the measured pad.
        let sink = CpalAudioSink::new("default");
        sink.extend_playback_deadline((SAMPLE_RATE / 100) as usize); // 10ms
        // Inject 500ms of measured device latency.
        sink.measured_latency_micros
            .store(500_000, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            sink.is_playing(),
            "10ms audio + ~550ms pad must still read playing after 300ms"
        );
    }
}
