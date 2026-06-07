use std::sync::atomic::{AtomicBool, Ordering};
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

/// Padding added to the computed playback deadline so `is_playing` stays true
/// until the sound is truly done. Covers the gap between queueing audio and it
/// physically sounding: stream-open latency on the first sentence plus the
/// device/OS output buffer the callback fills ahead of playout. Erring long is
/// safe — the daemon waits a hair past the tail; erring short re-arms the mic
/// mid-word and records the daemon's own voice.
const PLAYBACK_TAIL_PAD: Duration = Duration::from_millis(250);

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
}

impl CpalAudioSink {
    pub fn new(device_name: &str) -> Self {
        Self {
            device_name: device_name.to_string(),
            producer: Mutex::new(None),
            playback_end: Mutex::new(None),
            stream_running: Arc::new(AtomicBool::new(false)),
        }
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

                let stream = match device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
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

    /// Push the playback deadline out by the real-time duration of `samples`
    /// freshly queued frames (mono @ SAMPLE_RATE, so frames == samples). If
    /// audio is still playing, stack onto the current deadline so back-to-back
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
            self.extend_playback_deadline(written);
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
        // sounding, not merely been handed to the device.
        match self.playback_end.lock() {
            Ok(end) => match *end {
                Some(deadline) => Instant::now() < deadline + PLAYBACK_TAIL_PAD,
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
}
