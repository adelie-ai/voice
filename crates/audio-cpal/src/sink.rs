use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::{CHANNELS, SAMPLE_RATE};
use adele_voice_core::ports::audio::AudioSink;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use ringbuf::HeapRb;
use ringbuf::traits::{Producer, Split};

/// Ring buffer capacity for output: 2 seconds of audio at 16kHz.
const OUTPUT_RING_BUFFER_CAPACITY: usize = SAMPLE_RATE as usize * 2;

pub struct CpalAudioSink {
    device_name: String,
    producer: Mutex<Option<ringbuf::HeapProd<f32>>>,
    playing: Arc<AtomicBool>,
    stream_running: Arc<AtomicBool>,
}

impl CpalAudioSink {
    pub fn new(device_name: &str) -> Self {
        Self {
            device_name: device_name.to_string(),
            producer: Mutex::new(None),
            playing: Arc::new(AtomicBool::new(false)),
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
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
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
        let playing = Arc::clone(&self.playing);
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

                let dev_name = device.name().unwrap_or_else(|_| "unknown".into());
                tracing::info!(device = %dev_name, "opening output device");

                let config = StreamConfig {
                    channels: CHANNELS,
                    sample_rate: SAMPLE_RATE,
                    buffer_size: cpal::BufferSize::Default,
                };

                let playing_cb = playing;
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
                        playing_cb.store(filled > 0, Ordering::Relaxed);
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
        }

        Ok(())
    }

    fn stop(&self) -> Result<(), VoiceError> {
        self.stream_running.store(false, Ordering::SeqCst);
        self.playing.store(false, Ordering::Relaxed);

        let mut prod_guard = self
            .producer
            .lock()
            .map_err(|e| VoiceError::Audio(format!("lock poisoned: {e}")))?;
        *prod_guard = None;

        tracing::info!("audio playback stop requested");
        Ok(())
    }

    fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }
}
