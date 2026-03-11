use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::{CHANNELS, SAMPLE_RATE};
use adele_voice_core::ports::audio::AudioSource;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Producer, Split};
use tokio::sync::mpsc;

/// Ring buffer capacity: 1 second of audio at 16kHz.
const RING_BUFFER_CAPACITY: usize = SAMPLE_RATE as usize;

/// Chunk size sent through the channel: 20ms of audio.
const CHUNK_FRAMES: usize = SAMPLE_RATE as usize / 50;

pub struct CpalAudioSource {
    device_name: String,
    running: Arc<AtomicBool>,
}

impl CpalAudioSource {
    pub fn new(device_name: &str) -> Self {
        Self {
            device_name: device_name.to_string(),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    fn find_input_device(name: &str) -> Result<cpal::Device, VoiceError> {
        let host = cpal::default_host();

        if name == "default" {
            return host
                .default_input_device()
                .ok_or_else(|| VoiceError::Audio("no default input device".into()));
        }

        host.input_devices()
            .map_err(|e| VoiceError::Audio(format!("failed to enumerate input devices: {e}")))?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
            .ok_or_else(|| VoiceError::Audio(format!("input device '{name}' not found")))
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
                let device = match Self::find_input_device(&device_name) {
                    Ok(d) => d,
                    Err(e) => {
                        let _ = result_tx.send(Err(e));
                        return;
                    }
                };

                let dev_name = device.name().unwrap_or_else(|_| "unknown".into());
                tracing::info!(device = %dev_name, "opening input device");

                let config = StreamConfig {
                    channels: CHANNELS,
                    sample_rate: SampleRate(SAMPLE_RATE),
                    buffer_size: cpal::BufferSize::Default,
                };

                let rb = HeapRb::<f32>::new(RING_BUFFER_CAPACITY);
                let (mut producer, mut consumer) = rb.split();

                let stream = match device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let written = producer.push_slice(data);
                        if written < data.len() {
                            tracing::debug!(
                                dropped = data.len() - written,
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

                if let Err(e) = stream.play() {
                    let _ = result_tx.send(Err(VoiceError::Audio(format!(
                        "failed to start input stream: {e}"
                    ))));
                    return;
                }

                running_clone.store(true, Ordering::SeqCst);
                let _ = result_tx.send(Ok(()));

                // Drain loop: pull from ring buffer and send chunks
                let mut chunk_buf = vec![0.0f32; CHUNK_FRAMES];
                while running_clone.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(20));

                    let popped = consumer.pop_slice(&mut chunk_buf);
                    if popped > 0 {
                        if tx.blocking_send(chunk_buf[..popped].to_vec()).is_err() {
                            break; // receiver dropped
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
