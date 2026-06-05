//! Local neural TTS via Kokoro-82M (ONNX). Text is phonemized with espeak-ng
//! (IPA, with stress), each phoneme character is mapped to a Kokoro token, the
//! quantized ONNX model runs on the CPU with a per-voice style vector selected
//! by token-sequence length, and the 24kHz output is resampled to the pipeline
//! rate. Fully on-device — no cloud, no per-utterance cost.
//!
//! Reference: github.com/thewh1teagle/kokoro-onnx (vocab, IO, voice format).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use adele_voice_core::VoiceError;
use adele_voice_core::domain::SAMPLE_RATE;
use adele_voice_core::ports::tts::TextToSpeech;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use rubato::Resampler;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use tokio::process::Command;

const KOKORO_SAMPLE_RATE: u32 = 24_000;
/// Max phonemes the model accepts (it pads with token 0 at both ends).
const MAX_PHONEME_LENGTH: usize = 510;
/// Per-voice style-vector dimension.
const STYLE_DIM: usize = 256;
/// A voice `.bin` is a [510, 1, 256] f32 table: row L is the style for an
/// L-token utterance.
const VOICE_ROWS: usize = 510;

/// Kokoro phoneme→token map (from the model's config.json vocab; sparse).
#[rustfmt::skip]
static VOCAB_TABLE: &[(char, i64)] = &[
    ('\u{003B}', 1), ('\u{003A}', 2), ('\u{002C}', 3), ('\u{002E}', 4), ('\u{0021}', 5),
    ('\u{003F}', 6), ('\u{2014}', 9), ('\u{2026}', 10), ('\u{0022}', 11), ('\u{0028}', 12),
    ('\u{0029}', 13), ('\u{201C}', 14), ('\u{201D}', 15), ('\u{0020}', 16), ('\u{0303}', 17),
    ('\u{02A3}', 18), ('\u{02A5}', 19), ('\u{02A6}', 20), ('\u{02A8}', 21), ('\u{1D5D}', 22),
    ('\u{AB67}', 23), ('\u{0041}', 24), ('\u{0049}', 25), ('\u{004F}', 31), ('\u{0051}', 33),
    ('\u{0053}', 35), ('\u{0054}', 36), ('\u{0057}', 39), ('\u{0059}', 41), ('\u{1D4A}', 42),
    ('\u{0061}', 43), ('\u{0062}', 44), ('\u{0063}', 45), ('\u{0064}', 46), ('\u{0065}', 47),
    ('\u{0066}', 48), ('\u{0068}', 50), ('\u{0069}', 51), ('\u{006A}', 52), ('\u{006B}', 53),
    ('\u{006C}', 54), ('\u{006D}', 55), ('\u{006E}', 56), ('\u{006F}', 57), ('\u{0070}', 58),
    ('\u{0071}', 59), ('\u{0072}', 60), ('\u{0073}', 61), ('\u{0074}', 62), ('\u{0075}', 63),
    ('\u{0076}', 64), ('\u{0077}', 65), ('\u{0078}', 66), ('\u{0079}', 67), ('\u{007A}', 68),
    ('\u{0251}', 69), ('\u{0250}', 70), ('\u{0252}', 71), ('\u{00E6}', 72), ('\u{03B2}', 75),
    ('\u{0254}', 76), ('\u{0255}', 77), ('\u{00E7}', 78), ('\u{0256}', 80), ('\u{00F0}', 81),
    ('\u{02A4}', 82), ('\u{0259}', 83), ('\u{025A}', 85), ('\u{025B}', 86), ('\u{025C}', 87),
    ('\u{025F}', 90), ('\u{0261}', 92), ('\u{0265}', 99), ('\u{0268}', 101), ('\u{026A}', 102),
    ('\u{029D}', 103), ('\u{026F}', 110), ('\u{0270}', 111), ('\u{014B}', 112), ('\u{0273}', 113),
    ('\u{0272}', 114), ('\u{0274}', 115), ('\u{00F8}', 116), ('\u{0278}', 118), ('\u{03B8}', 119),
    ('\u{0153}', 120), ('\u{0279}', 123), ('\u{027E}', 125), ('\u{027B}', 126), ('\u{0281}', 128),
    ('\u{027D}', 129), ('\u{0282}', 130), ('\u{0283}', 131), ('\u{0288}', 132), ('\u{02A7}', 133),
    ('\u{028A}', 135), ('\u{028B}', 136), ('\u{028C}', 138), ('\u{0263}', 139), ('\u{0264}', 140),
    ('\u{03C7}', 142), ('\u{028E}', 143), ('\u{0292}', 147), ('\u{0294}', 148), ('\u{02C8}', 156),
    ('\u{02CC}', 157), ('\u{02D0}', 158), ('\u{02B0}', 162), ('\u{02B2}', 164), ('\u{2193}', 169),
    ('\u{2192}', 171), ('\u{2197}', 172), ('\u{2198}', 173), ('\u{1D7B}', 177),
];

fn vocab() -> &'static HashMap<char, i64> {
    static M: OnceLock<HashMap<char, i64>> = OnceLock::new();
    M.get_or_init(|| VOCAB_TABLE.iter().copied().collect())
}

/// A loaded voice: its style table, flattened [VOICE_ROWS * STYLE_DIM].
#[derive(Clone)]
struct KokoroVoice {
    name: String,
    styles: Arc<Vec<f32>>,
}

/// Text-to-speech via the Kokoro-82M ONNX model. Cloning shares the model
/// session and the active-voice state, mirroring the other backends.
#[derive(Clone)]
pub struct KokoroTts {
    session: Arc<Mutex<Session>>,
    voices_dir: PathBuf,
    voice: Arc<RwLock<KokoroVoice>>,
    espeak_lang: String,
}

impl KokoroTts {
    /// Load the model and the initial voice. `lang` is an espeak-ng voice such
    /// as "en-us" or "en-gb".
    pub fn new(
        model_path: &Path,
        voices_dir: &Path,
        voice_name: &str,
        lang: &str,
    ) -> Result<Self, VoiceError> {
        let session = Session::builder()
            .map_err(|e| VoiceError::Tts(format!("kokoro session builder: {e}")))?
            // Cap below the layout (NCHWc) transformer, which SEGVs on the
            // quantized Kokoro graph in this onnxruntime build.
            .with_optimization_level(GraphOptimizationLevel::Level2)
            .map_err(|e| VoiceError::Tts(format!("kokoro optimization level: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                VoiceError::Tts(format!("load kokoro model {}: {e}", model_path.display()))
            })?;
        let voice = load_voice(voices_dir, voice_name)?;
        tracing::info!(model = %model_path.display(), voice = voice_name, "Kokoro TTS initialized");
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            voices_dir: voices_dir.to_owned(),
            voice: Arc::new(RwLock::new(voice)),
            espeak_lang: lang.to_string(),
        })
    }

    /// Hot-swap the active voice (a `<name>.bin` in the voices dir).
    pub fn set_voice(&self, voice_name: &str) -> Result<(), VoiceError> {
        let v = load_voice(&self.voices_dir, voice_name)?;
        *self.voice.write().expect("kokoro voice lock poisoned") = v;
        Ok(())
    }

    /// The current voice name.
    pub fn current_voice(&self) -> String {
        self.voice
            .read()
            .expect("kokoro voice lock poisoned")
            .name
            .clone()
    }

    /// Voice names available in the voices dir (`*.bin`), sorted.
    pub fn list_voices(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.voices_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                (p.extension().and_then(|x| x.to_str()) == Some("bin"))
                    .then(|| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
                    .flatten()
            })
            .collect();
        names.sort();
        names
    }
}

/// Read a voice `.bin` (raw f32 [510,1,256], little-endian) into a flat table.
fn load_voice(dir: &Path, name: &str) -> Result<KokoroVoice, VoiceError> {
    let path = dir.join(format!("{name}.bin"));
    let bytes = std::fs::read(&path)
        .map_err(|e| VoiceError::Tts(format!("read voice {}: {e}", path.display())))?;
    let expected = VOICE_ROWS * STYLE_DIM * 4;
    if bytes.len() != expected {
        return Err(VoiceError::Tts(format!(
            "voice {name}: size {} != expected {expected}",
            bytes.len()
        )));
    }
    let styles: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(KokoroVoice {
        name: name.to_string(),
        styles: Arc::new(styles),
    })
}

/// Phonemize text to IPA (with stress) via espeak-ng, then keep only characters
/// the model's vocab knows. espeak emits one line per clause; words are space-
/// separated, so collapsing whitespace preserves word boundaries (token 16).
async fn phonemize(text: &str, lang: &str) -> Result<String, VoiceError> {
    let output = Command::new("espeak-ng")
        .arg("-q")
        .arg("-v")
        .arg(lang)
        .arg("--ipa")
        .arg(text.trim())
        .output()
        .await
        .map_err(|e| VoiceError::Tts(format!("failed to run espeak-ng (is it installed?): {e}")))?;
    if !output.status.success() {
        return Err(VoiceError::Tts(format!(
            "espeak-ng failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    let v = vocab();
    let phonemes: String = raw
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .filter(|c| v.contains_key(c))
        .collect();
    Ok(phonemes)
}

impl TextToSpeech for KokoroTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, VoiceError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let phonemes = phonemize(text, &self.espeak_lang).await?;
        let v = vocab();
        let tokens: Vec<i64> = phonemes
            .chars()
            .filter_map(|c| v.get(&c).copied())
            .take(MAX_PHONEME_LENGTH)
            .collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Style vector for this utterance length (the unpadded token count).
        let style: Vec<f32> = {
            let voice = self.voice.read().expect("kokoro voice lock poisoned");
            let idx = tokens.len().min(VOICE_ROWS - 1);
            voice.styles[idx * STYLE_DIM..(idx + 1) * STYLE_DIM].to_vec()
        };

        // input_ids = [0, <tokens>, 0]
        let mut input_ids = Vec::with_capacity(tokens.len() + 2);
        input_ids.push(0i64);
        input_ids.extend_from_slice(&tokens);
        input_ids.push(0i64);
        let n = input_ids.len() as i64;

        // The ONNX run is blocking + CPU-heavy; keep it off the async runtime.
        let session = Arc::clone(&self.session);
        let audio_24k: Vec<f32> =
            tokio::task::spawn_blocking(move || -> Result<Vec<f32>, VoiceError> {
                let map_err = |e| VoiceError::Tts(format!("kokoro inference: {e}"));
                let ids = Tensor::from_array(([1i64, n], input_ids)).map_err(map_err)?;
                let style =
                    Tensor::from_array(([1i64, STYLE_DIM as i64], style)).map_err(map_err)?;
                let speed = Tensor::from_array(([1i64], vec![1.0f32])).map_err(map_err)?;

                let mut sess = session.lock().expect("kokoro session lock poisoned");
                let outputs = sess
                    .run(ort::inputs![
                        "input_ids" => ids.upcast(),
                        "style" => style.upcast(),
                        "speed" => speed.upcast(),
                    ])
                    .map_err(map_err)?;
                let (_, data) = outputs["waveform"]
                    .try_extract_tensor::<f32>()
                    .map_err(map_err)?;
                Ok(data.to_vec())
            })
            .await
            .map_err(|e| VoiceError::Tts(format!("kokoro task join: {e}")))??;

        let out = if KOKORO_SAMPLE_RATE == SAMPLE_RATE {
            audio_24k
        } else {
            resample(&audio_24k, KOKORO_SAMPLE_RATE, SAMPLE_RATE)?
        };
        tracing::debug!(
            text_len = text.len(),
            tokens = tokens.len(),
            samples = out.len(),
            "Kokoro synthesis complete"
        );
        Ok(out)
    }
}

/// Resample mono f32 audio between integer rates with rubato's FFT resampler.
fn resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>, VoiceError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = 1024;
    let mut resampler = rubato::Fft::<f32>::new(
        src_rate as usize,
        dst_rate as usize,
        chunk_size,
        1,
        1,
        rubato::FixedSync::Input,
    )
    .map_err(|e| VoiceError::Tts(format!("resampler init: {e}")))?;

    let input_len = input.len();
    let output_len = resampler.process_all_needed_output_len(input_len);
    let input_data = vec![input.to_vec()];
    let mut output_data = vec![vec![0.0f32; output_len]];

    let in_adapter = SequentialSliceOfVecs::new(&input_data, 1, input_len)
        .map_err(|e| VoiceError::Tts(format!("resampler input adapter: {e}")))?;
    let mut out_adapter = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)
        .map_err(|e| VoiceError::Tts(format!("resampler output adapter: {e}")))?;

    let (_, nbr_out) = resampler
        .process_all_into_buffer(&in_adapter, &mut out_adapter, input_len, None)
        .map_err(|e| VoiceError::Tts(format!("resampler process: {e}")))?;

    let mut out = output_data.into_iter().next().unwrap();
    out.truncate(nbr_out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_maps_known_phonemes_and_drops_unknown() {
        let v = vocab();
        assert_eq!(v.get(&'ˈ'), Some(&156)); // primary stress
        assert_eq!(v.get(&' '), Some(&16)); // word boundary
        assert_eq!(v.get(&'ə'), Some(&83)); // schwa
        assert_eq!(v.get(&'😀'), None); // unknown dropped
        assert_eq!(v.len(), VOCAB_TABLE.len());
    }
}
