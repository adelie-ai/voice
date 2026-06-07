#!/usr/bin/env bash
# adele-voice setup: fetch the runtime models, check for the `piper` binary,
# and guide wake-word training. Re-runnable — existing files are skipped.
#
#   ./scripts/setup.sh
#
# Models land in $XDG_DATA_HOME/adele-voice/models (default
# ~/.local/share/adele-voice/models), matching the daemon's config defaults
# in crates/daemon/src/config.rs. Override any source URL below if needed.
set -euo pipefail

MODELS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/adele-voice/models"

# NOTE: the Silero URL is pinned to v4.0 deliberately — the vad-silero adapter
# uses the h/c/hn/cn LSTM tensor interface, which v5 replaced with a single
# "state" tensor. Do not bump to v5 without updating crates/vad-silero.
VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/v4.0/files/silero_vad.onnx"
STT_URL="https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin"
TTS_ONNX_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
TTS_JSON_URL="${TTS_ONNX_URL}.json"

# Kokoro — the default neural TTS. Quantized (q8f16) model + a few voices. The
# tts-kokoro adapter caps the onnxruntime optimizer so q8f16 loads cleanly.
KOKORO_BASE="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"
KOKORO_MODEL_URL="$KOKORO_BASE/onnx/model_q8f16.onnx"
KOKORO_VOICES="af_heart af_bella am_michael am_fenrir bf_emma bm_george"

c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_off=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$c_blue" "$c_off" "$*"; }
warn() { printf '%swarning:%s %s\n' "$c_yellow" "$c_off" "$*" >&2; }
err()  { printf '%serror:%s %s\n'   "$c_red"    "$c_off" "$*" >&2; }

# The runtime wake-word detector (crates/wake-rustpotter/src/lib.rs) configures
# rustpotter for 16 kHz, MONO, F32 — see SAMPLE_RATE = 16_000 in
# crates/core/src/domain.rs. rustpotter extracts MFCC features at *exactly* that
# rate/layout, so a model whose training clips were captured at the device default
# (commonly 44.1 kHz stereo / 32-bit float) has its features at the wrong rate and
# scores ~0 against live audio — the wake word never fires (#46). Every training
# clip MUST therefore be 16 kHz mono before `rustpotter-cli build`.
WAKE_SAMPLE_RATE=16000

# Convert a wav to 16 kHz mono 16-bit PCM in place-by-name, via ffmpeg or sox
# (whichever is on PATH). 16-bit int is what rustpotter-cli's own examples use and
# `build` happily accepts it (it normalizes internally); the detector's F32 runtime
# format is unrelated to the on-disk training format — only rate + channel count
# matter for feature extraction.
convert_16k_mono() { # convert_16k_mono <src.wav> <dst.wav>
  local src="$1" dst="$2"
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -hide_banner -loglevel error -y \
      -i "$src" -ac 1 -ar "$WAKE_SAMPLE_RATE" -sample_fmt s16 "$dst"
  elif command -v sox >/dev/null 2>&1; then
    sox "$src" -r "$WAKE_SAMPLE_RATE" -c 1 -b 16 "$dst"
  else
    err "neither ffmpeg nor sox is on PATH — cannot down-convert wake-word clips to 16 kHz mono."
    echo "    Install one of them and re-run:" >&2
    echo "      sudo pacman -S ffmpeg     # Arch / CachyOS" >&2
    echo "      sudo apt install ffmpeg   # Debian / Ubuntu" >&2
    return 1
  fi
}

fetch() { # fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    log "have $(basename "$dest") ($(du -h "$dest" | cut -f1)) — skipping"
    return 0
  fi
  log "downloading $(basename "$dest") ..."
  curl --fail --location --progress-bar -o "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
}

log "models dir: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

fetch "$VAD_URL"      "$MODELS_DIR/silero_vad.onnx"
fetch "$STT_URL"      "$MODELS_DIR/ggml-distil-large-v3.bin"
fetch "$TTS_ONNX_URL" "$MODELS_DIR/en_US-amy-medium.onnx"
fetch "$TTS_JSON_URL" "$MODELS_DIR/en_US-amy-medium.onnx.json"

fetch "$KOKORO_MODEL_URL" "$MODELS_DIR/kokoro.onnx"
mkdir -p "$MODELS_DIR/kokoro-voices"
for v in $KOKORO_VOICES; do
  fetch "$KOKORO_BASE/voices/$v.bin" "$MODELS_DIR/kokoro-voices/$v.bin"
done

# --- piper (text-to-speech) binary ---
if command -v piper >/dev/null 2>&1; then
  log "piper found: $(command -v piper)"
else
  warn "piper not found on PATH. Install it, for example:"
  echo "    pipx install piper-tts        # or: pip install --user piper-tts"
  echo "    # or a release binary: https://github.com/rhasspy/piper/releases"
fi

# --- espeak-ng (phonemizer for the default Kokoro TTS) ---
if command -v espeak-ng >/dev/null 2>&1; then
  log "espeak-ng found: $(command -v espeak-ng)"
else
  warn "espeak-ng not found — required by the default Kokoro TTS. Install it:"
  echo "    sudo pacman -S espeak-ng      # Arch / CachyOS"
  echo "    sudo apt install espeak-ng    # Debian / Ubuntu"
fi

# --- wake word model (interactive; needs your voice) ---
#
# The model must be TRAINED from your own recordings — rustpotter ships no
# "Hey Adele". This block records the clips, force-converts each to 16 kHz mono
# (see the WAKE_SAMPLE_RATE note above; this is the #46 fix), builds the .rpw, and
# then VALIDATES it by self-detecting against a training clip so we never silently
# ship a dead model.
#
# Set RETRAIN_WAKE=1 to re-record even when hey-adele.rpw already exists.
WAKE_MODEL="$MODELS_DIR/hey-adele.rpw"
WAKE_CLIPS="${WAKE_CLIPS:-8}"           # number of clips to record
WAKE_MS="${WAKE_MS:-2000}"              # record duration per clip (ms)
WAKE_NAME="hey adele"
# Validate at the runtime's effective gate: the detector disables the averaged-score
# pre-gate (avg_threshold = 0.0, see lib.rs), so we must too — otherwise a healthy
# model can look "dead" purely because of the averaging window. Threshold here is the
# self-detect floor, kept comfortably below a real fire (the rebuilt 16 kHz model in
# #46 self-scored 0.73).
WAKE_VALIDATE_THRESHOLD="${WAKE_VALIDATE_THRESHOLD:-0.3}"

train_wake_word() {
  if ! command -v rustpotter-cli >/dev/null 2>&1; then
    err "rustpotter-cli not found — needed to train the wake word. Install it:"
    echo "    cargo install rustpotter-cli" >&2
    return 1
  fi
  # Fail early if we have no way to guarantee 16 kHz mono.
  if ! command -v ffmpeg >/dev/null 2>&1 && ! command -v sox >/dev/null 2>&1; then
    err "neither ffmpeg nor sox is on PATH — cannot guarantee 16 kHz mono clips (required, #46)."
    echo "    Install one (e.g. 'sudo pacman -S ffmpeg' / 'sudo apt install ffmpeg') and re-run." >&2
    return 1
  fi

  local workdir; workdir="$(mktemp -d "${TMPDIR:-/tmp}/adele-wake.XXXXXX")"
  # shellcheck disable=SC2064  # expand workdir now, on trap-install
  trap "rm -rf '$workdir'" RETURN

  log "training wake word \"$WAKE_NAME\" — recording $WAKE_CLIPS clips of \"Hey Adele\""
  echo "    Speak clearly, same tone/distance each time, in a quiet room."
  echo "    Each clip records for $((WAKE_MS / 1000))s after you press Enter."

  local converted=()
  local i raw conv
  for i in $(seq 1 "$WAKE_CLIPS"); do
    raw="$workdir/raw-$i.wav"
    conv="$workdir/clip-$i.wav"
    read -r -p "    [$i/$WAKE_CLIPS] press Enter, then say \"Hey Adele\"... " _
    # `record --sample-rate` is only a *preferred* rate and never sets channel count
    # or sample format, so the device can still hand back 44.1 kHz stereo. We always
    # re-convert below — recording at 16k just minimizes the conversion when supported.
    rustpotter-cli record --sample-rate "$WAKE_SAMPLE_RATE" --ms "$WAKE_MS" "$raw"
    if ! convert_16k_mono "$raw" "$conv"; then
      err "failed to convert clip $i to 16 kHz mono — aborting wake-word training."
      return 1
    fi
    converted+=("$conv")
  done

  log "building $WAKE_MODEL from $WAKE_CLIPS 16 kHz mono clips"
  rustpotter-cli build --name "$WAKE_NAME" --path "$WAKE_MODEL" "${converted[@]}"

  # --- VALIDATE: the freshly built model must self-detect a training clip. A model
  # built from wrong-rate audio scores ~0 against 16 kHz input (the #46 bug), so this
  # is the guard against silently shipping a dead model. ---
  log "validating $WAKE_MODEL (self-detect against a training clip)"
  local hits
  hits="$(rustpotter-cli test \
            --threshold "$WAKE_VALIDATE_THRESHOLD" --averaged-threshold 0.0 --eager \
            "$WAKE_MODEL" "${converted[0]}" 2>/dev/null \
          | grep -c 'Wakeword detection' || true)"
  if [[ "$hits" -gt 0 ]]; then
    log "${c_green}wake-word model validated${c_off} — self-detected (>= $WAKE_VALIDATE_THRESHOLD) on a training clip."
  else
    err "wake-word model FAILED self-detection — it scored below $WAKE_VALIDATE_THRESHOLD on its OWN training clip."
    echo "    This usually means the clips were not truly 16 kHz mono, or the recordings" >&2
    echo "    were too quiet/noisy. The model at $WAKE_MODEL will almost certainly never" >&2
    echo "    fire on live audio. Re-run with RETRAIN_WAKE=1 to record again." >&2
    return 1
  fi

  echo "    Tune wake_word.sensitivity in ~/.config/adele-voice/config.toml (lower = more sensitive)."
}

if [[ -s "$WAKE_MODEL" && "${RETRAIN_WAKE:-0}" != "1" ]]; then
  log "have hey-adele.rpw — skipping (set RETRAIN_WAKE=1 to re-record)"
elif [[ ! -t 0 ]]; then
  # No TTY: recording needs the mic and an interactive prompt. Don't fail the whole
  # setup run (model fetches above still succeeded) — just explain how to train.
  warn "hey-adele.rpw is missing and stdin is not a TTY — wake-word training needs your"
  warn "microphone interactively. Re-run this script from a terminal to record it, or:"
  echo "    cargo install rustpotter-cli   # if not already installed" >&2
  echo "    ./scripts/setup.sh             # records 16 kHz mono clips, builds & validates" >&2
else
  if ! train_wake_word; then
    warn "wake-word training did not complete — see the error above. The rest of setup is done."
  fi
fi

# --- summary ---
echo
log "asset status in $MODELS_DIR:"
for f in silero_vad.onnx ggml-distil-large-v3.bin kokoro.onnx en_US-amy-medium.onnx en_US-amy-medium.onnx.json hey-adele.rpw; do
  if [[ -s "$MODELS_DIR/$f" ]]; then
    printf '  %sok%s   %s\n' "$c_green" "$c_off" "$f"
  else
    printf '  %sMISS%s %s\n' "$c_red" "$c_off" "$f"
  fi
done
if command -v piper >/dev/null 2>&1; then
  printf '  %sok%s   piper binary\n' "$c_green" "$c_off"
else
  printf '  %sMISS%s piper binary\n' "$c_red" "$c_off"
fi

echo
log "build & install the daemon:  cargo install --path crates/daemon"
log "then run:  adele-voice   (requires the desktop-assistant daemon on D-Bus)"
