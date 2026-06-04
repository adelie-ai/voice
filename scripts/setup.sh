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

c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_off=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$c_blue" "$c_off" "$*"; }
warn() { printf '%swarning:%s %s\n' "$c_yellow" "$c_off" "$*" >&2; }

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

# --- piper (text-to-speech) binary ---
if command -v piper >/dev/null 2>&1; then
  log "piper found: $(command -v piper)"
else
  warn "piper not found on PATH. Install it, for example:"
  echo "    pipx install piper-tts        # or: pip install --user piper-tts"
  echo "    # or a release binary: https://github.com/rhasspy/piper/releases"
fi

# --- wake word model (manual; needs your voice) ---
if [[ -s "$MODELS_DIR/hey-adele.rpw" ]]; then
  log "have hey-adele.rpw — skipping"
else
  warn "hey-adele.rpw is missing — it must be TRAINED from your own recordings:"
  cat <<TRAIN
    1. Record 5-10 clips of "Hey Adele" as 16kHz mono WAV, e.g.:
         arecord -f S16_LE -r 16000 -c 1 hey-adele-01.wav   # repeat, vary tone
    2. cargo install rustpotter-cli
    3. rustpotter-cli build-model --model-path hey-adele.rpw hey-adele-*.wav
    4. mv hey-adele.rpw "$MODELS_DIR"/
    Then tune wake_word.sensitivity in ~/.config/adele-voice/config.toml.
TRAIN
fi

# --- summary ---
echo
log "asset status in $MODELS_DIR:"
for f in silero_vad.onnx ggml-distil-large-v3.bin en_US-amy-medium.onnx en_US-amy-medium.onnx.json hey-adele.rpw; do
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
