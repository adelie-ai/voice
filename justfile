set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# --- Local verification ("local CI") ---
# Run locally instead of GitHub Actions. `install-hooks` wires `check` into a
# git pre-push hook so it runs automatically before every push.
check: fmt-check lint build test
fmt-check:
    cargo fmt --all --check
fmt:
    cargo fmt --all
lint:
    cargo clippy --workspace --all-targets -- -D warnings
build:
    cargo build --workspace
test:
    cargo test --workspace
test-integration:
    cargo test --workspace -- --ignored
premerge:
    git fetch origin
    git rebase origin/main
    just check
install-hooks:
    git config core.hooksPath .githooks
    @echo "pre-push hook active — bypass once with: git push --no-verify"

# --- provisioning, install & run ---------------------------------------------

models_dir := env_var_or_default("XDG_DATA_HOME", env_var("HOME") / ".local/share") / "adele-voice/models"
kokoro_base := "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"

# Base models every backend needs: Silero VAD + Whisper STT
models:
    #!/usr/bin/env bash
    set -euo pipefail
    d="{{models_dir}}"; mkdir -p "$d"
    get() { [ -s "$2" ] && echo "have $(basename "$2")" || { echo "fetching $(basename "$2")..."; curl -fL# "$1" -o "$2"; }; }
    get "https://raw.githubusercontent.com/snakers4/silero-vad/v4.0/files/silero_vad.onnx" "$d/silero_vad.onnx"
    get "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin" "$d/ggml-distil-large-v3.bin"

# Init the Kokoro backend (the default): quantized model + voices (needs espeak-ng)
init-kokoro:
    #!/usr/bin/env bash
    set -euo pipefail
    d="{{models_dir}}"; mkdir -p "$d/kokoro-voices"
    get() { [ -s "$2" ] && echo "have $(basename "$2")" || { echo "fetching $(basename "$2")..."; curl -fL# "$1" -o "$2"; }; }
    get "{{kokoro_base}}/onnx/model_q8f16.onnx" "$d/kokoro.onnx"
    for v in af_heart af_bella am_michael am_fenrir bf_emma bm_george; do get "{{kokoro_base}}/voices/$v.bin" "$d/kokoro-voices/$v.bin"; done
    command -v espeak-ng >/dev/null || echo "WARNING: espeak-ng not installed (sudo pacman -S espeak-ng | sudo apt install espeak-ng)"

# Init the Piper backend: the en_US-amy voice (needs the `piper` binary)
init-piper:
    #!/usr/bin/env bash
    set -euo pipefail
    d="{{models_dir}}"; mkdir -p "$d"
    base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
    get() { [ -s "$2" ] && echo "have $(basename "$2")" || { echo "fetching..."; curl -fL# "$1" -o "$2"; }; }
    get "$base" "$d/en_US-amy-medium.onnx"; get "$base.json" "$d/en_US-amy-medium.onnx.json"
    command -v piper >/dev/null || echo "WARNING: piper not installed (pipx install piper-tts)"

# Init the Polly backend: cloud — nothing to download; needs AWS credentials
init-polly:
    @echo 'Polly is cloud TTS (billable). Set [tts] backend = "polly" in config and provide AWS'
    @echo 'credentials (AWS_PROFILE / a static-key profile). See README > Text-to-speech backends.'

# Show which backends are provisioned/available and which is configured
check-setup:
    adele-voice check-setup

# Record wake-word clips into wake-samples/ (default 10), then build with `just train`
record count="10":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p wake-samples
    for i in $(seq -w 1 {{count}}); do read -rp "Press Enter, then say 'Hey Adele' (clip $i)... "; rustpotter-cli record --ms 2000 "wake-samples/hey-adele-$i.wav"; done
    echo "Recorded. Build the model with: just train"

# Build the wake-word model from wake-samples/ into the models dir
train:
    rustpotter-cli build --name "hey adele" --path "{{models_dir}}/hey-adele.rpw" wake-samples/hey-adele-*.wav

# Build + install the daemon to ~/.cargo/bin
install:
    cargo install --path crates/daemon --locked

# Install the systemd user service + D-Bus activation file
install-service:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p ~/.config/systemd/user ~/.local/share/dbus-1/services
    install -m644 systemd/adele-voice.service ~/.config/systemd/user/adele-voice.service
    install -m644 systemd/dbus-1/org.desktopAssistant.Voice.service ~/.local/share/dbus-1/services/
    systemctl --user daemon-reload
    echo "Installed. Enable with: just enable"

# Enable + start the daemon as a user service
enable:
    systemctl --user enable --now adele-voice

restart:
    systemctl --user restart adele-voice
logs:
    journalctl --user -u adele-voice -f
status:
    systemctl --user status adele-voice --no-pager

# Run the daemon in the foreground (dev; ctrl-c to stop)
run:
    RUST_LOG=info cargo run -p adele-voice

# Full setup: base models + the default (Kokoro) backend + build + service
setup: models init-kokoro install install-service enable
    @echo "Set up. Train the wake word with: just record && just train"
