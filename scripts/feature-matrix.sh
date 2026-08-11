#!/usr/bin/env bash
# Feature-combination check for adele-voice-module.
#
#   ./scripts/feature-matrix.sh
#
# The module selects its backend adapters with cargo features. A feature set
# compiles in one arrangement and breaks in another, so each arrangement gets
# an explicit check here rather than trusting the default build to cover them.
#
# The last check is the one that matters most: a consumer that selects no
# ONNX-Runtime backend must get a dependency graph with no `ort` in it. `ort`
# publishes no prebuilt ONNX Runtime for x86_64-apple-darwin, so an `ort` that
# arrives unasked-for fails the build of every consumer on that target, voice
# code or not. A build check alone cannot see this: `ort` can enter the graph
# and still compile on Linux, and only fails where no prebuilt exists.
#
# On a host that cannot link `ort` at all, the combinations that select an
# ONNX-Runtime backend are reported as skipped rather than failed. Skipping is
# not silent: the run names each combination it did not cover, because the
# point of this script is to say what was checked.
set -euo pipefail

PKG=adele-voice-module

failures=()
skipped=()

check() {
    local label="$1"
    shift
    echo
    echo "=== ${label}"
    if cargo check -p "$PKG" --all-targets "$@"; then
        echo "--- ok: ${label}"
    else
        echo "--- FAILED: ${label}"
        failures+=("$label")
    fi
}

# Combinations that pull ONNX Runtime. Run only where `ort` links.
check_ort() {
    local label="$1"
    shift
    if [ "$ORT_LINKS" = yes ]; then
        check "$label" "$@"
    else
        echo
        echo "=== ${label}"
        echo "--- skipped: this host cannot link ort"
        skipped+=("$label")
    fi
}

# Probe rather than encode ort's support matrix: build the smallest crate that
# needs it and see. A host without a prebuilt `ort` fails in the build script,
# before any of this crate's own code compiles.
echo "=== probing whether ort links on this host"
if cargo check -p adele-voice-vad-silero >/dev/null 2>&1; then
    ORT_LINKS=yes
    echo "--- ort links; the full matrix will run"
else
    ORT_LINKS=no
    echo "--- ort does not link here; ONNX-Runtime combinations will be skipped"
fi

# Nothing optional selected: the always-present floor (cpal audio + Piper TTS).
check "no default features" --no-default-features

# Each optional backend on its own, which is the arrangement most likely to
# expose a `cfg` that names a type another feature brings in.
check_ort "only vad-silero" --no-default-features --features vad-silero
check "only stt-whisper" --no-default-features --features stt-whisper
check "only tts-polly" --no-default-features --features tts-polly
check_ort "only tts-kokoro" --no-default-features --features tts-kokoro

# VAD and STT together, the pair `build_dictation` needs.
check_ort "vad-silero + stt-whisper" --no-default-features --features vad-silero,stt-whisper

# The set a macOS consumer selects today: speech input and cloud TTS, no ort.
check "no-ort set (stt-whisper + tts-polly)" --no-default-features --features stt-whisper,tts-polly

# Every backend, which is what `default` selects and what Linux consumers build.
check_ort "all features" --all-features

echo
echo "=== ort stays out of the no-ort set"
# `cargo tree -i` inverts the graph: it reports what depends on `ort`. It exits
# non-zero when the package is absent from the graph, which is the passing case
# here, so the success and failure branches are deliberately inverted. This
# check reads the graph, not the build, so it runs on every host.
if cargo tree -p "$PKG" --no-default-features \
    --features stt-whisper,tts-polly -i ort >/dev/null 2>&1; then
    echo "--- FAILED: ort is in the graph without vad-silero or tts-kokoro"
    cargo tree -p "$PKG" --no-default-features --features stt-whisper,tts-polly -i ort
    failures+=("ort absent from the no-ort set")
else
    echo "--- ok: no ort in the graph"
fi

echo
if [ ${#skipped[@]} -gt 0 ]; then
    echo "NOT COVERED on this host (${#skipped[@]}), because ort does not link here:"
    printf '  - %s\n' "${skipped[@]}"
    echo "Run this on a host with a prebuilt ort to cover them."
    echo
fi

if [ ${#failures[@]} -gt 0 ]; then
    echo "FAILED (${#failures[@]}):"
    printf '  - %s\n' "${failures[@]}"
    exit 1
fi
echo "All feature combinations checked here are green."
