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
# and still compile on Linux. The graph is therefore asserted directly.
set -euo pipefail

PKG=adele-voice-module

failures=()

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

# Nothing optional selected: the always-present floor (cpal audio + Piper TTS).
check "no default features" --no-default-features

# Each optional backend on its own, which is the arrangement most likely to
# expose a `cfg` that names a type another feature brings in.
for feat in vad-silero stt-whisper tts-polly tts-kokoro; do
    check "only ${feat}" --no-default-features --features "$feat"
done

# VAD and STT together, the pair `build_dictation` needs.
check "vad-silero + stt-whisper" --no-default-features --features vad-silero,stt-whisper

# The set a macOS consumer selects today: speech output and cloud TTS, no ort.
check "no-ort set (stt-whisper + tts-polly)" --no-default-features --features stt-whisper,tts-polly

# Every backend, which is what `default` selects and what Linux consumers build.
check "all features" --all-features

echo
echo "=== ort stays out of the no-ort set"
# `cargo tree -i` inverts the graph: it reports what depends on `ort`. It exits
# non-zero when the package is absent from the graph, which is the passing case
# here, so the success and failure branches are deliberately inverted.
if cargo tree -p "$PKG" --no-default-features \
    --features stt-whisper,tts-polly -i ort >/dev/null 2>&1; then
    echo "--- FAILED: ort is in the graph without vad-silero or tts-kokoro"
    cargo tree -p "$PKG" --no-default-features --features stt-whisper,tts-polly -i ort
    failures+=("ort absent from the no-ort set")
else
    echo "--- ok: no ort in the graph"
fi

echo
if [ ${#failures[@]} -gt 0 ]; then
    echo "FAILED (${#failures[@]}):"
    printf '  - %s\n' "${failures[@]}"
    exit 1
fi
echo "All feature combinations check out."
