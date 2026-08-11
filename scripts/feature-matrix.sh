#!/usr/bin/env bash
# Feature-combination check for adele-voice-module.
#
#   ./scripts/feature-matrix.sh
#
# The module selects its backend adapters with cargo features. A feature set
# compiles in one arrangement and breaks in another, and several of its tests
# exist only in an arrangement the default build never produces, so each
# arrangement is linted and RUN here rather than only compiled.
#
# This is not the power set. It checks the empty set, each optional backend on
# its own, each pair that a consumer actually selects, and the full set. A
# combination is worth adding when a `cfg` starts to distinguish it.
#
# ONNX Runtime. `ort` publishes no prebuilt binary for some targets, notably
# x86_64-apple-darwin, which is the reason these features exist at all. On such
# a host the combinations that select `vad-silero` or `tts-kokoro` are run with
# `adele-voice-vad-silero/load-dynamic`, which resolves the library at runtime
# instead of linking it, so they are still covered rather than skipped. They are
# skipped only when that does not work either, and then by name.
set -euo pipefail

PKG=adele-voice-module
# Pulls the vad-silero crate into the graph with dlopen linking. It does NOT
# enable the module's own `vad-silero` feature, so it can be added to any
# combination without changing which backends that combination selects.
ORT_DYNAMIC=adele-voice-vad-silero/load-dynamic

failures=()
skipped=()

# Lint and run one combination. Clippy rather than `cargo check`, because the
# workspace denies `clippy::all` as well as rustc warnings, and `--all-targets`
# so the test code is linted too.
check() {
    local label="$1"
    shift
    echo
    echo "=== ${label}"
    if ! cargo clippy -p "$PKG" --all-targets "$@" -- -D warnings; then
        echo "--- FAILED (lint): ${label}"
        failures+=("$label (lint)")
        return
    fi
    if ! cargo test -p "$PKG" "$@"; then
        echo "--- FAILED (test): ${label}"
        failures+=("$label (test)")
        return
    fi
    echo "--- ok: ${label}"
}

# A combination that pulls ONNX Runtime. Adds the dlopen escape where the host
# needs it, and skips only if even that fails.
check_ort() {
    local label="$1"
    shift
    if [ "$ORT_LINKS" = yes ]; then
        check "$label" "$@"
    elif [ "$ORT_DLOPEN_WORKS" = yes ]; then
        check "${label} [load-dynamic]" "$@" --features "$ORT_DYNAMIC"
    else
        echo
        echo "=== ${label}"
        echo "--- skipped: ort neither links nor loads dynamically on this host"
        skipped+=("$label")
    fi
}

# Probe how `ort` can be built here. Two probes, because "cannot link" and
# "vad-silero is broken" are different facts and only the first is a reason to
# change how the matrix runs. If the crate fails BOTH ways the fault is the
# crate, not the platform, and that is reported as a failure, not a skip.
echo "=== probing how ort builds on this host"
if cargo check -p adele-voice-vad-silero; then
    ORT_LINKS=yes
    ORT_DLOPEN_WORKS=yes
    echo "--- ort links; ONNX-Runtime combinations run as-is"
elif cargo check -p adele-voice-vad-silero --features load-dynamic; then
    ORT_LINKS=no
    ORT_DLOPEN_WORKS=yes
    echo "--- ort does not link here; ONNX-Runtime combinations run with load-dynamic"
else
    ORT_LINKS=no
    ORT_DLOPEN_WORKS=no
    echo "--- adele-voice-vad-silero builds NEITHER way; this is a crate fault, not a platform one"
    failures+=("adele-voice-vad-silero does not build with or without load-dynamic")
fi

# Nothing optional selected: the always-present floor (cpal audio + Piper TTS).
check "no default features" --no-default-features

# Each optional backend on its own.
check_ort "only vad-silero" --no-default-features --features vad-silero
check "only stt-whisper" --no-default-features --features stt-whisper
check "only tts-polly" --no-default-features --features tts-polly
check_ort "only tts-kokoro" --no-default-features --features tts-kokoro

# VAD and STT together, the pair `build_dictation` needs.
check_ort "vad-silero + stt-whisper" --no-default-features --features vad-silero,stt-whisper

# The set a macOS consumer selects today: Whisper STT paired with a platform VAD
# of its own, plus cloud TTS, and no ort.
check "no-ort set (stt-whisper + tts-polly)" --no-default-features --features stt-whisper,tts-polly

# One local ONNX backend without the other, which the `cfg`s treat separately.
check_ort "vad-silero + tts-kokoro" --no-default-features --features vad-silero,tts-kokoro

# Every backend, which is what `default` selects and what every current
# consumer builds.
check_ort "all features" --all-features

echo
echo "=== ort stays out of the no-ort set"
# Read the graph, not the build: `ort` can enter a graph unasked-for and still
# compile on a host that has a prebuilt for it, so a build check cannot see this
# property at all. This runs on every host.
#
# `cargo tree -i` inverts the graph and reports what depends on `ort`. It also
# exits non-zero for a mistyped feature, a renamed package, or a lockfile it
# cannot update - so a bare "did it fail?" would report those as success and
# assert nothing for ever. Establish that the command works at all first, then
# require that `ort` is absent from one graph and present in another.
ort_in_graph() {
    cargo tree -p "$PKG" --no-default-features --features "$1" --prefix none --edges normal \
        | grep -qE '^ort v'
}

if ! cargo tree -p "$PKG" --no-default-features --features stt-whisper,tts-polly >/dev/null; then
    echo "--- FAILED: cannot read the dependency tree at all; the check below proves nothing"
    failures+=("cargo tree is usable")
elif ort_in_graph stt-whisper,tts-polly; then
    echo "--- FAILED: ort is in the graph without vad-silero or tts-kokoro"
    cargo tree -p "$PKG" --no-default-features --features stt-whisper,tts-polly -i ort
    failures+=("ort absent from the no-ort set")
elif ! ort_in_graph tts-kokoro; then
    # The positive control. Without it, every way of failing to find `ort`
    # (renamed crate, changed output format) reads as the property holding.
    echo "--- FAILED: ort is not in the tts-kokoro graph either, so this check cannot detect it"
    failures+=("ort detectable at all")
else
    echo "--- ok: no ort in the no-ort set, and the check can still see ort when it is there"
fi

echo
if [ ${#skipped[@]} -gt 0 ]; then
    echo "NOT COVERED on this host (${#skipped[@]}):"
    printf '  - %s\n' "${skipped[@]}"
    echo
fi

if [ ${#failures[@]} -gt 0 ]; then
    echo "FAILED (${#failures[@]}):"
    printf '  - %s\n' "${failures[@]}"
    exit 1
fi
echo "All feature combinations checked here are green."
