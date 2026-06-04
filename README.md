# adele-voice

Local-first voice interface for the [Adelie](https://github.com/adelie-ai) desktop assistant — **wake word, speech-to-text, and text-to-speech**, wired to the assistant daemon over D-Bus.

Say **"Hey Adele"** (or press to talk), speak a prompt, and hear the assistant's reply read back. Speech recognition and speech synthesis both run **entirely on your machine**.

> **"Adele" vs "Adelie":** *Adele* is the assistant persona — and the wake word. *Adelie* is the project/brand. Same family, different scope.

---

## Privacy — what is actually listening, and where audio goes

Trust matters more than features here, so this section is deliberately precise. It describes **what the code does today**; anything not yet implemented is marked _planned_ and links to the issue that will deliver it.

### When is the microphone capturing?

The pipeline is a small state machine, and the microphone is treated very differently in each state:

| State | What the mic is doing | Where that audio goes |
|---|---|---|
| **Idle** | Local wake-word scanning only — each frame is fed to the on-device "Hey Adele" detector and immediately discarded | nowhere; never leaves RAM |
| **Listening** | Capturing your utterance (entered via the wake word **or** push-to-talk) | buffered locally for transcription |
| **Processing** | Mic input ignored | — |
| **Speaking** | Monitored only for *barge-in* so you can interrupt the reply | nowhere |

**Captured audio never leaves your machine, in any state.** There is no cloud speech service anywhere in this project.

### Where do speech-to-text and text-to-speech run?

- **Speech-to-text:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (via [`whisper-rs`](https://github.com/tazz4843/whisper-rs)), running the Whisper **distil-large-v3** model **in-process, on your own CPU**. Your voice is transcribed locally.
- **Text-to-speech:** [Piper](https://github.com/rhasspy/piper) — a **neural (VITS) on-device** synthesizer. This is *not* eSpeak / formant synthesis; replies are spoken with a real neural voice, locally.

### What actually leaves the machine?

Only the **transcribed prompt text** — and only as far as you've already chosen to send your typed messages. The flow is:

```
your voice ──(local Whisper STT)──▶ text ──D-Bus──▶ desktop-assistant daemon ──▶ your configured LLM backend
```

The LLM backend is whatever you have configured in the desktop assistant. It **may be local** (e.g. [Ollama](https://ollama.com)) **or a cloud provider** (e.g. Anthropic, OpenAI, Bedrock). If you've configured a cloud model, your prompt text goes there — exactly as it would if you had typed it. The voice layer adds **no** network egress of its own.

### Honest current limitations (being addressed)

- While the daemon is running, wake-word listening is **on by default**, and disabling it today gates *processing* but does **not yet release the microphone device**. A user-facing **"Enable 'Hey Adele'"** toggle and a true mic release are _planned_ ([#3](https://github.com/adelie-ai/voice/issues/3), [#5](https://github.com/adelie-ai/voice/issues/5)).
- The daemon currently runs **continuously** (systemd, from login). _Planned:_ on-demand start that exits when idle, so it isn't resident unless you're actually using it ([#5](https://github.com/adelie-ai/voice/issues/5)).
- Exposing the on-device voice to other apps (e.g. as an accessibility backend) via a `SayText` service is _planned_ ([#4](https://github.com/adelie-ai/voice/issues/4)).

---

## Architecture

Ports-and-adapters (hexagonal). `core` defines the domain, the state machine, and the port traits; each capability is an isolated adapter crate, so any engine can be swapped without touching the pipeline.

| Crate | Role | Backend |
|---|---|---|
| `core` | domain, state machine, port traits | — |
| `audio-cpal` | microphone input / speaker output | [cpal](https://github.com/RustAudio/cpal) |
| `wake-rustpotter` | wake-word detection | [rustpotter](https://github.com/GiviMAD/rustpotter) (`hey-adele.rpw`) |
| `vad-silero` | voice-activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX) |
| `stt-whisper` | speech-to-text | whisper.cpp (distil-large-v3) |
| `tts-piper` | text-to-speech | Piper |
| `assistant-dbus` | send prompts / receive streamed responses | zbus → `org.desktopAssistant.Conversations` |
| `dbus-interface` | control + status surface | zbus ← `org.desktopAssistant.Voice` |
| `daemon` | wires the pipeline together | — |

## D-Bus control surface

The daemon owns **`org.desktopAssistant.Voice`** at `/org/desktopAssistant/Voice`:

| Member | Kind | Purpose |
|---|---|---|
| `GetState() → s` | method | `Idle` / `Listening` / `Processing` / `Speaking` |
| `SetEnabled(b)` / `GetEnabled() → b` | method | toggle wake-word processing |
| `PushToTalk()` | method | skip the wake word, start listening now |
| `StopSpeaking()` | method | cancel current playback |
| `StateChanged` / `TranscriptReady` / `SpeakingText` | signals | UI updates |

If the voice service isn't running, this name simply isn't on the bus — so dependent UI should degrade gracefully.

## Install & run

> ⚠️ Models and the `piper` binary are **not bundled**.

**Quick start** — the setup script downloads the VAD/STT/TTS models into `~/.local/share/adele-voice/models/`, checks for `piper`, and reports what's still missing. It's idempotent (safe to re-run):

```sh
./scripts/setup.sh
```

The one thing it can't automate is the **wake word**: `hey-adele.rpw` is trained from your *own* recordings (rustpotter ships no "Hey Adele"). The script prints the exact [`rustpotter-cli`](https://github.com/GiviMAD/rustpotter-cli) steps — record a few 16 kHz mono clips of the phrase, then `rustpotter-cli build-model`.

Then build/install (and optionally configure input/output devices, model paths, wake sensitivity, STT language, and the silence timeout):

```sh
cargo install --path crates/daemon         # installs `adele-voice`
$EDITOR ~/.config/adele-voice/config.toml  # optional; sensible defaults if absent
```

The desktop-assistant daemon must be running and exposing `org.desktopAssistant.Conversations` for prompts to be answered.

## Status

Early but functional: the full pipeline (wake → VAD → STT → assistant → streamed TTS, with barge-in and push-to-talk) is implemented. Active work is tracked on the [Adelie AI Roadmap](https://github.com/orgs/adelie-ai/projects) board — provisioning, the "Enable 'Hey Adele'" / record controls in the clients, on-demand activation, a `SayText` accessibility service, continuous conversation, and TTS voice selection.

## License

[AGPL-3.0-or-later](LICENSE). Each crate also declares this in its manifest.
