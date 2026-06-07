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

**Captured audio never leaves your machine, in any state.** Speech *recognition* is always local. Speech *synthesis* is local by default (Kokoro or Piper); the only way any voice data leaves the machine is if you **opt into the AWS Polly TTS backend**, which sends the assistant's *reply text* (never your microphone audio) to AWS for synthesis — see [Text-to-speech backends](#text-to-speech-backends).

### Where do speech-to-text and text-to-speech run?

- **Speech-to-text:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (via [`whisper-rs`](https://github.com/tazz4843/whisper-rs)), running the Whisper **distil-large-v3** model **in-process, on your own CPU**. Your voice is transcribed locally.
- **Text-to-speech:** pluggable via `tts.backend`. The **default is [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** — a neural model running **on-device** (ONNX, roughly real-time on CPU). [Piper](https://github.com/rhasspy/piper) is an alternative local neural voice. **AWS Polly** (cloud) is available opt-in for its generative voices — see [Text-to-speech backends](#text-to-speech-backends). Only Polly sends text off-device; the local backends keep everything on your machine.

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
| `tts-kokoro` | text-to-speech (**default**) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (ONNX, local) |
| `tts-piper` | text-to-speech | [Piper](https://github.com/rhasspy/piper) (local) |
| `tts-polly` | text-to-speech | [AWS Polly](https://aws.amazon.com/polly/) (cloud, opt-in) |
| `assistant-connector` | send prompts / receive streamed responses | client-common `Connector` → UDS / WS / D-Bus |
| `dbus-interface` | control + status surface | zbus ← `org.desktopAssistant.Voice` |
| `module` | embeddable on-demand voice (`Dictation` + `Speaker`) | reuses the adapters above |
| `daemon` | the system **service** — consumes `module`, adds wake word + D-Bus | — |

### Embeddable module vs. system service

The reusable voice capabilities live in **`adele-voice-module`** — an embeddable library that does **dictation** and **speech playback** in-process, with **no wake word, no D-Bus, and no orchestrator coupling**. The daemon is just one consumer of it: the system-wide voice *service* that layers the wake word and the `org.desktopAssistant.Voice` surface on top.

| | Wake word | Dictation | Playback | D-Bus surface | Needs the daemon |
|---|:-:|:-:|:-:|:-:|:-:|
| **Daemon (service)** | ✅ | ✅ | ✅ | ✅ `SayText` / PTT / … | — |
| **Embedded (in a client)** | ❌ | ✅ | ✅ | ❌ | ❌ |

**The boundary:** the wake word and the `org.desktopAssistant.Voice` surface are **daemon-only**. Want hands-free "Hey Adele"? Run the daemon. Want a mic button in your own chat client that works even when no daemon is running? Embed the module — the client does STT/TTS locally and reaches only its own assistant connection for the LLM.

**Embedding it.** A client depends on `adele-voice-module` and wires the two verbs:

```rust
use adele_voice_module::{build_dictation, build_speaker};

// Dictation: tap a mic button → capture one utterance → transcript.
let mut dictation = build_dictation(&audio_cfg, &vad_cfg, &stt_cfg)?;
if let Some(text) = dictation.dictate().await? {
    // feed `text` into the app's normal "send a prompt" path
}

// Playback: read a reply aloud with the configured backend.
let speaker = build_speaker(&tts_cfg, &audio_cfg).await;
speaker.say(&reply).await?;
```

`build_dictation` / `build_speaker` take the same `[audio]` / `[vad]` / `[stt]` / `[tts]` config the daemon uses (re-exported as `adele_voice_module::config`), so a client gets the local-first backend selection (Kokoro → Piper fallback, **never** auto cloud) for free. The lower-level `Dictation` / `Speaker` / `Endpointer` / `Transcriber` / `TtsBackend` types are public too, for clients that manage their own audio devices. A client typically exposes this behind a config toggle (embedded vs. the daemon path), so a machine with no daemon still gets dictation and playback.

**Dependency approach.** Clients **path-dep** the voice crates for now (mirroring the existing adele-gtk ↔ desktop-assistant path-dep); a published or git dependency can follow once the API settles.

### Reaching the orchestrator

The voice service sends prompts to the desktop-assistant orchestrator and streams replies back. It runs **wherever the microphone is** — which need not be where the orchestrator runs — so the transport is configurable via `[assistant] transport` in `~/.config/adele-voice/config.toml`, reusing the chat clients' shared transport layer (`Connector`):

| `transport` | Reaches | Use |
|---|---|---|
| `uds` *(default)* | the local Unix socket (`$XDG_RUNTIME_DIR/adelie/sock`) | orchestrator on the same machine |
| `ws` | a (possibly remote) WebSocket — set `ws_url` | the mic and the orchestrator are on different machines |
| `dbus` | the local session bus (`org.desktopAssistant.Conversations`) | legacy local path |

```toml
[assistant]
transport = "uds"                            # "uds" (default) | "ws" | "dbus"
# ws_url      = "wss://host:11339/ws"        # when transport = "ws"
# socket_path = "/run/user/1000/adelie/sock" # override the default UDS path
```

The local UDS and WebSocket transports carry the per-request spoken-response hint as a native field; the legacy D-Bus transport folds it into the prompt. A bearer token is minted automatically via the local D-Bus minter (the same path the chat clients use); for a remote `ws` daemon, set `ws_jwt` or `ws_login_username` / `ws_login_password`.

## D-Bus control surface

The daemon owns **`org.desktopAssistant.Voice`** at `/org/desktopAssistant/Voice`:

| Member | Kind | Purpose |
|---|---|---|
| `GetState() → s` | method | `Idle` / `Listening` / `Processing` / `Speaking` |
| `SetEnabled(b)` / `GetEnabled() → b` | method | toggle wake-word processing |
| `PushToTalk()` | method | skip the wake word, start listening now — routes to the daemon's own session ("Voice Conversation") |
| `PushToTalkInConversation(s)` | method | like `PushToTalk()`, but dictate into the given orchestrator conversation id (`""` = own session). The in-chat mic button passes the focused conversation so the prompt + spoken reply land in that chat |
| `StopSpeaking()` | method | cancel current playback |
| `StateChanged` / `TranscriptReady` / `SpeakingText` | signals | UI updates |
| `SayText(s)` | method | speak text with the on-device voice — for **any** app, no orchestrator involved |
| `SynthesizeText(s) → ay` | method | return spoken text as WAV bytes (the caller routes its own audio) |
| `ListVoices() → a(sssu)` · `GetVoice() → si` · `SetVoice(si)` | methods | enumerate / read / switch the active voice at runtime |
| `Reload()` | method | re-read `config.toml` and apply changed tunables live (see below) |

**TTS is independent of the assistant orchestrator.** `SayText` and `SynthesizeText` are handled **directly by this daemon** — they synthesize speech without ever contacting `org.desktopAssistant.Conversations` or any LLM. Other apps (an accessibility tool, or a future Orca / speech-dispatcher shim) can therefore use the on-device voice purely as a system TTS service. If the voice service isn't running, the name simply isn't on the bus — so dependent UI should degrade gracefully.

### Live config reload

Tuning knobs in `~/.config/adele-voice/config.toml` take effect **without a service restart**. The daemon watches the config file (debounced) and re-reads it on edits made any way; a settings UI (e.g. the KDE KCM) can also call `Reload()` after writing for an instant apply. On change, the daemon diffs the new config against the running values and applies only what changed:

| Knob | On reload |
|---|---|
| `vad.speech_threshold` | hot-applied in place |
| `vad.silence_duration_ms` | hot-applied in place |
| `assistant.followup_timeout_ms` | hot-applied (next turn) |
| `assistant.conversation_mode` | hot-applied (next turn boundary) |
| `idle_exit_timeout_ms` | hot-applied (next idle check) |
| `wake_word.sensitivity` | wake detector **rebuilt** (rustpotter bakes the threshold in at construction) |
| `audio.input_device` / `audio.output_device` | **restart required** — the capture/playback stream isn't swapped live; the daemon logs a clear "restart required" note and applies every other changed knob |

Everything else (model paths, STT/TTS backend & voice, the orchestrator transport) still needs a restart — those rebuild expensive sessions or reconnect a socket. The TTS voice is the exception: switch it live via `SetVoice` (above).

## Text-to-speech backends

TTS is a swappable adapter chosen at startup via `tts.backend` in `~/.config/adele-voice/config.toml`. The default is **Kokoro** — local, natural, and free.

| Backend | Runs | Naturalness | Latency | Cost | Provision |
|---|---|---|---|---|---|
| **`kokoro`** (default) | on-device (ONNX) | high | ~real-time on CPU (RTF ≈ 0.7) | free | `just init-kokoro` (model + voices; needs `espeak-ng`) |
| `piper` | on-device | good | very fast | free | `just init-piper` (voice; needs the `piper` binary) |
| `polly` | **AWS cloud** | highest (generative) | network round-trip | per-character¹ | `just init-polly` (no model; AWS credentials) |

¹ Polly bills per character of synthesized text — roughly ½–2¢ per spoken reply (~$16 / 1M chars neural, ~$30 / 1M generative). Your microphone audio is never sent; only the assistant's reply text.

**Local-first, never billable by accident.** If the configured backend can't initialize, the daemon falls back to a **local** backend (Piper) — it will **never** switch to a billable cloud backend on its own. Run `adele-voice check-setup` to see what's available, what's configured, and the active voice.

### Switching backend and voice

```toml
[tts]
backend = "kokoro"             # "kokoro" | "piper" | "polly"

# kokoro (default)
kokoro_voice = "af_heart"      # any <name>.bin in the voices dir; af_*/am_* = US, bf_*/bm_* = GB
kokoro_lang  = "en-us"

# piper
# model_path = "~/.local/share/adele-voice/models/en_US-amy-medium.onnx"

# polly (cloud, opt-in)
# polly_voice  = "Ruth"
# polly_engine = "generative"  # "neural" (cheaper, every region) | "generative" (most natural)
# polly_region = "us-east-1"
```

The active voice can also be changed **at runtime** — without editing config — through the D-Bus voice API (`ListVoices` / `GetVoice` / `SetVoice`), which the KDE and GTK clients surface as a dropdown.

**Polly credentials** come from the standard AWS chain (env vars, a shared profile, STS, or IMDS). For the systemd service, point it at a profile with a drop-in (SSO / `credential_process` providers are excluded from the build — use a static-key or STS profile):

```ini
# ~/.config/systemd/user/adele-voice.service.d/aws.conf
[Service]
Environment=AWS_PROFILE=myprofile
Environment=AWS_REGION=us-east-1
```

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

**Running as a service.** `systemd/adele-voice.service` is D-Bus-activatable (`Type=dbus`): once installed it starts on demand when anything calls `org.desktopAssistant.Voice`, and also at login. Make it on-demand-only with `systemctl --user disable adele-voice`, or turn it off entirely with `systemctl --user mask adele-voice` (after which calls fail cleanly — the service simply isn't there). Set `idle_exit_timeout_ms` in the config to have the daemon exit when idle (wake word off, nothing playing) so it isn't resident between uses — activation restarts it on the next call. The session activation file is `systemd/dbus-1/org.desktopAssistant.Voice.service` (install to `~/.local/share/dbus-1/services/`).

## Status

Early but functional: the full pipeline (wake → VAD → STT → assistant → streamed TTS, with barge-in and push-to-talk) is implemented. Active work is tracked on the [Adelie AI Roadmap](https://github.com/orgs/adelie-ai/projects) board — provisioning, the "Enable 'Hey Adele'" / record controls in the clients, on-demand activation, a `SayText` accessibility service, continuous conversation, and TTS voice selection.

## License

[AGPL-3.0-or-later](LICENSE). Each crate also declares this in its manifest.
