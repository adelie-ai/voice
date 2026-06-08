//! cpal-backed audio capture and playback for the voice pipeline.
//!
//! # The audio-device model (and its sharp edges)
//!
//! The pipeline speaks exactly one format: **16 kHz mono `f32`**. Everything
//! here exists to bridge whatever a real input device offers to that contract.
//! Linux audio makes that surprisingly fiddly, so read this before touching
//! capture.
//!
//! ## Two kinds of input device
//!
//! cpal talks to **ALSA**, and on a PipeWire/Pulse desktop the ALSA PCMs split
//! into two very different categories:
//!
//! - **Shared sound-server routes** — `default`, `pipewire`, `pulse`, `jack`.
//!   These go through the sound server, which owns the hardware and *mixes*
//!   access. The server resamples/downmixes for us and lets many apps capture
//!   the same mic at once. This is almost always the right choice.
//! - **Raw ALSA cards** — `hw:CARD=…`, `plughw:…`, `sysdefault:CARD=…`,
//!   `front:CARD=…`, etc. These talk to the hardware directly. They expose only
//!   the device's *native* formats (a USB mic is often 48 kHz, multi-channel,
//!   `s24`/`s16` — never 16 kHz mono), and they take the card **exclusively**.
//!
//! ## Three gotchas, and how this crate handles them
//!
//! 1. **Don't force a format — negotiate it.** Building a stream with a
//!    hardcoded `StreamConfig { 16 kHz, mono }` makes a raw 48 kHz device reject
//!    it in `snd_pcm_hw_params` (`Invalid argument (22)`) and crash-loop capture.
//!    `negotiate_input_config` instead queries the device's supported configs,
//!    prefers an exact 16 kHz-mono passthrough (the server routes give us this
//!    for free), and otherwise opens the device's native config and **downmixes
//!    to mono + streaming-resamples to 16 kHz in software** (`to_mono_f32` + the
//!    rubato resampler). So any device works.
//!
//! 2. **Raw cards are exclusive — they can lock the mic away from everyone
//!    else.** Because a raw card takes the hardware directly, an *always-on*
//!    capture (the wake word never stops listening) on one blocks other apps —
//!    and, with two users logged in, the other session entirely (logind hands
//!    the `/dev/snd` ACLs to whichever session is active; each user runs its own
//!    PipeWire). Going through a shared route avoids this: the server mixes, so
//!    Teams/Chrome/etc. and the assistant coexist on one mic. To verify sharing
//!    works, run a second capturer (`parecord --device=<source>`) while the
//!    daemon runs — it should get real audio, not "device busy".
//!
//! 3. **Device selection is substring matching, so prefer shared routes.** The
//!    configured `input_device` is resolved by substring-matching cpal's device
//!    *description* (see `CpalAudioSource::find_input_device`), which is why a
//!    value like `"Mini"` can land on a raw `front:CARD=Mini` PCM.
//!    `resolve_input_device` guards against the exclusivity trap: if the
//!    configured device resolves to a raw card and a shared route exists, it
//!    captures via the shared route instead and logs a warning. `default` is
//!    left untouched — it's already the system's chosen (shared) input.
//!
//! ## Guidance
//!
//! - **Set `input_device` to `default` (or `pipewire`/`pulse`).** This shares
//!   the mic and resamples for free. It's the default and the recommended pick.
//! - **To target a *specific* mic, set it as the system default source**
//!   (KDE/GNOME sound settings or `wpctl set-default`) and let the assistant
//!   follow it — don't pin a raw `CARD=` device, which is exclusive.
//! - Raw cards remain selectable as a fallback (e.g. a headless box with no
//!   sound server), and the picker / `list-devices` flag them as exclusive.

mod sink;
mod source;

pub use sink::CpalAudioSink;
pub use source::{CpalAudioSource, InputDeviceInfo, probe_input_devices};
