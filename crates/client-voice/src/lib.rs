//! Shared client-side voice-output plumbing for the chat clients.
//!
//! The GTK and TUI clients (and, partially, the voice daemon) each grew their
//! own copy of the same client-side voice-output logic: the three-way `Adele:`
//! output level, the narration gate that decides whether a reply or a `say_this`
//! aside is spoken, the system-refinement prose attached on send while a
//! conversation is spoken, and the sentence chunker that feeds the one-shot TTS
//! synth. Three copies meant three chances to drift — and the two long
//! refinement constants in particular **must stay byte-identical across
//! clients**, which copy-paste cannot guarantee (they had already diverged).
//!
//! This crate is the single owner of those pieces (desktop-assistant#274). It
//! holds *decisions and data*, not UI: there is no GTK, no ratatui, no D-Bus,
//! and no transport here. Each client keeps only its own UI bindings (buttons,
//! keybinds, the daemon/embedded speaker handles) and consults this crate for
//! the model + the gate + the chunker.
//!
//! It lives in the `voice` workspace rather than `desktop-assistant`'s
//! `client-common` because [`into_speakable_sentences`] reuses
//! [`adele_voice_core::sentence_buffer::SentenceBuffer`] — the same chunking the
//! daemon's streaming pipeline uses — so the natural home is alongside the voice
//! domain crate the consumers already path-dep. Putting it in `client-common`
//! would instead force the orchestrator daemon (which also links `client-common`)
//! to take a dependency on the whole voice stack.

#[cfg(feature = "chunker")]
use std::time::Duration;

#[cfg(feature = "chunker")]
use adele_voice_core::sentence_buffer::SentenceBuffer;

/// The three-way voice-**output** level for a conversation, exposed by the
/// `Adele:` control. A dedicated enum (not two bools) because the level is
/// genuinely three-valued and the gate logic differs per variant; a bool pair
/// would admit a nonsensical "both" state and scatter the
/// `Disabled`/`OnDemand`/`Always` distinction across call sites. The default is
/// [`AdeleOutput::Disabled`] (never speaks).
///
/// It replaces the earlier pair of independent toggles, which mapped directly:
/// the read-aloud toggle was [`AdeleOutput::Always`] and the voice-mode toggle
/// was [`AdeleOutput::OnDemand`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdeleOutput {
    /// Never speaks (the default). A `say_this` aside downgrades to shown text.
    #[default]
    Disabled,
    /// Adele speaks **on demand**: the written reply is shown as text and is
    /// *not* auto-narrated — `say_this` is her sole spoken channel, so she
    /// chooses what to voice. Independent of `You`; the model's `request_voice`
    /// selects this. (Mirrors the voice daemon's `on_demand` speech mode.)
    OnDemand,
    /// Reads every reply aloud in full (made speakable, not shortened) —
    /// accessibility. A `say_this` aside is *not* separately spoken here (the
    /// whole reply already is), so it downgrades to shown text. Independent of
    /// `You`. (Mirrors the voice daemon's `always` speech mode.)
    Always,
}

impl AdeleOutput {
    /// The next level when the user cycles the control
    /// (`Disabled → OnDemand → Always → Disabled`).
    pub fn next(self) -> Self {
        match self {
            Self::Disabled => Self::OnDemand,
            Self::OnDemand => Self::Always,
            Self::Always => Self::Disabled,
        }
    }

    /// Short label for the status line / chat-title cue / dropdown.
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "Disabled",
            Self::OnDemand => "On Demand",
            Self::Always => "Always",
        }
    }

    /// Whether a *reply* is auto-narrated in full: `Always` only. `OnDemand`
    /// does *not* auto-narrate — its spoken channel is `say_this` — and
    /// `Disabled` never speaks. Decoupled from voice **input** (`You`): the
    /// `Adele:` level alone governs her output, so a conversation whose `Adele:`
    /// reads "off" can never talk (voice#126 — avoids the "Adele is off but
    /// she's talking!" bug).
    ///
    /// This is the reply-narration gate the clients consult, keyed by the
    /// *originating* conversation of the reply.
    pub fn narrates_reply(self) -> bool {
        matches!(self, Self::Always)
    }

    /// Whether a `say_this` aside is spoken aloud: `OnDemand` only — there
    /// `say_this` is Adele's sole spoken channel. `Always` already reads every
    /// reply in full (a separate aside would double-speak) and `Disabled` is
    /// silent, so both downgrade the aside to shown text. Keyed by the *call's*
    /// conversation (voice#126).
    pub fn speaks_aside(self) -> bool {
        matches!(self, Self::OnDemand)
    }

    /// The system refinement to attach on the next send for a conversation at
    /// this level, or `None` for `Disabled`. `OnDemand` →
    /// brief/conversational/speakable; `Always` → speakable-but-full (don't
    /// shorten). The pure decision the send path consults.
    pub fn send_refinement(self) -> Option<&'static str> {
        match self {
            Self::OnDemand => Some(ON_DEMAND_SYSTEM_REFINEMENT),
            Self::Always => Some(ALWAYS_SYSTEM_REFINEMENT),
            Self::Disabled => None,
        }
    }
}

/// System refinement attached on send while `Adele == OnDemand`.
///
/// In on-demand mode the written reply is shown to the user as text and is
/// *not* read aloud — `say_this` is Adele's only spoken channel. So this tells
/// the model to call `say_this` with whatever it wants voiced, kept brief and
/// shaped **for the ear**. Deliberately free of markdown markers so it can't
/// itself leak formatting. Refines the system prompt for that turn only — never
/// stored, never in the transcript.
pub const ON_DEMAND_SYSTEM_REFINEMENT: &str = "You are in on-demand voice mode: your written reply is shown to the user as text and is \
     not read aloud, so anything you want said out loud you must speak by calling the say_this \
     tool. Keep whatever you speak brief and conversational, a few short sentences at most, \
     written to be heard rather than read. In spoken text use no markdown or formatting of any \
     kind and no emoji, spell out acronyms and abbreviations as full words, and avoid symbols \
     that do not read well aloud (say 'and' not an ampersand, 'percent' not a percent sign, \
     'dollars' not a dollar sign). Do not speak URLs, file paths, or email addresses; describe \
     them in words instead, and write numbers, dates, and times the way you would say them.";

/// System refinement attached on send while `Adele == Always`.
///
/// Every reply is read aloud for accessibility, so make it **speakable but not
/// shortened**: keep the full content, just strip formatting and spell out
/// symbols. Crucially it does NOT ask for brevity (that's the `OnDemand` job) —
/// `Always` reads the whole answer. Free of markdown markers itself.
pub const ALWAYS_SYSTEM_REFINEMENT: &str = "This reply will be read aloud in full, so write it to be spoken, not read, without \
     leaving anything out. Do not shorten or summarize — cover everything you would normally \
     say, just phrased for the ear. Use no markdown or formatting of any kind, and no emoji. \
     Spell out acronyms and abbreviations as full words and avoid symbols that do not read well \
     aloud (say 'and' not an ampersand, 'percent' not a percent sign, 'dollars' not a dollar \
     sign). Do not read out URLs, file paths, or email addresses; describe them in words instead, \
     and write numbers, dates, and times the way you would say them.";

/// Split `text` into the chunks that should be fed to a one-shot synthesizer.
///
/// Both the voice daemon's `SayText` and the embedded `Speaker` are
/// **one-shot**: they assume a single short sentence and apply a per-synth
/// timeout (`adele_voice_module`'s `DEFAULT_SYNTH_TIMEOUT`, ~20s). A long reply
/// fed in one go would blow that timeout, so the *client* must chunk it the same
/// way the daemon's streaming pipeline does — via [`SentenceBuffer`].
///
/// This pushes the whole text through a `SentenceBuffer` (collecting every
/// complete sentence) and then appends the trailing remainder from `flush()`
/// (the last sentence has no trailing whitespace, so the buffer holds it back).
/// If chunking yields nothing it falls back to a single chunk of the trimmed
/// original when that text is non-blank, and to an empty `Vec` for
/// empty/whitespace input (nothing to speak).
///
/// The timeout passed to the buffer is irrelevant here: this is a synchronous,
/// one-shot push/flush with no streaming, so the time-based flush never fires.
///
/// Behind the default `chunker` feature — it pulls in `adele-voice-core`. wasm
/// consumers that only need [`AdeleOutput`] build with `default-features = false`.
#[cfg(feature = "chunker")]
pub fn into_speakable_sentences(text: &str) -> Vec<String> {
    // Timeout is unused on this synchronous push→flush path; any value works.
    let mut buf = SentenceBuffer::new(Duration::from_millis(500));
    let mut sentences = buf.push(text);
    let tail = buf.flush();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    if sentences.is_empty() && !text.trim().is_empty() {
        // No boundary produced a chunk but there *is* speakable text — speak it
        // whole rather than dropping it silently.
        sentences.push(text.trim().to_string());
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        assert_eq!(AdeleOutput::default(), AdeleOutput::Disabled);
    }

    #[test]
    fn cycles_disabled_on_demand_always() {
        assert_eq!(AdeleOutput::Disabled.next(), AdeleOutput::OnDemand);
        assert_eq!(AdeleOutput::OnDemand.next(), AdeleOutput::Always);
        assert_eq!(AdeleOutput::Always.next(), AdeleOutput::Disabled);
    }

    #[test]
    fn every_level_has_a_label() {
        assert_eq!(AdeleOutput::Disabled.label(), "Disabled");
        assert_eq!(AdeleOutput::OnDemand.label(), "On Demand");
        assert_eq!(AdeleOutput::Always.label(), "Always");
    }

    #[test]
    fn reply_narration_gate() {
        // Only Always auto-narrates the full reply. OnDemand does not (its
        // spoken channel is say_this); Disabled is silent. Decoupled from
        // voice input (`You`) — no argument any more (voice#126).
        assert!(AdeleOutput::Always.narrates_reply());
        assert!(!AdeleOutput::OnDemand.narrates_reply());
        assert!(!AdeleOutput::Disabled.narrates_reply());
    }

    #[test]
    fn say_this_aside_gate() {
        // say_this is spoken only in OnDemand (its sole spoken channel).
        // Always already narrates the whole reply and Disabled is silent, so
        // both downgrade the aside to shown text (voice#126).
        assert!(AdeleOutput::OnDemand.speaks_aside());
        assert!(!AdeleOutput::Always.speaks_aside());
        assert!(!AdeleOutput::Disabled.speaks_aside());
    }

    #[test]
    fn send_refinement_per_level() {
        assert_eq!(AdeleOutput::Disabled.send_refinement(), None);
        assert_eq!(
            AdeleOutput::OnDemand.send_refinement(),
            Some(ON_DEMAND_SYSTEM_REFINEMENT)
        );
        assert_eq!(
            AdeleOutput::Always.send_refinement(),
            Some(ALWAYS_SYSTEM_REFINEMENT)
        );
    }

    #[test]
    fn refinements_are_distinct_and_markdown_free() {
        assert_ne!(ON_DEMAND_SYSTEM_REFINEMENT, ALWAYS_SYSTEM_REFINEMENT);
        // OnDemand asks for brevity; Always must NOT (it reads the whole answer).
        assert!(ON_DEMAND_SYSTEM_REFINEMENT.to_lowercase().contains("brief"));
        assert!(!ALWAYS_SYSTEM_REFINEMENT.to_lowercase().contains("brief"));
        // Neither may carry markdown markers, or it could leak formatting.
        for refinement in [ON_DEMAND_SYSTEM_REFINEMENT, ALWAYS_SYSTEM_REFINEMENT] {
            assert!(!refinement.contains('*'));
            assert!(!refinement.contains('`'));
            assert!(!refinement.contains('#'));
        }
    }

    // --- Sentence chunking (behind the `chunker` feature) ---

    #[cfg(feature = "chunker")]
    #[test]
    fn chunks_multi_sentence_into_sentences() {
        let chunks = into_speakable_sentences("Hello there. How are you? I am fine.");
        assert_eq!(chunks, vec!["Hello there.", "How are you?", "I am fine."]);
    }

    #[cfg(feature = "chunker")]
    #[test]
    fn chunks_single_sentence_into_one() {
        let chunks = into_speakable_sentences("Just one sentence here.");
        assert_eq!(chunks, vec!["Just one sentence here."]);
    }

    #[cfg(feature = "chunker")]
    #[test]
    fn chunks_text_without_terminal_punctuation_into_one() {
        let chunks = into_speakable_sentences("no trailing punctuation here");
        assert_eq!(chunks, vec!["no trailing punctuation here"]);
    }

    #[cfg(feature = "chunker")]
    #[test]
    fn chunks_empty_or_whitespace_into_nothing() {
        assert!(into_speakable_sentences("").is_empty());
        assert!(into_speakable_sentences("   \n\t  ").is_empty());
    }

    #[cfg(feature = "chunker")]
    #[test]
    fn chunks_long_paragraph_into_multiple() {
        let paragraph = "The quick brown fox jumps over the lazy dog. \
             It then trots away to find a quiet spot. \
             Later, the dog wakes up and stretches lazily. \
             Neither animal pays the other any further mind. \
             The afternoon sun warms the empty field.";
        let chunks = into_speakable_sentences(paragraph);
        assert!(
            chunks.len() >= 4,
            "a five-sentence paragraph should split into several chunks, got {}: {chunks:?}",
            chunks.len()
        );
        assert!(chunks.iter().all(|c| !c.trim().is_empty()));
    }
}
