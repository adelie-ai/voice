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

use std::time::Duration;

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
    /// Never speaks (the default). A `say_this` aside downgrades to inline text.
    #[default]
    Disabled,
    /// Speaks replies only while `You == Enabled` (conversing by voice), shaped
    /// for the ear; speaks `say_this` asides regardless of `You`. The model's
    /// `request_voice` selects this.
    OnDemand,
    /// Reads every reply aloud in full (made speakable, not shortened) —
    /// accessibility. Independent of `You`.
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

    /// Whether a *reply* is spoken: `Always` always, `OnDemand` only while the
    /// conversation also has voice **input** enabled (`You == Enabled`, i.e. the
    /// user is conversing by voice), `Disabled` never. `voice_in` is that
    /// `You == Enabled` flag for the same conversation.
    ///
    /// This is the reply-narration gate the clients consult, keyed by the
    /// *originating* conversation of the reply.
    pub fn narrates_reply(self, voice_in: bool) -> bool {
        match self {
            Self::Always => true,
            Self::OnDemand => voice_in,
            Self::Disabled => false,
        }
    }

    /// Whether a `say_this` aside is spoken: spoken iff the level is `OnDemand`
    /// or `Always` (independent of `You`); `Disabled` downgrades the aside to
    /// inline text. Keyed by the *call's* conversation.
    pub fn speaks_aside(self) -> bool {
        !matches!(self, Self::Disabled)
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
/// Replies are spoken only while conversing by voice, so shape them **for the
/// ear**: brief, conversational, no markdown, symbols/acronyms spelled out.
/// Deliberately free of markdown markers so it can't itself leak formatting.
/// Refines the system prompt for that turn only — never stored, never in the
/// transcript.
pub const ON_DEMAND_SYSTEM_REFINEMENT: &str = "This reply will be read aloud, so write it to be spoken, not read. Keep it brief and \
     conversational, a few short sentences at most. Use no markdown or formatting of any kind, \
     and no emoji. Spell out acronyms and abbreviations as full words and avoid symbols that do \
     not read well aloud (say 'and' not an ampersand, 'percent' not a percent sign, 'dollars' not \
     a dollar sign). Do not read out URLs, file paths, or email addresses; describe them in words \
     instead, and write numbers, dates, and times the way you would say them.";

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
        // Always narrates regardless of voice input.
        assert!(AdeleOutput::Always.narrates_reply(false));
        assert!(AdeleOutput::Always.narrates_reply(true));
        // OnDemand only narrates while conversing by voice (You == Enabled).
        assert!(!AdeleOutput::OnDemand.narrates_reply(false));
        assert!(AdeleOutput::OnDemand.narrates_reply(true));
        // Disabled never narrates.
        assert!(!AdeleOutput::Disabled.narrates_reply(false));
        assert!(!AdeleOutput::Disabled.narrates_reply(true));
    }

    #[test]
    fn say_this_aside_gate() {
        // Asides are spoken whenever output isn't Disabled, independent of You.
        assert!(!AdeleOutput::Disabled.speaks_aside());
        assert!(AdeleOutput::OnDemand.speaks_aside());
        assert!(AdeleOutput::Always.speaks_aside());
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

    // --- Sentence chunking ---

    #[test]
    fn chunks_multi_sentence_into_sentences() {
        let chunks = into_speakable_sentences("Hello there. How are you? I am fine.");
        assert_eq!(chunks, vec!["Hello there.", "How are you?", "I am fine."]);
    }

    #[test]
    fn chunks_single_sentence_into_one() {
        let chunks = into_speakable_sentences("Just one sentence here.");
        assert_eq!(chunks, vec!["Just one sentence here."]);
    }

    #[test]
    fn chunks_text_without_terminal_punctuation_into_one() {
        let chunks = into_speakable_sentences("no trailing punctuation here");
        assert_eq!(chunks, vec!["no trailing punctuation here"]);
    }

    #[test]
    fn chunks_empty_or_whitespace_into_nothing() {
        assert!(into_speakable_sentences("").is_empty());
        assert!(into_speakable_sentences("   \n\t  ").is_empty());
    }

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
