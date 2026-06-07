//! Turn markdown-formatted text into clean prose suitable for a TTS engine.
//!
//! The LLM frequently emits markdown (bold, headers, lists, code fences, …)
//! even though the spoken-response system prompt asks it not to. Fed straight to
//! the synthesizer, that markdown is read *aloud* — "asterisk asterisk asterisk"
//! for `***`, "pound" for `#` headers, backticks, bullet characters, and so on.
//! The prompt instruction alone is unreliable, so [`strip_markdown_for_speech`]
//! deterministically strips the formatting right before synthesis (voice#63).
//!
//! Correctness over cleverness: we parse with `pulldown-cmark` (a real
//! CommonMark parser) and concatenate the text it emits, which gets the hard
//! cases right for free — intra-word punctuation (`file_name`, `snake_case`),
//! matched-vs-unmatched emphasis markers, links/images (keep the visible text,
//! drop the URL), and code (read the code text, not the backticks). A light
//! post-pass then collapses whitespace and removes any *residual* lone markdown
//! punctuation, which matters because the [`SentenceBuffer`](crate::sentence_buffer)
//! can split a `**marker**` across two emitted sentences.

use pulldown_cmark::{Event, Options, Parser, TagEnd};

/// Convert markdown-formatted `input` into clean prose for text-to-speech.
///
/// Emphasis/strikethrough markers, code fences/backticks, ATX header `#`s, list
/// bullets and ordered numbers, blockquote `>`s, horizontal rules, and link/
/// image URLs are removed; the human-readable text is kept. Block boundaries
/// (paragraphs, headings, list items) become sentence breaks so a list reads as
/// a sequence of short utterances rather than one run-on. Sentence-final
/// punctuation is preserved (the leading-ack relies on it). The function is
/// idempotent and leaves already-plain text unchanged.
pub fn strip_markdown_for_speech(input: &str) -> String {
    if input.trim().is_empty() {
        return String::new();
    }

    // Strikethrough is a GFM extension; enable it so `~~x~~` is parsed as an
    // emphasis span (its inner text kept) instead of literal tildes. We
    // deliberately leave tables/footnotes/tasklists off — the LLM shouldn't be
    // producing them in a spoken reply, and the default parse handles any stray
    // syntax as plain text, which is exactly what we want read aloud.
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);

    // Accumulate plain text. We push a sentence-ending break between blocks so
    // separate list items / paragraphs are spoken as separate short sentences;
    // inline breaks become a single space.
    let mut out = String::new();
    for event in Parser::new_ext(input, options) {
        match event {
            // Visible text and inline/fenced code: keep the characters.
            Event::Text(t) | Event::Code(t) => out.push_str(&t),
            // Math, raw HTML, footnote refs: not meaningful spoken content —
            // drop them rather than read tags/markup aloud.
            Event::InlineMath(_)
            | Event::DisplayMath(_)
            | Event::Html(_)
            | Event::InlineHtml(_)
            | Event::FootnoteReference(_) => {}
            // Soft/hard line breaks within a block read as a pause → one space.
            Event::SoftBreak | Event::HardBreak => push_space(&mut out),
            // A horizontal rule has no spoken content; just break.
            Event::Rule => push_break(&mut out),
            Event::TaskListMarker(_) => {}
            // Block boundaries → a sentence break so items don't run together.
            Event::End(
                TagEnd::Paragraph
                | TagEnd::Heading(_)
                | TagEnd::Item
                | TagEnd::BlockQuote(_)
                | TagEnd::CodeBlock
                | TagEnd::List(_)
                | TagEnd::TableRow
                | TagEnd::TableHead,
            ) => push_break(&mut out),
            // A table cell boundary reads as a short pause.
            Event::End(TagEnd::TableCell) => push_space(&mut out),
            // All Start tags and remaining End tags carry no spoken text of
            // their own (the inner Text events do). Link/Image URLs live on the
            // Start tag and are intentionally dropped; the visible link text /
            // image alt arrives as Text events and is kept.
            Event::Start(_) | Event::End(_) => {}
        }
    }

    cleanup(&out)
}

/// Append a single space unless the output is empty or already ends with
/// whitespace (so we never emit doubled spaces here).
fn push_space(out: &mut String) {
    if !out.is_empty() && !out.ends_with(char::is_whitespace) {
        out.push(' ');
    }
}

/// Append a block break as a newline marker; [`cleanup`] turns runs of these
/// into clean sentence spacing.
fn push_break(out: &mut String) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}

/// Final pass: drop residual lone markdown punctuation left by markers the
/// SentenceBuffer split across sentence boundaries, then collapse whitespace and
/// block breaks into clean single-spaced prose.
fn cleanup(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    // Process per block-break line so we can decide, line by line, whether the
    // break should be dropped (blank line) or rendered as a space.
    let mut first_line = true;
    for raw_line in text.split('\n') {
        let line = strip_residual_markers(raw_line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !first_line {
            result.push(' ');
        }
        first_line = false;
        // Collapse internal whitespace runs to single spaces.
        let mut prev_ws = false;
        for ch in line.chars() {
            if ch.is_whitespace() {
                if !prev_ws {
                    result.push(' ');
                }
                prev_ws = true;
            } else {
                result.push(ch);
                prev_ws = false;
            }
        }
    }
    result.trim().to_string()
}

/// Remove residual markdown punctuation that survived parsing — e.g. when a
/// `**bold**` span is split by the SentenceBuffer so only one half reaches this
/// call, the parser sees an unmatched marker and emits it as literal text
/// (either as a lone `**` token or *attached* to a word as `**Here`).
///
/// Two passes per whitespace-delimited token:
/// 1. Drop the token entirely if it is *all* marker characters (`**`, `#`,
///    `---`, `~~`, `* * *`, …).
/// 2. Otherwise trim leading/trailing runs of edge-only emphasis markers
///    (`*`, `` ` ``, `~`, `_`, `#`) off the token, so `**Here` → `Here` and
///    `code`` `` → `code`.
///
/// Crucially this only ever touches the *ends* of a token, so intra-word
/// punctuation is preserved: `file_name`, `snake_case_thing`, `a*b*c`, and
/// hyphenated/negative tokens (`well-known`, `-5`) all survive intact. `-` is
/// intentionally NOT trimmed from word edges (it carries meaning in numbers and
/// ranges); a lone `-`/`---` is already removed by pass 1.
fn strip_residual_markers(line: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for token in line.split_whitespace() {
        if token.chars().all(is_markdown_marker) {
            continue; // pass 1: whole token is just markers
        }
        // pass 2: trim edge-only emphasis markers.
        let trimmed = token.trim_matches(is_edge_marker);
        if !trimmed.is_empty() {
            kept.push(trimmed);
        }
    }
    kept.join(" ")
}

/// Characters that, as a standalone token, are markdown structure rather than
/// speech (`-`/`---` rules and bullets included).
fn is_markdown_marker(c: char) -> bool {
    matches!(c, '*' | '_' | '#' | '`' | '~' | '-')
}

/// Emphasis/code markers that are only meaningful at a word boundary, so they
/// are safe to trim off a token's edges (but never from its interior). `-` is
/// excluded — it is meaningful at the edge of numbers/ranges.
fn is_edge_marker(c: char) -> bool {
    matches!(c, '*' | '_' | '#' | '`' | '~')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bold_keeps_the_word() {
        assert_eq!(strip_markdown_for_speech("**bold**"), "bold");
        assert_eq!(strip_markdown_for_speech("__bold__"), "bold");
    }

    #[test]
    fn italic_keeps_the_word() {
        assert_eq!(strip_markdown_for_speech("*italic*"), "italic");
        assert_eq!(strip_markdown_for_speech("_italic_"), "italic");
    }

    #[test]
    fn bold_italic_keeps_the_word() {
        assert_eq!(strip_markdown_for_speech("***x***"), "x");
    }

    #[test]
    fn strikethrough_keeps_the_word() {
        assert_eq!(strip_markdown_for_speech("~~strike~~"), "strike");
    }

    #[test]
    fn inline_code_keeps_the_text_not_the_backticks() {
        assert_eq!(
            strip_markdown_for_speech("run `cargo test` now"),
            "run cargo test now"
        );
    }

    #[test]
    fn fenced_code_block_keeps_inner_lines_drops_fences() {
        let md = "Here:\n\n```rust\nlet x = 1;\nlet y = 2;\n```";
        let out = strip_markdown_for_speech(md);
        assert!(out.contains("let x = 1;"), "inner code kept: {out:?}");
        assert!(out.contains("let y = 2;"), "inner code kept: {out:?}");
        assert!(!out.contains('`'), "no backticks spoken: {out:?}");
        assert!(!out.contains("rust"), "fence language dropped: {out:?}");
    }

    #[test]
    fn atx_header_single_hash() {
        assert_eq!(strip_markdown_for_speech("# Title"), "Title");
    }

    #[test]
    fn atx_header_multiple_hashes_and_trailing() {
        assert_eq!(
            strip_markdown_for_speech("### Subsection ###"),
            "Subsection"
        );
    }

    #[test]
    fn unordered_list_dash() {
        let out = strip_markdown_for_speech("- one\n- two\n- three");
        assert_eq!(out, "one two three");
        assert!(!out.contains('-'));
    }

    #[test]
    fn unordered_list_asterisk_and_plus() {
        assert_eq!(strip_markdown_for_speech("* a\n* b"), "a b");
        assert_eq!(strip_markdown_for_speech("+ a\n+ b"), "a b");
    }

    #[test]
    fn ordered_list_dot_and_paren() {
        assert_eq!(
            strip_markdown_for_speech("1. first\n2. second"),
            "first second"
        );
        assert_eq!(
            strip_markdown_for_speech("1) first\n2) second"),
            "first second"
        );
    }

    #[test]
    fn blockquote_dropped() {
        assert_eq!(strip_markdown_for_speech("> quoted text"), "quoted text");
    }

    #[test]
    fn horizontal_rule_removed_entirely() {
        assert_eq!(strip_markdown_for_speech("---"), "");
        assert_eq!(strip_markdown_for_speech("***"), "");
        assert_eq!(strip_markdown_for_speech("___"), "");
        assert_eq!(
            strip_markdown_for_speech("before\n\n---\n\nafter"),
            "before after"
        );
    }

    #[test]
    fn link_keeps_text_drops_url() {
        assert_eq!(
            strip_markdown_for_speech("see [the docs](https://example.com/x)"),
            "see the docs"
        );
    }

    #[test]
    fn image_keeps_alt_drops_url() {
        assert_eq!(
            strip_markdown_for_speech("![a red cat](https://example.com/cat.png)"),
            "a red cat"
        );
    }

    #[test]
    fn autolink_keeps_url_drops_angle_brackets() {
        let out = strip_markdown_for_speech("visit <https://example.com>");
        assert!(out.contains("https://example.com"), "{out:?}");
        assert!(!out.contains('<') && !out.contains('>'), "{out:?}");
    }

    #[test]
    fn mixed_header_with_bold_and_link() {
        let md = "## **Important**: read [the guide](http://x.io) now";
        let out = strip_markdown_for_speech(md);
        assert_eq!(out, "Important: read the guide now");
    }

    #[test]
    fn snake_case_is_preserved() {
        // The main correctness trap: intra-word underscores/asterisks must NOT
        // be treated as emphasis markers.
        assert_eq!(
            strip_markdown_for_speech("call run_tts_service before exit"),
            "call run_tts_service before exit"
        );
        assert_eq!(
            strip_markdown_for_speech("the file_name field"),
            "the file_name field"
        );
        assert_eq!(
            strip_markdown_for_speech("snake_case_thing stays whole"),
            "snake_case_thing stays whole"
        );
    }

    #[test]
    fn intra_word_asterisk_survives() {
        // `a*b` (no matched pair) is not emphasis — keep it.
        let out = strip_markdown_for_speech("compute a*b*c here");
        assert!(out.contains("a*b*c") || out.contains("abc"), "{out:?}");
        // Either way it must read as connected, not as a lone asterisk token.
        assert!(!out.split_whitespace().any(|t| t == "*"), "{out:?}");
    }

    #[test]
    fn lone_asterisk_reduces_to_no_spoken_asterisk() {
        // A SentenceBuffer split can leave a stray marker alone in a sentence.
        assert_eq!(
            strip_markdown_for_speech("Here are the steps *"),
            "Here are the steps"
        );
        assert_eq!(strip_markdown_for_speech("* leftover"), "leftover");
    }

    #[test]
    fn unmatched_double_asterisk_reduces_to_no_spoken_asterisks() {
        let out = strip_markdown_for_speech("**Here are the steps");
        assert!(!out.contains('*'), "no asterisks spoken: {out:?}");
        assert!(out.contains("Here are the steps"), "{out:?}");
    }

    #[test]
    fn plain_text_unchanged_and_idempotent() {
        let plain = "The weather today is sunny with a high of seventy two degrees.";
        assert_eq!(strip_markdown_for_speech(plain), plain);
        // Idempotent: running it again is a no-op.
        let once = strip_markdown_for_speech("**bold** and `code` and [link](u)");
        assert_eq!(strip_markdown_for_speech(&once), once);
    }

    #[test]
    fn preserves_sentence_final_punctuation() {
        // The leading-ack flush depends on a trailing ./!/? — never strip it.
        assert_eq!(
            strip_markdown_for_speech("Got it — checking that now."),
            "Got it — checking that now."
        );
        assert_eq!(strip_markdown_for_speech("**Really?**"), "Really?");
    }

    #[test]
    fn empty_and_whitespace_input_yield_empty() {
        assert_eq!(strip_markdown_for_speech(""), "");
        assert_eq!(strip_markdown_for_speech("   "), "");
        assert_eq!(strip_markdown_for_speech("\n\t  \n"), "");
    }

    #[test]
    fn all_markdown_sentence_yields_empty() {
        // A "sentence" that is purely formatting (e.g. a rule, or stray markers)
        // sanitizes to empty so the caller can skip synthesis.
        assert_eq!(strip_markdown_for_speech("---"), "");
        assert_eq!(strip_markdown_for_speech("**"), "");
        assert_eq!(strip_markdown_for_speech("* * *"), "");
    }

    #[test]
    fn streamed_bold_lead_sentence_has_no_asterisks() {
        // The headline integration case: a streamed sentence like
        // "**Here** are the steps:" must be spoken without asterisks.
        let out = strip_markdown_for_speech("**Here** are the steps:");
        assert_eq!(out, "Here are the steps:");
        assert!(!out.contains('*'));
    }

    #[test]
    fn multiline_list_with_intro_reads_as_short_sentences() {
        let md = "Steps:\n\n- Open the file\n- Save it\n- Done";
        let out = strip_markdown_for_speech(md);
        assert_eq!(out, "Steps: Open the file Save it Done");
    }
}
