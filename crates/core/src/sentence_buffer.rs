use std::time::{Duration, Instant};

/// Accumulates streaming text chunks and yields complete sentences.
///
/// A sentence is considered complete when:
/// - A sentence-ending punctuation (`.` `!` `?`) is followed by whitespace or end of buffer
/// - A timeout expires after receiving a chunk without a sentence boundary
pub struct SentenceBuffer {
    buffer: String,
    timeout: Duration,
    last_chunk_at: Option<Instant>,
}

impl SentenceBuffer {
    pub fn new(timeout: Duration) -> Self {
        Self {
            buffer: String::new(),
            timeout,
            last_chunk_at: None,
        }
    }

    /// Push a text chunk into the buffer and return any complete sentences.
    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        self.last_chunk_at = Some(Instant::now());
        self.extract_sentences()
    }

    /// Flush a short leading acknowledgement the instant it looks complete,
    /// without waiting for the next chunk (#58).
    ///
    /// The spoken-response hint asks the model to open a larger turn with a
    /// brief ack ("Got it — checking that now.") terminated by a period.
    /// [`find_sentence_boundary`](Self::find_sentence_boundary) only splits when
    /// the punctuation is *followed* by more text, so an ack sitting alone at
    /// the buffer's end would otherwise stall until the next token arrives —
    /// exactly the early feedback the user is waiting for. This flushes such a
    /// terminal ack immediately, but ONLY when it is short (<= `max_words`),
    /// so a genuine first sentence of the answer isn't spoken twice.
    pub fn take_leading_ack(&mut self, max_words: usize) -> Option<String> {
        let trimmed = self.buffer.trim_end();
        if !trimmed.ends_with(['.', '!', '?']) {
            return None;
        }
        let ack = trimmed.trim();
        if ack.is_empty() || ack.split_whitespace().count() > max_words {
            return None;
        }
        Some(self.flush())
    }

    /// When the stream has gone quiet for `timeout` mid-sentence — tokens are
    /// still expected, but none have arrived — flush the largest speakable
    /// prefix that ends at a *natural pause* rather than dumping whatever raw
    /// bytes happen to be buffered.
    ///
    /// "Natural pause" means a clause boundary: a comma, semicolon, colon, or
    /// dash followed by whitespace (sentence-ending `.!?` are already handled
    /// eagerly by [`push`](Self::push)). Cutting there keeps each TTS unit
    /// prosodically whole, so a slow stream sounds paced rather than chopped
    /// mid-word. The remainder stays buffered and the quiet-clock resets, so the
    /// tail waits a fresh window for more tokens.
    ///
    /// Only if the quiet buffer has *no* clause boundary at all do we flush the
    /// whole fragment — at that point continued silence is worse than an
    /// imperfect cut. This is rare in practice: the timeout fires only on a true
    /// inter-token stall, and the caller's end-of-stream path flushes the tail
    /// immediately regardless (it never relies on this timer).
    pub fn flush_if_timeout(&mut self) -> Option<String> {
        let last = self.last_chunk_at?;
        if last.elapsed() < self.timeout || self.buffer.trim().is_empty() {
            return None;
        }
        match self.find_soft_boundary() {
            Some(pos) => {
                let head: String = self.buffer.drain(..pos).collect();
                self.drain_leading_whitespace();
                // Reset the quiet-clock so the retained tail waits a fresh
                // window for the next token rather than flushing immediately.
                self.last_chunk_at = Some(Instant::now());
                let head = head.trim().to_string();
                (!head.is_empty()).then_some(head)
            }
            // No clause boundary to cut at — speak the whole fragment rather
            // than leave the user in silence.
            None => Some(self.flush()),
        }
    }

    /// Force-flush the remaining buffer as a sentence.
    pub fn flush(&mut self) -> String {
        self.last_chunk_at = None;
        std::mem::take(&mut self.buffer).trim().to_string()
    }

    /// Returns true if the buffer has any non-whitespace content.
    pub fn has_content(&self) -> bool {
        !self.buffer.trim().is_empty()
    }

    fn extract_sentences(&mut self) -> Vec<String> {
        let mut sentences = Vec::new();

        loop {
            let boundary = self.find_sentence_boundary();
            match boundary {
                Some(pos) => {
                    let sentence: String = self.buffer.drain(..pos).collect();
                    let trimmed = sentence.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    self.drain_leading_whitespace();
                }
                None => break,
            }
        }

        sentences
    }

    /// Drop any run of leading whitespace left at the front of the buffer after
    /// draining a sentence or clause off the head.
    fn drain_leading_whitespace(&mut self) {
        let ws_count = self
            .buffer
            .chars()
            .take_while(|c| c.is_whitespace())
            .count();
        if ws_count > 0 {
            self.buffer.drain(..ws_count);
        }
    }

    fn find_sentence_boundary(&self) -> Option<usize> {
        let bytes = self.buffer.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'.' || b == b'!' || b == b'?' {
                // Check if followed by whitespace or end of string
                let next = i + 1;
                if next >= bytes.len() {
                    // At end of buffer — don't split yet, more text may come
                    return None;
                }
                if bytes[next].is_ascii_whitespace() {
                    return Some(next);
                }
            }
        }
        None
    }

    /// Find the *last* clause boundary in the buffer — a `,`, `;`, `:`, or dash
    /// (`-`/`—`) that is followed by whitespace — and return the byte offset
    /// just past the punctuation, so the head keeps the mark (its slight pause
    /// reads naturally) and the retained tail starts after the gap.
    ///
    /// The bare hyphen only counts when it's also *preceded* by whitespace, so a
    /// clause dash (`one moment - checking`) splits but a hyphenated compound
    /// (`built-in`) does not. Picking the last boundary flushes as much settled
    /// text as possible per timeout.
    fn find_soft_boundary(&self) -> Option<usize> {
        let chars: Vec<(usize, char)> = self.buffer.char_indices().collect();
        let mut last = None;
        for (idx, &(byte_pos, c)) in chars.iter().enumerate() {
            let is_clause_mark = matches!(c, ',' | ';' | ':' | '—')
                || (c == '-' && idx > 0 && chars[idx - 1].1.is_whitespace());
            if !is_clause_mark {
                continue;
            }
            // Require following whitespace so we cut at a real gap, never at the
            // buffer's trailing edge (more text may still complete the clause).
            if let Some(&(_, next_c)) = chars.get(idx + 1)
                && next_c.is_whitespace()
            {
                last = Some(byte_pos + c.len_utf8());
            }
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_period_followed_by_space() {
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        let sentences = buf.push("Hello world. How are you? ");
        assert_eq!(sentences, vec!["Hello world.", "How are you?"]);
    }

    #[test]
    fn accumulates_until_boundary() {
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        let s1 = buf.push("Hello ");
        assert!(s1.is_empty());
        let s2 = buf.push("world. ");
        assert_eq!(s2, vec!["Hello world."]);
    }

    #[test]
    fn does_not_split_at_end_of_buffer() {
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        let s = buf.push("Hello world.");
        assert!(
            s.is_empty(),
            "should not split when period is at end of buffer"
        );
    }

    #[test]
    fn flush_returns_remaining() {
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        buf.push("Hello world");
        let flushed = buf.flush();
        assert_eq!(flushed, "Hello world");
        assert!(!buf.has_content());
    }

    #[test]
    fn exclamation_and_question_marks() {
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        let sentences = buf.push("Wow! Really? Yes. ");
        assert_eq!(sentences, vec!["Wow!", "Really?", "Yes."]);
    }

    #[test]
    fn does_not_split_on_abbreviations() {
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        // "Dr.Smith" — no space after period, should not split
        let s = buf.push("Dr.Smith is here. ");
        assert_eq!(s, vec!["Dr.Smith is here."]);
    }

    #[test]
    fn empty_push() {
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        let s = buf.push("");
        assert!(s.is_empty());
        assert!(!buf.has_content());
    }

    #[test]
    fn take_leading_ack_flushes_a_short_terminal_opener() {
        // #58: a short ack alone at the end of the buffer is spoken immediately,
        // before the next token arrives.
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        assert!(buf.push("Got it — checking that now.").is_empty());
        let ack = buf.take_leading_ack(8);
        assert_eq!(ack.as_deref(), Some("Got it — checking that now."));
        assert!(
            !buf.has_content(),
            "the ack must be drained from the buffer"
        );
    }

    #[test]
    fn take_leading_ack_ignores_a_long_first_sentence() {
        // A real (long) first sentence of the answer must NOT be flushed as an
        // ack — that would speak it early and then again on the normal path.
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        buf.push("The weather today is sunny with a high of about seventy two degrees.");
        assert!(
            buf.take_leading_ack(8).is_none(),
            "a long terminal sentence is not a leading ack"
        );
        assert!(buf.has_content(), "a non-ack must stay buffered");
    }

    #[test]
    fn timeout_flushes_at_a_clause_boundary_not_mid_word() {
        // A slow stream goes quiet mid-sentence. The flush must cut at the
        // comma, keeping the clause whole and retaining the unfinished tail.
        let mut buf = SentenceBuffer::new(Duration::from_millis(0));
        assert!(buf.push("Okay, checking the weather no").is_empty());
        let flushed = buf.flush_if_timeout();
        assert_eq!(flushed.as_deref(), Some("Okay,"));
        // The tail (after the comma + space) stays buffered for more tokens.
        assert!(buf.has_content());
        assert_eq!(buf.flush(), "checking the weather no");
    }

    #[test]
    fn timeout_picks_the_last_clause_boundary() {
        // Flush as much settled text as possible: cut at the *last* boundary.
        let mut buf = SentenceBuffer::new(Duration::from_millis(0));
        buf.push("First, then second; and then the unfinished tai");
        let flushed = buf.flush_if_timeout();
        assert_eq!(flushed.as_deref(), Some("First, then second;"));
        assert_eq!(buf.flush(), "and then the unfinished tai");
    }

    #[test]
    fn timeout_with_no_clause_boundary_flushes_the_whole_fragment() {
        // No comma/clause mark to cut at — speaking the fragment beats silence.
        let mut buf = SentenceBuffer::new(Duration::from_millis(0));
        buf.push("checking the weather now");
        let flushed = buf.flush_if_timeout();
        assert_eq!(flushed.as_deref(), Some("checking the weather now"));
        assert!(!buf.has_content());
    }

    #[test]
    fn timeout_does_not_split_a_hyphenated_compound() {
        // A bare hyphen inside a word (no preceding whitespace) is not a clause
        // boundary, so "built-in" stays intact and the fragment flushes whole.
        let mut buf = SentenceBuffer::new(Duration::from_millis(0));
        buf.push("it is built-in support");
        let flushed = buf.flush_if_timeout();
        assert_eq!(flushed.as_deref(), Some("it is built-in support"));
    }

    #[test]
    fn timeout_splits_on_a_clause_dash() {
        // A spaced dash *is* a clause boundary.
        let mut buf = SentenceBuffer::new(Duration::from_millis(0));
        buf.push("one moment - still checking the foreca");
        let flushed = buf.flush_if_timeout();
        assert_eq!(flushed.as_deref(), Some("one moment -"));
        assert_eq!(buf.flush(), "still checking the foreca");
    }

    #[test]
    fn no_timeout_flush_before_the_window_elapses() {
        // A long timeout must not flush a buffer that is still actively filling.
        let mut buf = SentenceBuffer::new(Duration::from_secs(3600));
        buf.push("Okay, checking");
        assert!(buf.flush_if_timeout().is_none());
        assert!(buf.has_content());
    }

    #[test]
    fn take_leading_ack_waits_for_terminal_punctuation() {
        // An incomplete opener (no sentence-ending punctuation yet) isn't ready.
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        buf.push("Got it");
        assert!(buf.take_leading_ack(8).is_none());
    }
}
