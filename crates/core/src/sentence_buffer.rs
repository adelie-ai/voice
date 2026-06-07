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

    /// Check if the timeout has expired and flush the remaining buffer as a sentence.
    pub fn flush_if_timeout(&mut self) -> Option<String> {
        if let Some(last) = self.last_chunk_at
            && last.elapsed() >= self.timeout
            && !self.buffer.trim().is_empty()
        {
            return Some(self.flush());
        }
        None
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
                    // Skip leading whitespace after the boundary
                    let ws_count = self
                        .buffer
                        .chars()
                        .take_while(|c| c.is_whitespace())
                        .count();
                    if ws_count > 0 {
                        self.buffer.drain(..ws_count);
                    }
                }
                None => break,
            }
        }

        sentences
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
    fn take_leading_ack_waits_for_terminal_punctuation() {
        // An incomplete opener (no sentence-ending punctuation yet) isn't ready.
        let mut buf = SentenceBuffer::new(Duration::from_millis(500));
        buf.push("Got it");
        assert!(buf.take_leading_ack(8).is_none());
    }
}
