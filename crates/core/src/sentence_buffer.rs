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
}
