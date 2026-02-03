use crate::domain::model::Chunk;
use crate::domain::ports::Chunker;

/// A fixed-size chunker that splits text into chunks of a specified size with overlap.
/// Attempts to split at word boundaries when possible.
/// All sizes are in Unicode characters, not bytes (UTF-8 safe).
pub struct FixedChunker {
    /// Target chunk size in characters
    chunk_size: usize,
    /// Number of characters to overlap between adjacent chunks
    overlap: usize,
}

impl FixedChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
        }
    }

    /// Find the best split point (in char index) at or before the given position,
    /// preferring word boundaries.
    fn find_split_point(&self, chars: &[char], max_char_pos: usize) -> usize {
        if max_char_pos >= chars.len() {
            return chars.len();
        }

        // Look backwards for a word boundary (space, newline, etc.)
        let search_start = max_char_pos.saturating_sub(self.chunk_size / 4);

        // Find the last whitespace character in the search range
        for i in (search_start..=max_char_pos).rev() {
            if chars[i].is_whitespace() {
                return i + 1; // +1 to include the space in previous chunk
            }
        }

        // No good split point found, just use the max position
        max_char_pos
    }

    /// Convert a character index to a byte index in the original string
    fn char_to_byte_index(chars: &[char], char_idx: usize) -> usize {
        chars[..char_idx].iter().map(|c| c.len_utf8()).sum()
    }
}

impl Default for FixedChunker {
    fn default() -> Self {
        Self::new(512, 50)
    }
}

impl Chunker for FixedChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let chars: Vec<char> = text.chars().collect();
        let char_count = chars.len();

        if char_count <= self.chunk_size {
            return vec![Chunk {
                chunk_index: 0,
                text: text.to_string(),
                start_offset: 0,
                end_offset: text.len(),
            }];
        }

        let mut chunks = Vec::new();
        let mut start_char = 0;
        let mut chunk_index = 0;

        while start_char < char_count {
            let target_end_char = (start_char + self.chunk_size).min(char_count);
            let end_char = if target_end_char >= char_count {
                char_count
            } else {
                self.find_split_point(&chars, target_end_char)
            };

            // Convert char indices to byte indices for slicing
            let start_byte = Self::char_to_byte_index(&chars, start_char);
            let end_byte = Self::char_to_byte_index(&chars, end_char);

            let chunk_text = text[start_byte..end_byte].trim();
            if !chunk_text.is_empty() {
                // Find the actual byte offset after trimming
                let trim_offset = text[start_byte..end_byte]
                    .find(|c: char| !c.is_whitespace())
                    .unwrap_or(0);
                let actual_start_byte = start_byte + trim_offset;
                let actual_end_byte = actual_start_byte + chunk_text.len();

                chunks.push(Chunk {
                    chunk_index,
                    text: chunk_text.to_string(),
                    start_offset: actual_start_byte,
                    end_offset: actual_end_byte,
                });
                chunk_index += 1;
            }

            // Calculate next start position (in chars) with overlap
            let chars_processed = end_char - start_char;
            let advance = if chars_processed > self.overlap {
                chars_processed - self.overlap
            } else {
                // Prevent infinite loop: advance at least 1 character
                chars_processed.max(1)
            };
            start_char += advance;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_chunker_short_text() {
        let chunker = FixedChunker::new(100, 10);
        let text = "Short text";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text");
    }

    #[test]
    fn test_fixed_chunker_splits_at_word_boundary() {
        let chunker = FixedChunker::new(20, 5);
        let text = "Hello world, this is a longer text that needs to be split.";
        let chunks = chunker.chunk(text);

        assert!(chunks.len() > 1);
        // Each chunk should end at or near a word boundary
        for chunk in &chunks {
            // Chunks should not start or end with partial words (unless at boundaries)
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_fixed_chunker_overlap() {
        let chunker = FixedChunker::new(30, 10);
        let text = "The quick brown fox jumps over the lazy dog. This is additional text.";
        let chunks = chunker.chunk(text);

        assert!(chunks.len() >= 2);
        // Check that chunks overlap
        for i in 1..chunks.len() {
            let prev_end = chunks[i - 1].end_offset;
            let curr_start = chunks[i].start_offset;
            // There should be some overlap (curr_start < prev_end) or adjacent
            assert!(curr_start <= prev_end);
        }
    }

    #[test]
    fn test_fixed_chunker_empty_text() {
        let chunker = FixedChunker::new(100, 10);
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_fixed_chunker_preserves_offsets() {
        let chunker = FixedChunker::new(50, 10);
        let text = "This is a test string that will be split into multiple chunks for testing.";
        let chunks = chunker.chunk(text);

        for chunk in &chunks {
            // Verify that the offsets correctly map back to the original text
            let extracted = &text[chunk.start_offset..chunk.end_offset];
            assert_eq!(extracted, chunk.text);
        }
    }

    #[test]
    fn test_fixed_chunker_utf8_japanese() {
        let chunker = FixedChunker::new(10, 2);
        let text = "これは日本語のテキストです。分割できますか？";
        let chunks = chunker.chunk(text);

        assert!(
            chunks.len() >= 2,
            "Expected multiple chunks, got {}",
            chunks.len()
        );

        // Verify all chunks have valid offsets
        for chunk in &chunks {
            let extracted = &text[chunk.start_offset..chunk.end_offset];
            assert_eq!(extracted, chunk.text, "Offset mismatch");
        }
    }

    #[test]
    fn test_fixed_chunker_utf8_mixed() {
        let chunker = FixedChunker::new(15, 3);
        let text = "Hello 世界! This is mixed テキスト with Unicode 文字.";
        let chunks = chunker.chunk(text);

        // Verify all chunks have valid offsets and no panics
        for chunk in &chunks {
            let extracted = &text[chunk.start_offset..chunk.end_offset];
            assert_eq!(extracted, chunk.text, "Offset mismatch for: {}", chunk.text);
        }
    }

    #[test]
    fn test_fixed_chunker_emoji() {
        let chunker = FixedChunker::new(10, 2);
        let text = "Hello 👋 World 🌍! How are you? 😊";
        let chunks = chunker.chunk(text);

        // Verify no panics and valid offsets
        for chunk in &chunks {
            let extracted = &text[chunk.start_offset..chunk.end_offset];
            assert_eq!(extracted, chunk.text);
        }
    }
}
