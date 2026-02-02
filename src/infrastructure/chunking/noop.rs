use crate::domain::model::Chunk;
use crate::domain::ports::Chunker;

/// A no-op chunker that treats the entire document as a single chunk.
/// This is the default behavior when chunking is disabled.
pub struct NoopChunker;

impl NoopChunker {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoopChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for NoopChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        if text.is_empty() {
            return Vec::new();
        }
        vec![Chunk {
            chunk_index: 0,
            text: text.to_string(),
            start_offset: 0,
            end_offset: text.len(),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_chunker_single_chunk() {
        let chunker = NoopChunker::new();
        let text = "Hello, world! This is a test.";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].text, text);
        assert_eq!(chunks[0].start_offset, 0);
        assert_eq!(chunks[0].end_offset, text.len());
    }

    #[test]
    fn test_noop_chunker_empty_text() {
        let chunker = NoopChunker::new();
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }
}
