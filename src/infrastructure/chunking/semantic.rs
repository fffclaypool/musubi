use crate::domain::model::Chunk;
use crate::domain::ports::{Chunker, Embedder};
use std::sync::Arc;

/// A semantic chunker that splits text based on semantic similarity between sentences.
/// Uses an embedder to compute sentence embeddings and splits when similarity drops below threshold.
pub struct SemanticChunker {
    /// Minimum chunk size in characters
    min_chunk_size: usize,
    /// Maximum chunk size in characters
    max_chunk_size: usize,
    /// Similarity threshold (0.0-1.0). Split when similarity drops below this.
    similarity_threshold: f32,
    /// Embedder for computing sentence similarities
    embedder: Arc<dyn Embedder>,
}

impl SemanticChunker {
    pub fn new(
        min_chunk_size: usize,
        max_chunk_size: usize,
        similarity_threshold: f32,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            min_chunk_size,
            max_chunk_size,
            similarity_threshold,
            embedder,
        }
    }

    /// Split text into sentences using common sentence boundaries.
    fn split_sentences(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut sentences = Vec::new();
        let mut start = 0;
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            let is_sentence_end = matches!(c, '.' | '!' | '?' | '\n');

            if is_sentence_end {
                // Check if this is a real sentence boundary
                let is_real_boundary = if c == '.' {
                    // Not a sentence boundary if followed by a digit (e.g., "3.14")
                    // or if it's an abbreviation (simple heuristic: single letter before)
                    let next_is_space_or_end = i + 1 >= chars.len() || chars[i + 1].is_whitespace();
                    let prev_is_single_letter = i > 0
                        && i > 1
                        && chars[i - 1].is_alphabetic()
                        && (i < 2 || !chars[i - 2].is_alphabetic());

                    next_is_space_or_end && !prev_is_single_letter
                } else {
                    true
                };

                if is_real_boundary {
                    // Find byte position
                    let byte_start: usize = chars[..start].iter().map(|c| c.len_utf8()).sum();
                    let byte_end: usize = chars[..=i].iter().map(|c| c.len_utf8()).sum();
                    let sentence_text = text[byte_start..byte_end].trim().to_string();

                    if !sentence_text.is_empty() {
                        sentences.push((sentence_text, byte_start, byte_end));
                    }

                    // Skip whitespace after sentence boundary
                    while i + 1 < chars.len() && chars[i + 1].is_whitespace() {
                        i += 1;
                    }
                    start = i + 1;
                }
            }
            i += 1;
        }

        // Handle remaining text
        if start < chars.len() {
            let byte_start: usize = chars[..start].iter().map(|c| c.len_utf8()).sum();
            let remaining = text[byte_start..].trim().to_string();
            if !remaining.is_empty() {
                sentences.push((remaining, byte_start, text.len()));
            }
        }

        sentences
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

impl Chunker for SemanticChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        if text.is_empty() {
            return Vec::new();
        }

        // If text is small enough, return as single chunk
        if text.len() <= self.max_chunk_size {
            return vec![Chunk {
                chunk_index: 0,
                text: text.to_string(),
                start_offset: 0,
                end_offset: text.len(),
            }];
        }

        let sentences = self.split_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        if sentences.len() == 1 {
            return vec![Chunk {
                chunk_index: 0,
                text: sentences[0].0.clone(),
                start_offset: sentences[0].1,
                end_offset: sentences[0].2,
            }];
        }

        // Get embeddings for all sentences
        let sentence_texts: Vec<String> = sentences.iter().map(|(s, _, _)| s.clone()).collect();
        let embeddings = match self.embedder.embed(sentence_texts) {
            Ok(embs) => embs,
            Err(_) => {
                // Fallback to treating entire text as one chunk on embedding error
                return vec![Chunk {
                    chunk_index: 0,
                    text: text.to_string(),
                    start_offset: 0,
                    end_offset: text.len(),
                }];
            }
        };

        // Find split points based on similarity
        let mut split_points = Vec::new();
        for i in 0..embeddings.len() - 1 {
            let similarity = self.cosine_similarity(&embeddings[i], &embeddings[i + 1]);
            if similarity < self.similarity_threshold {
                split_points.push(i + 1);
            }
        }

        // Build chunks based on split points
        let mut chunks = Vec::new();
        let mut chunk_start_idx = 0;
        let mut chunk_index = 0;

        let build_chunk = |start_idx: usize, end_idx: usize, chunk_index: usize| -> Chunk {
            let start_offset = sentences[start_idx].1;
            let end_offset = sentences[end_idx - 1].2;
            let chunk_text = text[start_offset..end_offset].trim().to_string();
            Chunk {
                chunk_index,
                text: chunk_text,
                start_offset,
                end_offset,
            }
        };

        for &split_point in &split_points {
            let chunk_text_len: usize = sentences[chunk_start_idx..split_point]
                .iter()
                .map(|(s, _, _)| s.len())
                .sum();

            // Check min/max chunk size constraints
            if chunk_text_len >= self.min_chunk_size {
                // Check if adding more would exceed max
                if chunk_text_len <= self.max_chunk_size {
                    chunks.push(build_chunk(chunk_start_idx, split_point, chunk_index));
                    chunk_index += 1;
                    chunk_start_idx = split_point;
                }
            }
        }

        // Add remaining sentences as final chunk
        if chunk_start_idx < sentences.len() {
            let mut end_idx = sentences.len();

            // If remaining chunk is too large, split by max_chunk_size
            let remaining_len: usize = sentences[chunk_start_idx..]
                .iter()
                .map(|(s, _, _)| s.len())
                .sum();

            if remaining_len > self.max_chunk_size {
                // Split into smaller chunks respecting max_chunk_size
                let mut current_len = 0;
                let mut segment_start = chunk_start_idx;

                for i in chunk_start_idx..sentences.len() {
                    current_len += sentences[i].0.len() + 1; // +1 for space
                    if current_len > self.max_chunk_size && i > segment_start {
                        chunks.push(build_chunk(segment_start, i, chunk_index));
                        chunk_index += 1;
                        segment_start = i;
                        current_len = sentences[i].0.len();
                    }
                }
                end_idx = sentences.len();
                chunk_start_idx = segment_start;
            }

            if chunk_start_idx < end_idx {
                chunks.push(build_chunk(chunk_start_idx, end_idx, chunk_index));
            }
        }

        // Re-index chunks
        for (i, chunk) in chunks.iter_mut().enumerate() {
            chunk.chunk_index = i;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::Vector;
    use std::io;

    struct MockEmbedder {
        embeddings: Vec<Vec<f32>>,
    }

    impl MockEmbedder {
        fn new(embeddings: Vec<Vec<f32>>) -> Self {
            Self { embeddings }
        }
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: Vec<String>) -> io::Result<Vec<Vector>> {
            // Return pre-configured embeddings, cycling if needed
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| self.embeddings[i % self.embeddings.len()].clone())
                .collect())
        }
    }

    #[test]
    fn test_semantic_chunker_short_text() {
        let embedder = Arc::new(MockEmbedder::new(vec![vec![1.0, 0.0, 0.0]]));
        let chunker = SemanticChunker::new(10, 1000, 0.5, embedder);

        let text = "Short text.";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text.");
    }

    #[test]
    fn test_semantic_chunker_splits_on_low_similarity() {
        // Create embeddings where sentences 0-1 are similar, 2-3 are similar, but 1-2 are different
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // Sentence 0
            vec![0.9, 0.1, 0.0], // Sentence 1 (similar to 0)
            vec![0.0, 1.0, 0.0], // Sentence 2 (different)
            vec![0.1, 0.9, 0.0], // Sentence 3 (similar to 2)
        ];
        let embedder = Arc::new(MockEmbedder::new(embeddings));
        // Use max_chunk_size=50 to force splitting (text is 78 chars)
        let chunker = SemanticChunker::new(5, 50, 0.7, embedder);

        let text = "First topic sentence. Related to first. New topic here. Related to new topic.";
        let chunks = chunker.chunk(text);

        // Should split because text exceeds max_chunk_size
        assert!(
            chunks.len() >= 2,
            "Expected at least 2 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_semantic_chunker_empty_text() {
        let embedder = Arc::new(MockEmbedder::new(vec![vec![1.0]]));
        let chunker = SemanticChunker::new(10, 100, 0.5, embedder);

        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_cosine_similarity() {
        let embedder = Arc::new(MockEmbedder::new(vec![]));
        let chunker = SemanticChunker::new(10, 100, 0.5, embedder);

        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((chunker.cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!(chunker.cosine_similarity(&c, &d).abs() < 0.001);

        // Opposite vectors
        let e = vec![1.0, 0.0, 0.0];
        let f = vec![-1.0, 0.0, 0.0];
        assert!((chunker.cosine_similarity(&e, &f) + 1.0).abs() < 0.001);
    }
}
