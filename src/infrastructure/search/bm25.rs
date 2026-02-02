//! BM25 (Best Matching 25) implementation for keyword search.
//!
//! This module provides a simple BM25 index for text retrieval,
//! designed to work alongside vector search for hybrid retrieval.

use std::collections::{HashMap, HashSet};

/// BM25 parameters
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Term frequency saturation parameter (typically 1.2)
    pub k1: f64,
    /// Document length normalization parameter (typically 0.75)
    pub b: f64,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// A document in the BM25 index
#[derive(Debug, Clone)]
struct Bm25Document {
    /// Token frequencies in this document
    term_freqs: HashMap<String, u32>,
    /// Total number of tokens in this document
    doc_len: u32,
}

/// BM25 index for keyword-based retrieval
#[derive(Debug)]
pub struct Bm25Index {
    /// Configuration parameters
    config: Bm25Config,
    /// Documents indexed by their ID (position in records array)
    documents: HashMap<usize, Bm25Document>,
    /// Document frequency for each term (number of documents containing the term)
    doc_freqs: HashMap<String, u32>,
    /// Total number of documents
    num_docs: u32,
    /// Sum of all document lengths
    total_doc_len: u64,
}

impl Bm25Index {
    /// Create a new empty BM25 index
    pub fn new() -> Self {
        Self::with_config(Bm25Config::default())
    }

    /// Create a new BM25 index with custom configuration
    pub fn with_config(config: Bm25Config) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            doc_freqs: HashMap::new(),
            num_docs: 0,
            total_doc_len: 0,
        }
    }

    /// Add a document to the index
    pub fn add(&mut self, doc_id: usize, text: &str) {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }

        // Count term frequencies
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        // Update document frequencies (only count unique terms per doc)
        for term in term_freqs.keys() {
            *self.doc_freqs.entry(term.clone()).or_insert(0) += 1;
        }

        let doc_len = tokens.len() as u32;
        self.total_doc_len += doc_len as u64;
        self.num_docs += 1;

        self.documents.insert(
            doc_id,
            Bm25Document {
                term_freqs,
                doc_len,
            },
        );
    }

    /// Remove a document from the index
    pub fn remove(&mut self, doc_id: usize) {
        if let Some(doc) = self.documents.remove(&doc_id) {
            // Update document frequencies
            for term in doc.term_freqs.keys() {
                if let Some(df) = self.doc_freqs.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.doc_freqs.remove(term);
                    }
                }
            }

            self.total_doc_len = self.total_doc_len.saturating_sub(doc.doc_len as u64);
            self.num_docs = self.num_docs.saturating_sub(1);
        }
    }

    /// Update a document in the index
    pub fn update(&mut self, doc_id: usize, text: &str) {
        self.remove(doc_id);
        self.add(doc_id, text);
    }

    /// Search the index and return scored results
    /// Returns (doc_id, score) pairs sorted by score descending
    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, f64)> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Deduplicate query tokens for efficiency
        let unique_query_terms: HashSet<_> = query_tokens.into_iter().collect();

        let avg_doc_len = self.total_doc_len as f64 / self.num_docs as f64;

        let mut scores: Vec<(usize, f64)> = self
            .documents
            .iter()
            .map(|(&doc_id, doc)| {
                let score = self.score_document(doc, &unique_query_terms, avg_doc_len);
                (doc_id, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }

    /// Compute BM25 score for a document given query terms
    fn score_document(
        &self,
        doc: &Bm25Document,
        query_terms: &HashSet<String>,
        avg_doc_len: f64,
    ) -> f64 {
        let k1 = self.config.k1;
        let b = self.config.b;
        let n = self.num_docs as f64;

        let mut score = 0.0;

        for term in query_terms {
            // Get term frequency in this document
            let tf = doc.term_freqs.get(term).copied().unwrap_or(0) as f64;
            if tf == 0.0 {
                continue;
            }

            // Get document frequency
            let df = self.doc_freqs.get(term).copied().unwrap_or(0) as f64;
            if df == 0.0 {
                continue;
            }

            // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            // TF normalization: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            let dl = doc.doc_len as f64;
            let tf_norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avg_doc_len));

            score += idf * tf_norm;
        }

        score
    }

    /// Get the number of indexed documents
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokenize text for BM25 indexing and search (English-focused)
/// - Lowercase
/// - Split on non-alphanumeric characters
/// - Filter empty tokens
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a TEST.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("Version 2.0 released in 2024!");
        assert_eq!(tokens, vec!["version", "2", "0", "released", "in", "2024"]);
    }

    #[test]
    fn test_bm25_basic() {
        let mut index = Bm25Index::new();
        index.add(0, "the quick brown fox");
        index.add(1, "the lazy dog");
        index.add(2, "the quick brown dog");

        let results = index.search("quick fox", 10);
        assert!(!results.is_empty());
        // Document 0 should be ranked highest (has both "quick" and "fox")
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_bm25_remove() {
        let mut index = Bm25Index::new();
        index.add(0, "hello world");
        index.add(1, "goodbye world");

        assert_eq!(index.len(), 2);

        index.remove(0);
        assert_eq!(index.len(), 1);

        let results = index.search("hello", 10);
        assert!(results.is_empty()); // "hello" was only in doc 0
    }

    #[test]
    fn test_bm25_update() {
        let mut index = Bm25Index::new();
        index.add(0, "hello world");

        let results = index.search("hello", 10);
        assert_eq!(results.len(), 1);

        index.update(0, "goodbye world");

        let results = index.search("hello", 10);
        assert!(results.is_empty());

        let results = index.search("goodbye", 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bm25_empty_query() {
        let mut index = Bm25Index::new();
        index.add(0, "hello world");

        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_idf_ranking() {
        let mut index = Bm25Index::new();
        // "the" appears in all documents (low IDF)
        // "rare" appears only in one document (high IDF)
        index.add(0, "the common word");
        index.add(1, "the common phrase");
        index.add(2, "the rare unique word");

        let results = index.search("rare", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);

        // "common" appears in 2 docs, should rank them above doc with "rare"
        let results = index.search("common", 10);
        assert_eq!(results.len(), 2);
    }
}
