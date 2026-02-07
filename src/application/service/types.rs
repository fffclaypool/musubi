//! Type definitions for the document service.

use crate::domain::model::{Record, Tag};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Policy for tombstone-based compaction
#[derive(Debug, Clone)]
pub enum TombstonePolicy {
    /// No automatic compaction
    Disabled,
    /// Compact when tombstone count reaches this threshold
    MaxCount(usize),
    /// Compact when tombstone ratio (tombstones / total) reaches this threshold (0.0 - 1.0)
    MaxRatio(f64),
}

impl Default for TombstonePolicy {
    fn default() -> Self {
        // Default: compact when 30% are tombstones
        Self::MaxRatio(0.3)
    }
}

/// Configuration for tombstone-based compaction
#[derive(Debug, Clone)]
pub struct TombstoneConfig {
    /// Maximum number of tombstones before triggering compaction
    pub max_tombstones: Option<usize>,
    /// Maximum ratio of tombstones to total records before triggering compaction (0.0 - 1.0)
    pub max_tombstone_ratio: Option<f64>,
}

impl Default for TombstoneConfig {
    fn default() -> Self {
        Self {
            max_tombstones: None,
            max_tombstone_ratio: Some(0.3), // Default: compact when 30% are tombstones
        }
    }
}

impl TombstoneConfig {
    /// Convert to TombstonePolicy, validating that at most one option is set
    pub fn into_policy(self) -> Result<TombstonePolicy, &'static str> {
        match (self.max_tombstones, self.max_tombstone_ratio) {
            (None, None) => Ok(TombstonePolicy::Disabled),
            (Some(count), None) => Ok(TombstonePolicy::MaxCount(count)),
            (None, Some(ratio)) => {
                if ratio.is_nan() || !(0.0..=1.0).contains(&ratio) {
                    return Err("max_tombstone_ratio must be between 0.0 and 1.0");
                }
                Ok(TombstonePolicy::MaxRatio(ratio))
            }
            (Some(_), Some(_)) => Err("cannot set both max_tombstones and max_tombstone_ratio"),
        }
    }
}

/// Search mode determining the balance between vector and BM25 search
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchMode {
    /// Pure vector similarity search (alpha = 1.0)
    VectorOnly,
    /// Pure BM25 keyword search (alpha = 0.0)
    Bm25Only,
    /// Hybrid search combining vector and BM25 with given weight
    /// alpha is the weight for vector score (0.0 < alpha < 1.0)
    Hybrid { alpha: f64 },
}

impl SearchMode {
    /// Create SearchMode from alpha value with validation
    ///
    /// # Errors
    /// Returns error if alpha is NaN or outside [0.0, 1.0]
    pub fn from_alpha(alpha: f64) -> Result<Self, &'static str> {
        if alpha.is_nan() {
            return Err("alpha must not be NaN");
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err("alpha must be between 0.0 and 1.0");
        }

        if alpha == 1.0 {
            Ok(Self::VectorOnly)
        } else if alpha == 0.0 {
            Ok(Self::Bm25Only)
        } else {
            Ok(Self::Hybrid { alpha })
        }
    }

    /// Get the alpha value for this search mode
    pub fn alpha(&self) -> f64 {
        match self {
            Self::VectorOnly => 1.0,
            Self::Bm25Only => 0.0,
            Self::Hybrid { alpha } => *alpha,
        }
    }

    /// Returns true if vector search is needed
    pub fn needs_vector(&self) -> bool {
        !matches!(self, Self::Bm25Only)
    }

    /// Returns true if BM25 search is needed
    pub fn needs_bm25(&self) -> bool {
        !matches!(self, Self::VectorOnly)
    }
}

/// Configuration for document chunking
#[derive(Debug, Clone, Default)]
pub enum ChunkConfig {
    /// No chunking - treat each document as a single unit (default, backward compatible)
    #[default]
    None,
    /// Fixed-size chunking with overlap
    Fixed {
        /// Target chunk size in characters
        chunk_size: usize,
        /// Number of characters to overlap between chunks
        overlap: usize,
    },
    /// Semantic chunking based on sentence similarity
    Semantic {
        /// Minimum chunk size in characters
        min_chunk_size: usize,
        /// Maximum chunk size in characters
        max_chunk_size: usize,
        /// Similarity threshold (0.0-1.0) - split when similarity drops below this
        similarity_threshold: f32,
    },
}

/// Service configuration
pub struct ServiceConfig {
    pub snapshot_path: std::path::PathBuf,
    pub default_k: usize,
    pub default_ef: usize,
    pub wal_config: Option<crate::infrastructure::storage::wal::WalConfig>,
    pub tombstone_config: TombstoneConfig,
    pub chunk_config: ChunkConfig,
}

#[derive(Debug, Clone)]
pub struct InsertCommand {
    pub record: Record,
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UpdateCommand {
    pub title: Option<String>,
    pub body: Option<String>,
    pub source: Option<String>,
    pub updated_at: Option<String>,
    pub tags: Option<Vec<Tag>>,
    pub text: Option<String>,
}

/// Tag filter mode for SearchFilter
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum TagFilter {
    /// No tag filtering
    #[default]
    None,
    /// Match if any of these tags are present
    Any(Vec<Tag>),
    /// Match only if all of these tags are present
    All(Vec<Tag>),
}

impl TagFilter {
    /// Check if record tags match this filter
    fn matches(&self, record_tags: &[Tag]) -> bool {
        let record_set: HashSet<&str> = record_tags.iter().map(Tag::as_str).collect();
        match self {
            Self::None => true,
            Self::Any(filter_tags) => {
                filter_tags.is_empty()
                    || filter_tags.iter().any(|t| record_set.contains(t.as_str()))
            }
            Self::All(filter_tags) => {
                filter_tags.is_empty()
                    || filter_tags.iter().all(|t| record_set.contains(t.as_str()))
            }
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilter {
    /// Exact match on source field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Tag filter (none, any, or all)
    #[serde(default, flatten)]
    pub tag_filter: TagFilter,
    /// Match if updated_at >= this value (string comparison, YYYY-MM-DD)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at_gte: Option<String>,
    /// Match if updated_at <= this value (string comparison, YYYY-MM-DD)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at_lte: Option<String>,
}

impl SearchFilter {
    /// Check if a record matches all filter criteria
    pub fn matches(&self, record: &Record) -> bool {
        // source: exact match
        if let Some(ref filter_source) = self.source {
            match &record.source {
                Some(record_source) if record_source == filter_source => {}
                _ => return false,
            }
        }

        // tag filter
        if !self.tag_filter.matches(&record.tags) {
            return false;
        }

        // updated_at_gte
        if let Some(ref gte) = self.updated_at_gte {
            match &record.updated_at {
                Some(updated_at) if updated_at.as_str() >= gte.as_str() => {}
                _ => return false,
            }
        }

        // updated_at_lte
        if let Some(ref lte) = self.updated_at_lte {
            match &record.updated_at {
                Some(updated_at) if updated_at.as_str() <= lte.as_str() => {}
                _ => return false,
            }
        }

        true
    }
}

/// Search input - validated query content
#[derive(Debug, Clone)]
pub enum SearchInput {
    /// Text query only - will be embedded by the service
    Text(String),
    /// Pre-computed embedding only
    Embedding(Vec<f32>),
    /// Both text and embedding provided
    TextAndEmbedding { text: String, embedding: Vec<f32> },
}

impl SearchInput {
    /// Get text if available
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(t) | Self::TextAndEmbedding { text: t, .. } => Some(t),
            Self::Embedding(_) => None,
        }
    }

    /// Get embedding if available
    pub fn embedding(&self) -> Option<&[f32]> {
        match self {
            Self::Embedding(e) | Self::TextAndEmbedding { embedding: e, .. } => Some(e),
            Self::Text(_) => None,
        }
    }
}

/// Validated search parameters
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of results to return
    pub k: usize,
    /// Search expansion factor for HNSW
    pub ef: usize,
    /// Search mode (vector-only, BM25-only, or hybrid)
    pub mode: SearchMode,
}

/// Validated search query - all validation done at construction time
#[derive(Debug, Clone)]
pub struct ValidatedSearchQuery {
    /// The search input (text, embedding, or both)
    pub input: SearchInput,
    /// Search parameters
    pub params: SearchParams,
    /// Optional metadata filter
    pub filter: Option<SearchFilter>,
}

/// Error type for search query validation
#[derive(Debug, Clone)]
pub enum SearchValidationError {
    /// Alpha value is invalid
    InvalidAlpha(&'static str),
    /// BM25-only mode requires text
    Bm25RequiresText,
    /// Vector/hybrid mode requires text or embedding
    VectorRequiresInput,
    /// Text is empty after trimming
    EmptyText,
    /// Embedding vector is empty
    EmptyEmbedding,
}

impl std::fmt::Display for SearchValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha(msg) => write!(f, "{}", msg),
            Self::Bm25RequiresText => write!(f, "alpha=0.0 (BM25-only) requires 'text' parameter"),
            Self::VectorRequiresInput => write!(f, "text or embedding is required"),
            Self::EmptyText => write!(f, "text must not be empty"),
            Self::EmptyEmbedding => write!(f, "embedding must not be empty"),
        }
    }
}

impl std::error::Error for SearchValidationError {}

/// Raw search request - used for parsing before validation
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchRequest {
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub k: Option<usize>,
    pub ef: Option<usize>,
    pub alpha: Option<f64>,
    pub filter: Option<SearchFilter>,
}

impl ValidatedSearchQuery {
    /// Validate and construct a search query from a raw request.
    ///
    /// # Arguments
    /// * `req` - The raw search request
    /// * `default_k` - Default number of results
    /// * `default_ef` - Default search expansion factor
    pub fn from_request(
        req: SearchRequest,
        default_k: usize,
        default_ef: usize,
    ) -> Result<Self, SearchValidationError> {
        // Parse and validate search mode from alpha
        let mode = SearchMode::from_alpha(req.alpha.unwrap_or(0.7))
            .map_err(SearchValidationError::InvalidAlpha)?;

        // Validate text: explicit empty string is an error
        let text = match req.text {
            Some(t) => {
                let trimmed = t.trim();
                if trimmed.is_empty() {
                    return Err(SearchValidationError::EmptyText);
                }
                Some(trimmed.to_string())
            }
            None => None,
        };

        // Validate embedding: explicit empty vector is an error
        let embedding = match req.embedding {
            Some(e) if e.is_empty() => return Err(SearchValidationError::EmptyEmbedding),
            other => other,
        };

        // Build SearchInput based on mode and available inputs
        let input = match mode {
            SearchMode::Bm25Only => {
                // BM25-only requires text
                match text {
                    Some(t) => SearchInput::Text(t),
                    None => return Err(SearchValidationError::Bm25RequiresText),
                }
            }
            SearchMode::VectorOnly | SearchMode::Hybrid { .. } => {
                // Vector/hybrid requires at least text or embedding
                match (text, embedding) {
                    (Some(t), Some(e)) => SearchInput::TextAndEmbedding {
                        text: t,
                        embedding: e,
                    },
                    (Some(t), None) => SearchInput::Text(t),
                    (None, Some(e)) => SearchInput::Embedding(e),
                    (None, None) => return Err(SearchValidationError::VectorRequiresInput),
                }
            }
        };

        Ok(Self {
            input,
            params: SearchParams {
                k: req.k.unwrap_or(default_k),
                ef: req.ef.unwrap_or(default_ef),
                mode,
            },
            filter: req.filter,
        })
    }
}

/// Information about the best matching chunk within a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Index of the chunk within the parent document
    pub chunk_index: usize,
    /// Preview of the chunk text (first ~100 chars)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_preview: Option<String>,
}

/// Score breakdown by search mode - ensures type-level consistency
/// between search mode and available scores
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScoreBreakdown {
    /// Vector similarity search only
    VectorOnly {
        /// L2 distance (lower is better)
        distance: f32,
    },
    /// BM25 keyword search only
    Bm25Only {
        /// Raw BM25 score (higher is better)
        bm25_score: f64,
    },
    /// Hybrid search combining vector and BM25
    Hybrid {
        /// L2 distance (lower is better), None if only BM25 matched
        #[serde(skip_serializing_if = "Option::is_none")]
        distance: Option<f32>,
        /// Raw BM25 score (higher is better), None if only vector matched
        #[serde(skip_serializing_if = "Option::is_none")]
        bm25_score: Option<f64>,
        /// Combined hybrid score (higher is better)
        hybrid_score: f64,
    },
}

impl ScoreBreakdown {
    /// Get the primary score used for ranking (higher is better)
    pub fn ranking_score(&self) -> f64 {
        match self {
            Self::VectorOnly { distance } => -(*distance as f64), // Negate: lower distance = higher score
            Self::Bm25Only { bm25_score } => *bm25_score,
            Self::Hybrid { hybrid_score, .. } => *hybrid_score,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub index_id: usize,
    pub id: String,
    /// Score breakdown matching the search mode
    #[serde(flatten)]
    pub score: ScoreBreakdown,
    pub title: Option<String>,
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<Tag>,
    /// Best matching chunk info (only present when chunking is enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_chunk: Option<ChunkInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InsertResult {
    pub index_id: usize,
    pub id: String,
    pub dim: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentSummary {
    pub index_id: usize,
    #[serde(flatten)]
    pub record: Record,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentResponse {
    pub index_id: usize,
    #[serde(flatten)]
    pub record: Record,
    pub embedding: Vec<f32>,
}

/// Internal enum representing the effect of an update operation
pub(super) enum UpdateEffect {
    /// Only metadata changed, no re-embedding needed
    MetadataOnly,
    /// Content changed, re-embedding required
    ContentChanged { text: String },
}

/// Document indexing mode - determines how documents are stored and searched
pub(super) enum IndexingMode {
    /// Documents indexed as-is (no chunking)
    Direct,
    /// Documents split into chunks for finer-grained search
    Chunked {
        chunker: Box<dyn crate::domain::ports::Chunker>,
        chunk_store: Box<dyn crate::domain::ports::ChunkStore>,
        chunks: Vec<crate::domain::model::StoredChunk>,
        /// Maps index ID -> (record_index, chunk_index within that record)
        chunk_mapping: Vec<(usize, usize)>,
    },
}

impl IndexingMode {
    /// Returns true if chunking is enabled and has data
    pub(super) fn is_chunked_with_data(&self) -> bool {
        matches!(self, Self::Chunked { chunk_mapping, .. } if !chunk_mapping.is_empty())
    }

    /// Get chunks (empty slice for Direct mode)
    pub(super) fn chunks(&self) -> &[crate::domain::model::StoredChunk] {
        match self {
            Self::Direct => &[],
            Self::Chunked { chunks, .. } => chunks,
        }
    }

    /// Get chunk mapping (empty slice for Direct mode)
    pub(super) fn chunk_mapping(&self) -> &[(usize, usize)] {
        match self {
            Self::Direct => &[],
            Self::Chunked { chunk_mapping, .. } => chunk_mapping,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        id: &str,
        source: Option<&str>,
        tags: Option<&str>,
        updated_at: Option<&str>,
    ) -> Record {
        Record {
            id: id.to_string(),
            title: Some("Test".to_string()),
            body: None,
            source: source.map(|s| s.to_string()),
            updated_at: updated_at.map(|s| s.to_string()),
            tags: tags.map(Tag::parse_many).unwrap_or_default(),
        }
    }

    #[test]
    fn test_filter_empty_matches_all() {
        let filter = SearchFilter::default();
        let record = make_record("1", Some("news"), Some("ai, rust"), Some("2024-06-15"));
        assert!(filter.matches(&record));
    }

    #[test]
    fn test_filter_source_exact_match() {
        let filter = SearchFilter {
            source: Some("news".to_string()),
            ..Default::default()
        };
        let record = make_record("1", Some("news"), None, None);
        assert!(filter.matches(&record));

        let record2 = make_record("2", Some("blog"), None, None);
        assert!(!filter.matches(&record2));

        let record3 = make_record("3", None, None, None);
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_tags_any() {
        let filter = SearchFilter {
            tag_filter: TagFilter::Any(vec![Tag::new("ai"), Tag::new("ml")]),
            ..Default::default()
        };

        // Has "ai" tag
        let record1 = make_record("1", None, Some("ai, rust"), None);
        assert!(filter.matches(&record1));

        // Has "ml" tag
        let record2 = make_record("2", None, Some("ml, python"), None);
        assert!(filter.matches(&record2));

        // Has neither
        let record3 = make_record("3", None, Some("rust, go"), None);
        assert!(!filter.matches(&record3));

        // No tags at all
        let record4 = make_record("4", None, None, None);
        assert!(!filter.matches(&record4));
    }

    #[test]
    fn test_filter_tags_all() {
        let filter = SearchFilter {
            tag_filter: TagFilter::All(vec![Tag::new("ai"), Tag::new("rust")]),
            ..Default::default()
        };

        // Has both tags
        let record1 = make_record("1", None, Some("ai, rust, news"), None);
        assert!(filter.matches(&record1));

        // Has only one
        let record2 = make_record("2", None, Some("ai, python"), None);
        assert!(!filter.matches(&record2));

        // Has neither
        let record3 = make_record("3", None, Some("go, python"), None);
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_tags_case_insensitive() {
        let filter = SearchFilter {
            tag_filter: TagFilter::Any(vec![Tag::new("AI"), Tag::new("RUST")]),
            ..Default::default()
        };

        let record = make_record("1", None, Some("ai, rust"), None);
        assert!(filter.matches(&record));

        let filter2 = SearchFilter {
            tag_filter: TagFilter::All(vec![Tag::new("AI")]),
            ..Default::default()
        };
        assert!(filter2.matches(&record));
    }

    #[test]
    fn test_filter_tags_with_whitespace() {
        let filter = SearchFilter {
            tag_filter: TagFilter::Any(vec![Tag::new("ai")]),
            ..Default::default()
        };

        // Tags with extra whitespace
        let record = make_record("1", None, Some("  ai  ,  rust  "), None);
        assert!(filter.matches(&record));
    }

    #[test]
    fn test_filter_updated_at_gte() {
        let filter = SearchFilter {
            updated_at_gte: Some("2024-06-01".to_string()),
            ..Default::default()
        };

        let record1 = make_record("1", None, None, Some("2024-06-15"));
        assert!(filter.matches(&record1));

        let record2 = make_record("2", None, None, Some("2024-06-01"));
        assert!(filter.matches(&record2));

        let record3 = make_record("3", None, None, Some("2024-05-31"));
        assert!(!filter.matches(&record3));

        // No updated_at
        let record4 = make_record("4", None, None, None);
        assert!(!filter.matches(&record4));
    }

    #[test]
    fn test_filter_updated_at_lte() {
        let filter = SearchFilter {
            updated_at_lte: Some("2024-06-30".to_string()),
            ..Default::default()
        };

        let record1 = make_record("1", None, None, Some("2024-06-15"));
        assert!(filter.matches(&record1));

        let record2 = make_record("2", None, None, Some("2024-06-30"));
        assert!(filter.matches(&record2));

        let record3 = make_record("3", None, None, Some("2024-07-01"));
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_updated_at_range() {
        let filter = SearchFilter {
            updated_at_gte: Some("2024-01-01".to_string()),
            updated_at_lte: Some("2024-12-31".to_string()),
            ..Default::default()
        };

        let record1 = make_record("1", None, None, Some("2024-06-15"));
        assert!(filter.matches(&record1));

        let record2 = make_record("2", None, None, Some("2023-12-31"));
        assert!(!filter.matches(&record2));

        let record3 = make_record("3", None, None, Some("2025-01-01"));
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_combined() {
        let filter = SearchFilter {
            source: Some("news".to_string()),
            tag_filter: TagFilter::Any(vec![Tag::new("ai"), Tag::new("rust")]),
            updated_at_gte: Some("2024-01-01".to_string()),
            ..Default::default()
        };

        // All conditions match
        let record1 = make_record("1", Some("news"), Some("ai, tech"), Some("2024-06-15"));
        assert!(filter.matches(&record1));

        // Wrong source
        let record2 = make_record("2", Some("blog"), Some("ai, tech"), Some("2024-06-15"));
        assert!(!filter.matches(&record2));

        // No matching tags
        let record3 = make_record("3", Some("news"), Some("go, python"), Some("2024-06-15"));
        assert!(!filter.matches(&record3));

        // Too old
        let record4 = make_record("4", Some("news"), Some("ai, tech"), Some("2023-12-31"));
        assert!(!filter.matches(&record4));
    }

    #[test]
    fn test_tag_parse_many() {
        let tags = Tag::parse_many("ai, rust, ML");
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].as_str(), "ai");
        assert_eq!(tags[1].as_str(), "rust");
        assert_eq!(tags[2].as_str(), "ml"); // normalized to lowercase

        let empty = Tag::parse_many("");
        assert!(empty.is_empty());

        let with_spaces = Tag::parse_many("  ai  ,  ,  rust  ");
        assert_eq!(with_spaces.len(), 2);
        assert_eq!(with_spaces[0].as_str(), "ai");
        assert_eq!(with_spaces[1].as_str(), "rust");
    }
}
