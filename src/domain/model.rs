use serde::{Deserialize, Deserializer, Serialize};

/// A normalized tag (lowercase, trimmed)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Tag(String);

impl<'de> Deserialize<'de> for Tag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Tag::new(s))
    }
}

impl Tag {
    /// Create a new tag, normalizing to lowercase and trimmed
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into().trim().to_lowercase())
    }

    /// Get the tag value as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Parse comma-separated tags into a Vec<Tag>
    pub fn parse_many(tags: &str) -> Vec<Tag> {
        tags.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(Tag::new)
            .collect()
    }
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A chunk of text extracted from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Index of this chunk within the parent document (0-based)
    pub chunk_index: usize,
    /// The text content of this chunk
    pub text: String,
    /// Start offset in the original document (character position)
    pub start_offset: usize,
    /// End offset in the original document (character position)
    pub end_offset: usize,
}

/// A stored chunk with its embedding and parent reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    /// The ID of the parent document
    pub parent_id: String,
    /// The chunk data
    #[serde(flatten)]
    pub chunk: Chunk,
    /// The embedding vector for this chunk
    pub embedding: Vec<f32>,
    /// Tombstone flag for soft deletion
    #[serde(default)]
    pub deleted: bool,
}

impl StoredChunk {
    pub fn new(parent_id: String, chunk: Chunk, embedding: Vec<f32>) -> Self {
        Self {
            parent_id,
            chunk,
            embedding,
            deleted: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub title: Option<String>,
    pub body: Option<String>,
    pub source: Option<String>,
    pub updated_at: Option<String>,
    /// Tags associated with this record (normalized to lowercase)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<Tag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRecord {
    #[serde(flatten)]
    pub record: Record,
    pub embedding: Vec<f32>,
    /// Tombstone flag for soft deletion. Defaults to false for backwards compatibility.
    #[serde(default)]
    pub deleted: bool,
}

impl StoredRecord {
    pub fn new(record: Record, embedding: Vec<f32>) -> Self {
        Self {
            record,
            embedding,
            deleted: false,
        }
    }

    pub fn with_deleted(record: Record, embedding: Vec<f32>, deleted: bool) -> Self {
        Self {
            record,
            embedding,
            deleted,
        }
    }
}
