use serde::{Deserialize, Serialize};

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
    pub tags: Option<String>,
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
