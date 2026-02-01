use serde::{Deserialize, Serialize};

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
