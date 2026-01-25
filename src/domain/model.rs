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
}
