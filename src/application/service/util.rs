//! Utility functions for the document service.

use sha2::{Digest, Sha256};

use crate::domain::model::{Record, StoredChunk, StoredRecord};
use crate::domain::ports::VectorIndex;
use std::collections::HashMap;

/// Build text for embedding from explicit text or record fields.
pub(super) fn build_text(explicit: Option<&str>, record: &Record) -> Option<String> {
    if let Some(text) = explicit {
        if !text.trim().is_empty() {
            return Some(text.to_string());
        }
    }
    let title = record.title.as_deref().unwrap_or("").trim();
    let body = record.body.as_deref().unwrap_or("").trim();
    match (title.is_empty(), body.is_empty()) {
        (true, true) => None,
        (false, true) => Some(title.to_string()),
        (true, false) => Some(body.to_string()),
        (false, false) => Some(format!("{}\n\n{}", title, body)),
    }
}

/// Fill missing embeddings from the vector index.
pub(super) fn fill_missing_embeddings(
    records: &mut [StoredRecord],
    index: &dyn VectorIndex,
) -> bool {
    let mut filled = false;
    for (idx, record) in records.iter_mut().enumerate() {
        if record.embedding.is_empty() {
            if let Some(vector) = index.vector(idx) {
                record.embedding = vector.clone();
                filled = true;
            }
        }
    }
    filled
}

/// Build a mapping from vector index position to (record_index, chunk_index_within_record)
/// IMPORTANT: This includes ALL chunks (including deleted) to maintain 1:1 alignment with the vector index.
/// The search function must check chunk.deleted separately.
pub(super) fn build_chunk_mapping(
    chunks: &[StoredChunk],
    records: &[StoredRecord],
) -> Vec<(usize, usize)> {
    // Create a map from record ID to record index (include deleted records for mapping purposes)
    let id_to_idx: HashMap<&str, usize> = records
        .iter()
        .enumerate()
        .map(|(idx, r)| (r.record.id.as_str(), idx))
        .collect();

    // Include ALL chunks to maintain index alignment
    // If a chunk's parent is not found, use usize::MAX as a sentinel
    chunks
        .iter()
        .map(|c| {
            let record_idx = id_to_idx
                .get(c.parent_id.as_str())
                .copied()
                .unwrap_or(usize::MAX);
            (record_idx, c.chunk.chunk_index)
        })
        .collect()
}

/// Convert chunks to fake StoredRecords for index rebuilding
pub(super) fn chunks_to_fake_records(chunks: &[StoredChunk]) -> Vec<StoredRecord> {
    chunks
        .iter()
        .map(|c| {
            StoredRecord::with_deleted(
                Record {
                    id: format!("{}_{}", c.parent_id, c.chunk.chunk_index),
                    title: None,
                    body: None,
                    source: None,
                    updated_at: None,
                    tags: Vec::new(),
                },
                c.embedding.clone(),
                c.deleted,
            )
        })
        .collect()
}

/// Create a text preview from chunk text (first ~100 characters)
pub(super) fn create_text_preview(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        // Find a good break point (word boundary)
        let truncated = &text[..max_len];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &truncated[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

/// Calculate content hash with strict normalization for differential sync.
///
/// Normalization rules:
/// 1. title: trimmed, empty string if None
/// 2. body: trimmed, empty string if None
/// 3. text: trimmed, empty string if None (explicit embedding text)
/// 4. tags: lowercase, sorted alphabetically, joined with ","
/// 5. updated_at: "YYYY-MM-DD" or empty string
/// 6. Concatenate: "{title}\x00{body}\x00{text}\x00{tags}\x00{updated_at}"
/// 7. Hash: SHA-256, hex-encoded lowercase
pub fn calculate_content_hash(record: &Record, text: Option<&str>) -> String {
    let mut hasher = Sha256::new();

    // 1. title (trimmed, empty if None)
    let title = record.title.as_deref().unwrap_or("").trim();
    hasher.update(title.as_bytes());
    hasher.update(b"\x00");

    // 2. body (trimmed, empty if None)
    let body = record.body.as_deref().unwrap_or("").trim();
    hasher.update(body.as_bytes());
    hasher.update(b"\x00");

    // 3. text (trimmed, empty if None)
    let text_val = text.unwrap_or("").trim();
    hasher.update(text_val.as_bytes());
    hasher.update(b"\x00");

    // 4. tags (lowercase, sorted, comma-joined)
    let mut tags: Vec<&str> = record.tags.iter().map(|t| t.as_str()).collect();
    tags.sort();
    hasher.update(tags.join(",").as_bytes());
    hasher.update(b"\x00");

    // 5. updated_at (YYYY-MM-DD or empty)
    if let Some(date) = record.updated_at {
        hasher.update(date.format("%Y-%m-%d").to_string().as_bytes());
    }

    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::model::Tag;
    use chrono::NaiveDate;

    #[test]
    fn test_content_hash_deterministic() {
        let record = Record {
            id: "test".to_string(),
            title: Some("Title".to_string()),
            body: Some("Body".to_string()),
            source: None,
            updated_at: Some(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap()),
            tags: vec![Tag::new("rust"), Tag::new("ai")],
        };

        let hash1 = calculate_content_hash(&record, None);
        let hash2 = calculate_content_hash(&record, None);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_content_hash_tag_order_independent() {
        let record1 = Record {
            id: "test".to_string(),
            title: Some("Title".to_string()),
            body: None,
            source: None,
            updated_at: None,
            tags: vec![Tag::new("rust"), Tag::new("ai")],
        };

        let record2 = Record {
            id: "test".to_string(),
            title: Some("Title".to_string()),
            body: None,
            source: None,
            updated_at: None,
            tags: vec![Tag::new("ai"), Tag::new("rust")],
        };

        assert_eq!(
            calculate_content_hash(&record1, None),
            calculate_content_hash(&record2, None)
        );
    }

    #[test]
    fn test_content_hash_whitespace_normalized() {
        let record1 = Record {
            id: "test".to_string(),
            title: Some("  Title  ".to_string()),
            body: Some("  Body  ".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
        };

        let record2 = Record {
            id: "test".to_string(),
            title: Some("Title".to_string()),
            body: Some("Body".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
        };

        assert_eq!(
            calculate_content_hash(&record1, None),
            calculate_content_hash(&record2, None)
        );
    }

    #[test]
    fn test_content_hash_explicit_text() {
        let record = Record {
            id: "test".to_string(),
            title: Some("Title".to_string()),
            body: None,
            source: None,
            updated_at: None,
            tags: vec![],
        };

        let hash_without_text = calculate_content_hash(&record, None);
        let hash_with_text = calculate_content_hash(&record, Some("explicit text"));

        assert_ne!(hash_without_text, hash_with_text);
    }
}
