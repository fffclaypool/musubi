//! Read operations (get, list, embed, import) for DocumentService.

use std::collections::HashSet;

use crate::application::error::AppError;
use crate::domain::model::StoredRecord;

use super::core::DocumentService;
use super::traits::DocumentRead;
use super::types::{DocumentResponse, DocumentSummary};

impl DocumentRead for DocumentService {
    fn default_k(&self) -> usize {
        self.default_k
    }

    fn default_ef(&self) -> usize {
        self.default_ef
    }

    fn get(&self, id: &str) -> Result<DocumentResponse, AppError> {
        let index_id = self.find_index(id)?;
        let stored = self.records[index_id].clone();
        Ok(DocumentResponse {
            index_id,
            record: stored.record,
            embedding: stored.embedding,
        })
    }

    fn list(&self, offset: usize, limit: usize) -> (usize, Vec<DocumentSummary>) {
        // Filter out tombstones
        let active_records: Vec<(usize, &StoredRecord)> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.deleted)
            .collect();

        let total = active_records.len();
        if offset >= total {
            return (total, Vec::new());
        }

        let end = (offset + limit).min(total);
        let items = active_records[offset..end]
            .iter()
            .map(|(idx, record)| DocumentSummary {
                index_id: *idx,
                record: record.record.clone(),
            })
            .collect();

        (total, items)
    }

    fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Err(AppError::BadRequest("texts is required".to_string()));
        }
        Ok(self.embedder.embed(texts)?)
    }
}

// import_embeddings is in DocumentWrite trait, implemented here for code organization
impl DocumentService {
    /// Import pre-computed embeddings (part of DocumentWrite trait).
    pub(super) fn import_embeddings_impl(
        &mut self,
        records: Vec<StoredRecord>,
    ) -> Result<usize, AppError> {
        if records.is_empty() {
            return Ok(0);
        }

        let mut seen = HashSet::new();
        for record in &records {
            if record.record.id.trim().is_empty() {
                return Err(AppError::BadRequest("id is required".to_string()));
            }
            if record.embedding.is_empty() {
                return Err(AppError::BadRequest("embedding is required".to_string()));
            }
            if !seen.insert(&record.record.id) {
                return Err(AppError::Conflict(format!(
                    "duplicate id in import: {}",
                    record.record.id
                )));
            }
        }

        for record in &records {
            if self
                .records
                .iter()
                .any(|existing| existing.record.id == record.record.id && !existing.deleted)
            {
                return Err(AppError::Conflict(format!(
                    "id already exists: {}",
                    record.record.id
                )));
            }
        }

        // Write to WAL first
        if let Some(ref mut wal) = self.wal {
            for record in &records {
                wal.append_insert(record)?;
            }
        }

        // Add to index and records, tracking tombstones
        let deleted_count = records.iter().filter(|r| r.deleted).count();
        for record in &records {
            self.index.insert(record.embedding.clone());
        }
        self.records.extend(records);
        self.tombstone_count += deleted_count;

        self.index.save(&self.snapshot_path)?;
        self.record_store.save_all(&self.records)?;

        // Check if WAL rotation is needed
        self.maybe_rotate_wal()?;

        // Return count of active (non-deleted) records
        Ok(self.records.iter().filter(|r| !r.deleted).count())
    }
}
