//! Core DocumentService struct definition and private helpers.

use crate::application::error::AppError;
use crate::domain::model::StoredRecord;
use crate::domain::ports::{Embedder, RecordStore, VectorIndex, VectorIndexFactory};
use crate::infrastructure::search::Bm25Index;
use crate::infrastructure::storage::wal::{WalConfig, WalWriter};
use std::path::PathBuf;

use super::types::{ChunkConfig, IndexingMode, TombstonePolicy};

/// The main document service for managing documents with vector search.
pub struct DocumentService {
    pub(super) index: Box<dyn VectorIndex>,
    pub(super) index_factory: Box<dyn VectorIndexFactory>,
    pub(super) records: Vec<StoredRecord>,
    pub(super) embedder: Box<dyn Embedder>,
    pub(super) record_store: Box<dyn RecordStore>,
    pub(super) snapshot_path: PathBuf,
    pub(super) default_k: usize,
    pub(super) default_ef: usize,
    pub(super) wal: Option<WalWriter>,
    pub(super) wal_config: Option<WalConfig>,
    pub(super) tombstone_policy: TombstonePolicy,
    pub(super) tombstone_count: usize,
    pub(super) bm25_index: Bm25Index,
    pub(super) indexing_mode: IndexingMode,
    #[allow(dead_code)]
    pub(super) chunk_config: ChunkConfig,
}

impl DocumentService {
    /// Find the index of a record by ID.
    pub(super) fn find_index(&self, id: &str) -> Result<usize, AppError> {
        self.records
            .iter()
            .position(|record| record.record.id == id && !record.deleted)
            .ok_or_else(|| AppError::NotFound("record not found".to_string()))
    }

    /// Embed a single text string.
    pub(super) fn embed_single(&self, text: String) -> Result<Vec<f32>, AppError> {
        self.embedder
            .embed(vec![text])?
            .into_iter()
            .next()
            .ok_or_else(|| AppError::Io("embedding response is empty".to_string()))
    }
}
