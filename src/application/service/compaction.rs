//! Compaction and WAL rotation operations for DocumentService.

use crate::application::error::AppError;
use crate::infrastructure::search::Bm25Index;

use super::core::DocumentService;
use super::types::{IndexingMode, TombstonePolicy};
use super::util::{build_chunk_mapping, build_text, chunks_to_fake_records};

impl DocumentService {
    /// Get tombstone statistics.
    pub fn tombstone_stats(&self) -> (usize, usize, f64) {
        let total = self.records.len();
        let tombstones = self.tombstone_count;
        let ratio = if total > 0 {
            tombstones as f64 / total as f64
        } else {
            0.0
        };
        (tombstones, total, ratio)
    }

    /// Force compaction (rebuild index without tombstones).
    pub fn compact(&mut self) -> Result<(), AppError> {
        self.rebuild_without_tombstones()
    }

    /// Rebuild the index without tombstones.
    pub(super) fn rebuild_without_tombstones(&mut self) -> Result<(), AppError> {
        // Remove tombstones from records
        self.records.retain(|r| !r.deleted);
        self.tombstone_count = 0;

        // Handle chunking mode vs non-chunking mode
        match &mut self.indexing_mode {
            IndexingMode::Chunked {
                chunks,
                chunk_store,
                chunk_mapping,
                ..
            } if !chunks.is_empty() => {
                // Chunking mode: rebuild from chunks
                chunks.retain(|c| !c.deleted);

                // Create fake StoredRecords from chunks for index rebuilding
                let chunk_records = chunks_to_fake_records(chunks);

                self.index = self.index_factory.rebuild(&chunk_records);

                // Rebuild chunk_mapping
                *chunk_mapping = build_chunk_mapping(chunks, &self.records);

                // Save chunks
                chunk_store.save_all(chunks)?;
            }
            _ => {
                // Non-chunking mode or empty chunks: rebuild from records
                self.index = self.index_factory.rebuild(&self.records);
            }
        }

        // Rebuild BM25 index (always document-level)
        self.bm25_index = Bm25Index::new();
        for (idx, record) in self.records.iter().enumerate() {
            if let Some(text) = build_text(None, &record.record) {
                self.bm25_index.add(idx, &text);
            }
        }

        self.index.save(&self.snapshot_path)?;
        self.record_store.save_all(&self.records)?;

        // Truncate WAL since we have a fresh snapshot
        if let Some(ref mut wal) = self.wal {
            wal.truncate()?;
        }

        Ok(())
    }

    /// Check if compaction is needed and perform it if so.
    pub(super) fn maybe_compact(&mut self) -> Result<(), AppError> {
        let total = self.records.len();
        if total == 0 {
            return Ok(());
        }

        let should_compact = match self.tombstone_policy {
            TombstonePolicy::Disabled => false,
            TombstonePolicy::MaxCount(max) => self.tombstone_count >= max,
            TombstonePolicy::MaxRatio(max_ratio) => {
                let ratio = self.tombstone_count as f64 / total as f64;
                ratio >= max_ratio
            }
        };

        if should_compact {
            self.rebuild_without_tombstones()?;
        }

        Ok(())
    }

    /// Check if WAL rotation is needed and perform it if so.
    pub(super) fn maybe_rotate_wal(&mut self) -> Result<(), AppError> {
        if let (Some(ref mut wal), Some(ref config)) = (&mut self.wal, &self.wal_config) {
            if wal.should_rotate(config)? {
                // Save snapshot first, then truncate WAL
                self.index.save(&self.snapshot_path)?;
                self.record_store.save_all(&self.records)?;
                wal.truncate()?;
            }
        }
        Ok(())
    }
}
