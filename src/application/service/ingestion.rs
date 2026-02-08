//! Batch ingestion and differential sync operations.
//!
//! This module implements Bedrock-style batch document ingestion with differential sync:
//! - `batch_insert`: Add documents to pending_queue (NOT to records/index)
//! - `sync_pending`: Process pending documents, embedding and adding to records/index
//!
//! Key invariant: vector_id == record_id is maintained because pending documents
//! are only added to records/index together during sync.
//!
//! **Note**: Batch ingestion is NOT supported when chunking is enabled.

use crate::application::error::AppError;
use crate::domain::model::{PendingDocument, StoredRecord};

use super::core::DocumentService;
use super::types::{BatchDocument, BatchError, BatchInsertResult, IndexingMode, JobProgress};
use super::util::{build_text, calculate_content_hash};

/// Result of processing a single pending document during sync
#[derive(Debug)]
pub enum ProcessResult {
    /// Document was embedded (new content or content changed)
    Indexed,
    /// Document was skipped (content unchanged, embedding copied)
    Skipped,
    /// Document processing failed
    Failed(String),
}

/// Trait for document ingestion operations
pub trait DocumentIngestion {
    /// Batch insert documents to pending queue (not to records/index).
    /// Returns `Err(AppError::BadRequest)` if chunking is enabled.
    fn batch_insert(&mut self, documents: Vec<BatchDocument>) -> Result<BatchInsertResult, AppError>;

    /// Get list of pending document IDs
    fn get_pending_ids(&self) -> Vec<String>;

    /// Process a single pending document during sync
    fn process_pending_document(&mut self, id: &str) -> ProcessResult;

    /// Check if chunking is enabled (ingestion not supported)
    fn is_chunking_enabled(&self) -> bool;
}

impl DocumentIngestion for DocumentService {
    fn batch_insert(&mut self, documents: Vec<BatchDocument>) -> Result<BatchInsertResult, AppError> {
        // Reject if chunking is enabled
        if self.is_chunking_enabled() {
            return Err(AppError::BadRequest(
                "batch ingestion is not supported when chunking is enabled".to_string(),
            ));
        }

        let mut accepted = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        for (idx, doc) in documents.into_iter().enumerate() {
            // 1. Validate ID (non-empty after trim)
            let id = doc.id.trim().to_string();
            if id.is_empty() {
                errors.push(BatchError {
                    id: format!("[index {}]", idx),
                    error: "id is required".to_string(),
                });
                failed += 1;
                continue;
            }

            // 2. Build record
            let record = crate::domain::model::Record {
                id: id.clone(),
                title: doc.title,
                body: doc.body,
                source: doc.source,
                updated_at: doc.updated_at,
                tags: doc.tags,
            };

            // 3. Normalize explicit text (trim, None if empty)
            let embed_text = doc.text.and_then(|t| {
                let trimmed = t.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            });

            // 4. Calculate content_hash from normalized fields
            let content_hash = calculate_content_hash(&record, embed_text.as_deref());

            // 5. Create pending document
            let pending_doc = PendingDocument {
                record,
                content_hash,
                embed_text,
            };

            // 6. Persist to pending_store if available
            if let Some(ref ps) = self.pending_store {
                if let Err(e) = ps.upsert(&id, &pending_doc) {
                    errors.push(BatchError {
                        id: id.clone(),
                        error: format!("failed to persist pending: {}", e),
                    });
                    failed += 1;
                    continue;
                }
            }

            // 7. Add to pending_queue (overwrites any existing pending for same ID)
            // Note: We don't touch records/WAL/index here - that happens during sync
            self.pending_queue.insert(id, pending_doc);

            accepted += 1;
        }

        Ok(BatchInsertResult {
            accepted,
            failed,
            errors,
        })
    }

    fn get_pending_ids(&self) -> Vec<String> {
        self.pending_queue.keys().cloned().collect()
    }

    fn is_chunking_enabled(&self) -> bool {
        matches!(self.indexing_mode, IndexingMode::Chunked { .. })
    }

    fn process_pending_document(&mut self, id: &str) -> ProcessResult {
        // 1. Take pending document from queue (will restore on failure)
        let pending = match self.pending_queue.remove(id) {
            Some(p) => p,
            None => return ProcessResult::Failed("pending document not found".to_string()),
        };

        // 2. Find currently-indexed record with same ID (!deleted)
        let old_idx = self
            .records
            .iter()
            .position(|r| r.record.id == id && !r.deleted);

        // 3. Determine text for embedding
        // Use embed_text if available, otherwise derive from title+body
        let text_for_embed = pending
            .embed_text
            .clone()
            .or_else(|| build_text(None, &pending.record));

        // 4. Determine if we can skip embedding (content unchanged)
        let should_skip = if let Some(old_idx) = old_idx {
            let old = &self.records[old_idx];
            // Skip if:
            // - Old has content_hash matching pending's hash
            // - Old has a non-empty embedding
            old.content_hash.as_ref() == Some(&pending.content_hash)
                && !old.embedding.is_empty()
        } else {
            false
        };

        if should_skip {
            // CASE A: content_hash matches (SKIP embedding)
            let old_idx = old_idx.unwrap();

            // Check if this is a "cleanup only" case - old record is already indexed
            // No need to create new record, just delete from pending_store
            if self.records[old_idx].indexed {
                if let Some(ref ps) = self.pending_store {
                    let mut last_error = None;
                    for _ in 0..3 {
                        match ps.delete(id) {
                            Ok(()) => {
                                last_error = None;
                                break;
                            }
                            Err(e) => {
                                last_error = Some(e);
                            }
                        }
                    }
                    if let Some(e) = last_error {
                        // Restore to pending_queue for retry in same session
                        self.pending_queue.insert(id.to_string(), pending);
                        return ProcessResult::Failed(format!(
                            "pending_store cleanup failed after 3 attempts: {}",
                            e
                        ));
                    }
                }
                // Cleanup only succeeded - no indexing work needed
                return ProcessResult::Skipped;
            }

            // Normal skip path - old exists but needs to be replaced
            // Copy embedding from old record
            let old_embedding = self.records[old_idx].embedding.clone();
            let old_embed_text = self.records[old_idx].embed_text.clone();

            // Create new record with copied embedding
            // Clone pending data before consuming for potential restore
            let pending_for_restore = PendingDocument {
                record: pending.record.clone(),
                content_hash: pending.content_hash.clone(),
                embed_text: pending.embed_text.clone(),
            };
            let mut new_record = StoredRecord::new(pending.record, old_embedding);
            new_record.content_hash = Some(pending.content_hash);
            new_record.embed_text = old_embed_text;

            // Add new record to vector index FIRST (maintains vector_id == record_id)
            let new_record_idx = self.records.len();
            self.index.insert(new_record.embedding.clone());

            // Add to BM25 index
            if let Some(ref text) = text_for_embed {
                self.bm25_index.add(new_record_idx, text);
            }

            // Tombstone old record
            self.bm25_index.remove(old_idx);
            self.records[old_idx].deleted = true;
            self.tombstone_count += 1;

            // Write to WAL
            if let Some(ref mut wal) = self.wal {
                let _ = wal.append_delete(id);
                let _ = wal.append_append(&new_record);
            }

            // Add new record to records
            self.records.push(new_record);

            // Persist - restore to pending_queue on failure
            if let Err(e) = self.index.save(&self.snapshot_path) {
                self.pending_queue.insert(id.to_string(), pending_for_restore);
                return ProcessResult::Failed(format!("index save failed: {}", e));
            }

            if let Err(e) = self.record_store.save_all(&self.records) {
                self.pending_queue.insert(id.to_string(), pending_for_restore);
                return ProcessResult::Failed(format!("storage save failed: {}", e));
            }

            // Delete from pending_store on success - retry on failure
            if let Some(ref ps) = self.pending_store {
                let mut last_error = None;
                for _ in 0..3 {
                    match ps.delete(id) {
                        Ok(()) => {
                            last_error = None;
                            break;
                        }
                        Err(e) => {
                            last_error = Some(e);
                        }
                    }
                }
                if let Some(e) = last_error {
                    // Restore to pending_queue for retry
                    self.pending_queue.insert(id.to_string(), pending_for_restore);
                    return ProcessResult::Failed(format!(
                        "indexed but pending_store cleanup failed after 3 attempts: {}",
                        e
                    ));
                }
            }

            ProcessResult::Skipped
        } else {
            // CASE B: content_hash differs or old doesn't exist (EMBED)
            let text_to_embed = match text_for_embed {
                Some(t) if !t.trim().is_empty() => t,
                _ => {
                    // No text to embed - create record with empty embedding
                    // Document won't appear in vector search but is "indexed"
                    // Clone pending data before consuming for potential restore
                    let pending_for_restore = PendingDocument {
                        record: pending.record.clone(),
                        content_hash: pending.content_hash.clone(),
                        embed_text: pending.embed_text.clone(),
                    };
                    let mut new_record = StoredRecord::new(pending.record, Vec::new());
                    new_record.content_hash = Some(pending.content_hash);
                    new_record.embed_text = pending.embed_text;

                    // Tombstone old if exists
                    if let Some(old_idx) = old_idx {
                        self.bm25_index.remove(old_idx);
                        self.records[old_idx].deleted = true;
                        self.tombstone_count += 1;
                    }

                    // Write to WAL
                    if let Some(ref mut wal) = self.wal {
                        if old_idx.is_some() {
                            let _ = wal.append_delete(id);
                        }
                        let _ = wal.append_append(&new_record);
                    }

                    // Add to records (no index entry for empty embedding)
                    self.records.push(new_record);

                    if let Err(e) = self.record_store.save_all(&self.records) {
                        self.pending_queue.insert(id.to_string(), pending_for_restore);
                        return ProcessResult::Failed(format!("storage save failed: {}", e));
                    }

                    // Delete from pending_store on success - retry on failure
                    if let Some(ref ps) = self.pending_store {
                        let mut last_error = None;
                        for _ in 0..3 {
                            match ps.delete(id) {
                                Ok(()) => {
                                    last_error = None;
                                    break;
                                }
                                Err(e) => {
                                    last_error = Some(e);
                                }
                            }
                        }
                        if let Some(e) = last_error {
                            // Restore to pending_queue for retry
                            self.pending_queue.insert(id.to_string(), pending_for_restore);
                            return ProcessResult::Failed(format!(
                                "indexed but pending_store cleanup failed after 3 attempts: {}",
                                e
                            ));
                        }
                    }

                    return ProcessResult::Indexed;
                }
            };

            // Embed text
            let embedding = match self.embed_single(text_to_embed.clone()) {
                Ok(e) => e,
                Err(e) => {
                    // Put back in pending queue on failure
                    self.pending_queue.insert(id.to_string(), pending);
                    return ProcessResult::Failed(format!("embedding failed: {}", e));
                }
            };

            // Clone pending data before consuming for potential restore
            let pending_for_restore = PendingDocument {
                record: pending.record.clone(),
                content_hash: pending.content_hash.clone(),
                embed_text: pending.embed_text.clone(),
            };

            // Create new record with embedding
            let mut new_record = StoredRecord::new(pending.record, embedding.clone());
            new_record.content_hash = Some(pending.content_hash);
            new_record.embed_text = pending.embed_text;

            // Add to vector index FIRST (maintains vector_id == record_id)
            let new_record_idx = self.records.len();
            self.index.insert(embedding);

            // Add to BM25 index
            self.bm25_index.add(new_record_idx, &text_to_embed);

            // Tombstone old if exists
            if let Some(old_idx) = old_idx {
                self.bm25_index.remove(old_idx);
                self.records[old_idx].deleted = true;
                self.tombstone_count += 1;
            }

            // Write to WAL
            if let Some(ref mut wal) = self.wal {
                if old_idx.is_some() {
                    let _ = wal.append_delete(id);
                }
                let _ = wal.append_append(&new_record);
            }

            // Add new record to records
            self.records.push(new_record);

            // Persist - restore to pending_queue on failure
            if let Err(e) = self.index.save(&self.snapshot_path) {
                self.pending_queue.insert(id.to_string(), pending_for_restore);
                return ProcessResult::Failed(format!("index save failed: {}", e));
            }

            if let Err(e) = self.record_store.save_all(&self.records) {
                self.pending_queue.insert(id.to_string(), pending_for_restore);
                return ProcessResult::Failed(format!("storage save failed: {}", e));
            }

            // Delete from pending_store on success - retry on failure
            if let Some(ref ps) = self.pending_store {
                let mut last_error = None;
                for _ in 0..3 {
                    match ps.delete(id) {
                        Ok(()) => {
                            last_error = None;
                            break;
                        }
                        Err(e) => {
                            last_error = Some(e);
                        }
                    }
                }
                if let Some(e) = last_error {
                    // Restore to pending_queue for retry
                    self.pending_queue.insert(id.to_string(), pending_for_restore);
                    return ProcessResult::Failed(format!(
                        "indexed but pending_store cleanup failed after 3 attempts: {}",
                        e
                    ));
                }
            }

            ProcessResult::Indexed
        }
    }
}

/// Run the sync job, processing all pending documents.
/// Returns `Err(AppError::BadRequest)` if chunking is enabled.
pub fn run_sync(
    service: &mut DocumentService,
    progress_callback: impl Fn(&JobProgress),
) -> Result<JobProgress, AppError> {
    // Reject if chunking is enabled
    if service.is_chunking_enabled() {
        return Err(AppError::BadRequest(
            "sync is not supported when chunking is enabled".to_string(),
        ));
    }

    let pending_ids: Vec<String> = service.pending_queue.keys().cloned().collect();
    let total = pending_ids.len();

    let mut progress = JobProgress {
        total,
        processed: 0,
        indexed: 0,
        skipped: 0,
        failed: 0,
    };

    progress_callback(&progress);

    for id in pending_ids {
        let result = service.process_pending_document(&id);
        progress.processed += 1;

        match result {
            ProcessResult::Indexed => progress.indexed += 1,
            ProcessResult::Skipped => progress.skipped += 1,
            ProcessResult::Failed(_) => progress.failed += 1,
        }

        progress_callback(&progress);
    }

    // Check if compaction is needed after sync
    service.maybe_compact()?;

    // Check if WAL rotation is needed
    service.maybe_rotate_wal()?;

    Ok(progress)
}
