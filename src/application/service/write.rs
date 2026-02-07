//! Write operations (insert, update, delete) for DocumentService.

use crate::application::error::AppError;
use crate::domain::model::{Record, StoredChunk, StoredRecord};

use super::core::DocumentService;
use super::traits::DocumentWrite;
use super::types::{
    DocumentResponse, IndexingMode, InsertCommand, InsertResult, UpdateCommand, UpdateEffect,
};
use super::util::build_text;

impl DocumentWrite for DocumentService {
    fn insert(&mut self, cmd: InsertCommand) -> Result<InsertResult, AppError> {
        if cmd.record.id.trim().is_empty() {
            return Err(AppError::BadRequest("id is required".to_string()));
        }
        // Check for existing non-deleted record with same ID
        if self
            .records
            .iter()
            .any(|r| r.record.id == cmd.record.id && !r.deleted)
        {
            return Err(AppError::Conflict("id already exists".to_string()));
        }

        let text = build_text(cmd.text.as_deref(), &cmd.record)
            .ok_or_else(|| AppError::BadRequest("text/title/body is required".to_string()))?;

        match &mut self.indexing_mode {
            IndexingMode::Chunked {
                chunker,
                chunk_store,
                chunks,
                chunk_mapping,
                chunk_index,
            } => {
                // Chunking mode: split text into chunks and index each chunk
                let text_chunks = chunker.chunk(&text);
                if text_chunks.is_empty() {
                    return Err(AppError::BadRequest("text produced no chunks".to_string()));
                }

                // Get embeddings for all chunks
                let chunk_texts: Vec<String> = text_chunks.iter().map(|c| c.text.clone()).collect();
                let embeddings = self.embedder.embed(chunk_texts)?;

                // Store the document record with the first chunk's embedding (for compatibility)
                let doc_embedding = embeddings.first().cloned().unwrap_or_default();
                let stored = StoredRecord::new(cmd.record.clone(), doc_embedding.clone());

                // Write to WAL first
                if let Some(ref mut wal) = self.wal {
                    wal.append_insert(&stored)?;
                }

                // Add record to records list (but don't add to vector index - chunks go there)
                let record_idx = self.records.len();
                self.records.push(stored.clone());

                // Create and store chunks, add to vector index
                let mut first_index_id = None;
                for (chunk, embedding) in text_chunks.into_iter().zip(embeddings.into_iter()) {
                    let stored_chunk =
                        StoredChunk::new(cmd.record.id.clone(), chunk.clone(), embedding.clone());

                    // Add chunk embedding to vector index
                    let chunk_index_id = self.index.insert(embedding);
                    if first_index_id.is_none() {
                        first_index_id = Some(chunk_index_id);
                    }

                    // Update chunk mapping
                    chunk_mapping.push((record_idx, chunk.chunk_index));

                    // Add to O(1) chunk index
                    let chunk_pos = chunks.len();
                    chunk_index.insert((cmd.record.id.clone(), chunk.chunk_index), chunk_pos);

                    // Store chunk
                    chunks.push(stored_chunk.clone());
                    chunk_store.append(&stored_chunk)?;
                }

                // Update BM25 index with full document text (not chunks)
                self.bm25_index.add(record_idx, &text);

                self.index.save(&self.snapshot_path)?;
                self.record_store.append(&stored)?;

                // Check if WAL rotation is needed
                self.maybe_rotate_wal()?;

                Ok(InsertResult {
                    index_id: first_index_id.unwrap_or(0),
                    id: cmd.record.id,
                    dim: doc_embedding.len(),
                })
            }
            IndexingMode::Direct => {
                // Non-chunking mode: original behavior
                let embedding = self.embed_single(text.clone())?;
                let stored = StoredRecord::new(cmd.record.clone(), embedding.clone());

                // Write to WAL first
                if let Some(ref mut wal) = self.wal {
                    wal.append_insert(&stored)?;
                }

                // Then update index and records
                let index_id = self.index.insert(embedding.clone());
                self.records.push(stored.clone());

                // Update BM25 index
                self.bm25_index.add(index_id, &text);

                self.index.save(&self.snapshot_path)?;
                self.record_store.append(&stored)?;

                // Check if WAL rotation is needed
                self.maybe_rotate_wal()?;

                Ok(InsertResult {
                    index_id,
                    id: cmd.record.id,
                    dim: embedding.len(),
                })
            }
        }
    }

    fn update(&mut self, id: &str, cmd: UpdateCommand) -> Result<DocumentResponse, AppError> {
        let index_id = self.find_index(id)?;
        let current = self.records[index_id].clone();

        // Determine if content changed (requires re-embedding) or only metadata changed
        let content_changed = cmd.text.is_some() || cmd.title.is_some() || cmd.body.is_some();
        let updated = Record {
            id: current.record.id.clone(),
            title: cmd.title.or(current.record.title),
            body: cmd.body.or(current.record.body),
            source: cmd.source.or(current.record.source),
            updated_at: cmd.updated_at.or(current.record.updated_at),
            tags: cmd.tags.unwrap_or(current.record.tags),
        };

        // Determine update effect
        let effect = if content_changed {
            let text = cmd
                .text
                .or_else(|| build_text(None, &updated))
                .ok_or_else(|| AppError::BadRequest("text/title/body is required".to_string()))?;
            if text.trim().is_empty() {
                return Err(AppError::BadRequest("text must not be empty".to_string()));
            }
            UpdateEffect::ContentChanged { text }
        } else {
            UpdateEffect::MetadataOnly
        };

        match effect {
            UpdateEffect::ContentChanged { text } => {
                // Embedding changed: tombstone old record/chunks, append new

                match &mut self.indexing_mode {
                    IndexingMode::Chunked {
                        chunker,
                        chunk_store,
                        chunks,
                        chunk_mapping,
                        chunk_index,
                    } => {
                        // Chunking mode: re-chunk and re-index

                        // Mark old record as deleted
                        self.records[index_id].deleted = true;
                        self.tombstone_count += 1;

                        // Remove old chunks from O(1) index
                        chunk_index.retain(|(pid, _), _| pid != id);

                        // Mark all old chunks for this document as deleted
                        for chunk in chunks.iter_mut() {
                            if chunk.parent_id == id {
                                chunk.deleted = true;
                            }
                        }

                        // Update BM25: remove old
                        self.bm25_index.remove(index_id);

                        // Generate new chunks
                        let text_chunks = chunker.chunk(&text);
                        if text_chunks.is_empty() {
                            return Err(AppError::BadRequest(
                                "text produced no chunks".to_string(),
                            ));
                        }

                        // Get embeddings for all new chunks
                        let chunk_texts: Vec<String> =
                            text_chunks.iter().map(|c| c.text.clone()).collect();
                        let embeddings = self.embedder.embed(chunk_texts)?;

                        // Create new record
                        let doc_embedding = embeddings.first().cloned().unwrap_or_default();
                        let new_stored = StoredRecord::new(updated.clone(), doc_embedding.clone());

                        // Write to WAL
                        if let Some(ref mut wal) = self.wal {
                            wal.append_delete(&current.record.id)?;
                            wal.append_append(&new_stored)?;
                        }

                        // Add new record
                        let new_record_idx = self.records.len();
                        self.records.push(new_stored.clone());

                        // Create and store new chunks, add to vector index
                        let mut first_index_id = None;
                        for (chunk, emb) in text_chunks.into_iter().zip(embeddings.into_iter()) {
                            let stored_chunk =
                                StoredChunk::new(updated.id.clone(), chunk.clone(), emb.clone());

                            // Add chunk embedding to vector index
                            let chunk_index_id = self.index.insert(emb);
                            if first_index_id.is_none() {
                                first_index_id = Some(chunk_index_id);
                            }

                            // Update chunk mapping
                            chunk_mapping.push((new_record_idx, chunk.chunk_index));

                            // Add to O(1) chunk index
                            let chunk_pos = chunks.len();
                            chunk_index.insert((updated.id.clone(), chunk.chunk_index), chunk_pos);

                            // Store chunk
                            chunks.push(stored_chunk);
                        }

                        // Add to BM25 index
                        self.bm25_index.add(new_record_idx, &text);

                        // Save state
                        self.index.save(&self.snapshot_path)?;
                        self.record_store.save_all(&self.records)?;
                        chunk_store.save_all(chunks)?;

                        // Check if compaction is needed
                        self.maybe_compact()?;

                        // Check if WAL rotation is needed
                        self.maybe_rotate_wal()?;

                        Ok(DocumentResponse {
                            index_id: first_index_id.unwrap_or(0),
                            record: updated,
                            embedding: doc_embedding,
                        })
                    }
                    IndexingMode::Direct => {
                        // Non-chunking mode: original behavior
                        let embedding = self.embed_single(text)?;
                        let new_stored = StoredRecord::new(updated.clone(), embedding.clone());

                        // Write to WAL first: DELETE old, then APPEND new (idempotent replay)
                        if let Some(ref mut wal) = self.wal {
                            wal.append_delete(&current.record.id)?;
                            wal.append_append(&new_stored)?;
                        }

                        // Mark old record as deleted (tombstone)
                        self.records[index_id].deleted = true;
                        self.tombstone_count += 1;

                        // Update BM25: remove old, add new
                        self.bm25_index.remove(index_id);

                        // Append new record and add to index
                        let new_index_id = self.index.insert(embedding.clone());
                        self.records.push(new_stored.clone());

                        // Add to BM25 index
                        if let Some(bm25_text) = build_text(None, &updated) {
                            self.bm25_index.add(new_index_id, &bm25_text);
                        }

                        // Save state
                        self.index.save(&self.snapshot_path)?;
                        self.record_store.save_all(&self.records)?;

                        // Check if compaction is needed
                        self.maybe_compact()?;

                        // Check if WAL rotation is needed
                        self.maybe_rotate_wal()?;

                        Ok(DocumentResponse {
                            index_id: new_index_id,
                            record: updated,
                            embedding,
                        })
                    }
                }
            }
            UpdateEffect::MetadataOnly => {
                // Only metadata changed: update in place (no index change needed)
                let stored = StoredRecord::new(updated.clone(), current.embedding.clone());

                // Write to WAL first
                if let Some(ref mut wal) = self.wal {
                    wal.append_update(&stored)?;
                }

                // Update record in place
                self.records[index_id] = stored;

                // Update BM25 index
                if let Some(bm25_text) = build_text(None, &updated) {
                    self.bm25_index.update(index_id, &bm25_text);
                }

                self.record_store.save_all(&self.records)?;

                // Check if WAL rotation is needed
                self.maybe_rotate_wal()?;

                Ok(DocumentResponse {
                    index_id,
                    record: updated,
                    embedding: self.records[index_id].embedding.clone(),
                })
            }
        }
    }

    fn delete(&mut self, id: &str) -> Result<(), AppError> {
        let index_id = self.find_index(id)?;

        // Write to WAL first
        if let Some(ref mut wal) = self.wal {
            wal.append_delete(id)?;
        }

        // Mark as tombstone (soft delete)
        self.records[index_id].deleted = true;
        self.tombstone_count += 1;

        // Remove from BM25 index
        self.bm25_index.remove(index_id);

        // Mark all chunks for this document as deleted (if chunking enabled)
        if let IndexingMode::Chunked {
            chunks,
            chunk_store,
            chunk_index,
            ..
        } = &mut self.indexing_mode
        {
            // Remove from O(1) chunk index
            chunk_index.retain(|(pid, _), _| pid != id);

            for chunk in chunks.iter_mut() {
                if chunk.parent_id == id {
                    chunk.deleted = true;
                }
            }
            // Save updated chunks
            chunk_store.save_all(chunks)?;
        }

        self.record_store.save_all(&self.records)?;

        // Check if compaction is needed
        self.maybe_compact()?;

        // Check if WAL rotation is needed
        self.maybe_rotate_wal()?;

        Ok(())
    }

    fn import_embeddings(&mut self, records: Vec<StoredRecord>) -> Result<usize, AppError> {
        self.import_embeddings_impl(records)
    }
}
