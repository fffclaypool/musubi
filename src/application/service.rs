use crate::application::error::AppError;
use crate::domain::model::{Record, StoredChunk, StoredRecord};
use crate::domain::ports::{ChunkStore, Chunker, Embedder, RecordStore, VectorIndex, VectorIndexFactory};
use crate::infrastructure::search::Bm25Index;
use crate::infrastructure::storage::wal::{self, WalConfig, WalWriter};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Configuration for tombstone-based compaction
#[derive(Debug, Clone)]
pub struct TombstoneConfig {
    /// Maximum number of tombstones before triggering compaction
    pub max_tombstones: Option<usize>,
    /// Maximum ratio of tombstones to total records before triggering compaction (0.0 - 1.0)
    pub max_tombstone_ratio: Option<f64>,
}

impl Default for TombstoneConfig {
    fn default() -> Self {
        Self {
            max_tombstones: None,
            max_tombstone_ratio: Some(0.3), // Default: compact when 30% are tombstones
        }
    }
}

/// Configuration for document chunking
#[derive(Debug, Clone, Default)]
pub enum ChunkConfig {
    /// No chunking - treat each document as a single unit (default, backward compatible)
    #[default]
    None,
    /// Fixed-size chunking with overlap
    Fixed {
        /// Target chunk size in characters
        chunk_size: usize,
        /// Number of characters to overlap between chunks
        overlap: usize,
    },
    /// Semantic chunking based on sentence similarity
    Semantic {
        /// Minimum chunk size in characters
        min_chunk_size: usize,
        /// Maximum chunk size in characters
        max_chunk_size: usize,
        /// Similarity threshold (0.0-1.0) - split when similarity drops below this
        similarity_threshold: f32,
    },
}

pub struct ServiceConfig {
    pub snapshot_path: PathBuf,
    pub default_k: usize,
    pub default_ef: usize,
    pub wal_config: Option<WalConfig>,
    pub tombstone_config: TombstoneConfig,
    pub chunk_config: ChunkConfig,
}

pub struct DocumentService {
    index: Box<dyn VectorIndex>,
    index_factory: Box<dyn VectorIndexFactory>,
    records: Vec<StoredRecord>,
    embedder: Box<dyn Embedder>,
    record_store: Box<dyn RecordStore>,
    snapshot_path: PathBuf,
    default_k: usize,
    default_ef: usize,
    wal: Option<WalWriter>,
    wal_config: Option<WalConfig>,
    tombstone_config: TombstoneConfig,
    tombstone_count: usize,
    bm25_index: Bm25Index,
    // Chunking support
    chunks: Vec<StoredChunk>,
    /// Maps index ID -> (record_index, chunk_index within that record)
    chunk_mapping: Vec<(usize, usize)>,
    chunk_store: Option<Box<dyn ChunkStore>>,
    chunker: Option<Box<dyn Chunker>>,
    #[allow(dead_code)] // Reserved for future dynamic configuration
    chunk_config: ChunkConfig,
}

#[derive(Debug, Clone)]
pub struct InsertCommand {
    pub record: Record,
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UpdateCommand {
    pub title: Option<String>,
    pub body: Option<String>,
    pub source: Option<String>,
    pub updated_at: Option<String>,
    pub tags: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilter {
    /// Exact match on source field
    pub source: Option<String>,
    /// Match if any of these tags are present (comma-separated tags)
    pub tags_any: Option<Vec<String>>,
    /// Match only if all of these tags are present (comma-separated tags)
    pub tags_all: Option<Vec<String>>,
    /// Match if updated_at >= this value (string comparison, YYYY-MM-DD)
    pub updated_at_gte: Option<String>,
    /// Match if updated_at <= this value (string comparison, YYYY-MM-DD)
    pub updated_at_lte: Option<String>,
}

impl SearchFilter {
    /// Check if a record matches all filter criteria
    pub fn matches(&self, record: &Record) -> bool {
        // source: exact match
        if let Some(ref filter_source) = self.source {
            match &record.source {
                Some(record_source) if record_source == filter_source => {}
                _ => return false,
            }
        }

        // tags_any: at least one tag matches
        if let Some(ref filter_tags) = self.tags_any {
            if !filter_tags.is_empty() {
                let record_tags = parse_tags(record.tags.as_deref());
                let has_any = filter_tags
                    .iter()
                    .any(|t| record_tags.contains(&t.trim().to_lowercase()));
                if !has_any {
                    return false;
                }
            }
        }

        // tags_all: all tags must be present
        if let Some(ref filter_tags) = self.tags_all {
            if !filter_tags.is_empty() {
                let record_tags = parse_tags(record.tags.as_deref());
                let has_all = filter_tags
                    .iter()
                    .all(|t| record_tags.contains(&t.trim().to_lowercase()));
                if !has_all {
                    return false;
                }
            }
        }

        // updated_at_gte: string comparison
        if let Some(ref gte) = self.updated_at_gte {
            match &record.updated_at {
                Some(updated_at) if updated_at.as_str() >= gte.as_str() => {}
                _ => return false,
            }
        }

        // updated_at_lte: string comparison
        if let Some(ref lte) = self.updated_at_lte {
            match &record.updated_at {
                Some(updated_at) if updated_at.as_str() <= lte.as_str() => {}
                _ => return false,
            }
        }

        true
    }
}

/// Parse comma-separated tags into a set of lowercase trimmed strings
fn parse_tags(tags: Option<&str>) -> HashSet<String> {
    tags.map(|t| {
        t.split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
    .unwrap_or_default()
}

#[derive(Debug, Clone)]
pub struct SearchCommand {
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub k: Option<usize>,
    pub ef: Option<usize>,
    /// Weight for vector score in hybrid search (0.0 = BM25 only, 1.0 = vector only)
    /// Default is 0.7
    pub alpha: Option<f64>,
    /// Optional filter to narrow down search results
    pub filter: Option<SearchFilter>,
}

/// Information about the best matching chunk within a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Index of the chunk within the parent document
    pub chunk_index: usize,
    /// Preview of the chunk text (first ~100 chars)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub index_id: usize,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid_score: Option<f64>,
    pub title: Option<String>,
    pub source: Option<String>,
    pub tags: Option<String>,
    /// Best matching chunk info (only present when chunking is enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_chunk: Option<ChunkInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InsertResult {
    pub index_id: usize,
    pub id: String,
    pub dim: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentSummary {
    pub index_id: usize,
    #[serde(flatten)]
    pub record: Record,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentResponse {
    pub index_id: usize,
    #[serde(flatten)]
    pub record: Record,
    pub embedding: Vec<f32>,
}

impl DocumentService {
    pub fn load(
        config: ServiceConfig,
        embedder: Box<dyn Embedder>,
        record_store: Box<dyn RecordStore>,
        index_factory: Box<dyn VectorIndexFactory>,
        chunker: Option<Box<dyn Chunker>>,
        chunk_store: Option<Box<dyn ChunkStore>>,
    ) -> Result<Self, AppError> {
        // Validate: chunk_store and chunker must be provided together
        match (&chunker, &chunk_store) {
            (Some(_), None) => {
                return Err(AppError::Io(
                    "chunker requires chunk_store to be provided".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(AppError::Io(
                    "chunk_store requires chunker to be provided".to_string(),
                ));
            }
            _ => {} // Both Some or both None - OK
        }

        // Load records from record store
        let mut records = record_store.load()?;

        // Replay WAL if configured
        let (wal, wal_had_ops) = if let Some(ref wal_config) = config.wal_config {
            let ops = wal::replay(&wal_config.path)?;
            let had_ops = !ops.is_empty();
            if had_ops {
                wal::apply_ops_to_records(ops, &mut records);
                // Save merged records to record store
                record_store.save_all(&records)?;
            }
            (Some(WalWriter::new(&wal_config.path)?), had_ops)
        } else {
            (None, false)
        };

        // Count tombstones
        let tombstone_count = records.iter().filter(|r| r.deleted).count();

        // Load chunks first (if chunk store is provided) to determine chunking mode
        let (mut chunks, chunking_enabled) = if let Some(ref cs) = chunk_store {
            let loaded_chunks = cs.load()?;
            let has_chunks = !loaded_chunks.is_empty();
            (loaded_chunks, has_chunks || chunker.is_some())
        } else {
            (Vec::new(), false)
        };

        // Critical: Handle migration case where chunking is enabled but no chunks exist
        // When chunker is set but chunks is empty, we need to migrate existing records
        let needs_migration = chunking_enabled
            && chunks.is_empty()
            && !records.is_empty()
            && chunker.is_some();

        // Critical: Handle WAL recovery for chunking mode
        // WAL only tracks record operations, not chunks. After WAL replay in chunking mode,
        // we must regenerate chunks from the recovered record state to ensure consistency.
        let needs_chunk_rebuild = wal_had_ops && chunking_enabled && chunker.is_some();

        if needs_migration {
            println!("Migrating existing records to chunking mode...");
            // Migration will be done after embedder is available (handled below)
        } else if needs_chunk_rebuild {
            println!("WAL replay detected in chunking mode, will regenerate chunks...");
            // Clear stale chunks - they'll be regenerated from recovered records
            chunks.clear();
        }

        // Build or load index based on chunking mode
        // Note: If needs_migration or needs_chunk_rebuild, we'll rebuild after migration
        let index = if chunking_enabled && !chunks.is_empty() {
            // Chunking mode: index is built from chunks
            let chunk_records = chunks_to_fake_records(&chunks);
            if config.snapshot_path.exists() {
                index_factory.load_or_create(&config.snapshot_path, &chunk_records)?
            } else {
                index_factory.rebuild(&chunk_records)
            }
        } else if needs_migration || needs_chunk_rebuild {
            // Migration/rebuild mode: start with empty index, will rebuild after migration
            index_factory.rebuild(&[])
        } else {
            // Non-chunking mode: index is built from records
            if wal_had_ops {
                index_factory.rebuild(&records)
            } else if config.snapshot_path.exists() {
                index_factory.load_or_create(&config.snapshot_path, &records)?
            } else {
                index_factory.rebuild(&records)
            }
        };

        // Fill missing embeddings (only relevant for non-chunking mode)
        if !chunking_enabled {
            let updated = fill_missing_embeddings(&mut records, index.as_ref());
            if updated {
                let _ = record_store.save_all(&records);
            }
        }

        // Validate index count (skip for migration/rebuild mode - will rebuild after migration)
        if !needs_migration && !needs_chunk_rebuild {
            if chunking_enabled && !chunks.is_empty() {
                // Chunking mode: check chunks count (including tombstones - they have vectors)
                if chunks.len() != index.len() {
                    return Err(AppError::Io(format!(
                        "chunks count ({}) does not match index count ({})",
                        chunks.len(),
                        index.len()
                    )));
                }
            } else if !records.is_empty() && records.len() != index.len() {
                // Non-chunking mode: check records count
                return Err(AppError::Io(format!(
                    "records count ({}) does not match index count ({})",
                    records.len(),
                    index.len()
                )));
            }
        }

        // Build BM25 index from records (excluding tombstones, always document-level)
        let mut bm25_index = Bm25Index::new();
        for (idx, record) in records.iter().enumerate() {
            if !record.deleted {
                if let Some(text) = build_text(None, &record.record) {
                    bm25_index.add(idx, &text);
                }
            }
        }

        // Build chunk mapping (includes ALL chunks to maintain index alignment)
        let chunk_mapping = build_chunk_mapping(&chunks, &records);

        // After successful load, save snapshot and truncate WAL
        let mut service = Self {
            index,
            index_factory,
            records,
            embedder,
            record_store,
            snapshot_path: config.snapshot_path,
            default_k: config.default_k,
            default_ef: config.default_ef,
            wal,
            wal_config: config.wal_config,
            tombstone_config: config.tombstone_config,
            tombstone_count,
            bm25_index,
            chunks,
            chunk_mapping,
            chunk_store,
            chunker,
            chunk_config: config.chunk_config,
        };

        // Perform migration if needed (chunking enabled but no chunks exist)
        // or rebuild chunks after WAL replay
        if needs_migration || needs_chunk_rebuild {
            service.migrate_to_chunking()?;
            // WAL truncation happens inside migrate_to_chunking on success
        } else if wal_had_ops {
            // Non-migration case: truncate WAL after successful replay
            // Only truncate if we successfully replayed v2 operations
            // (v1 WAL can't be replayed, so wal_had_ops would be false)
            service.index.save(&service.snapshot_path)?;
            if let Some(ref mut wal) = service.wal {
                wal.truncate()?;
            }
        }

        Ok(service)
    }

    /// Migrate existing records to chunking mode by re-chunking all documents
    fn migrate_to_chunking(&mut self) -> Result<(), AppError> {
        let chunker = match &self.chunker {
            Some(c) => c,
            None => return Ok(()), // No chunker, nothing to do
        };

        println!("Starting migration: {} records to chunk...", self.records.len());

        // Collect all texts that need to be chunked
        let mut all_chunks_data: Vec<(usize, Vec<crate::domain::model::Chunk>)> = Vec::new();
        let mut all_chunk_texts: Vec<String> = Vec::new();
        let mut chunk_to_record: Vec<(usize, usize)> = Vec::new(); // (chunk_idx_in_all, record_idx)

        for (record_idx, record) in self.records.iter().enumerate() {
            if record.deleted {
                continue;
            }

            let text = match build_text(None, &record.record) {
                Some(t) => t,
                None => continue, // No text to chunk
            };

            let text_chunks = chunker.chunk(&text);
            if text_chunks.is_empty() {
                continue;
            }

            let start_idx = all_chunk_texts.len();
            for chunk in &text_chunks {
                all_chunk_texts.push(chunk.text.clone());
                chunk_to_record.push((all_chunk_texts.len() - 1 - start_idx, record_idx));
            }

            all_chunks_data.push((record_idx, text_chunks));
        }

        if all_chunk_texts.is_empty() {
            // Still need to truncate WAL even if no texts to embed
            if let Some(ref mut wal) = self.wal {
                wal.truncate()?;
            }
            println!("Migration complete: no texts to embed");
            return Ok(());
        }

        // Batch embed all chunk texts
        println!("Embedding {} chunks...", all_chunk_texts.len());
        let all_embeddings = self.embedder.embed(all_chunk_texts)?;

        // Build chunks and add to index
        let mut embedding_idx = 0;
        for (record_idx, text_chunks) in all_chunks_data {
            let record_id = &self.records[record_idx].record.id;

            for chunk in text_chunks {
                let embedding = all_embeddings[embedding_idx].clone();
                embedding_idx += 1;

                let stored_chunk = StoredChunk::new(
                    record_id.clone(),
                    chunk.clone(),
                    embedding.clone(),
                );

                // Add to vector index
                self.index.insert(embedding);

                // Update chunk mapping
                self.chunk_mapping.push((record_idx, chunk.chunk_index));

                // Store chunk
                self.chunks.push(stored_chunk);
            }
        }

        // Save everything
        self.index.save(&self.snapshot_path)?;
        if let Some(ref cs) = self.chunk_store {
            cs.save_all(&self.chunks)?;
        }

        // Truncate WAL since we have a fresh state
        if let Some(ref mut wal) = self.wal {
            wal.truncate()?;
        }

        println!("Migration complete: {} chunks created", self.chunks.len());
        Ok(())
    }

    pub fn insert(&mut self, cmd: InsertCommand) -> Result<InsertResult, AppError> {
        if cmd.record.id.trim().is_empty() {
            return Err(AppError::BadRequest("id is required".to_string()));
        }
        // Check for existing non-deleted record with same ID
        if self.records.iter().any(|r| r.record.id == cmd.record.id && !r.deleted) {
            return Err(AppError::Conflict("id already exists".to_string()));
        }

        let text = build_text(cmd.text.as_deref(), &cmd.record)
            .ok_or_else(|| AppError::BadRequest("text/title/body is required".to_string()))?;

        // Check if chunking is enabled
        if let Some(ref chunker) = self.chunker {
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
                let stored_chunk = StoredChunk::new(
                    cmd.record.id.clone(),
                    chunk.clone(),
                    embedding.clone(),
                );

                // Add chunk embedding to vector index
                let chunk_index_id = self.index.insert(embedding);
                if first_index_id.is_none() {
                    first_index_id = Some(chunk_index_id);
                }

                // Update chunk mapping
                self.chunk_mapping.push((record_idx, chunk.chunk_index));

                // Store chunk
                self.chunks.push(stored_chunk.clone());
                if let Some(ref cs) = self.chunk_store {
                    cs.append(&stored_chunk)?;
                }
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
        } else {
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

    pub fn update(&mut self, id: &str, cmd: UpdateCommand) -> Result<DocumentResponse, AppError> {
        let index_id = self.find_index(id)?;
        let current = self.records[index_id].clone();

        let needs_embedding = cmd.text.is_some() || cmd.title.is_some() || cmd.body.is_some();
        let updated = Record {
            id: current.record.id.clone(),
            title: cmd.title.or(current.record.title),
            body: cmd.body.or(current.record.body),
            source: cmd.source.or(current.record.source),
            updated_at: cmd.updated_at.or(current.record.updated_at),
            tags: cmd.tags.or(current.record.tags),
        };

        // Get text for embedding if needed
        let text_for_embedding = if needs_embedding {
            let text = cmd.text.or_else(|| build_text(None, &updated))
                .ok_or_else(|| AppError::BadRequest("text/title/body is required".to_string()))?;
            if text.trim().is_empty() {
                return Err(AppError::BadRequest("text must not be empty".to_string()));
            }
            Some(text)
        } else {
            None
        };

        if let Some(text) = text_for_embedding {
            // Embedding changed: tombstone old record/chunks, append new

            // Check if chunking is enabled
            if let Some(ref chunker) = self.chunker {
                // Chunking mode: re-chunk and re-index

                // Mark old record as deleted
                self.records[index_id].deleted = true;
                self.tombstone_count += 1;

                // Mark all old chunks for this document as deleted
                for chunk in &mut self.chunks {
                    if chunk.parent_id == id {
                        chunk.deleted = true;
                    }
                }

                // Update BM25: remove old
                self.bm25_index.remove(index_id);

                // Generate new chunks
                let text_chunks = chunker.chunk(&text);
                if text_chunks.is_empty() {
                    return Err(AppError::BadRequest("text produced no chunks".to_string()));
                }

                // Get embeddings for all new chunks
                let chunk_texts: Vec<String> = text_chunks.iter().map(|c| c.text.clone()).collect();
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
                    let stored_chunk = StoredChunk::new(
                        updated.id.clone(),
                        chunk.clone(),
                        emb.clone(),
                    );

                    // Add chunk embedding to vector index
                    let chunk_index_id = self.index.insert(emb);
                    if first_index_id.is_none() {
                        first_index_id = Some(chunk_index_id);
                    }

                    // Update chunk mapping
                    self.chunk_mapping.push((new_record_idx, chunk.chunk_index));

                    // Store chunk
                    self.chunks.push(stored_chunk);
                }

                // Add to BM25 index
                self.bm25_index.add(new_record_idx, &text);

                // Save state
                self.index.save(&self.snapshot_path)?;
                self.record_store.save_all(&self.records)?;
                if let Some(ref cs) = self.chunk_store {
                    cs.save_all(&self.chunks)?;
                }

                // Check if compaction is needed
                self.maybe_compact()?;

                // Check if WAL rotation is needed
                self.maybe_rotate_wal()?;

                Ok(DocumentResponse {
                    index_id: first_index_id.unwrap_or(0),
                    record: updated,
                    embedding: doc_embedding,
                })
            } else {
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
        } else {
            // Only metadata changed: update in place (no index change needed)
            let stored = StoredRecord::new(updated.clone(), current.embedding.clone());

            // Write to WAL first
            if let Some(ref mut wal) = self.wal {
                wal.append_update(&stored)?;
            }

            // Update record in place
            self.records[index_id] = stored;

            // Update BM25 index
            if let Some(text) = build_text(None, &updated) {
                self.bm25_index.update(index_id, &text);
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

    pub fn delete(&mut self, id: &str) -> Result<(), AppError> {
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

        // Mark all chunks for this document as deleted
        if self.chunker.is_some() {
            for chunk in &mut self.chunks {
                if chunk.parent_id == id {
                    chunk.deleted = true;
                }
            }
            // Save updated chunks
            if let Some(ref cs) = self.chunk_store {
                cs.save_all(&self.chunks)?;
            }
        }

        self.record_store.save_all(&self.records)?;

        // Check if compaction is needed
        self.maybe_compact()?;

        // Check if WAL rotation is needed
        self.maybe_rotate_wal()?;

        Ok(())
    }

    pub fn search(&self, cmd: SearchCommand) -> Result<Vec<SearchHit>, AppError> {
        let k = cmd.k.unwrap_or(self.default_k);
        let ef = cmd.ef.unwrap_or(self.default_ef);
        let alpha = cmd.alpha.unwrap_or(0.7).clamp(0.0, 1.0);

        let query_text = cmd.text.clone();

        // Validate parameters based on alpha
        // alpha=0.0 (BM25-only): requires text, no embedding needed
        // alpha=1.0 (vector-only): requires text or embedding
        // 0 < alpha < 1 (hybrid): requires text or embedding
        if alpha == 0.0 && query_text.is_none() {
            return Err(AppError::BadRequest(
                "alpha=0.0 (BM25-only) requires 'text' parameter".to_string(),
            ));
        }
        if alpha > 0.0 && query_text.is_none() && cmd.embedding.is_none() {
            return Err(AppError::BadRequest("text or embedding is required".to_string()));
        }

        let search_k = (k * 4).max(100);
        let search_ef = ef.max(search_k);

        // Check if chunking is enabled
        let chunking_enabled = self.chunker.is_some() && !self.chunk_mapping.is_empty();

        // Get vector search results (skip if alpha=0.0, BM25-only)
        // When chunking is enabled, this returns chunk-level results that need to be aggregated
        let (vector_scores, chunk_info_map): (HashMap<usize, f32>, HashMap<usize, (usize, f32)>) = if alpha == 0.0 {
            (HashMap::new(), HashMap::new())
        } else {
            let embedding = if let Some(embedding) = cmd.embedding.clone() {
                embedding
            } else if let Some(ref text) = query_text {
                self.embed_single(text.clone())?
            } else {
                return Err(AppError::BadRequest("text or embedding is required".to_string()));
            };

            let vector_results = self.index.search(&embedding, search_k, search_ef);

            if chunking_enabled {
                // Chunking mode: aggregate chunk results by parent document
                // chunk_mapping[index_id] = (record_idx, chunk_idx)
                let mut parent_scores: HashMap<usize, f32> = HashMap::new();
                let mut parent_best_chunk: HashMap<usize, (usize, f32)> = HashMap::new();

                for result in vector_results {
                    // Check if the chunk itself is deleted
                    if let Some(chunk) = self.chunks.get(result.id) {
                        if chunk.deleted {
                            continue;
                        }
                    }

                    if let Some(&(record_idx, chunk_idx)) = self.chunk_mapping.get(result.id) {
                        // Skip invalid mappings (usize::MAX sentinel)
                        if record_idx == usize::MAX {
                            continue;
                        }
                        if let Some(record) = self.records.get(record_idx) {
                            if !record.deleted {
                                // Use minimum distance (best match) for the parent
                                let current_dist = parent_scores.entry(record_idx).or_insert(f32::MAX);
                                if result.distance < *current_dist {
                                    *current_dist = result.distance;
                                    parent_best_chunk.insert(record_idx, (chunk_idx, result.distance));
                                }
                            }
                        }
                    }
                }
                (parent_scores, parent_best_chunk)
            } else {
                // Non-chunking mode: direct record lookup
                let scores: HashMap<usize, f32> = vector_results
                    .into_iter()
                    .filter_map(|result| {
                        self.records.get(result.id).and_then(|record| {
                            if record.deleted {
                                None
                            } else {
                                Some((result.id, result.distance))
                            }
                        })
                    })
                    .collect();
                (scores, HashMap::new())
            }
        };

        // Get BM25 results (document-level, not chunk-level)
        let bm25_scores: HashMap<usize, f64> = if let Some(ref text) = query_text {
            self.bm25_index
                .search(text, search_k)
                .into_iter()
                .filter(|(id, _)| {
                    self.records.get(*id).map(|r| !r.deleted).unwrap_or(false)
                })
                .collect()
        } else {
            HashMap::new()
        };

        // Combine candidates based on alpha:
        // alpha == 1.0 → vector only
        // alpha == 0.0 → BM25 only
        // otherwise   → union of both
        let all_candidates: HashSet<usize> = if alpha == 1.0 {
            vector_scores.keys().copied().collect()
        } else if alpha == 0.0 {
            bm25_scores.keys().copied().collect()
        } else {
            vector_scores
                .keys()
                .chain(bm25_scores.keys())
                .copied()
                .collect()
        };

        // Normalize scores and compute hybrid score
        let max_bm25 = bm25_scores.values().cloned().fold(0.0f64, f64::max);
        let min_distance = vector_scores.values().cloned().fold(f32::MAX, f32::min);
        let max_distance = vector_scores.values().cloned().fold(0.0f32, f32::max);

        let mut hits: Vec<SearchHit> = all_candidates
            .into_iter()
            .filter_map(|record_idx| {
                let record = self.records.get(record_idx)?;
                if record.deleted {
                    return None;
                }

                // Apply metadata filter if present
                if let Some(ref filter) = cmd.filter {
                    if !filter.matches(&record.record) {
                        return None;
                    }
                }

                let distance = vector_scores.get(&record_idx).copied();
                let bm25_raw = bm25_scores.get(&record_idx).copied();

                // Normalize vector score
                let vector_score = if let Some(d) = distance {
                    if max_distance > min_distance {
                        1.0 - ((d - min_distance) / (max_distance - min_distance)) as f64
                    } else {
                        1.0 - d as f64
                    }
                } else {
                    0.0
                };

                // Normalize BM25 score to [0, 1]
                let bm25_score = if let Some(bm25) = bm25_raw {
                    if max_bm25 > 0.0 { bm25 / max_bm25 } else { 0.0 }
                } else {
                    0.0
                };

                // Hybrid score: alpha * vector + (1 - alpha) * bm25
                let hybrid_score = alpha * vector_score + (1.0 - alpha) * bm25_score;

                // Build best_chunk info if chunking is enabled
                let best_chunk = if chunking_enabled {
                    chunk_info_map.get(&record_idx).map(|&(chunk_idx, _)| {
                        // Find the chunk text for preview
                        let text_preview = self.chunks
                            .iter()
                            .find(|c| c.parent_id == record.record.id && c.chunk.chunk_index == chunk_idx)
                            .map(|c| create_text_preview(&c.chunk.text, 100));
                        ChunkInfo {
                            chunk_index: chunk_idx,
                            text_preview,
                        }
                    })
                } else {
                    None
                };

                Some(SearchHit {
                    index_id: record_idx,
                    id: record.record.id.clone(),
                    distance,
                    bm25_score: bm25_raw,
                    hybrid_score: Some(hybrid_score),
                    title: record.record.title.clone(),
                    source: record.record.source.clone(),
                    tags: record.record.tags.clone(),
                    best_chunk,
                })
            })
            .collect();

        // Sort by hybrid score (descending)
        hits.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);

        Ok(hits)
    }

    pub fn get(&self, id: &str) -> Result<DocumentResponse, AppError> {
        let index_id = self.find_index(id)?;
        let stored = self.records[index_id].clone();
        Ok(DocumentResponse {
            index_id,
            record: stored.record,
            embedding: stored.embedding,
        })
    }

    pub fn list(&self, offset: usize, limit: usize) -> (usize, Vec<DocumentSummary>) {
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

    pub fn import_embeddings(&mut self, records: Vec<StoredRecord>) -> Result<usize, AppError> {
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

    pub fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Err(AppError::BadRequest("texts is required".to_string()));
        }
        Ok(self.embedder.embed(texts)?)
    }

    /// Get tombstone statistics
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

    /// Force compaction (rebuild index without tombstones)
    pub fn compact(&mut self) -> Result<(), AppError> {
        self.rebuild_without_tombstones()
    }

    fn rebuild_without_tombstones(&mut self) -> Result<(), AppError> {
        // Remove tombstones from records
        self.records.retain(|r| !r.deleted);
        self.tombstone_count = 0;

        // Handle chunking mode vs non-chunking mode
        if self.chunker.is_some() && !self.chunks.is_empty() {
            // Chunking mode: rebuild from chunks
            self.chunks.retain(|c| !c.deleted);

            // Create fake StoredRecords from chunks for index rebuilding
            let chunk_records: Vec<StoredRecord> = self.chunks
                .iter()
                .map(|c| StoredRecord::new(
                    Record {
                        id: format!("{}_{}", c.parent_id, c.chunk.chunk_index),
                        title: None,
                        body: None,
                        source: None,
                        updated_at: None,
                        tags: None,
                    },
                    c.embedding.clone(),
                ))
                .collect();

            self.index = self.index_factory.rebuild(&chunk_records);

            // Rebuild chunk_mapping
            self.chunk_mapping = build_chunk_mapping(&self.chunks, &self.records);

            // Save chunks
            if let Some(ref cs) = self.chunk_store {
                cs.save_all(&self.chunks)?;
            }
        } else {
            // Non-chunking mode: rebuild from records
            self.index = self.index_factory.rebuild(&self.records);
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

    fn maybe_compact(&mut self) -> Result<(), AppError> {
        let total = self.records.len();
        if total == 0 {
            return Ok(());
        }

        let should_compact = if let Some(max_tombstones) = self.tombstone_config.max_tombstones {
            self.tombstone_count >= max_tombstones
        } else if let Some(max_ratio) = self.tombstone_config.max_tombstone_ratio {
            let ratio = self.tombstone_count as f64 / total as f64;
            ratio >= max_ratio
        } else {
            false
        };

        if should_compact {
            self.rebuild_without_tombstones()?;
        }

        Ok(())
    }

    fn maybe_rotate_wal(&mut self) -> Result<(), AppError> {
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

    fn find_index(&self, id: &str) -> Result<usize, AppError> {
        self.records
            .iter()
            .position(|record| record.record.id == id && !record.deleted)
            .ok_or_else(|| AppError::NotFound("record not found".to_string()))
    }

    fn embed_single(&self, text: String) -> Result<Vec<f32>, AppError> {
        self.embedder
            .embed(vec![text])?
            .into_iter()
            .next()
            .ok_or_else(|| AppError::Io("embedding response is empty".to_string()))
    }
}

fn build_text(explicit: Option<&str>, record: &Record) -> Option<String> {
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

fn fill_missing_embeddings(records: &mut [StoredRecord], index: &dyn VectorIndex) -> bool {
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
fn build_chunk_mapping(chunks: &[StoredChunk], records: &[StoredRecord]) -> Vec<(usize, usize)> {
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
fn chunks_to_fake_records(chunks: &[StoredChunk]) -> Vec<StoredRecord> {
    chunks
        .iter()
        .map(|c| StoredRecord::with_deleted(
            Record {
                id: format!("{}_{}", c.parent_id, c.chunk.chunk_index),
                title: None,
                body: None,
                source: None,
                updated_at: None,
                tags: None,
            },
            c.embedding.clone(),
            c.deleted,
        ))
        .collect()
}

/// Create a text preview from chunk text (first ~100 characters)
fn create_text_preview(text: &str, max_len: usize) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        id: &str,
        source: Option<&str>,
        tags: Option<&str>,
        updated_at: Option<&str>,
    ) -> Record {
        Record {
            id: id.to_string(),
            title: Some("Test".to_string()),
            body: None,
            source: source.map(|s| s.to_string()),
            updated_at: updated_at.map(|s| s.to_string()),
            tags: tags.map(|s| s.to_string()),
        }
    }

    #[test]
    fn test_filter_empty_matches_all() {
        let filter = SearchFilter::default();
        let record = make_record("1", Some("news"), Some("ai, rust"), Some("2024-06-15"));
        assert!(filter.matches(&record));
    }

    #[test]
    fn test_filter_source_exact_match() {
        let filter = SearchFilter {
            source: Some("news".to_string()),
            ..Default::default()
        };
        let record = make_record("1", Some("news"), None, None);
        assert!(filter.matches(&record));

        let record2 = make_record("2", Some("blog"), None, None);
        assert!(!filter.matches(&record2));

        let record3 = make_record("3", None, None, None);
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_tags_any() {
        let filter = SearchFilter {
            tags_any: Some(vec!["ai".to_string(), "ml".to_string()]),
            ..Default::default()
        };

        // Has "ai" tag
        let record1 = make_record("1", None, Some("ai, rust"), None);
        assert!(filter.matches(&record1));

        // Has "ml" tag
        let record2 = make_record("2", None, Some("ml, python"), None);
        assert!(filter.matches(&record2));

        // Has neither
        let record3 = make_record("3", None, Some("rust, go"), None);
        assert!(!filter.matches(&record3));

        // No tags at all
        let record4 = make_record("4", None, None, None);
        assert!(!filter.matches(&record4));
    }

    #[test]
    fn test_filter_tags_all() {
        let filter = SearchFilter {
            tags_all: Some(vec!["ai".to_string(), "rust".to_string()]),
            ..Default::default()
        };

        // Has both tags
        let record1 = make_record("1", None, Some("ai, rust, news"), None);
        assert!(filter.matches(&record1));

        // Has only one
        let record2 = make_record("2", None, Some("ai, python"), None);
        assert!(!filter.matches(&record2));

        // Has neither
        let record3 = make_record("3", None, Some("go, python"), None);
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_tags_case_insensitive() {
        let filter = SearchFilter {
            tags_any: Some(vec!["AI".to_string(), "RUST".to_string()]),
            ..Default::default()
        };

        let record = make_record("1", None, Some("ai, rust"), None);
        assert!(filter.matches(&record));

        let filter2 = SearchFilter {
            tags_all: Some(vec!["AI".to_string()]),
            ..Default::default()
        };
        assert!(filter2.matches(&record));
    }

    #[test]
    fn test_filter_tags_with_whitespace() {
        let filter = SearchFilter {
            tags_any: Some(vec!["ai".to_string()]),
            ..Default::default()
        };

        // Tags with extra whitespace
        let record = make_record("1", None, Some("  ai  ,  rust  "), None);
        assert!(filter.matches(&record));
    }

    #[test]
    fn test_filter_updated_at_gte() {
        let filter = SearchFilter {
            updated_at_gte: Some("2024-06-01".to_string()),
            ..Default::default()
        };

        let record1 = make_record("1", None, None, Some("2024-06-15"));
        assert!(filter.matches(&record1));

        let record2 = make_record("2", None, None, Some("2024-06-01"));
        assert!(filter.matches(&record2));

        let record3 = make_record("3", None, None, Some("2024-05-31"));
        assert!(!filter.matches(&record3));

        // No updated_at
        let record4 = make_record("4", None, None, None);
        assert!(!filter.matches(&record4));
    }

    #[test]
    fn test_filter_updated_at_lte() {
        let filter = SearchFilter {
            updated_at_lte: Some("2024-06-30".to_string()),
            ..Default::default()
        };

        let record1 = make_record("1", None, None, Some("2024-06-15"));
        assert!(filter.matches(&record1));

        let record2 = make_record("2", None, None, Some("2024-06-30"));
        assert!(filter.matches(&record2));

        let record3 = make_record("3", None, None, Some("2024-07-01"));
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_updated_at_range() {
        let filter = SearchFilter {
            updated_at_gte: Some("2024-01-01".to_string()),
            updated_at_lte: Some("2024-12-31".to_string()),
            ..Default::default()
        };

        let record1 = make_record("1", None, None, Some("2024-06-15"));
        assert!(filter.matches(&record1));

        let record2 = make_record("2", None, None, Some("2023-12-31"));
        assert!(!filter.matches(&record2));

        let record3 = make_record("3", None, None, Some("2025-01-01"));
        assert!(!filter.matches(&record3));
    }

    #[test]
    fn test_filter_combined() {
        let filter = SearchFilter {
            source: Some("news".to_string()),
            tags_any: Some(vec!["ai".to_string(), "rust".to_string()]),
            updated_at_gte: Some("2024-01-01".to_string()),
            ..Default::default()
        };

        // All conditions match
        let record1 = make_record("1", Some("news"), Some("ai, tech"), Some("2024-06-15"));
        assert!(filter.matches(&record1));

        // Wrong source
        let record2 = make_record("2", Some("blog"), Some("ai, tech"), Some("2024-06-15"));
        assert!(!filter.matches(&record2));

        // No matching tags
        let record3 = make_record("3", Some("news"), Some("go, python"), Some("2024-06-15"));
        assert!(!filter.matches(&record3));

        // Too old
        let record4 = make_record("4", Some("news"), Some("ai, tech"), Some("2023-12-31"));
        assert!(!filter.matches(&record4));
    }

    #[test]
    fn test_parse_tags() {
        let tags = parse_tags(Some("ai, rust, ML"));
        assert!(tags.contains("ai"));
        assert!(tags.contains("rust"));
        assert!(tags.contains("ml"));
        assert_eq!(tags.len(), 3);

        let empty = parse_tags(None);
        assert!(empty.is_empty());

        let empty2 = parse_tags(Some(""));
        assert!(empty2.is_empty());

        let with_spaces = parse_tags(Some("  ai  ,  ,  rust  "));
        assert!(with_spaces.contains("ai"));
        assert!(with_spaces.contains("rust"));
        assert_eq!(with_spaces.len(), 2);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::domain::model::Chunk;
    use crate::domain::ports::{ChunkStore, Chunker, Embedder, RecordStore, VectorIndex, VectorIndexFactory};
    use crate::domain::types::SearchResult;
    use std::io;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Mock embedder that returns deterministic embeddings
    struct MockEmbedder {
        dim: usize,
        call_count: AtomicUsize,
    }

    impl MockEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: Vec<String>) -> io::Result<Vec<Vec<f32>>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    // Create deterministic embedding based on text hash
                    let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
                    (0..self.dim)
                        .map(|j| ((hash as f32 + i as f32 + j as f32) % 100.0) / 100.0)
                        .collect()
                })
                .collect())
        }
    }

    /// Simple in-memory record store
    struct MemoryRecordStore {
        records: Mutex<Vec<StoredRecord>>,
    }

    impl MemoryRecordStore {
        fn new() -> Self {
            Self {
                records: Mutex::new(Vec::new()),
            }
        }

        fn with_records(records: Vec<StoredRecord>) -> Self {
            Self {
                records: Mutex::new(records),
            }
        }
    }

    impl RecordStore for MemoryRecordStore {
        fn load(&self) -> io::Result<Vec<StoredRecord>> {
            Ok(self.records.lock().unwrap().clone())
        }

        fn save_all(&self, records: &[StoredRecord]) -> io::Result<()> {
            *self.records.lock().unwrap() = records.to_vec();
            Ok(())
        }

        fn append(&self, record: &StoredRecord) -> io::Result<()> {
            self.records.lock().unwrap().push(record.clone());
            Ok(())
        }

        fn path(&self) -> &Path {
            Path::new("memory://records")
        }
    }

    /// Simple in-memory chunk store
    struct MemoryChunkStore {
        chunks: Mutex<Vec<StoredChunk>>,
    }

    impl MemoryChunkStore {
        fn new() -> Self {
            Self {
                chunks: Mutex::new(Vec::new()),
            }
        }
    }

    impl ChunkStore for MemoryChunkStore {
        fn load(&self) -> io::Result<Vec<StoredChunk>> {
            Ok(self.chunks.lock().unwrap().clone())
        }

        fn save_all(&self, chunks: &[StoredChunk]) -> io::Result<()> {
            *self.chunks.lock().unwrap() = chunks.to_vec();
            Ok(())
        }

        fn append(&self, chunk: &StoredChunk) -> io::Result<()> {
            self.chunks.lock().unwrap().push(chunk.clone());
            Ok(())
        }

        fn path(&self) -> &Path {
            Path::new("memory://chunks")
        }
    }

    /// Simple fixed-size chunker for testing
    struct TestChunker {
        chunk_size: usize,
    }

    impl TestChunker {
        fn new(chunk_size: usize) -> Self {
            Self { chunk_size }
        }
    }

    impl Chunker for TestChunker {
        fn chunk(&self, text: &str) -> Vec<Chunk> {
            let chars: Vec<char> = text.chars().collect();
            if chars.is_empty() {
                return Vec::new();
            }
            if chars.len() <= self.chunk_size {
                return vec![Chunk {
                    chunk_index: 0,
                    text: text.to_string(),
                    start_offset: 0,
                    end_offset: text.len(),
                }];
            }

            let mut chunks = Vec::new();
            let mut start = 0;
            let mut chunk_idx = 0;

            while start < chars.len() {
                let end = (start + self.chunk_size).min(chars.len());
                let chunk_text: String = chars[start..end].iter().collect();
                let byte_start = chars[..start].iter().map(|c| c.len_utf8()).sum();
                let byte_end = chars[..end].iter().map(|c| c.len_utf8()).sum();

                chunks.push(Chunk {
                    chunk_index: chunk_idx,
                    text: chunk_text,
                    start_offset: byte_start,
                    end_offset: byte_end,
                });

                start = end;
                chunk_idx += 1;
            }
            chunks
        }
    }

    /// Simple in-memory vector index
    struct MemoryIndex {
        vectors: Vec<Vec<f32>>,
        deleted: Vec<bool>,
    }

    impl MemoryIndex {
        fn new() -> Self {
            Self {
                vectors: Vec::new(),
                deleted: Vec::new(),
            }
        }
    }

    impl VectorIndex for MemoryIndex {
        fn insert(&mut self, embedding: Vec<f32>) -> usize {
            let id = self.vectors.len();
            self.vectors.push(embedding);
            self.deleted.push(false);
            id
        }

        fn search(&self, query: &[f32], k: usize, _ef: usize) -> Vec<SearchResult> {
            let mut results: Vec<(usize, f32)> = self
                .vectors
                .iter()
                .enumerate()
                .filter(|(i, _)| !self.deleted.get(*i).copied().unwrap_or(true))
                .map(|(i, v)| {
                    let dist = v
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    (i, dist)
                })
                .collect();

            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results
                .into_iter()
                .take(k)
                .map(|(id, distance)| SearchResult { id, distance })
                .collect()
        }

        fn len(&self) -> usize {
            self.vectors.len()
        }

        fn dim(&self) -> usize {
            self.vectors.first().map(|v| v.len()).unwrap_or(0)
        }

        fn vector(&self, id: usize) -> Option<&Vec<f32>> {
            self.vectors.get(id)
        }

        fn save(&self, _path: &Path) -> io::Result<()> {
            Ok(())
        }
    }

    /// Simple index factory
    struct MemoryIndexFactory;

    impl VectorIndexFactory for MemoryIndexFactory {
        fn load_or_create(
            &self,
            _path: &Path,
            records: &[StoredRecord],
        ) -> io::Result<Box<dyn VectorIndex>> {
            let mut index = MemoryIndex::new();
            for record in records {
                index.insert(record.embedding.clone());
            }
            Ok(Box::new(index))
        }

        fn rebuild(&self, records: &[StoredRecord]) -> Box<dyn VectorIndex> {
            let mut index = MemoryIndex::new();
            for record in records {
                index.insert(record.embedding.clone());
            }
            Box::new(index)
        }
    }

    fn create_test_config(tmp_dir: &Path) -> ServiceConfig {
        ServiceConfig {
            snapshot_path: tmp_dir.join("index.bin"),
            default_k: 5,
            default_ef: 100,
            wal_config: None,
            tombstone_config: TombstoneConfig::default(),
            chunk_config: ChunkConfig::Fixed {
                chunk_size: 100,
                overlap: 0,
            },
        }
    }

    #[test]
    fn test_migration_from_records_to_chunking() {
        // Scenario: Existing records without chunks, then enable chunking
        let tmp_dir = tempfile::tempdir().unwrap();

        // Create existing records (simulating pre-chunking data)
        let existing_records = vec![
            StoredRecord::new(
                Record {
                    id: "doc1".to_string(),
                    title: Some("First Document".to_string()),
                    body: Some("This is a long document that should be split into multiple chunks when chunking is enabled.".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                vec![0.1; 8], // Pre-existing embedding (will be ignored in chunking mode)
            ),
            StoredRecord::new(
                Record {
                    id: "doc2".to_string(),
                    title: Some("Second Document".to_string()),
                    body: Some("Another document with some content.".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                vec![0.2; 8],
            ),
        ];

        let record_store = Box::new(MemoryRecordStore::with_records(existing_records));
        let chunk_store = Box::new(MemoryChunkStore::new());
        let chunker = Box::new(TestChunker::new(50)) as Box<dyn Chunker>;
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);

        let config = create_test_config(tmp_dir.path());

        // Load service with chunking enabled but no existing chunks
        // This should trigger migration
        let service = DocumentService::load(
            config,
            embedder,
            record_store,
            index_factory,
            Some(chunker),
            Some(chunk_store),
        )
        .expect("Failed to load service");

        // Verify migration happened
        assert!(!service.chunks.is_empty(), "Chunks should have been created during migration");
        assert!(
            !service.chunk_mapping.is_empty(),
            "Chunk mapping should have been created"
        );

        // Verify search works
        let results = service
            .search(SearchCommand {
                text: Some("document".to_string()),
                embedding: None,
                k: Some(5),
                ef: None,
                alpha: Some(1.0), // Vector only
                filter: None,
            })
            .expect("Search failed");

        assert!(!results.is_empty(), "Search should return results after migration");
    }

    #[test]
    fn test_insert_with_chunking_creates_chunks() {
        let tmp_dir = tempfile::tempdir().unwrap();

        let record_store = Box::new(MemoryRecordStore::new());
        let chunk_store = Box::new(MemoryChunkStore::new());
        let chunker = Box::new(TestChunker::new(50)) as Box<dyn Chunker>;
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);

        let config = create_test_config(tmp_dir.path());

        let mut service = DocumentService::load(
            config,
            embedder,
            record_store,
            index_factory,
            Some(chunker),
            Some(chunk_store),
        )
        .expect("Failed to load service");

        // Insert a document that should be chunked
        let result = service
            .insert(InsertCommand {
                record: Record {
                    id: "doc1".to_string(),
                    title: Some("Test Document".to_string()),
                    body: Some("This is a test document with enough text to be split into multiple chunks for testing purposes.".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                text: None,
            })
            .expect("Insert failed");

        assert_eq!(result.id, "doc1");

        // Verify chunks were created
        assert!(!service.chunks.is_empty(), "Chunks should have been created");

        // Verify chunk_mapping aligns with index
        assert_eq!(
            service.chunk_mapping.len(),
            service.chunks.len(),
            "Chunk mapping should match chunks count"
        );

        // Verify search returns results with chunk info
        let search_results = service
            .search(SearchCommand {
                text: Some("test document".to_string()),
                embedding: None,
                k: Some(5),
                ef: None,
                alpha: Some(1.0),
                filter: None,
            })
            .expect("Search failed");

        assert!(!search_results.is_empty(), "Search should return results");
        assert!(
            search_results[0].best_chunk.is_some(),
            "Result should include best_chunk info"
        );
    }

    #[test]
    fn test_chunk_mapping_alignment_after_multiple_inserts() {
        let tmp_dir = tempfile::tempdir().unwrap();

        let record_store = Box::new(MemoryRecordStore::new());
        let chunk_store = Box::new(MemoryChunkStore::new());
        let chunker = Box::new(TestChunker::new(30)) as Box<dyn Chunker>;
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);

        let config = create_test_config(tmp_dir.path());

        let mut service = DocumentService::load(
            config,
            embedder,
            record_store,
            index_factory,
            Some(chunker),
            Some(chunk_store),
        )
        .expect("Failed to load service");

        // Insert multiple documents
        for i in 0..5 {
            service
                .insert(InsertCommand {
                    record: Record {
                        id: format!("doc{}", i),
                        title: Some(format!("Document {}", i)),
                        body: Some(format!(
                            "Content for document {} that is long enough to be chunked into pieces.",
                            i
                        )),
                        source: None,
                        updated_at: None,
                        tags: None,
                    },
                    text: None,
                })
                .expect("Insert failed");
        }

        // Verify alignment: chunk_mapping.len() == chunks.len() == index.len()
        assert_eq!(
            service.chunk_mapping.len(),
            service.chunks.len(),
            "Chunk mapping should equal chunks count"
        );

        // Each mapping should point to valid record and chunk
        for (idx, &(record_idx, chunk_idx)) in service.chunk_mapping.iter().enumerate() {
            assert!(
                record_idx < service.records.len(),
                "Record index {} out of bounds at mapping {}",
                record_idx,
                idx
            );
            // Find corresponding chunk
            let chunk = &service.chunks[idx];
            assert_eq!(
                chunk.chunk.chunk_index, chunk_idx,
                "Chunk index mismatch at mapping {}", idx
            );
        }
    }

    #[test]
    fn test_delete_marks_chunks_as_deleted() {
        let tmp_dir = tempfile::tempdir().unwrap();

        let record_store = Box::new(MemoryRecordStore::new());
        let chunk_store = Box::new(MemoryChunkStore::new());
        let chunker = Box::new(TestChunker::new(30)) as Box<dyn Chunker>;
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);

        let config = create_test_config(tmp_dir.path());

        let mut service = DocumentService::load(
            config,
            embedder,
            record_store,
            index_factory,
            Some(chunker),
            Some(chunk_store),
        )
        .expect("Failed to load service");

        // Insert a document
        service
            .insert(InsertCommand {
                record: Record {
                    id: "doc1".to_string(),
                    title: Some("Test".to_string()),
                    body: Some("Long enough content to create chunks for testing delete.".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                text: None,
            })
            .expect("Insert failed");

        let chunks_before_delete: Vec<_> = service.chunks.iter().filter(|c| !c.deleted).collect();
        assert!(!chunks_before_delete.is_empty());

        // Delete the document
        service.delete("doc1").expect("Delete failed");

        // Verify all chunks for doc1 are marked as deleted
        let active_chunks_for_doc1: Vec<_> = service
            .chunks
            .iter()
            .filter(|c| c.parent_id == "doc1" && !c.deleted)
            .collect();
        assert!(
            active_chunks_for_doc1.is_empty(),
            "All chunks should be deleted"
        );

        // Search should not return deleted document
        let results = service
            .search(SearchCommand {
                text: Some("test content".to_string()),
                embedding: None,
                k: Some(5),
                ef: None,
                alpha: Some(1.0),
                filter: None,
            })
            .expect("Search failed");

        let doc1_in_results = results.iter().any(|r| r.id == "doc1");
        assert!(!doc1_in_results, "Deleted document should not appear in search");
    }

    #[test]
    fn test_update_re_chunks_document() {
        let tmp_dir = tempfile::tempdir().unwrap();

        let record_store = Box::new(MemoryRecordStore::new());
        let chunk_store = Box::new(MemoryChunkStore::new());
        let chunker = Box::new(TestChunker::new(30)) as Box<dyn Chunker>;
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);

        let config = create_test_config(tmp_dir.path());

        let mut service = DocumentService::load(
            config,
            embedder,
            record_store,
            index_factory,
            Some(chunker),
            Some(chunk_store),
        )
        .expect("Failed to load service");

        // Insert a document
        service
            .insert(InsertCommand {
                record: Record {
                    id: "doc1".to_string(),
                    title: Some("Original Title".to_string()),
                    body: Some("Original content that will be changed.".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                text: None,
            })
            .expect("Insert failed");

        let chunks_before = service.chunks.len();

        // Update the document with new body
        service
            .update(
                "doc1",
                UpdateCommand {
                    title: None,
                    body: Some("Completely new and different content that is much longer and will produce more chunks.".to_string()),
                    source: None,
                    updated_at: None,
                    tags: None,
                    text: None,
                },
            )
            .expect("Update failed");

        // Verify old chunks are deleted and new ones created
        let active_chunks: Vec<_> = service.chunks.iter().filter(|c| !c.deleted).collect();
        assert!(!active_chunks.is_empty(), "Should have new active chunks");

        // Total chunks should be more (old deleted + new active)
        assert!(
            service.chunks.len() >= chunks_before,
            "Total chunks should include both old (deleted) and new"
        );
    }
}
