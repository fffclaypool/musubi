//! Load and migration operations for DocumentService.

use crate::application::error::AppError;
use crate::domain::model::StoredChunk;
use crate::domain::ports::{ChunkStore, Chunker, Embedder, RecordStore, VectorIndexFactory};
use crate::infrastructure::search::Bm25Index;
use crate::infrastructure::storage::wal::{self, WalWriter};

use super::core::DocumentService;
use super::types::{IndexingMode, ServiceConfig};
use super::util::{
    build_chunk_mapping, build_text, chunks_to_fake_records, fill_missing_embeddings,
};

impl DocumentService {
    /// Load or create a new DocumentService.
    pub fn load(
        config: ServiceConfig,
        embedder: Box<dyn Embedder>,
        record_store: Box<dyn RecordStore>,
        index_factory: Box<dyn VectorIndexFactory>,
        chunker: Option<Box<dyn Chunker>>,
        chunk_store: Option<Box<dyn ChunkStore>>,
    ) -> Result<Self, AppError> {
        // Convert tombstone config to policy (validates and rejects conflicting settings)
        let tombstone_policy = config
            .tombstone_config
            .into_policy()
            .map_err(|e| AppError::Io(e.to_string()))?;

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
        let needs_migration =
            chunking_enabled && chunks.is_empty() && !records.is_empty() && chunker.is_some();

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

        // Build indexing mode based on chunker/chunk_store availability
        let indexing_mode = match (chunker, chunk_store) {
            (Some(chunker), Some(chunk_store)) => {
                let chunk_mapping = build_chunk_mapping(&chunks, &records);
                IndexingMode::Chunked {
                    chunker,
                    chunk_store,
                    chunks,
                    chunk_mapping,
                }
            }
            _ => IndexingMode::Direct,
        };

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
            tombstone_policy,
            tombstone_count,
            bm25_index,
            indexing_mode,
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
    pub(super) fn migrate_to_chunking(&mut self) -> Result<(), AppError> {
        let IndexingMode::Chunked {
            chunker,
            chunk_store,
            chunks,
            chunk_mapping,
        } = &mut self.indexing_mode
        else {
            return Ok(()); // Direct mode, nothing to do
        };

        println!(
            "Starting migration: {} records to chunk...",
            self.records.len()
        );

        // Collect all texts that need to be chunked
        let mut all_chunks_data: Vec<(usize, Vec<crate::domain::model::Chunk>)> = Vec::new();
        let mut all_chunk_texts: Vec<String> = Vec::new();

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

            for chunk in &text_chunks {
                all_chunk_texts.push(chunk.text.clone());
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

                let stored_chunk =
                    StoredChunk::new(record_id.clone(), chunk.clone(), embedding.clone());

                // Add to vector index
                self.index.insert(embedding);

                // Update chunk mapping
                chunk_mapping.push((record_idx, chunk.chunk_index));

                // Store chunk
                chunks.push(stored_chunk);
            }
        }

        // Save everything
        self.index.save(&self.snapshot_path)?;
        chunk_store.save_all(chunks)?;

        // Truncate WAL since we have a fresh state
        if let Some(ref mut wal) = self.wal {
            wal.truncate()?;
        }

        println!("Migration complete: {} chunks created", chunks.len());
        Ok(())
    }
}
