//! Integration tests for DocumentService.

#[cfg(test)]
mod integration_tests {
    use crate::application::service::{
        ChunkConfig, DocumentService, InsertCommand, SearchCommand, ServiceConfig, TombstoneConfig,
        UpdateCommand,
    };
    use crate::domain::model::{Chunk, Record, StoredChunk, StoredRecord};
    use crate::domain::ports::{
        ChunkStore, Chunker, Embedder, RecordStore, VectorIndex, VectorIndexFactory,
    };
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
        assert!(
            !service.indexing_mode.chunks().is_empty(),
            "Chunks should have been created during migration"
        );
        assert!(
            !service.indexing_mode.chunk_mapping().is_empty(),
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

        assert!(
            !results.is_empty(),
            "Search should return results after migration"
        );
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
        assert!(
            !service.indexing_mode.chunks().is_empty(),
            "Chunks should have been created"
        );

        // Verify chunk_mapping aligns with index
        assert_eq!(
            service.indexing_mode.chunk_mapping().len(),
            service.indexing_mode.chunks().len(),
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
        let chunks = service.indexing_mode.chunks();
        let chunk_mapping = service.indexing_mode.chunk_mapping();
        assert_eq!(
            chunk_mapping.len(),
            chunks.len(),
            "Chunk mapping should equal chunks count"
        );

        // Each mapping should point to valid record and chunk
        for (idx, &(record_idx, chunk_idx)) in chunk_mapping.iter().enumerate() {
            assert!(
                record_idx < service.records.len(),
                "Record index {} out of bounds at mapping {}",
                record_idx,
                idx
            );
            // Find corresponding chunk
            let chunk = &chunks[idx];
            assert_eq!(
                chunk.chunk.chunk_index, chunk_idx,
                "Chunk index mismatch at mapping {}",
                idx
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
                    body: Some(
                        "Long enough content to create chunks for testing delete.".to_string(),
                    ),
                    source: None,
                    updated_at: None,
                    tags: None,
                },
                text: None,
            })
            .expect("Insert failed");

        let chunks_before_delete: Vec<_> = service
            .indexing_mode
            .chunks()
            .iter()
            .filter(|c| !c.deleted)
            .collect();
        assert!(!chunks_before_delete.is_empty());

        // Delete the document
        service.delete("doc1").expect("Delete failed");

        // Verify all chunks for doc1 are marked as deleted
        let active_chunks_for_doc1: Vec<_> = service
            .indexing_mode
            .chunks()
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
        assert!(
            !doc1_in_results,
            "Deleted document should not appear in search"
        );
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

        let chunks_before = service.indexing_mode.chunks().len();

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
        let active_chunks: Vec<_> = service
            .indexing_mode
            .chunks()
            .iter()
            .filter(|c| !c.deleted)
            .collect();
        assert!(!active_chunks.is_empty(), "Should have new active chunks");

        // Total chunks should be more (old deleted + new active)
        assert!(
            service.indexing_mode.chunks().len() >= chunks_before,
            "Total chunks should include both old (deleted) and new"
        );
    }
}
