//! Integration tests for DocumentService.

#[cfg(test)]
mod integration_tests {
    use crate::application::service::{
        BatchDocument, ChunkConfig, DocumentDefaults, DocumentIngestion, DocumentSearch,
        DocumentService, SearchRequest, ServiceConfig, TombstoneConfig, ValidatedSearchQuery,
        run_sync, JobProgress,
    };
    use crate::domain::model::{Chunk, Record, StoredChunk, StoredRecord, Tag};
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
                if record.indexed && !record.deleted && !record.embedding.is_empty() {
                    index.insert(record.embedding.clone());
                }
            }
            Ok(Box::new(index))
        }

        fn rebuild(&self, records: &[StoredRecord]) -> Box<dyn VectorIndex> {
            let mut index = MemoryIndex::new();
            for record in records {
                if record.indexed && !record.deleted && !record.embedding.is_empty() {
                    index.insert(record.embedding.clone());
                }
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
            chunk_config: ChunkConfig::None, // Use direct indexing for simpler tests
            pending_store_path: None,        // Tests don't use pending_store by default
        }
    }

    fn create_test_service(tmp_dir: &Path) -> DocumentService {
        let record_store = Box::new(MemoryRecordStore::new());
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);
        let config = create_test_config(tmp_dir);

        DocumentService::load(config, embedder, record_store, index_factory, None, None, None)
            .expect("Failed to load service")
    }

    #[test]
    fn test_batch_insert_creates_pending_records() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        let result = service.batch_insert(vec![
            BatchDocument {
                id: "doc1".to_string(),
                title: Some("First Document".to_string()),
                body: Some("Content of the first document.".to_string()),
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            },
            BatchDocument {
                id: "doc2".to_string(),
                title: Some("Second Document".to_string()),
                body: Some("Content of the second document.".to_string()),
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            },
        ]).unwrap();

        assert_eq!(result.accepted, 2);
        assert_eq!(result.failed, 0);
        assert!(result.errors.is_empty());

        // Verify documents are pending (not searchable yet)
        let pending_ids = service.get_pending_ids();
        assert_eq!(pending_ids.len(), 2);
        assert!(pending_ids.contains(&"doc1".to_string()));
        assert!(pending_ids.contains(&"doc2".to_string()));
    }

    #[test]
    fn test_batch_insert_validates_empty_id() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        let result = service.batch_insert(vec![
            BatchDocument {
                id: "".to_string(), // Empty ID
                title: Some("Document".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            },
            BatchDocument {
                id: "   ".to_string(), // Whitespace-only ID
                title: Some("Another".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            },
        ]).unwrap();

        assert_eq!(result.accepted, 0);
        assert_eq!(result.failed, 2);
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn test_sync_embeds_pending_documents() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        // Insert documents as pending
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Test Document".to_string()),
            body: Some("Test content for embedding.".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: None,
        }]).unwrap();

        // Verify not searchable yet
        let query_before = ValidatedSearchQuery::from_request(
            SearchRequest {
                text: Some("test".to_string()),
                alpha: Some(1.0),
                k: Some(5),
                ..Default::default()
            },
            service.default_k(),
            service.default_ef(),
        )
        .unwrap();
        let results_before = service.search(query_before).unwrap();
        assert!(results_before.is_empty(), "Should not find pending documents");

        // Run sync
        let progress = run_sync(&mut service, |_: &JobProgress| {}).unwrap();
        assert_eq!(progress.total, 1);
        assert_eq!(progress.indexed, 1);
        assert_eq!(progress.skipped, 0);

        // Now searchable
        let query_after = ValidatedSearchQuery::from_request(
            SearchRequest {
                text: Some("test".to_string()),
                alpha: Some(1.0),
                k: Some(5),
                ..Default::default()
            },
            service.default_k(),
            service.default_ef(),
        )
        .unwrap();
        let results_after = service.search(query_after).unwrap();
        assert_eq!(results_after.len(), 1);
        assert_eq!(results_after[0].id, "doc1");
    }

    #[test]
    fn test_sync_skips_unchanged_content() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        // First: insert and sync
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Original Title".to_string()),
            body: Some("Original content.".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: None,
        }]).unwrap();
        let first_sync = run_sync(&mut service, |_: &JobProgress| {}).unwrap();
        assert_eq!(first_sync.indexed, 1);

        // Second: re-batch same content
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Original Title".to_string()),
            body: Some("Original content.".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: None,
        }]).unwrap();

        // Sync should skip (same content_hash)
        let second_sync = run_sync(&mut service, |_: &JobProgress| {}).unwrap();
        assert_eq!(second_sync.total, 1);
        assert_eq!(second_sync.skipped, 1);
        assert_eq!(second_sync.indexed, 0);
    }

    #[test]
    fn test_sync_re_embeds_changed_content() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        // First: insert and sync
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Original Title".to_string()),
            body: Some("Original content.".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: None,
        }]).unwrap();
        run_sync(&mut service, |_: &JobProgress| {}).unwrap();

        // Second: batch with different content
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Updated Title".to_string()),
            body: Some("Updated content.".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: None,
        }]).unwrap();

        // Sync should re-embed
        let sync_result = run_sync(&mut service, |_: &JobProgress| {}).unwrap();
        assert_eq!(sync_result.total, 1);
        assert_eq!(sync_result.indexed, 1);
        assert_eq!(sync_result.skipped, 0);
    }

    #[test]
    fn test_within_batch_duplicates_last_wins() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        // Insert same ID twice in one batch
        let result = service.batch_insert(vec![
            BatchDocument {
                id: "doc1".to_string(),
                title: Some("First version".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            },
            BatchDocument {
                id: "doc1".to_string(),
                title: Some("Second version".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            },
        ]).unwrap();

        assert_eq!(result.accepted, 2); // Both accepted, but later overwrites earlier

        // Only one pending
        let pending_ids = service.get_pending_ids();
        assert_eq!(pending_ids.len(), 1);

        // Sync and verify the title is "Second version"
        run_sync(&mut service, |_: &JobProgress| {}).unwrap();

        let query = ValidatedSearchQuery::from_request(
            SearchRequest {
                text: Some("version".to_string()),
                alpha: Some(1.0),
                k: Some(5),
                ..Default::default()
            },
            service.default_k(),
            service.default_ef(),
        )
        .unwrap();
        let results = service.search(query).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, Some("Second version".to_string()));
    }

    #[test]
    fn test_content_hash_tag_order_independent() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        // First sync with tags in one order
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Document".to_string()),
            body: None,
            source: None,
            updated_at: None,
            tags: vec![Tag::new("rust"), Tag::new("ai")],
            text: None,
        }]).unwrap();
        run_sync(&mut service, |_: &JobProgress| {}).unwrap();

        // Re-batch with tags in different order
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Document".to_string()),
            body: None,
            source: None,
            updated_at: None,
            tags: vec![Tag::new("ai"), Tag::new("rust")], // Different order
            text: None,
        }]).unwrap();

        // Should skip because content_hash is same (tags are sorted)
        let sync_result = run_sync(&mut service, |_: &JobProgress| {}).unwrap();
        assert_eq!(sync_result.skipped, 1);
        assert_eq!(sync_result.indexed, 0);
    }

    #[test]
    fn test_explicit_text_is_used_for_embedding() {
        // Verify that when explicit text is provided, it's used for embedding
        // (not title+body)
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut service = create_test_service(tmp_dir.path());

        // Insert with explicit text that differs from title+body
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Title about cats".to_string()),
            body: Some("Body about cats".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: Some("Explicit text about dogs".to_string()), // Different content
        }]).unwrap();

        // Sync
        run_sync(&mut service, |_: &JobProgress| {}).unwrap();

        // Re-batch with same explicit text
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Title about cats".to_string()),
            body: Some("Body about cats".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: Some("Explicit text about dogs".to_string()), // Same explicit text
        }]).unwrap();

        // Should skip because content_hash includes explicit text
        let sync_result = run_sync(&mut service, |_: &JobProgress| {}).unwrap();
        assert_eq!(sync_result.skipped, 1, "Should skip when explicit text is unchanged");
        assert_eq!(sync_result.indexed, 0);

        // Now change only the explicit text
        service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Title about cats".to_string()),
            body: Some("Body about cats".to_string()),
            source: None,
            updated_at: None,
            tags: vec![],
            text: Some("Explicit text about birds".to_string()), // Changed!
        }]).unwrap();

        // Should re-embed because explicit text changed
        let sync_result2 = run_sync(&mut service, |_: &JobProgress| {}).unwrap();
        assert_eq!(sync_result2.indexed, 1, "Should re-embed when explicit text changes");
        assert_eq!(sync_result2.skipped, 0);
    }

    #[test]
    fn test_pending_queue_is_in_memory_only() {
        // Verify that pending queue is in-memory only (not persisted across restarts)
        // This is by design: pending documents must be re-submitted after restart
        let tmp_dir = tempfile::tempdir().unwrap();

        // Use a shared record store to simulate restart
        let record_store = std::sync::Arc::new(MemoryRecordStore::new());

        // First session: batch insert same ID twice (second overwrites first in pending_queue)
        {
            let embedder = Box::new(MockEmbedder::new(8));
            let index_factory = Box::new(MemoryIndexFactory);
            let config = create_test_config(tmp_dir.path());

            let mut service = DocumentService::load(
                config,
                embedder,
                Box::new(RecordStoreWrapper(record_store.clone())),
                index_factory,
                None,
                None,
                None, // No pending_store in test
            )
            .expect("Failed to load service");

            // Insert two records with same ID in separate batches
            service.batch_insert(vec![BatchDocument {
                id: "doc1".to_string(),
                title: Some("First version".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            }]).unwrap();

            service.batch_insert(vec![BatchDocument {
                id: "doc1".to_string(),
                title: Some("Second version".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            }]).unwrap();

            // Should have 1 pending (second batch overwrites first in pending_queue)
            let pending_ids = service.get_pending_ids();
            assert_eq!(pending_ids.len(), 1);
        }

        // Second session: reload - pending queue is lost (by design, no pending_store)
        {
            let embedder = Box::new(MockEmbedder::new(8));
            let index_factory = Box::new(MemoryIndexFactory);
            let config = create_test_config(tmp_dir.path());

            let service = DocumentService::load(
                config,
                embedder,
                Box::new(RecordStoreWrapper(record_store.clone())),
                index_factory,
                None,
                None,
                None, // No pending_store in test
            )
            .expect("Failed to load service");

            // Pending queue is in-memory only when no pending_store is provided
            let pending_ids = service.get_pending_ids();
            assert_eq!(
                pending_ids.len(),
                0,
                "Pending queue should be empty after restart (no pending_store)"
            );
        }
    }

    /// Wrapper to allow Arc<MemoryRecordStore> to implement RecordStore
    struct RecordStoreWrapper(std::sync::Arc<MemoryRecordStore>);

    impl RecordStore for RecordStoreWrapper {
        fn load(&self) -> io::Result<Vec<StoredRecord>> {
            self.0.load()
        }

        fn save_all(&self, records: &[StoredRecord]) -> io::Result<()> {
            self.0.save_all(records)
        }

        fn append(&self, record: &StoredRecord) -> io::Result<()> {
            self.0.append(record)
        }

        fn path(&self) -> &Path {
            self.0.path()
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
                    tags: vec![],
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
                    tags: vec![],
                },
                vec![0.2; 8],
            ),
        ];

        let record_store = Box::new(MemoryRecordStore::with_records(existing_records));
        let chunk_store = Box::new(MemoryChunkStore::new());
        let chunker = Box::new(TestChunker::new(50)) as Box<dyn Chunker>;
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);

        let config = ServiceConfig {
            snapshot_path: tmp_dir.path().join("index.bin"),
            default_k: 5,
            default_ef: 100,
            wal_config: None,
            tombstone_config: TombstoneConfig::default(),
            chunk_config: ChunkConfig::Fixed {
                chunk_size: 100,
                overlap: 0,
            },
            pending_store_path: None, // Not used in chunking mode
        };

        // Load service with chunking enabled but no existing chunks
        // This should trigger migration
        let service = DocumentService::load(
            config,
            embedder,
            record_store,
            index_factory,
            Some(chunker),
            Some(chunk_store),
            None, // No pending_store - not needed for chunking mode
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
        let query = ValidatedSearchQuery::from_request(
            SearchRequest {
                text: Some("document".to_string()),
                alpha: Some(1.0), // Vector only
                k: Some(5),
                ..Default::default()
            },
            service.default_k(),
            service.default_ef(),
        )
        .expect("Validation failed");
        let results = service.search(query).expect("Search failed");

        assert!(
            !results.is_empty(),
            "Search should return results after migration"
        );
    }

    #[test]
    fn test_batch_insert_rejected_when_chunking_enabled() {
        // Verify that batch_insert returns error when chunking is enabled
        let tmp_dir = tempfile::tempdir().unwrap();

        let record_store = Box::new(MemoryRecordStore::new());
        let chunk_store = Box::new(MemoryChunkStore::new());
        let chunker = Box::new(TestChunker::new(50)) as Box<dyn Chunker>;
        let embedder = Box::new(MockEmbedder::new(8));
        let index_factory = Box::new(MemoryIndexFactory);

        let config = ServiceConfig {
            snapshot_path: tmp_dir.path().join("index.bin"),
            default_k: 5,
            default_ef: 100,
            wal_config: None,
            tombstone_config: TombstoneConfig::default(),
            chunk_config: ChunkConfig::Fixed {
                chunk_size: 100,
                overlap: 0,
            },
            pending_store_path: None,
        };

        let mut service = DocumentService::load(
            config,
            embedder,
            record_store,
            index_factory,
            Some(chunker),
            Some(chunk_store),
            None,
        )
        .expect("Failed to load service");

        // batch_insert should return error when chunking is enabled
        let result = service.batch_insert(vec![BatchDocument {
            id: "doc1".to_string(),
            title: Some("Test".to_string()),
            body: None,
            source: None,
            updated_at: None,
            tags: vec![],
            text: None,
        }]);

        assert!(result.is_err(), "batch_insert should fail when chunking is enabled");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("chunking"),
            "Error message should mention chunking: {}",
            err
        );
    }

    #[test]
    fn test_pending_store_persists_across_restart() {
        use crate::domain::ports::PendingStore;
        use crate::infrastructure::storage::pending_store::JsonlPendingStore;

        // Verify that pending documents are persisted and restored
        let tmp_dir = tempfile::tempdir().unwrap();
        let pending_path = tmp_dir.path().join("pending.jsonl");

        // Use a shared record store to simulate restart
        let record_store = std::sync::Arc::new(MemoryRecordStore::new());

        // First session: batch insert a document
        {
            let embedder = Box::new(MockEmbedder::new(8));
            let index_factory = Box::new(MemoryIndexFactory);
            let config = ServiceConfig {
                snapshot_path: tmp_dir.path().join("index.bin"),
                default_k: 5,
                default_ef: 100,
                wal_config: None,
                tombstone_config: TombstoneConfig::default(),
                chunk_config: ChunkConfig::None,
                pending_store_path: Some(pending_path.clone()),
            };
            let pending_store: Box<dyn PendingStore> =
                Box::new(JsonlPendingStore::new(&pending_path));

            let mut service = DocumentService::load(
                config,
                embedder,
                Box::new(RecordStoreWrapper(record_store.clone())),
                index_factory,
                None,
                None,
                Some(pending_store),
            )
            .expect("Failed to load service");

            // Insert a document as pending
            service.batch_insert(vec![BatchDocument {
                id: "doc1".to_string(),
                title: Some("Persisted pending".to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
                text: None,
            }]).unwrap();

            // Verify it's in pending queue
            let pending_ids = service.get_pending_ids();
            assert_eq!(pending_ids.len(), 1);
        }

        // Second session: reload and verify pending was restored
        {
            let embedder = Box::new(MockEmbedder::new(8));
            let index_factory = Box::new(MemoryIndexFactory);
            let config = ServiceConfig {
                snapshot_path: tmp_dir.path().join("index.bin"),
                default_k: 5,
                default_ef: 100,
                wal_config: None,
                tombstone_config: TombstoneConfig::default(),
                chunk_config: ChunkConfig::None,
                pending_store_path: Some(pending_path.clone()),
            };
            let pending_store: Box<dyn PendingStore> =
                Box::new(JsonlPendingStore::new(&pending_path));

            let service = DocumentService::load(
                config,
                embedder,
                Box::new(RecordStoreWrapper(record_store.clone())),
                index_factory,
                None,
                None,
                Some(pending_store),
            )
            .expect("Failed to load service");

            // Pending queue should be restored from pending_store
            let pending_ids = service.get_pending_ids();
            assert_eq!(
                pending_ids.len(),
                1,
                "Pending queue should be restored from pending_store"
            );
            assert!(pending_ids.contains(&"doc1".to_string()));
        }
    }
}
