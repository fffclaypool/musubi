use crate::domain::model::{Chunk, StoredChunk, StoredRecord};
use crate::domain::types::{SearchResult, Vector};
use std::io;
use std::path::Path;

pub trait Embedder: Send + Sync {
    fn embed(&self, texts: Vec<String>) -> io::Result<Vec<Vector>>;
}

pub trait VectorIndex: Send + Sync {
    fn insert(&mut self, vector: Vector) -> usize;
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult>;
    fn vector(&self, id: usize) -> Option<&Vector>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn dim(&self) -> usize;
    fn save(&self, path: &Path) -> io::Result<()>;
}

pub trait VectorIndexFactory: Send + Sync {
    fn load_or_create(
        &self,
        snapshot_path: &Path,
        records: &[StoredRecord],
    ) -> io::Result<Box<dyn VectorIndex>>;
    fn rebuild(&self, records: &[StoredRecord]) -> Box<dyn VectorIndex>;
}

pub trait RecordStore: Send + Sync {
    fn load(&self) -> io::Result<Vec<StoredRecord>>;
    fn append(&self, record: &StoredRecord) -> io::Result<()>;
    fn save_all(&self, records: &[StoredRecord]) -> io::Result<()>;
    fn path(&self) -> &Path;
}

/// Trait for splitting text into chunks
pub trait Chunker: Send + Sync {
    /// Split the given text into chunks
    fn chunk(&self, text: &str) -> Vec<Chunk>;
}

/// Trait for storing and loading chunks
pub trait ChunkStore: Send + Sync {
    /// Load all stored chunks
    fn load(&self) -> io::Result<Vec<StoredChunk>>;
    /// Save all chunks (overwrites existing)
    fn save_all(&self, chunks: &[StoredChunk]) -> io::Result<()>;
    /// Append a single chunk
    fn append(&self, chunk: &StoredChunk) -> io::Result<()>;
    /// Get the storage path
    fn path(&self) -> &Path;
}

use crate::domain::model::PendingDocument;

/// Trait for storing pending documents awaiting sync
pub trait PendingStore: Send + Sync {
    /// Load all pending documents
    fn load(&self) -> io::Result<Vec<(String, PendingDocument)>>;
    /// Save all pending documents (overwrites existing)
    fn save_all(&self, pending: &[(String, PendingDocument)]) -> io::Result<()>;
    /// Append or update a single pending document
    fn upsert(&self, id: &str, doc: &PendingDocument) -> io::Result<()>;
    /// Delete a pending document by ID
    fn delete(&self, id: &str) -> io::Result<()>;
    /// Get the storage path
    fn path(&self) -> &Path;
}
