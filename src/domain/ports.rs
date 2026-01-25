use crate::domain::model::StoredRecord;
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
