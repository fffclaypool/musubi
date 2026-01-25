use crate::domain::model::StoredRecord;
use crate::domain::ports::{VectorIndex, VectorIndexFactory};
use crate::infrastructure::index::HnswIndex;
use std::io;
use std::path::Path;

pub struct HnswIndexFactory {
    m: usize,
    ef_construction: usize,
}

impl HnswIndexFactory {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self { m, ef_construction }
    }
}

impl VectorIndexFactory for HnswIndexFactory {
    fn load_or_create(
        &self,
        snapshot_path: &Path,
        records: &[StoredRecord],
    ) -> io::Result<Box<dyn VectorIndex>> {
        if snapshot_path.exists() {
            let index = HnswIndex::load(snapshot_path)?;
            return Ok(Box::new(index));
        }

        if !records.is_empty() && records.iter().any(|record| record.embedding.is_empty()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "records are missing embeddings and no snapshot exists",
            ));
        }

        Ok(self.rebuild(records))
    }

    fn rebuild(&self, records: &[StoredRecord]) -> Box<dyn VectorIndex> {
        let mut index = HnswIndex::new(self.m, self.ef_construction);
        for record in records {
            index.insert(record.embedding.clone());
        }
        Box::new(index)
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, vector: Vec<f32>) -> usize {
        HnswIndex::insert(self, vector)
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<crate::domain::types::SearchResult> {
        HnswIndex::search(self, query, k, ef)
    }

    fn vector(&self, id: usize) -> Option<&Vec<f32>> {
        HnswIndex::vector(self, id)
    }

    fn len(&self) -> usize {
        HnswIndex::len(self)
    }

    fn dim(&self) -> usize {
        HnswIndex::dim(self)
    }

    fn save(&self, path: &Path) -> io::Result<()> {
        HnswIndex::save(self, path)
    }
}
