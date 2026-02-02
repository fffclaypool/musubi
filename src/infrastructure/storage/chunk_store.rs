use crate::domain::model::StoredChunk;
use crate::domain::ports::ChunkStore;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// JSONL-based chunk storage
pub struct JsonlChunkStore {
    path: PathBuf,
}

impl JsonlChunkStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl ChunkStore for JsonlChunkStore {
    fn load(&self) -> io::Result<Vec<StoredChunk>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = fs::File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut chunks = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let chunk: StoredChunk = serde_json::from_str(trimmed)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            chunks.push(chunk);
        }
        Ok(chunks)
    }

    fn save_all(&self, chunks: &[StoredChunk]) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(&self.path)?;
        for chunk in chunks {
            let line = serde_json::to_string(chunk)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            writeln!(file, "{}", line)?;
        }
        Ok(())
    }

    fn append(&self, chunk: &StoredChunk) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let line = serde_json::to_string(chunk)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::model::Chunk;
    use tempfile::tempdir;

    #[test]
    fn test_chunk_store_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chunks.jsonl");
        let store = JsonlChunkStore::new(&path);

        let chunks = vec![
            StoredChunk::new(
                "doc1".to_string(),
                Chunk {
                    chunk_index: 0,
                    text: "First chunk".to_string(),
                    start_offset: 0,
                    end_offset: 11,
                },
                vec![0.1, 0.2, 0.3],
            ),
            StoredChunk::new(
                "doc1".to_string(),
                Chunk {
                    chunk_index: 1,
                    text: "Second chunk".to_string(),
                    start_offset: 12,
                    end_offset: 24,
                },
                vec![0.4, 0.5, 0.6],
            ),
        ];

        store.save_all(&chunks).unwrap();
        let loaded = store.load().unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].parent_id, "doc1");
        assert_eq!(loaded[0].chunk.chunk_index, 0);
        assert_eq!(loaded[0].chunk.text, "First chunk");
        assert_eq!(loaded[1].chunk.chunk_index, 1);
    }

    #[test]
    fn test_chunk_store_append() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chunks.jsonl");
        let store = JsonlChunkStore::new(&path);

        let chunk1 = StoredChunk::new(
            "doc1".to_string(),
            Chunk {
                chunk_index: 0,
                text: "First".to_string(),
                start_offset: 0,
                end_offset: 5,
            },
            vec![0.1],
        );
        let chunk2 = StoredChunk::new(
            "doc1".to_string(),
            Chunk {
                chunk_index: 1,
                text: "Second".to_string(),
                start_offset: 6,
                end_offset: 12,
            },
            vec![0.2],
        );

        store.append(&chunk1).unwrap();
        store.append(&chunk2).unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_chunk_store_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.jsonl");
        let store = JsonlChunkStore::new(&path);

        let loaded = store.load().unwrap();
        assert!(loaded.is_empty());
    }
}
