//! JSONL-based pending document storage.
//!
//! Stores pending documents as JSONL with ID as the first field.
//! Supports upsert and delete operations by rewriting the file.

use crate::domain::model::PendingDocument;
use crate::domain::ports::PendingStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// A pending document entry with its ID for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingEntry {
    id: String,
    #[serde(flatten)]
    doc: PendingDocument,
}

/// JSONL-based pending document storage
pub struct JsonlPendingStore {
    path: PathBuf,
}

impl JsonlPendingStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Load all entries into a map for manipulation
    fn load_map(&self) -> io::Result<HashMap<String, PendingDocument>> {
        if !self.path.exists() {
            return Ok(HashMap::new());
        }
        let file = fs::File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut map = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: PendingEntry = serde_json::from_str(trimmed)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            map.insert(entry.id, entry.doc);
        }
        Ok(map)
    }

    /// Save the map back to the file
    fn save_map(&self, map: &HashMap<String, PendingDocument>) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(&self.path)?;
        for (id, doc) in map {
            let entry = PendingEntry {
                id: id.clone(),
                doc: doc.clone(),
            };
            let line = serde_json::to_string(&entry)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            writeln!(file, "{}", line)?;
        }
        Ok(())
    }
}

impl PendingStore for JsonlPendingStore {
    fn load(&self) -> io::Result<Vec<(String, PendingDocument)>> {
        let map = self.load_map()?;
        Ok(map.into_iter().collect())
    }

    fn save_all(&self, pending: &[(String, PendingDocument)]) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(&self.path)?;
        for (id, doc) in pending {
            let entry = PendingEntry {
                id: id.clone(),
                doc: doc.clone(),
            };
            let line = serde_json::to_string(&entry)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            writeln!(file, "{}", line)?;
        }
        Ok(())
    }

    fn upsert(&self, id: &str, doc: &PendingDocument) -> io::Result<()> {
        let mut map = self.load_map()?;
        map.insert(id.to_string(), doc.clone());
        self.save_map(&map)
    }

    fn delete(&self, id: &str) -> io::Result<()> {
        let mut map = self.load_map()?;
        map.remove(id);
        self.save_map(&map)
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::model::Record;
    use tempfile::tempdir;

    fn make_pending(id: &str, title: &str) -> PendingDocument {
        PendingDocument {
            record: Record {
                id: id.to_string(),
                title: Some(title.to_string()),
                body: None,
                source: None,
                updated_at: None,
                tags: vec![],
            },
            content_hash: format!("hash_{}", id),
            embed_text: None,
        }
    }

    #[test]
    fn test_pending_store_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pending.jsonl");
        let store = JsonlPendingStore::new(&path);

        let pending = vec![
            ("doc1".to_string(), make_pending("doc1", "First")),
            ("doc2".to_string(), make_pending("doc2", "Second")),
        ];

        store.save_all(&pending).unwrap();
        let loaded = store.load().unwrap();

        assert_eq!(loaded.len(), 2);
        let map: HashMap<_, _> = loaded.into_iter().collect();
        assert_eq!(
            map.get("doc1").unwrap().record.title,
            Some("First".to_string())
        );
        assert_eq!(
            map.get("doc2").unwrap().record.title,
            Some("Second".to_string())
        );
    }

    #[test]
    fn test_pending_store_upsert() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pending.jsonl");
        let store = JsonlPendingStore::new(&path);

        // Insert first document
        store
            .upsert("doc1", &make_pending("doc1", "First"))
            .unwrap();

        // Insert second document
        store
            .upsert("doc2", &make_pending("doc2", "Second"))
            .unwrap();

        // Update first document
        store
            .upsert("doc1", &make_pending("doc1", "Updated First"))
            .unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.len(), 2);

        let map: HashMap<_, _> = loaded.into_iter().collect();
        assert_eq!(
            map.get("doc1").unwrap().record.title,
            Some("Updated First".to_string())
        );
        assert_eq!(
            map.get("doc2").unwrap().record.title,
            Some("Second".to_string())
        );
    }

    #[test]
    fn test_pending_store_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pending.jsonl");
        let store = JsonlPendingStore::new(&path);

        let pending = vec![
            ("doc1".to_string(), make_pending("doc1", "First")),
            ("doc2".to_string(), make_pending("doc2", "Second")),
        ];
        store.save_all(&pending).unwrap();

        // Delete doc1
        store.delete("doc1").unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "doc2");
    }

    #[test]
    fn test_pending_store_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.jsonl");
        let store = JsonlPendingStore::new(&path);

        let loaded = store.load().unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_pending_store_delete_nonexistent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pending.jsonl");
        let store = JsonlPendingStore::new(&path);

        // Delete from empty store should succeed (no-op)
        store.delete("nonexistent").unwrap();
    }
}
