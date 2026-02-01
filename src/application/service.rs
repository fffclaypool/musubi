use crate::application::error::AppError;
use crate::domain::model::{Record, StoredRecord};
use crate::domain::ports::{Embedder, RecordStore, VectorIndex, VectorIndexFactory};
use crate::infrastructure::storage::wal::{self, WalConfig, WalWriter};
use serde::Serialize;
use std::collections::HashSet;
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

pub struct ServiceConfig {
    pub snapshot_path: PathBuf,
    pub default_k: usize,
    pub default_ef: usize,
    pub wal_config: Option<WalConfig>,
    pub tombstone_config: TombstoneConfig,
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

#[derive(Debug, Clone)]
pub struct SearchCommand {
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub k: Option<usize>,
    pub ef: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub index_id: usize,
    pub id: String,
    pub distance: f32,
    pub title: Option<String>,
    pub source: Option<String>,
    pub tags: Option<String>,
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
    ) -> Result<Self, AppError> {
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

        // Load or rebuild index from records (including tombstones - they still have vectors)
        let index = if wal_had_ops {
            // WAL had operations: always rebuild index from updated records
            index_factory.rebuild(&records)
        } else if config.snapshot_path.exists() {
            // No WAL ops: load snapshot
            index_factory.load_or_create(&config.snapshot_path, &records)?
        } else {
            // No snapshot: build from records
            index_factory.rebuild(&records)
        };

        let updated = fill_missing_embeddings(&mut records, index.as_ref());
        if updated {
            let _ = record_store.save_all(&records);
        }

        if !records.is_empty() && records.len() != index.len() {
            return Err(AppError::Io(format!(
                "records count ({}) does not match index count ({})",
                records.len(),
                index.len()
            )));
        }

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
        };

        // Only truncate WAL if we successfully replayed v2 operations
        // (v1 WAL can't be replayed, so wal_had_ops would be false)
        if wal_had_ops {
            service.index.save(&service.snapshot_path)?;
            if let Some(ref mut wal) = service.wal {
                wal.truncate()?;
            }
        }

        Ok(service)
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
        let embedding = self.embed_single(text)?;

        let stored = StoredRecord::new(cmd.record.clone(), embedding.clone());

        // Write to WAL first
        if let Some(ref mut wal) = self.wal {
            wal.append_insert(&stored)?;
        }

        // Then update index and records
        let index_id = self.index.insert(embedding.clone());
        self.records.push(stored.clone());
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

        let embedding = if needs_embedding {
            let text = if let Some(text) = cmd.text {
                if text.trim().is_empty() {
                    return Err(AppError::BadRequest("text must not be empty".to_string()));
                }
                text
            } else {
                build_text(None, &updated).ok_or_else(|| {
                    AppError::BadRequest("title/body is required to build embedding".to_string())
                })?
            };
            self.embed_single(text)?
        } else {
            current.embedding.clone()
        };

        if needs_embedding {
            // Embedding changed: tombstone old record, append new one
            let new_stored = StoredRecord::new(updated.clone(), embedding.clone());

            // Write to WAL first: DELETE old, then APPEND new (idempotent replay)
            if let Some(ref mut wal) = self.wal {
                wal.append_delete(&current.record.id)?;
                wal.append_append(&new_stored)?;
            }

            // Mark old record as deleted (tombstone)
            self.records[index_id].deleted = true;
            self.tombstone_count += 1;

            // Append new record and add to index
            let new_index_id = self.index.insert(embedding.clone());
            self.records.push(new_stored.clone());

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
        } else {
            // Only metadata changed: update in place (no index change needed)
            let stored = StoredRecord::new(updated.clone(), embedding.clone());

            // Write to WAL first
            if let Some(ref mut wal) = self.wal {
                wal.append_update(&stored)?;
            }

            // Update record in place
            self.records[index_id] = stored;
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
        let embedding = if let Some(embedding) = cmd.embedding {
            embedding
        } else if let Some(text) = cmd.text {
            self.embed_single(text)?
        } else {
            return Err(AppError::BadRequest("text or embedding is required".to_string()));
        };

        // Request more results to account for tombstones
        let mut search_k = k;
        let mut search_ef = ef;
        let mut hits = Vec::new();

        // Retry with larger k if we don't get enough non-deleted results
        loop {
            let results = self.index.search(&embedding, search_k, search_ef);
            hits.clear();

            for result in results {
                if let Some(record) = self.records.get(result.id) {
                    // Skip tombstones
                    if record.deleted {
                        continue;
                    }
                    hits.push(SearchHit {
                        index_id: result.id,
                        id: record.record.id.clone(),
                        distance: result.distance,
                        title: record.record.title.clone(),
                        source: record.record.source.clone(),
                        tags: record.record.tags.clone(),
                    });
                    if hits.len() >= k {
                        break;
                    }
                }
            }

            // If we have enough results or we've already expanded significantly, stop
            if hits.len() >= k || search_k >= k * 4 {
                break;
            }

            // Expand search to get more candidates
            search_k = search_k * 2;
            search_ef = search_ef.max(search_k);
        }

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

        // Rebuild index with only active records
        self.index = self.index_factory.rebuild(&self.records);
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
