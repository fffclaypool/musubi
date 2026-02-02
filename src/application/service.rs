use crate::application::error::AppError;
use crate::domain::model::{Record, StoredRecord};
use crate::domain::ports::{Embedder, RecordStore, VectorIndex, VectorIndexFactory};
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
    bm25_index: Bm25Index,
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

        // Build BM25 index from records (excluding tombstones)
        let mut bm25_index = Bm25Index::new();
        for (idx, record) in records.iter().enumerate() {
            if !record.deleted {
                if let Some(text) = build_text(None, &record.record) {
                    bm25_index.add(idx, &text);
                }
            }
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
            bm25_index,
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

            // Update BM25: remove old, add new
            self.bm25_index.remove(index_id);

            // Append new record and add to index
            let new_index_id = self.index.insert(embedding.clone());
            self.records.push(new_stored.clone());

            // Add to BM25 index
            if let Some(text) = build_text(None, &updated) {
                self.bm25_index.add(new_index_id, &text);
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
        } else {
            // Only metadata changed: update in place (no index change needed)
            let stored = StoredRecord::new(updated.clone(), embedding.clone());

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

        // Get vector search results (skip if alpha=0.0, BM25-only)
        let vector_scores: HashMap<usize, f32> = if alpha == 0.0 {
            HashMap::new()
        } else {
            let embedding = if let Some(embedding) = cmd.embedding.clone() {
                embedding
            } else if let Some(ref text) = query_text {
                self.embed_single(text.clone())?
            } else {
                // This shouldn't happen due to validation above
                return Err(AppError::BadRequest("text or embedding is required".to_string()));
            };

            let vector_results = self.index.search(&embedding, search_k, search_ef);
            vector_results
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
                .collect()
        };

        // Get BM25 results (skip if alpha=1.0 and no text, or if no text provided)
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
        // Vector: convert distance to similarity (1 - distance), then normalize
        // BM25: normalize to [0, 1]
        let max_bm25 = bm25_scores.values().cloned().fold(0.0f64, f64::max);
        let min_distance = vector_scores.values().cloned().fold(f32::MAX, f32::min);
        let max_distance = vector_scores.values().cloned().fold(0.0f32, f32::max);

        let mut hits: Vec<SearchHit> = all_candidates
            .into_iter()
            .filter_map(|index_id| {
                let record = self.records.get(index_id)?;
                if record.deleted {
                    return None;
                }

                // Apply metadata filter if present
                if let Some(ref filter) = cmd.filter {
                    if !filter.matches(&record.record) {
                        return None;
                    }
                }

                let distance = vector_scores.get(&index_id).copied();
                let bm25_raw = bm25_scores.get(&index_id).copied();

                // Normalize vector score: convert distance to [0, 1] similarity
                // BM25-only candidates (no distance) get vector_score = 0.0 for fair hybrid ranking
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

                Some(SearchHit {
                    index_id,
                    id: record.record.id.clone(),
                    distance,
                    bm25_score: bm25_raw,
                    hybrid_score: Some(hybrid_score),
                    title: record.record.title.clone(),
                    source: record.record.source.clone(),
                    tags: record.record.tags.clone(),
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

        // Rebuild vector index with only active records
        self.index = self.index_factory.rebuild(&self.records);

        // Rebuild BM25 index
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
