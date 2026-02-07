//! Search operations for DocumentService.

use std::collections::{HashMap, HashSet};

use crate::application::error::AppError;

use super::core::DocumentService;
use super::types::{ChunkInfo, SearchCommand, SearchHit, SearchMode};
use super::util::create_text_preview;

impl DocumentService {
    /// Search for documents matching the query.
    pub fn search(&self, cmd: SearchCommand) -> Result<Vec<SearchHit>, AppError> {
        let k = cmd.k.unwrap_or(self.default_k);
        let ef = cmd.ef.unwrap_or(self.default_ef);

        // Parse and validate alpha into SearchMode
        let search_mode = SearchMode::from_alpha(cmd.alpha.unwrap_or(0.7))
            .map_err(|e| AppError::BadRequest(e.to_string()))?;

        let query_text = cmd.text.clone();

        // Validate parameters based on search mode
        match search_mode {
            SearchMode::Bm25Only if query_text.is_none() => {
                return Err(AppError::BadRequest(
                    "alpha=0.0 (BM25-only) requires 'text' parameter".to_string(),
                ));
            }
            SearchMode::VectorOnly | SearchMode::Hybrid { .. }
                if query_text.is_none() && cmd.embedding.is_none() =>
            {
                return Err(AppError::BadRequest(
                    "text or embedding is required".to_string(),
                ));
            }
            _ => {}
        }

        let search_k = (k * 4).max(100);
        let search_ef = ef.max(search_k);

        // Check if chunking is enabled
        let chunking_enabled = self.indexing_mode.is_chunked_with_data();

        // Get vector search results (skip if BM25-only)
        // When chunking is enabled, this returns chunk-level results that need to be aggregated
        let (vector_scores, chunk_info_map): (HashMap<usize, f32>, HashMap<usize, (usize, f32)>) =
            if !search_mode.needs_vector() {
                (HashMap::new(), HashMap::new())
            } else {
                let embedding = if let Some(embedding) = cmd.embedding.clone() {
                    embedding
                } else if let Some(ref text) = query_text {
                    self.embed_single(text.clone())?
                } else {
                    return Err(AppError::BadRequest(
                        "text or embedding is required".to_string(),
                    ));
                };

                let vector_results = self.index.search(&embedding, search_k, search_ef);

                if chunking_enabled {
                    // Chunking mode: aggregate chunk results by parent document
                    // chunk_mapping[index_id] = (record_idx, chunk_idx)
                    let mut parent_scores: HashMap<usize, f32> = HashMap::new();
                    let mut parent_best_chunk: HashMap<usize, (usize, f32)> = HashMap::new();
                    let chunks = self.indexing_mode.chunks();
                    let chunk_mapping = self.indexing_mode.chunk_mapping();

                    for result in vector_results {
                        // Check if the chunk itself is deleted
                        if let Some(chunk) = chunks.get(result.id) {
                            if chunk.deleted {
                                continue;
                            }
                        }

                        if let Some(&(record_idx, chunk_idx)) = chunk_mapping.get(result.id) {
                            // Skip invalid mappings (usize::MAX sentinel)
                            if record_idx == usize::MAX {
                                continue;
                            }
                            if let Some(record) = self.records.get(record_idx) {
                                if !record.deleted {
                                    // Use minimum distance (best match) for the parent
                                    let current_dist =
                                        parent_scores.entry(record_idx).or_insert(f32::MAX);
                                    if result.distance < *current_dist {
                                        *current_dist = result.distance;
                                        parent_best_chunk
                                            .insert(record_idx, (chunk_idx, result.distance));
                                    }
                                }
                            }
                        }
                    }
                    (parent_scores, parent_best_chunk)
                } else {
                    // Non-chunking mode: direct record lookup
                    let scores: HashMap<usize, f32> = vector_results
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
                        .collect();
                    (scores, HashMap::new())
                }
            };

        // Get BM25 results (document-level, not chunk-level)
        let bm25_scores: HashMap<usize, f64> = match (&query_text, search_mode.needs_bm25()) {
            (Some(text), true) => self
                .bm25_index
                .search(text, search_k)
                .into_iter()
                .filter(|(id, _)| self.records.get(*id).is_some_and(|r| !r.deleted))
                .collect(),
            _ => HashMap::new(),
        };

        // Combine candidates based on search mode
        let all_candidates: HashSet<usize> = match search_mode {
            SearchMode::VectorOnly => vector_scores.keys().copied().collect(),
            SearchMode::Bm25Only => bm25_scores.keys().copied().collect(),
            SearchMode::Hybrid { .. } => vector_scores
                .keys()
                .chain(bm25_scores.keys())
                .copied()
                .collect(),
        };

        // Normalize scores and compute hybrid score
        let max_bm25 = bm25_scores.values().cloned().fold(0.0f64, f64::max);
        let min_distance = vector_scores.values().cloned().fold(f32::MAX, f32::min);
        let max_distance = vector_scores.values().cloned().fold(0.0f32, f32::max);

        let mut hits: Vec<SearchHit> = all_candidates
            .into_iter()
            .filter_map(|record_idx| {
                let record = self.records.get(record_idx)?;
                if record.deleted {
                    return None;
                }

                // Apply metadata filter if present
                if let Some(ref filter) = cmd.filter {
                    if !filter.matches(&record.record) {
                        return None;
                    }
                }

                let distance = vector_scores.get(&record_idx).copied();
                let bm25_raw = bm25_scores.get(&record_idx).copied();

                // Normalize vector score
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
                    if max_bm25 > 0.0 {
                        bm25 / max_bm25
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // Hybrid score: alpha * vector + (1 - alpha) * bm25
                let alpha = search_mode.alpha();
                let hybrid_score = alpha * vector_score + (1.0 - alpha) * bm25_score;

                // Build best_chunk info if chunking is enabled
                let best_chunk = if chunking_enabled {
                    chunk_info_map.get(&record_idx).map(|&(chunk_idx, _)| {
                        // Find the chunk text for preview
                        let text_preview = self
                            .indexing_mode
                            .chunks()
                            .iter()
                            .find(|c| {
                                c.parent_id == record.record.id && c.chunk.chunk_index == chunk_idx
                            })
                            .map(|c| create_text_preview(&c.chunk.text, 100));
                        ChunkInfo {
                            chunk_index: chunk_idx,
                            text_preview,
                        }
                    })
                } else {
                    None
                };

                Some(SearchHit {
                    index_id: record_idx,
                    id: record.record.id.clone(),
                    distance,
                    bm25_score: bm25_raw,
                    hybrid_score: Some(hybrid_score),
                    title: record.record.title.clone(),
                    source: record.record.source.clone(),
                    tags: record.record.tags.clone(),
                    best_chunk,
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
}
