//! Search operations for DocumentService.

use std::collections::{HashMap, HashSet};

use crate::application::error::AppError;

use super::core::DocumentService;
use super::traits::DocumentSearch;
use super::types::{
    ChunkInfo, ScoreBreakdown, SearchHit, SearchInput, SearchMode, ValidatedSearchQuery,
};
use super::util::create_text_preview;

/// Vector search result: (record_idx -> distance, record_idx -> (chunk_idx, distance))
type VectorSearchResult = (HashMap<usize, f32>, HashMap<usize, (usize, f32)>);

/// Normalization factors for hybrid scoring
struct Normalization {
    max_bm25: f64,
    min_distance: f32,
    max_distance: f32,
}

/// Scored candidate with index and score breakdown
struct ScoredCandidate {
    record_idx: usize,
    score: ScoreBreakdown,
}

impl DocumentSearch for DocumentService {
    fn search(&self, query: ValidatedSearchQuery) -> Result<Vec<SearchHit>, AppError> {
        let k = query.params.k;
        let mode = query.params.mode;
        let search_k = (k * 4).max(100);
        let search_ef = query.params.ef.max(search_k);

        let chunking_enabled = self.indexing_mode.is_chunked_with_data();

        // Get vector search results based on input type and mode
        let (vector_scores, chunk_info_map) =
            self.execute_vector_search(&query.input, mode, search_k, search_ef, chunking_enabled)?;

        // Get BM25 results (document-level)
        let bm25_scores = self.execute_bm25_search(&query.input, mode, search_k);

        // Combine and score results
        let hits = self.combine_and_score_results(
            vector_scores,
            bm25_scores,
            chunk_info_map,
            mode,
            query.filter.as_ref(),
            chunking_enabled,
            k,
        );

        Ok(hits)
    }
}

// Private helper methods for search operations
impl DocumentService {
    /// Execute vector search based on input and mode
    fn execute_vector_search(
        &self,
        input: &SearchInput,
        mode: SearchMode,
        search_k: usize,
        search_ef: usize,
        chunking_enabled: bool,
    ) -> Result<VectorSearchResult, AppError> {
        if !mode.needs_vector() {
            return Ok((HashMap::new(), HashMap::new()));
        }

        // Get or compute embedding
        let embedding = match input {
            SearchInput::Embedding(e) | SearchInput::TextAndEmbedding { embedding: e, .. } => {
                e.clone()
            }
            SearchInput::Text(text) => self.embed_single(text.clone())?,
        };

        let vector_results = self.index.search(&embedding, search_k, search_ef);

        if chunking_enabled {
            Ok(self.aggregate_chunk_results(vector_results))
        } else {
            Ok(self.direct_record_results(vector_results))
        }
    }

    /// Aggregate chunk-level results by parent document
    fn aggregate_chunk_results(
        &self,
        vector_results: Vec<crate::domain::types::SearchResult>,
    ) -> (HashMap<usize, f32>, HashMap<usize, (usize, f32)>) {
        let mut parent_scores: HashMap<usize, f32> = HashMap::new();
        let mut parent_best_chunk: HashMap<usize, (usize, f32)> = HashMap::new();
        let chunks = self.indexing_mode.chunks();
        let chunk_mapping = self.indexing_mode.chunk_mapping();

        for result in vector_results {
            // Skip deleted chunks
            if chunks.get(result.id).is_some_and(|c| c.deleted) {
                continue;
            }

            if let Some(&(record_idx, chunk_idx)) = chunk_mapping.get(result.id) {
                // Skip invalid mappings (usize::MAX sentinel)
                if record_idx == usize::MAX {
                    continue;
                }

                if let Some(record) = self.records.get(record_idx) {
                    if !record.deleted {
                        // Use minimum distance (best match) for the parent
                        let current_dist = parent_scores.entry(record_idx).or_insert(f32::MAX);
                        if result.distance < *current_dist {
                            *current_dist = result.distance;
                            parent_best_chunk.insert(record_idx, (chunk_idx, result.distance));
                        }
                    }
                }
            }
        }

        (parent_scores, parent_best_chunk)
    }

    /// Direct record lookup for non-chunking mode
    fn direct_record_results(
        &self,
        vector_results: Vec<crate::domain::types::SearchResult>,
    ) -> (HashMap<usize, f32>, HashMap<usize, (usize, f32)>) {
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

    /// Execute BM25 search if needed
    fn execute_bm25_search(
        &self,
        input: &SearchInput,
        mode: SearchMode,
        search_k: usize,
    ) -> HashMap<usize, f64> {
        if !mode.needs_bm25() {
            return HashMap::new();
        }

        match input.text() {
            Some(text) => self
                .bm25_index
                .search(text, search_k)
                .into_iter()
                .filter(|(id, _)| self.records.get(*id).is_some_and(|r| !r.deleted))
                .collect(),
            None => HashMap::new(),
        }
    }

    /// Combine vector and BM25 results, compute hybrid scores, and return sorted hits
    #[allow(clippy::too_many_arguments)]
    fn combine_and_score_results(
        &self,
        vector_scores: HashMap<usize, f32>,
        bm25_scores: HashMap<usize, f64>,
        chunk_info_map: HashMap<usize, (usize, f32)>,
        mode: SearchMode,
        filter: Option<&super::types::SearchFilter>,
        chunking_enabled: bool,
        k: usize,
    ) -> Vec<SearchHit> {
        // Phase 1: Collect candidates
        let candidates = self.collect_candidates(&vector_scores, &bm25_scores, mode, filter);

        // Phase 2: Score candidates
        let normalization = self.compute_normalization(&vector_scores, &bm25_scores);
        let scored = self.score_candidates(
            candidates,
            &vector_scores,
            &bm25_scores,
            mode,
            &normalization,
        );

        // Phase 3: Build hits with metadata
        self.build_hits(scored, chunk_info_map, chunking_enabled, k)
    }

    /// Phase 1: Collect candidate record indices based on mode and filter
    fn collect_candidates(
        &self,
        vector_scores: &HashMap<usize, f32>,
        bm25_scores: &HashMap<usize, f64>,
        mode: SearchMode,
        filter: Option<&super::types::SearchFilter>,
    ) -> Vec<usize> {
        let all_candidates: HashSet<usize> = match mode {
            SearchMode::VectorOnly => vector_scores.keys().copied().collect(),
            SearchMode::Bm25Only => bm25_scores.keys().copied().collect(),
            SearchMode::Hybrid { .. } => vector_scores
                .keys()
                .chain(bm25_scores.keys())
                .copied()
                .collect(),
        };

        all_candidates
            .into_iter()
            .filter(|&record_idx| {
                self.records
                    .get(record_idx)
                    .is_some_and(|r| !r.deleted && r.indexed && filter.is_none_or(|f| f.matches(&r.record)))
            })
            .collect()
    }

    /// Phase 2: Compute scores for each candidate
    fn score_candidates(
        &self,
        candidates: Vec<usize>,
        vector_scores: &HashMap<usize, f32>,
        bm25_scores: &HashMap<usize, f64>,
        mode: SearchMode,
        normalization: &Normalization,
    ) -> Vec<ScoredCandidate> {
        candidates
            .into_iter()
            .filter_map(|record_idx| {
                let distance = vector_scores.get(&record_idx).copied();
                let bm25_raw = bm25_scores.get(&record_idx).copied();
                let score = self.build_score_breakdown(mode, distance, bm25_raw, normalization)?;
                Some(ScoredCandidate { record_idx, score })
            })
            .collect()
    }

    /// Phase 3: Build final SearchHit results
    fn build_hits(
        &self,
        mut scored: Vec<ScoredCandidate>,
        chunk_info_map: HashMap<usize, (usize, f32)>,
        chunking_enabled: bool,
        k: usize,
    ) -> Vec<SearchHit> {
        // Sort by ranking score descending
        scored.sort_by(|a, b| {
            b.score
                .ranking_score()
                .partial_cmp(&a.score.ranking_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);

        scored
            .into_iter()
            .filter_map(|candidate| {
                let record = self.records.get(candidate.record_idx)?;

                let best_chunk = if chunking_enabled {
                    chunk_info_map
                        .get(&candidate.record_idx)
                        .map(|&(chunk_idx, _)| {
                            // O(1) chunk lookup via chunk_index
                            let text_preview = self
                                .indexing_mode
                                .get_chunk(&record.record.id, chunk_idx)
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
                    index_id: candidate.record_idx,
                    id: record.record.id.clone(),
                    score: candidate.score,
                    title: record.record.title.clone(),
                    source: record.record.source.clone(),
                    tags: record.record.tags.clone(),
                    best_chunk,
                })
            })
            .collect()
    }

    /// Compute normalization factors for hybrid scoring
    fn compute_normalization(
        &self,
        vector_scores: &HashMap<usize, f32>,
        bm25_scores: &HashMap<usize, f64>,
    ) -> Normalization {
        Normalization {
            max_bm25: bm25_scores.values().cloned().fold(0.0f64, f64::max),
            min_distance: vector_scores.values().cloned().fold(f32::MAX, f32::min),
            max_distance: vector_scores.values().cloned().fold(0.0f32, f32::max),
        }
    }

    /// Build ScoreBreakdown based on search mode and available scores
    fn build_score_breakdown(
        &self,
        mode: SearchMode,
        distance: Option<f32>,
        bm25_raw: Option<f64>,
        normalization: &Normalization,
    ) -> Option<ScoreBreakdown> {
        match mode {
            SearchMode::VectorOnly => {
                // VectorOnly requires distance
                let distance = distance?;
                Some(ScoreBreakdown::VectorOnly { distance })
            }
            SearchMode::Bm25Only => {
                // Bm25Only requires bm25_score
                let bm25_score = bm25_raw?;
                Some(ScoreBreakdown::Bm25Only { bm25_score })
            }
            SearchMode::Hybrid { alpha } => {
                // Normalize vector score: None -> 0.0, Some(d) -> normalized similarity
                let vector_score = distance.map_or(0.0, |d| {
                    if normalization.max_distance > normalization.min_distance {
                        1.0 - ((d - normalization.min_distance)
                            / (normalization.max_distance - normalization.min_distance))
                            as f64
                    } else {
                        1.0 - d as f64
                    }
                });

                // Normalize BM25 score: None -> 0.0, Some(s) -> normalized score
                let bm25_normalized = bm25_raw.map_or(0.0, |s| {
                    if normalization.max_bm25 > 0.0 {
                        s / normalization.max_bm25
                    } else {
                        0.0
                    }
                });

                // Compute hybrid score
                let hybrid_score = alpha * vector_score + (1.0 - alpha) * bm25_normalized;

                Some(ScoreBreakdown::Hybrid {
                    distance,
                    bm25_score: bm25_raw,
                    hybrid_score,
                })
            }
        }
    }
}
