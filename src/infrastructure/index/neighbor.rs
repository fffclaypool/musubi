use crate::domain::types::Vector;
use crate::infrastructure::index::metrics::cosine_distance;

/// 多様性を保つヒューリスティックで近傍を選択
pub fn select_neighbors_heuristic(
    vectors: &[Vector],
    candidates: &[(usize, f32)],
    m_max: usize,
) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::with_capacity(m_max);
    for &(candidate_id, candidate_dist) in candidates {
        let mut good = true;
        for &selected_id in &selected {
            let dist = cosine_distance(&vectors[candidate_id], &vectors[selected_id]);
            if dist < candidate_dist {
                good = false;
                break;
            }
        }
        if good {
            selected.push(candidate_id);
            if selected.len() >= m_max {
                break;
            }
        }
    }

    if selected.is_empty() {
        candidates.iter().take(m_max).map(|&(id, _)| id).collect()
    } else {
        selected
    }
}
