use crate::domain::types::Vector;

/// コサイン距離を計算する (ベクトルは正規化済み前提)
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// ベクトルをL2正規化する
pub fn normalize(vector: &[f32]) -> Option<Vector> {
    let norm_sq: f32 = vector.iter().map(|v| v * v).sum();
    if norm_sq == 0.0 {
        return None;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    Some(vector.iter().map(|v| v * inv_norm).collect())
}
