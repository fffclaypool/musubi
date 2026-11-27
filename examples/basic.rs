//! ランダムなベクトルで HNSW を構築し、総当たり検索と結果を比べる。
//!
//!   cargo run --release --example basic

use musubi::metrics::{cosine_distance, normalize};
use musubi::HnswIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

const DIM: usize = 32;
const NUM_VECTORS: usize = 10_000;
const NUM_QUERIES: usize = 100;
const K: usize = 10;
const EF: usize = 100;

fn random_vector(rng: &mut StdRng) -> Vec<f32> {
    (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// 全ベクトルとの距離を計算して上位 k 件の ID を返す (正解データ)
fn brute_force(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let query = normalize(query).expect("query must not be a zero vector");
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(id, v)| (id, cosine_distance(&query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<Vec<f32>> = (0..NUM_VECTORS).map(|_| random_vector(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..NUM_QUERIES).map(|_| random_vector(&mut rng)).collect();

    let start = Instant::now();
    let mut index = HnswIndex::new(16, 200);
    for v in &vectors {
        index.insert(v.clone());
    }
    println!(
        "built: {} vectors, dim={}, {:.2?}",
        index.len(),
        index.dim(),
        start.elapsed()
    );

    // 総当たり側は正規化済みベクトルで比較する (インデックス内部と同じ条件)
    let normalized: Vec<Vec<f32>> = vectors.iter().map(|v| normalize(v).unwrap()).collect();

    let mut hits = 0;
    let start = Instant::now();
    for query in &queries {
        let found: HashSet<usize> = index
            .search(query, K, EF)
            .into_iter()
            .map(|r| r.id)
            .collect();
        let truth = brute_force(&normalized, query, K);
        hits += truth.iter().filter(|id| found.contains(id)).count();
    }
    println!(
        "recall@{} = {:.4} (ef={}, {} queries, {:.2?} incl. brute force)",
        K,
        hits as f64 / (K * NUM_QUERIES) as f64,
        EF,
        NUM_QUERIES,
        start.elapsed()
    );

    let top = index.search(&queries[0], 3, EF);
    println!("query[0] top-3:");
    for r in top {
        println!("  id={:<6} distance={:.4}", r.id, r.distance);
    }
}
