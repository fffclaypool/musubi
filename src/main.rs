use musubi::{load_with_wal, HnswIndex, WalWriter};
use rand::Rng;
use std::fs;
use std::io;
use std::path::Path;

fn normalize(vector: &[f32]) -> Option<Vec<f32>> {
    let norm_sq: f32 = vector.iter().map(|v| v * v).sum();
    if norm_sq == 0.0 {
        return None;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    Some(vector.iter().map(|v| v * inv_norm).collect())
}

fn main() -> io::Result<()> {
    println!("=== HNSW Index Demo ===\n");

    // WAL追記 -> 復元の簡易デモ
    let wal_path = Path::new("hnsw.wal");
    let snapshot_path = Path::new("hnsw.bin");
    if wal_path.exists() {
        fs::remove_file(wal_path)?;
    }
    if snapshot_path.exists() {
        fs::remove_file(snapshot_path)?;
    }

    let mut wal = WalWriter::new(wal_path)?;
    let mut index = load_with_wal(snapshot_path, wal_path, 16, 200)?;
    index.insert_with_wal(vec![1.0, 0.0, 0.0], &mut wal)?;
    index.insert_with_wal(vec![0.0, 1.0, 0.0], &mut wal)?;
    index.save(snapshot_path)?;

    let recovered = load_with_wal(snapshot_path, wal_path, 16, 200)?;
    let demo_results = recovered.search(&[1.0, 0.0, 0.0], 1, 10);
    println!("WAL復元デモ結果: {:?}", demo_results);

    // パラメータ
    let dimensions = 1536;
    let k = 10; // 検索する最近傍の数

    let num_vectors_list = [1_000];
    let ef_construction_values = [200, 400];
    let ef_values = [50, 100];

    let mut rng = rand::thread_rng();

    let save_path = Path::new("hnsw.bin");

    for &num_vectors in &num_vectors_list {
        // ランダムなベクトルを生成
        println!("\n{}次元のランダムベクトルを{}件生成中...", dimensions, num_vectors);
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let normalized_vectors: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| normalize(v).expect("zero vector is not supported"))
            .collect();

        // ランダムなクエリベクトルを生成
        let query: Vec<f32> = (0..dimensions).map(|_| rng.gen::<f32>()).collect();
        let normalized_query = normalize(&query).expect("zero vector is not supported");

        // 総当たり検索で検証
        println!("総当たり検索で検証中...");
        let mut brute_force: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(id, _)| {
                let dist: f32 = normalized_vectors[id]
                    .iter()
                    .zip(normalized_query.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                (id, 1.0 - dist)
            })
            .collect();
        brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let true_ids: std::collections::HashSet<usize> =
            brute_force.iter().take(k).map(|(id, _)| *id).collect();

        for &ef_construction in &ef_construction_values {
            // M=16, ef_construction を変えてインデックスを作成
            let mut index = HnswIndex::new(16, ef_construction);

            println!(
                "\nHNSWインデックスにベクトルを挿入中... (n={}, ef_construction={})",
                num_vectors, ef_construction
            );
            for vector in &vectors {
                index.insert(vector.clone());
            }
            println!("インデックスサイズ: {}件\n", index.len());

            // スナップショット保存/復元のデモ
            if save_path.exists() {
                fs::remove_file(save_path)?;
            }
            index.save(save_path)?;
            let loaded = HnswIndex::load(save_path)?;

            for &ef in &ef_values {
                let results = index.search(&query, k, ef);
                let loaded_results = loaded.search(&query, k, ef);
                let hnsw_ids: std::collections::HashSet<usize> =
                    results.iter().map(|r| r.id).collect();
                let loaded_ids: std::collections::HashSet<usize> =
                    loaded_results.iter().map(|r| r.id).collect();
                let recall = hnsw_ids.intersection(&true_ids).count() as f32 / k as f32;
                let loaded_recall =
                    loaded_ids.intersection(&true_ids).count() as f32 / k as f32;
                println!(
                    "n={}, ef_construction={}, ef={}, Recall@{}: {:.1}% (loaded {:.1}%)",
                    num_vectors,
                    ef_construction,
                    ef,
                    k,
                    recall * 100.0,
                    loaded_recall * 100.0
                );
            }
        }
    }

    Ok(())
}
