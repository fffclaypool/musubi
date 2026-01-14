use musubi::HnswIndex;
use rand::Rng;

fn main() {
    println!("=== HNSW Index Demo ===\n");

    // パラメータ
    let num_vectors = 1000;
    let dimensions = 128;
    let k = 10; // 検索する最近傍の数

    // M=16, ef_construction=200 でインデックスを作成
    let mut index = HnswIndex::new(16, 200);

    // ランダムなベクトルを生成
    println!("{}次元のランダムベクトルを{}件生成中...", dimensions, num_vectors);
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // ベクトルをインデックスに挿入
    println!("HNSWインデックスにベクトルを挿入中...");
    for vector in &vectors {
        index.insert(vector.clone());
    }
    println!("インデックスサイズ: {}件\n", index.len());

    // ランダムなクエリベクトルを生成
    let query: Vec<f32> = (0..dimensions).map(|_| rng.gen::<f32>()).collect();

    // k個の最近傍を検索
    println!("{}個の最近傍を検索中...\n", k);
    let results = index.search(&query, k, 100);

    // 結果を表示
    println!("検索結果:");
    println!("{:-<40}", "");
    println!("{:<8} {:<15}", "ID", "距離");
    println!("{:-<40}", "");
    for result in &results {
        println!("{:<8} {:<15.6}", result.id, result.distance);
    }
    println!("{:-<40}", "");

    // 総当たり検索で検証
    println!("\n総当たり検索で検証中...");
    let mut brute_force: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(id, v)| {
            let dist: f32 = v.iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            (id, dist)
        })
        .collect();
    brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\n総当たり検索の上位{}件:", k);
    println!("{:-<40}", "");
    println!("{:<8} {:<15}", "ID", "距離");
    println!("{:-<40}", "");
    for (id, dist) in brute_force.iter().take(k) {
        println!("{:<8} {:<15.6}", id, dist);
    }
    println!("{:-<40}", "");

    // 再現率を計算
    let hnsw_ids: std::collections::HashSet<usize> = results.iter().map(|r| r.id).collect();
    let true_ids: std::collections::HashSet<usize> = brute_force.iter().take(k).map(|(id, _)| *id).collect();
    let recall = hnsw_ids.intersection(&true_ids).count() as f32 / k as f32;
    println!("\nRecall@{}: {:.1}%", k, recall * 100.0);
}
