//! ANN-Benchmarks evaluation tool for musubi HNSW implementation.
//!
//! Usage:
//!   cargo run --release --bin bench_ann -- --dataset data/ann/glove-100-angular.hdf5
//!
//! With subset (brute-force ground truth):
//!   cargo run --release --bin bench_ann -- --dataset data/ann/glove-100-angular.hdf5 \
//!       --train-limit 10000 --test-limit 1000

use clap::Parser;
use hdf5::File as H5File;
use musubi::infrastructure::index::hnsw::HnswIndex;
use ndarray::Array2;
use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "bench_ann")]
#[command(about = "ANN-Benchmarks evaluation for musubi HNSW")]
struct Args {
    /// Path to HDF5 dataset (e.g., glove-100-angular.hdf5)
    #[arg(short, long)]
    dataset: PathBuf,

    /// Number of neighbors to retrieve
    #[arg(short, long, default_value = "10")]
    k: usize,

    /// Search beam width
    #[arg(long, default_value = "100")]
    ef: usize,

    /// HNSW M parameter (max neighbors per node)
    #[arg(short, long, default_value = "16")]
    m: usize,

    /// HNSW ef_construction parameter
    #[arg(long, default_value = "200")]
    ef_construction: usize,

    /// Limit training vectors (enables brute-force ground truth)
    #[arg(long)]
    train_limit: Option<usize>,

    /// Limit test queries
    #[arg(long)]
    test_limit: Option<usize>,

    /// Output directory for results
    #[arg(short, long, default_value = "data/bench")]
    output: PathBuf,

    /// Save index snapshot for size measurement
    #[arg(long)]
    save_index: bool,
}

#[derive(Serialize)]
struct BenchResult {
    dataset: String,
    train: usize,
    test: usize,
    k: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    build_time_ms: u64,
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    recall_at_k: f64,
    index_size_bytes: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Open HDF5 file
    println!("Loading dataset: {:?}", args.dataset);
    let file = H5File::open(&args.dataset)?;

    // Read train and test data
    let train_ds = file.dataset("train")?;
    let test_ds = file.dataset("test")?;

    let train_data: Array2<f32> = train_ds.read()?;
    let test_data: Array2<f32> = test_ds.read()?;

    let (train_count, dim) = train_data.dim();
    let (test_count, _) = test_data.dim();

    println!(
        "Dataset: {} train vectors, {} test queries, {} dimensions",
        train_count, test_count, dim
    );

    // Apply limits
    let train_limit = args.train_limit.unwrap_or(train_count).min(train_count);
    let test_limit = args.test_limit.unwrap_or(test_count).min(test_count);

    // Validate limits
    if train_limit == 0 {
        eprintln!("Error: train_limit must be > 0");
        std::process::exit(1);
    }
    if test_limit == 0 {
        eprintln!("Error: test_limit must be > 0");
        std::process::exit(1);
    }

    println!(
        "Using: {} train vectors, {} test queries",
        train_limit, test_limit
    );

    // Prepare train and test vectors
    let train_vectors: Vec<Vec<f32>> = (0..train_limit)
        .map(|i| train_data.row(i).to_vec())
        .collect();

    let test_vectors: Vec<Vec<f32>> = (0..test_limit).map(|i| test_data.row(i).to_vec()).collect();

    // Get or compute ground truth
    let ground_truth = if args.train_limit.is_some() {
        // Subset mode: compute brute-force ground truth
        println!("Computing brute-force ground truth for subset...");
        compute_ground_truth(&train_vectors, &test_vectors, args.k)
    } else {
        // Full dataset: use HDF5 neighbors
        println!("Using HDF5 ground truth neighbors...");
        let neighbors_ds = file.dataset("neighbors")?;
        let neighbors: Array2<i32> = neighbors_ds.read()?;
        let (_, neighbors_k) = neighbors.dim();

        // Warn if k exceeds available ground truth
        if args.k > neighbors_k {
            eprintln!(
                "Warning: k={} exceeds HDF5 neighbors columns ({}). Recall will be computed against {} neighbors.",
                args.k, neighbors_k, neighbors_k
            );
        }

        let effective_k = args.k.min(neighbors_k);
        (0..test_limit)
            .map(|i| {
                neighbors
                    .row(i)
                    .iter()
                    .take(effective_k)
                    .map(|&x| x as usize)
                    .collect()
            })
            .collect()
    };

    // Build index
    println!(
        "Building HNSW index (M={}, ef_construction={})...",
        args.m, args.ef_construction
    );
    let build_start = Instant::now();
    let mut index = HnswIndex::new(args.m, args.ef_construction);

    for vec in &train_vectors {
        index.insert(vec.clone());
    }
    let build_time = build_start.elapsed();
    println!("Build time: {:?}", build_time);

    // Measure index size if requested
    let index_size = if args.save_index {
        let temp_path = args.output.join("temp_index.bin");
        fs::create_dir_all(&args.output)?;
        index.save(&temp_path)?;
        let size = fs::metadata(&temp_path)?.len();
        fs::remove_file(&temp_path)?;
        Some(size)
    } else {
        None
    };

    // Run search benchmark
    println!("Running search benchmark (k={}, ef={})...", args.k, args.ef);
    let mut latencies: Vec<Duration> = Vec::with_capacity(test_limit);
    let mut total_recall = 0.0;

    for (i, query) in test_vectors.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(query, args.k, args.ef);
        let elapsed = start.elapsed();
        latencies.push(elapsed);

        // Compute recall (use actual ground truth size to handle k > neighbors case)
        let result_ids: HashSet<usize> = results.iter().map(|r| r.id).collect();
        let true_ids: HashSet<usize> = ground_truth[i].iter().copied().collect();
        let hits = result_ids.intersection(&true_ids).count();
        let denominator = args.k.min(ground_truth[i].len());
        if denominator > 0 {
            total_recall += hits as f64 / denominator as f64;
        }
    }

    // Compute statistics
    latencies.sort();
    let avg_latency = latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>() / test_limit as f64;
    let p50_idx = (test_limit / 2).min(test_limit.saturating_sub(1));
    let p95_idx = ((test_limit as f64 * 0.95) as usize).min(test_limit.saturating_sub(1));
    let p99_idx = ((test_limit as f64 * 0.99) as usize).min(test_limit.saturating_sub(1));
    let p50_latency = latencies[p50_idx].as_secs_f64();
    let p95_latency = latencies[p95_idx].as_secs_f64();
    let p99_latency = latencies[p99_idx].as_secs_f64();
    let recall = total_recall / test_limit as f64;

    // Print results
    println!("\n=== Results ===");
    println!("Recall@{}: {:.4}", args.k, recall);
    println!("Latency (avg): {:.3} ms", avg_latency * 1000.0);
    println!("Latency (p50): {:.3} ms", p50_latency * 1000.0);
    println!("Latency (p95): {:.3} ms", p95_latency * 1000.0);
    println!("Latency (p99): {:.3} ms", p99_latency * 1000.0);
    if let Some(size) = index_size {
        println!(
            "Index size: {} bytes ({:.2} MB)",
            size,
            size as f64 / 1_000_000.0
        );
    }

    // Save results
    let dataset_name = args
        .dataset
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let result = BenchResult {
        dataset: dataset_name.to_string(),
        train: train_limit,
        test: test_limit,
        k: args.k,
        m: args.m,
        ef_construction: args.ef_construction,
        ef_search: args.ef,
        build_time_ms: build_time.as_millis() as u64,
        avg_ms: avg_latency * 1000.0,
        p50_ms: p50_latency * 1000.0,
        p95_ms: p95_latency * 1000.0,
        p99_ms: p99_latency * 1000.0,
        recall_at_k: recall,
        index_size_bytes: index_size,
    };

    fs::create_dir_all(&args.output)?;
    let output_file = args.output.join(format!(
        "{}_m{}_ef{}_k{}.json",
        dataset_name, args.m, args.ef, args.k
    ));
    let json = serde_json::to_string_pretty(&result)?;
    let mut file = fs::File::create(&output_file)?;
    file.write_all(json.as_bytes())?;
    println!("\nResults saved to: {:?}", output_file);

    Ok(())
}

/// Compute brute-force ground truth for subset evaluation
fn compute_ground_truth(train: &[Vec<f32>], test: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    let train_normalized: Vec<Vec<f32>> = train.iter().map(|v| normalize(v)).collect();
    let test_normalized: Vec<Vec<f32>> = test.iter().map(|v| normalize(v)).collect();

    test_normalized
        .iter()
        .map(|query| {
            let mut distances: Vec<(usize, f32)> = train_normalized
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_distance(query, v)))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}
