use musubi::{load_with_wal, HnswIndex, WalWriter};
use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Deserialize)]
struct EmbeddingRecord {
    id: String,
    embedding: Vec<f32>,
    title: Option<String>,
    body: Option<String>,
    source: Option<String>,
    updated_at: Option<String>,
    tags: Option<String>,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let embeddings_path = args.get(1).map(String::as_str).unwrap_or("data/embeddings.jsonl");
    let snapshot_path = args.get(2).map(String::as_str).unwrap_or("hnsw.bin");
    let wal_path = args.get(3).map(String::as_str).unwrap_or("hnsw.wal");

    let m = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let ef_construction = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(200);

    let mut index = load_with_wal(snapshot_path, wal_path, m, ef_construction)?;
    let mut wal = WalWriter::new(wal_path)?;

    let file = File::open(embeddings_path)?;
    let reader = BufReader::new(file);

    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: EmbeddingRecord = serde_json::from_str(&line)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        index.insert_with_wal(record.embedding, &mut wal)?;
        count += 1;
    }

    index.save(snapshot_path)?;
    println!(
        "imported {} vectors into {} (snapshot: {}, wal: {})",
        count,
        index.len(),
        snapshot_path,
        wal_path
    );
    Ok(())
}
