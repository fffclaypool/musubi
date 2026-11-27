//! 事前計算済みの埋め込み (JSONL) を読み込み、指定したドキュメントに似たものを探す。
//!
//! 埋め込みの生成:
//!   uv run python/embed_csv.py --input data/sample.csv --output data/embeddings.jsonl
//!
//! 実行:
//!   cargo run --release --example similar_docs -- data/embeddings.jsonl <doc-id>
//!
//! doc-id を省略すると先頭のドキュメントを使う。

use musubi::HnswIndex;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Deserialize)]
struct Doc {
    id: String,
    title: Option<String>,
    embedding: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "data/embeddings.jsonl".to_string());
    let target_id = args.next();

    let mut docs: Vec<Doc> = Vec::new();
    for line in BufReader::new(File::open(&path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        docs.push(serde_json::from_str(&line)?);
    }
    if docs.is_empty() {
        return Err(format!("no documents in {}", path).into());
    }

    // インデックスの ID は挿入順 (= docs の添字) になる
    let mut index = HnswIndex::new(16, 200);
    for doc in &docs {
        index.insert(doc.embedding.clone());
    }

    let target = match &target_id {
        Some(id) => docs
            .iter()
            .position(|d| &d.id == id)
            .ok_or_else(|| format!("document id not found: {}", id))?,
        None => 0,
    };

    println!(
        "similar to [{}] {}",
        docs[target].id,
        docs[target].title.as_deref().unwrap_or("")
    );
    let k = 6.min(docs.len());
    for r in index.search(&docs[target].embedding, k, 100) {
        if r.id == target {
            continue;
        }
        println!(
            "  distance={:.4}  [{}] {}",
            r.distance,
            docs[r.id].id,
            docs[r.id].title.as_deref().unwrap_or("")
        );
    }
    Ok(())
}
