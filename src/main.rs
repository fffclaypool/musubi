use musubi::application::error::AppError;
use musubi::application::service::{DocumentService, ServiceConfig, TombstoneConfig};
use musubi::domain::ports::Embedder;
use musubi::infrastructure::embedding::http::HttpEmbedder;
use musubi::infrastructure::embedding::python::PythonEmbedder;
use musubi::infrastructure::index::adapter::HnswIndexFactory;
use musubi::infrastructure::storage::record_store::JsonlRecordStore;
use musubi::infrastructure::storage::wal::WalConfig;
use musubi::interface::http::server::serve;
use std::io;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> io::Result<()> {
    let snapshot_path = PathBuf::from("hnsw.bin");
    let records_path = PathBuf::from("data/records.jsonl");
    let addr = std::env::var("MUSUBI_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    // WAL configuration
    let wal_config = {
        let wal_path = std::env::var("MUSUBI_WAL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("hnsw.wal"));

        let max_bytes = std::env::var("MUSUBI_WAL_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());

        let max_records = std::env::var("MUSUBI_WAL_MAX_RECORDS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        // WAL is enabled by default
        let wal_enabled = std::env::var("MUSUBI_WAL_ENABLED")
            .map(|s| s != "0" && s.to_lowercase() != "false")
            .unwrap_or(true);

        if wal_enabled {
            println!("WAL enabled: {:?}", wal_path);
            if let Some(max_bytes) = max_bytes {
                println!("  max_bytes: {}", max_bytes);
            }
            if let Some(max_records) = max_records {
                println!("  max_records: {}", max_records);
            }
            Some(WalConfig {
                path: wal_path,
                max_bytes,
                max_records,
            })
        } else {
            println!("WAL disabled");
            None
        }
    };

    // Tombstone configuration
    let tombstone_config = {
        let max_tombstones = std::env::var("MUSUBI_TOMBSTONE_MAX_COUNT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let max_ratio = std::env::var("MUSUBI_TOMBSTONE_MAX_RATIO")
            .ok()
            .and_then(|s| s.parse::<f64>().ok());

        if max_tombstones.is_some() || max_ratio.is_some() {
            println!("Tombstone compaction:");
            if let Some(max) = max_tombstones {
                println!("  max_count: {}", max);
            }
            if let Some(ratio) = max_ratio {
                println!("  max_ratio: {:.1}%", ratio * 100.0);
            }
            TombstoneConfig {
                max_tombstones,
                max_tombstone_ratio: max_ratio,
            }
        } else {
            // Use default (30% ratio)
            println!("Tombstone compaction: default (30% ratio)");
            TombstoneConfig::default()
        }
    };

    let config = ServiceConfig {
        snapshot_path,
        default_k: 5,
        default_ef: 100,
        wal_config,
        tombstone_config,
    };

    let embedder: Box<dyn Embedder> = if let Ok(url) = std::env::var("MUSUBI_EMBED_URL") {
        println!("Using HTTP embedder: {}", url);
        Box::new(HttpEmbedder::new(url)?)
    } else {
        let python_bin =
            std::env::var("MUSUBI_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let model = std::env::var("MUSUBI_MODEL").unwrap_or_else(|_| {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string()
        });
        println!("Using Python embedder: {}", model);
        Box::new(PythonEmbedder::new(python_bin, model))
    };

    let record_store = Box::new(JsonlRecordStore::new(records_path));
    let index_factory = Box::new(HnswIndexFactory::new(16, 200));

    let service = DocumentService::load(config, embedder, record_store, index_factory)
        .map_err(map_app_error)?;

    serve(addr, service).await
}

fn map_app_error(err: AppError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err.to_string())
}
