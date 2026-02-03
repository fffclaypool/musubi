use musubi::application::error::AppError;
use musubi::application::service::{ChunkConfig, DocumentService, ServiceConfig, TombstoneConfig};
use musubi::domain::ports::{ChunkStore, Chunker, Embedder};
use musubi::infrastructure::chunking::{FixedChunker, SemanticChunker};
use musubi::infrastructure::embedding::http::HttpEmbedder;
use musubi::infrastructure::embedding::python::PythonEmbedder;
use musubi::infrastructure::index::adapter::HnswIndexFactory;
use musubi::infrastructure::storage::chunk_store::JsonlChunkStore;
use musubi::infrastructure::storage::record_store::JsonlRecordStore;
use musubi::infrastructure::storage::wal::WalConfig;
use musubi::interface::http::server::serve;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> io::Result<()> {
    let snapshot_path = PathBuf::from("hnsw.bin");
    let records_path = PathBuf::from("data/records.jsonl");
    let chunks_path = PathBuf::from("data/chunks.jsonl");
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

        // Early validation for conflicting settings (better UX than failing at load time)
        if max_tombstones.is_some() && max_ratio.is_some() {
            eprintln!(
                "Error: Cannot set both MUSUBI_TOMBSTONE_MAX_COUNT and MUSUBI_TOMBSTONE_MAX_RATIO"
            );
            std::process::exit(1);
        }

        if let Some(ratio) = max_ratio {
            if ratio.is_nan() || !(0.0..=1.0).contains(&ratio) {
                eprintln!("Error: MUSUBI_TOMBSTONE_MAX_RATIO must be between 0.0 and 1.0");
                std::process::exit(1);
            }
        }

        // Build config and log
        match (max_tombstones, max_ratio) {
            (Some(max), None) => {
                println!("Tombstone compaction: max_count = {}", max);
                TombstoneConfig {
                    max_tombstones: Some(max),
                    max_tombstone_ratio: None,
                }
            }
            (None, Some(ratio)) => {
                println!("Tombstone compaction: max_ratio = {:.1}%", ratio * 100.0);
                TombstoneConfig {
                    max_tombstones: None,
                    max_tombstone_ratio: Some(ratio),
                }
            }
            (None, None) => {
                println!("Tombstone compaction: default (30% ratio)");
                TombstoneConfig::default()
            }
            (Some(_), Some(_)) => unreachable!(), // Already handled above
        }
    };

    // Embedder configuration (needed before chunk config for semantic chunker)
    let embedder: Arc<dyn Embedder> = if let Ok(url) = std::env::var("MUSUBI_EMBED_URL") {
        println!("Using HTTP embedder: {}", url);
        Arc::new(HttpEmbedder::new(url)?)
    } else {
        let python_bin = std::env::var("MUSUBI_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let model = std::env::var("MUSUBI_MODEL").unwrap_or_else(|_| {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string()
        });
        println!("Using Python embedder: {}", model);
        Arc::new(PythonEmbedder::new(python_bin, model))
    };

    // Chunk configuration
    let (chunk_config, chunker, chunk_store): (
        ChunkConfig,
        Option<Box<dyn Chunker>>,
        Option<Box<dyn ChunkStore>>,
    ) = {
        let chunk_type = std::env::var("MUSUBI_CHUNK_TYPE")
            .unwrap_or_else(|_| "none".to_string())
            .to_lowercase();

        match chunk_type.as_str() {
            "fixed" => {
                let chunk_size = std::env::var("MUSUBI_CHUNK_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(512);
                let overlap = std::env::var("MUSUBI_CHUNK_OVERLAP")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(50);

                println!("Chunking: fixed (size={}, overlap={})", chunk_size, overlap);

                (
                    ChunkConfig::Fixed {
                        chunk_size,
                        overlap,
                    },
                    Some(Box::new(FixedChunker::new(chunk_size, overlap)) as Box<dyn Chunker>),
                    Some(Box::new(JsonlChunkStore::new(&chunks_path)) as Box<dyn ChunkStore>),
                )
            }
            "semantic" => {
                let min_chunk_size = std::env::var("MUSUBI_CHUNK_MIN_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(100);
                let max_chunk_size = std::env::var("MUSUBI_CHUNK_MAX_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1000);
                let similarity_threshold = std::env::var("MUSUBI_CHUNK_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);

                println!(
                    "Chunking: semantic (min={}, max={}, threshold={})",
                    min_chunk_size, max_chunk_size, similarity_threshold
                );

                (
                    ChunkConfig::Semantic {
                        min_chunk_size,
                        max_chunk_size,
                        similarity_threshold,
                    },
                    Some(Box::new(SemanticChunker::new(
                        min_chunk_size,
                        max_chunk_size,
                        similarity_threshold,
                        embedder.clone(),
                    )) as Box<dyn Chunker>),
                    Some(Box::new(JsonlChunkStore::new(&chunks_path)) as Box<dyn ChunkStore>),
                )
            }
            _ => {
                // "none" or any other value - no chunking (backward compatible)
                println!("Chunking: disabled (documents indexed as-is)");
                (ChunkConfig::None, None, None)
            }
        }
    };

    let config = ServiceConfig {
        snapshot_path,
        default_k: 5,
        default_ef: 100,
        wal_config,
        tombstone_config,
        chunk_config,
    };

    // Convert Arc<dyn Embedder> to Box<dyn Embedder> for the service
    let embedder_box: Box<dyn Embedder> = Box::new(ArcEmbedder(embedder));

    let record_store = Box::new(JsonlRecordStore::new(records_path));
    let index_factory = Box::new(HnswIndexFactory::new(16, 200));

    let service = DocumentService::load(
        config,
        embedder_box,
        record_store,
        index_factory,
        chunker,
        chunk_store,
    )
    .map_err(map_app_error)?;

    serve(addr, service).await
}

/// Wrapper to convert Arc<dyn Embedder> to Box<dyn Embedder>
struct ArcEmbedder(Arc<dyn Embedder>);

impl Embedder for ArcEmbedder {
    fn embed(&self, texts: Vec<String>) -> io::Result<Vec<Vec<f32>>> {
        self.0.embed(texts)
    }
}

fn map_app_error(err: AppError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err.to_string())
}
