//! Application configuration loaded from environment variables.

use crate::application::service::{ChunkConfig, ServiceConfig, TombstoneConfig};
use crate::infrastructure::storage::wal::WalConfig;
use std::path::PathBuf;
use std::{env, io};

/// Application configuration loaded from environment variables.
#[derive(Debug)]
pub struct AppConfig {
    /// HTTP server address
    pub addr: String,
    /// Path to HNSW index snapshot
    pub snapshot_path: PathBuf,
    /// Path to records JSONL file
    pub records_path: PathBuf,
    /// Path to chunks JSONL file
    pub chunks_path: PathBuf,
    /// WAL configuration (None if disabled)
    pub wal_config: Option<WalConfig>,
    /// Tombstone compaction configuration
    pub tombstone_config: TombstoneConfig,
    /// Chunk configuration
    pub chunk_config: ChunkConfig,
    /// Embedder configuration
    pub embedder_config: EmbedderConfig,
}

/// Embedder configuration
#[derive(Debug)]
pub enum EmbedderConfig {
    /// HTTP-based embedder
    Http { url: String },
    /// Python subprocess embedder
    Python { python_bin: String, model: String },
}

/// Error type for configuration validation
#[derive(Debug)]
pub enum ConfigError {
    /// Conflicting configuration options
    Conflict(String),
    /// Invalid value
    InvalidValue(String),
    /// IO error
    Io(io::Error),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict(msg) => write!(f, "Configuration conflict: {}", msg),
            Self::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
            Self::Io(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<io::Error> for ConfigError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl AppConfig {
    /// Load configuration from environment variables.
    ///
    /// # Environment Variables
    ///
    /// - `MUSUBI_ADDR`: Server address (default: "127.0.0.1:8080")
    /// - `MUSUBI_WAL_PATH`: WAL file path (default: "hnsw.wal")
    /// - `MUSUBI_WAL_ENABLED`: Enable WAL (default: true)
    /// - `MUSUBI_WAL_MAX_BYTES`: Max WAL size before rotation
    /// - `MUSUBI_WAL_MAX_RECORDS`: Max WAL records before rotation
    /// - `MUSUBI_TOMBSTONE_MAX_COUNT`: Max tombstones before compaction
    /// - `MUSUBI_TOMBSTONE_MAX_RATIO`: Max tombstone ratio before compaction
    /// - `MUSUBI_EMBED_URL`: HTTP embedder URL (if set, uses HTTP embedder)
    /// - `MUSUBI_PYTHON`: Python binary path (default: "python3")
    /// - `MUSUBI_MODEL`: Embedding model name
    /// - `MUSUBI_CHUNK_TYPE`: Chunking type (none, fixed, semantic)
    /// - `MUSUBI_CHUNK_SIZE`: Fixed chunk size
    /// - `MUSUBI_CHUNK_OVERLAP`: Fixed chunk overlap
    /// - `MUSUBI_CHUNK_MIN_SIZE`: Semantic chunk min size
    /// - `MUSUBI_CHUNK_MAX_SIZE`: Semantic chunk max size
    /// - `MUSUBI_CHUNK_THRESHOLD`: Semantic chunk similarity threshold
    pub fn from_env() -> Result<Self, ConfigError> {
        let addr = env::var("MUSUBI_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
        let snapshot_path = PathBuf::from("hnsw.bin");
        let records_path = PathBuf::from("data/records.jsonl");
        let chunks_path = PathBuf::from("data/chunks.jsonl");

        let wal_config = Self::parse_wal_config();
        let tombstone_config = Self::parse_tombstone_config()?;
        let embedder_config = Self::parse_embedder_config();
        let chunk_config = Self::parse_chunk_config();

        Ok(Self {
            addr,
            snapshot_path,
            records_path,
            chunks_path,
            wal_config,
            tombstone_config,
            chunk_config,
            embedder_config,
        })
    }

    /// Build ServiceConfig from AppConfig
    pub fn service_config(&self) -> ServiceConfig {
        ServiceConfig {
            snapshot_path: self.snapshot_path.clone(),
            default_k: 5,
            default_ef: 100,
            wal_config: self.wal_config.clone(),
            tombstone_config: self.tombstone_config.clone(),
            chunk_config: self.chunk_config.clone(),
        }
    }

    /// Log configuration to stdout
    pub fn log(&self) {
        // WAL
        match &self.wal_config {
            Some(wal) => {
                println!("WAL enabled: {:?}", wal.path);
                if let Some(max_bytes) = wal.max_bytes {
                    println!("  max_bytes: {}", max_bytes);
                }
                if let Some(max_records) = wal.max_records {
                    println!("  max_records: {}", max_records);
                }
            }
            None => println!("WAL disabled"),
        }

        // Tombstone
        match (
            &self.tombstone_config.max_tombstones,
            &self.tombstone_config.max_tombstone_ratio,
        ) {
            (Some(max), None) => println!("Tombstone compaction: max_count = {}", max),
            (None, Some(ratio)) => {
                println!("Tombstone compaction: max_ratio = {:.1}%", ratio * 100.0)
            }
            _ => println!("Tombstone compaction: default (30% ratio)"),
        }

        // Embedder
        match &self.embedder_config {
            EmbedderConfig::Http { url } => println!("Using HTTP embedder: {}", url),
            EmbedderConfig::Python { model, .. } => println!("Using Python embedder: {}", model),
        }

        // Chunking
        match &self.chunk_config {
            ChunkConfig::None => println!("Chunking: disabled (documents indexed as-is)"),
            ChunkConfig::Fixed {
                chunk_size,
                overlap,
            } => println!("Chunking: fixed (size={}, overlap={})", chunk_size, overlap),
            ChunkConfig::Semantic {
                min_chunk_size,
                max_chunk_size,
                similarity_threshold,
            } => println!(
                "Chunking: semantic (min={}, max={}, threshold={})",
                min_chunk_size, max_chunk_size, similarity_threshold
            ),
        }
    }

    fn parse_wal_config() -> Option<WalConfig> {
        let wal_enabled = env::var("MUSUBI_WAL_ENABLED")
            .map(|s| s != "0" && s.to_lowercase() != "false")
            .unwrap_or(true);

        if !wal_enabled {
            return None;
        }

        let path = env::var("MUSUBI_WAL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("hnsw.wal"));

        let max_bytes = env::var("MUSUBI_WAL_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());

        let max_records = env::var("MUSUBI_WAL_MAX_RECORDS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        Some(WalConfig {
            path,
            max_bytes,
            max_records,
        })
    }

    fn parse_tombstone_config() -> Result<TombstoneConfig, ConfigError> {
        let max_tombstones = env::var("MUSUBI_TOMBSTONE_MAX_COUNT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let max_ratio = env::var("MUSUBI_TOMBSTONE_MAX_RATIO")
            .ok()
            .and_then(|s| s.parse::<f64>().ok());

        // Validate: cannot set both
        if max_tombstones.is_some() && max_ratio.is_some() {
            return Err(ConfigError::Conflict(
                "Cannot set both MUSUBI_TOMBSTONE_MAX_COUNT and MUSUBI_TOMBSTONE_MAX_RATIO"
                    .to_string(),
            ));
        }

        // Validate: ratio must be in range
        if let Some(ratio) = max_ratio {
            if ratio.is_nan() || !(0.0..=1.0).contains(&ratio) {
                return Err(ConfigError::InvalidValue(
                    "MUSUBI_TOMBSTONE_MAX_RATIO must be between 0.0 and 1.0".to_string(),
                ));
            }
        }

        // Use default (30% ratio) when neither is specified
        match (max_tombstones, max_ratio) {
            (None, None) => Ok(TombstoneConfig::default()),
            _ => Ok(TombstoneConfig {
                max_tombstones,
                max_tombstone_ratio: max_ratio,
            }),
        }
    }

    fn parse_embedder_config() -> EmbedderConfig {
        if let Ok(url) = env::var("MUSUBI_EMBED_URL") {
            EmbedderConfig::Http { url }
        } else {
            let python_bin = env::var("MUSUBI_PYTHON").unwrap_or_else(|_| "python3".to_string());
            let model = env::var("MUSUBI_MODEL").unwrap_or_else(|_| {
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string()
            });
            EmbedderConfig::Python { python_bin, model }
        }
    }

    fn parse_chunk_config() -> ChunkConfig {
        let chunk_type = env::var("MUSUBI_CHUNK_TYPE")
            .unwrap_or_else(|_| "none".to_string())
            .to_lowercase();

        match chunk_type.as_str() {
            "fixed" => {
                let chunk_size = env::var("MUSUBI_CHUNK_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(512);
                let overlap = env::var("MUSUBI_CHUNK_OVERLAP")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(50);
                ChunkConfig::Fixed {
                    chunk_size,
                    overlap,
                }
            }
            "semantic" => {
                let min_chunk_size = env::var("MUSUBI_CHUNK_MIN_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(100);
                let max_chunk_size = env::var("MUSUBI_CHUNK_MAX_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1000);
                let similarity_threshold = env::var("MUSUBI_CHUNK_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                ChunkConfig::Semantic {
                    min_chunk_size,
                    max_chunk_size,
                    similarity_threshold,
                }
            }
            _ => ChunkConfig::None,
        }
    }
}
