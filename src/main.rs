use musubi::application::config::{AppConfig, ConfigError, EmbedderConfig};
use musubi::application::error::AppError;
use musubi::application::service::{ChunkConfig, DocumentService};
use musubi::domain::ports::{ChunkStore, Chunker, Embedder, PendingStore};
use musubi::infrastructure::chunking::{FixedChunker, SemanticChunker};
use musubi::infrastructure::embedding::http::HttpEmbedder;
use musubi::infrastructure::embedding::python::PythonEmbedder;
use musubi::infrastructure::index::adapter::HnswIndexFactory;
use musubi::infrastructure::storage::chunk_store::JsonlChunkStore;
use musubi::infrastructure::storage::pending_store::JsonlPendingStore;
use musubi::infrastructure::storage::record_store::JsonlRecordStore;
use musubi::interface::http::server::serve;
use std::io;
use std::sync::Arc;

/// Optional chunker and chunk store pair for document chunking
type ChunkerPair = (Option<Box<dyn Chunker>>, Option<Box<dyn ChunkStore>>);

/// Optional pending store for batch ingestion
type MaybePendingStore = Option<Box<dyn PendingStore>>;

#[tokio::main]
async fn main() -> io::Result<()> {
    let config = AppConfig::from_env().map_err(map_config_error)?;
    config.log();

    // Create embedder based on configuration
    let embedder: Arc<dyn Embedder> = match &config.embedder_config {
        EmbedderConfig::Http { url } => Arc::new(HttpEmbedder::new(url.clone())?),
        EmbedderConfig::Python { python_bin, model } => {
            Arc::new(PythonEmbedder::new(python_bin.clone(), model.clone()))
        }
    };

    // Create chunker and chunk store based on configuration
    let (chunker, chunk_store): ChunkerPair = match &config.chunk_config {
        ChunkConfig::Fixed {
            chunk_size,
            overlap,
        } => (
            Some(Box::new(FixedChunker::new(*chunk_size, *overlap))),
            Some(Box::new(JsonlChunkStore::new(&config.chunks_path))),
        ),
        ChunkConfig::Semantic {
            min_chunk_size,
            max_chunk_size,
            similarity_threshold,
        } => (
            Some(Box::new(SemanticChunker::new(
                *min_chunk_size,
                *max_chunk_size,
                *similarity_threshold,
                embedder.clone(),
            ))),
            Some(Box::new(JsonlChunkStore::new(&config.chunks_path))),
        ),
        ChunkConfig::None => (None, None),
    };

    let service_config = config.service_config();
    let embedder_box: Box<dyn Embedder> = Box::new(ArcEmbedder(embedder));
    let record_store = Box::new(JsonlRecordStore::new(config.records_path));
    let index_factory = Box::new(HnswIndexFactory::new(16, 200));

    // Create pending store only when chunking is disabled
    // (batch ingestion is not supported with chunking)
    let pending_store: MaybePendingStore = match &config.chunk_config {
        ChunkConfig::None => Some(Box::new(JsonlPendingStore::new(&config.pending_path))),
        _ => None,
    };

    let service = DocumentService::load(
        service_config,
        embedder_box,
        record_store,
        index_factory,
        chunker,
        chunk_store,
        pending_store,
    )
    .map_err(map_app_error)?;

    serve(config.addr, service).await
}

/// Wrapper to convert Arc<dyn Embedder> to Box<dyn Embedder>
struct ArcEmbedder(Arc<dyn Embedder>);

impl Embedder for ArcEmbedder {
    fn embed(&self, texts: Vec<String>) -> io::Result<Vec<Vec<f32>>> {
        self.0.embed(texts)
    }
}

fn map_app_error(err: AppError) -> io::Error {
    io::Error::other(err.to_string())
}

fn map_config_error(err: ConfigError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, err.to_string())
}
