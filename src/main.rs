use musubi::application::error::AppError;
use musubi::application::service::{DocumentService, ServiceConfig};
use musubi::infrastructure::embedding::python::PythonEmbedder;
use musubi::infrastructure::index::adapter::HnswIndexFactory;
use musubi::infrastructure::storage::record_store::JsonlRecordStore;
use musubi::interface::http::server::serve;
use std::io;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> io::Result<()> {
    let snapshot_path = PathBuf::from("hnsw.bin");
    let records_path = PathBuf::from("data/records.jsonl");
    let python_bin = std::env::var("MUSUBI_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let model = std::env::var("MUSUBI_MODEL")
        .unwrap_or_else(|_| "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string());
    let addr = std::env::var("MUSUBI_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let config = ServiceConfig {
        snapshot_path,
        default_k: 5,
        default_ef: 100,
    };
    let embedder = Box::new(PythonEmbedder::new(python_bin, model));
    let record_store = Box::new(JsonlRecordStore::new(records_path));
    let index_factory = Box::new(HnswIndexFactory::new(16, 200));

    let service = DocumentService::load(config, embedder, record_store, index_factory)
        .map_err(map_app_error)?;

    serve(addr, service).await
}

fn map_app_error(err: AppError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err.to_string())
}
