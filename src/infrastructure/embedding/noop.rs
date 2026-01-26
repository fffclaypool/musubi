use crate::domain::ports::Embedder;
use crate::domain::types::Vector;
use std::io;

#[derive(Debug)]
pub struct NoopEmbedder;

impl NoopEmbedder {
    pub fn new() -> Self {
        Self
    }
}

impl Embedder for NoopEmbedder {
    fn embed(&self, _texts: Vec<String>) -> io::Result<Vec<Vector>> {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "embedder is not configured for this command",
        ))
    }
}
