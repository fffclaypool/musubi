use crate::domain::ports::Embedder;
use crate::domain::types::Vector;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::io;
use std::time::Duration;

pub struct HttpEmbedder {
    client: Client,
    url: String,
}

impl HttpEmbedder {
    pub fn new(url: impl Into<String>) -> io::Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(io::Error::other)?;
        Ok(Self {
            client,
            url: url.into(),
        })
    }
}

#[derive(Debug, Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    detail: String,
}

impl Embedder for HttpEmbedder {
    fn embed(&self, texts: Vec<String>) -> io::Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let response = self
            .client
            .post(&self.url)
            .json(&EmbedRequest { texts })
            .send()
            .map_err(|err| io::Error::new(io::ErrorKind::ConnectionRefused, err))?;

        if !response.status().is_success() {
            let status = response.status();
            let error: Result<ErrorResponse, _> = response.json();
            let message = error
                .map(|e| e.detail)
                .unwrap_or_else(|_| format!("HTTP {}", status));
            return Err(io::Error::other(message));
        }

        let result: EmbedResponse = response
            .json()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        Ok(result.embeddings)
    }
}
