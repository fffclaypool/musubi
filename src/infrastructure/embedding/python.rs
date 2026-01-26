use crate::domain::ports::Embedder;
use crate::domain::types::Vector;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::process::{Command, Stdio};

#[derive(Debug)]
pub struct PythonEmbedder {
    python_bin: String,
    model: String,
}

impl PythonEmbedder {
    pub fn new(python_bin: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            python_bin: python_bin.into(),
            model: model.into(),
        }
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

impl Embedder for PythonEmbedder {
    fn embed(&self, texts: Vec<String>) -> io::Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut child = Command::new(&self.python_bin)
            .arg("python/embed_text.py")
            .arg("--model")
            .arg(&self.model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            let payload = EmbedRequest { texts };
            let data = serde_json::to_vec(&payload)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            stdin.write_all(&data)?;
        }

        let output = child.wait_with_output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("embedding failed: {}", stderr.trim()),
            ));
        }

        let response: EmbedResponse = serde_json::from_slice(&output.stdout)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        Ok(response.embeddings)
    }
}
