//! Trait definitions for DocumentService capabilities.
//!
//! These traits allow for trait-based polymorphism and better testability
//! by splitting DocumentService functionality into focused concerns:
//!
//! - `DocumentRead`: Read-only document operations
//! - `DocumentWrite`: Document mutation operations
//! - `DocumentSearch`: Search operations

use crate::application::error::AppError;
use crate::domain::model::StoredRecord;

use super::types::{
    DocumentResponse, DocumentSummary, InsertCommand, InsertResult, SearchHit, UpdateCommand,
    ValidatedSearchQuery,
};

/// Read-only document operations.
///
/// This trait provides access to documents and embeddings without modification.
pub trait DocumentRead {
    /// Get the default number of results (k) for search.
    fn default_k(&self) -> usize;

    /// Get the default search expansion factor (ef) for search.
    fn default_ef(&self) -> usize;

    /// Get a document by ID.
    fn get(&self, id: &str) -> Result<DocumentResponse, AppError>;

    /// List documents with pagination.
    fn list(&self, offset: usize, limit: usize) -> (usize, Vec<DocumentSummary>);

    /// Embed a list of texts and return their embeddings.
    fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError>;
}

/// Document mutation operations.
///
/// This trait provides insert, update, and delete capabilities.
pub trait DocumentWrite {
    /// Insert a new document.
    fn insert(&mut self, cmd: InsertCommand) -> Result<InsertResult, AppError>;

    /// Update an existing document.
    fn update(&mut self, id: &str, cmd: UpdateCommand) -> Result<DocumentResponse, AppError>;

    /// Delete a document by ID.
    fn delete(&mut self, id: &str) -> Result<(), AppError>;

    /// Import pre-computed embeddings.
    fn import_embeddings(&mut self, records: Vec<StoredRecord>) -> Result<usize, AppError>;
}

/// Search operations.
///
/// This trait provides document search functionality.
pub trait DocumentSearch {
    /// Search for documents matching the validated query.
    fn search(&self, query: ValidatedSearchQuery) -> Result<Vec<SearchHit>, AppError>;
}
