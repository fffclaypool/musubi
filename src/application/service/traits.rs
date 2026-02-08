//! Trait definitions for DocumentService capabilities.
//!
//! These traits allow for trait-based polymorphism and better testability
//! by splitting DocumentService functionality into focused concerns:
//!
//! - `DocumentSearch`: Search operations
//! - `DocumentIngestion`: Batch ingestion and sync operations (defined in ingestion.rs)
//! - `DocumentDefaults`: Read defaults for search parameters

use crate::application::error::AppError;

use super::types::{SearchHit, ValidatedSearchQuery};

/// Search operations.
///
/// This trait provides document search functionality.
pub trait DocumentSearch {
    /// Search for documents matching the validated query.
    fn search(&self, query: ValidatedSearchQuery) -> Result<Vec<SearchHit>, AppError>;
}

/// Read defaults for search parameters
pub trait DocumentDefaults {
    /// Get the default number of results (k) for search.
    fn default_k(&self) -> usize;

    /// Get the default search expansion factor (ef) for search.
    fn default_ef(&self) -> usize;
}
