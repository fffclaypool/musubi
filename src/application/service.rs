//! Document service module.
//!
//! This module provides the main document management service with vector search capabilities.
//! It is organized into the following submodules:
//!
//! - `types`: Type definitions for commands, responses, and configuration
//! - `core`: Core DocumentService struct definition
//! - `load`: Service loading and migration operations
//! - `write`: Write operations (insert, update, delete)
//! - `search`: Search operations
//! - `read`: Read operations (get, list, embed, import)
//! - `compaction`: Compaction and WAL rotation operations
//! - `util`: Internal utility functions

mod compaction;
mod core;
mod load;
mod read;
mod search;
mod types;
mod util;
mod write;

#[cfg(test)]
mod tests;

// Re-export public types
pub use types::{
    ChunkConfig, ChunkInfo, DocumentResponse, DocumentSummary, InsertCommand, InsertResult,
    SearchFilter, SearchHit, SearchInput, SearchMode, SearchParams, SearchRequest,
    SearchValidationError, ServiceConfig, TagFilter, TombstoneConfig, TombstonePolicy,
    UpdateCommand, ValidatedSearchQuery,
};

// Re-export Tag from domain
pub use crate::domain::model::Tag;

// Re-export the main service struct
pub use core::DocumentService;
