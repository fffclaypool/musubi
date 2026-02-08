//! Document service module.
//!
//! This module provides the main document management service with vector search capabilities.
//! It is organized into the following submodules:
//!
//! - `types`: Type definitions for commands, responses, and configuration
//! - `core`: Core DocumentService struct definition
//! - `load`: Service loading and migration operations
//! - `ingestion`: Batch ingestion and differential sync operations
//! - `search`: Search operations
//! - `compaction`: Compaction and WAL rotation operations
//! - `job_store`: In-memory job state management
//! - `util`: Internal utility functions

mod compaction;
mod core;
pub mod ingestion;
pub mod job_store;
mod load;
mod read;
mod search;
mod traits;
mod types;
mod util;
mod write;

#[cfg(test)]
mod tests;

// Re-export public types
pub use types::{
    BatchDocument, BatchError, BatchInsertResult, ChunkConfig, ChunkInfo, IngestionJob,
    JobProgress, JobStatus, LastSyncInfo, SearchFilter, SearchHit, SearchInput, SearchMode,
    SearchParams, SearchRequest, SearchValidationError, ServiceConfig, TagFilter, TombstoneConfig,
    TombstonePolicy, ValidatedSearchQuery,
};

// Re-export Tag from domain
pub use crate::domain::model::Tag;

// Re-export the main service struct
pub use core::DocumentService;

// Re-export traits for trait-based polymorphism
pub use traits::{DocumentDefaults, DocumentSearch};
pub use ingestion::DocumentIngestion;

// Re-export ingestion utilities
pub use ingestion::{run_sync, ProcessResult};
