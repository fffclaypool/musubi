//! musubi: 学習用の HNSW (Hierarchical Navigable Small World) 実装。
//!
//! ```
//! use musubi::HnswIndex;
//!
//! let mut index = HnswIndex::new(16, 200);
//! index.insert(vec![1.0, 0.0]);
//! index.insert(vec![0.0, 1.0]);
//!
//! let results = index.search(&[0.9, 0.1], 1, 50);
//! assert_eq!(results[0].id, 0);
//! ```

pub mod hnsw;
pub mod metrics;
pub mod neighbor;
pub mod snapshot;
pub mod types;

pub use crate::hnsw::HnswIndex;
pub use crate::types::{SearchResult, Vector};
