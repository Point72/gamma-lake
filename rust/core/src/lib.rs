//! The storage-independent core of gamma-lake.
//!
//! A lake stores each feature table with the same sort keys and row order as its index.
//! Reads line tables up by position and concatenate them horizontally, without joins. A
//! [`Lake`] keeps its tables in a [`TableStore`]; [`MemoryStore`] keeps them in memory, and
//! other crates provide persistent stores, such as `gamma_lake_delta` for Delta Lake.

mod admin;
mod align;
mod error;
mod expressions;
mod joined;
mod lake;
mod lock;
mod predicate;
mod provider;
pub mod runtime;
mod scan;
mod sort;
pub mod store;
mod stream;
mod write;

pub use admin::Annotation;
pub use arrow;
pub use datafusion;
pub use error::{Error, Result};
pub use expressions::Expression;
pub use lake::{Lake, LakeOptions, SelectedFeature};
pub use lock::{LockError, LockGuard, TableLock};
pub use predicate::{Dnf, Op, Term};
pub use provider::FeatureTable;
pub use store::{MemoryStore, StoredTable, TableStore};
#[doc(hidden)]
pub use stream::live_streams;
pub use stream::{AsyncBatchStream, BatchStream};
pub use write::{AddOptions, OverlapMode, SignalType};
