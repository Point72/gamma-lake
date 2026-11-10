//! Delta-backed Rust API combining [`gamma_lake_core`] with [`gamma_lake_delta`].

use std::sync::Arc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub use gamma_lake_core::*;
pub use gamma_lake_delta::{DeltaOptions, DeltaStore};

/// Opens a lake using the configured Delta store.
pub fn open(store: DeltaStore, options: LakeOptions) -> Result<Lake> {
    Lake::open_with_store(Arc::new(store), options)
}

/// Creates any missing index and metadata tables, then opens the lake.
pub fn create(
    store: DeltaStore,
    index_schema: arrow::datatypes::SchemaRef,
    options: LakeOptions,
) -> Result<Lake> {
    Lake::create_with_store(Arc::new(store), index_schema, options)
}

/// Package version of the Rust library.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
