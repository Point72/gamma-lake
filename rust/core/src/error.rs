use arrow::error::ArrowError;
use datafusion::error::DataFusionError;

/// Errors returned by the gamma-lake core.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("lake at {0} is not initialized: a required index or metadata table is missing")]
    NotInitialized(String),
    #[error("no feature tables exist for any of the requested features: {0:?}")]
    MissingFeatures(Vec<String>),
    #[error("not registered as target signals: {0:?}")]
    MisregisteredTargets(Vec<String>),
    #[error(
        "feature {feature:?} has signal type {signal_type:?}, which the Rust core cannot read yet"
    )]
    Unsupported {
        feature: String,
        signal_type: String,
    },
    /// A write the Rust core does not handle yet, detected before anything was written.
    #[error("unsupported input: {0}")]
    UnsupportedInput(String),
    /// A write failed after it started committing, so the lake may hold part of it.
    #[error(
        "write failed while {phase}; tables already committed: {committed:?}. The lake may \
         hold part of this write: {source}"
    )]
    Write {
        phase: String,
        committed: Vec<String>,
        source: Box<Error>,
    },
    /// Table `table` changed after the write read it, so the write's commit to it was refused
    /// and wrote nothing. Another writer committed to the table in between.
    #[error("table {table} changed after this write read it, so nothing was committed to it")]
    Conflict { table: String },
    /// A [`TableLock`](crate::TableLock) failed to lock or release table `target`.
    #[error("lock on {target} failed: {source}")]
    Lock {
        target: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("invalid predicate: {0}")]
    InvalidPredicate(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("integrity check failed for table {table_addr}: {kind}")]
    Integrity { table_addr: String, kind: String },
    /// An error of the [`TableStore`](crate::TableStore), such as a failed commit or an
    /// unreachable object store.
    #[error(transparent)]
    Store(Box<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    DataFusion(#[from] DataFusionError),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

impl Error {
    /// Stable snake_case name of the error category.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::NotInitialized(_) => "not_initialized",
            Error::MissingFeatures(_) => "missing_features",
            Error::MisregisteredTargets(_) => "misregistered_targets",
            Error::Unsupported { .. } | Error::UnsupportedInput(_) => "unsupported",
            Error::Write { .. } => "write",
            Error::Conflict { .. } => "conflict",
            Error::Lock { .. } => "lock",
            Error::InvalidPredicate(_) => "invalid_predicate",
            Error::InvalidArgument(_) => "invalid_argument",
            Error::Integrity { .. } => "integrity",
            Error::Store(_) | Error::DataFusion(_) | Error::Arrow(_) => "engine",
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
