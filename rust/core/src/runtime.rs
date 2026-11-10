use std::future::Future;
use std::sync::OnceLock;

use datafusion::error::DataFusionError;
use datafusion::prelude::{SessionConfig, SessionContext};
use tokio::sync::Semaphore;

use tokio::runtime::{Handle, Runtime};

use crate::error::{Error, Result};

/// Environment variable that sets the number of threads the core computes on.
pub const THREADS_VAR: &str = "GAMMA_LAKE_THREADS";

/// A positive thread count, or `None` when `value` is unset, empty or invalid.
fn parse_threads(value: Option<&str>) -> Option<usize> {
    value?.trim().parse().ok().filter(|&n| n > 0)
}

/// Parallelism for runtime workers, CPU-bound blocking tasks and DataFusion
/// partitions, read once per process from `GAMMA_LAKE_THREADS` and otherwise the
/// available parallelism.
pub fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        parse_threads(std::env::var(THREADS_VAR).ok().as_deref())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    })
}

/// Permits for CPU-bound work on the blocking pool. The pool itself is not capped,
/// because Delta log replay waits there for file I/O that also needs a blocking thread.
pub(crate) fn cpu_permits() -> &'static Semaphore {
    static PERMITS: OnceLock<Semaphore> = OnceLock::new();
    PERMITS.get_or_init(|| Semaphore::new(threads()))
}

/// A DataFusion session for reading lakes, with `target_partitions` set to [`threads`] and
/// `overrides` applied on top.
///
/// Identifiers are case-sensitive, Parquet scans yield string and binary view types, and hash
/// joins do not push their keys into scans as IN lists, which can fail on dictionary keys.
pub fn session_context(overrides: &[(&str, String)]) -> Result<SessionContext> {
    let mut config = SessionConfig::new()
        .set_bool("datafusion.sql_parser.enable_ident_normalization", false)
        .set_bool("datafusion.execution.parquet.schema_force_view_types", true)
        .set_usize("datafusion.optimizer.hash_join_inlist_pushdown_max_size", 0)
        .set_usize(
            "datafusion.optimizer.hash_join_inlist_pushdown_max_distinct_values",
            0,
        )
        .set_usize("datafusion.execution.target_partitions", threads());
    let options = config.options_mut();
    for (key, value) in overrides {
        options.set(key, value)?;
    }
    Ok(SessionContext::new_with_config(config))
}

/// Process-wide runtime. It is never dropped, so no runtime is ever torn down from
/// inside an async context or an FFI release callback.
fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(threads())
            .thread_name("gamma-lake")
            .build()
            .expect("failed to start the gamma-lake tokio runtime")
    })
}

/// Handle to the shared runtime, for work started from other executors.
pub(crate) fn handle() -> Handle {
    runtime().handle().clone()
}

/// Runs `future` to completion on the shared runtime. Fails when called from inside a tokio
/// runtime, where blocking would panic.
pub(crate) fn block_on<F: Future>(future: F) -> Result<F::Output> {
    if Handle::try_current().is_ok() {
        return Err(Error::InvalidArgument(
            "the blocking gamma-lake API cannot be called from inside a tokio runtime".into(),
        ));
    }
    Ok(runtime().block_on(future))
}

/// Runs the CPU-bound `f` on the blocking pool, holding one of the [`cpu_permits`].
pub(crate) async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let permit = cpu_permits()
        .acquire()
        .await
        .expect("the CPU permits are never closed");
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await
    .map_err(|e| Error::DataFusion(DataFusionError::External(Box::new(e))))?
}

#[cfg(test)]
mod tests {
    use super::{parse_threads, session_context, threads};

    #[test]
    fn parses_thread_counts() {
        assert_eq!(parse_threads(Some("4")), Some(4));
        assert_eq!(parse_threads(Some(" 2 ")), Some(2));
        for value in [None, Some(""), Some("0"), Some("-1"), Some("four")] {
            assert_eq!(parse_threads(value), None);
        }
        assert_eq!(parse_threads(Some("99999999999999999999999")), None);
    }

    #[test]
    fn sessions_use_the_thread_count() {
        let ctx = session_context(&[("datafusion.execution.batch_size", "7".into())]).unwrap();
        let config = ctx.state().config().clone();
        assert_eq!(config.target_partitions(), threads());
        assert_eq!(config.batch_size(), 7);
        let options = config.options();
        assert!(!options.sql_parser.enable_ident_normalization);
        assert!(options.execution.parquet.schema_force_view_types);
        assert_eq!(options.optimizer.hash_join_inlist_pushdown_max_size, 0);
        assert_eq!(
            options
                .optimizer
                .hash_join_inlist_pushdown_max_distinct_values,
            0
        );
        assert!(session_context(&[("datafusion.not.a.key", "1".into())]).is_err());
    }
}
