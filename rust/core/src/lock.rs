//! Locking the tables a write commits to.
//!
//! Some object stores cannot commit a Delta table version atomically, so concurrent writers
//! need an external lock. A [`TableLock`] in [`LakeOptions::lock`](crate::LakeOptions::lock)
//! is held around every commit to one table: its creation, appends, merges, metadata
//! changes and restores. Reads never take it.
//!
//! The lock serialises commits; it does not make a whole write atomic. Writers that extend
//! the index concurrently must be coordinated around the whole write by the caller.
//!
//! A failure to release after a successful commit is an [`Error::Write`] that lists the
//! table as committed, since its commit may be visible.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::sync::{mpsc, Arc};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use futures::channel::oneshot;

use crate::error::{Error, Result};
use crate::predicate::Dnf;
use crate::store::{CreateOptions, StoredTable, TableScan, TableStore};

/// The error a lock callback reports.
pub type LockError = Box<dyn StdError + Send + Sync>;

/// A lock on the tables of a lake, keyed by [`TableStore::table_location`].
///
/// One acquisition is acquired and released on the same thread, which blocks meanwhile, so
/// thread-affine locks work. Acquisitions for different tables may run concurrently; one
/// write never acquires the same table twice at a time.
pub trait TableLock: Send + Sync {
    /// Blocks until table `target` is locked.
    fn acquire(&self, target: &str) -> Result<Box<dyn LockGuard>, LockError>;
}

/// A held [`TableLock`].
pub trait LockGuard {
    /// Releases the lock.
    fn release(self: Box<Self>) -> Result<(), LockError>;
}

/// Runs `op` on table `name` while `target`, its location, is locked. An error of `op` takes
/// precedence over a failure to release. A failure to release after `op` succeeded is an
/// [`Error::Write`] listing `name` as committed, since `op` may have committed to it.
async fn locked<T, F>(lock: &Arc<dyn TableLock>, target: String, name: &str, op: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let (held_tx, held_rx) = oneshot::channel::<Result<(), LockError>>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let (done_tx, done_rx) = oneshot::channel::<Result<(), LockError>>();
    let thread_lock = lock.clone();
    let thread_target = target.clone();
    std::thread::Builder::new()
        .name("gamma-lake-lock".into())
        .spawn(move || {
            let guard = match thread_lock.acquire(&thread_target) {
                Ok(guard) => guard,
                Err(e) => {
                    let _ = held_tx.send(Err(e));
                    return;
                }
            };
            let _ = held_tx.send(Ok(()));
            // Returns when told to release or when the waiting operation is dropped.
            let _ = release_rx.recv();
            let _ = done_tx.send(guard.release());
        })
        .map_err(|e| lock_error(&target, Box::new(e)))?;
    let error = |source| lock_error(&target, source);
    match held_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(error(e)),
        Err(_) => return Err(error("the lock thread exited".into())),
    }
    let result = op.await;
    let _ = release_tx.send(());
    let released = match done_rx.await {
        Ok(released) => released,
        Err(_) => Err("the lock thread exited".into()),
    };
    match (result, released) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(e)) => Err(Error::Write {
            phase: format!("releasing the lock on table {name}"),
            committed: vec![name.to_string()],
            source: Box::new(error(e)),
        }),
        (Err(e), _) => Err(e),
    }
}

fn lock_error(target: &str, source: LockError) -> Error {
    Error::Lock {
        target: target.to_string(),
        source,
    }
}

/// A [`TableStore`] whose commits hold `lock`. A commit goes through the handle it was made
/// on, so it fails with [`Error::Conflict`] if another commit came first; with the lock
/// held, the inner store's check of that is reliable even without atomic commits.
#[derive(Clone)]
pub(crate) struct LockingStore {
    inner: Arc<dyn TableStore>,
    lock: Arc<dyn TableLock>,
}

impl LockingStore {
    pub(crate) fn new(inner: Arc<dyn TableStore>, lock: Arc<dyn TableLock>) -> Self {
        Self { inner, lock }
    }
}

impl fmt::Debug for LockingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LockingStore")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableStore for LockingStore {
    fn location(&self) -> &str {
        self.inner.location()
    }

    fn table_location(&self, name: &str) -> Result<String> {
        self.inner.table_location(name)
    }

    async fn open(&self, name: &str) -> Result<Option<Box<dyn StoredTable>>> {
        Ok(self
            .inner
            .open(name)
            .await?
            .map(|table| self.wrap(name, table)))
    }

    async fn create(&self, name: &str, batch: RecordBatch, options: &CreateOptions) -> Result<()> {
        let target = self.inner.table_location(name)?;
        locked(
            &self.lock,
            target,
            name,
            self.inner.create(name, batch, options),
        )
        .await
    }
}

impl LockingStore {
    fn wrap(&self, name: &str, table: Box<dyn StoredTable>) -> Box<dyn StoredTable> {
        Box::new(LockingTable {
            store: self.clone(),
            name: name.to_string(),
            table,
        })
    }

    /// Runs `op`, a commit to table `name`, while the table is locked.
    async fn commit<T>(
        &self,
        name: &str,
        op: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        let target = self.inner.table_location(name)?;
        locked(&self.lock, target, name, op).await
    }
}

struct LockingTable {
    store: LockingStore,
    name: String,
    table: Box<dyn StoredTable>,
}

#[async_trait]
impl StoredTable for LockingTable {
    fn schema(&self) -> SchemaRef {
        self.table.schema()
    }

    async fn scan(
        &self,
        columns: Option<&[String]>,
        dnf: &Dnf,
        batch_size: usize,
    ) -> Result<TableScan> {
        self.table.scan(columns, dnf, batch_size).await
    }

    async fn append(self: Box<Self>, batch: RecordBatch) -> Result<()> {
        let Self { store, name, table } = *self;
        store.commit(&name, table.append(batch)).await
    }

    async fn append_merging(self: Box<Self>, batch: RecordBatch) -> Result<()> {
        let Self { store, name, table } = *self;
        store.commit(&name, table.append_merging(batch)).await
    }

    async fn merge(
        self: Box<Self>,
        source: RecordBatch,
        keys: &[String],
        update: &[String],
    ) -> Result<()> {
        let Self { store, name, table } = *self;
        store.commit(&name, table.merge(source, keys, update)).await
    }

    async fn set_column_metadata(
        self: Box<Self>,
        column: &str,
        metadata: HashMap<String, String>,
    ) -> Result<Box<dyn StoredTable>> {
        let Self { store, name, table } = *self;
        let table = store
            .commit(&name, table.set_column_metadata(column, metadata))
            .await?;
        Ok(store.wrap(&name, table))
    }

    async fn restore(self: Box<Self>, at: i64) -> Result<bool> {
        let Self { store, name, table } = *self;
        store.commit(&name, table.restore(at)).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use futures::FutureExt;

    use super::*;

    /// Counts acquisitions and releases; `acquire` waits until `go` is sent.
    #[derive(Default)]
    struct Counting {
        acquired: AtomicUsize,
        released: Arc<AtomicUsize>,
        gate: Mutex<Option<mpsc::Receiver<()>>>,
    }

    struct CountingGuard(Arc<AtomicUsize>);

    impl TableLock for Counting {
        fn acquire(&self, _: &str) -> Result<Box<dyn LockGuard>, LockError> {
            if let Some(gate) = self.gate.lock().unwrap().take() {
                gate.recv().unwrap();
            }
            self.acquired.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(CountingGuard(self.released.clone())))
        }
    }

    impl LockGuard for CountingGuard {
        fn release(self: Box<Self>) -> Result<(), LockError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn released_once(lock: &Counting) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while lock.released.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(lock.acquired.load(Ordering::SeqCst), 1);
        assert_eq!(lock.released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropped_while_acquiring_releases_once_acquired() {
        let (go, gate) = mpsc::channel();
        let counting = Arc::new(Counting {
            gate: Mutex::new(Some(gate)),
            ..Default::default()
        });
        let lock: Arc<dyn TableLock> = counting.clone();
        let pending = locked(&lock, "t".into(), "t", async { Ok(()) }).now_or_never();
        assert!(pending.is_none());
        go.send(()).unwrap();
        released_once(&counting);
    }

    #[test]
    fn dropped_while_holding_releases() {
        let counting = Arc::new(Counting::default());
        let lock: Arc<dyn TableLock> = counting.clone();
        let mut op = Box::pin(locked(
            &lock,
            "t".into(),
            "t",
            futures::future::pending::<Result<()>>(),
        ));
        let deadline = Instant::now() + Duration::from_secs(10);
        while counting.acquired.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            assert!((&mut op).now_or_never().is_none());
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!((&mut op).now_or_never().is_none());
        assert_eq!(counting.released.load(Ordering::SeqCst), 0);
        drop(op);
        released_once(&counting);
    }

    #[test]
    fn a_panicking_operation_releases() {
        let counting = Arc::new(Counting::default());
        let lock: Arc<dyn TableLock> = counting.clone();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            futures::executor::block_on(locked::<(), _>(&lock, "t".into(), "t", async {
                panic!("boom")
            }))
        }));
        assert!(panicked.is_err());
        released_once(&counting);
    }
}
