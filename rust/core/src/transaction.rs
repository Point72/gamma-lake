//! The transaction log: an append-only table with a row for each state a write goes through,
//! as `GammaFeatureLake` keeps it. A write that fails after committing is rolled back: its
//! commits, marked with its transaction id, are undone on each table it committed to.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};

use crate::admin::after;
use crate::error::{Error, Result};
use crate::lake::Lake;
use crate::store::{CreateOptions, Entangled};
use crate::write::{assemble, now_micros, Commits};

/// Characters of an error message the log keeps.
const MAX_ERROR_CHARS: usize = 2000;
/// Attempts at appending one row while other writers' appends conflict with it.
const ATTEMPTS: usize = 10;

/// Schema of a new transaction log.
pub(crate) fn schema() -> SchemaRef {
    let string = |name| Field::new(name, DataType::Utf8, true);
    let time = |name| {
        Field::new(
            name,
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        )
    };
    Arc::new(Schema::new(vec![
        string("transaction_id"),
        string("status"),
        string("owner"),
        string("signal_type"),
        time("started_at"),
        time("updated_at"),
        string("error"),
    ]))
}

#[derive(Clone, Copy)]
enum Status {
    Pending,
    Success,
    Failed,
    RolledBack,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Pending => "pending",
            Status::Success => "success",
            Status::Failed => "failed",
            Status::RolledBack => "rolled_back",
        }
    }
}

/// The phase, tables that may hold the write and cause of `error`, a failed write or undo.
fn parts(error: Error) -> (String, Vec<String>, Box<Error>) {
    match error {
        Error::Write {
            phase,
            committed,
            source,
        } => (phase, committed, source),
        error => ("writing".to_string(), vec![], Box::new(error)),
    }
}

/// `committed` and the tables of `more` not in it.
fn union(mut committed: Vec<String>, more: Vec<String>) -> Vec<String> {
    let more: Vec<_> = more
        .into_iter()
        .filter(|t| !committed.contains(t))
        .collect();
    committed.extend(more);
    committed
}

/// The outcome of undoing a failed write.
enum Undo {
    /// Every table was undone; whether any commits were.
    Done(bool),
    /// Why the commits to a table cannot be undone without undoing another writer's, and the
    /// tables that may still hold the write.
    Refused(String, Vec<String>),
}

/// One logged write.
struct Transaction<'a> {
    id: String,
    owner: &'a str,
    signal_type: &'a str,
    started_at: i64,
}

impl Lake {
    /// Runs `body`, a write that returns the tables it committed and counts its commits in
    /// `commits`, as a transaction whose id marks those commits.
    ///
    /// Logs `pending`, then `success` once `body` succeeds. When `body` or that row fails,
    /// logs `failed`, undoes the commits `body` made, newest table first, and logs
    /// `rolled_back`; the error then names no committed tables, or is the bare conflict, as
    /// when a write fails before its first commit. Commits are found by their mark, so one
    /// that failed after it landed is undone too, and other writers' commits are kept. A failed
    /// undo leaves the transaction `failed` and is an [`Error::Write`] naming the tables that
    /// may hold the write; when another writer rewrote what the write committed, so that
    /// undoing it would undo that writer's commit too, its cause is the write's own error.
    /// Must be called with the write lock held.
    pub(crate) async fn transaction(
        &self,
        owner: &str,
        signal_type: &str,
        commits: &Commits,
        body: impl Future<Output = Result<Vec<String>>>,
    ) -> Result<()> {
        let tx = Transaction {
            id: commits.tag().to_string(),
            owner,
            signal_type,
            started_at: now_micros(),
        };
        self.log(&tx, Status::Pending, None).await?;
        let error = match body.await {
            Ok(committed) => match self.log(&tx, Status::Success, None).await {
                Ok(()) => return Ok(()),
                Err(e) => after("logging the transaction".into(), committed, e),
            },
            Err(e) => e,
        };
        Err(self.roll_back(&tx, commits, error).await)
    }

    async fn roll_back(&self, tx: &Transaction<'_>, commits: &Commits, error: Error) -> Error {
        let message = error.to_string();
        // A row that cannot be logged is not reported: the write's error is the one to act on.
        self.log(tx, Status::Failed, Some(&message)).await.ok();
        match self.undo(commits).await {
            Ok(Undo::Done(undone)) => {
                self.log(tx, Status::RolledBack, Some(&message)).await.ok();
                match error {
                    // As if nothing had been committed; see `commit`.
                    Error::Write {
                        phase,
                        committed,
                        source,
                    } if !committed.is_empty() || undone => match *source {
                        conflict @ Error::Conflict { .. } => conflict,
                        source => Error::Write {
                            phase: format!("{phase} (rolled back)"),
                            committed: vec![],
                            source: Box::new(source),
                        },
                    },
                    error => error,
                }
            }
            Ok(Undo::Refused(reason, held)) => {
                let (phase, committed, source) = parts(error);
                Error::Write {
                    phase: format!("{phase} (not rolled back: {reason})"),
                    committed: union(committed, held),
                    source,
                }
            }
            Err(e) => {
                let (undoing, held, cause) = parts(e);
                let (phase, committed, source) = parts(error);
                Error::Write {
                    phase: format!(
                        "{undoing}, rolling back a write that failed while {phase}: {source}"
                    ),
                    committed: union(committed, held),
                    source: cause,
                }
            }
        }
    }

    /// Undoes the commits of `commits` on each of its tables, newest first, retrying an undo
    /// that conflicts with another writer's commit. Stops at a table whose commits cannot be
    /// undone alone, and otherwise returns whether any were undone. A failure is an
    /// [`Error::Write`] naming the tables that may still hold the write.
    async fn undo(&self, commits: &Commits) -> Result<Undo> {
        let tables = commits.tables();
        let mut undone = false;
        for (i, commit) in tables.iter().enumerate().rev() {
            let addr = &commit.addr;
            let held = || tables[..=i].iter().map(|c| c.addr.clone()).collect();
            let Some(read) = commit.read else {
                let reason = format!("the store does not number the versions of table {addr}");
                return Ok(Undo::Refused(reason, held()));
            };
            let revert = async {
                let mut attempt = 1;
                loop {
                    let table = self.table(addr).await?;
                    if table.version() == Some(read) {
                        return Ok(false);
                    }
                    match table.revert(commits.tag(), read).await {
                        Err(Error::Conflict { .. }) if attempt < ATTEMPTS => attempt += 1,
                        result => return result,
                    }
                }
            };
            match revert.await {
                Ok(reverted) => undone |= reverted,
                Err(Error::Store(e)) if e.is::<Entangled>() => {
                    return Ok(Undo::Refused(e.to_string(), held()));
                }
                Err(e) => return Err(after(format!("restoring table {addr}"), held(), e)),
            }
        }
        Ok(Undo::Done(undone))
    }

    /// Appends the row of `tx` at `status`, creating the log when it is missing.
    async fn log(&self, tx: &Transaction<'_>, status: Status, error: Option<&str>) -> Result<()> {
        let path = &self.options.transaction_log_path;
        let error: Option<String> = error.map(|e| e.chars().take(MAX_ERROR_CHARS).collect());
        let row = || {
            let string = |s: Option<&str>| Arc::new(StringArray::from(vec![s])) as ArrayRef;
            let time = |t: i64| {
                Arc::new(TimestampMicrosecondArray::from(vec![t]).with_timezone("UTC")) as ArrayRef
            };
            HashMap::from([
                ("transaction_id", string(Some(&tx.id))),
                ("status", string(Some(status.as_str()))),
                ("owner", string(Some(tx.owner))),
                ("signal_type", string(Some(tx.signal_type))),
                ("started_at", time(tx.started_at)),
                ("updated_at", time(now_micros())),
                ("error", string(error.as_deref())),
            ])
        };
        let mut attempt = 1;
        loop {
            let result = match self.store.open(path).await? {
                Some(table) => {
                    let batch = assemble(&table.schema(), row(), 1, path)?;
                    table.append(batch).await
                }
                None => {
                    let batch = assemble(&schema(), row(), 1, path)?;
                    let created = self
                        .store
                        .create(path, batch, &CreateOptions::default())
                        .await;
                    // Another writer may have created the log first.
                    match created {
                        Err(_) if attempt < ATTEMPTS && self.store.open(path).await?.is_some() => {
                            Err(Error::Conflict {
                                table: path.clone(),
                            })
                        }
                        result => result,
                    }
                }
            };
            match result {
                Err(Error::Conflict { .. }) if attempt < ATTEMPTS => attempt += 1,
                result => return result,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
    use arrow::compute::{cast, concat_batches};
    use arrow::record_batch::RecordBatchReader;

    use super::*;
    use crate::predicate::Dnf;
    use crate::runtime::block_on;
    use crate::{AddOptions, LakeOptions, LockError, LockGuard, MemoryStore, TableLock};

    /// Fails the acquisitions of each named table whose outcome is `true`, in order.
    #[derive(Default)]
    struct Failing(Mutex<HashMap<&'static str, Vec<bool>>>);

    impl Failing {
        fn fail(&self, name: &'static str, outcomes: &[bool]) {
            let outcomes = outcomes.iter().rev().copied().collect();
            self.0.lock().unwrap().insert(name, outcomes);
        }
    }

    struct Released;

    impl LockGuard for Released {
        fn release(self: Box<Self>) -> std::result::Result<(), LockError> {
            Ok(())
        }
    }

    impl TableLock for Failing {
        fn acquire(&self, target: &str) -> std::result::Result<Box<dyn LockGuard>, LockError> {
            let name = target.rsplit('/').next().unwrap();
            match self.0.lock().unwrap().get_mut(name).and_then(Vec::pop) {
                Some(true) => Err(format!("cannot lock {name}").into()),
                _ => Ok(Box::new(Released)),
            }
        }
    }

    fn lake(lock: &Arc<Failing>, store: &MemoryStore, log: &str) -> Lake {
        let index = Schema::new(vec![
            Field::new("ts", DataType::Int64, true),
            Field::new("sym", DataType::Utf8, true),
        ]);
        let options = LakeOptions {
            primary_sort_key: "ts".into(),
            transaction_log_path: log.into(),
            lock: Some(lock.clone()),
            ..Default::default()
        };
        Lake::create_with_store(Arc::new(store.clone()), Arc::new(index), options).unwrap()
    }

    fn frame(ts: &[i64], value: f64) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("ts", DataType::Int64, true),
            Field::new("sym", DataType::Utf8, true),
            Field::new("f", DataType::Float64, true),
        ]);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ts.to_vec())),
            Arc::new(StringArray::from_iter_values(ts.iter().map(|_| "a"))),
            Arc::new(Float64Array::from(vec![value; ts.len()])),
        ];
        RecordBatch::try_new(Arc::new(schema), columns).unwrap()
    }

    fn add(lake: &Lake, ts: &[i64], value: f64) -> Result<()> {
        let options = AddOptions {
            owner: "me".into(),
            ..Default::default()
        };
        lake.add(vec![frame(ts, value)], &options)
    }

    fn collect(stream: crate::BatchStream) -> RecordBatch {
        let schema = stream.schema();
        let batches = stream.collect::<std::result::Result<Vec<_>, _>>().unwrap();
        concat_batches(&schema, &batches).unwrap()
    }

    fn strings(batch: &RecordBatch, column: &str) -> Vec<Option<String>> {
        let values = cast(batch.column_by_name(column).unwrap(), &DataType::Utf8).unwrap();
        let values = values.as_any().downcast_ref::<StringArray>().unwrap();
        values.iter().map(|v| v.map(str::to_string)).collect()
    }

    fn statuses(lake: &Lake) -> Vec<String> {
        let log = collect(lake.transaction_log(&Dnf::default()).unwrap());
        strings(&log, "status").into_iter().flatten().collect()
    }

    fn values(lake: &Lake) -> Vec<f64> {
        let read = collect(lake.read(&["f"], &[], &Dnf::default()).unwrap());
        let f = read.column_by_name("f").unwrap();
        let f = f.as_any().downcast_ref::<Float64Array>().unwrap();
        f.iter().flatten().collect()
    }

    #[test]
    fn a_write_is_logged_pending_then_success() {
        let lake = lake(&Arc::default(), &MemoryStore::new(), "transaction_log");
        assert_eq!(lake.transaction_log_schema().unwrap(), schema());
        assert_eq!(statuses(&lake), Vec::<String>::new());

        add(&lake, &[1, 2], 1.0).unwrap();
        let log = collect(lake.transaction_log(&Dnf::default()).unwrap());
        assert_eq!(statuses(&lake), ["pending", "success"]);
        let ids = strings(&log, "transaction_id");
        assert!(ids[0].is_some() && ids[0] == ids[1]);
        assert_eq!(
            strings(&log, "owner"),
            [Some("me".into()), Some("me".into())]
        );
        let signal = Some("feature".to_string());
        assert_eq!(strings(&log, "signal_type"), [signal.clone(), signal]);
        assert_eq!(strings(&log, "error"), [None, None]);
    }

    #[test]
    fn rejected_input_is_not_logged() {
        let lake = lake(&Arc::default(), &MemoryStore::new(), "transaction_log");
        assert!(lake.add(vec![], &AddOptions::default()).is_err());
        assert_eq!(statuses(&lake), Vec::<String>::new());
    }

    #[test]
    fn a_failed_write_is_rolled_back_and_later_writes_do_not_see_it() {
        let lock = Arc::new(Failing::default());
        let lake = lake(&lock, &MemoryStore::new(), "transaction_log");
        add(&lake, &[1, 2], 1.0).unwrap();

        // The write appends to the table of `f`, then cannot lock the index.
        lock.fail("index", &[true]);
        let err = add(&lake, &[3, 4], 2.0).unwrap_err();
        let Error::Write {
            phase,
            committed,
            source,
        } = &err
        else {
            panic!("{err}");
        };
        assert_eq!(phase, "writing table index (rolled back)");
        assert!(committed.is_empty(), "{err}");
        assert!(matches!(**source, Error::Lock { .. }), "{err}");
        assert_eq!(values(&lake), [1.0, 1.0]);
        assert_eq!(statuses(&lake)[2..], ["pending", "failed", "rolled_back"]);
        let log = collect(lake.transaction_log(&Dnf::default()).unwrap());
        let logged = strings(&log, "error")[4].clone().unwrap();
        assert!(logged.contains("cannot lock index"), "{logged}");

        add(&lake, &[5], 3.0).unwrap();
        assert_eq!(values(&lake), [1.0, 1.0, 3.0]);
    }

    #[test]
    fn a_rolled_back_write_of_a_new_feature_leaves_its_table_unreferenced() {
        let lock = Arc::new(Failing::default());
        let lake = lake(&lock, &MemoryStore::new(), "transaction_log");
        lock.fail("feature_metadata", &[true]);
        let err = add(&lake, &[1, 2], 1.0).unwrap_err();
        assert!(
            matches!(&err, Error::Write { committed, .. } if committed.is_empty()),
            "{err}"
        );
        assert_eq!(collect(lake.index(&Dnf::default()).unwrap()).num_rows(), 0);
        assert!(lake.read(&["f"], &[], &Dnf::default()).is_err());

        add(&lake, &[1, 2], 1.0).unwrap();
        assert_eq!(values(&lake), [1.0, 1.0]);
    }

    #[test]
    fn a_failed_rollback_leaves_the_transaction_failed() {
        let lock = Arc::new(Failing::default());
        let lake = lake(&lock, &MemoryStore::new(), "transaction_log");
        add(&lake, &[1, 2], 1.0).unwrap();

        // The write commits the table of `f` and the index, then cannot lock the table
        // metadata; the rollback cannot lock the index.
        lock.fail("table_metadata", &[true]);
        lock.fail("index", &[false, true]);
        let err = add(&lake, &[3, 4], 2.0).unwrap_err();
        let Error::Write {
            phase,
            committed,
            source,
        } = &err
        else {
            panic!("{err}");
        };
        let expected = "restoring table index, rolling back a write that failed while writing \
                        table table_metadata";
        assert!(phase.starts_with(expected), "{err}");
        assert_eq!(committed.len(), 2, "{err}");
        assert!(matches!(**source, Error::Lock { .. }), "{err}");
        assert_eq!(statuses(&lake)[2..], ["pending", "failed"]);
    }

    fn index_ts(lake: &Lake) -> Vec<i64> {
        let index = collect(lake.index(&Dnf::default()).unwrap());
        let ts = index.column_by_name("ts").unwrap();
        let mut ts: Vec<i64> = ts
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        ts.sort();
        ts
    }

    fn row(ts: i64) -> RecordBatch {
        frame(&[ts], 0.0).project(&[0, 1]).unwrap()
    }

    #[derive(Clone, Copy)]
    enum Other {
        Append(i64),
        Merge(i64),
    }

    async fn commit_other(lake: &Lake, other: Option<Other>) -> Result<()> {
        let path = &lake.options.index_path;
        let keys = ["ts".to_string(), "sym".to_string()];
        match other {
            None => Ok(()),
            Some(Other::Append(ts)) => lake.table(path).await?.append(row(ts)).await,
            Some(Other::Merge(ts)) => lake.table(path).await?.merge(row(ts), &keys, &[]).await,
        }
    }

    fn fail_appending(lake: &Lake, before: Option<Other>, after: Option<Other>) -> Error {
        let path = &lake.options.index_path;
        let commits = Commits::default();
        let body = async {
            commit_other(lake, before).await?;
            let index = commits.tagged(lake.table(path).await?);
            let read = index.version();
            commits.count(path, read, index.append(row(8)).await)?;
            commit_other(lake, after).await?;
            Err(Error::Write {
                phase: "testing".into(),
                committed: vec![path.clone()],
                source: Box::new(Error::InvalidArgument("boom".into())),
            })
        };
        block_on(lake.transaction("me", "feature", &commits, body))
            .unwrap()
            .unwrap_err()
    }

    #[test]
    fn a_rollback_keeps_another_writers_commits() {
        let lake = lake(&Arc::default(), &MemoryStore::new(), "transaction_log");
        add(&lake, &[1, 2], 1.0).unwrap();
        let err = fail_appending(&lake, Some(Other::Append(7)), None);
        assert!(err.to_string().contains("(rolled back)"), "{err}");
        let err = fail_appending(&lake, None, Some(Other::Append(9)));
        assert!(err.to_string().contains("(rolled back)"), "{err}");
        assert_eq!(index_ts(&lake), [1, 2, 7, 9]);
        let rolled_back = ["pending", "failed", "rolled_back"].repeat(2);
        assert_eq!(statuses(&lake)[2..], rolled_back);
    }

    #[test]
    fn a_commit_that_failed_after_it_landed_is_rolled_back() {
        let lake = lake(&Arc::default(), &MemoryStore::new(), "transaction_log");
        add(&lake, &[1, 2], 1.0).unwrap();
        let path = &lake.options.index_path;
        let commits = Commits::default();
        let body = async {
            let index = commits.tagged(lake.table(path).await?);
            let read = index.version();
            index.append(row(8)).await?;
            // As a Delta commit whose checkpoint fails after it landed.
            commits.count(
                path,
                read,
                Err::<(), _>(Error::Store("checkpoint failed".into())),
            )?;
            Ok(vec![])
        };
        let err = block_on(lake.transaction("me", "feature", &commits, body))
            .unwrap()
            .unwrap_err();
        assert!(matches!(err, Error::Store(_)), "{err}");
        assert_eq!(index_ts(&lake), [1, 2]);
        assert_eq!(statuses(&lake)[2..], ["pending", "failed", "rolled_back"]);
    }

    #[test]
    fn a_write_another_writer_rewrote_is_not_rolled_back() {
        let lake = lake(&Arc::default(), &MemoryStore::new(), "transaction_log");
        add(&lake, &[1, 2], 1.0).unwrap();
        let err = fail_appending(&lake, None, Some(Other::Merge(9)));
        let Error::Write {
            phase, committed, ..
        } = &err
        else {
            panic!("{err}");
        };
        assert!(
            phase.starts_with("testing (not rolled back: version"),
            "{err}"
        );
        assert!(phase.contains("rewrote what this write committed"), "{err}");
        assert_eq!(committed, &[lake.options.index_path.clone()]);
        assert_eq!(index_ts(&lake), [1, 2, 8, 9]);
        assert_eq!(statuses(&lake)[2..], ["pending", "failed"]);
    }

    #[test]
    fn a_lake_without_a_log_creates_it_on_first_write() {
        let store = MemoryStore::new();
        let options = LakeOptions {
            transaction_log_path: "other_log".into(),
            ..lake(&Arc::default(), &store, "transaction_log").options
        };
        let lake = Lake::open_with_store(Arc::new(store), options).unwrap();
        let missing = block_on(lake.store.open("other_log")).unwrap().unwrap();
        assert!(missing.is_none());
        assert_eq!(lake.transaction_log_schema().unwrap(), schema());
        assert_eq!(statuses(&lake), Vec::<String>::new());

        add(&lake, &[1, 2], 1.0).unwrap();
        assert_eq!(statuses(&lake), ["pending", "success"]);
    }
}
