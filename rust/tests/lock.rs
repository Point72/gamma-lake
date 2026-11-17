//! Table locks: every commit holds the lock on its table, reads take none, and lock failures
//! fail the operation.

mod common;

use std::collections::{BTreeSet, HashSet};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use common::{add, delta_store, frame, index_schema, keys, tick};
use gamma_lake::arrow::record_batch::RecordBatch;
use gamma_lake::{
    AddOptions, Annotation, Dnf, Error, Lake, LakeOptions, LockError, LockGuard, MemoryStore,
    OverlapMode, TableLock, TableStore,
};
use tempfile::TempDir;

#[derive(Default)]
struct State {
    held: HashSet<String>,
    acquired: Vec<String>,
    released: usize,
}

/// Records acquisitions, fails on reentry and checks each release runs on its acquiring
/// thread.
#[derive(Default)]
struct RecordingLock {
    state: Mutex<State>,
    fail_acquire: Option<&'static str>,
    fail_release: Option<&'static str>,
}

struct Guard {
    lock: Arc<RecordingLock>,
    target: String,
    thread: ThreadId,
}

struct Shared(Arc<RecordingLock>);

impl TableLock for Shared {
    fn acquire(&self, target: &str) -> Result<Box<dyn LockGuard>, LockError> {
        if self.0.fail_acquire.is_some_and(|f| target.contains(f)) {
            return Err("no lock for you".into());
        }
        let mut state = self.0.state.lock().unwrap();
        if !state.held.insert(target.to_string()) {
            return Err(format!("{target} is already locked").into());
        }
        state.acquired.push(target.to_string());
        Ok(Box::new(Guard {
            lock: self.0.clone(),
            target: target.to_string(),
            thread: std::thread::current().id(),
        }))
    }
}

impl LockGuard for Guard {
    fn release(self: Box<Self>) -> Result<(), LockError> {
        assert_eq!(std::thread::current().id(), self.thread);
        let mut state = self.lock.state.lock().unwrap();
        assert!(state.held.remove(&self.target));
        state.released += 1;
        if self
            .lock
            .fail_release
            .is_some_and(|f| self.target.contains(f))
        {
            return Err("cannot unlock".into());
        }
        Ok(())
    }
}

/// Rows of symbol `a` at `ts`, with the feature `f`.
fn rows(ts: impl IntoIterator<Item = i64>) -> RecordBatch {
    frame(&keys(ts, &["a"]), &[("f", 1.0)])
}

fn options(lock: &Arc<RecordingLock>) -> LakeOptions {
    LakeOptions {
        primary_sort_key: "ts".into(),
        lock: Some(Arc::new(Shared(lock.clone()))),
        ..Default::default()
    }
}

/// A Delta lake (`delta`) or an in-memory lake, with its store.
fn lake(
    lock: &Arc<RecordingLock>,
    delta: bool,
) -> (TempDir, Result<Lake, Error>, Arc<dyn TableStore>) {
    let dir = TempDir::new().unwrap();
    if delta {
        let path = dir.path().to_str().unwrap().to_string();
        let store = delta_store(&path).unwrap();
        let lake = gamma_lake::create(store.clone(), index_schema(), options(lock));
        (dir, lake, Arc::new(store))
    } else {
        let memory = MemoryStore::new();
        let lake = Lake::create_with_store(Arc::new(memory.clone()), index_schema(), options(lock));
        (dir, lake, Arc::new(memory))
    }
}

fn acquired(lock: &RecordingLock) -> Vec<String> {
    let state = lock.state.lock().unwrap();
    assert!(state.held.is_empty());
    assert_eq!(state.released, state.acquired.len());
    state.acquired.clone()
}

#[test]
fn every_commit_holds_the_lock_on_its_table_and_reads_take_none() {
    for delta in [true, false] {
        let lock = Arc::new(RecordingLock::default());
        let (_dir, lake, store) = lake(&lock, delta);
        let lake = lake.unwrap();
        let location = |name: &str| store.table_location(name).unwrap();
        let metadata = BTreeSet::from([
            location("index"),
            location("feature_metadata"),
            location("table_metadata"),
            location("transaction_log"),
        ]);
        assert_eq!(BTreeSet::from_iter(acquired(&lock)), metadata);

        let created = tick();
        let mut seen = acquired(&lock).len();
        let mut step = |what: &str, run: &dyn Fn()| {
            run();
            let all = acquired(&lock);
            assert!(all.len() > seen, "{what} took no lock");
            seen = all.len();
        };
        step("a new table", &|| {
            add(&lake, rows([1, 2]), OverlapMode::Copy)
        });
        step("an append", &|| add(&lake, rows([3]), OverlapMode::Copy));
        step("a copy", &|| add(&lake, rows([2, 3]), OverlapMode::Copy));
        step("a merge", &|| add(&lake, rows([3, 4]), OverlapMode::Merge));
        step("padding", &|| {
            lake.add_index_rows(vec![frame(&keys([9], &["a"]), &[])])
                .unwrap()
        });
        step("annotations", &|| {
            let note = |name: &str, comment: &str| Annotation {
                feature_name: name.into(),
                table_addr: None,
                comment: Some(comment.into()),
                tags: None,
            };
            lake.annotate(&[note("f", "one"), note("f", "two")])
                .unwrap()
        });

        let before = acquired(&lock).len();
        lake.read(&["f"], &[], &Dnf::default())
            .unwrap()
            .for_each(|b| drop(b.unwrap()));
        lake.describe(&[]).unwrap();
        assert_eq!(acquired(&lock).len(), before);
        step("a restore", &|| lake.restore(created).unwrap());
        // Every target is a table of this lake.
        let prefix = format!("{}/", location("index").strip_suffix("/index").unwrap());
        assert!(acquired(&lock).iter().all(|t| t.starts_with(&prefix)));
    }
}

#[test]
fn lock_failures_fail_the_operation() {
    let lock = Arc::new(RecordingLock {
        fail_acquire: Some("feature_metadata"),
        ..Default::default()
    });
    for delta in [true, false] {
        let err = lake(&lock, delta).1.unwrap_err();
        assert!(matches!(err, Error::Lock { .. }), "{err}");
        assert!(err.to_string().contains("no lock for you"), "{err}");
    }

    // A failed release fails the commit it ends, after the commit is made.
    let lock = Arc::new(RecordingLock {
        fail_release: Some("feature_metadata"),
        ..Default::default()
    });
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let err =
        gamma_lake::create(delta_store(path).unwrap(), index_schema(), options(&lock)).unwrap_err();
    assert!(err.to_string().contains("cannot unlock"), "{err}");
    acquired(&lock);
    assert!(dir.path().join("feature_metadata").exists());
    assert!(matches!(
        gamma_lake::open(delta_store(path).unwrap(), options(&Arc::default())),
        Err(Error::NotInitialized(_))
    ));
}

#[test]
fn a_failed_release_after_a_commit_counts_the_table_as_committed() {
    // The first table a write commits, and a later one. The rollback restores them, and
    // fails to release them again.
    for (fails, expected) in [("delta_", 1), ("/index", 2)] {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        let lake = gamma_lake::create(
            delta_store(path).unwrap(),
            index_schema(),
            options(&Arc::default()),
        )
        .unwrap();
        add(&lake, rows([1, 2]), OverlapMode::Copy);
        let lock = Arc::new(RecordingLock {
            fail_release: Some(fails),
            ..Default::default()
        });
        let lake = gamma_lake::open(delta_store(path).unwrap(), options(&lock)).unwrap();
        let err = lake
            .add(vec![rows([3])], &AddOptions::default())
            .unwrap_err();
        let Error::Write {
            committed, source, ..
        } = &err
        else {
            panic!("{err}");
        };
        assert_eq!(committed.len(), expected, "{err}");
        assert!(committed[0].starts_with("delta_"), "{err}");
        assert!(matches!(**source, Error::Lock { .. }), "{err}");
        assert!(err.to_string().contains("cannot unlock"), "{err}");
        assert!(err.to_string().contains("rolling back a write"), "{err}");
        acquired(&lock);
    }
}
