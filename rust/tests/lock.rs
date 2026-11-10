//! Table locks: every commit holds the lock on its table, reads take none, and lock failures
//! fail the operation.

use std::collections::{BTreeSet, HashSet};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use gamma_lake::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use gamma_lake::arrow::datatypes::{DataType, Field, Schema};
use gamma_lake::arrow::record_batch::RecordBatch;
use gamma_lake::{
    AddOptions, Annotation, DeltaStore, Dnf, Error, Lake, LakeOptions, LockError, LockGuard,
    MemoryStore, OverlapMode, TableLock, TableStore,
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

fn index_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, true),
        Field::new("sym", DataType::Utf8, true),
    ]))
}

fn frame(ts: &[i64], feature: &str) -> RecordBatch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ts.to_vec())),
        Arc::new(StringArray::from_iter_values(ts.iter().map(|_| "a"))),
        Arc::new(Float64Array::from_iter_values(ts.iter().map(|&t| t as f64))),
    ];
    let schema = Schema::new(vec![
        Field::new("ts", DataType::Int64, true),
        Field::new("sym", DataType::Utf8, true),
        Field::new(feature, DataType::Float64, true),
    ]);
    RecordBatch::try_new(Arc::new(schema), columns).unwrap()
}

fn options(lock: &Arc<RecordingLock>) -> LakeOptions {
    LakeOptions {
        primary_sort_key: "ts".into(),
        lock: Some(Arc::new(Shared(lock.clone()))),
        ..Default::default()
    }
}

fn delta_store(path: &str) -> DeltaStore {
    DeltaStore::new(path, Default::default()).unwrap()
}

/// A Delta lake (`delta`) or an in-memory lake, with its store.
fn lake(
    lock: &Arc<RecordingLock>,
    delta: bool,
) -> (TempDir, Result<Lake, Error>, Arc<dyn TableStore>) {
    let dir = TempDir::new().unwrap();
    if delta {
        let path = dir.path().to_str().unwrap().to_string();
        let store = delta_store(&path);
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

fn add(lake: &Lake, batch: RecordBatch, overlap_mode: OverlapMode) {
    let options = AddOptions {
        overlap_mode,
        ..Default::default()
    };
    lake.add(vec![batch], &options).unwrap();
}

fn micros_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
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
        ]);
        assert_eq!(BTreeSet::from_iter(acquired(&lock)), metadata);

        std::thread::sleep(std::time::Duration::from_millis(5));
        let created = micros_now();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let mut seen = acquired(&lock).len();
        let mut step = |what: &str, run: &dyn Fn()| {
            run();
            let all = acquired(&lock);
            assert!(all.len() > seen, "{what} took no lock");
            seen = all.len();
        };
        step("a new table", &|| {
            add(&lake, frame(&[1, 2], "f"), OverlapMode::Copy)
        });
        step("an append", &|| {
            add(&lake, frame(&[3], "f"), OverlapMode::Copy)
        });
        step("a copy", &|| {
            add(&lake, frame(&[2, 3], "f"), OverlapMode::Copy)
        });
        step("a merge", &|| {
            add(&lake, frame(&[3, 4], "f"), OverlapMode::Merge)
        });
        step("padding", &|| {
            lake.add_index_rows(vec![frame(&[9], "ignored")]).unwrap()
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
    let err = gamma_lake::create(delta_store(path), index_schema(), options(&lock)).unwrap_err();
    assert!(err.to_string().contains("cannot unlock"), "{err}");
    acquired(&lock);
    assert!(dir.path().join("feature_metadata").exists());
    assert!(matches!(
        gamma_lake::open(delta_store(path), options(&Arc::default())),
        Err(Error::NotInitialized(_))
    ));
}

#[test]
fn a_table_has_one_lock_target_however_its_lake_is_named() {
    let dir = TempDir::new().unwrap();
    let real = dir.path().join("real");
    std::fs::create_dir(&real).unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let targets: BTreeSet<String> = [
        real.display().to_string(),
        format!("{}/", real.display()),
        format!("{}/./", link.display()),
        format!("file://{}", link.display()),
    ]
    .iter()
    .map(|base| {
        DeltaStore::new(base, Default::default())
            .unwrap()
            .table_location("features/f1")
            .unwrap()
    })
    .collect();
    assert_eq!(
        targets,
        BTreeSet::from([format!(
            "{}/features/f1",
            std::fs::canonicalize(&real).unwrap().display()
        )])
    );
    let s3 = DeltaStore::new("s3://bucket/lake/", Default::default()).unwrap();
    assert_eq!(
        s3.table_location("index").unwrap(),
        "s3://bucket/lake/index"
    );
}

#[test]
fn a_failed_release_after_a_commit_counts_the_table_as_committed() {
    // The first table a write commits, and a later one.
    for (fails, expected) in [("delta_", 1), ("/index", 2)] {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        gamma_lake::create(delta_store(path), index_schema(), options(&Arc::default())).unwrap();
        let lock = Arc::new(RecordingLock {
            fail_release: Some(fails),
            ..Default::default()
        });
        let lake = gamma_lake::open(delta_store(path), options(&lock)).unwrap();
        let err = lake
            .add(vec![frame(&[1, 2], "f")], &AddOptions::default())
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
        acquired(&lock);
    }
}
