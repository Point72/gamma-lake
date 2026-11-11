//! Native writes: every scenario ends by checking the alignment invariant of every dense
//! table, historical ones included, and reading the written values back with validation.

mod common;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use common::{
    add, collect, create_delta, delta_store, frame, index_schema, keys, open_delta, tick,
};
use futures::TryStreamExt;
use gamma_lake::arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, Int32Array, Int64Array, StringArray, UInt16Array,
    UInt32Array, UInt64Array,
};
use gamma_lake::arrow::compute::{cast, concat_batches, take};
use gamma_lake::arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema, SchemaRef};
use gamma_lake::arrow::record_batch::RecordBatch;
use gamma_lake::store::{CreateOptions, TableScan};
use gamma_lake::{
    AddOptions, Annotation, DeltaStore, Dnf, Error, Expression, Lake, LakeOptions, MemoryStore,
    OverlapMode, SignalType, StoredTable, TableStore,
};
use tempfile::TempDir;

type Key = (i64, String);

fn options() -> LakeOptions {
    LakeOptions {
        primary_sort_key: "ts".into(),
        validate: true,
        ..Default::default()
    }
}

fn create() -> (TempDir, Lake) {
    let dir = TempDir::new().unwrap();
    let lake = create_delta(dir.path().to_str().unwrap(), index_schema(), options()).unwrap();
    (dir, lake)
}

/// `feature` per key, read with validation.
fn read(lake: &Lake, feature: &str) -> BTreeMap<Key, Option<f64>> {
    values(
        &collect(lake.read(&[feature], &[], &Dnf::default()).unwrap()),
        feature,
    )
}

fn values(batch: &RecordBatch, feature: &str) -> BTreeMap<Key, Option<f64>> {
    let values = batch
        .column_by_name(feature)
        .unwrap()
        .as_primitive::<Float64Type>();
    key_list(batch).into_iter().zip(values).collect()
}

fn expect(
    rows: &[(i64, &str)],
    value: impl Fn(i64, &str) -> Option<f64>,
) -> BTreeMap<Key, Option<f64>> {
    rows.iter()
        .map(|(t, s)| ((*t, s.to_string()), value(*t, s)))
        .collect()
}

fn delta(dir: &Path) -> DeltaStore {
    delta_store(dir.to_str().unwrap()).unwrap()
}

fn scan(dir: &Path, name: &str, columns: &[&str]) -> RecordBatch {
    scan_store(&delta(dir), name, columns)
}

fn scan_store(store: &dyn TableStore, name: &str, columns: &[&str]) -> RecordBatch {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let table = store.open(name).await.unwrap().unwrap();
        let columns = columns
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        let scan = table
            .scan(Some(&columns), &Dnf::default(), 65536)
            .await
            .unwrap();
        concat_batches(
            &scan.schema,
            &scan.stream.try_collect::<Vec<_>>().await.unwrap(),
        )
        .unwrap()
    })
}

fn strings(batch: &RecordBatch, name: &str) -> Vec<Option<String>> {
    let column = cast(batch.column_by_name(name).unwrap(), &DataType::Utf8).unwrap();
    column
        .as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_string))
        .collect()
}

fn key_list(batch: &RecordBatch) -> Vec<Key> {
    let ts = cast(batch.column_by_name("ts").unwrap(), &DataType::Int64).unwrap();
    let ts = ts.as_primitive::<Int64Type>();
    let sym = strings(batch, "sym");
    (0..batch.num_rows())
        .map(|i| (ts.value(i), sym[i].clone().unwrap()))
        .collect()
}

fn check_invariant(dir: &Path) {
    check_store_invariant(&delta(dir))
}

/// Every dense table holds exactly the index keys up to its coverage, each once.
fn check_store_invariant(store: &dyn TableStore) {
    let fm = scan_store(store, "feature_metadata", &["table_addr", "signal_type"]);
    let joined: BTreeSet<String> = strings(&fm, "table_addr")
        .into_iter()
        .zip(strings(&fm, "signal_type"))
        .filter(|(_, t)| !matches!(t.as_deref(), Some("feature" | "target")))
        .map(|(a, _)| a.unwrap())
        .collect();
    let index = key_list(&scan_store(store, "index", &["ts", "sym"]));
    let unique: BTreeSet<Key> = index.iter().cloned().collect();
    assert_eq!(unique.len(), index.len(), "duplicate index keys");
    let tm = scan_store(
        store,
        "table_metadata",
        &["table_addr", "last_updated", "update_timestamp"],
    );
    let addrs = strings(&tm, "table_addr");
    let last = tm
        .column_by_name("last_updated")
        .unwrap()
        .as_primitive::<Int64Type>()
        .clone();
    let stamps = cast(
        tm.column_by_name("update_timestamp").unwrap(),
        &DataType::Int64,
    )
    .unwrap();
    let stamps = stamps.as_primitive::<Int64Type>();
    let mut coverage: BTreeMap<String, (i64, i64)> = BTreeMap::new();
    for (i, addr) in addrs.iter().enumerate() {
        let entry = coverage
            .entry(addr.clone().unwrap())
            .or_insert((i64::MIN, 0));
        if stamps.value(i) >= entry.0 {
            *entry = (stamps.value(i), last.value(i));
        }
    }
    assert!(!coverage.is_empty());
    for (addr, (_, l)) in coverage {
        if joined.contains(&addr) {
            continue;
        }
        let mut table = key_list(&scan_store(store, &addr, &["ts", "sym"]));
        table.sort();
        let expected: Vec<Key> = unique.iter().filter(|k| k.0 <= l).cloned().collect();
        assert_eq!(table, expected, "table {addr} with coverage {l}");
    }
}

fn feature_metadata(lake: &Lake) -> Vec<(String, i64, String, String, String)> {
    let batch = collect(lake.feature_metadata(&Dnf::default()).unwrap());
    let names = strings(&batch, "feature_name");
    let addrs = strings(&batch, "table_addr");
    let types = strings(&batch, "signal_type");
    let params = strings(&batch, "feature_params");
    let versions = batch
        .column_by_name("version")
        .unwrap()
        .as_primitive::<Int64Type>();
    let mut rows: Vec<_> = (0..batch.num_rows())
        .map(|i| {
            (
                names[i].clone().unwrap(),
                versions.value(i),
                addrs[i].clone().unwrap(),
                types[i].clone().unwrap(),
                params[i].clone().unwrap(),
            )
        })
        .collect();
    rows.sort();
    rows
}

fn commits(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .map(|p| {
            if p.is_dir() {
                commits(&p)
            } else {
                usize::from(p.extension().is_some_and(|e| e == "json"))
            }
        })
        .sum()
}

#[test]
fn memory_limit_reads_batches_that_fit_its_share() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(0..50_000, &["A", "B"]), &[("f1", 1.0), ("f2", 2.0)]),
        OverlapMode::Copy,
    );
    // A batch of the default size takes several times a scan's share of the limit.
    let bounded = open_delta(
        dir.path().to_str().unwrap(),
        LakeOptions {
            memory_limit: Some(256 << 10),
            ..options()
        },
    )
    .unwrap();
    assert_eq!(read(&bounded, "f2"), read(&lake, "f2"));
}

#[test]
fn create_is_idempotent_and_matches_the_python_schemas() {
    let (dir, _) = create();
    let again = create_delta(dir.path().to_str().unwrap(), index_schema(), options()).unwrap();
    assert_eq!(again.sort_keys(), ["ts", "sym"]);
    let names = |batch: &RecordBatch| {
        let schema = batch.schema();
        schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
    };
    let tm = collect(again.table_metadata(&Dnf::default()).unwrap());
    assert_eq!(
        names(&tm),
        ["table_addr", "last_updated", "update_timestamp"]
    );
    assert_eq!(tm.schema().field(1).data_type(), &DataType::Int64);
    let fm = collect(again.feature_metadata(&Dnf::default()).unwrap());
    assert_eq!(fm.num_rows(), 0);
    assert_eq!(
        names(&fm),
        [
            "feature_name",
            "version",
            "table_addr",
            "owner",
            "signal_type",
            "feature_params"
        ]
    );
    let err = create_delta(
        dir.path().join("other").to_str().unwrap(),
        index_schema(),
        LakeOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
}

#[test]
fn new_table_then_append_beyond_coverage() {
    let (dir, lake) = create();
    let first = keys(1..=3, &["a", "b"]);
    add(&lake, frame(&first, &[("f", 1.0)]), OverlapMode::Copy);
    let second = keys(4..=5, &["a", "b"]);
    add(&lake, frame(&second, &[("f", 1.0)]), OverlapMode::Copy);
    check_invariant(dir.path());
    let all = keys(1..=5, &["a", "b"]);
    assert_eq!(read(&lake, "f"), expect(&all, |t, _| Some(t as f64)));
    let fm = feature_metadata(&lake);
    assert_eq!(fm.len(), 1);
    assert_eq!(fm[0].1, 0);
    assert_eq!(fm[0].3, "feature");
    assert_eq!(fm[0].4, "{}");
}

#[test]
fn all_new_keys_pad_the_missing_tail() {
    // Point72/gamma-lake#45: f covers 1..3 while the index reaches 5; appending only new keys
    // at 6 must also add 4 and 5 to f.
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=3, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    add(
        &lake,
        frame(&keys(1..=5, &["a"]), &[("g", 2.0)]),
        OverlapMode::Copy,
    );
    add(
        &lake,
        frame(&keys(6..=6, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    check_invariant(dir.path());
    let all = keys(1..=6, &["a"]);
    assert_eq!(
        read(&lake, "f"),
        expect(&all, |t, _| (t <= 3 || t == 6).then_some(t as f64))
    );
    assert_eq!(
        read(&lake, "g"),
        expect(&all, |t, _| (t <= 5).then_some(2.0 * t as f64))
    );
}

#[test]
fn all_new_keys_inside_coverage_are_appended_and_padded() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=3, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    add(
        &lake,
        frame(&keys(1..=3, &["a"]), &[("g", 2.0)]),
        OverlapMode::Copy,
    );
    // New keys inside the coverage of f, the last at its end (pk == L).
    add(
        &lake,
        frame(&[(2, "b"), (3, "b")], &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    check_invariant(dir.path());
    let all = [(1, "a"), (2, "a"), (2, "b"), (3, "a"), (3, "b")];
    assert_eq!(read(&lake, "f"), expect(&all, |t, _| Some(t as f64)));
    assert_eq!(
        read(&lake, "g"),
        expect(&all, |t, s| (s == "a").then_some(2.0 * t as f64))
    );
    assert!(feature_metadata(&lake).iter().all(|r| r.1 == 0));
}

#[test]
fn overlap_copy_keeps_unwritten_keys_and_the_old_tail() {
    // Point72/gamma-lake#46: copy mode must keep keys of the window missing from the input
    // and the rows after the window.
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=6, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    add(
        &lake,
        frame(&keys(1..=6, &["a", "b"]), &[("g", 2.0)]),
        OverlapMode::Copy,
    );
    let before = feature_metadata(&lake);
    add(
        &lake,
        frame(&keys(2..=3, &["a"]), &[("f", 10.0)]),
        OverlapMode::Copy,
    );
    check_invariant(dir.path());
    let all = keys(1..=6, &["a", "b"]);
    assert_eq!(
        read(&lake, "f"),
        expect(&all, |t, s| match (t, s) {
            (2..=3, "a") => Some(10.0 * t as f64),
            (2..=3, _) => None,
            (_, "a") => Some(t as f64),
            _ => None,
        })
    );
    let after = feature_metadata(&lake);
    let f: Vec<_> = after.iter().filter(|r| r.0 == "f").collect();
    assert_eq!(f.len(), 2);
    assert_eq!(f[1].1, 1);
    assert_ne!(f[1].2, f[0].2, "copy writes a new table");
    assert_eq!(after.len(), before.len() + 1);
    // The coverage of the copy includes the old tail, so a later append adds no duplicates.
    add(
        &lake,
        frame(&keys(7..=7, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    check_invariant(dir.path());
}

#[test]
fn overlap_copy_with_new_keys_before_coverage() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=4, &["a"]), &[("f", 1.0), ("h", 3.0)]),
        OverlapMode::Copy,
    );
    // Overlaps at 2 and adds the new key (3, "b") before f's coverage.
    add(
        &lake,
        frame(&[(2, "a"), (3, "b")], &[("f", 10.0)]),
        OverlapMode::Copy,
    );
    check_invariant(dir.path());
    let all = [(1, "a"), (2, "a"), (3, "a"), (3, "b"), (4, "a")];
    assert_eq!(
        read(&lake, "f"),
        expect(&all, |t, s| match (t, s) {
            (2, _) | (3, "b") => Some(10.0 * t as f64),
            (3, _) => None,
            _ => Some(t as f64),
        })
    );
    assert_eq!(
        read(&lake, "h"),
        expect(&all, |t, s| (s == "a").then_some(3.0 * t as f64))
    );
}

#[test]
fn overlap_merge_updates_only_the_input_features() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=4, &["a", "b"]), &[("f", 1.0), ("h", 3.0)]),
        OverlapMode::Copy,
    );
    let addr = feature_metadata(&lake)[0].2.clone();
    add(
        &lake,
        frame(&[(2, "a"), (3, "a"), (5, "a")], &[("f", 10.0)]),
        OverlapMode::Merge,
    );
    check_invariant(dir.path());
    let mut all = keys(1..=4, &["a", "b"]);
    all.push((5, "a"));
    assert_eq!(
        read(&lake, "f"),
        // The input replaces f over its whole primary key range [2, 5].
        expect(&all, |t, s| match (t, s) {
            (2 | 3 | 5, "a") => Some(10.0 * t as f64),
            (2..=4, _) => None,
            _ => Some(t as f64),
        })
    );
    assert_eq!(
        read(&lake, "h"),
        expect(&all, |t, _| (t <= 4).then_some(3.0 * t as f64))
    );
    let fm = feature_metadata(&lake);
    let f: Vec<_> = fm.iter().filter(|r| r.0 == "f").collect();
    assert_eq!((f[1].1, &f[1].2), (1, &addr), "merge keeps the table");
}

#[test]
fn padding_reaches_every_covering_table_once() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=2, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    add(
        &lake,
        frame(&keys(3..=4, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    add(
        &lake,
        frame(&keys(1..=2, &["a"]), &[("g", 2.0)]),
        OverlapMode::Copy,
    );
    // f has two table metadata rows; new keys at 1 and 3 pad f once and g only at 1.
    lake.add_index_rows(vec![frame(&[(1, "b"), (3, "b")], &[("ignored", 1.0)])])
        .unwrap();
    check_invariant(dir.path());
    let all = [(1, "a"), (1, "b"), (2, "a"), (3, "a"), (3, "b"), (4, "a")];
    assert_eq!(
        read(&lake, "f"),
        expect(&all, |t, s| (s == "a").then_some(t as f64))
    );
    assert_eq!(
        read(&lake, "g"),
        expect(&all, |t, s| (s == "a" && t <= 2).then_some(2.0 * t as f64))
    );
    assert_eq!(feature_metadata(&lake).len(), 2);
}

#[test]
fn input_is_cleaned_and_validated() {
    let (dir, lake) = create();
    let schema = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("ts", DataType::Int32, true),
        Field::new("f", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("a"),
                None,
                Some("a"),
            ])),
            Arc::new(Int32Array::from(vec![Some(2), Some(2), Some(3), Some(1)])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        ],
    )
    .unwrap();
    lake.add(vec![batch], &AddOptions::default()).unwrap();
    check_invariant(dir.path());
    assert_eq!(
        read(&lake, "f"),
        expect(&[(1, "a"), (2, "a")], |t, _| Some(if t == 1 {
            4.0
        } else {
            1.0
        }))
    );

    let err = lake.add(vec![], &AddOptions::default()).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    let no_ts = frame(&[(1, "a")], &[("f", 1.0)]).project(&[1, 2]).unwrap();
    let err = lake.add(vec![no_ts], &AddOptions::default()).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    let floats = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Float64, true),
            Field::new("sym", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Float64Array::from(vec![1.5])),
            Arc::new(StringArray::from(vec!["a"])),
        ],
    )
    .unwrap();
    let err = lake.add_index_rows(vec![floats]).unwrap_err();
    assert!(matches!(err, Error::UnsupportedInput(_)), "{err}");
    let err = lake
        .add(
            vec![frame(&[(1, "a")], &[("f", 1.0)])],
            &AddOptions {
                feature_params: Some("[1]".into()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
}

#[test]
fn unsupported_input_writes_nothing() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=2, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    let before = commits(dir.path());
    // Unsigned columns are stored as the signed type of the same width, so u64::MAX does not fit.
    let row = keys(3..=3, &["a"]);
    for batch in [
        with_feature(&row, "u", Arc::new(UInt64Array::from(vec![u64::MAX]))),
        // A string written into f's float column.
        with_feature(&row, "f", Arc::new(StringArray::from(vec!["x"]))),
    ] {
        let err = lake.add(vec![batch], &AddOptions::default()).unwrap_err();
        assert!(matches!(err, Error::UnsupportedInput(_)), "{err}");
    }
    assert_eq!(commits(dir.path()), before);

    let batch = with_feature(&row, "u", Arc::new(UInt16Array::from(vec![255])));
    lake.add(vec![batch], &AddOptions::default()).unwrap();
    let u = collect(lake.read(&["u"], &[], &Dnf::default()).unwrap());
    assert_eq!(
        u.schema().field_with_name("u").unwrap().data_type(),
        &DataType::Int16
    );
}

#[test]
fn new_index_values_can_be_disallowed() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=3, &["a", "b"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    let before = commits(dir.path());
    let options = AddOptions {
        allow_new_index_values: false,
        ..Default::default()
    };
    let err = error(lake.add(
        vec![frame(&[(2, "a"), (2, "c"), (4, "a")], &[("g", 1.0)])],
        &options,
    ));
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    assert!(err.to_string().contains("2 input keys"), "{err}");
    assert_eq!(commits(dir.path()), before);

    lake.add(vec![frame(&[(2, "a"), (3, "b")], &[("g", 10.0)])], &options)
        .unwrap();
    check_invariant(dir.path());
    assert_eq!(
        read(&lake, "g"),
        expect(&keys(1..=3, &["a", "b"]), |t, s| match (t, s) {
            (2, "a") | (3, "b") => Some(10.0 * t as f64),
            _ => None,
        })
    );
}

#[test]
fn targets_and_metadata_options_are_recorded() {
    let (dir, lake) = create();
    let options = AddOptions {
        owner: "me".into(),
        signal_type: SignalType::Target,
        overlap_mode: OverlapMode::Copy,
        feature_params: Some("{\"w\": 1}".into()),
        ..Default::default()
    };
    lake.add(vec![frame(&keys(1..=2, &["a"]), &[("y", 1.0)])], &options)
        .unwrap();
    lake.add(
        vec![frame(&keys(1..=1, &["a"]), &[("y", 5.0)])],
        &AddOptions {
            signal_type: SignalType::Target,
            ..Default::default()
        },
    )
    .unwrap();
    check_invariant(dir.path());
    let fm = feature_metadata(&lake);
    assert_eq!(fm[0].3, "target");
    assert_eq!(fm[0].4, "{\"w\": 1}");
    assert_eq!(fm[1].1, 1);
    assert_eq!(fm[1].4, "{\"w\": 1}", "a new version keeps the parameters");
    let batch = collect(lake.read(&[], &["y"], &Dnf::default()).unwrap());
    assert_eq!(batch.num_rows(), 2);
    let owners = strings(
        &collect(lake.feature_metadata(&Dnf::default()).unwrap()),
        "owner",
    );
    assert!(owners.contains(&Some("me".into())));
    assert!(owners.contains(&Some("missing_owner".into())));
}

#[cfg(unix)]
#[test]
fn a_failed_commit_is_rolled_back() {
    use std::os::unix::fs::PermissionsExt;
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=2, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    let log = dir.path().join("index").join("_delta_log");
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o555)).unwrap();
    let result = lake.add(
        vec![frame(&keys(3..=3, &["a"]), &[("f", 1.0)])],
        &AddOptions::default(),
    );
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o755)).unwrap();
    match result.unwrap_err() {
        Error::Write {
            phase, committed, ..
        } => assert_eq!(
            (phase.as_str(), committed.len()),
            ("writing table index (rolled back)", 0)
        ),
        other => panic!("expected a write error, got {other}"),
    }
    check_invariant(dir.path());
    add(
        &lake,
        frame(&keys(3..=3, &["a"]), &[("f", 2.0)]),
        OverlapMode::Copy,
    );
    let f: Vec<_> = read(&lake, "f").into_values().collect();
    assert_eq!(f, [Some(1.0), Some(2.0), Some(6.0)]);
}

/// Writes covering every kind of commit, then every feature read back.
fn scenario(lake: &Lake) -> (RecordBatch, Vec<(String, i64, String)>) {
    use OverlapMode::{Copy, Merge};
    add(
        lake,
        frame(&keys(1..=3, &["a", "b"]), &[("f", 1.0), ("g", 2.0)]),
        Copy,
    );
    add(lake, frame(&keys(1..=5, &["a"]), &[("h", 3.0)]), Copy);
    add(
        lake,
        frame(&keys(4..=6, &["a", "b"]), &[("f", 4.0), ("g", 5.0)]),
        Copy,
    );
    add(lake, frame(&keys(2..=3, &["a"]), &[("f", 6.0)]), Merge);
    add(lake, frame(&keys(2..=3, &["b", "c"]), &[("h", 7.0)]), Copy);
    lake.add_index_rows(vec![frame(&keys(7..=7, &["a"]), &[])])
        .unwrap();
    add(lake, frame(&keys(3..=4, &["d"]), &[("g", 8.0)]), Merge);
    let batch = collect(lake.read(&["f", "g", "h"], &[], &Dnf::default()).unwrap());
    let metadata = feature_metadata(lake)
        .into_iter()
        .map(|(name, version, _, signal_type, _)| (name, version, signal_type))
        .collect();
    (batch, metadata)
}

#[test]
fn memory_and_delta_stores_hold_the_same_lake() {
    let stores = lakes();
    assert!(stores[1].1.location().starts_with("memory://"));
    let held: Vec<_> = stores
        .iter()
        .map(|(_, lake, store)| {
            let held = scenario(lake);
            check_store_invariant(store.as_ref());
            held
        })
        .collect();
    assert_eq!(held[0], held[1]);
    assert!(held[0].0.num_rows() > 10);
    let joined: Vec<_> = lakes()
        .iter()
        .map(|(_, lake, store)| {
            joined_scenario(lake);
            check_store_invariant(store.as_ref());
            collect(lake.read(&["f", "x", "s"], &[], &Dnf::default()).unwrap())
        })
        .collect();
    assert_eq!(joined[0], joined[1]);
}

/// A [`MemoryStore`] whose scans return rows in reverse order.
#[derive(Debug)]
struct Reversed(MemoryStore);

struct ReversedTable(Box<dyn StoredTable>);

#[async_trait]
impl TableStore for Reversed {
    fn location(&self) -> &str {
        self.0.location()
    }

    async fn open(&self, name: &str) -> Result<Option<Box<dyn StoredTable>>, Error> {
        Ok(self
            .0
            .open(name)
            .await?
            .map(|t| Box::new(ReversedTable(t)) as Box<dyn StoredTable>))
    }

    async fn create(
        &self,
        name: &str,
        batch: RecordBatch,
        options: &CreateOptions,
    ) -> Result<(), Error> {
        self.0.create(name, batch, options).await
    }
}

#[async_trait]
impl StoredTable for ReversedTable {
    fn schema(&self) -> SchemaRef {
        self.0.schema()
    }

    async fn scan(
        &self,
        columns: Option<&[String]>,
        dnf: &Dnf,
        batch_size: usize,
    ) -> Result<TableScan, Error> {
        let scan = self.0.scan(columns, dnf, batch_size).await?;
        let schema = scan.schema;
        let batch = concat_batches(&schema, &scan.stream.try_collect::<Vec<_>>().await?)?;
        let reversed = UInt32Array::from_iter_values((0..batch.num_rows() as u32).rev());
        let columns = batch
            .columns()
            .iter()
            .map(|c| take(c, &reversed, None))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        Ok(TableScan {
            schema,
            stream: Box::pin(futures::stream::iter([Ok(batch)])),
        })
    }

    async fn append(self: Box<Self>, batch: RecordBatch) -> Result<(), Error> {
        self.0.append(batch).await
    }

    async fn append_merging(self: Box<Self>, batch: RecordBatch) -> Result<(), Error> {
        self.0.append_merging(batch).await
    }

    async fn merge(
        self: Box<Self>,
        source: RecordBatch,
        keys: &[String],
        update: &[String],
    ) -> Result<(), Error> {
        self.0.merge(source, keys, update).await
    }

    async fn set_column_metadata(
        self: Box<Self>,
        column: &str,
        metadata: HashMap<String, String>,
    ) -> Result<Box<dyn StoredTable>, Error> {
        Ok(Box::new(ReversedTable(
            self.0.set_column_metadata(column, metadata).await?,
        )))
    }

    async fn restore(self: Box<Self>, at: i64) -> Result<bool, Error> {
        self.0.restore(at).await
    }
}

#[test]
fn reads_do_not_depend_on_the_order_scans_return_rows_in() {
    let memory = MemoryStore::new();
    let ordered =
        Lake::create_with_store(Arc::new(memory.clone()), index_schema(), options()).unwrap();
    let (batch, _) = scenario(&ordered);
    assert_ne!(
        scan_store(&Reversed(memory.clone()), "index", &["ts", "sym"]),
        scan_store(&memory, "index", &["ts", "sym"]),
    );
    let reversed = Lake::open_with_store(Arc::new(Reversed(memory)), options()).unwrap();
    let read = |lake: &Lake| collect(lake.read(&["f", "g", "h"], &[], &Dnf::default()).unwrap());
    assert_eq!(read(&reversed), batch);
    assert!(batch.num_rows() > 10);
}

fn add_joined(lake: &Lake, batch: RecordBatch, signal_type: SignalType, params: Option<&str>) {
    let options = AddOptions {
        signal_type,
        feature_params: params.map(str::to_string),
        ..Default::default()
    };
    lake.add(vec![batch], &options).unwrap();
}

fn error<T>(result: Result<T, Error>) -> Error {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

fn key(t: i64, s: &str) -> Key {
    (t, s.to_string())
}

/// Index keys after [`joined_scenario`].
fn joined_keys() -> Vec<(i64, &'static str)> {
    let mut keys = keys(1..=4, &["a", "b"]);
    keys.extend([(5, "b"), (6, "a"), (7, "a"), (7, "b")]);
    keys
}

/// Dense `f`, backward as-of `x` written three ways, and sparse `s`.
fn joined_scenario(lake: &Lake) {
    let backward = Some(r#"{"strategy": "backward"}"#);
    add(
        lake,
        frame(&keys(1..=4, &["a", "b"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    // A new as-of table is padded with the missing index keys up to its last key.
    add_joined(
        lake,
        frame(&[(2, "a"), (4, "a")], &[("x", 10.0)]),
        SignalType::AsOf,
        backward,
    );
    // Keys new to the index are appended alone.
    add_joined(
        lake,
        frame(&[(6, "a")], &[("x", 10.0)]),
        SignalType::AsOf,
        backward,
    );
    lake.add_index_rows(vec![frame(&keys(7..=7, &["a", "b"]), &[])])
        .unwrap();
    // An overwrite keeps the rows before and after it, without padding.
    add_joined(
        lake,
        frame(&[(3, "a")], &[("x", 100.0)]),
        SignalType::AsOf,
        backward,
    );
    // A new sparse table holds only the input rows.
    add_joined(
        lake,
        frame(&[(2, "b"), (5, "b")], &[("s", 1.0)]),
        SignalType::Sparse,
        None,
    );
}

#[test]
fn as_of_and_sparse_features_are_written_and_joined_to_the_index() {
    let (dir, lake) = create();
    joined_scenario(&lake);
    check_invariant(dir.path());
    let all = joined_keys();
    assert_eq!(
        read(&lake, "f"),
        expect(&all, |t, _| (t <= 4).then_some(t as f64))
    );
    let x = |t: i64, s: &str| match (t, s) {
        (2, "a") => Some(20.0),
        (3, "a") => Some(300.0),
        (4 | 5, "a") => Some(40.0),
        (6.., "a") => Some(60.0),
        _ => None,
    };
    assert_eq!(read(&lake, "x"), expect(&all, x));
    let s = |t: i64, s: &str| (s == "b" && (t == 2 || t == 5)).then_some(t as f64);
    assert_eq!(read(&lake, "s"), expect(&all, s));

    let fm = feature_metadata(&lake);
    let types: Vec<(&str, i64, &str, &str)> = fm
        .iter()
        .map(|(n, v, _, t, p)| (n.as_str(), *v, t.as_str(), p.as_str()))
        .collect();
    assert_eq!(
        types,
        [
            ("f", 0, "feature", "{}"),
            ("s", 0, "sparse_feature", "{}"),
            ("x", 0, "as_of_feature", r#"{"strategy": "backward"}"#),
            ("x", 1, "as_of_feature", r#"{"strategy": "backward"}"#),
        ]
    );
    // The overwrite kept the padded rows before it and the rows after it.
    let x_table = key_list(&scan(dir.path(), &fm[3].2, &["ts", "sym"]));
    let x_table: BTreeSet<Key> = x_table.into_iter().collect();
    assert!(x_table.contains(&key(6, "a")) && x_table.contains(&key(1, "b")));
    assert!(!x_table.contains(&key(3, "b")) && !x_table.contains(&key(7, "a")));
    assert_eq!(
        key_list(&scan(dir.path(), &fm[1].2, &["ts", "sym"])),
        [key(2, "b"), key(5, "b")]
    );

    // A filtered read matches the filtered full read: x at 7 still comes from 6.
    let dnf = Dnf(vec![vec![gamma_lake::Term {
        column: "ts".into(),
        op: gamma_lake::Op::Eq,
        values: Arc::new(Int64Array::from(vec![7])),
    }]]);
    let batch = collect(lake.read(&["x", "s"], &[], &dnf).unwrap());
    assert_eq!(
        values(&batch, "x"),
        expect(&[(7, "a"), (7, "b")], |_, s| (s == "a").then_some(60.0))
    );
}

#[test]
fn reads_pin_versions_and_selected_rows() {
    let (_dir, lake) = create();
    joined_scenario(&lake);
    let all = joined_keys();
    let batch = collect(
        lake.read_versions(&[("x", 0), ("f", 0)], &Dnf::default())
            .unwrap(),
    );
    assert_eq!(
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["x", "f", "ts", "sym"]
    );
    // Version 0 still holds the padding row at (3, "a").
    let x0 = |t: i64, s: &str| match (t, s) {
        (2, "a") => Some(20.0),
        (4 | 5, "a") => Some(40.0),
        (6.., "a") => Some(60.0),
        _ => None,
    };
    assert_eq!(values(&batch, "x"), expect(&all, x0));
    assert_eq!(
        lake.read_versions_schema(&[("x", 1)])
            .unwrap()
            .field(0)
            .name(),
        "x"
    );
    let err = error(lake.read_versions(&[("x", 2)], &Dnf::default()));
    assert!(
        matches!(err, Error::MissingFeatures(ref m) if m == &["x version 2"]),
        "{err}"
    );
    let err = error(lake.read_versions(&[("x", 0), ("x", 1)], &Dnf::default()));
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");

    // Rows from the metadata read as stored, or with the parameters they are given.
    let fm = feature_metadata(&lake);
    let x1 = |signal_type: &str, params: &str| gamma_lake::SelectedFeature {
        name: "x".into(),
        table_addr: fm[3].2.clone(),
        signal_type: signal_type.into(),
        feature_params: Some(params.into()),
    };
    let stored = x1("as_of_feature", r#"{"strategy": "backward"}"#);
    let batch = collect(lake.read_selected(&[stored], &Dnf::default()).unwrap());
    assert_eq!(values(&batch, "x"), read(&lake, "x"));
    let forward = x1("as_of_feature", r#"{"strategy": "forward"}"#);
    let batch = collect(lake.read_selected(&[forward], &Dnf::default()).unwrap());
    let x = values(&batch, "x");
    assert_eq!(x[&key(7, "a")], None, "nothing follows 7");
    assert_eq!(x[&key(4, "a")], Some(40.0));
    assert_eq!(
        x[&key(1, "a")],
        None,
        "the padded row at 1 is matched exactly"
    );
    let err = error(lake.read_selected(&[x1("runtime_computed_feature", "{}")], &Dnf::default()));
    assert!(matches!(err, Error::Unsupported { .. }), "{err}");
    let err = error(lake.read_selected(
        &[x1("as_of_feature", r#"{"strategy": "sideways"}"#)],
        &Dnf::default(),
    ));
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
}

#[test]
fn an_empty_selection_reads_the_index() {
    let (_dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=3, &["a", "b"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    let batch = collect(lake.read_selected(&[], &Dnf::default()).unwrap());
    let schema = batch.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["ts", "sym"]);
    let mut got = key_list(&batch);
    got.sort();
    let want: Vec<Key> = keys(1..=3, &["a", "b"])
        .into_iter()
        .map(|(t, s)| (t, s.to_string()))
        .collect();
    assert_eq!(got, want);
}

fn note(
    name: &str,
    addr: Option<&str>,
    comment: Option<&str>,
    tags: Option<&[&str]>,
) -> Annotation {
    Annotation {
        feature_name: name.into(),
        table_addr: addr.map(str::to_string),
        comment: comment.map(str::to_string),
        tags: tags.map(|t| t.iter().map(|s| s.to_string()).collect()),
    }
}

/// `(comment, tags)` per row of an annotations or describe batch.
fn notes(batch: &RecordBatch) -> Vec<(Option<String>, Vec<String>)> {
    let comments = strings(batch, "comment");
    let tags = batch.column_by_name("tags").unwrap().as_list::<i32>();
    (0..batch.num_rows())
        .map(|i| {
            let row = tags.value(i);
            let row = row.as_string::<i32>();
            let row = row.iter().map(|v| v.unwrap().to_string()).collect();
            (comments[i].clone(), row)
        })
        .collect()
}

fn described(lake: &Lake, names: &[&str]) -> Vec<(Option<String>, Vec<String>)> {
    notes(&lake.describe(names).unwrap())
}

fn n(comment: Option<&str>, tags: &[&str]) -> (Option<String>, Vec<String>) {
    (
        comment.map(str::to_string),
        tags.iter().map(|s| s.to_string()).collect(),
    )
}

fn addr_of(lake: &Lake, name: &str) -> String {
    feature_metadata(lake)
        .into_iter()
        .rfind(|r| r.0 == name)
        .unwrap()
        .2
}

#[test]
fn annotations_belong_to_feature_versions() {
    let (dir, lake) = create();
    let options = AddOptions {
        annotations: vec![note("f", None, Some("first f"), Some(&["raw"]))],
        ..Default::default()
    };
    lake.add(
        vec![frame(&keys(1..=3, &["a"]), &[("f", 1.0), ("g", 2.0)])],
        &options,
    )
    .unwrap();
    let first = addr_of(&lake, "f");
    lake.annotate(&[
        note("g", None, Some("g"), None),
        note("g", None, None, Some(&["x", "y"])),
        note("f", None, None, None),
    ])
    .unwrap();
    assert_eq!(
        described(&lake, &["g", "f"]),
        [n(Some("g"), &["x", "y"]), n(Some("first f"), &["raw"])]
    );
    // Merges and padding rewrite the table but keep its annotations.
    add(
        &lake,
        frame(&keys(2..=3, &["a"]), &[("g", 5.0)]),
        OverlapMode::Merge,
    );
    lake.add_index_rows(vec![frame(&keys(4..=4, &["a"]), &[])])
        .unwrap();
    add(
        &lake,
        frame(&keys(5..=5, &["a"]), &[("h", 1.0)]),
        OverlapMode::Copy,
    );
    assert_eq!(addr_of(&lake, "g"), first);
    assert_eq!(described(&lake, &["g"]), [n(Some("g"), &["x", "y"])]);
    // A new version of f starts without annotations; the old one keeps them.
    add(
        &lake,
        frame(&keys(1..=2, &["a"]), &[("f", 3.0)]),
        OverlapMode::Copy,
    );
    let second = addr_of(&lake, "f");
    assert_ne!(second, first);
    assert_eq!(described(&lake, &["f"]), [n(None, &[])]);
    lake.annotate(&[note("f", Some(&first), Some("old f"), None)])
        .unwrap();
    let pairs = [
        ("f", first.as_str()),
        ("f", second.as_str()),
        ("g", first.as_str()),
    ];
    assert_eq!(
        notes(&lake.annotations(&pairs).unwrap()),
        [
            n(Some("old f"), &["raw"]),
            n(None, &[]),
            n(Some("g"), &["x", "y"])
        ]
    );
    // An empty selection describes every feature.
    let all = lake.describe(&[]).unwrap();
    assert_eq!(strings(&all, "feature_name").len(), 3);
    check_invariant(dir.path());
    assert_eq!(
        read(&lake, "g"),
        expect(&keys(1..=5, &["a"]), |t, _| match t {
            1 => Some(2.0),
            2 | 3 => Some(t as f64 * 5.0),
            _ => None,
        })
    );
}

#[test]
fn bad_annotations_store_nothing() {
    let (dir, lake) = create();
    add(
        &lake,
        frame(&keys(1..=2, &["a"]), &[("f", 1.0)]),
        OverlapMode::Copy,
    );
    let before = commits(dir.path());
    let addr = addr_of(&lake, "f");
    let bad = [
        vec![
            note("f", None, Some("ok"), None),
            note("nope", None, Some("x"), None),
        ],
        vec![note("f", Some("not_a_table"), Some("x"), None)],
        vec![
            note("f", None, Some("ok"), None),
            note("ts", Some(&addr), Some("x"), None),
        ],
    ];
    for annotations in bad {
        assert!(lake.annotate(&annotations).is_err(), "{annotations:?}");
    }
    assert!(matches!(
        error(lake.describe(&["f", "nope"])),
        Error::MissingFeatures(m) if m == ["nope"]
    ));
    for annotation in [
        note("ts", None, Some("x"), None),
        note("g", None, Some("x"), None),
        note("f", Some(&addr), Some("x"), None),
    ] {
        let options = AddOptions {
            annotations: vec![annotation],
            ..Default::default()
        };
        let input = vec![frame(&keys(3..=3, &["a"]), &[("f", 1.0)])];
        assert!(matches!(
            error(lake.add(input, &options)),
            Error::InvalidArgument(_)
        ));
    }
    assert_eq!(commits(dir.path()), before);
    assert_eq!(described(&lake, &["f"]), [n(None, &[])]);
}

#[test]
fn restore_undoes_later_writes_and_leaves_untouched_tables() {
    for (_dir, lake, store) in lakes() {
        add(
            &lake,
            frame(&keys(1..=3, &["a"]), &[("f", 1.0)]),
            OverlapMode::Copy,
        );
        add(
            &lake,
            frame(&keys(1..=3, &["a"]), &[("g", 2.0)]),
            OverlapMode::Copy,
        );
        lake.annotate(&[note("f", None, Some("before"), None)])
            .unwrap();
        let cutoff = tick();
        let metadata = feature_metadata(&lake);
        // A bad merge into f's table, an annotation, a new index row and a new feature;
        // g's table gets no later commit.
        add(
            &lake,
            frame(&keys(2..=3, &["a"]), &[("f", 9.0)]),
            OverlapMode::Merge,
        );
        lake.annotate(&[note("f", None, Some("after"), None)])
            .unwrap();
        add(
            &lake,
            frame(&keys(4..=4, &["b"]), &[("h", 1.0)]),
            OverlapMode::Copy,
        );
        lake.restore(cutoff).unwrap();
        assert_eq!(feature_metadata(&lake), metadata);
        assert_eq!(
            read(&lake, "f"),
            expect(&keys(1..=3, &["a"]), |t, _| Some(t as f64))
        );
        assert_eq!(
            read(&lake, "g"),
            expect(&keys(1..=3, &["a"]), |t, _| Some(t as f64 * 2.0))
        );
        assert_eq!(described(&lake, &["f"]), [n(Some("before"), &[])]);
        assert!(matches!(
            error(lake.read(&["h"], &[], &Dnf::default())),
            Error::MissingFeatures(_)
        ));
        check_store_invariant(store.as_ref());
        // Writing continues from the restored lake.
        add(
            &lake,
            frame(&keys(4..=4, &["a"]), &[("f", 1.0)]),
            OverlapMode::Copy,
        );
        check_store_invariant(store.as_ref());
    }
}

/// A Delta lake and a memory lake, with their stores.
fn lakes() -> Vec<(Option<TempDir>, Lake, Arc<dyn TableStore>)> {
    let (dir, delta_lake) = create();
    let delta_store: Arc<dyn TableStore> = Arc::new(delta(dir.path()));
    let memory = MemoryStore::new();
    let memory_lake =
        Lake::create_with_store(Arc::new(memory.clone()), index_schema(), options()).unwrap();
    vec![
        (Some(dir), delta_lake, delta_store),
        (None, memory_lake, Arc::new(memory)),
    ]
}

#[test]
fn consolidation_moves_features_into_one_table_and_keeps_reads() {
    for (_dir, lake, store) in lakes() {
        let (before, _) = scenario(&lake);
        add_joined(
            &lake,
            frame(&keys(2..=2, &["a"]), &[("s", 1.0)]),
            SignalType::Sparse,
            None,
        );
        lake.annotate(&[
            note("f", None, Some("f"), Some(&["x"])),
            note("h", None, None, Some(&["y"])),
        ])
        .unwrap();
        let old = feature_metadata(&lake);
        assert_ne!(addr_of(&lake, "f"), addr_of(&lake, "h"));
        let addr = lake.consolidate(&["f", "g", "h", "f"]).unwrap().unwrap();
        let after = collect(lake.read(&["f", "g", "h"], &[], &Dnf::default()).unwrap());
        assert_eq!(after, before);
        let new = feature_metadata(&lake);
        for name in ["f", "g", "h"] {
            let was = old.iter().rfind(|r| r.0 == name).unwrap();
            let now = new.iter().rfind(|r| r.0 == name).unwrap();
            assert_eq!(
                (now.1, &now.2, &now.3, &now.4),
                (was.1 + 1, &addr, &was.3, &was.4)
            );
        }
        assert_eq!(
            described(&lake, &["f", "g", "h"]),
            [n(Some("f"), &["x"]), n(None, &[]), n(None, &["y"])]
        );
        check_store_invariant(store.as_ref());
        assert_eq!(lake.consolidate(&["f", "h"]).unwrap(), None);
        assert!(matches!(
            error(lake.consolidate(&["f", "nope"])),
            Error::MissingFeatures(m) if m == ["nope"]
        ));
        assert!(matches!(
            error(lake.consolidate(&["f", "s"])),
            Error::InvalidArgument(m) if m.contains("sparse_feature")
        ));
        // Later writes reach the consolidated table.
        lake.add_index_rows(vec![frame(&keys(8..=8, &["a"]), &[])])
            .unwrap();
        add(
            &lake,
            frame(&keys(2..=2, &["a"]), &[("g", 1.0)]),
            OverlapMode::Merge,
        );
        check_store_invariant(store.as_ref());
        assert_eq!(read(&lake, "g")[&key(2, "a")], Some(2.0));
        assert_eq!(read(&lake, "f")[&key(8, "a")], None);
    }
}

fn expression(name: &str, roots: &[&str]) -> Expression {
    Expression {
        name: name.into(),
        expression: format!("{name} payload"),
        roots: roots.iter().map(|r| r.to_string()).collect(),
    }
}

#[test]
fn runtime_computed_features_are_registered_and_not_read() {
    for (dir, lake, _) in lakes() {
        add(
            &lake,
            frame(&keys(0..3, &["a"]), &[("f", 1.0)]),
            OverlapMode::Copy,
        );
        let exprs = [expression("c", &["f", "d"]), expression("d", &["f"])];
        lake.add_expressions(&exprs, "me").unwrap();
        let rows = feature_metadata(&lake);
        let rc: Vec<_> = rows.iter().filter(|r| r.0 != "f").cloned().collect();
        let row = |n: &str| {
            let params = format!("{n} payload");
            (
                n.to_string(),
                0,
                "n/a".into(),
                "runtime_computed".into(),
                params,
            )
        };
        assert_eq!(rc, [row("c"), row("d")]);
        let fm = collect(lake.feature_metadata(&Dnf::default()).unwrap());
        assert_eq!(
            strings(&fm, "owner")
                .iter()
                .flatten()
                .filter(|o| *o == "me")
                .count(),
            2
        );

        let again = error(lake.add_expressions(&[expression("c", &["f"])], "me"));
        assert!(matches!(again, Error::InvalidArgument(_)), "{again}");
        let missing = error(lake.add_expressions(&[expression("e", &["zz"])], "me"));
        assert!(matches!(missing, Error::InvalidArgument(_)), "{missing}");
        assert_eq!(feature_metadata(&lake).len(), 3);

        let read = error(lake.read(&["c"], &[], &Dnf::default()));
        assert!(
            matches!(&read, Error::Unsupported { feature, signal_type }
                if feature == "c" && signal_type == "runtime_computed"),
            "{read}"
        );
        if let Some(dir) = dir {
            check_invariant(dir.path());
        }
    }
}

fn memory_lake(primary_sort_key: &str) -> Lake {
    let options = LakeOptions {
        primary_sort_key: primary_sort_key.into(),
        ..options()
    };
    Lake::create_with_store(Arc::new(MemoryStore::new()), index_schema(), options).unwrap()
}

/// A frame with the keys `rows` and one feature column.
fn with_feature(rows: &[(i64, &str)], name: &str, values: ArrayRef) -> RecordBatch {
    let keys = frame(rows, &[]);
    let mut fields = keys.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(name, values.data_type().clone(), true)));
    let mut columns = keys.columns().to_vec();
    columns.push(values);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn shown(batch: &RecordBatch, name: &str) -> Vec<String> {
    let column = batch.column_by_name(name).unwrap();
    (0..column.len())
        .map(|i| gamma_lake::arrow::util::display::array_value_to_string(column, i).unwrap())
        .collect()
}

#[test]
fn nested_features_are_stored_and_widened() {
    use gamma_lake::arrow::array::{ListArray, StructArray};
    use gamma_lake::arrow::datatypes::Int32Type;
    for (_dir, lake, _) in lakes() {
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            None,
        ]);
        add(
            &lake,
            with_feature(&keys(1..=2, &["a"]), "l", Arc::new(list)),
            OverlapMode::Copy,
        );
        let x: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let s = StructArray::from(vec![(Arc::new(Field::new("x", DataType::Int32, true)), x)]);
        add(
            &lake,
            with_feature(&keys(1..=2, &["a"]), "s", Arc::new(s)),
            OverlapMode::Copy,
        );
        // Copy mode stores l at the wider item type.
        let wider = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![Some(vec![Some(3)])]);
        add(
            &lake,
            with_feature(&keys(2..=2, &["a"]), "l", Arc::new(wider)),
            OverlapMode::Copy,
        );
        let batch = collect(lake.read(&["l", "s"], &[], &Dnf::default()).unwrap());
        let DataType::List(item) = batch
            .schema()
            .field_with_name("l")
            .unwrap()
            .data_type()
            .clone()
        else {
            panic!("l is not a list")
        };
        assert_eq!(item.data_type(), &DataType::Int64);
        assert_eq!(shown(&batch, "l"), ["[1, 2]", "[3]"]);
        assert_eq!(shown(&batch, "s"), ["{x: 1}", "{x: }"]);
    }
}

#[test]
fn copy_mode_promotes_numbers_as_polars_does() {
    use DataType::*;
    let cases = [
        (Int32, Int64, Some(Int64)),
        (Int64, Int8, Some(Int64)),
        (Float32, Int16, Some(Float32)),
        (Int16, Float32, Some(Float32)),
        (Int8, Int16, Some(Int16)),
        (Int32, Float32, Some(Float64)),
        (Float32, Int64, Some(Float64)),
        (Float64, Float32, Some(Float64)),
        (Int32, Utf8, None),
        (Utf8, Int64, None),
    ];
    for (stored, incoming, want) in cases {
        let lake = memory_lake("ts");
        // 2^24 + 1 is the smallest integer a Float32 cannot hold.
        let first_values: ArrayRef = Arc::new(Int64Array::from(vec![16_777_217, 2]));
        let first = cast(&first_values, &stored).unwrap();
        add(
            &lake,
            with_feature(&keys(1..=2, &["a"]), "f", first),
            OverlapMode::Copy,
        );
        let second: ArrayRef = Arc::new(Int64Array::from(vec![3]));
        let second = cast(&second, &incoming).unwrap();
        let result = lake.add(
            vec![with_feature(&keys(2..=2, &["a"]), "f", second)],
            &AddOptions::default(),
        );
        let Some(want) = want else {
            let err = error(result);
            assert!(
                matches!(err, Error::UnsupportedInput(_)),
                "{stored} {incoming}: {err}"
            );
            continue;
        };
        result.unwrap();
        let batch = collect(lake.read(&["f"], &[], &Dnf::default()).unwrap());
        let f = batch.column_by_name("f").unwrap();
        assert_eq!(f.data_type(), &want, "{stored} + {incoming}");
        let f = cast(f, &Float64).unwrap();
        let first = cast(&cast(&first_values, &stored).unwrap(), &Float64).unwrap();
        assert_eq!(
            f.as_primitive::<Float64Type>().iter().collect::<Vec<_>>(),
            [
                first.as_primitive::<Float64Type>().iter().next().unwrap(),
                Some(3.0)
            ],
            "{stored} + {incoming}"
        );
    }
}

#[test]
fn floating_point_sort_keys_are_rejected() {
    let ts = Field::new("ts", DataType::Int64, true);
    let x = Field::new("x", DataType::Float64, true);
    // A floating point primary sort key, then a floating point secondary one.
    for (fields, primary_sort_key) in [([ts.clone(), x.clone()], "x"), ([x, ts], "ts")] {
        let options = LakeOptions {
            primary_sort_key: primary_sort_key.into(),
            ..options()
        };
        let schema = Arc::new(Schema::new(fields.to_vec()));
        let err = error(Lake::create_with_store(
            Arc::new(MemoryStore::new()),
            schema,
            options,
        ));
        assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    }
}

#[test]
fn as_of_parameters_are_checked_before_writing() {
    let lake = memory_lake("ts");
    let batch = || frame(&keys(1..=2, &["a"]), &[("a", 1.0)]);
    for (params, invalid) in [
        (r#"{"tolerance": "1h"}"#, true),
        (r#"{"strategy": "sideways"}"#, true),
        (r#"{"by": "sym"}"#, false),
    ] {
        let options = AddOptions {
            signal_type: SignalType::AsOf,
            feature_params: Some(params.into()),
            ..Default::default()
        };
        let err = error(lake.add(vec![batch()], &options));
        assert_eq!(
            matches!(err, Error::InvalidArgument(_)),
            invalid,
            "{params}: {err}"
        );
        assert!(lake.read(&["a"], &[], &Dnf::default()).is_err());
    }
    add_joined(
        &lake,
        batch(),
        SignalType::AsOf,
        Some(r#"{"tolerance": 1}"#),
    );

    // A string primary key cannot be matched as of, whatever the parameters.
    let lake = memory_lake("sym");
    let err = error(lake.add(
        vec![batch()],
        &AddOptions {
            signal_type: SignalType::AsOf,
            ..Default::default()
        },
    ));
    assert!(matches!(err, Error::UnsupportedInput(_)), "{err}");
}
