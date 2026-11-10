//! The table stores behind a lake: addressing, snapshots and merges, on every store.

use std::collections::HashMap;
use std::sync::Arc;

use futures::TryStreamExt;
use gamma_lake::arrow::array::{Array, AsArray, Float64Array, Int64Array, RecordBatch};
use gamma_lake::arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
use gamma_lake::arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use gamma_lake::store::CreateOptions;
use gamma_lake::{DeltaOptions, DeltaStore, Dnf, MemoryStore, StoredTable, TableStore};
use tempfile::TempDir;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
        Field::new("g", DataType::Float64, true),
    ]))
}

fn batch(k: &[i64], f: &[f64], g: &[f64]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(k.to_vec())),
            Arc::new(Float64Array::from(f.to_vec())),
            Arc::new(Float64Array::from(g.to_vec())),
        ],
    )
    .unwrap()
}

/// Every store under test, each with its own empty location.
fn stores() -> Vec<(TempDir, Arc<dyn TableStore>)> {
    let local = TempDir::new().unwrap();
    let delta = delta_store(local.path().to_str().unwrap(), HashMap::new()).unwrap();
    vec![
        (local, Arc::new(delta)),
        (TempDir::new().unwrap(), Arc::new(MemoryStore::new())),
    ]
}

fn delta_store(
    base: &str,
    storage_options: HashMap<String, String>,
) -> gamma_lake::Result<DeltaStore> {
    DeltaStore::new(
        base,
        DeltaOptions {
            storage_options,
            ..Default::default()
        },
    )
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

/// All rows of `table`, sorted by `k`.
async fn rows(table: &dyn StoredTable) -> RecordBatch {
    let scan = table.scan(None, &Dnf::default(), 65536).await.unwrap();
    let batch = concat_batches(
        &scan.schema,
        &scan.stream.try_collect::<Vec<_>>().await.unwrap(),
    )
    .unwrap();
    let order = sort_to_indices(batch.column_by_name("k").unwrap(), None, None).unwrap();
    take_record_batch(&batch, &order).unwrap()
}

type Row = (i64, Option<f64>, Option<f64>);

fn values(batch: &RecordBatch) -> Vec<Row> {
    let k = batch.column(0).as_primitive::<Int64Type>();
    let f = batch.column(1).as_primitive::<Float64Type>();
    let g = batch.column(2).as_primitive::<Float64Type>();
    let value = |a: &gamma_lake::arrow::array::Float64Array, i| a.is_valid(i).then(|| a.value(i));
    (0..batch.num_rows())
        .map(|i| (k.value(i), value(f, i), value(g, i)))
        .collect()
}

#[test]
fn table_names_are_paths_with_any_characters() {
    let names = ["x:y", "a?b", "c#d", "sp ace", "nested/dir/t", "./dot", "é"];
    for (dir, store) in stores() {
        runtime().block_on(async {
            for name in names {
                assert!(store.open(name).await.unwrap().is_none(), "{name}");
                store
                    .create(name, batch(&[1], &[1.0], &[2.0]), &CreateOptions::default())
                    .await
                    .unwrap();
                let table = store.open(name).await.unwrap().unwrap();
                assert_eq!(table.schema().fields().len(), 3);
                assert_eq!(rows(table.as_ref()).await.num_rows(), 1, "{name}");
            }
            if store.location().starts_with('/') {
                for name in names {
                    assert!(dir.path().join(name).join("_delta_log").is_dir(), "{name}");
                }
            }
            for name in ["", "/abs", "../up", "a/../b", "50%", "back\\slash"] {
                assert!(store.open(name).await.is_err(), "{name}");
            }
            let err = store
                .create(
                    "x:y",
                    batch(&[1], &[1.0], &[2.0]),
                    &CreateOptions::default(),
                )
                .await;
            assert!(err.is_err(), "creating an existing table must fail");
        });
    }
}

#[test]
fn an_opened_table_keeps_its_version() {
    for (_dir, store) in stores() {
        runtime().block_on(async {
            let options = CreateOptions::default();
            store
                .create("t", batch(&[1, 2], &[1.0, 2.0], &[0.0, 0.0]), &options)
                .await
                .unwrap();
            let before = store.open("t").await.unwrap().unwrap();
            let writer = store.open("t").await.unwrap().unwrap();
            writer.append(batch(&[3], &[3.0], &[0.0])).await.unwrap();
            // The provider is requested only after the commit.
            assert_eq!(rows(before.as_ref()).await.num_rows(), 2);
            let after = store.open("t").await.unwrap().unwrap();
            assert_eq!(rows(after.as_ref()).await.num_rows(), 3);
            let source = RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int64, true),
                    Field::new("f", DataType::Float64, true),
                ])),
                vec![
                    Arc::new(Int64Array::from(vec![4])),
                    Arc::new(Float64Array::from(vec![4.0])),
                ],
            )
            .unwrap();
            store
                .open("t")
                .await
                .unwrap()
                .unwrap()
                .merge(source, &["k".into()], &["f".into()])
                .await
                .unwrap();
            assert_eq!(rows(after.as_ref()).await.num_rows(), 3);
            assert_eq!(rows(before.as_ref()).await.num_rows(), 2);
        });
    }
}

#[test]
fn merge_updates_matched_columns_and_inserts_the_rest() {
    let mut results = vec![];
    for (_dir, store) in stores() {
        let result = runtime().block_on(async {
            store
                .create(
                    "t",
                    batch(&[1, 2], &[1.0, 2.0], &[10.0, 20.0]),
                    &CreateOptions {
                        stats_columns: vec!["k".into()],
                    },
                )
                .await
                .unwrap();
            let source = RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int64, true),
                    Field::new("f", DataType::Float64, true),
                ])),
                vec![
                    Arc::new(Int64Array::from(vec![3, 2])),
                    Arc::new(Float64Array::from(vec![33.0, 99.0])),
                ],
            )
            .unwrap();
            let table = store.open("t").await.unwrap().unwrap();
            table
                .merge(source, &["k".into()], &["f".into()])
                .await
                .unwrap();
            let table = store.open("t").await.unwrap().unwrap();
            assert_eq!(table.schema().fields().len(), 3);
            values(&rows(table.as_ref()).await)
        });
        assert_eq!(
            result,
            vec![
                (1, Some(1.0), Some(10.0)),
                (2, Some(99.0), Some(20.0)),
                (3, Some(33.0), None),
            ],
            "{store:?}"
        );
        results.push(result);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn appends_must_match_the_table_schema() {
    for (_dir, store) in stores() {
        runtime().block_on(async {
            store
                .create("t", batch(&[1], &[1.0], &[1.0]), &CreateOptions::default())
                .await
                .unwrap();
            let narrow = batch(&[2], &[2.0], &[2.0]).project(&[0, 1]).unwrap();
            let table = store.open("t").await.unwrap().unwrap();
            assert!(table.append(narrow).await.is_err(), "{store:?}");
        });
    }
}

#[test]
fn merging_appends_add_nullable_columns() {
    for (_dir, store) in stores() {
        runtime().block_on(async {
            let narrow = |k: &[i64], f: &[f64]| batch(k, f, f).project(&[0, 1]).unwrap();
            store
                .create("t", narrow(&[1], &[1.0]), &CreateOptions::default())
                .await
                .unwrap();
            let table = store.open("t").await.unwrap().unwrap();
            let reordered = batch(&[2], &[2.0], &[2.0]).project(&[0, 2, 1]).unwrap();
            assert!(table.append_merging(reordered).await.is_err(), "{store:?}");
            let table = store.open("t").await.unwrap().unwrap();
            table
                .append_merging(batch(&[2], &[2.0], &[3.0]))
                .await
                .unwrap();
            let table = store.open("t").await.unwrap().unwrap();
            assert_eq!(table.schema().fields().len(), 3, "{store:?}");
            let expected = [(1, Some(1.0), None), (2, Some(2.0), Some(3.0))];
            assert_eq!(values(&rows(table.as_ref()).await), expected, "{store:?}");
        });
    }
}

#[test]
fn a_commit_through_a_stale_handle_conflicts_and_writes_nothing() {
    let source = || {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, true),
                Field::new("f", DataType::Float64, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![20.0])),
            ],
        )
        .unwrap()
    };
    let wider = || {
        let mut fields = schema().fields().to_vec();
        fields.push(Arc::new(Field::new("h", DataType::Float64, true)));
        let b = batch(&[3], &[3.0], &[3.0]);
        let mut columns = b.columns().to_vec();
        columns.push(Arc::new(Float64Array::from(vec![3.0])));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    };
    let mutations = ["append", "append_merging", "merge", "set_column_metadata"];
    for (_dir, store) in stores() {
        runtime().block_on(async {
            store
                .create(
                    "t",
                    batch(&[1, 2], &[1.0, 2.0], &[0.0, 0.0]),
                    &CreateOptions::default(),
                )
                .await
                .unwrap();
            for (i, mutation) in mutations.into_iter().enumerate() {
                let stale = store.open("t").await.unwrap().unwrap();
                let fresh = store.open("t").await.unwrap().unwrap();
                let k = 100 + i as i64;
                fresh.append(batch(&[k], &[0.0], &[0.0])).await.unwrap();
                let before = store.open("t").await.unwrap().unwrap();
                let before = (before.schema(), values(&rows(before.as_ref()).await));
                let err = match mutation {
                    "append" => stale.append(batch(&[3], &[3.0], &[3.0])).await.err(),
                    "append_merging" => stale.append_merging(wider()).await.err(),
                    "merge" => stale
                        .merge(source(), &["k".into()], &["f".into()])
                        .await
                        .err(),
                    _ => stale
                        .set_column_metadata("f", HashMap::from([("a".into(), "b".into())]))
                        .await
                        .err(),
                }
                .unwrap_or_else(|| panic!("{store:?}: {mutation} committed"));
                assert!(
                    matches!(&err, gamma_lake::Error::Conflict { table } if table == "t"),
                    "{store:?}: {mutation}: {err}"
                );
                let after = store.open("t").await.unwrap().unwrap();
                let after = (after.schema(), values(&rows(after.as_ref()).await));
                assert_eq!(before, after, "{store:?}: {mutation}");
            }
        });
    }
}

#[cfg(unix)]
#[test]
fn every_path_to_a_new_lake_has_one_location() {
    let dir = TempDir::new().unwrap();
    let real = dir.path().canonicalize().unwrap().join("real");
    std::fs::create_dir(&real).unwrap();
    let alias = dir.path().join("alias");
    std::os::unix::fs::symlink(&real, &alias).unwrap();
    let created = delta_store(alias.join("lake").to_str().unwrap(), HashMap::new()).unwrap();
    std::fs::create_dir(real.join("lake")).unwrap();
    let reopened = delta_store(real.join("lake").to_str().unwrap(), HashMap::new()).unwrap();
    assert_eq!(
        created.location(),
        format!("{}/", real.join("lake").display())
    );
    assert_eq!(created.location(), reopened.location());
    assert!(delta_store(alias.join("x/../lake").to_str().unwrap(), HashMap::new()).is_err());
}

#[test]
fn delta_store_locations_and_secrets() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().canonicalize().unwrap();
    let store = delta_store(path.to_str().unwrap(), HashMap::new()).unwrap();
    assert_eq!(store.location(), format!("{}/", path.display()));
    let url = url::Url::from_directory_path(&path).unwrap();
    let store = delta_store(url.as_str(), HashMap::new()).unwrap();
    assert_eq!(store.location(), format!("{}/", path.display()));

    let secret = HashMap::from([("AWS_SECRET_ACCESS_KEY".into(), "hunter2".into())]);
    let store = delta_store("s3://bucket/lake", secret).unwrap();
    assert_eq!(store.location(), "s3://bucket/lake/");
    assert!(!format!("{store:?}").contains("hunter2"));
    assert!(delta_store("s3://bucket/50%25", HashMap::new()).is_err());
    assert!(delta_store("s3://bucket/lake#x", HashMap::new()).is_err());
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}

/// Waits long enough for commits on either side to have distinct times.
fn tick() -> i64 {
    std::thread::sleep(std::time::Duration::from_millis(30));
    let now = now_micros();
    std::thread::sleep(std::time::Duration::from_millis(30));
    now
}

#[test]
fn column_metadata_is_upserted_and_kept_by_later_commits() {
    for (_dir, store) in stores() {
        runtime().block_on(async {
            let options = CreateOptions::default();
            store
                .create("t", batch(&[1, 2], &[1.0, 2.0], &[0.0, 0.0]), &options)
                .await
                .unwrap();
            let table = store.open("t").await.unwrap().unwrap();
            let one = |k: &str, v: &str| HashMap::from([(k.to_string(), v.to_string())]);
            let table = table
                .set_column_metadata("f", one("comment", "first"))
                .await
                .unwrap();
            let table = table
                .set_column_metadata("f", one("tags", "[\"x\"]"))
                .await
                .unwrap();
            table
                .set_column_metadata("g", one("comment", "other"))
                .await
                .unwrap();
            let metadata = |schema: &Schema, column: &str| {
                schema.field_with_name(column).unwrap().metadata().clone()
            };
            let opened = store.open("t").await.unwrap().unwrap();
            let expected = HashMap::from([
                ("comment".to_string(), "first".to_string()),
                ("tags".to_string(), "[\"x\"]".to_string()),
            ]);
            assert_eq!(metadata(&opened.schema(), "f"), expected);
            assert_eq!(metadata(&opened.schema(), "g"), one("comment", "other"));
            assert!(metadata(&opened.schema(), "k").is_empty());
            let missing = opened.set_column_metadata("nope", one("a", "b")).await;
            assert!(missing.is_err());

            let opened = store.open("t").await.unwrap().unwrap();
            let next = batch(&[3], &[3.0], &[0.0]);
            let next = RecordBatch::try_new(opened.schema(), next.columns().to_vec()).unwrap();
            opened.append(next).await.unwrap();
            let opened = store.open("t").await.unwrap().unwrap();
            assert_eq!(metadata(&opened.schema(), "f"), expected);
            assert_eq!(rows(opened.as_ref()).await.num_rows(), 3);
        });
    }
}

#[test]
fn restore_returns_to_the_newest_version_at_a_time() {
    for (_dir, store) in stores() {
        runtime().block_on(async {
            let options = CreateOptions::default();
            let before_create = tick();
            store
                .create("t", batch(&[1], &[1.0], &[0.0]), &options)
                .await
                .unwrap();
            let open = || async { store.open("t").await.unwrap().unwrap() };
            open()
                .await
                .append(batch(&[2], &[2.0], &[0.0]))
                .await
                .unwrap();
            let cutoff = tick();
            assert!(
                !open().await.restore(cutoff).await.unwrap(),
                "nothing to undo"
            );
            open()
                .await
                .append(batch(&[3], &[3.0], &[0.0]))
                .await
                .unwrap();
            open()
                .await
                .set_column_metadata("f", HashMap::from([("comment".into(), "x".into())]))
                .await
                .unwrap();
            assert!(open().await.restore(cutoff).await.unwrap());
            let restored = open().await;
            assert_eq!(values(&rows(restored.as_ref()).await).len(), 2);
            assert!(restored
                .schema()
                .field_with_name("f")
                .unwrap()
                .metadata()
                .is_empty());
            // Restoring again commits once more but keeps the same contents.
            open().await.restore(cutoff).await.unwrap();
            assert_eq!(values(&rows(open().await.as_ref()).await).len(), 2);
            assert!(open().await.restore(before_create).await.is_err());
            // Writes continue from the restored version.
            open()
                .await
                .append(batch(&[4], &[4.0], &[0.0]))
                .await
                .unwrap();
            let keys: Vec<i64> = values(&rows(open().await.as_ref()).await)
                .into_iter()
                .map(|r| r.0)
                .collect();
            assert_eq!(keys, vec![1, 2, 4]);
        });
    }
}
