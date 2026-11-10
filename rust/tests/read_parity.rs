//! Parity with the Python reference, driven by `tests/fixtures/queries.json`.
//!
//! `make fixtures` generates the fixtures, which these tests need.

mod common;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::open_delta;
use gamma_lake::arrow::array::{Array, ArrayRef, StringArray};
use gamma_lake::arrow::compute::{
    cast, cast_with_options, concat_batches, lexsort_to_indices, take_record_batch, CastOptions,
    SortColumn,
};
use gamma_lake::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use gamma_lake::arrow::ipc::reader::FileReader;
use gamma_lake::arrow::record_batch::RecordBatch;
use gamma_lake::{BatchStream, Dnf, Error, Lake, LakeOptions, Op, Term};
use serde_json::Value;

const MISSING: &str = "parity fixtures missing; run `make fixtures`";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn queries() -> Vec<Value> {
    let text = std::fs::read_to_string(fixtures().join("queries.json")).expect(MISSING);
    serde_json::from_str::<Value>(&text)
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn lake_path(lake: &str) -> PathBuf {
    let path = fixtures().join("lakes").join(lake);
    assert!(path.is_dir(), "{MISSING}");
    path
}

fn open_with(lake: &str, options: LakeOptions) -> Lake {
    open_delta(lake_path(lake).to_str().unwrap(), options).unwrap()
}

fn literals(values: &[Value]) -> ArrayRef {
    let kind = values
        .first()
        .map(|v| v["type"].as_str().unwrap())
        .unwrap_or("utf8");
    let raw = || values.iter().map(|v| v["value"].clone());
    let strict = CastOptions {
        safe: false,
        ..Default::default()
    };
    let timestamps = |unit| {
        let strings: ArrayRef = Arc::new(StringArray::from_iter_values(
            raw().map(|v| v.as_str().unwrap().to_string()),
        ));
        cast_with_options(
            &strings,
            &DataType::Timestamp(unit, Some("UTC".into())),
            &strict,
        )
        .unwrap()
    };
    match kind {
        "timestamp_us_utc" => timestamps(TimeUnit::Microsecond),
        "timestamp_ns_utc" => timestamps(TimeUnit::Nanosecond),
        "utf8" => Arc::new(StringArray::from_iter_values(
            raw().map(|v| v.as_str().unwrap().to_string()),
        )),
        "int64" => Arc::new(gamma_lake::arrow::array::Int64Array::from_iter_values(
            raw().map(|v| v.as_i64().unwrap()),
        )),
        "float64" => Arc::new(gamma_lake::arrow::array::Float64Array::from_iter_values(
            raw().map(|v| v.as_f64().unwrap()),
        )),
        other => panic!("unknown literal type {other}"),
    }
}

fn dnf(value: &Value) -> Dnf {
    Dnf(value
        .as_array()
        .unwrap()
        .iter()
        .map(|conjunction| {
            conjunction
                .as_array()
                .unwrap()
                .iter()
                .map(|t| Term {
                    column: t["column"].as_str().unwrap().into(),
                    op: match t["op"].as_str().unwrap() {
                        "eq" => Op::Eq,
                        "not_eq" => Op::NotEq,
                        "lt" => Op::Lt,
                        "lt_eq" => Op::LtEq,
                        "gt" => Op::Gt,
                        "gt_eq" => Op::GtEq,
                        "in" => Op::In,
                        "not_in" => Op::NotIn,
                        "is_null" => Op::IsNull,
                        "is_not_null" => Op::IsNotNull,
                        other => panic!("unknown op {other}"),
                    },
                    values: literals(t["values"].as_array().unwrap()),
                })
                .collect()
        })
        .collect())
}

fn names(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

fn run(lake: &Lake, query: &Value) -> Result<BatchStream, Error> {
    let filter = dnf(&query["dnf"]);
    match query["kind"].as_str().unwrap() {
        "read" => {
            let features = names(&query["features"]);
            let targets = names(&query["targets"]);
            let features: Vec<&str> = features.iter().map(String::as_str).collect();
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
            lake.read(&features, &targets, &filter)
        }
        "feature_metadata" => lake.feature_metadata(&filter),
        "latest_features" => lake.latest_features(),
        "table_metadata" => lake.table_metadata(&filter),
        "index" => lake.index(&filter),
        other => panic!("unknown kind {other}"),
    }
}

/// Plain string layout, so view and large strings compare equal to `Utf8`.
fn plain(batch: &RecordBatch) -> RecordBatch {
    let mut fields = vec![];
    let mut columns = vec![];
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        let target = match field.data_type() {
            DataType::Utf8View | DataType::LargeUtf8 => DataType::Utf8,
            DataType::BinaryView | DataType::LargeBinary => DataType::Binary,
            other => other.clone(),
        };
        columns.push(cast(column, &target).unwrap());
        fields.push(Field::new(field.name(), target, true));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn try_collect(stream: BatchStream) -> Result<RecordBatch, String> {
    let schema = gamma_lake::arrow::record_batch::RecordBatchReader::schema(&stream);
    let batches = stream
        .collect::<Result<Vec<RecordBatch>, _>>()
        .map_err(|e| e.to_string())?;
    for batch in &batches {
        assert_eq!(
            batch.schema(),
            schema,
            "batch schema differs from stream schema"
        );
    }
    Ok(plain(&concat_batches(&schema, &batches).unwrap()))
}

fn expected(name: &str) -> RecordBatch {
    let file = File::open(fixtures().join("expected").join(format!("{name}.arrow"))).unwrap();
    let reader = FileReader::try_new(file, None).unwrap();
    let schema = reader.schema();
    let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
    plain(&concat_batches(&schema, &batches).unwrap())
}

fn sorted(batch: &RecordBatch, by: &[String]) -> RecordBatch {
    let columns: Vec<SortColumn> = by
        .iter()
        .map(|c| SortColumn {
            values: batch.column_by_name(c).unwrap().clone(),
            options: None,
        })
        .collect();
    take_record_batch(batch, &lexsort_to_indices(&columns, None).unwrap()).unwrap()
}

fn assert_same(name: &str, actual: &RecordBatch, expected: &RecordBatch) {
    let types = |b: &RecordBatch| {
        b.schema()
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(types(actual), types(expected), "{name}: schema");
    assert_eq!(actual.num_rows(), expected.num_rows(), "{name}: row count");
    for (i, (a, e)) in actual.columns().iter().zip(expected.columns()).enumerate() {
        assert_eq!(
            a.to_data(),
            e.to_data(),
            "{name}: column {}",
            actual.schema().field(i).name()
        );
    }
}

#[test]
fn queries_match_reference() {
    let queries = queries();
    assert!(!queries.is_empty());
    let default = LakeOptions::default().batch_size;
    for (validate, batch_size, memory_limit) in [
        (false, default, None),
        (true, default, None),
        (false, default, Some(1 << 20)),
        (true, default, Some(1 << 20)),
        (true, 2, None),
        (true, 2, Some(1 << 20)),
    ] {
        let options = LakeOptions {
            validate,
            batch_size,
            memory_limit,
            ..Default::default()
        };
        let context =
            format!("validate {validate}, batch size {batch_size}, memory limit {memory_limit:?}");
        check_all(&queries, &options, &context);
    }
}

fn check_all(queries: &[Value], options: &LakeOptions, context: &str) {
    let validate = options.validate;
    for query in queries {
        let file = query["name"].as_str().unwrap();
        let name = &format!("{file} ({context})");
        let lake = open_with(query["lake"].as_str().unwrap(), options.clone());
        let result = run(&lake, query);
        let expect = match query.get("validate_expect") {
            Some(e) if validate => e.as_str().unwrap(),
            _ => query["expect"].as_str().unwrap(),
        };
        match expect {
            // Alignment is checked while streaming, so the stream opens and then fails.
            "error:integrity" if query["kind"] == "read" => {
                let stream = result.unwrap_or_else(|e| panic!("{name}: {e}"));
                let error = try_collect(stream).expect_err(name);
                assert!(error.contains("integrity check failed"), "{name}: {error}");
            }
            "ok" => {
                let stream = result.unwrap_or_else(|e| panic!("{name}: {e}"));
                let actual = try_collect(stream).unwrap_or_else(|e| panic!("{name}: {e}"));
                let mut expected = expected(file);
                let mut actual = actual;
                if let Some(by) = query.get("sort_by") {
                    let by = names(by);
                    actual = sorted(&actual, &by);
                    expected = sorted(&expected, &by);
                }
                assert_same(name, &actual, &expected);
            }
            expect => {
                let kind = expect.strip_prefix("error:").unwrap();
                match result {
                    Ok(_) => panic!("{name}: expected {kind}, got a stream"),
                    Err(e) => assert_eq!(e.kind(), kind, "{name}: {e}"),
                }
            }
        }
    }
}

#[test]
fn memory_limit_too_small_for_a_batch_fails_the_read() {
    let query = queries()
        .into_iter()
        .find(|q| q["name"] == "basic_all")
        .unwrap();
    let lake = open_with(
        "basic",
        LakeOptions {
            memory_limit: Some(1),
            ..Default::default()
        },
    );
    let error = run(&lake, &query)
        .map_err(|e| e.to_string())
        .and_then(try_collect)
        .unwrap_err();
    assert!(error.contains("Resources exhausted"), "{error}");
}

#[test]
fn relocated_lake_reads_the_same() {
    let dir = tempfile::tempdir().unwrap();
    copy_dir(&lake_path("basic"), dir.path());
    let query = queries()
        .into_iter()
        .find(|q| q["name"] == "basic_all")
        .unwrap();
    let lake = open_delta(dir.path().to_str().unwrap(), LakeOptions::default()).unwrap();
    assert_same(
        "relocated",
        &try_collect(run(&lake, &query).unwrap()).unwrap(),
        &expected("basic_all"),
    );
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn missing_lake_is_not_initialized() {
    let dir = tempfile::tempdir().unwrap();
    let empty_log = dir.path().join("empty_log");
    std::fs::create_dir_all(empty_log.join("index/_delta_log")).unwrap();
    for path in [
        dir.path().to_path_buf(),
        dir.path().join("absent"),
        empty_log,
    ] {
        let err = open_delta(path.to_str().unwrap(), LakeOptions::default()).unwrap_err();
        assert!(matches!(err, Error::NotInitialized(_)), "{err}");
    }
}

#[test]
fn rejects_unknown_schemes_bad_addresses_and_unknown_primary_key() {
    // A scheme without an object store is an error, not an absent lake.
    let err = open_delta("nosuch://bucket/lake", LakeOptions::default()).unwrap_err();
    assert!(matches!(err, Error::Store(_)), "{err}");
    let err = open_delta("s3://bucket/lake?x=1", LakeOptions::default()).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    // Decodes to a non-UTF-8 path.
    let err = open_delta("file:///nonexistent/%FF", LakeOptions::default()).unwrap_err();
    assert!(matches!(err, Error::NotInitialized(_)), "{err}");
    let path = lake_path("basic");
    for index_path in ["../basic/index", "/index", "", "ind%65x"] {
        let options = LakeOptions {
            index_path: index_path.into(),
            ..Default::default()
        };
        let err = open_delta(path.to_str().unwrap(), options).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "{index_path}: {err}"
        );
    }
    let options = LakeOptions {
        primary_sort_key: "nope".into(),
        ..Default::default()
    };
    let err = open_delta(path.to_str().unwrap(), options).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
}

#[test]
fn small_batches_and_stream_outlives_lake() {
    let options = LakeOptions {
        batch_size: 3,
        ..Default::default()
    };
    let lake = open_with("basic", options.clone());
    let stream = lake
        .read(&["f1", "f2", "f3", "f4"], &["t1"], &Dnf::default())
        .unwrap();
    drop(lake);
    let batches: Vec<RecordBatch> = stream.map(|b| b.unwrap()).collect();
    assert!(batches.len() > 1);
    assert!(batches.iter().all(|b| b.num_rows() <= 3));
    let schema = batches[0].schema();
    let all = plain(&concat_batches(&schema, &batches).unwrap());
    assert_same("small batches", &all, &expected("basic_all"));

    let lake = open_with("basic", options);
    for stream in [
        lake.latest_features().unwrap(),
        lake.feature_metadata(&Dnf::default()).unwrap(),
    ] {
        let sizes: Vec<usize> = stream.map(|b| b.unwrap().num_rows()).collect();
        assert_eq!(sizes.iter().sum::<usize>(), 5);
        assert!(sizes.iter().all(|&n| n <= 3), "{sizes:?}");
    }
}
