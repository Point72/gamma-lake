//! `FeatureTable`: SQL over a lake equals SQL over the materialized read, with or without
//! pushdown.

use std::sync::Arc;

use datafusion::datasource::MemTable;
use datafusion::physical_plan::displayable;
use datafusion::prelude::SessionContext;
use gamma_lake_core::arrow::array::{ArrayRef, AsArray, Float64Array, Int64Array, StringArray};
use gamma_lake_core::arrow::compute::concat_batches;
use gamma_lake_core::arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use gamma_lake_core::arrow::record_batch::RecordBatch;
use gamma_lake_core::{
    AddOptions, BatchStream, Dnf, Error, FeatureTable, Lake, LakeOptions, MemoryStore, OverlapMode,
    SignalType,
};
use tokio::runtime::Runtime;

const FEATURES: [&str; 5] = ["f", "g", "h", "x", "s"];

fn frame(rows: &[(i64, &str)], features: &[(&str, f64)]) -> RecordBatch {
    let mut fields = vec![
        Field::new("ts", DataType::Int64, true),
        Field::new("sym", DataType::Utf8, true),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
        Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.1))),
    ];
    for (name, scale) in features {
        fields.push(Field::new(*name, DataType::Float64, true));
        columns.push(Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.0 as f64 * scale),
        )));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn keys(ts: impl IntoIterator<Item = i64>) -> Vec<(i64, &'static str)> {
    ts.into_iter()
        .flat_map(|t| ["a", "b"].map(move |s| (t, s)))
        .collect()
}

fn add(lake: &Lake, batch: RecordBatch, signal_type: SignalType, params: Option<&str>) {
    let options = AddOptions {
        signal_type,
        feature_params: params.map(str::to_string),
        ..Default::default()
    };
    lake.add(vec![batch], &options).unwrap();
}

/// Dense `f` and `g` in one table over ts 1..=6, dense `h` over 1..=4 (shorter than the
/// index), backward as-of `x` and sparse `s`.
fn lake(validate: bool) -> Lake {
    let index = Schema::new(vec![
        Field::new("ts", DataType::Int64, true),
        Field::new("sym", DataType::Utf8, true),
    ]);
    let options = LakeOptions {
        primary_sort_key: "ts".into(),
        validate,
        batch_size: 3,
        ..Default::default()
    };
    let lake =
        Lake::create_with_store(Arc::new(MemoryStore::new()), Arc::new(index), options).unwrap();
    let dense = SignalType::Feature;
    add(
        &lake,
        frame(&keys(1..=6), &[("f", 1.0), ("g", 2.0)]),
        dense,
        None,
    );
    add(&lake, frame(&keys(1..=4), &[("h", 3.0)]), dense, None);
    let backward = Some(r#"{"strategy": "backward"}"#);
    add(
        &lake,
        frame(&[(2, "a"), (5, "b")], &[("x", 10.0)]),
        SignalType::AsOf,
        backward,
    );
    add(
        &lake,
        frame(&[(3, "a"), (6, "b")], &[("s", 1.0)]),
        SignalType::Sparse,
        None,
    );
    lake
}

fn collect(stream: BatchStream) -> RecordBatch {
    let schema = gamma_lake_core::arrow::record_batch::RecordBatchReader::schema(&stream);
    let batches = stream.collect::<Result<Vec<_>, _>>().unwrap();
    concat_batches(&schema, &batches).unwrap()
}

/// A session with the feature table as `t` and its materialized read as `m`.
fn session(lake: &Lake, features: &[&str]) -> SessionContext {
    let table = lake.feature_table(features, &[]).unwrap();
    let full = collect(lake.read(features, &[], &Dnf::default()).unwrap());
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(table)).unwrap();
    let memory = MemTable::try_new(full.schema(), vec![vec![full]]).unwrap();
    ctx.register_table("m", Arc::new(memory)).unwrap();
    ctx
}

fn runtime() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

async fn sql(ctx: &SessionContext, query: &str) -> RecordBatch {
    let df = ctx.sql(query).await.unwrap();
    let schema = df.schema().inner().clone();
    concat_batches(&schema, &df.collect().await.unwrap()).unwrap()
}

async fn plan(ctx: &SessionContext, query: &str) -> String {
    let physical = ctx
        .sql(query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text = displayable(physical.as_ref()).indent(true).to_string();
    text
}

#[test]
fn select_star_equals_the_read() {
    let lake = lake(true);
    let ctx = session(&lake, &FEATURES);
    runtime().block_on(async {
        let ours = sql(&ctx, "SELECT * FROM t").await;
        let read = sql(&ctx, "SELECT * FROM m").await;
        assert_eq!(ours.num_rows(), 12);
        assert_eq!(ours, read);
    });
}

#[test]
fn pushed_filters_give_the_rows_of_the_unfiltered_read() {
    for validate in [true, false] {
        let lake = lake(validate);
        let ctx = session(&lake, &FEATURES);
        runtime().block_on(async {
            let pushed = [
                "ts >= 3",
                "3 > ts",
                "ts < 3 OR sym = 'b'",
                "ts IN (2, 5)",
                "ts NOT IN (2, 5)",
                "ts BETWEEN 2 AND 4",
                "ts NOT BETWEEN 2 AND 4",
                "sym IS NULL",
                "sym IS NOT NULL AND ts <> 4",
                "ts = 5 AND f > 2.5",
                "(ts = 1 OR ts = 3) AND (sym = 'a' OR sym > 'a')",
            ];
            let kept = ["ts % 2 = 0", "f > 2.5", "length(sym) = 1"];
            for (filter, expect_pushed) in pushed
                .iter()
                .map(|f| (f, true))
                .chain(kept.iter().map(|f| (f, false)))
            {
                let query =
                    |table| format!("SELECT * FROM {table} WHERE {filter} ORDER BY ts, sym");
                let ours = sql(&ctx, &query("t")).await;
                let read = sql(&ctx, &query("m")).await;
                assert_eq!(ours, read, "{filter}");
                let text = plan(&ctx, &query("t")).await;
                assert_eq!(text.contains("pushed=["), expect_pushed, "{filter}\n{text}");
            }
        });
    }
}

#[test]
fn projection_reads_only_the_projected_features_and_keeps_every_row() {
    let lake = lake(true);
    let ctx = session(&lake, &FEATURES);
    runtime().block_on(async {
        for template in [
            "SELECT h, sym FROM {} ORDER BY ts, sym",
            "SELECT sym, ts, s FROM {} ORDER BY ts, sym",
            "SELECT count(*) AS n FROM {}",
            "SELECT g, f FROM {} WHERE ts > 4 ORDER BY ts, sym",
        ] {
            let query = |table: &str| template.replace("{}", table);
            assert_eq!(
                sql(&ctx, &query("t")).await,
                sql(&ctx, &query("m")).await,
                "{template}"
            );
        }
        let text = plan(&ctx, "SELECT h FROM t").await;
        assert!(text.contains("features=[h], projection=[h]"), "{text}");
        let text = plan(&ctx, "SELECT count(*) FROM t").await;
        assert!(text.contains("features=[]"), "{text}");
    });
}

#[test]
fn sort_key_order_needs_no_sort() {
    let lake = lake(true);
    let ctx = session(&lake, &FEATURES);
    runtime().block_on(async {
        let sorted = |order: &str| format!("SELECT f, ts, sym FROM t ORDER BY {order}");
        for order in ["ts NULLS FIRST", "ts NULLS FIRST, sym NULLS FIRST"] {
            let text = plan(&ctx, &sorted(order)).await;
            assert!(!text.contains("SortExec"), "{order}\n{text}");
        }
        for order in ["ts DESC", "sym NULLS FIRST", "ts NULLS LAST"] {
            let text = plan(&ctx, &sorted(order)).await;
            assert!(text.contains("SortExec"), "{order}\n{text}");
        }
        // Without the primary key the order is unknown.
        let text = plan(&ctx, "SELECT sym FROM t ORDER BY sym NULLS FIRST").await;
        assert!(text.contains("SortExec"), "{text}");
    });
}

#[test]
fn the_table_reads_its_versions_with_their_current_contents() {
    let lake = lake(true);
    let table = Arc::new(lake.feature_table(&["f"], &[]).unwrap());
    let ctx = SessionContext::new();
    ctx.register_table("t", table.clone()).unwrap();
    let rt = runtime();
    let query = "SELECT sum(f) AS f, count(*) AS n FROM t";
    let totals = |ctx: &SessionContext| {
        let batch = rt.block_on(sql(ctx, query));
        let f = batch.column(0).as_primitive::<Float64Type>().value(0);
        let n = batch.column(1).as_primitive::<Int64Type>().value(0);
        (f, n)
    };
    assert_eq!(totals(&ctx), (42.0, 12));

    // Appended rows are in the same table.
    add(
        &lake,
        frame(&keys(7..=7), &[("f", 1.0)]),
        SignalType::Feature,
        None,
    );
    assert_eq!(totals(&ctx), (56.0, 14));
    // A merge-mode overwrite also writes the same table.
    let merge = AddOptions {
        overlap_mode: OverlapMode::Merge,
        ..Default::default()
    };
    lake.add(vec![frame(&keys(1..=1), &[("f", 101.0)])], &merge)
        .unwrap();
    assert_eq!(totals(&ctx), (256.0, 14));
    // A copy-mode overwrite writes a new version, which only a new table reads.
    let copy = AddOptions::default();
    lake.add(vec![frame(&keys(1..=1), &[("f", 1001.0)])], &copy)
        .unwrap();
    assert_eq!(totals(&ctx), (256.0, 14));
    let fresh = SessionContext::new();
    fresh
        .register_table("t", Arc::new(lake.feature_table(&["f"], &[]).unwrap()))
        .unwrap();
    assert_eq!(totals(&fresh), (2056.0, 14));
    assert_eq!(table.features()[0].name, "f");
}

#[test]
fn creation_resolves_the_features() {
    let lake = lake(true);
    let err = lake.feature_table(&["nope"], &[]).unwrap_err();
    assert!(matches!(err, Error::MissingFeatures(_)), "{err}");
    let table = lake.feature_table(&["x", "f"], &[]).unwrap();
    let names: Vec<_> = table.features().iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["x", "f"]);
    let selected = FeatureTable::from_selected(lake.clone(), table.features().to_vec()).unwrap();
    assert_eq!(
        datafusion::catalog::TableProvider::schema(&selected),
        lake.read_schema(&["x", "f"], &[]).unwrap()
    );
}

#[test]
fn a_dropped_query_stops_and_others_still_run() {
    let lake = lake(true);
    let ctx = session(&lake, &FEATURES);
    let rt = runtime();
    rt.block_on(async {
        use futures::StreamExt;
        let mut stream = ctx
            .sql("SELECT * FROM t")
            .await
            .unwrap()
            .execute_stream()
            .await
            .unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.num_rows(), 3);
        drop(stream);
        let ours = sql(&ctx, "SELECT * FROM t").await;
        assert_eq!(ours.num_rows(), 12);
    });
}
