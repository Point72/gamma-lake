//! Helpers shared by the integration tests; each test crate uses some of them.
#![allow(dead_code)]

use std::sync::Arc;

use gamma_lake::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use gamma_lake::arrow::compute::concat_batches;
use gamma_lake::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use gamma_lake::arrow::record_batch::{RecordBatch, RecordBatchReader};
use gamma_lake::{AddOptions, BatchStream, DeltaStore, Lake, LakeOptions, OverlapMode};

pub fn delta_store(base: &str) -> gamma_lake::Result<DeltaStore> {
    DeltaStore::new(base, Default::default())
}

pub fn open_delta(base: &str, options: LakeOptions) -> gamma_lake::Result<Lake> {
    gamma_lake::open(delta_store(base)?, options)
}

pub fn create_delta(
    base: &str,
    schema: SchemaRef,
    options: LakeOptions,
) -> gamma_lake::Result<Lake> {
    gamma_lake::create(delta_store(base)?, schema, options)
}

pub fn index_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, true),
        Field::new("sym", DataType::Utf8, true),
    ]))
}

/// A frame with the keys `rows` and, per feature, the value `ts * scale`.
pub fn frame(rows: &[(i64, &str)], features: &[(&str, f64)]) -> RecordBatch {
    let mut fields = index_schema().fields().to_vec();
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
        Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.1))),
    ];
    for (name, scale) in features {
        fields.push(Arc::new(Field::new(*name, DataType::Float64, true)));
        columns.push(Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|r| r.0 as f64 * scale),
        )));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// Every `(ts, sym)` pair.
pub fn keys(ts: impl IntoIterator<Item = i64>, syms: &[&'static str]) -> Vec<(i64, &'static str)> {
    ts.into_iter()
        .flat_map(|t| syms.iter().map(move |s| (t, *s)))
        .collect()
}

pub fn add(lake: &Lake, batch: RecordBatch, overlap_mode: OverlapMode) {
    let options = AddOptions {
        overlap_mode,
        ..Default::default()
    };
    lake.add(vec![batch], &options).unwrap();
}

pub fn collect(stream: BatchStream) -> RecordBatch {
    let schema = stream.schema();
    let batches = stream.collect::<Result<Vec<_>, _>>().unwrap();
    concat_batches(&schema, &batches).unwrap()
}

/// Waits long enough for commits on either side to have distinct times; returns the time
/// in between, in microseconds.
pub fn tick() -> i64 {
    std::thread::sleep(std::time::Duration::from_millis(30));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    std::thread::sleep(std::time::Duration::from_millis(30));
    now
}
