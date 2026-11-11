//! Positional alignment of feature tables to the index.
//!
//! A gamma-lake keeps every feature table row-aligned with the sorted index: row `i` of a
//! table, sorted by the sort keys, belongs to row `i` of the sorted index. Reads therefore
//! zip columns by position instead of joining on keys. This follows Polars'
//! `concat(how="horizontal")`: shorter inputs are padded with nulls at the end, and the
//! output is as long as the longest input. A bounded zip is as long as the index instead.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{new_null_array, ArrayRef};
use arrow::compute::{cast_with_options, concat, concat_batches, CastOptions};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::future::try_join_all;
use futures::StreamExt;

use crate::error::integrity;

/// A sorted input to [`zip_aligned`].
pub(crate) struct AlignInput {
    /// Table address, for error messages.
    pub name: String,
    pub stream: SendableRecordBatchStream,
    /// Number of leading sort-key columns, compared with the index and not output.
    pub keys: usize,
}

/// One output column: `(input, column)`, where input 0 is the index.
pub(crate) type Source = (usize, usize);

struct Buffered {
    name: String,
    stream: SendableRecordBatchStream,
    keys: usize,
    batches: VecDeque<RecordBatch>,
    rows: usize,
    done: bool,
}

impl Buffered {
    async fn fill(&mut self, wanted: usize) -> DfResult<()> {
        while self.rows < wanted && !self.done {
            match self.stream.next().await {
                Some(batch) => {
                    let batch = batch?;
                    if batch.num_rows() > 0 {
                        self.rows += batch.num_rows();
                        self.batches.push_back(batch);
                    }
                }
                None => self.done = true,
            }
        }
        Ok(())
    }

    /// Removes up to `n` rows from the front of the buffer.
    fn take(&mut self, n: usize) -> DfResult<RecordBatch> {
        let schema = self.stream.schema();
        let mut parts = vec![];
        let mut needed = n.min(self.rows);
        while needed > 0 {
            let batch = self.batches.pop_front().expect("rows are buffered");
            if batch.num_rows() > needed {
                parts.push(batch.slice(0, needed));
                self.batches
                    .push_front(batch.slice(needed, batch.num_rows() - needed));
                self.rows -= needed;
                needed = 0;
            } else {
                needed -= batch.num_rows();
                self.rows -= batch.num_rows();
                parts.push(batch);
            }
        }
        Ok(concat_batches(&schema, &parts)?)
    }
}

/// Key checks for `validate`: tables must match the index row for row, and index keys
/// must be unique.
struct Validator {
    converter: RowConverter,
    last: Option<OwnedRow>,
}

impl Validator {
    fn new(index: &Schema, keys: usize) -> DfResult<Self> {
        let fields = index.fields()[..keys]
            .iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();
        Ok(Self {
            converter: RowConverter::new(fields)?,
            last: None,
        })
    }

    /// Row-encodes the first `keys` columns of `batch` as the index key types. A key that
    /// does not convert to the index type and back unchanged fails the read for `name`.
    fn rows(&self, name: &str, batch: &RecordBatch, keys: usize, index: &Schema) -> DfResult<Rows> {
        let columns = (0..keys)
            .map(|i| {
                let target = index.field(i).data_type();
                let column = batch.column(i);
                if column.data_type() == target {
                    return Ok(column.clone());
                }
                let strict = CastOptions {
                    safe: false,
                    ..Default::default()
                };
                let exact = cast_with_options(column, target, &strict)
                    .ok()
                    .filter(|c| {
                        cast_with_options(c, column.data_type(), &strict)
                            .is_ok_and(|back| back.to_data() == column.to_data())
                    });
                exact.ok_or_else(|| {
                    integrity(
                        name,
                        format!(
                            "sort key {:?} of type {} does not convert exactly to the index type {target}",
                            index.field(i).name(),
                            column.data_type()
                        ),
                    )
                    .into()
                })
            })
            .collect::<DfResult<Vec<ArrayRef>>>()?;
        Ok(self.converter.convert_columns(&columns)?)
    }

    fn check_index(&mut self, rows: &Rows, offset: usize) -> DfResult<()> {
        for i in 0..rows.num_rows() {
            let previous = if i == 0 {
                self.last.as_ref().map(OwnedRow::row)
            } else {
                Some(rows.row(i - 1))
            };
            if previous == Some(rows.row(i)) {
                return Err(integrity(
                    "index",
                    format!("duplicate sort keys at row {}", offset + i),
                )
                .into());
            }
        }
        if rows.num_rows() > 0 {
            self.last = Some(rows.row(rows.num_rows() - 1).owned());
        }
        Ok(())
    }
}

/// Zips `tables` onto `index` by row position. `layout` lists the output columns. The
/// first `keys` columns of `index` are the sort keys.
///
/// When `bounded`, the output has one row per index row and rows of longer tables are
/// dropped, or reported as an integrity error when validating.
pub(crate) fn zip_aligned(
    index: SendableRecordBatchStream,
    keys: usize,
    tables: Vec<AlignInput>,
    layout: Vec<Source>,
    batch_size: usize,
    validate: bool,
    bounded: bool,
) -> DfResult<SendableRecordBatchStream> {
    let index_schema = index.schema();
    let index = AlignInput {
        name: "index".into(),
        stream: index,
        keys,
    };
    let inputs: Vec<Buffered> = std::iter::once(index)
        .chain(tables)
        .map(|t| Buffered {
            name: t.name,
            stream: t.stream,
            keys: t.keys,
            batches: VecDeque::new(),
            rows: 0,
            done: false,
        })
        .collect();
    // Padding introduces nulls in every column, including the index keys when a table is
    // longer than the index.
    let schema: SchemaRef = Arc::new(Schema::new(
        layout
            .iter()
            .map(|&(input, column)| {
                let field = inputs[input].stream.schema().field(column).clone();
                Field::new(field.name(), field.data_type().clone(), true)
            })
            .collect::<Vec<_>>(),
    ));
    let validator = validate
        .then(|| Validator::new(&index_schema, keys))
        .transpose()?;

    struct State {
        inputs: Vec<Buffered>,
        validator: Option<Validator>,
        offset: usize,
    }
    let state = State {
        inputs,
        validator,
        offset: 0,
    };
    let out_schema = schema.clone();
    let stream = futures::stream::try_unfold(state, move |mut state| {
        let schema = out_schema.clone();
        let layout = layout.clone();
        let index_schema = index_schema.clone();
        async move {
            try_join_all(state.inputs.iter_mut().map(|i| i.fill(batch_size))).await?;
            let n = if bounded {
                state.inputs[0].rows.min(batch_size)
            } else {
                state
                    .inputs
                    .iter()
                    .map(|i| i.rows.min(batch_size))
                    .max()
                    .unwrap_or(0)
            };
            if n == 0 {
                if bounded && state.validator.is_some() {
                    if let Some(input) = state.inputs.iter().skip(1).find(|i| i.rows > 0) {
                        return Err(integrity(
                            &input.name,
                            format!("more rows than the index (from row {})", state.offset),
                        )
                        .into());
                    }
                }
                return Ok(None);
            }
            let taken = state
                .inputs
                .iter_mut()
                .map(|i| i.take(n))
                .collect::<DfResult<Vec<_>>>()?;
            if let Some(validator) = state.validator.as_mut() {
                let index_rows = validator.rows("index", &taken[0], keys, &index_schema)?;
                validator.check_index(&index_rows, state.offset)?;
                // Inputs without keys are built from the index and so match it.
                let keyed = state
                    .inputs
                    .iter()
                    .zip(&taken)
                    .skip(1)
                    .filter(|(i, _)| i.keys > 0);
                for (input, batch) in keyed {
                    if batch.num_rows() > taken[0].num_rows() {
                        return Err(integrity(
                            &input.name,
                            format!(
                                "more rows than the index (from row {})",
                                state.offset + taken[0].num_rows()
                            ),
                        )
                        .into());
                    }
                    let rows = validator.rows(&input.name, batch, input.keys, &index_schema)?;
                    if let Some(i) =
                        (0..rows.num_rows()).find(|&i| rows.row(i) != index_rows.row(i))
                    {
                        return Err(integrity(
                            &input.name,
                            format!(
                                "sort keys at row {} differ from the index",
                                state.offset + i
                            ),
                        )
                        .into());
                    }
                }
            }
            let columns = layout
                .iter()
                .zip(schema.fields())
                .map(|(&(input, column), field)| {
                    let values = taken[input].column(column);
                    Ok(if values.len() == n {
                        values.clone()
                    } else {
                        let padding = new_null_array(field.data_type(), n - values.len());
                        concat(&[values.as_ref(), padding.as_ref()])?
                    })
                })
                .collect::<DfResult<Vec<_>>>()?;
            state.offset += n;
            let batch = RecordBatch::try_new_with_options(
                schema,
                columns,
                &RecordBatchOptions::new().with_row_count(Some(n)),
            )?;
            Ok(Some((batch, state)))
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Float64Type, Int64Type};

    fn stream(
        name: &str,
        batches: Vec<Vec<Option<i64>>>,
        value: Option<&str>,
    ) -> SendableRecordBatchStream {
        let mut fields = vec![Field::new(name, DataType::Int64, true)];
        if let Some(v) = value {
            fields.push(Field::new(v, DataType::Float64, true));
        }
        let schema = Arc::new(Schema::new(fields));
        let batches = batches
            .into_iter()
            .map(|keys| {
                let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys.clone()))];
                if value.is_some() {
                    columns.push(Arc::new(Float64Array::from(
                        keys.iter()
                            .map(|k| k.map(|k| k as f64 * 10.0))
                            .collect::<Vec<_>>(),
                    )));
                }
                Ok(RecordBatch::try_new(schema.clone(), columns).unwrap())
            })
            .collect::<Vec<_>>();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches),
        ))
    }

    fn table(name: &str, batches: Vec<Vec<Option<i64>>>) -> AlignInput {
        AlignInput {
            name: name.into(),
            stream: stream("k", batches, Some(name)),
            keys: 1,
        }
    }

    fn run(
        index: Vec<Vec<Option<i64>>>,
        tables: Vec<AlignInput>,
        batch_size: usize,
        validate: bool,
    ) -> DfResult<Vec<RecordBatch>> {
        zip(index, tables, batch_size, validate, false)
    }

    fn zip(
        index: Vec<Vec<Option<i64>>>,
        tables: Vec<AlignInput>,
        batch_size: usize,
        validate: bool,
        bounded: bool,
    ) -> DfResult<Vec<RecordBatch>> {
        let layout = (1..=tables.len()).map(|i| (i, 1)).chain([(0, 0)]).collect();
        let stream = zip_aligned(
            stream("k", index, None),
            1,
            tables,
            layout,
            batch_size,
            validate,
            bounded,
        )?;
        crate::runtime::block_on(stream.collect::<Vec<_>>())
            .unwrap()
            .into_iter()
            .collect()
    }

    fn column(batches: &[RecordBatch], i: usize) -> Vec<Option<f64>> {
        batches
            .iter()
            .flat_map(|b| {
                let c = arrow::compute::cast(b.column(i), &DataType::Float64).unwrap();
                c.as_primitive::<Float64Type>().iter().collect::<Vec<_>>()
            })
            .collect()
    }

    fn keys(batches: &[RecordBatch], i: usize) -> Vec<Option<i64>> {
        batches
            .iter()
            .flat_map(|b| {
                b.column(i)
                    .as_primitive::<Int64Type>()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn zips_by_position_across_batch_boundaries() {
        let index = vec![vec![Some(1), Some(2)], vec![Some(3), Some(4), Some(5)]];
        let a = table(
            "a",
            vec![vec![Some(1)], vec![Some(2), Some(3), Some(4), Some(5)]],
        );
        let b = table("b", vec![vec![Some(1), Some(2), Some(3)]]);
        let out = run(index, vec![a, b], 2, true).unwrap();
        assert_eq!(
            out.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(keys(&out, 2), (1..=5).map(Some).collect::<Vec<_>>());
        assert_eq!(
            column(&out, 0),
            [10., 20., 30., 40., 50.].map(Some).to_vec()
        );
        assert_eq!(
            column(&out, 1),
            vec![Some(10.), Some(20.), Some(30.), None, None]
        );
    }

    #[test]
    fn longer_tables_extend_the_output_without_validation() {
        let out = run(
            vec![vec![Some(1)]],
            vec![table("a", vec![vec![Some(1), Some(2)]])],
            8,
            false,
        )
        .unwrap();
        assert_eq!(keys(&out, 1), vec![Some(1), None]);
        assert_eq!(column(&out, 0), vec![Some(10.), Some(20.)]);
    }

    #[test]
    fn bounded_zips_end_with_the_index() {
        let tables = || {
            vec![
                table("short", vec![vec![Some(1)]]),
                table("long", vec![vec![Some(1), Some(2)], vec![Some(3)]]),
            ]
        };
        let out = zip(vec![vec![Some(1), Some(2)]], tables(), 1, false, true).unwrap();
        assert_eq!(keys(&out, 2), vec![Some(1), Some(2)]);
        assert_eq!(column(&out, 0), vec![Some(10.), None]);
        assert_eq!(column(&out, 1), vec![Some(10.), Some(20.)]);
        let message = integrity_message(zip(vec![vec![Some(1), Some(2)]], tables(), 1, true, true));
        assert!(
            message.contains("table long")
                && message.contains("more rows than the index (from row 2)"),
            "{message}"
        );
    }

    #[test]
    fn empty_inputs_produce_no_batches() {
        let out = run(vec![], vec![table("a", vec![])], 8, true).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn positional_zip_does_not_look_at_keys_without_validation() {
        let out = run(
            vec![vec![Some(1), Some(2), Some(3)]],
            vec![table("a", vec![vec![Some(1), Some(3)]])],
            8,
            false,
        )
        .unwrap();
        assert_eq!(column(&out, 0), vec![Some(10.), Some(30.), None]);
    }

    fn integrity_message(result: DfResult<Vec<RecordBatch>>) -> String {
        let message = result.unwrap_err().to_string();
        assert!(message.contains("integrity check failed"), "{message}");
        message
    }

    #[test]
    fn validation_rejects_key_mismatch() {
        let message = integrity_message(run(
            vec![vec![Some(1), Some(2), Some(3)]],
            vec![table("a", vec![vec![Some(1), Some(3)]])],
            8,
            true,
        ));
        assert!(
            message.contains("table a") && message.contains("row 1"),
            "{message}"
        );
    }

    #[test]
    fn validation_rejects_tables_longer_than_the_index() {
        let message = integrity_message(run(
            vec![vec![Some(1)]],
            vec![table("a", vec![vec![Some(1), Some(2)]])],
            8,
            true,
        ));
        assert!(message.contains("more rows than the index"), "{message}");
    }

    #[test]
    fn validation_rejects_duplicate_index_keys_across_batches() {
        let message = integrity_message(run(
            vec![vec![Some(1), Some(2)], vec![Some(2)]],
            vec![],
            2,
            true,
        ));
        assert!(
            message.contains("duplicate sort keys at row 2"),
            "{message}"
        );
    }

    fn float_keyed(name: &str, keys: Vec<f64>) -> AlignInput {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Float64, true)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Float64Array::from(keys))]).unwrap();
        AlignInput {
            name: name.into(),
            stream: Box::pin(RecordBatchStreamAdapter::new(
                schema,
                futures::stream::iter([Ok(batch)]),
            )),
            keys: 1,
        }
    }

    #[test]
    fn validation_compares_other_key_types_exactly() {
        let index = || vec![vec![Some(1), Some(2)]];
        let layout = || vec![(0, 0)];
        let read = |table| {
            let stream = zip_aligned(
                stream("k", index(), None),
                1,
                vec![table],
                layout(),
                8,
                true,
                false,
            )?;
            crate::runtime::block_on(stream.collect::<Vec<_>>())
                .unwrap()
                .into_iter()
                .collect::<DfResult<Vec<_>>>()
        };
        assert!(read(float_keyed("exact", vec![1.0, 2.0])).is_ok());
        let message = integrity_message(read(float_keyed("lossy", vec![1.9, 2.0])));
        assert!(
            message.contains("table lossy") && message.contains("does not convert exactly"),
            "{message}"
        );
    }
}
