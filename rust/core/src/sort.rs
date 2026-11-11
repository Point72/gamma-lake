//! Sorting read inputs with DataFusion.

use std::sync::Arc;

use datafusion::dataframe::DataFrame;
use datafusion::execution::memory_pool::FairSpillPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{execute_stream, SendableRecordBatchStream};
use futures::StreamExt;

use crate::error::Result;

/// `frame` sorted by its first `keys` columns (ascending, nulls first), without the key
/// columns unless `keep_keys`. With a `budget`, the sort is limited to that many bytes and
/// spills to disk.
///
/// DataFusion produces the first batch only after sorting the complete scan. It is polled
/// before returning, so callers awaiting one sort at a time materialize one scan at a time.
pub(crate) async fn sorted_stream(
    frame: DataFrame,
    keys: usize,
    keep_keys: bool,
    budget: Option<usize>,
) -> Result<SendableRecordBatchStream> {
    let mut stream = sort(frame, keys, keep_keys, budget).await?;
    let first = stream.next().await.transpose()?;
    let schema = stream.schema();
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::iter(first.map(Ok)).chain(stream),
    )))
}

async fn sort(
    frame: DataFrame,
    keys: usize,
    keep_keys: bool,
    budget: Option<usize>,
) -> Result<SendableRecordBatchStream> {
    let columns = frame.schema().columns();
    let mut frame = frame.sort(
        columns[..keys]
            .iter()
            .map(|column| Expr::Column(column.clone()).sort(true, true))
            .collect(),
    )?;
    if !keep_keys {
        frame = frame.select(columns[keys..].iter().cloned().map(Expr::Column))?;
    }
    let task = frame.task_ctx();
    let plan = frame.create_physical_plan().await?;
    let task = match budget {
        None => task,
        Some(budget) => {
            let row_bytes: usize = frame
                .schema()
                .fields()
                .iter()
                .map(|field| field.data_type().primitive_width().unwrap_or(32))
                .sum();
            let config = task.session_config().clone();
            // A batch is reserved whole before the sort can spill, so keep batches to an
            // eighth of the budget.
            let batch_size = config
                .batch_size()
                .min(budget / 8 / row_bytes.max(1))
                .max(1);
            let mut config = config.with_batch_size(batch_size);
            config.options_mut().execution.sort_spill_reservation_bytes =
                (budget / 8).min(10 << 20);
            let runtime = RuntimeEnvBuilder::from_runtime_env(&task.runtime_env())
                .with_memory_pool(Arc::new(FairSpillPool::new(budget)))
                .build_arc()?;
            task.with_session_config(config).with_runtime(runtime)
        }
    };
    Ok(execute_stream(plan, Arc::new(task))?)
}

#[cfg(test)]
mod tests {
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use futures::TryStreamExt;

    use super::*;

    fn frame(values: Vec<i64>) -> DataFrame {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(values.clone())),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .unwrap();
        let context = SessionContext::new_with_config(SessionConfig::new().with_batch_size(2));
        context
            .read_table(Arc::new(
                MemTable::try_new(schema, vec![vec![batch]]).unwrap(),
            ))
            .unwrap()
    }

    #[test]
    fn datafusion_sorts_and_projects() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let batches = runtime.block_on(async {
            let stream = sorted_stream(frame(vec![3, 1, 2]), 1, false, None)
                .await
                .unwrap();
            stream.try_collect::<Vec<_>>().await.unwrap()
        });
        let values = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, [1, 2, 3]);
    }

    #[test]
    fn spilling_sort_stays_ordered() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let batches = runtime.block_on(async {
            let frame = frame((0..10_000).rev().collect());
            let stream = sorted_stream(frame, 1, true, Some(1 << 20)).await.unwrap();
            stream.try_collect::<Vec<_>>().await.unwrap()
        });
        let values = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, (0..10_000).collect::<Vec<_>>());
    }
}
