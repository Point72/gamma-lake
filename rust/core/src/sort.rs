//! Sorting read inputs with DataFusion.

use std::sync::Arc;

use datafusion::dataframe::DataFrame;
use datafusion::error::DataFusionError;
use datafusion::execution::memory_pool::FairSpillPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{execute_stream, ExecutionPlan, SendableRecordBatchStream};
use futures::StreamExt;

use crate::error::{Error, Result};

/// Executes each scan sequentially and returns it sorted by its first `keys` columns
/// (ascending, nulls first). Key columns are dropped unless `keep_keys`.
pub(crate) async fn sorted_streams(
    frames: Vec<DataFrame>,
    keys: usize,
    keep_keys: bool,
) -> Result<Vec<SendableRecordBatchStream>> {
    sort_streams(frames, keys, keep_keys, None).await
}

/// As [`sorted_streams`], with each sort limited to `budget` bytes and spilling to disk.
pub(crate) async fn spilling_sorted_streams(
    frames: Vec<DataFrame>,
    keys: usize,
    keep_keys: bool,
    budget: usize,
) -> Result<Vec<SendableRecordBatchStream>> {
    sort_streams(frames, keys, keep_keys, Some(budget)).await
}

async fn sort_streams(
    frames: Vec<DataFrame>,
    keys: usize,
    keep_keys: bool,
    budget: Option<usize>,
) -> Result<Vec<SendableRecordBatchStream>> {
    let mut streams = Vec::with_capacity(frames.len());
    for frame in frames {
        let mut stream = sort(frame, keys, keep_keys, budget).await?.1;
        // DataFusion produces the first batch only after sorting the complete scan. Polling
        // here ensures only one physical table scan is materialized at a time.
        let first = stream.next().await.transpose()?;
        let schema = stream.schema();
        streams.push(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(first.map(Ok)).chain(stream),
        )) as SendableRecordBatchStream);
    }
    Ok(streams)
}

async fn sort(
    frame: DataFrame,
    keys: usize,
    keep_keys: bool,
    budget: Option<usize>,
) -> Result<(Arc<dyn ExecutionPlan>, SendableRecordBatchStream)> {
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
    let stream = execute_stream(plan.clone(), Arc::new(task))?;
    Ok((plan, stream))
}

pub(crate) async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let permit = crate::runtime::cpu_permits()
        .acquire()
        .await
        .map_err(|e| Error::DataFusion(DataFusionError::External(Box::new(e))))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await
    .map_err(|e| Error::DataFusion(DataFusionError::External(Box::new(e))))?
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

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
            let mut streams = sorted_streams(vec![frame(vec![3, 1, 2])], 1, false)
                .await
                .unwrap();
            streams.remove(0).try_collect::<Vec<_>>().await.unwrap()
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
            let mut streams =
                spilling_sorted_streams(vec![frame((0..10_000).rev().collect())], 1, true, 1 << 20)
                    .await
                    .unwrap();
            streams.remove(0).try_collect::<Vec<_>>().await.unwrap()
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
