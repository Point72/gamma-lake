//! Adapts storage-neutral Arrow scans to DataFusion data frames used inside the core.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::dataframe::DataFrame;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion::prelude::SessionContext;
use futures::{StreamExt, TryStreamExt};

use crate::error::{Error, Result};
use crate::predicate::Dnf;
use crate::store::{StoredTable, TableScan};

/// `schema` projected to `columns`, or all of it for `None`.
fn projected(schema: SchemaRef, columns: Option<&[String]>) -> Result<SchemaRef> {
    let Some(columns) = columns else {
        return Ok(schema);
    };
    let positions = columns
        .iter()
        .map(|name| schema.index_of(name))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Arc::new(schema.project(&positions)?))
}

/// Fails unless `actual` has the column names and types of `expected`, in order.
fn check_columns(expected: &Schema, actual: &Schema) -> Result<()> {
    let same = expected.fields().len() == actual.fields().len()
        && expected
            .fields()
            .iter()
            .zip(actual.fields())
            .all(|(e, a)| e.name() == a.name() && e.data_type() == a.data_type());
    if same {
        return Ok(());
    }
    let describe = |schema: &Schema| {
        schema
            .fields()
            .iter()
            .map(|f| format!("{}: {}", f.name(), f.data_type()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(Error::Store(
        format!(
            "a storage scan returned columns [{}] where the table's schema has [{}]",
            describe(actual),
            describe(expected)
        )
        .into(),
    ))
}

/// [`StoredTable::scan`], failing when the scan or one of its batches does not have the
/// columns of the table's schema that it was asked for.
pub(crate) async fn checked_scan(
    table: &dyn StoredTable,
    columns: Option<&[String]>,
    dnf: &Dnf,
    batch_size: usize,
) -> Result<TableScan> {
    let expected = projected(table.schema(), columns)?;
    let scan = table.scan(columns, dnf, batch_size).await?;
    check_columns(&expected, &scan.schema)?;
    let stream = scan.stream.map(move |batch| {
        let batch = batch?;
        check_columns(&expected, &batch.schema())?;
        Ok(batch)
    });
    Ok(TableScan {
        schema: scan.schema,
        stream: Box::pin(stream),
    })
}

struct ScanProvider {
    table: Arc<dyn StoredTable>,
    dnf: Dnf,
}

impl fmt::Debug for ScanProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanProvider").finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for ScanProvider {
    fn schema(&self) -> SchemaRef {
        self.table.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let schema = self.table.schema();
        let columns = projection.map(|positions| {
            positions
                .iter()
                .map(|&i| schema.field(i).name().clone())
                .collect::<Vec<_>>()
        });
        let schema = projected(schema, columns.as_deref()).map_err(DataFusionError::from)?;
        let partition = Arc::new(BackendPartition {
            schema: schema.clone(),
            table: self.table.clone(),
            columns,
            dnf: self.dnf.clone(),
        });
        Ok(Arc::new(StreamingTableExec::try_new(
            schema,
            vec![partition],
            None,
            [],
            false,
            limit,
        )?))
    }
}

struct BackendPartition {
    schema: SchemaRef,
    table: Arc<dyn StoredTable>,
    columns: Option<Vec<String>>,
    dnf: Dnf,
}

impl fmt::Debug for BackendPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendPartition").finish_non_exhaustive()
    }
}

impl PartitionStream for BackendPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let schema = self.schema.clone();
        let (table, columns, dnf) = (self.table.clone(), self.columns.clone(), self.dnf.clone());
        let batch_size = ctx.session_config().batch_size();
        let stream = futures::stream::once(async move {
            let scan = checked_scan(table.as_ref(), columns.as_deref(), &dnf, batch_size)
                .await
                .map_err(DataFusionError::from)?;
            Ok::<_, DataFusionError>(
                scan.stream
                    .map(|batch| batch.map_err(DataFusionError::from)),
            )
        })
        .try_flatten();
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }
}

/// A DataFusion frame over one storage scan, which applies `dnf`. DataFusion is an internal
/// execution adapter; the storage contract itself remains Arrow-only.
pub(crate) fn frame(
    ctx: &SessionContext,
    table: Arc<dyn StoredTable>,
    dnf: &Dnf,
) -> Result<DataFrame> {
    // Rejects predicates the table cannot evaluate now rather than when the scan runs.
    dnf.to_expr(&table.schema(), "t")?;
    Ok(ctx.read_table(Arc::new(ScanProvider {
        table,
        dnf: dnf.clone(),
    }))?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::array::{Float64Array, Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field};

    use super::*;

    /// Stores `id: Int64` but scans it as `Float64`.
    struct Mistyped;

    #[async_trait]
    impl StoredTable for Mistyped {
        fn schema(&self) -> SchemaRef {
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]))
        }

        async fn scan(&self, _: Option<&[String]>, _: &Dnf, _: usize) -> Result<TableScan> {
            let batch =
                RecordBatch::try_from_iter([("id", Arc::new(Float64Array::from(vec![1.0])) as _)])?;
            Ok(TableScan {
                schema: self.schema(),
                stream: Box::pin(futures::stream::iter([Ok(batch)])),
            })
        }

        async fn append(self: Box<Self>, _: RecordBatch) -> Result<()> {
            unreachable!()
        }

        async fn append_merging(self: Box<Self>, _: RecordBatch) -> Result<()> {
            unreachable!()
        }

        async fn merge(self: Box<Self>, _: RecordBatch, _: &[String], _: &[String]) -> Result<()> {
            unreachable!()
        }

        async fn set_column_metadata(
            self: Box<Self>,
            _: &str,
            _: HashMap<String, String>,
        ) -> Result<Box<dyn StoredTable>> {
            unreachable!()
        }

        async fn restore(self: Box<Self>, _: i64) -> Result<bool> {
            unreachable!()
        }
    }

    #[test]
    fn a_scan_batch_of_another_type_is_an_error() {
        let batches = futures::executor::block_on(async {
            let scan = checked_scan(&Mistyped, None, &Dnf::default(), 1).await?;
            scan.stream.try_collect::<Vec<_>>().await
        });
        let message = batches.unwrap_err().to_string();
        assert!(message.contains("[id: Float64]"), "{message}");
        assert!(message.contains("[id: Int64]"), "{message}");
    }

    #[test]
    fn a_projected_scan_checks_only_the_projected_columns() {
        let table = crate::store::MemoryStore::new();
        let batches = futures::executor::block_on(async {
            use crate::store::{CreateOptions, TableStore};
            let batch = RecordBatch::try_from_iter([
                ("id", Arc::new(Int64Array::from(vec![1, 2])) as _),
                ("x", Arc::new(Float64Array::from(vec![1.0, 2.0])) as _),
            ])?;
            table.create("t", batch, &CreateOptions::default()).await?;
            let stored = table.open("t").await?.unwrap();
            let scan =
                checked_scan(stored.as_ref(), Some(&["x".into()]), &Dnf::default(), 1).await?;
            scan.stream.try_collect::<Vec<_>>().await
        })
        .unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].schema().field(0).name(), "x");
    }
}
