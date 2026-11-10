//! [`FeatureTable`]: features of a lake as a DataFusion table.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::ArrayRef;
use arrow::compute::SortOptions;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{
    Between, BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType,
};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::execution_plan::{
    Boundedness, ChildrenPropertiesMode, EmissionType, ReplaceChildrenOptions,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::channel::mpsc;
use futures::{FutureExt, SinkExt, Stream, StreamExt};
use tokio::task::JoinHandle;

use crate::error::{Error, Result};
use crate::lake::{Lake, SelectedFeature, Selection};
use crate::predicate::{Dnf, Op, Term};
use crate::runtime;

/// Most conjunctions a pushed-down predicate may expand to.
const MAX_CONJUNCTIONS: usize = 64;

/// Features of a lake as a DataFusion [`TableProvider`].
///
/// The table has one row per index row: the features in order, then the sort keys, all
/// nullable, as [`Lake::read`] returns them. Rows of feature tables that extend beyond the
/// index (left by an interrupted write) are not part of it; with
/// [`LakeOptions::validate`](crate::LakeOptions::validate) they fail the query instead.
///
/// The table reads the feature versions it was created with. Their contents are read when a
/// query runs, so appends, merge-mode overwrites and new index rows are visible to later
/// queries, while a copy-mode overwrite, which writes a new version, is not.
///
/// Filters on the sort keys are pushed into the read (comparisons with literals, `IN`,
/// `BETWEEN`, `IS [NOT] NULL`, and `AND`/`OR` of these); DataFusion applies every filter
/// again, so on an aligned lake the result is the same with or without pushdown. A write
/// that fails with [`Error::Write`] can leave a table misaligned before the end of the
/// index; then, as with [`Lake::read`], results depend on the filters, and validation only
/// checks the rows a query reads. Batches come in sort-key
/// order, ascending with nulls first, which DataFusion uses to skip sorts.
///
/// Queries run on the library's own runtime, so the table works inside any executor, but
/// creating it blocks and must happen outside async code. The
/// read decodes its inputs before the first batch, like [`Lake::read`]. Dropping a query's
/// stream stops the read, though sorting or joining work already started finishes first.
///
/// ```no_run
/// use std::sync::Arc;
///
/// use gamma_lake_core::datafusion::prelude::SessionContext;
/// use gamma_lake_core::store::{MemoryStore, TableStore};
/// use gamma_lake_core::Lake;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let store: Arc<dyn TableStore> = Arc::new(MemoryStore::new());
/// // Created outside async code, like every blocking call of the library.
/// let lake = Lake::open_with_store(store, Default::default())?;
/// let table = lake.feature_table(&["feature_a", "feature_b"], &[])?;
/// let ctx = SessionContext::new();
/// ctx.register_table("features", Arc::new(table))?;
/// let runtime = tokio::runtime::Runtime::new()?;
/// let batches = runtime.block_on(async {
///     let frame = ctx.sql("SELECT * FROM features WHERE symbol = 'AAPL'").await?;
///     frame.collect().await
/// })?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct FeatureTable {
    lake: Lake,
    features: Vec<SelectedFeature>,
    /// `features`, each with the signal type and parameters of the first feature at its
    /// address, so that reading a subset reads every table the same way.
    reads: Vec<SelectedFeature>,
    schema: SchemaRef,
}

impl Lake {
    /// The latest versions of `features` and `targets` as a DataFusion table, resolved as
    /// [`Lake::read`] resolves them.
    pub fn feature_table(&self, features: &[&str], targets: &[&str]) -> Result<FeatureTable> {
        let selected = self.resolve(Selection::Latest { features, targets })?;
        FeatureTable::from_selected(self.clone(), selected)
    }
}

impl FeatureTable {
    /// A table of the features `features` describes, read as [`Lake::read_selected`] reads
    /// them.
    pub fn from_selected(lake: Lake, features: Vec<SelectedFeature>) -> Result<Self> {
        let schema = lake.read_selected_schema(&features)?;
        let reads = features
            .iter()
            .map(|f| {
                let first = features
                    .iter()
                    .find(|g| g.table_addr == f.table_addr)
                    .expect("f is in features");
                SelectedFeature {
                    signal_type: first.signal_type.clone(),
                    feature_params: first.feature_params.clone(),
                    ..f.clone()
                }
            })
            .collect();
        Ok(Self {
            lake,
            features,
            reads,
            schema,
        })
    }

    /// The feature metadata rows the table reads.
    pub fn features(&self) -> &[SelectedFeature] {
        &self.features
    }

    /// The predicate `filter` pushes into the read, if it can be pushed.
    fn pushed(&self, filter: &Expr) -> Option<Vec<Vec<Term>>> {
        let dnf = convert(filter, self.lake.sort_keys())?;
        Dnf(dnf.clone())
            .to_expr(&self.lake.index_schema, "index")
            .ok()
            .map(|_| dnf)
    }
}

#[async_trait]
impl TableProvider for FeatureTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|f| match self.pushed(f) {
                Some(_) => TableProviderFilterPushDown::Inexact,
                None => TableProviderFilterPushDown::Unsupported,
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let mut dnf: Option<Vec<Vec<Term>>> = None;
        let mut pushed = vec![];
        for filter in filters {
            let Some(next) = self.pushed(filter) else {
                continue;
            };
            dnf = match dnf.take() {
                None => Some(next),
                Some(prev) => match and(&prev, &next) {
                    Some(both) => Some(both),
                    None => {
                        dnf = Some(prev);
                        continue;
                    }
                },
            };
            pushed.push(filter.clone());
        }

        let all: Vec<usize> = (0..self.schema.fields().len()).collect();
        let projection = projection.unwrap_or(&all);
        let width = self.features.len();
        let mut wanted: Vec<usize> = projection.iter().copied().filter(|&i| i < width).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let columns = projection
            .iter()
            .map(|&i| match wanted.binary_search(&i) {
                Ok(position) => position,
                Err(_) => wanted.len() + i - width,
            })
            .collect();
        let schema = Arc::new(self.schema.project(projection)?);
        Ok(Arc::new(FeatureTableExec::new(
            self.lake.clone(),
            wanted.iter().map(|&i| self.reads[i].clone()).collect(),
            Dnf(dnf.unwrap_or_default()),
            pushed,
            columns,
            schema,
        )))
    }
}

/// `filter` as a DNF over the sort keys, or `None` when it has a part that is not one.
fn convert(filter: &Expr, keys: &[String]) -> Option<Vec<Vec<Term>>> {
    let key = |expr: &Expr| match expr {
        Expr::Column(c) if keys.contains(&c.name) => Some(c.name.clone()),
        _ => None,
    };
    let literal = |expr: &Expr| match expr {
        Expr::Literal(value, _) if !value.is_null() => value.to_array().ok(),
        _ => None,
    };
    let term =
        |column: String, op: Op, values: ArrayRef| Some(vec![vec![Term { column, op, values }]]);
    match filter {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
            Operator::And => and(&convert(left, keys)?, &convert(right, keys)?),
            Operator::Or => {
                let (mut left, right) = (convert(left, keys)?, convert(right, keys)?);
                if left.len() + right.len() > MAX_CONJUNCTIONS {
                    return None;
                }
                left.extend(right);
                Some(left)
            }
            _ => {
                let (column, value, op) = match (key(left), literal(right)) {
                    (Some(column), Some(value)) => (column, value, *op),
                    _ => (key(right)?, literal(left)?, op.swap()?),
                };
                let op = match op {
                    Operator::Eq => Op::Eq,
                    Operator::NotEq => Op::NotEq,
                    Operator::Lt => Op::Lt,
                    Operator::LtEq => Op::LtEq,
                    Operator::Gt => Op::Gt,
                    Operator::GtEq => Op::GtEq,
                    _ => return None,
                };
                term(column, op, value)
            }
        },
        Expr::IsNull(expr) => term(key(expr)?, Op::IsNull, ScalarValue::Null.to_array().ok()?),
        Expr::IsNotNull(expr) => term(
            key(expr)?,
            Op::IsNotNull,
            ScalarValue::Null.to_array().ok()?,
        ),
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => {
            // `x NOT IN ()` is true for null `x`, which `Op::NotIn` does not match.
            if *negated && list.is_empty() {
                return None;
            }
            let column = key(expr)?;
            let values = list
                .iter()
                .map(|v| match v {
                    Expr::Literal(value, _) if !value.is_null() => Some(value.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let values = if values.is_empty() {
                ScalarValue::Null.to_array().ok()?.slice(0, 0)
            } else {
                ScalarValue::iter_to_array(values).ok()?
            };
            let op = if *negated { Op::NotIn } else { Op::In };
            term(column, op, values)
        }
        Expr::Between(Between {
            expr,
            negated: false,
            low,
            high,
        }) => {
            let column = key(expr)?;
            Some(vec![vec![
                Term {
                    column: column.clone(),
                    op: Op::GtEq,
                    values: literal(low)?,
                },
                Term {
                    column,
                    op: Op::LtEq,
                    values: literal(high)?,
                },
            ]])
        }
        _ => None,
    }
}

/// The DNF of `a AND b`, or `None` when it has too many conjunctions.
fn and(a: &[Vec<Term>], b: &[Vec<Term>]) -> Option<Vec<Vec<Term>>> {
    if a.len().checked_mul(b.len())? > MAX_CONJUNCTIONS {
        return None;
    }
    Some(
        a.iter()
            .flat_map(|x| b.iter().map(move |y| x.iter().chain(y).cloned().collect()))
            .collect(),
    )
}

/// One partition reading a [`FeatureTable`].
#[derive(Debug)]
struct FeatureTableExec {
    lake: Lake,
    features: Vec<SelectedFeature>,
    dnf: Dnf,
    /// The filters `dnf` was built from, for display.
    pushed: Vec<Expr>,
    /// Output columns, as positions in the read's schema.
    columns: Vec<usize>,
    properties: Arc<PlanProperties>,
}

impl FeatureTableExec {
    fn new(
        lake: Lake,
        features: Vec<SelectedFeature>,
        dnf: Dnf,
        pushed: Vec<Expr>,
        columns: Vec<usize>,
        schema: SchemaRef,
    ) -> Self {
        // Rows come from the sorted index: the longest prefix of the sort keys in the
        // output gives their order.
        let ascending = SortOptions {
            descending: false,
            nulls_first: true,
        };
        let ordering: Vec<PhysicalSortExpr> = lake
            .sort_keys()
            .iter()
            .map_while(|k| schema.index_of(k).ok().map(|i| (k, i)))
            .map(|(k, i)| PhysicalSortExpr::new(Arc::new(Column::new(k, i)), ascending))
            .collect();
        let equivalence = if ordering.is_empty() {
            EquivalenceProperties::new(schema)
        } else {
            EquivalenceProperties::new_with_orderings(schema, [ordering])
        };
        let properties = PlanProperties::new(
            equivalence,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            lake,
            features,
            dnf,
            pushed,
            columns,
            properties: Arc::new(properties),
        }
    }
}

impl DisplayAs for FeatureTableExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let names: Vec<&str> = self.features.iter().map(|f| f.name.as_str()).collect();
        let columns: Vec<&str> = self
            .properties
            .eq_properties
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        write!(
            f,
            "FeatureTableExec: features=[{}], projection=[{}]",
            names.join(", "),
            columns.join(", ")
        )?;
        if !self.pushed.is_empty() {
            let pushed: Vec<String> = self.pushed.iter().map(ToString::to_string).collect();
            write!(f, ", pushed=[{}]", pushed.join(", "))?;
        }
        Ok(())
    }
}

impl ExecutionPlan for FeatureTableExec {
    fn name(&self) -> &str {
        "FeatureTableExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DfResult<TreeNodeRecursion>,
    ) -> DfResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "FeatureTableExec has one partition, not {}",
                partition + 1
            )));
        }
        let schema = self.schema();
        let (lake, features, dnf) = (self.lake.clone(), self.features.clone(), self.dnf.clone());
        let (columns, out) = (self.columns.clone(), schema.clone());
        let (mut sender, receiver) = mpsc::channel(2);
        let task = runtime::handle().spawn(async move {
            let stream = lake.read_selected_async(&features, &dnf, true).await;
            let mut stream = match stream {
                Ok(stream) => stream,
                Err(e) => {
                    let _ = sender.send(Err(external(e))).await;
                    return;
                }
            };
            while let Some(batch) = stream.next().await {
                let batch = batch
                    .map_err(external)
                    .and_then(|b| project(&b, &columns, &out));
                let failed = batch.is_err();
                if sender.send(batch).await.is_err() || failed {
                    return;
                }
            }
        });
        let stream = Forwarded {
            receiver,
            task: Some(task),
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

fn external(error: Error) -> DataFusionError {
    match error {
        Error::DataFusion(e) => e,
        other => DataFusionError::External(Box::new(other)),
    }
}

fn project(batch: &RecordBatch, columns: &[usize], schema: &SchemaRef) -> DfResult<RecordBatch> {
    Ok(RecordBatch::try_new_with_options(
        schema.clone(),
        columns.iter().map(|&i| batch.column(i).clone()).collect(),
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

/// The batches a read task sends. Dropping it aborts the task; a task that panicked or was
/// cancelled ends the stream with an error.
struct Forwarded {
    receiver: mpsc::Receiver<DfResult<RecordBatch>>,
    task: Option<JoinHandle<()>>,
}

impl Stream for Forwarded {
    type Item = DfResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_next_unpin(cx) {
            Poll::Ready(None) => {}
            other => return other,
        }
        let Some(task) = self.task.as_mut() else {
            return Poll::Ready(None);
        };
        let result = std::task::ready!(task.poll_unpin(cx));
        self.task = None;
        Poll::Ready(result.err().map(|e| {
            Err(DataFusionError::Execution(format!(
                "the gamma-lake read stopped: {e}"
            )))
        }))
    }
}

impl Drop for Forwarded {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Schema;
    use datafusion::prelude::{col, lit};

    fn keys() -> Vec<String> {
        vec!["ts".into(), "sym".into()]
    }

    fn shape(filter: Expr) -> Option<Vec<Vec<(String, Op, usize)>>> {
        convert(&filter, &keys()).map(|dnf| {
            dnf.into_iter()
                .map(|c| {
                    c.into_iter()
                        .map(|t| (t.column, t.op, t.values.len()))
                        .collect()
                })
                .collect()
        })
    }

    fn t(column: &str, op: Op, n: usize) -> (String, Op, usize) {
        (column.into(), op, n)
    }

    #[test]
    fn comparisons_flip_when_the_literal_is_on_the_left() {
        assert_eq!(
            shape(lit(5i64).lt(col("ts"))),
            Some(vec![vec![t("ts", Op::Gt, 1)]])
        );
        assert_eq!(
            shape(col("ts").not_eq(lit(5i64))),
            Some(vec![vec![t("ts", Op::NotEq, 1)]])
        );
    }

    #[test]
    fn and_distributes_over_or() {
        let filter = col("ts")
            .eq(lit(1i64))
            .or(col("ts").eq(lit(2i64)))
            .and(col("sym").between(lit("a"), lit("c")));
        assert_eq!(
            shape(filter),
            Some(vec![
                vec![
                    t("ts", Op::Eq, 1),
                    t("sym", Op::GtEq, 1),
                    t("sym", Op::LtEq, 1)
                ],
                vec![
                    t("ts", Op::Eq, 1),
                    t("sym", Op::GtEq, 1),
                    t("sym", Op::LtEq, 1)
                ],
            ])
        );
    }

    #[test]
    fn membership_and_null_tests() {
        assert_eq!(
            shape(col("ts").in_list(vec![lit(1i64), lit(2i64)], false)),
            Some(vec![vec![t("ts", Op::In, 2)]])
        );
        assert_eq!(
            shape(col("ts").in_list(vec![], false)),
            Some(vec![vec![t("ts", Op::In, 0)]])
        );
        assert_eq!(
            shape(col("ts").in_list(vec![lit(1i64)], true)),
            Some(vec![vec![t("ts", Op::NotIn, 1)]])
        );
        assert_eq!(shape(col("ts").in_list(vec![], true)), None);
        assert_eq!(
            shape(col("ts").in_list(vec![lit(1i64), lit(ScalarValue::Int64(None))], false)),
            None
        );
        assert_eq!(
            shape(col("sym").is_null()),
            Some(vec![vec![t("sym", Op::IsNull, 1)]])
        );
    }

    #[test]
    fn other_filters_are_not_pushed() {
        let unsupported = [
            col("f").eq(lit(1.0)),
            col("ts").eq(col("ts")),
            col("ts").eq(lit(ScalarValue::Int64(None))),
            !col("ts").eq(lit(1i64)),
            col("ts").not_between(lit(1i64), lit(2i64)),
            col("ts").eq(lit(1i64)).or(col("f").eq(lit(1.0))),
            Expr::Cast(datafusion::logical_expr::Cast::new(
                Box::new(col("ts")),
                arrow::datatypes::DataType::Utf8,
            ))
            .eq(lit("1")),
        ];
        for filter in unsupported {
            assert_eq!(shape(filter.clone()), None, "{filter}");
        }
        // An AND with an unsupported side is not pushed; DataFusion splits top-level ANDs
        // into separate filters before asking.
        assert_eq!(
            shape(col("ts").eq(lit(1i64)).and(col("f").gt(lit(0.0)))),
            None
        );
    }

    #[test]
    fn dropping_the_stream_aborts_the_read() {
        use futures::channel::oneshot;
        use std::time::Duration;

        // Before the first batch, and while the read waits for room in the channel.
        for batches in [0, 3] {
            let (mut sender, receiver) = mpsc::channel(2);
            let (alive, aborted) = oneshot::channel::<()>();
            let (sent_tx, sent_rx) = std::sync::mpsc::channel();
            let task = runtime::handle().spawn(async move {
                let _alive = alive;
                for sent in 0..batches {
                    sent_tx.send(sent).unwrap();
                    let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
                    sender.send(Ok(batch)).await.unwrap();
                }
                sent_tx.send(batches).unwrap();
                futures::future::pending::<()>().await;
            });
            let stream = Forwarded {
                receiver,
                task: Some(task),
            };
            // The channel holds 2 batches plus 1 per sender, so the third send waits.
            let reached = batches.min(2);
            while sent_rx.recv_timeout(Duration::from_secs(10)).unwrap() < reached {}
            drop(stream);
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                done_tx
                    .send(futures::executor::block_on(aborted).is_err())
                    .unwrap();
            });
            assert!(done_rx.recv_timeout(Duration::from_secs(10)).unwrap());
        }
    }

    #[test]
    fn a_panicking_read_ends_the_stream_with_an_error() {
        for batches in [0, 1] {
            let (mut sender, receiver) = mpsc::channel(2);
            let task = runtime::handle().spawn(async move {
                for _ in 0..batches {
                    let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
                    sender.send(Ok(batch)).await.unwrap();
                }
                panic!("boom");
            });
            let stream = Forwarded {
                receiver,
                task: Some(task),
            };
            let items = runtime::block_on(stream.collect::<Vec<_>>()).unwrap();
            assert_eq!(items.len(), batches + 1);
            let message = items.last().unwrap().as_ref().unwrap_err().to_string();
            assert!(message.contains("the gamma-lake read stopped"), "{message}");
            assert!(items[..batches].iter().all(Result::is_ok));
        }
    }

    #[test]
    fn expansion_beyond_the_budget_is_not_pushed() {
        let pair = |k: i64| col("ts").eq(lit(k)).or(col("ts").eq(lit(k + 100)));
        // (a|b) & (c|d) & ... six times is 64 conjunctions; seven is 128.
        let product = |n: i64| (1..n).fold(pair(0), |acc, k| acc.and(pair(k)));
        assert_eq!(shape(product(6)).map(|d| d.len()), Some(64));
        assert_eq!(shape(product(7)), None);
        // Nested inside an OR, the inner expansion is still checked.
        assert_eq!(shape(product(7).or(col("ts").eq(lit(1i64)))), None);
        let wide = (0..65).fold(col("ts").eq(lit(-1i64)), |acc, k| {
            acc.or(col("ts").eq(lit(k)))
        });
        assert_eq!(shape(wide), None);
    }
}
