//! Reading a gamma-lake: metadata tables, the index and feature columns.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, UInt32Array};
use arrow::compute::{cast, concat_batches, take_record_batch};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion::common::Column;
use datafusion::dataframe::DataFrame;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{cast as cast_expr, Expr};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::prelude::SessionContext;
use futures::{StreamExt, TryStreamExt};

use crate::align::{zip_aligned, AlignInput, Source};
use crate::error::{integrity, Error, Result};
use crate::joined::{table_predicate, AsOfParams, AsOfTable, SparseTable};
use crate::lock::{LockingStore, TableLock};
use crate::predicate::{column, Dnf};
use crate::runtime::{block_on, blocking};
use crate::scan::{checked_scan, frame as scan_frame};
use crate::sort::sorted_stream;
use crate::store::{StoredTable, TableStore};
use crate::stream::{AsyncBatchStream, BatchStream};
use crate::transaction;
use crate::write::SignalType;

const INDEX: &str = "idx";

/// Options for [`Lake::open_with_store`]. The defaults match `GammaFeatureLake`.
#[derive(Clone)]
pub struct LakeOptions {
    pub primary_sort_key: String,
    pub index_path: String,
    pub feature_metadata_path: String,
    pub table_metadata_path: String,
    /// The transaction log, created by the first write when missing; see [`Lake::add`].
    pub transaction_log_path: String,
    /// Upper bound on rows per output batch.
    pub batch_size: usize,
    /// While reading, check that every feature table's sort keys match the index row by
    /// row and that index keys are unique. Reads the sort keys of every table as well.
    pub validate: bool,
    /// Prefix of the addresses of feature tables created by writes.
    pub table_name_prefix: String,
    /// Held around every commit to a table; see [`TableLock`].
    pub lock: Option<Arc<dyn TableLock>>,
    /// Bytes a read may hold while sorting its scans; `None` sorts in memory. See
    /// [`Lake::read`].
    pub memory_limit: Option<usize>,
}

impl std::fmt::Debug for LakeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LakeOptions")
            .field("primary_sort_key", &self.primary_sort_key)
            .field("index_path", &self.index_path)
            .field("feature_metadata_path", &self.feature_metadata_path)
            .field("table_metadata_path", &self.table_metadata_path)
            .field("transaction_log_path", &self.transaction_log_path)
            .field("batch_size", &self.batch_size)
            .field("validate", &self.validate)
            .field("table_name_prefix", &self.table_name_prefix)
            .field("lock", &self.lock.as_ref().map(|_| "TableLock"))
            .field("memory_limit", &self.memory_limit)
            .finish()
    }
}

impl Default for LakeOptions {
    fn default() -> Self {
        Self {
            primary_sort_key: "timestamp".into(),
            index_path: "index".into(),
            feature_metadata_path: "feature_metadata".into(),
            table_metadata_path: "table_metadata".into(),
            transaction_log_path: "transaction_log".into(),
            batch_size: 65536,
            validate: false,
            table_name_prefix: String::new(),
            lock: None,
            memory_limit: None,
        }
    }
}

/// `store`, holding `options.lock` around its commits when there is one.
pub(crate) fn locking(store: Arc<dyn TableStore>, options: &LakeOptions) -> Arc<dyn TableStore> {
    match &options.lock {
        Some(lock) => Arc::new(LockingStore::new(store, lock.clone())),
        None => store,
    }
}

/// A gamma-lake: an index, two metadata tables and feature tables in a [`TableStore`].
///
/// Every call reads the latest version of the tables involved.
#[derive(Clone, Debug)]
pub struct Lake {
    pub(crate) store: Arc<dyn TableStore>,
    pub(crate) options: LakeOptions,
    pub(crate) sort_keys: Vec<String>,
    pub(crate) index_schema: SchemaRef,
}

/// One scan of a feature table.
struct TableInput {
    addr: String,
    /// Unsorted; the sort keys followed by `names`.
    frame: DataFrame,
    /// Number of key columns kept in the sorted stream: all of them when validating, else 0.
    keys: usize,
    names: Vec<String>,
    /// How the table is matched to the index when it is not read by position.
    join: Option<Join>,
}

/// A feature to read, given as the feature metadata row that describes it.
///
/// Features at one `table_addr` are read together, with the signal type and parameters of
/// the first of them, as `GammaFeatureLake.read(feature_metadata_frame)` does.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedFeature {
    pub name: String,
    pub table_addr: String,
    pub signal_type: String,
    /// Required for `as_of_feature`, where it holds the `join_asof` arguments as JSON.
    pub feature_params: Option<String>,
}

/// Which features a read returns.
#[derive(Clone, Copy)]
pub(crate) enum Selection<'a> {
    /// The latest version of each name.
    Latest {
        features: &'a [&'a str],
        targets: &'a [&'a str],
    },
    /// Each name at the given version.
    Versions(&'a [(&'a str, i64)]),
    Rows(&'a [SelectedFeature]),
}

/// How a table that is not read by position is matched to the index.
enum Join {
    AsOf(AsOfParams),
    Sparse,
}

impl Join {
    /// How `feature`'s table is read: `None` for by position.
    fn of(feature: &SelectedFeature) -> Result<Option<Self>> {
        Ok(match SignalType::parse(&feature.signal_type) {
            Some(SignalType::Feature | SignalType::Target) => None,
            Some(SignalType::AsOf) => match &feature.feature_params {
                Some(params) => Some(Join::AsOf(AsOfParams::parse(params)?)),
                None => {
                    return Err(Error::InvalidArgument(format!(
                        "as-of feature {:?} has no feature_params",
                        feature.name
                    )))
                }
            },
            Some(SignalType::Sparse) => Some(Join::Sparse),
            None => {
                return Err(Error::Unsupported {
                    feature: feature.name.clone(),
                    signal_type: feature.signal_type.clone(),
                })
            }
        })
    }
}

/// A read input ready to align: a sorted stream, or a table to look up index rows in.
enum Prepared {
    Aligned(AlignInput),
    Joined(Box<JoinedTable>),
}

/// A table prepared for looking up index rows: an as-of or sparse table.
enum Lookup {
    AsOf(AsOfTable),
    Sparse(SparseTable),
}

/// An as-of or sparse table ready to look up the rows each index row matches.
struct JoinedTable {
    lookup: Lookup,
    /// The feature columns of the table.
    values: RecordBatch,
}

impl JoinedTable {
    /// Prepares `table`, the sort keys followed by the feature columns. `pk` is the position
    /// of the primary sort key among the `keys` sort keys.
    fn new(join: &Join, addr: &str, table: &RecordBatch, keys: usize, pk: usize) -> Result<Self> {
        let values = table.project(&(keys..table.num_columns()).collect::<Vec<_>>())?;
        let lookup = match join {
            Join::AsOf(params) => {
                let (table_pk, table_by) = split_keys(table, keys, pk);
                Lookup::AsOf(AsOfTable::new(params, &table_by, table_pk.as_ref())?)
            }
            Join::Sparse => Lookup::Sparse(SparseTable::new(&table.columns()[..keys], addr)?),
        };
        Ok(Self { lookup, values })
    }

    /// The feature columns of the rows that each row of `index` matches, null where none
    /// does. `index` starts with the sort keys.
    fn columns(&self, index: &RecordBatch, keys: usize, pk: usize) -> Result<Vec<ArrayRef>> {
        let rows = match &self.lookup {
            Lookup::AsOf(lookup) => {
                let (pk, by) = split_keys(index, keys, pk);
                lookup.lookup(&by, &pk)?
            }
            Lookup::Sparse(lookup) => lookup.lookup(&index.columns()[..keys])?,
        };
        Ok(take_record_batch(&self.values, &rows)?.columns().to_vec())
    }
}

/// The primary sort key column of `batch` and its other sort key columns.
fn split_keys(batch: &RecordBatch, keys: usize, pk: usize) -> (ArrayRef, Vec<ArrayRef>) {
    let by = (0..keys)
        .filter(|&k| k != pk)
        .map(|k| batch.column(k).clone())
        .collect();
    (batch.column(pk).clone(), by)
}

/// `index` with the feature columns of `tables` appended to every batch, in order. The
/// lookups run on the blocking pool, several batches at a time, keeping batch order.
fn with_joined(
    index: SendableRecordBatchStream,
    tables: Vec<JoinedTable>,
    keys: usize,
    pk: usize,
) -> Result<SendableRecordBatchStream> {
    let mut fields = index.schema().fields().to_vec();
    for table in &tables {
        fields.extend(table.values.schema().fields().iter().cloned());
    }
    let schema = SchemaRef::new(Schema::new(fields));
    let tables = Arc::new(tables);
    let out = schema.clone();
    let stream = index
        .map(move |batch| {
            let (tables, schema) = (tables.clone(), out.clone());
            async move {
                let batch = batch?;
                blocking(move || {
                    let mut columns = batch.columns().to_vec();
                    for table in tables.iter() {
                        columns.extend(table.columns(&batch, keys, pk)?);
                    }
                    let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                    Ok(RecordBatch::try_new_with_options(
                        schema, columns, &options,
                    )?)
                })
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))
            }
        })
        .buffered(crate::runtime::threads());
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
}

/// The inputs of a read, sorted by [`sorted_stream`] and zipped by [`zip_aligned`].
struct ReadInputs {
    /// Resolved feature names in output order.
    names: Vec<String>,
    /// Unsorted index keys.
    index: DataFrame,
    tables: Vec<TableInput>,
}

impl ReadInputs {
    /// Output columns as `(input, column)`; input 0 is the index.
    fn layout(&self) -> Vec<Source> {
        let mut sources: HashMap<&str, Source> = HashMap::new();
        for (i, table) in self.tables.iter().enumerate() {
            for (position, name) in table.names.iter().enumerate() {
                sources
                    .entry(name.as_str())
                    .or_insert((i + 1, table.keys + position));
            }
        }
        let features = self.names.iter().map(|name| sources[name.as_str()]);
        let keys = (0..self.index.schema().fields().len()).map(|k| (0, k));
        features.chain(keys).collect()
    }

    fn schema(&self) -> SchemaRef {
        let keys = self.index.schema().fields().len();
        let field = |input: usize, column: usize| {
            let field = if input == 0 {
                self.index.schema().field(column)
            } else {
                let table = &self.tables[input - 1];
                table.frame.schema().field(keys + column - table.keys)
            };
            Field::new(field.name(), field.data_type().clone(), true)
        };
        Arc::new(Schema::new(
            self.layout()
                .into_iter()
                .map(|(i, c)| field(i, c))
                .collect::<Vec<_>>(),
        ))
    }
}

pub(crate) struct FeatureRow {
    /// Row number in the feature metadata batch.
    pub(crate) row: usize,
    pub(crate) name: String,
    pub(crate) version: Option<i64>,
    pub(crate) table_addr: Option<String>,
    pub(crate) signal_type: Option<String>,
    pub(crate) feature_params: Option<String>,
}

impl Lake {
    /// Opens the lake whose tables are in `store`.
    pub fn open_with_store(store: Arc<dyn TableStore>, options: LakeOptions) -> Result<Self> {
        Self::open_locked(locking(store, &options), options)
    }

    /// Opens the lake in `store`, which already holds `options.lock`.
    pub(crate) fn open_locked(store: Arc<dyn TableStore>, options: LakeOptions) -> Result<Self> {
        if options.batch_size == 0 {
            return Err(Error::InvalidArgument("batch_size must be positive".into()));
        }
        let tables = block_on(async {
            futures::try_join!(
                store.open(&options.index_path),
                store.open(&options.feature_metadata_path),
                store.open(&options.table_metadata_path),
            )
        })??;
        let (Some(index), Some(_), Some(_)) = tables else {
            return Err(Error::NotInitialized(store.location().to_string()));
        };
        let index_schema = index.schema();
        let sort_keys: Vec<String> = index_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        if !sort_keys.contains(&options.primary_sort_key) {
            return Err(Error::InvalidArgument(format!(
                "primary sort key {:?} is not an index column {sort_keys:?}",
                options.primary_sort_key
            )));
        }
        Ok(Self {
            store,
            options,
            sort_keys,
            index_schema,
        })
    }

    /// Where the lake's tables are.
    pub fn location(&self) -> &str {
        self.store.location()
    }

    /// Names of the index columns, in index order.
    pub fn sort_keys(&self) -> &[String] {
        &self.sort_keys
    }

    pub fn primary_sort_key(&self) -> &str {
        &self.options.primary_sort_key
    }

    /// Schema of the index, as returned by [`Lake::index`].
    pub fn index_schema(&self) -> SchemaRef {
        self.index_schema.clone()
    }

    /// Rows of the feature metadata table matching `dnf`.
    pub fn feature_metadata(&self, dnf: &Dnf) -> Result<BatchStream> {
        self.scan(&self.options.feature_metadata_path, dnf)
    }

    /// Rows of the table metadata table matching `dnf`.
    pub fn table_metadata(&self, dnf: &Dnf) -> Result<BatchStream> {
        self.scan(&self.options.table_metadata_path, dnf)
    }

    /// Rows of the transaction log matching `dnf`: none when the lake has no log yet.
    pub fn transaction_log(&self, dnf: &Dnf) -> Result<BatchStream> {
        block_on(async {
            match self.store.open(&self.options.transaction_log_path).await? {
                Some(table) => self.scan_table(table, dnf).await,
                None => Ok(BatchStream::from_batches(transaction::schema(), vec![])),
            }
        })?
    }

    /// Rows of the index matching `dnf`, unordered.
    pub fn index(&self, dnf: &Dnf) -> Result<BatchStream> {
        self.scan(&self.options.index_path, dnf)
    }

    /// The feature metadata row of the highest version of each feature (null versions last).
    pub fn latest_features(&self) -> Result<BatchStream> {
        block_on(async {
            let ctx = self.context()?;
            let table = Arc::<dyn StoredTable>::from(
                self.table(&self.options.feature_metadata_path).await?,
            );
            let df = scan_frame(&ctx, table, &Dnf::default())?;
            let df = normalize(df)?;
            let schema = df.schema().inner().clone();
            let batch = concat_batches(&schema, &df.collect().await?)?;
            let rows = feature_rows(&batch)?;
            let mut chosen: Vec<u32> = latest(&rows)
                .into_values()
                .map(|i| rows[i].row as u32)
                .collect();
            chosen.sort_unstable();
            let batch = take_record_batch(&batch, &UInt32Array::from(chosen))?;
            let size = self.options.batch_size;
            let batches = (0..batch.num_rows())
                .step_by(size)
                .map(|offset| batch.slice(offset, size.min(batch.num_rows() - offset)))
                .collect();
            Ok(BatchStream::from_batches(schema, batches))
        })?
    }

    /// Reads features and targets aligned to the index.
    ///
    /// Each requested name resolves to its latest version; names without metadata are
    /// skipped. Every target must be registered as a `target` signal. Columns are the
    /// resolved names in request order followed by the sort keys. `dnf` may only reference
    /// sort keys; it filters the index and every feature table before they are aligned, as
    /// `GammaFeatureLake.read(start=, end=)` does. On row-aligned lakes that equals filtering
    /// the aligned result.
    ///
    /// `feature` and `target` tables are matched to the index by row position after sorting
    /// each by the sort keys (nulls first), as `GammaFeatureLake.read` does; no join is
    /// performed. Tables shorter than the index read as null at the end. Writers must keep
    /// these tables row-aligned with the index; with [`LakeOptions::validate`] the keys of
    /// each table are compared with the index while streaming and a mismatch fails the
    /// stream with [`Error::Integrity`].
    ///
    /// `as_of_feature` tables are matched with Polars' `join_asof` on the primary sort key,
    /// within equal values of the other sort keys, using the `strategy`, `tolerance` and
    /// `allow_exact_matches` stored in `feature_params`. `sparse_feature` tables are
    /// left-joined to the index on all sort keys. Features stored in one table are read the
    /// way the first of them in the request is.
    ///
    /// By default the requested columns are decoded into memory before the first batch is
    /// returned, and a table whose rows are not stored in key order is copied once to sort
    /// it, which can double its share of peak memory. With [`LakeOptions::memory_limit`], the
    /// index and each scan of a feature table share the limit equally and are sorted one at
    /// a time by an external sort that spills to the system temporary directory beyond its
    /// share. Scans then decode batches of fewer than `batch_size` rows when a batch would
    /// take more than an eighth of their share, estimating 32 bytes for each value of a
    /// variable-width column. The first batch still comes after every scan is sorted. A
    /// share too small to sort one batch fails the read with "Resources exhausted"; for
    /// long strings or nested values, a smaller `batch_size` avoids this. As-of and sparse
    /// tables are read into memory in both modes. The external sort does not keep the order of rows with
    /// equal sort keys, so tables that repeat sort keys may not align.
    pub fn read(&self, features: &[&str], targets: &[&str], dnf: &Dnf) -> Result<BatchStream> {
        self.read_selection(Selection::Latest { features, targets }, dnf)
    }

    /// Asynchronously reads the latest versions of `features` and `targets` as Arrow batches.
    pub async fn read_async(
        &self,
        features: &[&str],
        targets: &[&str],
        dnf: &Dnf,
    ) -> Result<AsyncBatchStream> {
        let stream = self
            .stream_selection(Selection::Latest { features, targets }, dnf, false)
            .await?;
        Ok(AsyncBatchStream::from_plan(stream))
    }

    /// Schema of [`Lake::read`], resolved without reading feature data.
    pub fn read_schema(&self, features: &[&str], targets: &[&str]) -> Result<SchemaRef> {
        self.selection_schema(Selection::Latest { features, targets })
    }

    /// Reads each feature at the given version, as [`Lake::read`] reads the latest ones.
    ///
    /// Every `(name, version)` must have a feature metadata row, or the read fails with
    /// [`Error::MissingFeatures`]; a name listed twice is [`Error::InvalidArgument`].
    pub fn read_versions(&self, features: &[(&str, i64)], dnf: &Dnf) -> Result<BatchStream> {
        self.read_selection(Selection::Versions(features), dnf)
    }

    /// Schema of [`Lake::read_versions`], resolved without reading feature data.
    pub fn read_versions_schema(&self, features: &[(&str, i64)]) -> Result<SchemaRef> {
        self.selection_schema(Selection::Versions(features))
    }

    /// Reads the features that `features` describe, as [`Lake::read`] reads the latest
    /// ones. The rows are not checked against the feature metadata, so they can come from
    /// an edited [`Lake::feature_metadata`] frame. A name listed twice is
    /// [`Error::InvalidArgument`]; no rows reads the sort keys of the index only.
    pub fn read_selected(&self, features: &[SelectedFeature], dnf: &Dnf) -> Result<BatchStream> {
        self.read_selection(Selection::Rows(features), dnf)
    }

    pub(crate) async fn read_selected_async(
        &self,
        features: &[SelectedFeature],
        dnf: &Dnf,
        bounded: bool,
    ) -> Result<AsyncBatchStream> {
        let stream = self
            .stream_selection(Selection::Rows(features), dnf, bounded)
            .await?;
        Ok(AsyncBatchStream::from_plan(stream))
    }

    /// Schema of [`Lake::read_selected`], resolved without reading feature data.
    pub fn read_selected_schema(&self, features: &[SelectedFeature]) -> Result<SchemaRef> {
        self.selection_schema(Selection::Rows(features))
    }

    /// The feature metadata rows `selection` reads, in output order.
    pub(crate) fn resolve(&self, selection: Selection<'_>) -> Result<Vec<SelectedFeature>> {
        block_on(async { self.select(&self.context()?, selection).await })?
    }

    fn selection_schema(&self, selection: Selection<'_>) -> Result<SchemaRef> {
        block_on(async {
            let plan = self.read_inputs(selection, &Dnf::default()).await?;
            Ok(plan.schema())
        })?
    }

    fn read_selection(&self, selection: Selection<'_>, dnf: &Dnf) -> Result<BatchStream> {
        block_on(async {
            let stream = self.stream_selection(selection, dnf, false).await?;
            Ok(BatchStream::from_plan(stream))
        })?
    }

    /// The aligned read of `selection`. When `bounded`, rows of feature tables beyond the
    /// index are not part of the output (see [`zip_aligned`]).
    pub(crate) async fn stream_selection(
        &self,
        selection: Selection<'_>,
        dnf: &Dnf,
        bounded: bool,
    ) -> Result<SendableRecordBatchStream> {
        let plan = self.read_inputs(selection, dnf).await?;
        let mut layout = plan.layout();
        let keys = self.sort_keys.len();
        let pk = self
            .sort_keys
            .iter()
            .position(|k| k == self.primary_sort_key())
            .expect("the primary sort key is a sort key");
        // The memory limit is shared equally by the scans that are sorted.
        let budget = self.options.memory_limit.map(|limit| {
            let scans = 1 + plan.tables.iter().filter(|t| t.join.is_none()).count();
            (limit / scans).max(1)
        });
        let sort = move |frame, keep_keys| sorted_stream(frame, keys, keep_keys, budget);
        let index = sort(plan.index, true);
        let prepare = |mut input: TableInput| async move {
            if let Some(join) = input.join.take() {
                let schema = input.frame.schema().inner().clone();
                let table = concat_batches(&schema, &input.frame.collect().await?)?;
                let addr = input.addr;
                let table =
                    blocking(move || JoinedTable::new(&join, &addr, &table, keys, pk)).await?;
                return Ok::<_, Error>(Prepared::Joined(Box::new(table)));
            }
            Ok(Prepared::Aligned(AlignInput {
                stream: sort(input.frame, input.keys > 0).await?,
                name: input.addr,
                keys: input.keys,
            }))
        };
        // Sort one physical table at a time to bound Parquet buffers and intermediate data.
        let index = index.await?;
        let tables = futures::stream::iter(plan.tables)
            .then(prepare)
            .try_collect::<Vec<_>>()
            .await?;
        // The feature columns of joined tables are appended to the index, input 0. `moved`
        // maps each input of the layout to its new input and column offset.
        let (mut aligned, mut joined) = (vec![], vec![]);
        let mut moved = vec![(0, 0)];
        let mut width = keys;
        for table in tables {
            match table {
                Prepared::Aligned(input) => {
                    aligned.push(input);
                    moved.push((aligned.len(), 0));
                }
                Prepared::Joined(table) => {
                    moved.push((0, width));
                    width += table.values.num_columns();
                    joined.push(*table);
                }
            }
        }
        for source in &mut layout {
            let (input, offset) = moved[source.0];
            *source = (input, offset + source.1);
        }
        let index = if joined.is_empty() {
            index
        } else {
            with_joined(index, joined, keys, pk)?
        };
        Ok(zip_aligned(
            index,
            keys,
            aligned,
            layout,
            self.options.batch_size,
            self.options.validate,
            bounded,
        )?)
    }

    /// Schema of [`Lake::feature_metadata`].
    pub fn feature_metadata_schema(&self) -> Result<SchemaRef> {
        self.table_schema(&self.options.feature_metadata_path)
    }

    /// Schema of [`Lake::table_metadata`].
    pub fn table_metadata_schema(&self) -> Result<SchemaRef> {
        self.table_schema(&self.options.table_metadata_path)
    }

    /// Schema of [`Lake::transaction_log`].
    pub fn transaction_log_schema(&self) -> Result<SchemaRef> {
        block_on(async {
            let table = self.store.open(&self.options.transaction_log_path).await?;
            Ok::<_, Error>(table.map_or_else(transaction::schema, |t| t.schema()))
        })?
    }

    fn table_schema(&self, name: &str) -> Result<SchemaRef> {
        block_on(async { Ok::<_, Error>(self.table(name).await?.schema()) })?
    }

    /// Errors unless as-of and sparse tables can be matched on the other sort keys. Floats
    /// are not: Polars treats `-0.0` and `0.0` as one key, and these lookups would not.
    fn check_join_keys(&self) -> Result<()> {
        for key in self
            .sort_keys
            .iter()
            .filter(|k| *k != self.primary_sort_key())
        {
            let data_type = self.index_schema.field_with_name(key)?.data_type();
            if data_type.is_floating() {
                return Err(Error::UnsupportedInput(format!(
                    "as-of and sparse features cannot be matched on floating-point sort key {key:?}"
                )));
            }
        }
        Ok(())
    }

    async fn read_inputs(&self, selection: Selection<'_>, dnf: &Dnf) -> Result<ReadInputs> {
        if let Some(c) = dnf
            .columns()
            .find(|c| !self.sort_keys.iter().any(|k| k == c))
        {
            return Err(Error::InvalidPredicate(format!(
                "column {c:?} is not a sort key; read() predicates may only reference {:?}",
                self.sort_keys
            )));
        }
        let ctx = self.context()?;
        let (selected, index_table) = futures::try_join!(
            self.select(&ctx, selection),
            self.table(&self.options.index_path)
        )?;

        // Features at one address are read together, the way the first of them is read.
        let pk_type = self
            .index_schema
            .field_with_name(self.primary_sort_key())?
            .data_type()
            .clone();
        let mut groups: Vec<(&str, Option<Join>, Vec<&str>)> = vec![];
        let mut group_indices: HashMap<&str, usize> = HashMap::new();
        for feature in &selected {
            let addr = feature.table_addr.as_str();
            if let Some(&index) = group_indices.get(addr) {
                groups[index].2.push(&feature.name);
                continue;
            }
            let join = Join::of(feature)?;
            if let Some(Join::AsOf(params)) = &join {
                params.check(&pk_type)?;
            }
            if join.is_some() {
                self.check_join_keys()?;
            }
            group_indices.insert(addr, groups.len());
            groups.push((addr, join, vec![&feature.name]));
        }

        let keys = self
            .sort_keys
            .iter()
            .map(|k| column(INDEX, k))
            .collect::<Vec<_>>();
        let index = scan_frame(&ctx, Arc::from(index_table), dnf)?;
        let index = normalize(index.alias(INDEX)?.select(keys)?)?;

        let opens: Vec<_> = groups.iter().map(|(addr, _, _)| self.table(addr)).collect();
        let opened: Vec<Box<dyn StoredTable>> = futures::stream::iter(opens)
            .buffered(16)
            .try_collect()
            .await?;
        let mut tables = vec![];
        let key_names = self
            .sort_keys
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        for ((addr, join, names), table) in groups.into_iter().zip(opened) {
            let relation = "t";
            let table = Arc::<dyn StoredTable>::from(table);
            if let Some(missing) = names
                .iter()
                .chain(key_names.iter())
                .find(|n| table.schema().field_with_name(n).is_err())
            {
                return Err(integrity(addr, format!("missing column {missing:?}")));
            }
            let scan = |names: &[&str], dnf: &Dnf| {
                let projection = key_names
                    .iter()
                    .chain(names.iter())
                    .map(|n| column(relation, n).alias(*n))
                    .collect::<Vec<_>>();
                let frame = scan_frame(&ctx, table.clone(), dnf)?.alias(relation)?;
                normalize(frame.select(projection)?)
            };
            let frame = match &join {
                Some(Join::AsOf(params)) => scan(
                    &names,
                    &table_predicate(dnf, self.primary_sort_key(), params.strategy),
                )?,
                _ => scan(&names, dnf)?,
            };
            // Joined tables carry no keys; without validation, aligned tables only need them
            // for sorting.
            let keys = if join.is_none() && self.options.validate {
                self.sort_keys.len()
            } else {
                0
            };
            tables.push(TableInput {
                addr: addr.to_string(),
                frame,
                keys,
                names: names.iter().map(|n| n.to_string()).collect(),
                join,
            });
        }
        Ok(ReadInputs {
            names: selected.into_iter().map(|f| f.name).collect(),
            index,
            tables,
        })
    }

    /// The feature metadata rows `selection` reads, in output order.
    async fn select(
        &self,
        ctx: &SessionContext,
        selection: Selection<'_>,
    ) -> Result<Vec<SelectedFeature>> {
        let unique = |names: Vec<&str>| {
            let mut seen = HashSet::new();
            match names.into_iter().find(|n| !seen.insert(*n)) {
                Some(name) => Err(Error::InvalidArgument(format!(
                    "feature {name:?} is selected more than once"
                ))),
                None => Ok(()),
            }
        };
        if let Selection::Rows(rows) = selection {
            unique(rows.iter().map(|r| r.name.as_str()).collect())?;
            return Ok(rows.to_vec());
        }
        let metadata_table =
            Arc::<dyn StoredTable>::from(self.table(&self.options.feature_metadata_path).await?);
        let metadata = scan_frame(ctx, metadata_table, &Dnf::default())?;
        let metadata = normalize(metadata)?;
        let schema = metadata.schema().inner().clone();
        let rows = feature_rows(&concat_batches(&schema, &metadata.collect().await?)?)?;
        let selected: Vec<&FeatureRow> = match selection {
            Selection::Rows(_) => unreachable!(),
            Selection::Latest { features, targets } => {
                let registered: HashSet<&str> = rows
                    .iter()
                    .filter(|r| r.signal_type.as_deref() == Some(SignalType::Target.as_str()))
                    .map(|r| r.name.as_str())
                    .collect();
                let misregistered: Vec<String> = targets
                    .iter()
                    .filter(|t| !registered.contains(**t))
                    .map(|t| t.to_string())
                    .collect();
                if !misregistered.is_empty() {
                    return Err(Error::MisregisteredTargets(misregistered));
                }
                let mut seen = HashSet::new();
                let requested: Vec<&str> = features
                    .iter()
                    .chain(targets)
                    .copied()
                    .filter(|n| seen.insert(*n))
                    .collect();
                let best = latest(&rows);
                let resolved: Vec<&FeatureRow> = requested
                    .iter()
                    .filter_map(|name| best.get(*name).map(|&i| &rows[i]))
                    .collect();
                if resolved.is_empty() {
                    return Err(Error::MissingFeatures(
                        requested.iter().map(|s| s.to_string()).collect(),
                    ));
                }
                resolved
            }
            Selection::Versions(pairs) => {
                unique(pairs.iter().map(|(n, _)| *n).collect())?;
                let mut versions: HashMap<(&str, i64), &FeatureRow> = HashMap::new();
                for row in &rows {
                    if let Some(version) = row.version {
                        versions.entry((row.name.as_str(), version)).or_insert(row);
                    }
                }
                let mut missing = vec![];
                let mut resolved = vec![];
                for &(name, version) in pairs {
                    match versions.get(&(name, version)) {
                        Some(&row) => resolved.push(row),
                        None => missing.push(format!("{name} version {version}")),
                    }
                }
                if !missing.is_empty() || resolved.is_empty() {
                    return Err(Error::MissingFeatures(missing));
                }
                resolved
            }
        };
        selected
            .into_iter()
            .map(|row| match &row.table_addr {
                Some(addr) => Ok(SelectedFeature {
                    name: row.name.clone(),
                    table_addr: addr.clone(),
                    signal_type: row.signal_type.clone().unwrap_or_default(),
                    feature_params: row.feature_params.clone(),
                }),
                None => Err(Error::Unsupported {
                    feature: row.name.clone(),
                    signal_type: row.signal_type.clone().unwrap_or_default(),
                }),
            })
            .collect()
    }

    fn scan(&self, name: &str, dnf: &Dnf) -> Result<BatchStream> {
        block_on(async { self.scan_table(self.table(name).await?, dnf).await })?
    }

    async fn scan_table(&self, table: Box<dyn StoredTable>, dnf: &Dnf) -> Result<BatchStream> {
        let ctx = self.context()?;
        let df = normalize(scan_frame(&ctx, Arc::from(table), dnf)?)?;
        Ok(BatchStream::from_plan(df.execute_stream().await?))
    }

    pub(crate) fn context(&self) -> Result<SessionContext> {
        let mut overrides = vec![(
            "datafusion.execution.batch_size",
            self.options.batch_size.to_string(),
        )];
        if self.options.memory_limit.is_some() {
            // One sort per scan, so that each scan's share of the limit is one sort's budget,
            // and no repartition buffers drawing on that share.
            overrides.push(("datafusion.optimizer.repartition_sorts", "false".into()));
            overrides.push((
                "datafusion.optimizer.enable_round_robin_repartition",
                "false".into(),
            ));
        }
        crate::runtime::session_context(&overrides)
    }

    /// Every row of table `name`, with dictionary columns decoded.
    pub(crate) async fn read_batch(&self, name: &str) -> Result<RecordBatch> {
        self.read_all(self.table(name).await?.as_ref()).await
    }

    /// Every row of `table`.
    pub(crate) async fn read_all(&self, table: &dyn StoredTable) -> Result<RecordBatch> {
        let scan = checked_scan(table, None, &Dnf::default(), self.options.batch_size).await?;
        let schema = scan.schema.clone();
        let batches = scan.stream.try_collect::<Vec<_>>().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    /// The latest version of table `name`, which must exist.
    pub(crate) async fn table(&self, name: &str) -> Result<Box<dyn StoredTable>> {
        self.store
            .open(name)
            .await?
            .ok_or_else(|| integrity(name, "the table does not exist"))
    }
}

/// Decodes dictionary-encoded columns so outputs have stable, plain types.
pub(crate) fn normalize(df: DataFrame) -> Result<DataFrame> {
    let mut changed = false;
    let exprs = df
        .schema()
        .iter()
        .map(|(qualifier, field)| {
            let col = Expr::Column(Column::new(qualifier.cloned(), field.name()));
            match field.data_type() {
                DataType::Dictionary(_, value) => {
                    changed = true;
                    cast_expr(col, value.as_ref().clone()).alias(field.name())
                }
                _ => col,
            }
        })
        .collect::<Vec<_>>();
    Ok(if changed { df.select(exprs)? } else { df })
}

pub(crate) fn string_column(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<Vec<Option<String>>> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| integrity("feature_metadata", format!("missing column {name:?}")))?;
    let array = cast(array, &DataType::Utf8)?;
    Ok(array
        .as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_string))
        .collect())
}

pub(crate) fn feature_rows(batch: &arrow::record_batch::RecordBatch) -> Result<Vec<FeatureRow>> {
    let names = string_column(batch, "feature_name")?;
    let addrs = string_column(batch, "table_addr")?;
    let types = string_column(batch, "signal_type")?;
    let params = match batch.column_by_name("feature_params") {
        Some(_) => string_column(batch, "feature_params")?,
        None => vec![None; batch.num_rows()],
    };
    let versions = batch
        .column_by_name("version")
        .ok_or_else(|| integrity("feature_metadata", "missing column \"version\""))?;
    let versions = cast(versions, &DataType::Int64)?;
    let versions = versions.as_primitive::<Int64Type>();
    Ok((0..batch.num_rows())
        .filter_map(|i| {
            Some(FeatureRow {
                row: i,
                name: names[i].clone()?,
                version: versions.is_valid(i).then(|| versions.value(i)),
                table_addr: addrs[i].clone(),
                signal_type: types[i].clone(),
                feature_params: params[i].clone(),
            })
        })
        .collect())
}

/// Index of the highest-version row per feature name; null versions rank last.
pub(crate) fn latest(rows: &[FeatureRow]) -> HashMap<String, usize> {
    let mut best: HashMap<String, usize> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        // `None < Some(_)`, so null versions only win when nothing else exists.
        match best.get(&row.name) {
            Some(&j) if row.version <= rows[j].version => {}
            _ => {
                best.insert(row.name.clone(), i);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;

    #[test]
    fn latest_ranks_null_versions_last_and_keeps_batch_rows() {
        let strings = |v: Vec<Option<&str>>| Arc::new(StringArray::from(v)) as _;
        let schema = Schema::new(
            ["feature_name", "version", "table_addr", "signal_type"]
                .map(|n| {
                    Field::new(
                        n,
                        if n == "version" {
                            DataType::Int64
                        } else {
                            DataType::Utf8
                        },
                        true,
                    )
                })
                .to_vec(),
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                strings(vec![None, Some("a"), Some("a"), Some("b"), Some("b")]),
                Arc::new(Int64Array::from(vec![
                    Some(9),
                    Some(2),
                    None,
                    None,
                    Some(1),
                ])),
                strings(vec![None; 5]),
                strings(vec![None; 5]),
            ],
        )
        .unwrap();
        let rows = feature_rows(&batch).unwrap();
        let best = latest(&rows);
        assert_eq!(best.len(), 2);
        assert_eq!(rows[best["a"]].row, 1);
        assert_eq!(rows[best["b"]].row, 4);
    }
}
