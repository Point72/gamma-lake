//! Writing dense features, targets and index rows.
//!
//! A write first plans: it reads the metadata, the index and any table it copies, and
//! builds every batch it will commit. Inputs the Rust core cannot write yet fail here with
//! [`Error::UnsupportedInput`] or [`Error::Unsupported`] before anything is written. It
//! then commits the feature tables, the index, the table metadata and the feature
//! metadata, in that order. A failure from then on is reported as [`Error::Write`].
//!
//! Every dense table holds exactly the index keys up to its coverage (the `last_updated`
//! of its newest table metadata row), each once. Reads align tables to the index by row
//! position, so every write keeps this true for every dense table it touches.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasher, RandomState};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use arrow::array::{
    new_null_array, Array, ArrayRef, AsArray, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray, UInt32Array,
};
use arrow::compute::{cast, concat, concat_batches, take, take_record_batch};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef, TimeUnit};
use arrow::row::{RowConverter, Rows, SortField};
use arrow::util::display::array_value_to_string;
use futures::{StreamExt, TryStreamExt};

use crate::admin::Annotation;
use crate::error::{integrity, unsupported, Error, Result};
use crate::joined::AsOfParams;
use crate::lake::{feature_rows, latest, string_column, FeatureRow, Lake, LakeOptions};
use crate::lock::write_lock;
use crate::predicate::{lossless_cast, Dnf, Op, Term};
use crate::runtime::block_on;
use crate::scan::checked_scan;
use crate::store::{now_micros, CreateOptions, StoredTable, TableStore};

/// Signal type of written columns.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SignalType {
    #[default]
    Feature,
    Target,
    /// Read with Polars' `join_asof` semantics; `feature_params` holds its arguments.
    AsOf,
    /// Stored only where there are values and left-joined to the index when read.
    Sparse,
}

/// The stored `signal_type` of runtime-computed features, which have no table.
pub(crate) const RUNTIME_COMPUTED: &str = "runtime_computed";

impl SignalType {
    /// The signal type a stored `signal_type` names, or `None` for one the core cannot read.
    pub(crate) fn parse(stored: &str) -> Option<Self> {
        [Self::Feature, Self::Target, Self::AsOf, Self::Sparse]
            .into_iter()
            .find(|t| t.as_str() == stored)
    }

    /// The `signal_type` stored in the feature metadata.
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalType::Feature => "feature",
            SignalType::Target => "target",
            SignalType::AsOf => "as_of_feature",
            SignalType::Sparse => "sparse_feature",
        }
    }

    /// Whether tables of this type are read by position and so are padded to the index.
    pub(crate) fn dense(&self) -> bool {
        matches!(self, SignalType::Feature | SignalType::Target)
    }
}

/// How [`Lake::add`] writes features whose table already covers some of the input keys.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OverlapMode {
    /// Write the features to a new table and keep the old one unchanged.
    #[default]
    Copy,
    /// Upsert into the existing table with a Delta MERGE on the sort keys.
    Merge,
}

/// Options of [`Lake::add`].
#[derive(Clone, Debug)]
pub struct AddOptions {
    pub owner: String,
    pub signal_type: SignalType,
    pub overlap_mode: OverlapMode,
    /// JSON object stored with new feature versions. Without it, new features store `{}`
    /// and new versions of existing features keep their previous parameters.
    pub feature_params: Option<String>,
    /// Comments and tags for the written versions, applied after the write; see
    /// [`Lake::annotate`]. Each names an input feature and has no `table_addr`.
    pub annotations: Vec<Annotation>,
    /// Whether the write may add keys to the index. If not, a write whose input holds a key
    /// the index does not is rejected before anything is written.
    pub allow_new_index_values: bool,
}

impl Default for AddOptions {
    fn default() -> Self {
        Self {
            owner: "missing_owner".into(),
            signal_type: SignalType::Feature,
            overlap_mode: OverlapMode::Copy,
            feature_params: None,
            annotations: vec![],
            allow_new_index_values: true,
        }
    }
}

/// The coverage of every table: the `last_updated` of its table metadata row with the
/// newest `update_timestamp`, as a one-row array. Tables whose newest row has a null
/// `last_updated` are left out.
pub(crate) fn coverage_values(tm: &RecordBatch) -> Result<BTreeMap<String, ArrayRef>> {
    let addrs = string_column(tm, "table_addr")?;
    let stamps = tm
        .column_by_name("update_timestamp")
        .ok_or_else(|| integrity("table_metadata", "missing column \"update_timestamp\""))?;
    let stamps = cast(stamps, &DataType::Int64)?;
    let stamps = stamps.as_primitive::<Int64Type>();
    let last = tm
        .column_by_name("last_updated")
        .ok_or_else(|| integrity("table_metadata", "missing column \"last_updated\""))?;
    let mut newest: BTreeMap<String, usize> = BTreeMap::new();
    for (r, addr) in addrs.iter().enumerate() {
        let (Some(addr), true) = (addr, stamps.is_valid(r)) else {
            continue;
        };
        if newest
            .get(addr)
            .is_none_or(|&o| stamps.value(o) < stamps.value(r))
        {
            newest.insert(addr.clone(), r);
        }
    }
    Ok(newest
        .into_iter()
        .filter(|&(_, r)| last.is_valid(r))
        .map(|(addr, r)| (addr, last.slice(r, 1)))
        .collect())
}

pub(crate) fn new_table_addr(prefix: &str) -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}delta_{}", &id[..8])
}

/// The Delta-compatible type new tables store for an input column of type `data_type`.
fn storage_type(data_type: &DataType) -> Option<DataType> {
    use DataType::*;
    Some(match data_type {
        Boolean | Int8 | Int16 | Int32 | Int64 | Float32 | Float64 | Date32 | Utf8 | Binary
        | Decimal128(..) => data_type.clone(),
        // Stored as the signed type of the same width; values that do not fit are rejected.
        UInt8 => Int8,
        UInt16 => Int16,
        UInt32 => Int32,
        UInt64 => Int64,
        LargeUtf8 | Utf8View => Utf8,
        LargeBinary | BinaryView => Binary,
        Dictionary(_, value) => storage_type(value)?,
        Timestamp(_, None) => Timestamp(TimeUnit::Microsecond, None),
        // Delta stores zoned timestamps in UTC; the instants are unchanged.
        Timestamp(_, Some(_)) => Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        List(f) | LargeList(f) | ListView(f) | LargeListView(f) | FixedSizeList(f, _) => List(
            Arc::new(Field::new(f.name(), storage_type(f.data_type())?, true)),
        ),
        Struct(fields) => Struct(
            fields
                .iter()
                .map(|f| Some(Field::new(f.name(), storage_type(f.data_type())?, true)))
                .collect::<Option<Vec<_>>>()?
                .into(),
        ),
        _ => return None,
    })
}

/// `schema`, the sort keys of a new lake, with the types its index stores.
fn index_storage_schema(schema: &Schema) -> Result<SchemaRef> {
    let fields = schema.fields().iter().map(|field| {
        if is_float(field.data_type()) {
            return Err(Error::InvalidArgument(format!(
                "sort key {:?} is floating-point; sort keys are matched by equality, \
                 which is not reliable for floats",
                field.name()
            )));
        }
        match storage_type(field.data_type()) {
            Some(stored) if !stored.is_nested() => {
                Ok(field.as_ref().clone().with_data_type(stored))
            }
            _ => Err(unsupported(format!(
                "index column {:?} has type {}, which the Rust core cannot store",
                field.name(),
                field.data_type()
            ))),
        }
    });
    Ok(Arc::new(Schema::new_with_metadata(
        fields.collect::<Result<Vec<_>>>()?,
        schema.metadata().clone(),
    )))
}

/// The type that holds values of both `stored` and `incoming` (storage types), following
/// Polars' supertypes for the numeric types; `None` when they do not mix.
fn promote(stored: &DataType, incoming: &DataType) -> Option<DataType> {
    use DataType::*;
    let int_rank = |t: &DataType| match t {
        Int8 => Some(1),
        Int16 => Some(2),
        Int32 => Some(3),
        Int64 => Some(4),
        _ => None,
    };
    let numeric = |t: &DataType| int_rank(t).is_some() || matches!(t, Float32 | Float64);
    let small = |t: &DataType| matches!(t, Int8 | Int16 | Float32);
    Some(match (stored, incoming) {
        _ if stored == incoming => stored.clone(),
        (List(a), List(b)) => List(Arc::new(Field::new(
            a.name(),
            promote(a.data_type(), b.data_type())?,
            true,
        ))),
        (Struct(a), Struct(b)) if a.len() == b.len() => Struct(
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| {
                    (a.name() == b.name())
                        .then(|| promote(a.data_type(), b.data_type()))
                        .flatten()
                        .map(|t| Field::new(a.name(), t, true))
                })
                .collect::<Option<Vec<_>>>()?
                .into(),
        ),
        _ => match (int_rank(stored), int_rank(incoming)) {
            (Some(a), Some(b)) => (if a >= b { stored } else { incoming }).clone(),
            _ if !numeric(stored) || !numeric(incoming) => return None,
            _ if small(stored) && small(incoming) => Float32,
            _ => Float64,
        },
    })
}

fn is_float(data_type: &DataType) -> bool {
    match data_type {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => true,
        DataType::Dictionary(_, value) => is_float(value),
        _ => false,
    }
}

fn nullable(schema: &Schema) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| Field::new(f.name(), f.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ))
}

/// `n` rows as an exclusive bound of `take` indices, which are `u32`.
fn row_count(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| {
        Error::InvalidArgument(format!(
            "cannot write {n} rows at once, more than {}",
            u32::MAX
        ))
    })
}

/// `batch` sorted by its `keys` columns, ascending with nulls first as reads sort.
fn sort_by_keys(batch: RecordBatch, keys: &[String]) -> Result<RecordBatch> {
    let columns = keys
        .iter()
        .map(|k| {
            batch
                .column_by_name(k)
                .cloned()
                .ok_or_else(|| Error::InvalidArgument(format!("missing sort key {k:?}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let converter = RowConverter::new(
        columns
            .iter()
            .map(|c| SortField::new(c.data_type().clone()))
            .collect(),
    )?;
    let rows = converter.convert_columns(&columns)?;
    if (1..rows.num_rows()).all(|i| rows.row(i - 1) <= rows.row(i)) {
        return Ok(batch);
    }
    let mut order: Vec<u32> = (0..row_count(rows.num_rows())?).collect();
    order.sort_by_key(|&r| rows.row(r as usize));
    Ok(take_record_batch(&batch, &UInt32Array::from(order))?)
}

/// Casts `values` into column `name` of a written table, rejecting lossy casts.
pub(crate) fn cast_into(values: &ArrayRef, field: &Field, table: &str) -> Result<ArrayRef> {
    lossless_cast(values, field.data_type()).map_err(|why| {
        unsupported(format!(
            "cannot write {} values into column {:?} of type {} in {table}: {why}",
            values.data_type(),
            field.name(),
            field.data_type()
        ))
    })
}

/// Builds a batch of `schema` from named columns of `len` rows; other columns are null.
pub(crate) fn assemble(
    schema: &SchemaRef,
    mut columns: HashMap<&str, ArrayRef>,
    len: usize,
    table: &str,
) -> Result<RecordBatch> {
    let arrays = schema
        .fields()
        .iter()
        .map(|field| match columns.remove(field.name().as_str()) {
            Some(values) => cast_into(&values, field, table),
            None => Ok(new_null_array(field.data_type(), len)),
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| unsupported(format!("cannot write to {table}: {e}")))
}

/// One planned commit.
pub(crate) enum Action {
    /// Appends `batch` to the existing table.
    Append {
        addr: String,
        table: Box<dyn StoredTable>,
        batch: RecordBatch,
    },
    /// Creates the table with `batch` as its first version.
    Create { addr: String, batch: RecordBatch },
    /// Upserts `source` on the sort keys, updating only `features` of matched rows.
    Merge {
        addr: String,
        table: Box<dyn StoredTable>,
        source: RecordBatch,
        features: Vec<String>,
    },
}

impl Action {
    fn addr(&self) -> &str {
        match self {
            Action::Append { addr, .. }
            | Action::Create { addr, .. }
            | Action::Merge { addr, .. } => addr,
        }
    }
}

/// A table a write commits to.
#[derive(Clone)]
pub(crate) struct Commit {
    pub(crate) addr: String,
    /// The version the write first committed on.
    pub(crate) read: Option<u64>,
}

/// The tables a write commits to, in order, and the mark of its commits, so that a failed
/// write can be undone. Creating a table is not tracked: the table is left in place, and once
/// the tables referring to it are undone, nothing does.
pub(crate) struct Commits {
    tag: String,
    tables: Mutex<Vec<Commit>>,
}

impl Default for Commits {
    fn default() -> Self {
        Self {
            tag: uuid::Uuid::new_v4().simple().to_string(),
            tables: Mutex::default(),
        }
    }
}

impl Commits {
    /// `table`, marking its commits as this write's.
    pub(crate) fn tagged(&self, mut table: Box<dyn StoredTable>) -> Box<dyn StoredTable> {
        table.tag_commits(&self.tag);
        table
    }

    /// `result`, of a commit to `addr` made on version `read`, recorded unless it certainly
    /// did not land: a conflict, or a lock that could not be taken.
    pub(crate) fn count<T>(&self, addr: &str, read: Option<u64>, result: Result<T>) -> Result<T> {
        if !matches!(result, Err(Error::Conflict { .. } | Error::Lock { .. })) {
            let mut tables = self.tables.lock().unwrap_or_else(PoisonError::into_inner);
            if !tables.iter().any(|c| c.addr == addr) {
                tables.push(Commit {
                    addr: addr.to_string(),
                    read,
                });
            }
        }
        result
    }

    /// The mark of this write's commits.
    pub(crate) fn tag(&self) -> &str {
        &self.tag
    }

    pub(crate) fn tables(&self) -> Vec<Commit> {
        self.tables
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// A key value of the primary sort key, as an array of one row and its row encoding.
#[derive(Clone)]
struct Bound {
    value: ArrayRef,
    row: Vec<u8>,
}

/// The input after key casts, null-key removal and deduplication.
struct Input {
    /// The sort keys in index order, then the feature columns.
    batch: RecordBatch,
    features: Vec<String>,
    /// Row encoding of the sort keys of every row.
    keys: Rows,
    /// Row encoding of the primary sort key of every row.
    pk: Rows,
    min: Bound,
    max: Bound,
}

/// Index rows with a primary key up to the input maximum.
struct Index {
    batch: RecordBatch,
    pk: Rows,
    /// Input rows whose keys are not in the index.
    new: Vec<u32>,
    /// Index rows whose keys are not in the input.
    missing: Vec<u32>,
}

struct Planner<'a> {
    lake: &'a Lake,
    pk_converter: RowConverter,
    input: Input,
    index: Index,
}

impl Planner<'_> {
    /// Rows of `schema` from the input rows `input` (with the values of `features`) and the
    /// index rows `index` (keys only); every other column is null. Sorted by the keys.
    fn frame(
        &self,
        schema: &SchemaRef,
        features: &[String],
        input: &[u32],
        index: &[u32],
        table: &str,
    ) -> Result<RecordBatch> {
        let input_rows = UInt32Array::from(input.to_vec());
        let index_rows = UInt32Array::from(index.to_vec());
        let len = input.len() + index.len();
        let keys: HashSet<&str> = self.lake.sort_keys.iter().map(String::as_str).collect();
        let features: HashSet<&str> = features.iter().map(String::as_str).collect();
        let mut columns: HashMap<&str, ArrayRef> = HashMap::new();
        for field in schema.fields() {
            let name = field.name().as_str();
            let is_key = keys.contains(name);
            if !is_key && !features.contains(name) {
                continue;
            }
            let values = self.input.batch.column_by_name(name).expect("input column");
            // Input rows are distinct and ascending, so all of them are in order.
            let values = if input.len() == values.len() {
                values.clone()
            } else {
                take(values, &input_rows, None)?
            };
            let head = cast_into(&values, field, table)?;
            if index.is_empty() {
                columns.insert(name, head);
                continue;
            }
            let tail = if is_key {
                let keys = self.index.batch.column_by_name(name).expect("index key");
                cast_into(&take(keys, &index_rows, None)?, field, table)?
            } else {
                new_null_array(field.data_type(), index.len())
            };
            let column = if input.is_empty() {
                tail
            } else {
                concat(&[head.as_ref(), tail.as_ref()])?
            };
            columns.insert(name, column);
        }
        sort_by_keys(assemble(schema, columns, len, table)?, &self.lake.sort_keys)
    }

    /// `rows` whose primary key satisfies `keep`.
    fn select(rows: &[u32], pk: &Rows, keep: impl Fn(&[u8]) -> bool) -> Vec<u32> {
        rows.iter()
            .copied()
            .filter(|&r| keep(pk.row(r as usize).as_ref()))
            .collect()
    }

    /// Rows of `table`, table `addr`, outside the primary key range `[min, max]`, in `schema`.
    async fn outside(
        &self,
        table: &dyn StoredTable,
        addr: &str,
        schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let term = |op, bound: &Bound| Term {
            column: self.lake.options.primary_sort_key.clone(),
            op,
            values: bound.value.clone(),
        };
        let dnf = Dnf(vec![
            vec![term(Op::Lt, &self.input.min)],
            vec![term(Op::Gt, &self.input.max)],
        ]);
        let columns = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();
        let scan = checked_scan(table, Some(&columns), &dnf, self.lake.options.batch_size).await?;
        let source = scan.schema.clone();
        let batch = concat_batches(&source, &scan.stream.try_collect::<Vec<_>>().await?)?;
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                (
                    f.name().as_str(),
                    batch.column_by_name(f.name()).unwrap().clone(),
                )
            })
            .collect();
        assemble(schema, columns, batch.num_rows(), addr)
    }
}

/// Attempts at a write while it conflicts with other writers; see [`Lake::add`].
const ATTEMPTS: u32 = 10;
/// The longest wait before the second attempt at a write. It doubles before each later
/// attempt, up to [`MAX_BACKOFF`].
const BACKOFF: Duration = Duration::from_millis(50);
const MAX_BACKOFF: Duration = Duration::from_secs(1);

/// The random wait after attempt number `attempt` at a write conflicted, so that writers
/// that conflicted do not run again in step.
fn backoff(attempt: u32) -> Duration {
    let longest = BACKOFF
        .saturating_mul(2u32.saturating_pow(attempt - 1))
        .min(MAX_BACKOFF);
    // 53 random bits, as a fraction in [0, 1).
    let random = RandomState::new().hash_one(attempt) >> 11;
    longest.mul_f64(random as f64 / (1u64 << 53) as f64)
}

impl Lake {
    /// Creates the index, metadata and transaction log tables of a lake in `store` where they
    /// are missing, and opens it.
    ///
    /// `index_schema` holds the sort keys; it must contain `options.primary_sort_key`.
    pub fn create_with_store(
        store: Arc<dyn TableStore>,
        index_schema: SchemaRef,
        options: LakeOptions,
    ) -> Result<Lake> {
        let store = crate::lake::locking(store, &options);
        let index_schema = index_storage_schema(&index_schema)?;
        let pk = index_schema
            .field_with_name(&options.primary_sort_key)
            .map_err(|_| {
                Error::InvalidArgument(format!(
                    "primary sort key {:?} is not an index column {:?}",
                    options.primary_sort_key,
                    index_schema
                        .fields()
                        .iter()
                        .map(|f| f.name())
                        .collect::<Vec<_>>()
                ))
            })?
            .clone();
        let mut names = HashSet::new();
        if let Some(f) = index_schema
            .fields()
            .iter()
            .find(|f| !names.insert(f.name()))
        {
            return Err(Error::InvalidArgument(format!(
                "index column {:?} appears twice",
                f.name()
            )));
        }
        let lock = write_lock(store.location())?;
        let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);

        let string = |name: &str| Field::new(name, DataType::Utf8, true);
        let tables = [
            (options.index_path.as_str(), nullable(&index_schema)),
            (
                options.feature_metadata_path.as_str(),
                Arc::new(Schema::new(vec![
                    string("feature_name"),
                    Field::new("version", DataType::Int64, true),
                    string("table_addr"),
                    string("owner"),
                    string("signal_type"),
                    string("feature_params"),
                ])),
            ),
            (
                options.table_metadata_path.as_str(),
                Arc::new(Schema::new(vec![
                    string("table_addr"),
                    Field::new("last_updated", pk.data_type().clone(), true),
                    Field::new(
                        "update_timestamp",
                        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                        true,
                    ),
                ])),
            ),
            (
                options.transaction_log_path.as_str(),
                crate::transaction::schema(),
            ),
        ];
        block_on(async {
            for (name, schema) in tables {
                if store.open(name).await?.is_none() {
                    let batch = RecordBatch::new_empty(schema);
                    store.create(name, batch, &CreateOptions::default()).await?;
                }
            }
            Ok::<_, Error>(())
        })??;
        drop(_guard);
        Lake::open_locked(store, options)
    }

    /// Adds dense features or targets. Non-key columns of `input` are the features; it must
    /// hold every sort key.
    ///
    /// Rows with a null key are dropped and, of rows sharing keys, the first is kept. Keys
    /// new to the index are added to it, and every dense table whose coverage includes them
    /// gets them as rows of nulls. If [`AddOptions::allow_new_index_values`] is false, an input
    /// with keys new to the index fails with [`Error::InvalidArgument`] instead.
    ///
    /// Features new to the lake go to a new table. Features whose table covers none of
    /// the input keys already in the index are appended to it. Otherwise the input
    /// replaces the features over the input's primary key range: [`OverlapMode::Copy`]
    /// writes them with the table's rows outside that range to a new table, and
    /// [`OverlapMode::Merge`] upserts them into the table; both record a new version.
    ///
    /// Writes to one lake are serialized within this process, and readers may observe a write
    /// half done. A commit to a table that another writer changed after this write read it
    /// fails with [`Error::Conflict`]. A write that adds no index rows does not commit to the
    /// index, so it can miss keys added by a concurrent writer without failing; writers in
    /// other processes must not add index rows while it runs.
    ///
    /// Each write that passes validation is a transaction in the transaction log: `pending`,
    /// then `success`. A write that fails after committing a
    /// table is logged `failed`, each table it committed to is restored to the version it
    /// read, newest commit first, and it is logged `rolled_back`; the write then fails as one
    /// that committed nothing does. Tables it created are left in place, unreferenced.
    /// Restoring a table another writer committed to since would undo that commit too, so
    /// then nothing is restored and the write returns an [`Error::Write`] that says so; a
    /// restore that fails is an [`Error::Write`] too. Either leaves the transaction `failed`.
    ///
    /// A write that fails with [`Error::Conflict`] leaves the lake as it was, apart from the
    /// tables it created, so it is planned and run again, as a new transaction, up to 10 times
    /// in all, after a random wait of up to 50 ms that doubles with each attempt, to at most
    /// 1 s. Each attempt plans from the lake as the writers it conflicted with left it, so a
    /// feature another writer added meanwhile is written to as any later write would, for
    /// example as a new version. The conflict is returned once every attempt has conflicted.
    pub fn add(&self, input: Vec<RecordBatch>, options: &AddOptions) -> Result<()> {
        if let Some(params) = &options.feature_params {
            if !matches!(
                serde_json::from_str(params),
                Ok(serde_json::Value::Object(_))
            ) {
                return Err(Error::InvalidArgument(
                    "feature_params must be a JSON object".into(),
                ));
            }
        }
        if let Some(first) = input.first() {
            let schema = first.schema();
            for a in &options.annotations {
                if a.table_addr.is_some() {
                    return Err(Error::InvalidArgument(format!(
                        "the annotation of {:?} names a table; annotations of an add apply to \
                         the versions it writes",
                        a.feature_name
                    )));
                }
                if self.sort_keys.contains(&a.feature_name)
                    || schema.field_with_name(&a.feature_name).is_err()
                {
                    return Err(Error::InvalidArgument(format!(
                        "annotated feature {:?} is not a feature column of the input",
                        a.feature_name
                    )));
                }
            }
        }
        self.write(input, options)
    }

    /// Adds keys to the index without features.
    /// Non-key columns of `input` are ignored. Transactions are logged, rolled back and retried
    /// as in [`Lake::add`], with the owner and signal type of [`AddOptions::default`].
    pub fn add_index_rows(&self, input: Vec<RecordBatch>) -> Result<()> {
        let keys = self.sort_keys.clone();
        let input = input
            .into_iter()
            .map(|batch| {
                let indices = keys
                    .iter()
                    .map(|k| batch.schema().index_of(k))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|_| {
                        Error::InvalidArgument(format!("input must hold the sort keys {keys:?}"))
                    })?;
                Ok(batch.project(&indices)?)
            })
            .collect::<Result<Vec<_>>>()?;
        self.write(input, &AddOptions::default())
    }

    fn write(&self, input: Vec<RecordBatch>, options: &AddOptions) -> Result<()> {
        // Plan (read and build every batch), commit, then annotate the committed versions, all as
        // one logged transaction that undoes its commits if it fails.
        self.retrying(|| {
            block_on(async {
                let (actions, index, metadata) = self.plan(input.clone(), options).await?;
                let commits = Commits::default();
                let body = async {
                    let committed = commit(self, actions, Some(index), metadata, &commits).await?;
                    if !options.annotations.is_empty() {
                        self.annotate_locked(&options.annotations, committed.clone(), &commits)
                            .await?
                    }
                    Ok(committed)
                };
                self.transaction(&options.owner, options.signal_type.as_str(), &commits, body)
                    .await
            })?
        })
    }

    /// Runs `write` with the write lock held, and again while it fails with
    /// [`Error::Conflict`], up to [`ATTEMPTS`] times in all. Between attempts it waits a
    /// [`backoff`] without the lock, so that other writes in this process can run meanwhile.
    /// `write` must plan anew on each attempt, and a conflict must leave the lake as it was,
    /// apart from tables it created.
    pub(crate) fn retrying<T>(&self, mut write: impl FnMut() -> Result<T>) -> Result<T> {
        let lock = write_lock(self.store.location())?;
        let mut attempt = 1;
        loop {
            let result = {
                let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
                write()
            };
            match result {
                Err(Error::Conflict { .. }) if attempt < ATTEMPTS => {
                    std::thread::sleep(backoff(attempt));
                    attempt += 1;
                }
                result => return result,
            }
        }
    }

    /// Table `name` and its rows that satisfy `dnf`, read through the handle later commits use.
    pub(crate) async fn read_table(
        &self,
        name: &str,
        dnf: &Dnf,
    ) -> Result<(Box<dyn StoredTable>, RecordBatch)> {
        let table = self.table(name).await?;
        let scan = checked_scan(table.as_ref(), None, dnf, self.options.batch_size).await?;
        let batch = concat_batches(&scan.schema, &scan.stream.try_collect::<Vec<_>>().await?)?;
        Ok((table, batch))
    }

    fn prepare(
        &self,
        input: Vec<RecordBatch>,
        key_converter: &RowConverter,
        pk_converter: &RowConverter,
    ) -> Result<Input> {
        let Some(first) = input.first() else {
            return Err(Error::InvalidArgument("the input has no rows".into()));
        };
        let schema = first.schema();
        let batch = concat_batches(&schema, &input)?;
        let mut seen = HashSet::new();
        if let Some(f) = schema.fields().iter().find(|f| !seen.insert(f.name())) {
            return Err(Error::InvalidArgument(format!(
                "input column {:?} appears twice",
                f.name()
            )));
        }
        let mut columns = vec![];
        let mut fields = vec![];
        for key in self.index_schema.fields() {
            let values = batch.column_by_name(key.name()).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "the input has no sort key column {:?}; it must hold {:?}",
                    key.name(),
                    self.sort_keys
                ))
            })?;
            if is_float(key.data_type()) || is_float(values.data_type()) {
                return Err(unsupported(format!(
                    "floating-point sort key {:?} is not supported",
                    key.name()
                )));
            }
            columns.push(cast_into(values, key, "the index")?);
            fields.push(Field::new(key.name(), key.data_type().clone(), true));
        }
        let mut features = vec![];
        for (field, values) in schema.fields().iter().zip(batch.columns()) {
            if !self.sort_keys.contains(field.name()) {
                features.push(field.name().clone());
                columns.push(values.clone());
                fields.push(Field::new(field.name(), field.data_type().clone(), true));
            }
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;

        let nkeys = self.sort_keys.len();
        let keys = key_converter.convert_columns(&batch.columns()[..nkeys])?;
        let mut kept = HashSet::new();
        let rows: Vec<u32> = (0..row_count(batch.num_rows())?)
            .filter(|&r| {
                batch.columns()[..nkeys]
                    .iter()
                    .all(|c| c.is_valid(r as usize))
            })
            .filter(|&r| kept.insert(keys.row(r as usize)))
            .collect();
        if rows.is_empty() {
            return Err(Error::InvalidArgument(
                "the input has no rows with non-null sort keys".into(),
            ));
        }
        let batch = if rows.len() == batch.num_rows() {
            batch
        } else {
            take_record_batch(&batch, &UInt32Array::from(rows))?
        };
        let keys = key_converter.convert_columns(&batch.columns()[..nkeys])?;
        let pk_index = self.index_schema.index_of(&self.options.primary_sort_key)?;
        let pk = pk_converter.convert_columns(&[batch.column(pk_index).clone()])?;
        let rows = 0..pk.num_rows();
        let min = rows.clone().min_by_key(|&r| pk.row(r)).unwrap_or_default();
        let max = rows.max_by_key(|&r| pk.row(r)).unwrap_or_default();
        let bound = |r: usize| Bound {
            value: batch.column(pk_index).slice(r, 1),
            row: pk.row(r).as_ref().to_vec(),
        };
        Ok(Input {
            min: bound(min),
            max: bound(max),
            batch,
            features,
            keys,
            pk,
        })
    }

    async fn plan(
        &self,
        input: Vec<RecordBatch>,
        options: &AddOptions,
    ) -> Result<(Vec<Action>, Action, Vec<Action>)> {
        let pk_name = &self.options.primary_sort_key;
        let pk_type = self.index_schema.field_with_name(pk_name)?.data_type();
        let pk_converter = RowConverter::new(vec![SortField::new(pk_type.clone())])?;
        let key_converter = RowConverter::new(
            self.index_schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;
        let input = self.prepare(input, &key_converter, &pk_converter)?;
        // Index rows up to the input maximum, and how they relate to the input keys.
        let dnf = Dnf(vec![vec![Term {
            column: pk_name.clone(),
            op: Op::LtEq,
            values: input.max.value.clone(),
        }]]);
        // Each table is read and committed to through one handle, so a commit fails if another
        // writer committed to the table after this write read it.
        let all = Dnf::default();
        let ((index_table, index_batch), fm, tm) = futures::try_join!(
            self.read_table(&self.options.index_path, &dnf),
            self.read_table(&self.options.feature_metadata_path, &all),
            self.read_table(&self.options.table_metadata_path, &all),
        )?;
        let index_keys = key_converter.convert_columns(index_batch.columns())?;
        let in_index: HashSet<_> = index_keys.iter().collect();
        let in_input: HashSet<_> = input.keys.iter().collect();
        if in_index.len() != index_keys.num_rows() {
            return Err(integrity(
                &self.options.index_path,
                "the index holds duplicate keys",
            ));
        }
        let new: Vec<u32> = (0..row_count(input.keys.num_rows())?)
            .filter(|&r| !in_index.contains(&input.keys.row(r as usize)))
            .collect();
        if let (false, Some(&r)) = (options.allow_new_index_values, new.first()) {
            let key = self
                .index_schema
                .fields()
                .iter()
                .zip(input.batch.columns())
                .map(|(f, c)| {
                    Ok(format!(
                        "{}={}",
                        f.name(),
                        array_value_to_string(c, r as usize)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            return Err(Error::InvalidArgument(format!(
                "{} input keys are not in the index, such as ({}), and new index values are not allowed",
                new.len(),
                key.join(", ")
            )));
        }
        let missing: Vec<u32> = (0..row_count(index_keys.num_rows())?)
            .filter(|&r| !in_input.contains(&index_keys.row(r as usize)))
            .collect();
        let pk_index = self.index_schema.index_of(pk_name)?;
        let index_pk = pk_converter.convert_columns(&[index_batch.column(pk_index).clone()])?;
        let planner = Planner {
            lake: self,
            pk_converter,
            input,
            index: Index {
                batch: index_batch,
                pk: index_pk,
                new,
                missing,
            },
        };
        planner.plan(options, index_table, fm, tm).await
    }
}

impl Planner<'_> {
    /// The actions of the write, given the index, feature metadata and table metadata tables
    /// and their rows.
    async fn plan(
        self,
        options: &AddOptions,
        index_table: Box<dyn StoredTable>,
        (fm_table, fm): (Box<dyn StoredTable>, RecordBatch),
        (tm_table, tm): (Box<dyn StoredTable>, RecordBatch),
    ) -> Result<(Vec<Action>, Action, Vec<Action>)> {
        let lake = self.lake;
        let rows = feature_rows(&fm)?;
        let best = latest(&rows);

        // Input features grouped by the table holding their latest version.
        let mut groups: BTreeMap<Option<String>, Vec<String>> = BTreeMap::new();
        for name in &self.input.features {
            let addr = match best.get(name).map(|&i| &rows[i]) {
                None => None,
                Some(row) => {
                    let signal_type = row.signal_type.as_deref().unwrap_or_default();
                    if SignalType::parse(signal_type).is_none() {
                        return Err(Error::Unsupported {
                            feature: name.clone(),
                            signal_type: signal_type.to_string(),
                        });
                    }
                    row.table_addr.clone()
                }
            };
            groups.entry(addr).or_default().push(name.clone());
        }

        let signal_type = options.signal_type;
        let pk_type = lake
            .index_schema
            .field_with_name(&lake.options.primary_sort_key)?
            .data_type();
        if signal_type == SignalType::AsOf {
            // Reject parameters that every later read of these features would fail on.
            for name in &self.input.features {
                let params = options.feature_params.as_deref().or_else(|| {
                    best.get(name)
                        .and_then(|&i| rows[i].feature_params.as_deref())
                });
                AsOfParams::parse(params.unwrap_or("{}"))?.check(pk_type)?;
            }
        }

        let mut coverage: BTreeMap<String, Bound> = BTreeMap::new();
        for (addr, last) in coverage_values(&tm)? {
            let value = lossless_cast(&last, pk_type).map_err(|why| integrity(&addr, why))?;
            let row = self
                .pk_converter
                .convert_columns(std::slice::from_ref(&value))?;
            let row = row.row(0).as_ref().to_vec();
            coverage.insert(addr, Bound { value, row });
        }

        // Tables that ever held an as-of or sparse feature are never padded.
        let types = string_column(&fm, "signal_type")?;
        let fm_addrs = string_column(&fm, "table_addr")?;
        let unpadded: HashSet<&str> = types
            .iter()
            .zip(&fm_addrs)
            .filter(|(t, _)| {
                t.as_deref()
                    .and_then(SignalType::parse)
                    .is_some_and(|t| !t.dense())
            })
            .filter_map(|(_, a)| a.as_deref())
            .collect();

        let input_rows: Vec<u32> = (0..row_count(self.input.batch.num_rows())?).collect();
        let all_new = self.index.new.len() == input_rows.len();
        let (min, max) = (&self.input.min, &self.input.max);
        let mut actions = vec![];
        let mut tm_rows: Vec<(String, Bound)> = vec![];
        // (base feature metadata row, feature name, version, table address)
        let mut fm_rows: Vec<(Option<&FeatureRow>, String, i64, String)> = vec![];
        let keys = &lake.sort_keys;
        let index_schema = index_table.schema();
        let opens: Vec<_> = groups.keys().flatten().map(|a| lake.table(a)).collect();
        let mut opened = futures::stream::iter(opens)
            .buffered(16)
            .try_collect::<Vec<_>>()
            .await?
            .into_iter();
        for (addr, features) in &groups {
            let Some(addr) = addr else {
                let addr = new_table_addr(&lake.options.table_name_prefix);
                let schema = self.new_schema(&index_schema, features)?;
                // New as-of tables are padded with the missing index rows; sparse tables are not.
                let missing = match signal_type {
                    SignalType::Sparse => &[][..],
                    _ => &self.index.missing[..],
                };
                let batch = self.frame(&schema, features, &input_rows, missing, &addr)?;
                actions.push(Action::Create {
                    addr: addr.clone(),
                    batch,
                });
                tm_rows.push((addr.clone(), max.clone()));
                for f in features {
                    fm_rows.push((None, f.clone(), 0, addr.clone()));
                }
                continue;
            };
            let table = opened.next().expect("a table per existing group");
            let covered = coverage
                .get(addr)
                .ok_or_else(|| integrity(addr, "the table has no table metadata row"))?;
            let l = covered.row.as_slice();
            let schema = table.schema();
            let coverage_after = if covered.row >= max.row {
                covered.clone()
            } else {
                max.clone()
            };
            if all_new || min.row.as_slice() > l {
                // Padding keeps dense tables aligned; in an as-of table it would hide the
                // carried-forward value, so all-new rows are appended alone.
                let tail = if all_new && !signal_type.dense() {
                    vec![]
                } else {
                    Self::select(&self.index.missing, &self.index.pk, |pk| pk > l)
                };
                let batch = self.frame(&schema, features, &input_rows, &tail, addr)?;
                actions.push(Action::Append {
                    addr: addr.clone(),
                    table,
                    batch,
                });
                tm_rows.push((addr.clone(), coverage_after));
                continue;
            }
            let window = Self::select(&self.index.missing, &self.index.pk, |pk| {
                pk >= min.row.as_slice() && pk <= max.row.as_slice()
            });
            let version = features
                .iter()
                .filter_map(|f| rows[best[f]].version)
                .max()
                .unwrap_or(-1)
                + 1;
            let target = match options.overlap_mode {
                OverlapMode::Merge => {
                    let fields: Vec<Field> = keys
                        .iter()
                        .chain(features)
                        .map(|n| schema.field_with_name(n).cloned())
                        .collect::<std::result::Result<_, _>>()
                        .map_err(|e| integrity(addr, e.to_string()))?;
                    let source_schema = Arc::new(Schema::new(fields));
                    let source =
                        self.frame(&source_schema, features, &input_rows, &window, addr)?;
                    actions.push(Action::Merge {
                        addr: addr.clone(),
                        table,
                        source,
                        features: features.clone(),
                    });
                    addr.clone()
                }
                OverlapMode::Copy => {
                    // The copy holds each feature at the supertype of its stored and incoming
                    // types, as concatenating the two in Polars would.
                    let mut copy_types = HashMap::new();
                    for name in features {
                        let stored = schema
                            .field_with_name(name)
                            .map_err(|e| integrity(addr, e.to_string()))?
                            .data_type();
                        let incoming = self
                            .input
                            .batch
                            .schema_ref()
                            .field_with_name(name)?
                            .data_type()
                            .clone();
                        let wide = storage_type(&incoming)
                            .and_then(|incoming| promote(stored, &incoming))
                            .ok_or_else(|| {
                                unsupported(format!(
                                    "copy mode cannot combine feature '{name}' of type {stored} \
                                     with new values of type {incoming}"
                                ))
                            })?;
                        copy_types.insert(name.as_str(), wide);
                    }
                    let early = if unpadded.contains(addr.as_str()) {
                        vec![]
                    } else {
                        Self::select(&self.index.new, &self.input.pk, |pk| pk <= l)
                    };
                    let frozen = self.frame(&schema, &[], &early, &[], addr)?;
                    let fields: Vec<Field> = keys
                        .iter()
                        .chain(features)
                        .map(|n| {
                            schema.field_with_name(n).map(|f| {
                                let data_type = copy_types.get(n.as_str()).unwrap_or(f.data_type());
                                Field::new(n, data_type.clone(), true)
                            })
                        })
                        .collect::<std::result::Result<_, _>>()
                        .map_err(|e| integrity(addr, e.to_string()))?;
                    let copy_schema = Arc::new(Schema::new(fields));
                    let old = self.outside(table.as_ref(), addr, &copy_schema).await?;
                    let window = if signal_type.dense() {
                        &window[..]
                    } else {
                        &[][..]
                    };
                    let fresh = self.frame(&copy_schema, features, &input_rows, window, addr)?;
                    let batch = sort_by_keys(concat_batches(&copy_schema, [&old, &fresh])?, keys)?;
                    if frozen.num_rows() > 0 {
                        actions.push(Action::Append {
                            addr: addr.clone(),
                            table,
                            batch: frozen,
                        });
                    }
                    let copy = new_table_addr(&lake.options.table_name_prefix);
                    actions.push(Action::Create {
                        addr: copy.clone(),
                        batch,
                    });
                    copy
                }
            };
            tm_rows.push((target.clone(), coverage_after));
            for f in features {
                fm_rows.push((Some(&rows[best[f]]), f.clone(), version, target.clone()));
            }
        }

        // Pad other dense tables with the new keys they cover.
        let homes: HashSet<&str> = groups.keys().flatten().map(String::as_str).collect();
        let pads: Vec<(&String, Vec<u32>)> = coverage
            .iter()
            .filter(|(addr, _)| !homes.contains(addr.as_str()) && !unpadded.contains(addr.as_str()))
            .map(|(addr, covered)| {
                let l = covered.row.as_slice();
                (
                    addr,
                    Self::select(&self.index.new, &self.input.pk, |pk| pk <= l),
                )
            })
            .filter(|(_, pad)| !pad.is_empty())
            .collect();
        let opens: Vec<_> = pads.iter().map(|(addr, _)| lake.table(addr)).collect();
        let tables: Vec<_> = futures::stream::iter(opens)
            .buffered(16)
            .try_collect()
            .await?;
        for ((addr, pad), table) in pads.into_iter().zip(tables) {
            let batch = self.frame(&table.schema(), &[], &pad, &[], addr)?;
            actions.push(Action::Append {
                addr: addr.clone(),
                table,
                batch,
            });
        }

        let index_batch = self.frame(
            &index_schema,
            &[],
            &self.index.new,
            &[],
            &lake.options.index_path,
        )?;
        let index = Action::Append {
            addr: lake.options.index_path.clone(),
            table: index_table,
            batch: index_batch,
        };

        let mut metadata = vec![];
        if !tm_rows.is_empty() {
            let schema = tm_table.schema();
            let n = tm_rows.len();
            let values: Vec<&dyn Array> = tm_rows.iter().map(|(_, b)| b.value.as_ref()).collect();
            let columns = HashMap::from([
                (
                    "table_addr",
                    Arc::new(StringArray::from_iter_values(
                        tm_rows.iter().map(|(a, _)| a),
                    )) as ArrayRef,
                ),
                ("last_updated", concat(&values)?),
                (
                    "update_timestamp",
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![now_micros(); n]).with_timezone("UTC"),
                    ),
                ),
            ]);
            let batch = assemble(&schema, columns, n, &lake.options.table_metadata_path)?;
            metadata.push(Action::Append {
                addr: lake.options.table_metadata_path.clone(),
                table: tm_table,
                batch,
            });
        }
        if !fm_rows.is_empty() {
            let schema = fm_table.schema();
            let n = fm_rows.len();
            let base: UInt32Array = fm_rows
                .iter()
                .map(|(r, ..)| r.map(|r| r.row as u32))
                .collect();
            let params = fm_rows
                .iter()
                .map(|(r, ..)| match (&options.feature_params, r) {
                    (Some(p), _) => Some(p.clone()),
                    (None, None) => Some("{}".to_string()),
                    (None, Some(r)) => r.feature_params.clone(),
                });
            let mut columns: HashMap<&str, ArrayRef> = HashMap::from([
                (
                    "feature_name",
                    Arc::new(StringArray::from_iter_values(fm_rows.iter().map(|r| &r.1)))
                        as ArrayRef,
                ),
                (
                    "version",
                    Arc::new(Int64Array::from_iter_values(fm_rows.iter().map(|r| r.2))),
                ),
                (
                    "table_addr",
                    Arc::new(StringArray::from_iter_values(fm_rows.iter().map(|r| &r.3))),
                ),
                (
                    "owner",
                    Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                        options.owner.as_str(),
                        n,
                    ))),
                ),
                (
                    "signal_type",
                    Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                        options.signal_type.as_str(),
                        n,
                    ))),
                ),
                ("feature_params", Arc::new(StringArray::from_iter(params))),
            ]);
            for field in schema.fields() {
                if !columns.contains_key(field.name().as_str()) {
                    if let Some(values) = fm.column_by_name(field.name()) {
                        columns.insert(field.name().as_str(), take(values, &base, None)?);
                    }
                }
            }
            let batch = assemble(&schema, columns, n, &lake.options.feature_metadata_path)?;
            metadata.push(Action::Append {
                addr: lake.options.feature_metadata_path.clone(),
                table: fm_table,
                batch,
            });
        }
        Ok((actions, index, metadata))
    }

    /// Schema of a new table: the `index` columns, then `features` as the input holds them.
    fn new_schema(&self, index: &Schema, features: &[String]) -> Result<SchemaRef> {
        let mut fields: Vec<Field> = self
            .lake
            .sort_keys
            .iter()
            .map(|k| {
                index
                    .field_with_name(k)
                    .map(|f| Field::new(k, f.data_type().clone(), true))
            })
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| integrity(&self.lake.options.index_path, e.to_string()))?;
        for name in features {
            let field = self.input.batch.schema().field_with_name(name)?.clone();
            let data_type = storage_type(field.data_type()).ok_or_else(|| {
                unsupported(format!(
                    "column {name:?} has type {}, which the Rust core cannot store yet",
                    field.data_type()
                ))
            })?;
            fields.push(Field::new(name, data_type, true));
        }
        Ok(Arc::new(Schema::new(fields)))
    }
}

async fn run(
    action: Action,
    store: &dyn TableStore,
    keys: &[String],
    commits: &Commits,
) -> Result<()> {
    match action {
        Action::Append { addr, table, batch } => {
            if batch.num_rows() > 0 {
                let table = commits.tagged(table);
                let read = table.version();
                commits.count(&addr, read, table.append(batch).await)?;
            }
        }
        Action::Create { addr, batch } => {
            let options = CreateOptions {
                stats_columns: keys.to_vec(),
            };
            store.create(&addr, batch, &options).await?;
        }
        Action::Merge {
            addr,
            table,
            source,
            features,
        } => {
            let table = commits.tagged(table);
            let read = table.version();
            commits.count(&addr, read, table.merge(source, keys, &features).await)?;
        }
    }
    Ok(())
}

/// `error` without the tables it reports as written, which are added to `committed`. A table
/// whose lock failed to release after its commit is reported this way.
fn written(error: Error, committed: &mut Vec<String>) -> Error {
    match error {
        Error::Write {
            committed: written,
            source,
            ..
        } => {
            committed.extend(written);
            *source
        }
        error => error,
    }
}

/// Commits the feature tables concurrently, up to 16 at a time, then the index, then the
/// metadata tables, and returns the addresses of the tables committed, counting the commits
/// in `commits`. A failure is an [`Error::Write`], unless nothing was committed and every
/// failure was a conflict; the first conflict is then returned as is. Other errors may come
/// after a commit, so they never count as conflicts, and are reported ahead of them.
pub(crate) async fn commit(
    lake: &Lake,
    actions: Vec<Action>,
    index: Option<Action>,
    metadata: Vec<Action>,
    commits: &Commits,
) -> Result<Vec<String>> {
    let attempt = |action: Action| async move {
        let addr = action.addr().to_string();
        let result = run(action, lake.store.as_ref(), &lake.sort_keys, commits).await;
        (addr, result)
    };
    let mut committed = vec![];
    let mut failure = None;
    let mut only_conflicts = true;
    let attempts = futures::stream::iter(actions).map(attempt).buffered(16);
    for (addr, result) in attempts.collect::<Vec<_>>().await {
        match result {
            Ok(()) => committed.push(addr),
            Err(e) => {
                let e = written(e, &mut committed);
                let conflict = matches!(e, Error::Conflict { .. });
                only_conflicts &= conflict;
                // A conflict is reported only when every failure was one, so that a rolled
                // back write fails as a conflict only when retrying it can succeed.
                let conflicted = matches!(failure, Some((_, Error::Conflict { .. })));
                if failure.is_none() || (!conflict && conflicted) {
                    failure = Some((format!("writing table {addr}"), e));
                }
            }
        }
    }
    for action in index.into_iter().chain(metadata) {
        if failure.is_some() {
            break;
        }
        let (addr, result) = attempt(action).await;
        match result {
            Ok(()) => committed.push(addr),
            Err(e) => failure = Some((format!("writing table {addr}"), written(e, &mut committed))),
        }
    }
    match failure {
        None => Ok(committed),
        Some((_, conflict @ Error::Conflict { .. })) if committed.is_empty() && only_conflicts => {
            Err(conflict)
        }
        Some((phase, source)) => Err(Error::Write {
            phase,
            committed,
            source: Box::new(source),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;
    use crate::store::TableScan;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use async_trait::async_trait;

    #[test]
    fn row_count_rejects_more_rows_than_take_indices_hold() {
        assert_eq!(row_count(u32::MAX as usize).unwrap(), u32::MAX);
        assert!(matches!(
            row_count(u32::MAX as usize + 1),
            Err(Error::InvalidArgument(_))
        ));
    }

    fn frame(ts: &[i64]) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("ts", DataType::Int64, true),
            Field::new("sym", DataType::Utf8, true),
            Field::new("x", DataType::Float64, true),
        ]);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ts.to_vec())),
            Arc::new(StringArray::from_iter_values(ts.iter().map(|_| "a"))),
            Arc::new(Float64Array::from_iter_values(ts.iter().map(|&t| t as f64))),
        ];
        RecordBatch::try_new(Arc::new(schema), columns).unwrap()
    }

    /// A lake in `store` keyed on `ts` and `sym`.
    fn lake(store: Arc<dyn TableStore>) -> Lake {
        let index = Schema::new(vec![
            Field::new("ts", DataType::Int64, true),
            Field::new("sym", DataType::Utf8, true),
        ]);
        let options = LakeOptions {
            primary_sort_key: "ts".into(),
            ..Default::default()
        };
        Lake::create_with_store(store, Arc::new(index), options).unwrap()
    }

    /// Plans a write of a new key for an existing feature, lets another writer commit to its
    /// feature table or else the index, then commits the plan. Returns the error and the number
    /// of index rows after it.
    fn raced_write(feature: bool) -> (Error, usize) {
        let lake = lake(Arc::new(MemoryStore::new()));
        lake.add(vec![frame(&[1, 2])], &AddOptions::default())
            .unwrap();
        block_on(async {
            let (actions, index, metadata) = lake
                .plan(vec![frame(&[3])], &AddOptions::default())
                .await
                .unwrap();
            let raced = if feature {
                actions[0].addr().to_string()
            } else {
                lake.options.index_path.clone()
            };
            let other = lake.store.open(&raced).await.unwrap().unwrap();
            other
                .set_column_metadata("ts", HashMap::from([("k".into(), "v".into())]))
                .await
                .unwrap();
            let error = commit(&lake, actions, Some(index), metadata, &Commits::default())
                .await
                .unwrap_err();
            let index = lake.read_batch(&lake.options.index_path).await.unwrap();
            (error, index.num_rows())
        })
        .unwrap()
    }

    #[test]
    fn a_write_racing_another_commit_fails() {
        let (error, rows) = raced_write(true);
        assert!(matches!(error, Error::Conflict { .. }), "{error}");
        assert_eq!(rows, 2);

        let (error, rows) = raced_write(false);
        let Error::Write {
            committed, source, ..
        } = error
        else {
            panic!("{error}");
        };
        assert_eq!(committed.len(), 1);
        assert!(matches!(*source, Error::Conflict { .. }), "{source}");
        assert_eq!(rows, 2);
    }

    /// A store in which another writer commits to table `raced` as soon as a write reads it,
    /// as many times as `races` says: it appends `rows` if given, and otherwise changes the
    /// metadata of a column.
    #[derive(Debug)]
    struct Racing {
        inner: MemoryStore,
        raced: String,
        races: Arc<Mutex<usize>>,
        rows: Option<RecordBatch>,
    }

    struct RacingTable {
        inner: Box<dyn StoredTable>,
        store: MemoryStore,
        name: String,
        /// For the raced table, the races left and the rows they append.
        race: Option<(Arc<Mutex<usize>>, Option<RecordBatch>)>,
    }

    /// A lake in a [`Racing`] store, and its count of races left, at first none.
    fn racing(raced: &str, rows: Option<RecordBatch>) -> (Lake, Arc<Mutex<usize>>) {
        let races = Arc::new(Mutex::new(0));
        let store = Racing {
            inner: MemoryStore::new(),
            raced: raced.to_string(),
            races: races.clone(),
            rows,
        };
        (lake(Arc::new(store)), races)
    }

    #[async_trait]
    impl TableStore for Racing {
        fn location(&self) -> &str {
            self.inner.location()
        }

        async fn open(&self, name: &str) -> Result<Option<Box<dyn StoredTable>>> {
            Ok(self.inner.open(name).await?.map(|inner| {
                Box::new(RacingTable {
                    inner,
                    store: self.inner.clone(),
                    name: name.to_string(),
                    race: (name == self.raced).then(|| (self.races.clone(), self.rows.clone())),
                }) as Box<dyn StoredTable>
            }))
        }

        async fn create(
            &self,
            name: &str,
            batch: RecordBatch,
            options: &CreateOptions,
        ) -> Result<()> {
            self.inner.create(name, batch, options).await
        }
    }

    #[async_trait]
    impl StoredTable for RacingTable {
        fn schema(&self) -> SchemaRef {
            self.inner.schema()
        }

        async fn scan(
            &self,
            columns: Option<&[String]>,
            dnf: &Dnf,
            batch_size: usize,
        ) -> Result<TableScan> {
            if let Some((races, rows)) = &self.race {
                let race = {
                    let mut left = races.lock().unwrap();
                    left.checked_sub(1).map(|n| *left = n).is_some()
                };
                if race {
                    let other = self.store.open(&self.name).await?.unwrap();
                    match rows {
                        Some(rows) => other.append(rows.clone()).await?,
                        None => {
                            let column = other.schema().field(0).name().clone();
                            let metadata = HashMap::from([("k".into(), "v".into())]);
                            other.set_column_metadata(&column, metadata).await?;
                        }
                    }
                }
            }
            self.inner.scan(columns, dnf, batch_size).await
        }

        async fn append(self: Box<Self>, batch: RecordBatch) -> Result<()> {
            self.inner.append(batch).await
        }

        async fn append_merging(self: Box<Self>, batch: RecordBatch) -> Result<()> {
            self.inner.append_merging(batch).await
        }

        async fn merge(
            self: Box<Self>,
            source: RecordBatch,
            keys: &[String],
            update: &[String],
        ) -> Result<()> {
            self.inner.merge(source, keys, update).await
        }

        async fn set_column_metadata(
            self: Box<Self>,
            column: &str,
            metadata: HashMap<String, String>,
        ) -> Result<Box<dyn StoredTable>> {
            self.inner.set_column_metadata(column, metadata).await
        }

        async fn restore(self: Box<Self>, at: i64) -> Result<bool> {
            self.inner.restore(at).await
        }

        fn version(&self) -> Option<u64> {
            self.inner.version()
        }

        fn tag_commits(&mut self, tag: &str) {
            self.inner.tag_commits(tag)
        }

        async fn revert(self: Box<Self>, tag: &str, since: u64) -> Result<bool> {
            self.inner.revert(tag, since).await
        }

        async fn check_revert(&self, tag: &str, since: u64) -> Result<()> {
            self.inner.check_revert(tag, since).await
        }
    }

    /// `frame(ts)` with a new feature `y`, so that a write commits to the metadata tables too.
    fn with_y(ts: &[i64]) -> RecordBatch {
        let batch = frame(ts);
        let mut fields = batch.schema().fields().to_vec();
        fields.push(Arc::new(Field::new("y", DataType::Float64, true)));
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(Float64Array::from_iter_values(
            ts.iter().map(|&t| t as f64),
        )));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    /// The status and error of each row of the transaction log.
    fn log(lake: &Lake) -> Vec<(String, Option<String>)> {
        let path = &lake.options.transaction_log_path;
        let log = block_on(lake.read_batch(path)).unwrap().unwrap();
        let statuses = string_column(&log, "status").unwrap().into_iter().flatten();
        statuses
            .zip(string_column(&log, "error").unwrap())
            .collect()
    }

    /// The table of the latest version of each feature.
    fn tables(lake: &Lake) -> HashMap<String, Option<String>> {
        let path = &lake.options.feature_metadata_path;
        let rows = feature_rows(&block_on(lake.read_batch(path)).unwrap().unwrap()).unwrap();
        latest(&rows)
            .into_iter()
            .map(|(name, i)| (name, rows[i].table_addr.clone()))
            .collect()
    }

    #[test]
    fn a_write_commits_to_each_table_through_the_version_it_read_and_runs_again() {
        let defaults = LakeOptions::default();
        for raced in [
            defaults.index_path,
            defaults.feature_metadata_path,
            defaults.table_metadata_path,
        ] {
            let (lake, races) = racing(&raced, None);
            lake.add(vec![frame(&[1, 2])], &AddOptions::default())
                .unwrap();
            *races.lock().unwrap() = 1;
            lake.add(vec![with_y(&[3])], &AddOptions::default())
                .unwrap();
            assert_eq!(*races.lock().unwrap(), 0, "{raced} was not read");
            // The first attempt conflicted on the raced table and was rolled back.
            let log = log(&lake);
            let statuses: Vec<&str> = log.iter().map(|(s, _)| s.as_str()).collect();
            let retried = ["pending", "failed", "rolled_back", "pending", "success"];
            assert_eq!(statuses[2..], retried, "{raced}");
            let error = log[3].1.as_deref().unwrap();
            assert!(error.contains(&format!("table {raced} changed")), "{error}");
            let index = block_on(lake.read_batch(&lake.options.index_path)).unwrap();
            assert_eq!(index.unwrap().num_rows(), 3, "{raced}");
            assert!(tables(&lake).contains_key("y"), "{raced}");
        }
    }

    #[test]
    fn a_write_that_conflicts_on_every_attempt_returns_the_conflict() {
        let tm = LakeOptions::default().table_metadata_path;
        let (lake, races) = racing(&tm, None);
        lake.add(vec![frame(&[1, 2])], &AddOptions::default())
            .unwrap();
        *races.lock().unwrap() = usize::MAX;
        // A new feature over existing keys, as concurrent writers of different features write.
        let error = lake
            .add(vec![with_y(&[1, 2])], &AddOptions::default())
            .unwrap_err();
        assert!(
            matches!(&error, Error::Conflict { table } if *table == tm),
            "{error}"
        );
        let attempts = ATTEMPTS as usize;
        assert_eq!(usize::MAX - *races.lock().unwrap(), attempts);
        let statuses: Vec<String> = log(&lake).into_iter().map(|(s, _)| s).collect();
        let rolled_back = ["pending", "failed", "rolled_back"].repeat(attempts);
        assert_eq!(statuses[2..], rolled_back);
        assert!(!tables(&lake).contains_key("y"));
    }

    #[test]
    fn a_consolidation_whose_table_metadata_conflicts_runs_again() {
        let (lake, races) = racing(&LakeOptions::default().table_metadata_path, None);
        lake.add(vec![frame(&[1, 2])], &AddOptions::default())
            .unwrap();
        lake.add(vec![with_y(&[3])], &AddOptions::default())
            .unwrap();
        *races.lock().unwrap() = 1;
        let addr = lake.consolidate(&["x", "y"]).unwrap().unwrap();
        assert_eq!(*races.lock().unwrap(), 0);
        let tables = tables(&lake);
        for name in ["x", "y"] {
            assert_eq!(tables[name].as_deref(), Some(addr.as_str()), "{name}");
        }
    }

    #[test]
    fn a_runtime_computed_feature_registered_meanwhile_is_not_registered_again() {
        // Another writer registers `c` as soon as the feature metadata is read.
        let fm = LakeOptions::default().feature_metadata_path;
        let schema = lake(Arc::new(MemoryStore::new()))
            .feature_metadata_schema()
            .unwrap();
        let name = Arc::new(StringArray::from(vec!["c"])) as ArrayRef;
        let row = assemble(&schema, HashMap::from([("feature_name", name)]), 1, &fm).unwrap();
        let (lake, races) = racing(&fm, Some(row));
        lake.add(vec![frame(&[1, 2])], &AddOptions::default())
            .unwrap();
        *races.lock().unwrap() = 1;
        let expression = crate::Expression {
            name: "c".into(),
            expression: "{}".into(),
            roots: vec!["x".into()],
        };
        let error = lake.add_expressions(&[expression], "me").unwrap_err();
        assert!(
            matches!(&error, Error::InvalidArgument(m) if m.contains("already registered")),
            "{error}"
        );
    }

    #[test]
    fn a_conflict_beside_another_failure_is_not_reported_as_a_bare_conflict() {
        let store = Arc::new(MemoryStore::new());
        let lake = lake(store.clone());
        block_on(async {
            for name in ["a", "b"] {
                store
                    .create(name, frame(&[1]), &CreateOptions::default())
                    .await
                    .unwrap();
            }
            let a = store.open("a").await.unwrap().unwrap();
            let b = store.open("b").await.unwrap().unwrap();
            store
                .open("a")
                .await
                .unwrap()
                .unwrap()
                .append(frame(&[2]))
                .await
                .unwrap();
            let mismatched = frame(&[3]).project(&[0, 1]).unwrap();
            let actions = vec![
                Action::Append {
                    addr: "a".into(),
                    table: a,
                    batch: frame(&[3]),
                },
                Action::Append {
                    addr: "b".into(),
                    table: b,
                    batch: mismatched,
                },
            ];
            let error = commit(&lake, actions, None, vec![], &Commits::default())
                .await
                .unwrap_err();
            let Error::Write { source, .. } = &error else {
                panic!("{error}");
            };
            // The other failure is reported, so that a rolled back write is not retried.
            assert!(!matches!(**source, Error::Conflict { .. }), "{error}");
        })
        .unwrap()
    }
}
