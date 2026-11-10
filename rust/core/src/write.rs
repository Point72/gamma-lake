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
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    new_null_array, Array, ArrayRef, AsArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    UInt32Array,
};
use arrow::compute::{cast, concat, concat_batches, take, take_record_batch};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef, TimeUnit};
use arrow::row::{RowConverter, Rows, SortField};
use futures::future::join_all;
use futures::TryStreamExt;

use crate::admin::Annotation;
use crate::error::{Error, Result};
use crate::joined::AsOfParams;
use crate::lake::{feature_rows, latest, string_column, Lake, LakeOptions};
use crate::predicate::{lossless_cast, Dnf, Op, Term};
use crate::runtime::block_on;
use crate::scan::checked_scan;
use crate::store::{CreateOptions, StoredTable, TableStore};

/// Signal types whose tables never receive padding rows.
const UNPADDED_SIGNAL_TYPES: [&str; 2] = ["as_of_feature", "sparse_feature"];

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

impl SignalType {
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
    fn dense(&self) -> bool {
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

/// Options of [`Lake::add`]. The defaults match `GammaFeatureLake.add_features`.
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
}

impl Default for AddOptions {
    fn default() -> Self {
        Self {
            owner: "missing_owner".into(),
            signal_type: SignalType::Feature,
            overlap_mode: OverlapMode::Copy,
            feature_params: None,
            annotations: vec![],
        }
    }
}

/// Serializes writes to one lake within this process.
pub(crate) fn write_lock(location: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    LOCKS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(location.to_string())
        .or_default()
        .clone()
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::UnsupportedInput(message.into())
}

pub(crate) fn integrity(table_addr: &str, kind: impl Into<String>) -> Error {
    Error::Integrity {
        table_addr: table_addr.into(),
        kind: kind.into(),
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
        match newest.get(addr) {
            Some(&o) if stamps.value(o) >= stamps.value(r) => {}
            _ => {
                newest.insert(addr.clone(), r);
            }
        }
    }
    Ok(newest
        .into_iter()
        .filter(|&(_, r)| last.is_valid(r))
        .map(|(addr, r)| (addr, last.slice(r, 1)))
        .collect())
}

/// Microseconds since the Unix epoch.
pub(crate) fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or_default()
}

pub(crate) fn new_table_addr(prefix: &str) -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}delta_{}", &id[..8])
}

/// The Delta-compatible type new tables store for an input column of type `data_type`.
fn storage_type(data_type: &DataType) -> Option<DataType> {
    use DataType::*;
    Some(match data_type {
        Boolean | Int8 | Int16 | Int32 | Int64 | Float32 | Float64 | Date32 | Utf8 | Binary => {
            data_type.clone()
        }
        Decimal128(..) => data_type.clone(),
        // Stored as the signed type of the same width; values that do not fit are rejected.
        UInt8 => Int8,
        UInt16 => Int16,
        UInt32 => Int32,
        UInt64 => Int64,
        LargeUtf8 | Utf8View => Utf8,
        LargeBinary | BinaryView => Binary,
        Dictionary(_, value) => storage_type(value)?,
        Timestamp(_, None) => Timestamp(TimeUnit::Microsecond, None),
        Timestamp(_, Some(tz)) if tz.as_ref() == "UTC" => {
            Timestamp(TimeUnit::Microsecond, Some(tz.clone()))
        }
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

/// Whether `data_type` can be a sort key: a stored scalar that is not floating-point.
fn key_type_ok(data_type: &DataType) -> bool {
    !data_type.is_nested()
        && !is_float(data_type)
        && storage_type(data_type).as_ref() == Some(data_type)
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
    let mut order: Vec<u32> = (0..rows.num_rows() as u32).collect();
    order.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
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
            Action::Append { addr, .. } | Action::Create { addr, .. } => addr,
            Action::Merge { addr, .. } => addr,
        }
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

/// Coverage of a dense table: the newest table metadata row.
struct Coverage {
    bound: Bound,
}

struct Planner<'a> {
    lake: &'a Lake,
    pk_converter: RowConverter,
    input: Input,
    index: Index,
}

impl Planner<'_> {
    fn sort_keys(&self) -> &[String] {
        &self.lake.sort_keys
    }

    fn pk_name(&self) -> &str {
        &self.lake.options.primary_sort_key
    }

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
        let mut columns: HashMap<&str, ArrayRef> = HashMap::new();
        for field in schema.fields() {
            let name = field.name().as_str();
            let from_input = self
                .sort_keys()
                .iter()
                .any(|k| k == name)
                .then_some(true)
                .or_else(|| features.iter().any(|f| f == name).then_some(false));
            let Some(is_key) = from_input else { continue };
            let values = self.input.batch.column_by_name(name).expect("input column");
            let head = cast_into(&take(values, &input_rows, None)?, field, table)?;
            let tail = if is_key {
                let keys = self.index.batch.column_by_name(name).expect("index key");
                cast_into(&take(keys, &index_rows, None)?, field, table)?
            } else {
                new_null_array(field.data_type(), index.len())
            };
            columns.insert(name, concat(&[head.as_ref(), tail.as_ref()])?);
        }
        sort_by_keys(assemble(schema, columns, len, table)?, self.sort_keys())
    }

    /// `rows` whose primary key satisfies `keep`.
    fn select(rows: &[u32], pk: &Rows, keep: impl Fn(&[u8]) -> bool) -> Vec<u32> {
        rows.iter()
            .copied()
            .filter(|&r| keep(pk.row(r as usize).as_ref()))
            .collect()
    }

    fn bound(&self, value: ArrayRef, what: &str) -> Result<Bound> {
        let pk_type = self
            .lake
            .index_schema
            .field_with_name(self.pk_name())?
            .data_type();
        let value = lossless_cast(&value, pk_type).map_err(|why| integrity(what, why))?;
        let row = self
            .pk_converter
            .convert_columns(std::slice::from_ref(&value))?;
        Ok(Bound {
            row: row.row(0).as_ref().to_vec(),
            value,
        })
    }

    /// Rows of `table`, table `addr`, outside the primary key range `[min, max]`, in `schema`.
    async fn outside(
        &self,
        table: &dyn StoredTable,
        addr: &str,
        schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let pk = self.pk_name().to_string();
        let term = |op, bound: &Bound| Term {
            column: pk.clone(),
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

impl Lake {
    /// Creates the index and metadata tables of a lake in `store` where they are missing, as
    /// `GammaFeatureLake.initialize` does, and opens it.
    ///
    /// `index_schema` holds the sort keys; it must contain `options.primary_sort_key`.
    pub fn create_with_store(
        store: Arc<dyn TableStore>,
        index_schema: SchemaRef,
        options: LakeOptions,
    ) -> Result<Lake> {
        let store = crate::lake::locking(store, &options);
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
        for field in index_schema.fields() {
            if is_float(field.data_type()) {
                return Err(Error::InvalidArgument(format!(
                    "sort key {:?} is floating-point; sort keys are matched by equality, \
                     which is not reliable for floats",
                    field.name()
                )));
            }
            if !key_type_ok(field.data_type()) {
                return Err(unsupported(format!(
                    "index column {:?} has type {}, which the Rust core cannot store",
                    field.name(),
                    field.data_type()
                )));
            }
        }
        let lock = write_lock(store.location());
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

    /// Adds dense features or targets, as `GammaFeatureLake.add_features` and `add_targets`
    /// do. Non-key columns of `input` are the features; it must hold every sort key.
    ///
    /// Rows with a null key are dropped and, of rows sharing keys, the first is kept. Keys
    /// new to the index are added to it, and every dense table whose coverage includes them
    /// gets them as rows of nulls.
    ///
    /// Features new to the lake go to a new table. Features whose table covers none of
    /// the input keys already in the index are appended to it. Otherwise the input
    /// replaces the features over the input's primary key range: [`OverlapMode::Copy`]
    /// writes them with the table's rows outside that range to a new table, and
    /// [`OverlapMode::Merge`] upserts them into the table; both record a new version.
    ///
    /// Writes to one lake are serialized within this process. Writers in other processes
    /// must not run at the same time, and readers may observe a write half done. A commit to
    /// a table that another writer changed after this write read it fails; the write returns
    /// [`Error::Conflict`] when nothing was committed yet, and [`Error::Write`] otherwise.
    /// A write that adds no index rows does not commit to the index, so it can miss keys
    /// added by a concurrent writer without failing.
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
        self.write(input, Some(options))
    }

    /// Adds keys to the index without features, as `GammaFeatureLake.add_index_rows` does.
    /// Non-key columns of `input` are ignored.
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
        self.write(input, None)
    }

    fn write(&self, input: Vec<RecordBatch>, options: Option<&AddOptions>) -> Result<()> {
        let lock = write_lock(self.store.location());
        let _guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
        block_on(async {
            let (actions, index, metadata) = self.plan(input, options).await?;
            let committed = commit(self, actions, Some(index), metadata).await?;
            match options {
                Some(options) if !options.annotations.is_empty() => {
                    self.annotate_locked(&options.annotations, committed).await
                }
                _ => Ok(()),
            }
        })?
    }

    fn prepare(&self, input: Vec<RecordBatch>, pk_converter: &RowConverter) -> Result<Input> {
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
        let features: Vec<String> = schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| !self.sort_keys.contains(n))
            .collect();
        for name in &features {
            let field = schema.field_with_name(name)?;
            columns.push(batch.column_by_name(name).unwrap().clone());
            fields.push(Field::new(name, field.data_type().clone(), true));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;

        let nkeys = self.sort_keys.len();
        let key_converter = RowConverter::new(
            self.index_schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;
        let keys = key_converter.convert_columns(&batch.columns()[..nkeys])?;
        let mut kept = HashSet::new();
        let rows: Vec<u32> = (0..batch.num_rows())
            .filter(|&r| batch.columns()[..nkeys].iter().all(|c| c.is_valid(r)))
            .filter(|&r| kept.insert(keys.row(r)))
            .map(|r| r as u32)
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
        let (mut min, mut max) = (0, 0);
        for r in 1..pk.num_rows() {
            if pk.row(r) < pk.row(min) {
                min = r;
            }
            if pk.row(r) > pk.row(max) {
                max = r;
            }
        }
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
        options: Option<&AddOptions>,
    ) -> Result<(Vec<Action>, Action, Vec<Action>)> {
        let pk_name = self.options.primary_sort_key.clone();
        let pk_type = self
            .index_schema
            .field_with_name(&pk_name)?
            .data_type()
            .clone();
        let pk_converter = RowConverter::new(vec![SortField::new(pk_type)])?;
        let input = self.prepare(input, &pk_converter)?;
        // Index rows up to the input maximum, and how they relate to the input keys.
        let dnf = Dnf(vec![vec![Term {
            column: pk_name.clone(),
            op: Op::LtEq,
            values: input.max.value.clone(),
        }]]);
        // Each table is read and committed to through one handle, so a commit fails if another
        // writer committed to the table after this write read it.
        let index_table = self.table(&self.options.index_path).await?;
        let scan = checked_scan(index_table.as_ref(), None, &dnf, self.options.batch_size).await?;
        let index_batch =
            concat_batches(&scan.schema, &scan.stream.try_collect::<Vec<_>>().await?)?;
        let key_converter = RowConverter::new(
            self.index_schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
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
        let new: Vec<u32> = (0..input.keys.num_rows())
            .filter(|&r| !in_index.contains(&input.keys.row(r)))
            .map(|r| r as u32)
            .collect();
        let missing: Vec<u32> = (0..index_keys.num_rows())
            .filter(|&r| !in_input.contains(&index_keys.row(r)))
            .map(|r| r as u32)
            .collect();
        let pk_index = self.index_schema.index_of(&pk_name)?;
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
        planner.plan(options, index_table).await
    }
}

impl Planner<'_> {
    async fn plan(
        self,
        options: Option<&AddOptions>,
        index_table: Box<dyn StoredTable>,
    ) -> Result<(Vec<Action>, Action, Vec<Action>)> {
        let lake = self.lake;
        let fm_table = lake.table(&lake.options.feature_metadata_path).await?;
        let tm_table = lake.table(&lake.options.table_metadata_path).await?;
        let fm = lake.read_all(fm_table.as_ref()).await?;
        let tm = lake.read_all(tm_table.as_ref()).await?;
        let rows = feature_rows(&fm)?;
        let best = latest(&rows);

        // Input features grouped by the table holding their latest version.
        let mut groups: BTreeMap<Option<String>, Vec<String>> = BTreeMap::new();
        for name in &self.input.features {
            let addr = match best.get(name).map(|&i| &rows[i]) {
                None => None,
                Some(row) => {
                    let signal_type = row.signal_type.as_deref().unwrap_or_default();
                    if !crate::lake::READABLE_SIGNAL_TYPES.contains(&signal_type) {
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

        let signal_type = options.map(|o| o.signal_type).unwrap_or_default();
        if signal_type == SignalType::AsOf {
            // Reject parameters that every later read of these features would fail on.
            let pk = lake
                .index_schema
                .field_with_name(&lake.options.primary_sort_key)?
                .data_type();
            for name in &self.input.features {
                let params = match options.and_then(|o| o.feature_params.as_deref()) {
                    Some(p) => Some(p),
                    None => best
                        .get(name)
                        .and_then(|&i| rows[i].feature_params.as_deref()),
                };
                AsOfParams::parse(params.unwrap_or("{}"))?.check(pk)?;
            }
        }

        let mut coverage: BTreeMap<String, Coverage> = BTreeMap::new();
        for (addr, last) in coverage_values(&tm)? {
            let bound = self.bound(last, &addr)?;
            coverage.insert(addr, Coverage { bound });
        }

        // Tables that ever held an as-of or sparse feature are never padded.
        let types = string_column(&fm, "signal_type")?;
        let fm_addrs = string_column(&fm, "table_addr")?;
        let unpadded: HashSet<&str> = types
            .iter()
            .zip(&fm_addrs)
            .filter(|(t, _)| {
                t.as_deref()
                    .is_some_and(|t| UNPADDED_SIGNAL_TYPES.contains(&t))
            })
            .filter_map(|(_, a)| a.as_deref())
            .collect();

        let input_rows: Vec<u32> = (0..self.input.batch.num_rows() as u32).collect();
        let all_new = self.index.new.len() == input_rows.len();
        let (min, max) = (&self.input.min, &self.input.max);
        let mut actions = vec![];
        let mut tm_rows: Vec<(String, Bound)> = vec![];
        // (base feature metadata row, feature name, version, table address)
        let mut fm_rows: Vec<(Option<usize>, String, i64, String)> = vec![];
        let keys = self.sort_keys().to_vec();
        for (addr, features) in &groups {
            let Some(addr) = addr else {
                let addr = new_table_addr(&lake.options.table_name_prefix);
                let schema = self.new_schema(features).await?;
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
            let covered = coverage
                .get(addr)
                .ok_or_else(|| integrity(addr, "the table has no table metadata row"))?;
            let l = covered.bound.row.as_slice();
            let table = lake.table(addr).await?;
            let schema = table.schema();
            let newer = |a: &Bound, b: &Bound| if a.row >= b.row { a.clone() } else { b.clone() };
            let coverage_after = newer(&covered.bound, max);
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
            let target = match options.map(|o| o.overlap_mode).unwrap_or_default() {
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
                    let batch = sort_by_keys(concat_batches(&copy_schema, [&old, &fresh])?, &keys)?;
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
                fm_rows.push((Some(rows[best[f]].row), f.clone(), version, target.clone()));
            }
        }

        // Pad other dense tables with the new keys they cover.
        let homes: HashSet<&str> = groups.keys().flatten().map(String::as_str).collect();
        for (addr, covered) in &coverage {
            if homes.contains(addr.as_str()) || unpadded.contains(addr.as_str()) {
                continue;
            }
            let l = covered.bound.row.as_slice();
            let pad = Self::select(&self.index.new, &self.input.pk, |pk| pk <= l);
            if pad.is_empty() {
                continue;
            }
            let table = lake.table(addr).await?;
            let schema = table.schema();
            let batch = self.frame(&schema, &[], &pad, &[], addr)?;
            actions.push(Action::Append {
                addr: addr.clone(),
                table,
                batch,
            });
        }

        let index_schema = index_table.schema();
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
            let table = tm_table;
            let schema = table.schema();
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
                table,
                batch,
            });
        }
        if !fm_rows.is_empty() {
            let options = options.cloned().unwrap_or_default();
            let table = fm_table;
            let schema = table.schema();
            let n = fm_rows.len();
            let base = UInt32Array::from(
                fm_rows
                    .iter()
                    .map(|(r, ..)| r.map(|r| r as u32))
                    .collect::<Vec<_>>(),
            );
            let params = string_column(&fm, "feature_params").unwrap_or_default();
            let params = fm_rows
                .iter()
                .map(|(r, ..)| match (&options.feature_params, r) {
                    (Some(p), _) => Some(p.clone()),
                    (None, None) => Some("{}".to_string()),
                    (None, Some(r)) => params.get(*r).cloned().flatten(),
                });
            let mut columns: HashMap<&str, ArrayRef> = HashMap::from([
                (
                    "feature_name",
                    Arc::new(StringArray::from_iter_values(fm_rows.iter().map(|r| &r.1)))
                        as ArrayRef,
                ),
                (
                    "version",
                    Arc::new(arrow::array::Int64Array::from_iter_values(
                        fm_rows.iter().map(|r| r.2),
                    )),
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
                table,
                batch,
            });
        }
        Ok((actions, index, metadata))
    }

    /// Schema of a new table: the index columns, then `features` as the input holds them.
    async fn new_schema(&self, features: &[String]) -> Result<SchemaRef> {
        let index = self.lake.table(&self.lake.options.index_path).await?;
        let index = index.schema();
        let mut fields: Vec<Field> = self
            .sort_keys()
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

async fn run(action: Action, store: &dyn TableStore, keys: &[String]) -> Result<()> {
    match action {
        Action::Append { table, batch, .. } => {
            if batch.num_rows() > 0 {
                table.append(batch).await?;
            }
        }
        Action::Create { addr, batch } => {
            let options = CreateOptions {
                stats_columns: keys.to_vec(),
            };
            store.create(&addr, batch, &options).await?;
        }
        Action::Merge {
            table,
            source,
            features,
            ..
        } => table.merge(source, keys, &features).await?,
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

/// Commits the feature tables concurrently, then the index, then the metadata tables, and
/// returns the addresses of the tables committed. A failure is an [`Error::Write`], unless
/// nothing was committed and every failure was a conflict; the first conflict is then
/// returned as is. Other errors may come after a commit, so they never count as conflicts.
pub(crate) async fn commit(
    lake: &Lake,
    actions: Vec<Action>,
    index: Option<Action>,
    metadata: Vec<Action>,
) -> Result<Vec<String>> {
    let run = |action: Action| async move {
        let addr = action.addr().to_string();
        (
            addr,
            run(action, lake.store.as_ref(), &lake.sort_keys).await,
        )
    };
    let mut committed = vec![];
    let mut failure = None;
    let mut only_conflicts = true;
    for (addr, result) in join_all(actions.into_iter().map(run)).await {
        match result {
            Ok(()) => committed.push(addr),
            Err(e) => {
                let e = written(e, &mut committed);
                only_conflicts &= matches!(e, Error::Conflict { .. });
                if failure.is_none() {
                    failure = Some((format!("writing table {addr}"), e));
                }
            }
        }
    }
    for action in index.into_iter().chain(metadata) {
        if failure.is_some() {
            break;
        }
        let (addr, result) = run(action).await;
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
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::store::MemoryStore;
    use crate::store::TableScan;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use async_trait::async_trait;

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

    /// Plans a write of a new key for an existing feature, lets another writer commit to its
    /// feature table or else the index, then commits the plan. Returns the error and the number
    /// of index rows after it.
    fn raced_write(feature: bool) -> (Error, usize) {
        let index = Schema::new(vec![
            Field::new("ts", DataType::Int64, true),
            Field::new("sym", DataType::Utf8, true),
        ]);
        let options = LakeOptions {
            primary_sort_key: "ts".into(),
            ..Default::default()
        };
        let store = Arc::new(MemoryStore::new());
        let lake = Lake::create_with_store(store, Arc::new(index), options).unwrap();
        lake.add(vec![frame(&[1, 2])], &AddOptions::default())
            .unwrap();
        block_on(async {
            let (actions, index, metadata) = lake.plan(vec![frame(&[3])], None).await.unwrap();
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
            let error = commit(&lake, actions, Some(index), metadata)
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

    /// A store in which, once `armed`, another writer commits to table `raced` as soon as a
    /// write reads it.
    #[derive(Debug)]
    struct Racing {
        inner: MemoryStore,
        raced: String,
        armed: Arc<AtomicBool>,
    }

    struct RacingTable {
        inner: Box<dyn StoredTable>,
        store: MemoryStore,
        name: String,
        armed: Option<Arc<AtomicBool>>,
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
                    armed: (name == self.raced).then(|| self.armed.clone()),
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
            if self
                .armed
                .as_ref()
                .is_some_and(|a| a.swap(false, Ordering::SeqCst))
            {
                let other = self.store.open(&self.name).await?.unwrap();
                let column = other.schema().field(0).name().clone();
                other
                    .set_column_metadata(&column, HashMap::from([("k".into(), "v".into())]))
                    .await?;
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
    }

    #[test]
    fn a_write_commits_to_each_table_through_the_version_it_read() {
        let defaults = LakeOptions::default();
        for raced in [
            defaults.index_path,
            defaults.feature_metadata_path,
            defaults.table_metadata_path,
        ] {
            let armed = Arc::new(AtomicBool::new(false));
            let store = Racing {
                inner: MemoryStore::new(),
                raced: raced.clone(),
                armed: armed.clone(),
            };
            let index = Schema::new(vec![
                Field::new("ts", DataType::Int64, true),
                Field::new("sym", DataType::Utf8, true),
            ]);
            let options = LakeOptions {
                primary_sort_key: "ts".into(),
                ..Default::default()
            };
            let lake = Lake::create_with_store(Arc::new(store), Arc::new(index), options).unwrap();
            lake.add(vec![frame(&[1, 2])], &AddOptions::default())
                .unwrap();
            // A new feature, so that the write commits to the metadata tables too.
            let batch = frame(&[3]);
            let mut fields = batch.schema().fields().to_vec();
            fields.push(Arc::new(Field::new("y", DataType::Float64, true)));
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(Float64Array::from(vec![3.0])));
            let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
            armed.store(true, Ordering::SeqCst);
            let error = lake.add(vec![batch], &AddOptions::default()).unwrap_err();
            assert!(!armed.load(Ordering::SeqCst), "{raced} was not read");
            let conflict = match &error {
                Error::Write { source, .. } => source.as_ref(),
                other => other,
            };
            assert!(
                matches!(conflict, Error::Conflict { table } if *table == raced),
                "{raced}: {error}"
            );
        }
    }

    #[test]
    fn a_conflict_beside_another_failure_is_not_reported_as_a_bare_conflict() {
        let store = Arc::new(MemoryStore::new());
        let index = Schema::new(vec![
            Field::new("ts", DataType::Int64, true),
            Field::new("sym", DataType::Utf8, true),
        ]);
        let options = LakeOptions {
            primary_sort_key: "ts".into(),
            ..Default::default()
        };
        let lake = Lake::create_with_store(store.clone(), Arc::new(index), options).unwrap();
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
            let error = commit(&lake, actions, None, vec![]).await.unwrap_err();
            let Error::Write { source, .. } = &error else {
                panic!("{error}");
            };
            assert!(matches!(**source, Error::Conflict { .. }), "{error}");
        })
        .unwrap()
    }
}
