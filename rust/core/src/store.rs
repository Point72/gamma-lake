//! Where a lake keeps its tables.
//!
//! A [`Lake`](crate::Lake) reads and writes its index, metadata and feature tables through a
//! [`TableStore`]. [`MemoryStore`] keeps them in memory; other crates implement persistent
//! stores, using the helpers here to check table names, schemas and restore times the same way.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Component, Path};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::{new_null_array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::{cast, concat_batches, interleave, take};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::row::{RowConverter, SortField};
use async_trait::async_trait;
use futures::Stream;

use crate::error::{Error, Result};
use crate::predicate::Dnf;

/// An asynchronous stream of Arrow record batches returned by a storage backend.
pub type TableStream = Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>;

/// One storage scan and the schema of every batch it yields.
pub struct TableScan {
    pub schema: SchemaRef,
    pub stream: TableStream,
}

/// Options of [`TableStore::create`].
#[derive(Clone, Debug, Default)]
pub struct CreateOptions {
    /// Columns to collect file statistics on for data skipping. Empty means the store's
    /// default.
    pub stats_columns: Vec<String>,
}

/// The tables of one lake, addressed by relative names such as `index` or `features/f1`.
///
/// Commits are per table. A lake write that changes several tables commits them one at a
/// time, so a failure part-way leaves the earlier ones committed; the lake reports those in
/// [`Error::Write`].
///
/// A [`StoredTable`] mutation commits only if its table is still at the handle's version.
/// When another commit came first, it writes nothing and fails with [`Error::Conflict`].
///
/// A write ([`TableStore::create`] or a [`StoredTable`] mutation) that fails before
/// committing may return any error. One that fails after it may have committed, for example
/// because the commit was sent but its outcome is unknown, returns [`Error::Write`] with the
/// table's name in `committed`.
#[async_trait]
pub trait TableStore: Send + Sync + fmt::Debug {
    /// Where the tables are, for messages. Writes through one [`Lake`](crate::Lake) process
    /// are serialized per location.
    fn location(&self) -> &str;

    /// The key of table `name` for a [`TableLock`](crate::TableLock): the same for every
    /// store that addresses that table.
    fn table_location(&self, name: &str) -> Result<String> {
        Ok(format!("{}/{name}", self.location().trim_end_matches('/')))
    }

    /// The latest version of table `name`, or `None` when no such table exists.
    async fn open(&self, name: &str) -> Result<Option<Box<dyn StoredTable>>>;

    /// Creates table `name` with `batch` as its first version; fails if the table exists.
    async fn create(&self, name: &str, batch: RecordBatch, options: &CreateOptions) -> Result<()>;
}

/// One version of a stored table. Reads see that version even after later commits.
/// Mutations commit only while this version is the latest, and otherwise fail with
/// [`Error::Conflict`].
#[async_trait]
pub trait StoredTable: Send + Sync {
    /// The schema the table stores, which appended batches must match.
    fn schema(&self) -> SchemaRef;

    /// Scans this version as Arrow batches.
    ///
    /// `columns = None` reads every column. The scan and each of its batches have the names and
    /// types of [`StoredTable::schema`] projected to `columns`. The scan must apply `dnf`
    /// exactly and may return matching rows in any order, in batches of at most `batch_size`
    /// rows; `batch_size` is positive.
    async fn scan(
        &self,
        columns: Option<&[String]>,
        dnf: &Dnf,
        batch_size: usize,
    ) -> Result<TableScan>;

    /// Commits `batch` as new rows. `batch` has the columns of [`StoredTable::schema`].
    async fn append(self: Box<Self>, batch: RecordBatch) -> Result<()>;

    /// Appends `batch`, whose columns are the table's followed by new nullable columns, in
    /// one commit that adds those columns. Earlier rows read them as null.
    async fn append_merging(self: Box<Self>, batch: RecordBatch) -> Result<()>;

    /// Upserts `source` on the `keys` columns. Matched rows take the `update` columns of
    /// `source` and keep the rest; unmatched source rows are inserted, with nulls in columns
    /// `source` lacks. `source` holds `keys`, `update` and possibly other table columns, with
    /// at most one row per key.
    async fn merge(
        self: Box<Self>,
        source: RecordBatch,
        keys: &[String],
        update: &[String],
    ) -> Result<()>;

    /// Sets `metadata` keys on column `column`, keeping its other keys and the other
    /// columns, and returns the table at the new version.
    async fn set_column_metadata(
        self: Box<Self>,
        column: &str,
        metadata: HashMap<String, String>,
    ) -> Result<Box<dyn StoredTable>>;

    /// Makes the latest version a copy of the newest version committed at or before `at`,
    /// in microseconds since the Unix epoch. Does nothing and returns `false` when that is
    /// already this version. Fails when every version is newer than `at`.
    async fn restore(self: Box<Self>, at: i64) -> Result<bool>;

    /// This version's number: 0 for the table's first version, then one more per commit, or
    /// `None` when the store does not number versions. A lake rolls back a failed write only
    /// on tables whose versions are numbered.
    fn version(&self) -> Option<u64> {
        None
    }

    /// Marks the commits made through this handle, and through the handle
    /// [`StoredTable::set_column_metadata`] returns, with `tag`, so that
    /// [`StoredTable::revert`] can find them. Stores that cannot mark commits ignore it.
    fn tag_commits(&mut self, _tag: &str) {}

    /// Undoes the commits marked `tag` made after version `since`, keeping every other
    /// commit, in one commit on this version, and returns whether any were undone, now or by
    /// an earlier revert of `tag`. Fails with [`entangled`], without committing, when another
    /// commit rewrote what a marked one wrote, since undoing the marked one would undo it too.
    /// Stores that do not number versions cannot.
    async fn revert(self: Box<Self>, tag: &str, since: u64) -> Result<bool> {
        let _ = (tag, since);
        Err(Error::InvalidArgument(
            "this store does not number versions, so it cannot undo commits".into(),
        ))
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "version {version} of table {table}, another writer's, rewrote what this write committed, \
     so undoing this write would undo that version too"
)]
pub(crate) struct Entangled {
    table: String,
    version: u64,
}

/// The error of [`StoredTable::revert`] on table `table`, whose commit `version`, not one to
/// undo, rewrote what one to undo wrote.
pub fn entangled(table: &str, version: u64) -> Error {
    Error::Store(Box::new(Entangled {
        table: table.to_string(),
        version,
    }))
}

/// The newest of versions `earliest..=latest` committed at or before `at`, given each
/// version's commit time. Commit times must not decrease with the version; equal times are
/// allowed.
pub async fn version_at<F, Fut>(
    earliest: u64,
    latest: u64,
    at: i64,
    committed: F,
) -> Result<Option<u64>>
where
    F: Fn(u64) -> Fut,
    Fut: std::future::Future<Output = Result<i64>>,
{
    // Rightmost version whose commit time is <= at.
    let (mut lo, mut hi) = (earliest, latest + 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if committed(mid).await? <= at {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok((lo > earliest).then(|| lo - 1))
}

/// The error of [`StoredTable::restore`] when every retained version of `table` is newer than
/// `at`.
pub fn before_first_commit(table: &str, at: i64) -> Error {
    Error::InvalidArgument(format!(
        "table {table} has no retained version committed at or before {at} microseconds since \
         the epoch"
    ))
}

/// Checks that `name` is a plain relative path and returns its segments. Names containing
/// `%` are rejected.
pub fn segments(name: &str) -> Result<Vec<&str>> {
    let invalid = || {
        Error::InvalidArgument(format!(
            "table address {name:?} must be a relative path inside the lake"
        ))
    };
    let plain = Path::new(name)
        .components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    if name.is_empty() || !plain || name.contains('\\') {
        return Err(invalid());
    }
    // delta-rs decodes percent escapes twice, so `%` would address another table.
    if name.contains('%') {
        return Err(Error::InvalidArgument(format!(
            "table address {name:?} contains '%', which is not supported"
        )));
    }
    let parts: Vec<&str> = name
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    if parts.is_empty() {
        return Err(invalid());
    }
    Ok(parts)
}

/// A commit that conflicts with another commit made since the table was opened.
/// Tables held in memory, for tests and short-lived lakes. Clones share the tables.
#[derive(Clone)]
pub struct MemoryStore {
    location: String,
    tables: Arc<Mutex<HashMap<String, Arc<MemoryVersion>>>>,
}

impl fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryStore")
            .field("location", &self.location)
            .finish_non_exhaustive()
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore {
    /// An empty store.
    pub fn new() -> Self {
        Self {
            location: format!("memory://{}", uuid::Uuid::new_v4()),
            tables: Default::default(),
        }
    }

    fn tables(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<MemoryVersion>>> {
        self.tables.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Replaces version `base` of `name` with `f` of it, marked `tag`, if `base` is still the
    /// latest.
    fn commit(
        &self,
        name: &str,
        base: &Arc<MemoryVersion>,
        tag: Option<&str>,
        f: impl FnOnce(&MemoryVersion) -> Result<(SchemaRef, Vec<RecordBatch>)>,
    ) -> Result<Arc<MemoryVersion>> {
        let mut tables = self.tables();
        let latest = tables
            .get(name)
            .ok_or_else(|| Error::InvalidArgument(format!("table {name:?} was removed")))?
            .clone();
        if !Arc::ptr_eq(&latest, base) {
            return Err(Error::Conflict {
                table: name.to_string(),
            });
        }
        let (schema, batches) = f(&latest)?;
        let version = Arc::new(MemoryVersion {
            schema,
            batches,
            number: latest.number + 1,
            committed: now_micros(),
            tag: tag.map(str::to_string),
            previous: Some(latest),
        });
        tables.insert(name.to_string(), version.clone());
        Ok(version)
    }
}

/// Microseconds since the Unix epoch.
fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or_default()
}

struct MemoryVersion {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    number: u64,
    /// Commit time in microseconds since the Unix epoch.
    committed: i64,
    /// The mark of [`StoredTable::tag_commits`] the commit was made with.
    tag: Option<String>,
    previous: Option<Arc<MemoryVersion>>,
}

impl MemoryVersion {
    /// This version and the older ones, newest first.
    fn history(self: &Arc<Self>) -> impl Iterator<Item = &Arc<MemoryVersion>> {
        std::iter::successors(Some(self), |v| v.previous.as_ref())
    }
}

/// Checks that `wider` is `schema` followed by new nullable columns, as
/// [`StoredTable::append_merging`] requires of its batch.
pub fn extends(schema: &SchemaRef, wider: &SchemaRef, table: &str) -> Result<()> {
    let (old, new) = wider
        .fields()
        .split_at(schema.fields().len().min(wider.fields().len()));
    let prefix = old.len() == schema.fields().len()
        && old
            .iter()
            .zip(schema.fields())
            .all(|(a, b)| a.name() == b.name() && a.data_type().equals_datatype(b.data_type()));
    let fresh = new
        .iter()
        .all(|f| f.is_nullable() && schema.field_with_name(f.name()).is_err());
    if !prefix || !fresh {
        return Err(Error::InvalidArgument(format!(
            "batch schema {wider:?} does not extend table {table} {schema:?} with new nullable \
             columns"
        )));
    }
    Ok(())
}

/// `batch` with the columns of `schema`, which must have the same names and types.
fn conform(schema: &SchemaRef, batch: RecordBatch, table: &str) -> Result<RecordBatch> {
    let fields = batch.schema_ref().fields();
    let matches = fields.len() == schema.fields().len()
        && fields
            .iter()
            .zip(schema.fields())
            .all(|(a, b)| a.name() == b.name() && a.data_type().equals_datatype(b.data_type()));
    if !matches {
        return Err(Error::InvalidArgument(format!(
            "batch schema {:?} does not match table {table} {:?}",
            batch.schema(),
            schema
        )));
    }
    Ok(RecordBatch::try_new(
        schema.clone(),
        batch.columns().to_vec(),
    )?)
}

#[async_trait]
impl TableStore for MemoryStore {
    fn location(&self) -> &str {
        &self.location
    }

    async fn open(&self, name: &str) -> Result<Option<Box<dyn StoredTable>>> {
        segments(name)?;
        Ok(self.tables().get(name).map(|version| {
            Box::new(MemoryStored {
                store: self.clone(),
                name: name.to_string(),
                version: version.clone(),
                tag: None,
            }) as Box<dyn StoredTable>
        }))
    }

    async fn create(&self, name: &str, batch: RecordBatch, _options: &CreateOptions) -> Result<()> {
        segments(name)?;
        let mut tables = self.tables();
        if tables.contains_key(name) {
            return Err(Error::InvalidArgument(format!(
                "table {name:?} already exists"
            )));
        }
        let schema = batch.schema();
        tables.insert(
            name.to_string(),
            Arc::new(MemoryVersion {
                schema,
                batches: vec![batch],
                number: 0,
                committed: now_micros(),
                tag: None,
                previous: None,
            }),
        );
        Ok(())
    }
}

struct MemoryStored {
    store: MemoryStore,
    name: String,
    version: Arc<MemoryVersion>,
    tag: Option<String>,
}

#[async_trait]
impl StoredTable for MemoryStored {
    fn schema(&self) -> SchemaRef {
        self.version.schema.clone()
    }

    async fn scan(
        &self,
        columns: Option<&[String]>,
        dnf: &Dnf,
        batch_size: usize,
    ) -> Result<TableScan> {
        if batch_size == 0 {
            return Err(Error::InvalidArgument("batch_size must be positive".into()));
        }
        // Scans yield the types Delta scans do, so both stores read the same.
        let scan_type = |t: &DataType| match t {
            DataType::Utf8 | DataType::LargeUtf8 => DataType::Utf8View,
            DataType::Binary | DataType::LargeBinary => DataType::BinaryView,
            other => other.clone(),
        };
        let schema = Arc::new(Schema::new(
            self.version
                .schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone().with_data_type(scan_type(f.data_type())))
                .collect::<Vec<_>>(),
        ));
        let positions = match columns {
            Some(columns) => columns
                .iter()
                .map(|name| self.version.schema.index_of(name))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            None => (0..self.version.schema.fields().len()).collect(),
        };
        let output_schema = Arc::new(self.version.schema.project(&positions)?);
        let batches = self.version.batches.iter().try_fold(
            Vec::new(),
            |mut output, batch| -> Result<Vec<RecordBatch>> {
                let columns = batch
                    .columns()
                    .iter()
                    .zip(schema.fields())
                    .map(|(c, f)| cast(c, f.data_type()))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let batch = RecordBatch::try_new(schema.clone(), columns)?;
                let batch = dnf.filter_batch(&batch)?;
                let columns = positions
                    .iter()
                    .zip(output_schema.fields())
                    .map(|(&position, field)| {
                        cast(batch.column(position), field.data_type()).map_err(Error::from)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let batch = RecordBatch::try_new(output_schema.clone(), columns)?;
                for offset in (0..batch.num_rows()).step_by(batch_size) {
                    output.push(batch.slice(offset, batch_size.min(batch.num_rows() - offset)));
                }
                Ok(output)
            },
        )?;
        Ok(TableScan {
            schema: output_schema,
            stream: Box::pin(futures::stream::iter(batches.into_iter().map(Ok))),
        })
    }

    async fn append(self: Box<Self>, batch: RecordBatch) -> Result<()> {
        let batch = conform(&self.version.schema, batch, &self.name)?;
        self.store
            .commit(&self.name, &self.version, self.tag.as_deref(), |latest| {
                let mut batches = latest.batches.clone();
                batches.push(batch);
                Ok((latest.schema.clone(), batches))
            })?;
        Ok(())
    }

    async fn append_merging(self: Box<Self>, batch: RecordBatch) -> Result<()> {
        extends(&self.version.schema, batch.schema_ref(), &self.name)?;
        self.store
            .commit(&self.name, &self.version, self.tag.as_deref(), |latest| {
                // The new schema is the batch's wider one. Every existing batch is padded with
                // all-null arrays for the added columns (`schema.fields()[n..]` is the tail
                // slice beyond the old column count), then the new batch is appended.
                let schema = batch.schema();
                let mut batches = latest
                    .batches
                    .iter()
                    .map(|old| {
                        let mut columns = old.columns().to_vec();
                        for field in &schema.fields()[old.num_columns()..] {
                            columns.push(new_null_array(field.data_type(), old.num_rows()));
                        }
                        Ok(RecordBatch::try_new(schema.clone(), columns)?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                batches.push(batch);
                Ok((schema, batches))
            })?;
        Ok(())
    }

    async fn merge(
        self: Box<Self>,
        source: RecordBatch,
        keys: &[String],
        update: &[String],
    ) -> Result<()> {
        self.store
            .commit(&self.name, &self.version, self.tag.as_deref(), |latest| {
                let batches =
                    merge_batches(&latest.schema, &latest.batches, &source, keys, update)?;
                Ok((latest.schema.clone(), batches))
            })?;
        Ok(())
    }

    async fn set_column_metadata(
        self: Box<Self>,
        column: &str,
        metadata: HashMap<String, String>,
    ) -> Result<Box<dyn StoredTable>> {
        let name = self.name.clone();
        let version = self
            .store
            .commit(&name, &self.version, self.tag.as_deref(), |latest| {
                // Copy each field, merging `metadata` into the matching column's existing map.
                let fields = latest
                    .schema
                    .fields()
                    .iter()
                    .map(|f| {
                        let mut f = f.as_ref().clone();
                        if f.name() == column {
                            f.metadata_mut().extend(metadata.clone());
                        }
                        f
                    })
                    .collect::<Vec<_>>();
                if latest.schema.field_with_name(column).is_err() {
                    return Err(Error::InvalidArgument(format!(
                        "table {name} has no column {column:?}"
                    )));
                }
                // Keep the table-level metadata; rebuild every batch onto the new schema.
                let schema = Arc::new(Schema::new_with_metadata(
                    fields,
                    latest.schema.metadata().clone(),
                ));
                let batches = latest
                    .batches
                    .iter()
                    .map(|b| conform(&schema, b.clone(), &name))
                    .collect::<Result<Vec<_>>>()?;
                Ok((schema, batches))
            })?;
        Ok(Box::new(MemoryStored {
            store: self.store,
            name,
            version,
            tag: self.tag,
        }))
    }

    async fn restore(self: Box<Self>, at: i64) -> Result<bool> {
        let Some(target) = self.version.history().find(|v| v.committed <= at).cloned() else {
            return Err(before_first_commit(&self.name, at));
        };
        if Arc::ptr_eq(&target, &self.version) {
            return Ok(false);
        }
        self.store
            .commit(&self.name, &self.version, self.tag.as_deref(), |_| {
                Ok((target.schema.clone(), target.batches.clone()))
            })?;
        Ok(true)
    }

    fn version(&self) -> Option<u64> {
        Some(self.version.number)
    }

    fn tag_commits(&mut self, tag: &str) {
        self.tag = Some(tag.to_string());
    }

    async fn revert(self: Box<Self>, tag: &str, since: u64) -> Result<bool> {
        // The versions after `since`, oldest first, each with the one before it.
        let mut versions: Vec<_> = self
            .version
            .history()
            .take_while(|v| v.number > since)
            .filter_map(|v| Some((v, v.previous.as_ref()?)))
            .collect();
        versions.reverse();
        let ours = |v: &MemoryVersion| v.tag.as_deref() == Some(tag);
        let Some(first) = versions.iter().position(|(v, _)| ours(v)) else {
            return Ok(false);
        };
        let versions = &versions[first..];
        let base = versions[0].1;
        if versions.iter().all(|(v, _)| ours(v)) {
            self.store.commit(&self.name, &self.version, None, |_| {
                Ok((base.schema.clone(), base.batches.clone()))
            })?;
            return Ok(true);
        }
        // A commit that keeps a batch keeps its first column.
        let id = |b: &RecordBatch| Arc::as_ptr(b.column(0)) as *const () as usize;
        let ids = |v: &MemoryVersion| v.batches.iter().map(id).collect::<HashSet<_>>();
        // The batches the marked commits added and removed, and whether they changed the
        // schema and its columns.
        let mut added = HashSet::new();
        let mut removed: Vec<(usize, RecordBatch)> = vec![];
        let (mut reschemed, mut recolumned) = (false, false);
        // The first commit not marked that changed the schema.
        let mut rescheming = None;
        for (v, previous) in versions {
            let (after, before) = (ids(v), ids(previous));
            if ours(v) {
                for batch in &v.batches {
                    let key = id(batch);
                    if !before.contains(&key) && !removed.iter().any(|(k, _)| *k == key) {
                        added.insert(key);
                    }
                }
                removed.retain(|(k, _)| !after.contains(k));
                for batch in &previous.batches {
                    let key = id(batch);
                    if !after.contains(&key) && !added.remove(&key) {
                        removed.push((key, batch.clone()));
                    }
                }
                reschemed |= v.schema != previous.schema;
                recolumned |= v
                    .schema
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .ne(previous.schema.fields().iter().map(|field| field.name()));
            } else if added.iter().any(|k| !after.contains(k))
                // Its rows may fill columns the marked commits added.
                || (recolumned && after.iter().any(|k| !before.contains(k)))
            {
                return Err(entangled(&self.name, v.number));
            } else if v.schema != previous.schema {
                rescheming.get_or_insert(v.number);
            }
        }
        if let Some(number) = rescheming.filter(|_| reschemed) {
            return Err(entangled(&self.name, number));
        }
        let schema = if reschemed {
            base.schema.clone()
        } else {
            self.version.schema.clone()
        };
        let latest = ids(&self.version);
        let kept = self
            .version
            .batches
            .iter()
            .filter(|b| !added.contains(&id(b)));
        let restored = removed
            .iter()
            .filter(|(k, _)| !latest.contains(k))
            .map(|(_, b)| b);
        let batches = kept
            .chain(restored)
            .map(|b| fit(&schema, b))
            .collect::<Result<Vec<_>>>()?;
        self.store
            .commit(&self.name, &self.version, None, |_| Ok((schema, batches)))?;
        Ok(true)
    }
}

/// `batch` with the columns of `schema` by name; columns it lacks are null.
fn fit(schema: &SchemaRef, batch: &RecordBatch) -> Result<RecordBatch> {
    let columns = schema
        .fields()
        .iter()
        .map(|f| match batch.column_by_name(f.name()) {
            Some(column) => column.clone(),
            None => new_null_array(f.data_type(), batch.num_rows()),
        })
        .collect();
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

/// The rows of `batches` with `source` merged in as [`StoredTable::merge`] describes.
fn merge_batches(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    source: &RecordBatch,
    keys: &[String],
    update: &[String],
) -> Result<Vec<RecordBatch>> {
    let target = concat_batches(schema, batches)?;
    let key_columns = |batch: &RecordBatch| -> Result<Vec<ArrayRef>> {
        keys.iter()
            .map(|k| {
                batch.column_by_name(k).cloned().ok_or_else(|| {
                    Error::InvalidArgument(format!("merge source lacks key column {k:?}"))
                })
            })
            .collect()
    };
    let converter = RowConverter::new(
        keys.iter()
            .map(|k| {
                Ok(SortField::new(
                    schema.field_with_name(k)?.data_type().clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?,
    )?;
    let target_rows = converter.convert_columns(&key_columns(&target)?)?;
    let source_rows = converter.convert_columns(&key_columns(source)?)?;
    let mut by_key = HashMap::with_capacity(source.num_rows());
    for i in 0..source.num_rows() {
        if by_key.insert(source_rows.row(i), i).is_some() {
            return Err(Error::InvalidArgument(
                "merge source has two rows with the same key".into(),
            ));
        }
    }
    let matches: Vec<Option<usize>> = (0..target.num_rows())
        .map(|i| by_key.get(&target_rows.row(i)).copied())
        .collect();
    let matched: HashSet<usize> = matches.iter().flatten().copied().collect();
    let inserted = UInt32Array::from_iter_values(
        (0..source.num_rows())
            .filter(|i| !matched.contains(i))
            .map(|i| i as u32),
    );
    let update: HashSet<&str> = update.iter().map(String::as_str).collect();
    let mut updated = vec![];
    let mut new = vec![];
    for field in schema.fields() {
        let old = target.column_by_name(field.name()).unwrap();
        let incoming = source.column_by_name(field.name());
        updated.push(match incoming {
            Some(incoming) if update.contains(field.name().as_str()) => {
                let indices: Vec<(usize, usize)> = matches
                    .iter()
                    .enumerate()
                    .map(|(i, m)| m.map_or((0, i), |j| (1, j)))
                    .collect();
                interleave(&[old.as_ref(), incoming.as_ref()], &indices)?
            }
            _ => old.clone(),
        });
        new.push(match incoming {
            Some(incoming) => take(incoming, &inserted, None)?,
            None => new_null_array(field.data_type(), inserted.len()),
        });
    }
    let mut out = vec![RecordBatch::try_new(schema.clone(), updated)?];
    if !inserted.is_empty() {
        out.push(RecordBatch::try_new(schema.clone(), new)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_memory_scan_rejects_an_empty_batch_size() {
        let store = MemoryStore::new();
        let error = futures::executor::block_on(async {
            let batch = RecordBatch::try_from_iter([(
                "id",
                Arc::new(UInt32Array::from(vec![1])) as ArrayRef,
            )])?;
            store.create("t", batch, &CreateOptions::default()).await?;
            store
                .open("t")
                .await?
                .unwrap()
                .scan(None, &Dnf::default(), 0)
                .await
                .map(|_| ())
        })
        .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(_)), "{error}");
    }

    #[test]
    fn version_at_finds_the_newest_of_equal_times() {
        let times = [10, 20, 20, 20, 30];
        let at =
            |t: i64| {
                futures::executor::block_on(version_at(0, 4, t, |v| async move {
                    Ok(times[v as usize])
                }))
                .unwrap()
            };
        assert_eq!(at(5), None);
        assert_eq!(at(10), Some(0));
        assert_eq!(at(19), Some(0));
        assert_eq!(at(20), Some(3));
        assert_eq!(at(25), Some(3));
        assert_eq!(at(30), Some(4));
        assert_eq!(at(99), Some(4));
        let retained =
            |t: i64| {
                futures::executor::block_on(version_at(2, 4, t, |v| async move {
                    Ok(times[v as usize])
                }))
                .unwrap()
            };
        assert_eq!(retained(15), None);
        assert_eq!(retained(20), Some(3));
    }
}
