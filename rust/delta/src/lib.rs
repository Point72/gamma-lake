//! Delta Lake tables for gamma-lake.
//!
//! [`DeltaStore`] keeps a lake's tables as Delta tables under a local directory or an object
//! store URL, for use with [`Lake::open_with_store`](gamma_lake_core::Lake::open_with_store)
//! and [`Lake::create_with_store`](gamma_lake_core::Lake::create_with_store).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use deltalake::arrow::array::RecordBatch;
use deltalake::arrow::compute::cast;
use deltalake::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use deltalake::datafusion::catalog::Session;
use deltalake::datafusion::common::{Column, TableReference};
use deltalake::datafusion::logical_expr::{cast as cast_expr, Expr};
use deltalake::datafusion::prelude::{DataFrame, SessionContext};
use deltalake::delta_datafusion::{DeltaColumn, DeltaSessionContext};
use deltalake::kernel::transaction::{CommitBuilder, CommitProperties, TransactionError};
use deltalake::kernel::{Action, Metadata, MetadataExt, MetadataValue, StructField, StructType};
use deltalake::logstore::get_actions;
use deltalake::logstore::object_store::ObjectStoreExt;
use deltalake::parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use deltalake::parquet::file::properties::WriterProperties;
use deltalake::protocol::{DeltaOperation, SaveMode};
use deltalake::{DeltaTable, DeltaTableBuilder, DeltaTableError, Path as StorePath};
use futures::StreamExt;
use gamma_lake_core::runtime::threads;
use gamma_lake_core::store::{
    before_first_commit, entangled, extends, segments, version_at, CreateOptions, StoredTable,
    TableScan, TableStore,
};
use gamma_lake_core::{Dnf, Error, Result};
use serde_json::Value;
use url::Url;

/// Options of a [`DeltaStore`].
#[derive(Clone)]
pub struct DeltaOptions {
    /// Parquet compression of written files: `uncompressed`, `snappy`, `gzip`, `brotli`,
    /// `lz4`, `lz4_raw` or `zstd`, case-insensitive.
    pub compression: String,
    /// Target size in bytes of Parquet files written by appends and new tables.
    pub target_file_size: u64,
    /// delta-rs object store options for remote lakes, such as `AWS_REGION` or
    /// `AWS_ENDPOINT_URL`; see [`DeltaStore`].
    pub storage_options: HashMap<String, String>,
}

impl fmt::Debug for DeltaOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Storage options can hold secrets, so only their keys are shown.
        let mut keys: Vec<_> = self.storage_options.keys().collect();
        keys.sort();
        f.debug_struct("DeltaOptions")
            .field("compression", &self.compression)
            .field("target_file_size", &self.target_file_size)
            .field("storage_options", &keys)
            .finish()
    }
}

impl Default for DeltaOptions {
    fn default() -> Self {
        Self {
            compression: "zstd".into(),
            target_file_size: DEFAULT_TARGET_FILE_SIZE,
            storage_options: HashMap::new(),
        }
    }
}

/// `e` as a gamma-lake error.
fn delta_err(e: DeltaTableError) -> Error {
    Error::Store(Box::new(e))
}

/// The frame over delta-rs's DataFusion provider that [`StoredTable::scan`] executes, and
/// the schema of its output.
fn scan_frame(
    provider: Arc<dyn deltalake::datafusion::catalog::TableProvider>,
    stored_schema: SchemaRef,
    columns: Option<&[String]>,
    dnf: &Dnf,
    batch_size: usize,
) -> Result<(DataFrame, SchemaRef)> {
    let relation = "t";
    let ctx = gamma_lake_core::runtime::session_context(&[(
        "datafusion.execution.batch_size",
        batch_size.to_string(),
    )])?;
    let mut frame = ctx.read_table(provider)?.alias(relation)?;
    if let Some(predicate) = dnf.to_expr(frame.schema().as_arrow(), relation)? {
        frame = frame.filter(predicate)?;
    }
    if let Some(columns) = columns {
        frame = frame.select(
            columns
                .iter()
                .map(|name| {
                    Expr::Column(Column::new(Some(TableReference::bare(relation)), name))
                        .alias(name)
                })
                .collect::<Vec<_>>(),
        )?;
    }
    let schema = match columns {
        Some(columns) => Arc::new(
            stored_schema.project(
                &columns
                    .iter()
                    .map(|name| stored_schema.index_of(name))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )?,
        ),
        None => stored_schema,
    };
    let expressions = schema
        .fields()
        .iter()
        .map(|field| {
            let value = Expr::Column(Column::from_name(field.name()));
            match frame.schema().field_with_unqualified_name(field.name()) {
                Ok(source) if source.data_type() != field.data_type() => {
                    cast_expr(value, field.data_type().clone()).alias(field.name())
                }
                _ => value,
            }
        })
        .collect::<Vec<_>>();
    Ok((frame.select(expressions)?, schema))
}

/// Key of a commit's `commitInfo` holding the mark of [`StoredTable::tag_commits`].
const TAG: &str = "gammaLakeTag";
/// Key of a commit's `commitInfo` holding the mark whose commits it undid.
const REVERTS: &str = "gammaLakeReverts";

/// Commits that fail rather than retry on top of a newer version, with `info` in their
/// `commitInfo`.
fn strict(info: Option<(&str, &str)>) -> CommitProperties {
    let properties = CommitProperties::default().with_max_retries(0);
    match info {
        Some((key, value)) => properties.with_metadata([(key.to_string(), value.into())]),
        None => properties,
    }
}

/// The column names of the schema `metadata` sets.
fn columns(metadata: &Metadata) -> Result<Vec<String>> {
    let invalid = |e: serde_json::Error| Error::Store(Box::new(e));
    let metadata = serde_json::to_value(metadata).map_err(invalid)?;
    let schema = metadata["schemaString"].as_str().unwrap_or("{}");
    let schema: Value = serde_json::from_str(schema).map_err(invalid)?;
    Ok(schema["fields"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| f["name"].as_str().map(str::to_string))
        .collect())
}

/// `e`, from a commit to table `name`, as [`Error::Conflict`] when another commit came first.
fn commit_err(name: &str, e: DeltaTableError) -> Error {
    match e {
        DeltaTableError::VersionAlreadyExists(_)
        | DeltaTableError::Transaction {
            source: TransactionError::MaxCommitAttempts(_) | TransactionError::CommitConflict(_),
        } => Error::Conflict {
            table: name.to_string(),
        },
        e => delta_err(e),
    }
}

/// The column `name` of `relation`, without parsing `name`.
fn column(relation: &str, name: &str) -> Expr {
    Expr::Column(Column::new(Some(relation), name))
}

/// A Delta session context with `overrides` and `target_partitions` set to [`threads`].
fn delta_context(overrides: &[(&str, String)]) -> Result<SessionContext> {
    let mut map = HashMap::from([(
        "datafusion.execution.target_partitions".to_string(),
        threads().to_string(),
    )]);
    map.extend(overrides.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Ok(DeltaSessionContext::new_with_session_overrides(&map)
        .map_err(delta_err)?
        .into_inner())
}

/// Session state for plain Delta writes, configured like [`delta_context`] except that
/// rows are not repartitioned, so files hold rows in input order as the Python `deltalake`
/// writer leaves them.
fn delta_session() -> Result<Arc<dyn Session>> {
    let in_order = [(
        "datafusion.optimizer.enable_round_robin_repartition",
        "false".to_string(),
    )];
    Ok(Arc::new(delta_context(&in_order)?.state()))
}

/// Session state for Delta merges. Scans keep plain string and binary types, so merged
/// files record the same Arrow types as files the Python `deltalake` merge writes.
fn delta_merge_session() -> Result<Arc<dyn Session>> {
    let plain = [(
        "datafusion.execution.parquet.schema_force_view_types",
        "false".to_string(),
    )];
    Ok(Arc::new(delta_context(&plain)?.state()))
}

/// Parses a compression name as `GammaFeatureLake.compression` accepts it.
fn compression(name: &str) -> Result<Compression> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "uncompressed" => Compression::UNCOMPRESSED,
        "snappy" => Compression::SNAPPY,
        "gzip" => Compression::GZIP(GzipLevel::default()),
        "brotli" => Compression::BROTLI(BrotliLevel::default()),
        "lz4" => Compression::LZ4,
        "lz4_raw" => Compression::LZ4_RAW,
        "zstd" => Compression::ZSTD(ZstdLevel::default()),
        _ => {
            return Err(Error::InvalidArgument(format!(
                "unknown compression {name:?}; expected one of uncompressed, snappy, gzip, \
                 brotli, lz4, lz4_raw or zstd"
            )))
        }
    })
}

/// The oldest version of a Delta table whose commit file is still in its log.
async fn earliest_commit(store: &dyn deltalake::ObjectStore) -> Result<u64> {
    use futures::TryStreamExt;
    let log = deltalake::Path::from("_delta_log");
    let files: Vec<_> = store
        .list(Some(&log))
        .try_collect()
        .await
        .map_err(|e| delta_err(e.into()))?;
    files
        .iter()
        .filter_map(|meta| {
            let version = meta.location.filename()?.strip_suffix(".json")?;
            (version.len() == 20).then(|| version.parse::<u64>().ok())?
        })
        .min()
        .ok_or_else(|| Error::InvalidArgument("a Delta log holds no commits".into()))
}

/// Default Parquet file size target of [`DeltaStore`], in bytes.
pub const DEFAULT_TARGET_FILE_SIZE: u64 = 128 * 1024 * 1024;

/// Delta tables under a local directory or an object store URL.
///
/// Local paths and `file://`, `s3://` (feature `s3`, on by default), `az://`, `abfss://`
/// (feature `azure`) and `gs://` (feature `gcs`) URLs are supported. `storage_options` are
/// the delta-rs object store options, such as `AWS_REGION` or `AWS_ENDPOINT_URL`. Credentials
/// missing from them come from the environment: on S3, the AWS default credential chain,
/// which reads `AWS_PROFILE` from the process environment and not from `storage_options`.
#[derive(Clone)]
pub struct DeltaStore {
    root: Url,
    location: String,
    storage_options: HashMap<String, String>,
    properties: WriterProperties,
    target_file_size: u64,
}

impl fmt::Debug for DeltaStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Storage options can hold secrets.
        f.debug_struct("DeltaStore")
            .field("location", &self.location)
            .field(
                "storage_options",
                &self.storage_options.keys().collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl DeltaStore {
    /// A store rooted at `base`, a local path or URL, which need not exist yet.
    pub fn new(base: &str, options: DeltaOptions) -> Result<Self> {
        if options.target_file_size == 0 {
            return Err(Error::InvalidArgument(
                "target_file_size must be positive".into(),
            ));
        }
        let root = match Url::parse(base) {
            // Single-letter schemes are Windows drive letters, not URLs.
            Ok(url) if url.scheme().len() > 1 && url.scheme() != "file" => {
                if url.query().is_some() || url.fragment().is_some() {
                    return Err(Error::InvalidArgument(format!(
                        "lake URL {base:?} must not have a query or fragment"
                    )));
                }
                if url.path().contains('%') {
                    return Err(Error::InvalidArgument(format!(
                        "lake URL {base:?} contains '%', which is not supported"
                    )));
                }
                let mut url = url;
                if !url.path().ends_with('/') {
                    let path = format!("{}/", url.path());
                    url.set_path(&path);
                }
                url
            }
            Ok(url) if url.scheme() == "file" => local_root(
                &url.to_file_path()
                    .map_err(|_| Error::InvalidArgument(format!("invalid file URL {base:?}")))?,
            )?,
            _ => local_root(Path::new(base))?,
        };
        let location = match root.to_file_path() {
            Ok(path) if root.scheme() == "file" => path.display().to_string(),
            _ => root.to_string(),
        };
        Ok(Self {
            root,
            location,
            storage_options: options.storage_options,
            properties: WriterProperties::builder()
                .set_compression(compression(&options.compression)?)
                .build(),
            target_file_size: options.target_file_size,
        })
    }

    fn table_url(&self, name: &str) -> Result<Url> {
        let mut url = self.root.clone();
        {
            let mut path = url.path_segments_mut().map_err(|_| {
                Error::InvalidArgument(format!("{} cannot hold tables", self.location))
            })?;
            path.pop_if_empty();
            path.extend(segments(name)?);
            path.push("");
        }
        Ok(url)
    }

    /// The local directory of `url` when the store is on the local file system.
    fn local(url: &Url) -> Option<PathBuf> {
        (url.scheme() == "file")
            .then(|| url.to_file_path().ok())
            .flatten()
    }
}

/// The URL of a local lake directory, with symlinks resolved in its existing ancestors so
/// every path to one directory yields the same URL.
fn local_root(path: &Path) -> Result<Url> {
    let invalid =
        |e: &dyn fmt::Display| Error::InvalidArgument(format!("invalid lake path {path:?}: {e}"));
    let absolute = std::path::absolute(path).map_err(|e| invalid(&e))?;
    let (existing, mut path) = absolute
        .ancestors()
        .find_map(|a| std::fs::canonicalize(a).ok().map(|c| (a, c)))
        .ok_or_else(|| invalid(&"no ancestor exists"))?;
    for part in absolute.strip_prefix(existing).unwrap().components() {
        match part {
            std::path::Component::Normal(part) => path.push(part),
            std::path::Component::CurDir => {}
            _ => return Err(invalid(&"'..' below a directory that does not exist")),
        }
    }
    // delta-rs decodes percent escapes twice, so `%` would address another directory.
    if path.to_string_lossy().contains('%') {
        return Err(Error::InvalidArgument(format!(
            "lake path {path:?} contains '%', which is not supported"
        )));
    }
    Url::from_directory_path(&path)
        .map_err(|_| Error::InvalidArgument(format!("invalid lake path {path:?}")))
}

#[async_trait]
impl TableStore for DeltaStore {
    fn location(&self) -> &str {
        &self.location
    }

    /// The absolute path of a local table, with symlinks resolved where it exists, or its
    /// URL without the trailing `/`.
    fn table_location(&self, name: &str) -> Result<String> {
        let url = self.table_url(name)?;
        let location = match Self::local(&url) {
            Some(path) => match local_root(&path)?.to_file_path() {
                Ok(path) => path.display().to_string(),
                Err(()) => path.display().to_string(),
            },
            None => url.to_string(),
        };
        Ok(location.trim_end_matches('/').to_string())
    }

    async fn open(&self, name: &str) -> Result<Option<Box<dyn StoredTable>>> {
        let url = self.table_url(name)?;
        if let Some(path) = Self::local(&url) {
            if !path.join("_delta_log").is_dir() {
                return Ok(None);
            }
        }
        let table = match DeltaTableBuilder::from_url(url)
            .map_err(delta_err)?
            .with_storage_options(self.storage_options.clone())
            .load()
            .await
        {
            Ok(table) => table,
            Err(DeltaTableError::NotATable(_)) => return Ok(None),
            Err(e) => return Err(delta_err(e)),
        };
        let schema = table
            .snapshot()
            .map_err(delta_err)?
            .snapshot()
            .arrow_schema();
        Ok(Some(Box::new(DeltaStored {
            name: name.to_string(),
            table,
            schema,
            properties: self.properties.clone(),
            target_file_size: self.target_file_size,
            tag: None,
        })))
    }

    async fn create(&self, name: &str, batch: RecordBatch, options: &CreateOptions) -> Result<()> {
        let url = self.table_url(name)?;
        if let Some(path) = Self::local(&url) {
            std::fs::create_dir_all(&path).map_err(|e| {
                Error::InvalidArgument(format!("cannot create {}: {e}", path.display()))
            })?;
        }
        let table =
            DeltaTable::try_from_url_with_storage_options(url, self.storage_options.clone())
                .await
                .map_err(delta_err)?;
        let mut write = table
            .write(vec![as_view_types(batch)?])
            .with_save_mode(SaveMode::ErrorIfExists)
            .with_session_state(delta_session()?)
            .with_writer_properties(self.properties.clone())
            .with_target_file_size(NonZeroU64::new(self.target_file_size));
        if !options.stats_columns.is_empty() {
            write = write.with_configuration([(
                "delta.dataSkippingStatsColumns",
                Some(options.stats_columns.join(",")),
            )]);
        }
        write.await.map_err(delta_err)?;
        Ok(())
    }
}

/// `batch` with the Arrow types Polars hands the Python `deltalake` writer, which records
/// them in each file's Arrow schema: strings and binaries as view types and lists as large
/// lists, at any depth. Plain Delta writes here record the same types. The Delta schema is
/// the same either way.
fn as_view_types(batch: RecordBatch) -> Result<RecordBatch> {
    fn field(f: &Field) -> Field {
        f.clone().with_data_type(view(f.data_type()))
    }
    fn view(t: &DataType) -> DataType {
        match t {
            DataType::Utf8 | DataType::LargeUtf8 => DataType::Utf8View,
            DataType::Binary | DataType::LargeBinary => DataType::BinaryView,
            DataType::List(item) | DataType::LargeList(item) => {
                DataType::LargeList(Arc::new(field(item)))
            }
            DataType::FixedSizeList(item, n) => DataType::FixedSizeList(Arc::new(field(item)), *n),
            DataType::Struct(fields) => DataType::Struct(fields.iter().map(|f| field(f)).collect()),
            other => other.clone(),
        }
    }
    let schema = batch.schema();
    if schema
        .fields()
        .iter()
        .all(|f| view(f.data_type()) == *f.data_type())
    {
        return Ok(batch);
    }
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (f, array) in schema.fields().iter().zip(batch.columns()) {
        let target = field(f);
        columns.push(if target.data_type() == f.data_type() {
            array.clone()
        } else {
            cast(array, target.data_type())?
        });
        fields.push(target);
    }
    let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
}

struct DeltaStored {
    name: String,
    table: DeltaTable,
    schema: SchemaRef,
    properties: WriterProperties,
    target_file_size: u64,
    tag: Option<String>,
}

impl DeltaStored {
    /// [`strict`] properties marking the commit with [`DeltaStored::tag`].
    fn committing(&self) -> CommitProperties {
        strict(self.tag.as_deref().map(|tag| (TAG, tag)))
    }

    /// Upserts the metadata of each `(column, metadata)` of `updates` into its column's, in one
    /// commit, as delta-rs's `update_field_metadata` does for one column.
    async fn commit_metadata(&self, updates: Vec<(String, HashMap<String, String>)>) -> Result<()> {
        let name = &self.name;
        let snapshot = self.table.snapshot().map_err(delta_err)?;
        let mut fields: Vec<StructField> = snapshot.schema().fields().cloned().collect();
        for (column, metadata) in updates {
            let field = fields
                .iter_mut()
                .find(|f| f.name == column)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!("table {name} has no column {column:?}"))
                })?;
            for (key, value) in metadata {
                if key.starts_with("delta.") {
                    return Err(delta_err(DeltaTableError::Generic(
                        "Not allowed to modify protected metadata e.g. `delta.columnMapping.id`"
                            .into(),
                    )));
                }
                field.metadata.insert(key, MetadataValue::String(value));
            }
        }
        let schema = StructType::try_new(fields).map_err(|e| delta_err(e.into()))?;
        let metadata = snapshot
            .metadata()
            .clone()
            .with_schema(&schema)
            .map_err(|e| delta_err(e.into()))?;
        let operation = DeltaOperation::UpdateFieldMetadata {
            fields: schema.fields().cloned().collect(),
        };
        CommitBuilder::from(self.committing())
            .with_actions(vec![Action::Metadata(metadata)])
            .build(Some(snapshot), self.table.log_store(), operation)
            .await
            .map_err(|e| commit_err(name, e.into()))?;
        Ok(())
    }
}

#[async_trait]
impl StoredTable for DeltaStored {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn scan(
        &self,
        columns: Option<&[String]>,
        dnf: &Dnf,
        batch_size: usize,
    ) -> Result<TableScan> {
        let (frame, schema) = scan_frame(
            self.table.table_provider().await?,
            self.schema(),
            columns,
            dnf,
            batch_size,
        )?;
        let stream = frame
            .execute_stream()
            .await?
            .map(|batch| batch.map_err(Error::from));
        Ok(TableScan {
            schema,
            stream: Box::pin(stream),
        })
    }

    async fn append(self: Box<Self>, batch: RecordBatch) -> Result<()> {
        let committing = self.committing();
        self.table
            .write(vec![as_view_types(batch)?])
            .with_save_mode(SaveMode::Append)
            .with_session_state(delta_session()?)
            .with_writer_properties(self.properties)
            .with_target_file_size(NonZeroU64::new(self.target_file_size))
            .with_commit_properties(committing)
            .await
            .map_err(|e| commit_err(&self.name, e))?;
        Ok(())
    }

    async fn append_merging(self: Box<Self>, batch: RecordBatch) -> Result<()> {
        extends(&self.schema, batch.schema_ref(), &self.name)?;
        let committing = self.committing();
        self.table
            .write(vec![as_view_types(batch)?])
            .with_save_mode(SaveMode::Append)
            .with_schema_mode(deltalake::operations::write::SchemaMode::Merge)
            .with_session_state(delta_session()?)
            .with_writer_properties(self.properties)
            .with_target_file_size(NonZeroU64::new(self.target_file_size))
            .with_commit_properties(committing)
            .await
            .map_err(|e| commit_err(&self.name, e))?;
        Ok(())
    }

    async fn merge(
        self: Box<Self>,
        source: RecordBatch,
        keys: &[String],
        update: &[String],
    ) -> Result<()> {
        let predicate = keys
            .iter()
            .map(|k| column("target", k).eq(column("source", k)))
            .reduce(Expr::and)
            .ok_or_else(|| Error::InvalidArgument("merge needs at least one key".into()))?;
        let names: Vec<String> = source
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let source = delta_context(&[])?.read_batch(source)?;
        let plain = |name: &str| DeltaColumn::from(Column::new_unqualified(name));
        let committing = self.committing();
        self.table
            .merge(source, predicate)
            .with_session_state(delta_merge_session()?)
            .with_source_alias("source")
            .with_target_alias("target")
            .with_writer_properties(self.properties)
            .with_commit_properties(committing)
            .when_matched_update(|mut u| {
                for f in update {
                    u = u.update(plain(f), column("source", f));
                }
                u
            })
            .map_err(delta_err)?
            .when_not_matched_insert(|mut i| {
                for n in &names {
                    i = i.set(plain(n), column("source", n));
                }
                i
            })
            .map_err(delta_err)?
            .await
            .map_err(|e| commit_err(&self.name, e))?;
        Ok(())
    }

    async fn set_column_metadata(
        self: Box<Self>,
        column: &str,
        metadata: HashMap<String, String>,
    ) -> Result<Box<dyn StoredTable>> {
        self.schema.field_with_name(column).map_err(|_| {
            Error::InvalidArgument(format!("table {} has no column {column:?}", self.name))
        })?;
        let metadata = metadata
            .into_iter()
            .map(|(k, v)| (k, MetadataValue::String(v)))
            .collect();
        let committing = self.committing();
        let table = self
            .table
            .update_field_metadata()
            .with_field_name(column)
            .with_metadata(metadata)
            .with_commit_properties(committing)
            .await
            .map_err(|e| commit_err(&self.name, e))?;
        let schema = table
            .snapshot()
            .map_err(delta_err)?
            .snapshot()
            .arrow_schema();
        Ok(Box::new(DeltaStored {
            table,
            schema,
            ..*self
        }))
    }

    async fn set_columns_metadata(
        self: Box<Self>,
        updates: Vec<(String, HashMap<String, String>)>,
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        self.commit_metadata(updates).await
    }

    async fn restore(self: Box<Self>, at: i64) -> Result<bool> {
        let latest = self.table.version().ok_or_else(|| {
            Error::InvalidArgument(format!("table {} has no versions", self.name))
        })?;
        // Delta dates a version by the modification time of its commit file. delta-rs's own
        // lookup can stop at any of several versions with equal times, so search here.
        let store = self.table.object_store();
        let committed = |version: u64| {
            let store = store.clone();
            async move {
                let path = deltalake::Path::from(format!("_delta_log/{version:020}.json"));
                let meta = store.head(&path).await.map_err(|e| delta_err(e.into()))?;
                Ok(meta.last_modified.timestamp_micros())
            }
        };
        // Old commit files may have been cleaned up, so check the latest first and then
        // search only the retained ones.
        if committed(latest).await? <= at {
            return Ok(false);
        }
        let earliest = earliest_commit(store.as_ref()).await?;
        let Some(version) = version_at(earliest, latest, at, committed).await? else {
            return Err(before_first_commit(&self.name, at));
        };
        self.table
            .restore()
            .with_version_to_restore(version)
            .await
            .map_err(|e| commit_err(&self.name, e))?;
        Ok(true)
    }

    fn version(&self) -> Option<u64> {
        self.table.version()
    }

    fn tag_commits(&mut self, tag: &str) {
        self.tag = Some(tag.to_string());
    }

    async fn revert(self: Box<Self>, tag: &str, since: u64) -> Result<bool> {
        let name = &self.name;
        let latest = self.table.version().unwrap_or(since);
        let log_store = self.table.log_store();
        // The commits from the first one marked `tag`, oldest first, with whether each is.
        let mut commits = vec![];
        let mut undone = false;
        for version in since + 1..=latest {
            let bytes = log_store
                .read_commit_entry(version)
                .await
                .map_err(delta_err)?
                .ok_or_else(|| {
                    Error::Store(
                        format!("table {name} has no log entry for version {version}").into(),
                    )
                })?;
            let actions = get_actions(version, &bytes).map_err(delta_err)?;
            let marked = |key: &str| {
                actions.iter().any(|a| {
                    matches!(a, Action::CommitInfo(info)
                        if info.info.get(key).and_then(Value::as_str) == Some(tag))
                })
            };
            let (ours, reverts) = (marked(TAG), marked(REVERTS));
            if reverts {
                // An earlier revert, whose result was lost, undid the commits before it.
                commits.clear();
                undone = true;
            } else if ours || !commits.is_empty() {
                commits.push((version, ours, actions));
            }
        }
        let Some(&(first, ..)) = commits.first() else {
            return Ok(undone);
        };
        let base = first - 1;
        let properties = strict(Some((REVERTS, tag)));
        if commits.iter().all(|(_, ours, _)| *ours) {
            self.table
                .restore()
                .with_version_to_restore(base)
                .with_commit_properties(properties)
                .await
                .map_err(|e| commit_err(name, e))?;
            return Ok(true);
        }
        let load = |version: u64| {
            let mut table = self.table.clone();
            async move {
                table.load_version(version).await.map_err(delta_err)?;
                Ok::<_, Error>(table)
            }
        };
        let base_metadata = load(base)
            .await?
            .snapshot()
            .map_err(delta_err)?
            .metadata()
            .clone();
        // Log paths are already percent-decoded; key them as `object_store_path` does.
        let path = |p: &str| StorePath::parse(p).unwrap_or_else(|_| StorePath::from(p));
        // The files the marked commits added and removed, with the version that removed each,
        // and whether they changed the metadata and its columns.
        let mut added = HashSet::new();
        let mut removed: Vec<(StorePath, u64)> = vec![];
        let (mut reschemed, mut recolumned) = (false, false);
        // The first commit not marked that changed the metadata.
        let mut rescheming = None;
        let mut columns_now = columns(&base_metadata)?;
        for (version, ours, actions) in &commits {
            for action in actions {
                match action {
                    Action::Add(add) if *ours => {
                        let file = path(&add.path);
                        match removed.iter().position(|(r, _)| *r == file) {
                            Some(i) => {
                                removed.remove(i);
                            }
                            None => {
                                added.insert(file);
                            }
                        }
                    }
                    Action::Remove(remove) if *ours => {
                        let file = path(&remove.path);
                        if !added.remove(&file) {
                            removed.push((file, *version));
                        }
                    }
                    Action::Remove(remove) if added.contains(&path(&remove.path)) => {
                        return Err(entangled(name, *version));
                    }
                    // Its rows may fill columns the marked commits added.
                    Action::Add(_) if recolumned => return Err(entangled(name, *version)),
                    Action::Metadata(metadata) => {
                        let columns = columns(metadata)?;
                        if *ours {
                            reschemed = true;
                            recolumned |= columns != columns_now;
                        } else {
                            rescheming.get_or_insert(*version);
                        }
                        columns_now = columns;
                    }
                    _ => {}
                }
            }
        }
        if let Some(version) = rescheming.filter(|_| reschemed) {
            return Err(entangled(name, version));
        }
        let snapshot = self.table.snapshot().map_err(delta_err)?;
        let current: HashMap<_, _> = snapshot
            .log_data()
            .iter()
            .map(|file| (file.object_store_path(), file))
            .collect();
        let mut actions = vec![];
        for file in &added {
            let file = current.get(file).ok_or_else(|| {
                Error::Store(format!("table {name} has lost file {file} this write added").into())
            })?;
            actions.push(Action::Remove(file.remove_action(true)));
        }
        // Files the marked commits removed, as the version before each such commit held them.
        let mut versions: Vec<u64> = removed.iter().map(|(_, v)| *v).collect();
        versions.dedup();
        for version in versions {
            let before = load(version - 1).await?;
            for file in before.snapshot().map_err(delta_err)?.log_data().iter() {
                let path = file.object_store_path();
                if removed.contains(&(path.clone(), version)) && !current.contains_key(&path) {
                    #[allow(deprecated)]
                    let mut add = file.add_action();
                    add.data_change = true;
                    actions.push(Action::Add(add));
                }
            }
        }
        if reschemed {
            actions.push(Action::Metadata(base_metadata));
        }
        if !actions.is_empty() {
            let operation = DeltaOperation::Restore {
                version: Some(base),
                datetime: None,
            };
            CommitBuilder::from(properties)
                .with_actions(actions)
                .build(Some(snapshot), log_store, operation)
                .await
                .map_err(|e| commit_err(name, e))?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;

    #[test]
    fn restore_searches_only_the_retained_log() {
        use deltalake::arrow::array::Int64Array;
        use deltalake::arrow::datatypes::Field;

        let dir = tempfile::TempDir::new().unwrap();
        let store = DeltaStore::new(dir.path().to_str().unwrap(), DeltaOptions::default()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let batch = |k: i64| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![k]))]).unwrap()
        };
        let now = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        };
        let tick = || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let at = now().as_micros() as i64;
            std::thread::sleep(std::time::Duration::from_millis(20));
            at
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            store
                .create("t", batch(0), &CreateOptions::default())
                .await
                .unwrap();
            for k in 1..6 {
                store
                    .open("t")
                    .await
                    .unwrap()
                    .unwrap()
                    .append(batch(k))
                    .await
                    .unwrap();
            }
            let table =
                deltalake::open_table(Url::from_directory_path(dir.path().join("t")).unwrap())
                    .await
                    .unwrap();
            deltalake::checkpoints::create_checkpoint(&table, None)
                .await
                .unwrap();
            for v in 0..5 {
                std::fs::remove_file(dir.path().join(format!("t/_delta_log/{v:020}.json")))
                    .unwrap();
            }
            let cutoff = tick();
            store
                .open("t")
                .await
                .unwrap()
                .unwrap()
                .append(batch(6))
                .await
                .unwrap();
            let open = || async { store.open("t").await.unwrap().unwrap() };
            assert!(!open().await.restore(i64::MAX).await.unwrap());
            assert!(open().await.restore(cutoff).await.unwrap());
            let scan = open()
                .await
                .scan(None, &Dnf::default(), 65536)
                .await
                .unwrap();
            let rows: usize = scan
                .stream
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum();
            assert_eq!(rows, 6);
            assert!(open().await.restore(0).await.is_err());
        });
    }

    #[test]
    fn revert_undoes_only_the_tagged_commits() {
        use deltalake::arrow::array::{ArrayRef, Int64Array};
        use deltalake::arrow::datatypes::Field;

        let dir = tempfile::TempDir::new().unwrap();
        let store = DeltaStore::new(dir.path().to_str().unwrap(), DeltaOptions::default()).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Int64, true),
        ]));
        let batch = |k: &[i64], v: i64| {
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from(k.to_vec())),
                Arc::new(Int64Array::from(vec![v; k.len()])),
            ];
            RecordBatch::try_new(schema.clone(), columns).unwrap()
        };
        let (keys, update) = (["k".to_string()], ["v".to_string()]);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let open = || async { store.open("t").await.unwrap().unwrap() };
            let tagged = |tag: &'static str| async move {
                let mut table = open().await;
                table.tag_commits(tag);
                table
            };
            let rows = || async {
                let scan = open()
                    .await
                    .scan(None, &Dnf::default(), 1024)
                    .await
                    .unwrap();
                let batches: Vec<RecordBatch> = scan.stream.try_collect().await.unwrap();
                let column = |b: &RecordBatch, i| {
                    let array = b.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
                    array.values().to_vec()
                };
                let mut rows: Vec<(i64, i64)> = batches
                    .iter()
                    .flat_map(|b| column(b, 0).into_iter().zip(column(b, 1)))
                    .collect();
                rows.sort();
                rows
            };
            store
                .create("t", batch(&[0], 0), &CreateOptions::default())
                .await
                .unwrap();

            // Only the write's commits follow the read, so the table is restored.
            tagged("a").await.append(batch(&[1], 0)).await.unwrap();
            assert!(open().await.revert("a", 0).await.unwrap());
            assert_eq!(rows().await, [(0, 0)]);
            // Reverting again, as after a revert whose outcome was lost, commits nothing.
            assert!(open().await.revert("a", 0).await.unwrap());
            assert_eq!(open().await.version(), Some(2));
            assert!(!open().await.revert("unused", 0).await.unwrap());

            // Another writer's commits before and after the write's are kept, and the file
            // the write rewrote is restored.
            open().await.append(batch(&[2], 0)).await.unwrap();
            let merging = tagged("b").await;
            merging
                .merge(batch(&[0, 3], 1), &keys, &update)
                .await
                .unwrap();
            // Annotating several columns is one commit.
            let annotating = tagged("b").await;
            let before = annotating.version();
            let unit = |u: &str| HashMap::from([("unit".to_string(), u.to_string())]);
            let updates = vec![("k".to_string(), unit("n")), ("v".to_string(), unit("s"))];
            annotating.set_columns_metadata(updates).await.unwrap();
            let annotated = open().await;
            assert_eq!(annotated.version(), before.map(|v| v + 1));
            assert_eq!(annotated.schema().field(1).metadata()["unit"], "s");
            open().await.append(batch(&[4], 0)).await.unwrap();
            assert!(open().await.revert("b", 2).await.unwrap());
            assert_eq!(rows().await, [(0, 0), (2, 0), (4, 0)]);
            let reverted = open().await;
            assert!(reverted
                .schema()
                .fields()
                .iter()
                .all(|f| f.metadata().is_empty()));

            // Another writer rewrote the write's rows, so it cannot be undone alone.
            let read = reverted.version().unwrap();
            tagged("c").await.append(batch(&[5], 0)).await.unwrap();
            open()
                .await
                .merge(batch(&[5], 2), &keys, &update)
                .await
                .unwrap();
            let err = open().await.revert("c", read).await.unwrap_err();
            let expected = format!("version {} of table t", read + 2);
            assert!(err.to_string().contains(&expected), "{err}");
            assert_eq!(rows().await, [(0, 0), (2, 0), (4, 0), (5, 2)]);
        });
    }

    #[test]
    fn window_scans_read_fewer_files() {
        use deltalake::arrow::array::Int64Array;
        use deltalake::arrow::datatypes::Field;
        use deltalake::datafusion::physical_plan::displayable;
        use gamma_lake_core::{Op, Term};

        let dir = tempfile::TempDir::new().unwrap();
        let store = DeltaStore::new(dir.path().to_str().unwrap(), DeltaOptions::default()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let batch = |k: i64| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![k]))]).unwrap()
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            store
                .create("t", batch(1), &CreateOptions::default())
                .await
                .unwrap();
            for k in 2..=8 {
                let table = store.open("t").await.unwrap().unwrap();
                table.append(batch(k)).await.unwrap();
            }
            let table = DeltaTableBuilder::from_url(store.table_url("t").unwrap())
                .unwrap()
                .load()
                .await
                .unwrap();
            let mut files = vec![];
            let window = Dnf(vec![vec![Term {
                column: "k".into(),
                op: Op::GtEq,
                values: Arc::new(Int64Array::from(vec![7])),
            }]]);
            for dnf in [Dnf::default(), window] {
                let provider = table.table_provider().await.unwrap();
                let (frame, _) = scan_frame(provider, schema.clone(), None, &dnf, 1024).unwrap();
                let plan = frame.create_physical_plan().await.unwrap();
                let text = displayable(plan.as_ref()).indent(true).to_string();
                files.push(text.matches(".parquet").count());
            }
            let (all, window) = (files[0], files[1]);
            assert!(
                window > 0 && window < all,
                "window scans {window} files, all scans {all}"
            );
        });
    }
}
