//! Delta Lake tables for gamma-lake.
//!
//! [`DeltaStore`] keeps a lake's tables as Delta tables under a local directory or an object
//! store URL, for use with [`Lake::open_with_store`](gamma_lake_core::Lake::open_with_store)
//! and [`Lake::create_with_store`](gamma_lake_core::Lake::create_with_store).

use std::collections::HashMap;
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
use deltalake::kernel::transaction::{CommitProperties, TransactionError};
use deltalake::kernel::MetadataValue;
use deltalake::logstore::object_store::ObjectStoreExt;
use deltalake::parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use deltalake::parquet::file::properties::WriterProperties;
use deltalake::protocol::SaveMode;
use deltalake::{DeltaTable, DeltaTableBuilder, DeltaTableError};
use futures::StreamExt;
use gamma_lake_core::runtime::threads;
use gamma_lake_core::store::{
    before_first_commit, extends, segments, version_at, CreateOptions, StoredTable, TableScan,
    TableStore,
};
use gamma_lake_core::{Dnf, Error, Result};
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

/// Commits that fail rather than retry on top of a newer version.
fn strict() -> CommitProperties {
    CommitProperties::default().with_max_retries(0)
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
        self.table
            .write(vec![as_view_types(batch)?])
            .with_save_mode(SaveMode::Append)
            .with_session_state(delta_session()?)
            .with_writer_properties(self.properties)
            .with_target_file_size(NonZeroU64::new(self.target_file_size))
            .with_commit_properties(strict())
            .await
            .map_err(|e| commit_err(&self.name, e))?;
        Ok(())
    }

    async fn append_merging(self: Box<Self>, batch: RecordBatch) -> Result<()> {
        extends(&self.schema, batch.schema_ref(), &self.name)?;
        self.table
            .write(vec![as_view_types(batch)?])
            .with_save_mode(SaveMode::Append)
            .with_schema_mode(deltalake::operations::write::SchemaMode::Merge)
            .with_session_state(delta_session()?)
            .with_writer_properties(self.properties)
            .with_target_file_size(NonZeroU64::new(self.target_file_size))
            .with_commit_properties(strict())
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
        self.table
            .merge(source, predicate)
            .with_session_state(delta_merge_session()?)
            .with_source_alias("source")
            .with_target_alias("target")
            .with_writer_properties(self.properties)
            .with_commit_properties(strict())
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
        let table = self
            .table
            .update_field_metadata()
            .with_field_name(column)
            .with_metadata(metadata)
            .with_commit_properties(strict())
            .await
            .map_err(|e| commit_err(&self.name, e))?;
        let schema = table
            .snapshot()
            .map_err(delta_err)?
            .snapshot()
            .arrow_schema();
        Ok(Box::new(DeltaStored {
            name: self.name,
            table,
            schema,
            properties: self.properties,
            target_file_size: self.target_file_size,
        }))
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
