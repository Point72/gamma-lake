//! Python bindings: `gammalake.gammalake`.
//!
//! Results are exported through the Arrow PyCapsule interface, so any consumer of
//! `__arrow_c_stream__` (polars, pyarrow, nanoarrow, duckdb) can read them without a
//! dependency on this module. The Polars-facing API lives in `gammalake/native.py`.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use gamma_lake::arrow::array::{make_array, new_empty_array, ArrayRef};
use gamma_lake::arrow::datatypes::Schema;
use gamma_lake::arrow::datatypes::{DataType, SchemaRef};
use gamma_lake::arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use gamma_lake::arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use gamma_lake::arrow::record_batch::RecordBatch;
use gamma_lake::{
    AddOptions, Annotation, BatchStream, DeltaOptions, DeltaStore, Dnf, Error, Expression,
    FeatureTable as CoreFeatureTable, Lake as CoreLake, LakeOptions, LockError, LockGuard, Op,
    OverlapMode, SelectedFeature, SignalType, TableLock, Term,
};
use pyo3::exceptions::{PyImportError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods, PyTuple};

// `import_exception!` declares a Rust handle to an exception class defined in Python, looked up
// lazily (the first time such an error is raised) from the package's public namespace.
pyo3::import_exception!(gammalake, MissingFeaturesException);
pyo3::import_exception!(gammalake, MissingOrMisregisteredSignalsException);
pyo3::import_exception!(gammalake, UninitializedDeltaLakeException);
pyo3::import_exception!(gammalake, UnsupportedError);
pyo3::import_exception!(gammalake, ConflictError);
pyo3::import_exception!(gammalake, PartialWriteError);

/// The exception a Python lock raised, if `error` is one.
fn lock_exception(error: &Error) -> Option<PyErr> {
    let Error::Lock { source, .. } = error else {
        return None;
    };
    let error = source.downcast_ref::<PyErr>()?;
    Some(Python::attach(|py| error.clone_ref(py)))
}

fn to_py(error: Error) -> PyErr {
    if let Some(raised) = lock_exception(&error) {
        return raised;
    }
    let message = error.to_string();
    // `match` consumes `error` (moved, not borrowed) so the arms can take ownership of fields.
    match error {
        Error::NotInitialized(_) => UninitializedDeltaLakeException::new_err(message),
        Error::MissingFeatures(_) => MissingFeaturesException::new_err(message),
        Error::MisregisteredTargets(_) => MissingOrMisregisteredSignalsException::new_err(message),
        Error::Unsupported {
            feature,
            signal_type,
        } => UnsupportedError::new_err((message, feature, signal_type)),
        Error::UnsupportedInput(_) => UnsupportedError::new_err(message),
        Error::Conflict { table } => ConflictError::new_err((message, table)),
        Error::Write {
            phase,
            committed,
            source,
        } => {
            let raised = lock_exception(&source);
            match raised {
                // Nothing was written, so the lock's own exception is the whole story.
                Some(raised) if committed.is_empty() => raised,
                // Otherwise raise PartialWriteError, chaining any lock exception as `__cause__`.
                _ => {
                    let error = PartialWriteError::new_err((message, phase, committed));
                    Python::attach(|py| error.set_cause(py, raised));
                    error
                }
            }
        }
        Error::InvalidPredicate(_) | Error::InvalidArgument(_) => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// An Arrow schema exported through `__arrow_c_schema__`.
#[pyclass(frozen, module = "gammalake.gammalake")]
struct ArrowSchema {
    schema: SchemaRef,
}

fn export_schema<'py>(py: Python<'py>, schema: &SchemaRef) -> PyResult<Bound<'py, PyCapsule>> {
    let ffi = FFI_ArrowSchema::try_from(schema.as_ref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    PyCapsule::new_with_value(py, ffi, c"arrow_schema")
}

#[pymethods]
impl ArrowSchema {
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        export_schema(py, &self.schema)
    }

    fn __repr__(&self) -> String {
        format!("ArrowSchema({})", self.schema)
    }
}

/// A single-use stream of record batches exported through `__arrow_c_stream__`.
#[pyclass(frozen, module = "gammalake.gammalake")]
struct ArrowStream {
    schema: SchemaRef,
    stream: Mutex<Option<BatchStream>>,
}

impl ArrowStream {
    fn new(stream: BatchStream) -> Self {
        use gamma_lake::arrow::record_batch::RecordBatchReader;
        Self {
            schema: stream.schema(),
            stream: Mutex::new(Some(stream)),
        }
    }
}

#[pymethods]
impl ArrowStream {
    /// Schema of the batches, available without consuming the stream.
    #[getter]
    fn schema(&self) -> ArrowSchema {
        ArrowSchema {
            schema: self.schema.clone(),
        }
    }

    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        export_schema(py, &self.schema)
    }

    /// Hands the stream to the consumer. `requested_schema` is ignored.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        let stream = self
            .stream
            .lock()
            .map_err(|_| PyRuntimeError::new_err("stream lock poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("stream already consumed"))?;
        let ffi = FFI_ArrowArrayStream::new(Box::new(stream));
        PyCapsule::new_with_value(py, ffi, c"arrow_array_stream")
    }

    /// Whether `__arrow_c_stream__` has already been called.
    #[getter]
    fn consumed(&self) -> bool {
        self.stream.lock().map(|s| s.is_none()).unwrap_or(true)
    }
}

/// Imports a predicate given as `[[(column, op, values_or_None), ...], ...]`.
fn import_dnf(predicate: Option<&Bound<'_, PyAny>>) -> PyResult<Dnf> {
    let Some(predicate) = predicate else {
        return Ok(Dnf::default());
    };
    let mut conjunctions = vec![];
    for conjunction in predicate.try_iter()? {
        let mut terms = vec![];
        for term in conjunction?.try_iter()? {
            let term = term?;
            let (column, op, values): (String, i32, Option<Bound<'_, PyAny>>) = term.extract()?;
            let op = Op::try_from(op).map_err(to_py)?;
            let values = match values {
                Some(values) => import_array(&values)?,
                None => new_empty_array(&DataType::Null),
            };
            terms.push(Term { column, op, values });
        }
        conjunctions.push(terms);
    }
    Ok(Dnf(conjunctions))
}

/// Imports an object implementing `__arrow_c_array__`, taking ownership of its data.
fn import_array(values: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    if !values.hasattr("__arrow_c_array__")? {
        return Err(PyTypeError::new_err(format!(
            "predicate values must implement __arrow_c_array__ (e.g. pyarrow.Array), got {}",
            values.get_type().name()?
        )));
    }
    let pair = values.call_method0("__arrow_c_array__")?;
    let pair = pair.cast::<PyTuple>()?;
    if pair.len() != 2 {
        return Err(PyTypeError::new_err(
            "__arrow_c_array__ must return a (schema, array) capsule pair",
        ));
    }
    let schema_capsule = pair.get_item(0)?;
    let array_capsule = pair.get_item(1)?;
    let schema_capsule = schema_capsule.cast::<PyCapsule>()?;
    let array_capsule = array_capsule.cast::<PyCapsule>()?;
    let schema_ptr: NonNull<FFI_ArrowSchema> = schema_capsule
        .pointer_checked(Some(c"arrow_schema"))?
        .cast();
    let array_ptr: NonNull<FFI_ArrowArray> =
        array_capsule.pointer_checked(Some(c"arrow_array"))?.cast();
    // SAFETY: the capsule names guarantee the pointee types. Both structs are moved out
    // and replaced by released ones, so the capsule destructors only free the allocations.
    unsafe {
        if schema_ptr.as_ref().release.is_none() || array_ptr.as_ref().is_released() {
            return Err(PyValueError::new_err(
                "__arrow_c_array__ returned a released schema or array",
            ));
        }
        let schema = std::ptr::replace(schema_ptr.as_ptr(), FFI_ArrowSchema::empty());
        let array = std::ptr::replace(array_ptr.as_ptr(), FFI_ArrowArray::empty());
        let data = from_ffi(array, &schema).map_err(|e| to_py(Error::Arrow(e)))?;
        Ok(make_array(data))
    }
}

/// Imports an object implementing `__arrow_c_stream__` and reads every batch.
fn import_stream(data: &Bound<'_, PyAny>) -> PyResult<Vec<RecordBatch>> {
    if !data.hasattr("__arrow_c_stream__")? {
        return Err(PyTypeError::new_err(format!(
            "input must implement __arrow_c_stream__ (e.g. polars.DataFrame), got {}",
            data.get_type().name()?
        )));
    }
    let capsule = data.call_method0("__arrow_c_stream__")?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let ptr: NonNull<FFI_ArrowArrayStream> =
        capsule.pointer_checked(Some(c"arrow_array_stream"))?.cast();
    // SAFETY: the capsule name guarantees the pointee type. The stream is moved out and
    // replaced by a released one, so the capsule destructor only frees the allocation.
    let reader = unsafe {
        if ptr.as_ref().release.is_none() {
            return Err(PyValueError::new_err(
                "__arrow_c_stream__ returned a released stream",
            ));
        }
        ArrowArrayStreamReader::from_raw(ptr.as_ptr()).map_err(|e| to_py(Error::Arrow(e)))?
    };
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| to_py(Error::Arrow(e)))
}

/// Imports an object implementing `__arrow_c_schema__`.
fn import_schema(schema: &Bound<'_, PyAny>) -> PyResult<SchemaRef> {
    if !schema.hasattr("__arrow_c_schema__")? {
        return Err(PyTypeError::new_err(format!(
            "index_schema must implement __arrow_c_schema__ (e.g. pyarrow.Schema), got {}",
            schema.get_type().name()?
        )));
    }
    let capsule = schema.call_method0("__arrow_c_schema__")?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let ptr: NonNull<FFI_ArrowSchema> = capsule.pointer_checked(Some(c"arrow_schema"))?.cast();
    // SAFETY: the capsule name guarantees the pointee type.
    if unsafe { ptr.as_ref() }.release.is_none() {
        return Err(PyValueError::new_err(
            "the Arrow schema capsule has already been consumed",
        ));
    }
    // SAFETY: the schema is live (checked above) and only borrowed.
    let schema = unsafe { Schema::try_from(ptr.as_ref()) }.map_err(|e| to_py(Error::Arrow(e)))?;
    Ok(std::sync::Arc::new(schema))
}

#[allow(clippy::too_many_arguments)]
fn lake_options(
    primary_sort_key: Option<String>,
    index_path: Option<String>,
    feature_metadata_path: Option<String>,
    table_metadata_path: Option<String>,
    batch_size: Option<usize>,
    validate: bool,
    table_name_prefix: Option<String>,
    compression: Option<String>,
    target_file_size: Option<u64>,
    storage_options: Option<HashMap<String, String>>,
    lock: Option<Py<PyAny>>,
    memory_limit: Option<usize>,
) -> (LakeOptions, DeltaOptions) {
    let defaults = LakeOptions::default();
    let delta = DeltaOptions::default();
    let options = LakeOptions {
        primary_sort_key: primary_sort_key.unwrap_or(defaults.primary_sort_key),
        index_path: index_path.unwrap_or(defaults.index_path),
        feature_metadata_path: feature_metadata_path.unwrap_or(defaults.feature_metadata_path),
        table_metadata_path: table_metadata_path.unwrap_or(defaults.table_metadata_path),
        batch_size: batch_size.unwrap_or(defaults.batch_size),
        validate,
        table_name_prefix: table_name_prefix.unwrap_or(defaults.table_name_prefix),
        lock: lock.map(|factory| Arc::new(PyLock { factory }) as Arc<dyn TableLock>),
        memory_limit,
    };
    let delta = DeltaOptions {
        compression: compression.unwrap_or(delta.compression),
        target_file_size: target_file_size.unwrap_or(delta.target_file_size),
        storage_options: storage_options.unwrap_or_default(),
    };
    (options, delta)
}

/// A lock made by a Python callable that takes the table location and returns a context
/// manager.
struct PyLock {
    factory: Py<PyAny>,
}

struct PyGuard {
    context: Py<PyAny>,
}

impl TableLock for PyLock {
    fn acquire(&self, target: &str) -> Result<Box<dyn LockGuard>, LockError> {
        Python::attach(|py| {
            let context = self.factory.call1(py, (target,))?;
            context.call_method0(py, "__enter__")?;
            Ok(Box::new(PyGuard { context }) as Box<dyn LockGuard>)
        })
        .map_err(|e: PyErr| Box::new(e) as LockError)
    }
}

impl LockGuard for PyGuard {
    fn release(self: Box<Self>) -> Result<(), LockError> {
        Python::attach(|py| {
            let none = py.None();
            self.context.call_method1(
                py,
                "__exit__",
                (none.clone_ref(py), none.clone_ref(py), none),
            )?;
            Ok(())
        })
        .map_err(|e: PyErr| Box::new(e) as LockError)
    }
}

fn pairs(features: &[(String, i64)]) -> Vec<(&str, i64)> {
    features.iter().map(|(n, v)| (n.as_str(), *v)).collect()
}

type SelectedRow = (String, String, String, Option<String>);

fn selected(rows: Vec<SelectedRow>) -> Vec<SelectedFeature> {
    rows.into_iter()
        .map(
            |(name, table_addr, signal_type, feature_params)| SelectedFeature {
                name,
                table_addr,
                signal_type,
                feature_params,
            },
        )
        .collect()
}

/// `(feature_name, table_addr, comment, tags)`.
type AnnotationRow = (String, Option<String>, Option<String>, Option<Vec<String>>);

fn annotation((feature_name, table_addr, comment, tags): AnnotationRow) -> Annotation {
    Annotation {
        feature_name,
        table_addr,
        comment,
        tags,
    }
}

fn batch_stream(batch: RecordBatch) -> ArrowStream {
    ArrowStream::new(BatchStream::from_batches(batch.schema(), vec![batch]))
}

fn names(items: Option<Vec<String>>) -> Vec<String> {
    items.unwrap_or_default()
}

/// A gamma-lake opened by the native core. Every call reads the latest table versions.
#[pyclass(frozen, module = "gammalake.gammalake")]
struct Lake {
    inner: CoreLake,
}

#[pymethods]
impl Lake {
    #[new]
    #[pyo3(signature = (
        base_path,
        *,
        primary_sort_key = None,
        index_path = None,
        feature_metadata_path = None,
        table_metadata_path = None,
        batch_size = None,
        validate = false,
        table_name_prefix = None,
        compression = None,
        target_file_size = None,
        storage_options = None,
        lock = None,
        memory_limit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        base_path: String,
        primary_sort_key: Option<String>,
        index_path: Option<String>,
        feature_metadata_path: Option<String>,
        table_metadata_path: Option<String>,
        batch_size: Option<usize>,
        validate: bool,
        table_name_prefix: Option<String>,
        compression: Option<String>,
        target_file_size: Option<u64>,
        storage_options: Option<HashMap<String, String>>,
        lock: Option<Py<PyAny>>,
        memory_limit: Option<usize>,
    ) -> PyResult<Self> {
        let (options, delta) = lake_options(
            primary_sort_key,
            index_path,
            feature_metadata_path,
            table_metadata_path,
            batch_size,
            validate,
            table_name_prefix,
            compression,
            target_file_size,
            storage_options,
            lock,
            memory_limit,
        );
        let inner = py
            .detach(|| {
                let store = DeltaStore::new(&base_path, delta)?;
                gamma_lake::open(store, options)
            })
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Creates the metadata and index tables of a lake, or opens it if they exist.
    #[staticmethod]
    #[pyo3(signature = (
        base_path,
        index_schema,
        *,
        primary_sort_key = None,
        index_path = None,
        feature_metadata_path = None,
        table_metadata_path = None,
        batch_size = None,
        validate = false,
        table_name_prefix = None,
        compression = None,
        target_file_size = None,
        storage_options = None,
        lock = None,
        memory_limit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create(
        py: Python<'_>,
        base_path: String,
        index_schema: Bound<'_, PyAny>,
        primary_sort_key: Option<String>,
        index_path: Option<String>,
        feature_metadata_path: Option<String>,
        table_metadata_path: Option<String>,
        batch_size: Option<usize>,
        validate: bool,
        table_name_prefix: Option<String>,
        compression: Option<String>,
        target_file_size: Option<u64>,
        storage_options: Option<HashMap<String, String>>,
        lock: Option<Py<PyAny>>,
        memory_limit: Option<usize>,
    ) -> PyResult<Self> {
        let schema = import_schema(&index_schema)?;
        let (options, delta) = lake_options(
            primary_sort_key,
            index_path,
            feature_metadata_path,
            table_metadata_path,
            batch_size,
            validate,
            table_name_prefix,
            compression,
            target_file_size,
            storage_options,
            lock,
            memory_limit,
        );
        let inner = py
            .detach(|| {
                let store = DeltaStore::new(&base_path, delta)?;
                gamma_lake::create(store, schema, options)
            })
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Adds the non-key columns of `data` as features or targets.
    #[pyo3(signature = (
        data,
        *,
        owner = None,
        signal_type = "feature",
        overlap_mode = "copy",
        feature_params = None,
        annotations = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        py: Python<'_>,
        data: Bound<'_, PyAny>,
        owner: Option<String>,
        signal_type: &str,
        overlap_mode: &str,
        feature_params: Option<String>,
        annotations: Option<Vec<AnnotationRow>>,
    ) -> PyResult<()> {
        let mut options = AddOptions {
            annotations: annotations.into_iter().flatten().map(annotation).collect(),
            ..Default::default()
        };
        if let Some(owner) = owner {
            options.owner = owner;
        }
        options.signal_type = match signal_type {
            "feature" => SignalType::Feature,
            "target" => SignalType::Target,
            "as_of_feature" => SignalType::AsOf,
            "sparse_feature" => SignalType::Sparse,
            other => {
                return Err(to_py(Error::UnsupportedInput(format!(
                    "signal type {other:?}; writes accept features, targets, as-of and \
                     sparse features"
                ))))
            }
        };
        options.overlap_mode = match overlap_mode {
            "copy" => OverlapMode::Copy,
            "merge" => OverlapMode::Merge,
            other => {
                return Err(PyValueError::new_err(format!(
                    "overlap_mode must be 'copy' or 'merge', got {other:?}"
                )))
            }
        };
        options.feature_params = feature_params;
        let batches = import_stream(&data)?;
        py.detach(|| self.inner.add(batches, &options))
            .map_err(to_py)
    }

    /// Adds the sort keys of `data` to the index without features.
    fn add_index_rows(&self, py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<()> {
        let batches = import_stream(&data)?;
        py.detach(|| self.inner.add_index_rows(batches))
            .map_err(to_py)
    }

    /// Where the lake's tables are: a local directory or URL.
    #[getter]
    fn location(&self) -> String {
        self.inner.location().to_string()
    }

    #[getter]
    fn sort_keys(&self) -> Vec<String> {
        self.inner.sort_keys().to_vec()
    }

    #[getter]
    fn primary_sort_key(&self) -> String {
        self.inner.primary_sort_key().to_string()
    }

    #[getter]
    fn index_schema(&self) -> ArrowSchema {
        ArrowSchema {
            schema: self.inner.index_schema(),
        }
    }

    /// The latest versions of `features` and `targets` as a DataFusion table.
    #[pyo3(signature = (features, targets = None))]
    fn feature_table(
        &self,
        py: Python<'_>,
        features: Vec<String>,
        targets: Option<Vec<String>>,
    ) -> PyResult<FeatureTable> {
        let targets = names(targets);
        py.detach(|| {
            let features: Vec<&str> = features.iter().map(String::as_str).collect();
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
            self.inner.feature_table(&features, &targets)
        })
        .map(|table| FeatureTable {
            inner: Arc::new(table),
        })
        .map_err(to_py)
    }

    /// Reads features and targets aligned to the index.
    #[pyo3(signature = (features, targets = None, predicate = None))]
    fn read(
        &self,
        py: Python<'_>,
        features: Vec<String>,
        targets: Option<Vec<String>>,
        predicate: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ArrowStream> {
        let dnf = import_dnf(predicate.as_ref())?;
        let targets = names(targets);
        py.detach(|| {
            let features: Vec<&str> = features.iter().map(String::as_str).collect();
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
            self.inner.read(&features, &targets, &dnf)
        })
        .map(ArrowStream::new)
        .map_err(to_py)
    }

    /// Schema [`read`] would produce, without reading any feature data.
    #[pyo3(signature = (features, targets = None))]
    fn read_schema(
        &self,
        py: Python<'_>,
        features: Vec<String>,
        targets: Option<Vec<String>>,
    ) -> PyResult<ArrowSchema> {
        let targets = names(targets);
        py.detach(|| {
            let features: Vec<&str> = features.iter().map(String::as_str).collect();
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
            self.inner.read_schema(&features, &targets)
        })
        .map(|schema| ArrowSchema { schema })
        .map_err(to_py)
    }

    /// Reads each `(name, version)` pair as [`read`] reads the latest versions.
    #[pyo3(signature = (features, predicate = None))]
    fn read_versions(
        &self,
        py: Python<'_>,
        features: Vec<(String, i64)>,
        predicate: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ArrowStream> {
        let dnf = import_dnf(predicate.as_ref())?;
        py.detach(|| self.inner.read_versions(&pairs(&features), &dnf))
            .map(ArrowStream::new)
            .map_err(to_py)
    }

    fn read_versions_schema(
        &self,
        py: Python<'_>,
        features: Vec<(String, i64)>,
    ) -> PyResult<ArrowSchema> {
        py.detach(|| self.inner.read_versions_schema(&pairs(&features)))
            .map(|schema| ArrowSchema { schema })
            .map_err(to_py)
    }

    /// Reads the features that `(name, table_addr, signal_type, feature_params)` rows of
    /// the feature metadata describe.
    #[pyo3(signature = (features, predicate = None))]
    fn read_selected(
        &self,
        py: Python<'_>,
        features: Vec<SelectedRow>,
        predicate: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ArrowStream> {
        let dnf = import_dnf(predicate.as_ref())?;
        let features = selected(features);
        py.detach(|| self.inner.read_selected(&features, &dnf))
            .map(ArrowStream::new)
            .map_err(to_py)
    }

    fn read_selected_schema(
        &self,
        py: Python<'_>,
        features: Vec<SelectedRow>,
    ) -> PyResult<ArrowSchema> {
        let features = selected(features);
        py.detach(|| self.inner.read_selected_schema(&features))
            .map(|schema| ArrowSchema { schema })
            .map_err(to_py)
    }

    #[pyo3(signature = (predicate = None))]
    fn index(&self, py: Python<'_>, predicate: Option<Bound<'_, PyAny>>) -> PyResult<ArrowStream> {
        let dnf = import_dnf(predicate.as_ref())?;
        py.detach(|| self.inner.index(&dnf))
            .map(ArrowStream::new)
            .map_err(to_py)
    }

    #[pyo3(signature = (predicate = None))]
    fn feature_metadata(
        &self,
        py: Python<'_>,
        predicate: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ArrowStream> {
        let dnf = import_dnf(predicate.as_ref())?;
        py.detach(|| self.inner.feature_metadata(&dnf))
            .map(ArrowStream::new)
            .map_err(to_py)
    }

    #[pyo3(signature = (predicate = None))]
    fn table_metadata(
        &self,
        py: Python<'_>,
        predicate: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ArrowStream> {
        let dnf = import_dnf(predicate.as_ref())?;
        py.detach(|| self.inner.table_metadata(&dnf))
            .map(ArrowStream::new)
            .map_err(to_py)
    }

    fn feature_metadata_schema(&self, py: Python<'_>) -> PyResult<ArrowSchema> {
        py.detach(|| self.inner.feature_metadata_schema())
            .map(|schema| ArrowSchema { schema })
            .map_err(to_py)
    }

    fn table_metadata_schema(&self, py: Python<'_>) -> PyResult<ArrowSchema> {
        py.detach(|| self.inner.table_metadata_schema())
            .map(|schema| ArrowSchema { schema })
            .map_err(to_py)
    }

    /// The latest version of every feature, one metadata row each.
    fn latest_features(&self, py: Python<'_>) -> PyResult<ArrowStream> {
        py.detach(|| self.inner.latest_features())
            .map(ArrowStream::new)
            .map_err(to_py)
    }

    /// Sets comments and tags; each row is `(feature_name, table_addr, comment, tags)` and
    /// `None` keeps the stored value or, for `table_addr`, selects the latest version.
    fn annotate(&self, py: Python<'_>, annotations: Vec<AnnotationRow>) -> PyResult<()> {
        let annotations: Vec<Annotation> = annotations.into_iter().map(annotation).collect();
        py.detach(|| self.inner.annotate(&annotations))
            .map_err(to_py)
    }

    /// Registers runtime-computed features; each row is
    /// `(feature_name, expression, root_names)`.
    fn add_expressions(
        &self,
        py: Python<'_>,
        expressions: Vec<(String, String, Vec<String>)>,
        owner: &str,
    ) -> PyResult<()> {
        let expressions: Vec<Expression> = expressions
            .into_iter()
            .map(|(name, expression, roots)| Expression {
                name,
                expression,
                roots,
            })
            .collect();
        py.detach(|| self.inner.add_expressions(&expressions, owner))
            .map_err(to_py)
    }

    /// The comment and tags of each `(feature_name, table_addr)`.
    fn annotations(
        &self,
        py: Python<'_>,
        features: Vec<(String, String)>,
    ) -> PyResult<ArrowStream> {
        py.detach(|| {
            let pairs: Vec<(&str, &str)> = features
                .iter()
                .map(|(n, a)| (n.as_str(), a.as_str()))
                .collect();
            self.inner.annotations(&pairs)
        })
        .map(batch_stream)
        .map_err(to_py)
    }

    /// The latest metadata rows of `features` (all when empty) with comments and tags.
    fn describe(&self, py: Python<'_>, features: Vec<String>) -> PyResult<ArrowStream> {
        py.detach(|| {
            let features: Vec<&str> = features.iter().map(String::as_str).collect();
            self.inner.describe(&features)
        })
        .map(batch_stream)
        .map_err(to_py)
    }

    /// Restores the lake to how it was at `at`, in microseconds since the Unix epoch.
    fn restore(&self, py: Python<'_>, at: i64) -> PyResult<()> {
        py.detach(|| self.inner.restore(at)).map_err(to_py)
    }

    /// Moves the latest versions of `features` into one new table; returns its address,
    /// or `None` when they already share one.
    fn consolidate(&self, py: Python<'_>, features: Vec<String>) -> PyResult<Option<String>> {
        py.detach(|| {
            let features: Vec<&str> = features.iter().map(String::as_str).collect();
            self.inner.consolidate(&features)
        })
        .map_err(to_py)
    }

}

/// Package version of the native library.
#[pyfunction]
fn version() -> &'static str {
    gamma_lake::version()
}

/// Number of native streams not yet released, for leak tests.
/// Features of a lake as a DataFusion table, for datafusion-python's
/// `SessionContext.register_table`.
#[pyclass(frozen, module = "gammalake.gammalake")]
struct FeatureTable {
    inner: Arc<CoreFeatureTable>,
}

#[pymethods]
impl FeatureTable {
    #[getter]
    fn schema(&self) -> ArrowSchema {
        use gamma_lake::datafusion::catalog::TableProvider;
        ArrowSchema {
            schema: self.inner.schema(),
        }
    }

    /// Names of the features the table reads.
    #[getter]
    fn features(&self) -> Vec<String> {
        self.inner
            .features()
            .iter()
            .map(|f| f.name.clone())
            .collect()
    }

    /// The table as a `datafusion_table_provider` capsule for `session`, a datafusion-python
    /// `SessionContext` of the same DataFusion major version.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = session_codec(&session)?;
        // Queries move to the core's runtime when they execute, so no runtime is needed here.
        let provider = FFI_TableProvider::new_with_ffi_codec(self.inner.clone(), true, None, codec);
        PyCapsule::new_with_value(py, provider, c"datafusion_table_provider")
    }
}

/// The logical extension codec of a datafusion-python session.
fn session_codec(session: &Bound<'_, PyAny>) -> PyResult<FFI_LogicalExtensionCodec> {
    const NAME: &str = "datafusion_logical_extension_codec";
    let capsule = session.call_method0("__datafusion_logical_extension_codec__")?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let name = capsule.name()?;
    // SAFETY: the name is read while the capsule is alive.
    let name = name.map(|n| unsafe { n.as_cstr() }.to_string_lossy().into_owned());
    if name.as_deref() != Some(NAME) {
        return Err(PyValueError::new_err(format!(
            "expected a {NAME:?} capsule from the session, got {name:?}"
        )));
    }
    let data: NonNull<FFI_LogicalExtensionCodec> = capsule
        .pointer_checked(Some(c"datafusion_logical_extension_codec"))?
        .cast();
    // SAFETY: the capsule name guarantees the type, and the capsule outlives this borrow.
    let codec = unsafe { data.as_ref() };
    let (found, expected) = (unsafe { (codec.version)() }, datafusion_ffi::version());
    if found != expected {
        return Err(PyImportError::new_err(format!(
            "the session uses DataFusion {found}, but gamma-lake was built with DataFusion \
             {expected}; install a datafusion package with major version {expected}"
        )));
    }
    Ok(codec.clone())
}

#[pyfunction]
fn _live_streams() -> usize {
    gamma_lake::live_streams()
}

#[pymodule]
fn gammalake(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(_live_streams, m)?)?;
    m.add_class::<Lake>()?;
    m.add_class::<FeatureTable>()?;
    m.add_class::<ArrowStream>()?;
    m.add_class::<ArrowSchema>()?;
    Ok(())
}
