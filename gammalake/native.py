"""Read and write a gamma-lake through the native (Rust) core.

``NativeLake`` reads and writes the same Delta tables as :class:`~gammalake.GammaFeatureLake`
and returns Polars frames. Feature tables are aligned to the index by row position, as
``GammaFeatureLake.read`` does, and writes pad every table so that this alignment holds.
As-of and sparse features are stored unpadded and joined to the index when read.
``GammaFeatureLake`` is built on this module.

Errors raised while a stream is being consumed (for example a failed integrity check
with ``validate=True``) surface as ``RuntimeError`` from :meth:`NativeLake.read` and as a
Polars ``ComputeError`` from a collected :meth:`NativeLake.scan`.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import polars as pl
import pyarrow as pa
from polars_io_tools.io_sources.base import BaseExprNode, ErrorNode, FunctionNode, UnknownNode, get_parsed_expr
from polars_io_tools.io_sources.dnf_visitor import convert_expr_to_dnf
from polars_io_tools.io_sources.enum import BooleanFunctionType
from polars_io_tools.io_sources.util import register_io_source_with_is_pure
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gammalake import gammalake as _native
from gammalake._topo import deserialize_expression, with_columns_topo
from gammalake.abstract import MissingFeaturesException, MissingOrMisregisteredSignalsException

__all__ = ("ConflictError", "DeltaOptions", "NativeLake", "PartialWriteError", "UnsupportedError")

_OPS = {"=": "==", "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=", "in": "in", "not in": "not_in"}
_COMPRESSIONS = ("uncompressed", "snappy", "gzip", "brotli", "lz4", "lz4_raw", "zstd")


class DeltaOptions(BaseModel):
    """Configuration owned by the default Delta storage backend."""

    compression: str = "zstd"
    target_file_size: int = Field(128 * 1024 * 1024, gt=0)
    storage_options: dict[str, str] = Field(default_factory=dict, repr=False)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("compression")
    @classmethod
    def validate_compression(cls, value: str) -> str:
        if value.lower() not in _COMPRESSIONS:
            raise ValueError(f"expected one of {_COMPRESSIONS}, got {value!r}")
        return value


class UnsupportedError(NotImplementedError):
    """The lake cannot serve this request, such as a read of a feature whose signal type it
    does not know or a write of a column type it cannot store."""

    def __init__(self, message: str, feature: str | None = None, signal_type: str | None = None):
        super().__init__(message)
        self.feature = feature
        self.signal_type = signal_type


class ConflictError(RuntimeError):
    """Another writer committed to ``table`` after this write read it, so nothing was written.

    The lake is unchanged by the failed write, which can be run again.
    """

    def __init__(self, message: str, table: str):
        super().__init__(message)
        self.table = table


class PartialWriteError(RuntimeError):
    """A write failed after some of its tables may have committed.

    ``committed`` names the tables that did, and is empty when the write was rolled back;
    ``phase`` names the step that failed. When a lock failed, its exception is the ``__cause__``.
    """

    def __init__(self, message: str, phase: str, committed: list[str]):
        super().__init__(message)
        self.phase = phase
        self.committed = committed


LockFactory = Callable[[str], AbstractContextManager]
"""Takes the absolute path or URL of a table and returns a context manager holding a lock on it."""


def _coerce(column: str, dtype: pl.DataType, value: Any) -> Any:
    """``value`` as a Python object of ``dtype``, compared as Polars would compare it."""
    ok = False
    if isinstance(dtype, pl.Datetime):
        if isinstance(value, dt.datetime):
            literal_tz = pl.Series([value]).dtype.time_zone
            ok = literal_tz == dtype.time_zone and (dtype.time_unit != "ms" or value.microsecond % 1000 == 0)
        elif isinstance(value, dt.date) and dtype.time_zone is not None:
            # Polars compares a date with a zoned datetime as midnight UTC.
            value, ok = dt.datetime.combine(value, dt.time(), tzinfo=dt.UTC), True
    elif dtype == pl.Date:
        ok = isinstance(value, dt.date) and not isinstance(value, dt.datetime)
    elif dtype == pl.String:
        ok = isinstance(value, str)
    elif dtype == pl.Boolean:
        ok = isinstance(value, bool)
    elif dtype.is_integer():
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif dtype.is_float():
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    if not ok:
        raise UnsupportedError(f"cannot compare {column!r} ({dtype}) with {value!r} ({type(value).__name__}); convert the value to the column's type")
    return value


def _encode(column: str, dtype: pl.DataType, values: Sequence[Any]) -> pa.Array:
    coerced = [_coerce(column, dtype, v) for v in values]
    try:
        series = pl.Series(column, coerced, dtype=dtype, strict=True)
    except (OverflowError, pl.exceptions.PolarsError) as error:
        raise UnsupportedError(f"cannot represent {values!r} as {column!r} ({dtype})") from error
    if dtype == pl.String:
        # The core compares Utf8 literals with any string column.
        return series.to_arrow().cast(pa.string())
    return series.to_arrow()


Term = tuple[str, str, pa.Array | None]


def _terms(filters, schema: pl.Schema) -> list[list[Term]]:
    """pyarrow-style ``filters`` as the core's DNF: a list of OR-ed lists of AND-ed terms."""
    if not filters:
        return [[]]
    clauses = [filters] if isinstance(filters[0], tuple) else filters
    result = []
    for clause in clauses:
        terms = []
        for column, op, value in clause:
            if column not in schema:
                raise ValueError(f"unknown filter column {column!r}; expected one of {list(schema)}")
            op = op.lower() if isinstance(op, str) else op
            if value is None and op in ("=", "==", "is"):
                terms.append((column, "is_null", None))
            elif value is None and op in ("!=", "is not"):
                terms.append((column, "is_not_null", None))
            elif op in ("in", "not in"):
                terms.append((column, _OPS[op], _encode(column, schema[column], list(value))))
            elif op in _OPS:
                terms.append((column, _OPS[op], _encode(column, schema[column], [value])))
            else:
                raise ValueError(f"unsupported filter operator {op!r}")
        result.append(terms)
    return result


def _and(left: list[list[Term]], right: list[list[Term]]) -> list[list[Term]]:
    return [a + b for a in left for b in right]


_MAX_CLAUSES = 64
_PUSHED_OPS = {**_OPS, "!in": _OPS["not in"]}


def _pushdown(predicate: pl.Expr | None, schema: pl.Schema, columns: Iterable[str]) -> list[list[Term]]:
    """The parts of ``predicate`` on ``columns`` that the core evaluates, as DNF.

    ``convert_expr_to_dnf`` gives a DNF selecting every row that ``predicate`` selects. Terms
    the core cannot evaluate, or on other or float columns (the core orders NaNs differently
    from Polars), are dropped from their clause, which only widens it, so ``predicate`` must
    still be applied to what is read.
    """
    node = _parse(predicate)
    if node is None or _row_aligned(node):
        return [[]]
    dnf = convert_expr_to_dnf(node, schema)
    if not dnf or len(dnf) > _MAX_CLAUSES:
        return [[]]
    pushable = {c for c in columns if not schema[c].is_float()}
    return [[t for c, op, v in clause if c in pushable and (t := _pushed_term(c, op, v, schema[c])) is not None] for clause in dnf]


def _pushed_term(column: str, op: str, value: Any, dtype: pl.DataType) -> Term | None:
    if value is None and op in ("is", "is not"):
        return (column, "is_null" if op == "is" else "is_not_null", None)
    if op not in _PUSHED_OPS:
        return None
    try:
        return (column, _PUSHED_OPS[op], _encode(column, dtype, value if op in ("in", "not in", "!in") else [value]))
    except UnsupportedError:
        return None


def _parse(predicate: pl.Expr | None) -> BaseExprNode | None:
    if predicate is None:
        return None
    try:
        return get_parsed_expr(predicate)
    except Exception as error:  # noqa: BLE001
        return ErrorNode(expr=predicate, error=str(error))


def _row_aligned(node: BaseExprNode) -> bool:
    """Whether ``node`` may hold a literal with a value per row, which must see every row in order.

    Parts that could not be parsed count as row-aligned.
    """
    if isinstance(node, UnknownNode | ErrorNode):
        return True
    if isinstance(node, FunctionNode) and node.function_type == BooleanFunctionType.IS_IN and len(node.inputs) == 2:
        values = node.inputs[1]
        if (
            values.can_extract_literal
            and isinstance(values.value, list | tuple)
            and all(v is None or not isinstance(v, list | tuple | dict | pl.Series) for v in values.value)
        ):
            return _row_aligned(node.inputs[0])
    if node.can_extract_literal and isinstance(node.value, list | tuple | dict | pl.Series):
        return True
    return any(_row_aligned(child) for child in node.get_children())


def _dnf(clauses: list[list[Term]]) -> list[list[Term]]:
    """``clauses`` for the core; an empty clause matches every row, so nothing is filtered."""
    return [] if any(not c for c in clauses) else clauses


def _scan(schema: pl.Schema, open_stream, base: list[list[Term]], filter_schema: pl.Schema, pushable: list[str]) -> pl.LazyFrame:
    """A lazy frame of ``schema`` whose batches come from ``open_stream(columns, dnf)``.

    ``dnf`` is ``base`` and the part of a later predicate on ``pushable`` columns that the
    core can evaluate; ``columns`` is the projection, or ``None`` for every column.
    """

    def source(with_columns: list[str] | None, predicate: pl.Expr | None, n_rows: int | None, batch_size: int | None) -> Iterator[pl.DataFrame]:
        node = _parse(predicate)
        # Polars passes n_rows for rows taken before the predicate, so it cannot be pushed.
        pushed = [[]] if n_rows is not None else _pushdown(predicate, filter_schema, pushable)
        reader = pa.RecordBatchReader.from_stream(open_stream(with_columns, _dnf(_and(base, pushed))))
        frames = _head((pl.from_arrow(batch) for batch in reader), n_rows)
        if node is not None and _row_aligned(node):
            collected = list(frames)
            frames = [pl.concat(collected, rechunk=True) if collected else pl.DataFrame(schema=schema)]
        for frame in frames:
            if predicate is not None:
                frame = frame.filter(predicate)
            if with_columns is not None:
                frame = frame.select(with_columns)
            yield frame

    return register_io_source_with_is_pure(source, schema=schema, explain_name="gammalake_native")


def _head(frames: Iterator[pl.DataFrame], n_rows: int | None) -> Iterator[pl.DataFrame]:
    """The first ``n_rows`` rows of ``frames``, or all of them."""
    remaining = n_rows
    for frame in frames:
        if remaining is not None:
            if remaining == 0:
                return
            frame = frame.head(remaining)
            remaining -= frame.height
        yield frame


def _schema(exported) -> pl.Schema:
    return pl.from_arrow(pa.schema(exported).empty_table()).schema


def _selected_rows(features: pl.DataFrame) -> list[tuple[str, str, str, str | None]]:
    """``(name, table_addr, signal_type, feature_params)`` rows of a feature metadata frame."""
    params = features["feature_params"] if "feature_params" in features.columns else pl.repeat(None, features.height, dtype=pl.String, eager=True)
    rows = []
    for name, addr, signal_type, param in zip(features["feature_name"], features["table_addr"], features["signal_type"], params):
        if addr is None:
            raise UnsupportedError(f"feature {name!r} has no table", name, signal_type)
        rows.append((name, addr, signal_type or "", param))
    return rows


def _by_address(rows: list[tuple[str, str, str, str | None]]) -> dict[str, tuple[str, str, str, str | None]]:
    """``rows`` by name, each read the way the first row at its table is read.

    A table's signal type and params come from the first of its selected rows, so a lazy
    frame that reads a subset of the rows must keep them.
    """
    first: dict[str, tuple[str, str, str, str | None]] = {}
    for row in rows:
        first.setdefault(row[1], row)
    return {name: (name, addr, first[addr][2], first[addr][3]) for name, addr, _, _ in rows}


def _ordered_rows(metadata: pl.DataFrame, keys: list[tuple], key_columns: list[str]) -> list[tuple[str, str, str, str | None]]:
    """``_selected_rows`` of the metadata rows whose ``key_columns`` match ``keys``, in ``keys`` order."""
    index = {key: i for i, key in enumerate(metadata.select(key_columns).rows())}
    return _selected_rows(metadata[[index[key] for key in keys]])


_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
_RUNTIME_COMPUTED = "runtime_computed"


def _runtime_only(error: UnsupportedError) -> None:
    """Re-raises ``error`` unless the core refused a runtime-computed feature."""
    if error.signal_type != _RUNTIME_COMPUTED:
        raise error


def _from_stream(stream) -> pl.DataFrame:
    try:
        table = pa.RecordBatchReader.from_stream(stream).read_all()
    except pa.ArrowException as error:
        raise RuntimeError(str(error)) from error
    return pl.from_arrow(table)


class NativeLake:
    """A gamma-lake read and written through the native core.

    Every call reads the latest version of each table. ``start`` and ``end`` are inclusive
    bounds on the primary sort key and filter the index and every feature table before
    they are aligned, as in ``GammaFeatureLake.read``. ``filters`` take the pyarrow form:
    a list of ``(column, op, value)`` tuples, or a list of such lists OR-ed together.
    Read filters may only reference sort keys.

    ``base_path`` is a local directory or a ``file://``, ``s3://``, ``az://``, ``abfss://`` or
    ``gs://`` URL. ``storage_options`` are delta-rs object store options such as
    ``AWS_REGION`` or ``AWS_ENDPOINT_URL``; credentials they omit come from the environment.

    ``lock`` is called with the absolute path or URL of each table a write commits to, and
    the context manager it returns is held around that commit. Writes from several processes
    to an object store without atomic commits need one. The lock serialises commits; it does
    not make a whole write atomic, so writers that extend the index concurrently must be
    coordinated by the caller. Reads take no lock. An exception raised by the lock propagates
    unchanged, unless part of the write committed and was not rolled back: then it is the
    ``__cause__`` of a :class:`PartialWriteError`. A table whose lock failed to release after
    its commit counts as committed.

    Each write appends ``pending`` and then ``success`` or ``failed`` rows to the transaction
    log at ``transaction_log_path``, created on the first write when missing. A write that
    fails after committing part of itself undoes its commits, which are marked with its
    transaction id, on each table it committed to, keeping other writers' commits; it logs
    ``rolled_back`` and then fails as a write that committed nothing does. Tables it created
    are left in place, unreferenced. It is left ``failed``, naming the tables that may hold
    it, when an undo fails, as when another writer rewrote rows the write committed.

    By default a read sorts each table it reads in memory. With ``memory_limit`` set, the
    tables of a read share that many bytes for sorting and spill to the system temporary
    directory (``TMPDIR``) beyond it, so reads larger than memory can be streamed with
    :meth:`scan`.

    Runtime-computed features are Polars expressions stored in the feature metadata and
    evaluated when read. Deserializing an expression can run code, so reading or registering
    them requires ``runtime_computed_features=True``, set only when the metadata is trusted.
    A read of a runtime-computed feature also returns the stored and runtime-computed features
    its expression reads, at their latest versions, after the requested ones.
    """

    def __init__(
        self,
        base_path: str,
        *,
        primary_sort_key: str = "timestamp",
        index_path: str = "index",
        feature_metadata_path: str = "feature_metadata",
        table_metadata_path: str = "table_metadata",
        batch_size: int | None = None,
        validate: bool = False,
        table_name_prefix: str = "",
        delta_options: DeltaOptions | None = None,
        runtime_computed_features: bool = False,
        lock: LockFactory | None = None,
        memory_limit: int | None = None,
        transaction_log_path: str = "transaction_log",
    ):
        delta_options = delta_options or DeltaOptions()
        lake = _native.Lake(
            base_path,
            primary_sort_key=primary_sort_key,
            index_path=index_path,
            feature_metadata_path=feature_metadata_path,
            table_metadata_path=table_metadata_path,
            batch_size=batch_size,
            validate=validate,
            table_name_prefix=table_name_prefix,
            compression=delta_options.compression,
            target_file_size=delta_options.target_file_size,
            storage_options=delta_options.storage_options or None,
            lock=lock,
            memory_limit=memory_limit,
            transaction_log_path=transaction_log_path,
        )
        self._init(lake, runtime_computed_features)

    def _init(self, lake, runtime_computed_features: bool) -> None:
        self._lake = lake
        self._runtime_computed_features = runtime_computed_features
        self._index_schema = _schema(lake.index_schema)

    @classmethod
    def create(cls, base_path: str, index_schema: pa.Schema, **options) -> NativeLake:
        """Creates the metadata and index tables of a lake, or opens it if they exist.

        ``index_schema`` holds the sort keys; ``options`` are those of the constructor.
        """
        self = cls.__new__(cls)
        runtime_computed_features = options.pop("runtime_computed_features", False)
        delta_options = options.pop("delta_options", None) or DeltaOptions()
        options.update(
            compression=delta_options.compression,
            target_file_size=delta_options.target_file_size,
            storage_options=delta_options.storage_options or None,
        )
        self._init(_native.Lake.create(base_path, index_schema, **options), runtime_computed_features)
        return self

    @property
    def location(self) -> str:
        """Where the lake's tables are: a local directory or URL."""
        return self._lake.location

    @property
    def sort_keys(self) -> list[str]:
        return self._lake.sort_keys

    @property
    def primary_sort_key(self) -> str:
        return self._lake.primary_sort_key

    @property
    def index_schema(self) -> pl.Schema:
        return self._index_schema

    def _window(self, start, end, filters) -> list[list[Term]]:
        bounds = [(self.primary_sort_key, op, v) for op, v in ((">=", start), ("<=", end)) if v is not None]
        return _and(_terms(bounds, self._index_schema) if bounds else [[]], _terms(filters, self._index_schema))

    def read_schema(self, features: list[str], targets: list[str] | None = None) -> pl.Schema:
        """Schema of :meth:`read`: resolved names in request order, then the sort keys."""
        try:
            return _schema(self._lake.read_schema(features, targets))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(self._requested(features, targets or []), None, None, None).collect_schema()

    def read_arrow(self, features: list[str], targets: list[str] | None = None, start=None, end=None, filters=None):
        """The result of :meth:`read` as an Arrow C stream (``__arrow_c_stream__``), consumable once."""
        try:
            return self._lake.read(features, targets, _dnf(self._window(start, end, filters)))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(self._requested(features, targets or []), start, end, filters).collect().to_arrow()

    def read(self, features: list[str], targets: list[str] | None = None, start=None, end=None, filters=None) -> pl.DataFrame:
        """Features and targets aligned to the index, sorted by the sort keys."""
        return _from_stream(self.read_arrow(features, targets, start, end, filters))

    def feature_table(self, features: list[str], targets: list[str] | None = None):
        """The latest versions of ``features`` and ``targets`` as a DataFusion table.

        Register the result with ``datafusion.SessionContext.register_table``; it needs the
        ``datafusion`` extra, at the DataFusion major version gamma-lake was built with. The table
        has one row per index row, with :meth:`read_schema` as its schema, and is sorted by the
        sort keys. Filters on sort keys are pushed down; DataFusion applies the rest. The features
        are resolved now, but each query reads their tables as they are then. Runtime-computed
        features are not supported.
        """
        return self._lake.feature_table(features, targets)

    def scan(self, features: list[str], targets: list[str] | None = None, start=None, end=None, filters=None) -> pl.LazyFrame:
        """A lazy :meth:`read`.

        Only the selected features are read. Predicates added to the returned lazy frame are
        pushed into the native scan where they constrain the sort keys, and are applied again
        after it.
        """
        try:
            schema = _schema(self._lake.read_schema(features, targets))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(self._requested(features, targets or []), start, end, filters)
        names = [(n,) for n in schema if n not in self.sort_keys]
        rows = _by_address(_ordered_rows(self.latest_features(), names, ["feature_name"]))
        return self._lazy(schema, lambda wanted, dnf: self._lake.read_selected([rows[n] for n in wanted], dnf), start, end, filters)

    def read_versions_schema(self, features: Mapping[str, int]) -> pl.Schema:
        """Schema of :meth:`read_versions`."""
        try:
            return _schema(self._lake.read_versions_schema(list(features.items())))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(self._pinned(features), None, None, None).collect_schema()

    def read_versions(self, features: Mapping[str, int], start=None, end=None, filters=None) -> pl.DataFrame:
        """Each feature at the version it maps to, read as :meth:`read` reads the latest versions.

        The features a runtime-computed feature reads are taken at their latest versions.
        """
        try:
            stream = self._lake.read_versions(list(features.items()), _dnf(self._window(start, end, filters)))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(self._pinned(features), start, end, filters).collect()
        return _from_stream(stream)

    def scan_versions(self, features: Mapping[str, int], start=None, end=None, filters=None) -> pl.LazyFrame:
        """A lazy :meth:`read_versions`."""
        try:
            schema = _schema(self._lake.read_versions_schema(list(features.items())))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(self._pinned(features), start, end, filters)
        rows = _by_address(_ordered_rows(self.feature_metadata(), list(features.items()), ["feature_name", "version"]))
        return self._lazy(schema, lambda wanted, dnf: self._lake.read_selected([rows[n] for n in wanted], dnf), start, end, filters)

    def read_selected_schema(self, features: pl.DataFrame) -> pl.Schema:
        """Schema of :meth:`read_selected`."""
        try:
            rows = _selected_rows(features)
            return _schema(self._lake.read_selected_schema(rows))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(features, None, None, None).collect_schema()

    def read_selected(self, features: pl.DataFrame, start=None, end=None, filters=None) -> pl.DataFrame:
        """The features that rows of :meth:`feature_metadata` describe, one row per name.

        Uses the ``feature_name``, ``table_addr``, ``signal_type`` and ``feature_params``
        columns. The rows are not checked against the
        stored metadata. Features that runtime-computed rows read and the rows do not select
        are taken at their latest versions.
        """
        try:
            rows = _selected_rows(features)
            stream = self._lake.read_selected(rows, _dnf(self._window(start, end, filters)))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(features, start, end, filters).collect()
        return _from_stream(stream)

    def scan_selected(self, features: pl.DataFrame, start=None, end=None, filters=None) -> pl.LazyFrame:
        """A lazy :meth:`read_selected`."""
        try:
            selected = _selected_rows(features)
            schema = _schema(self._lake.read_selected_schema(selected))
        except UnsupportedError as error:
            _runtime_only(error)
            return self._runtime(features, start, end, filters)
        rows = _by_address(selected)
        return self._lazy(schema, lambda wanted, dnf: self._lake.read_selected([rows[n] for n in wanted], dnf), start, end, filters)

    def _requested(self, features: list[str], targets: list[str]) -> pl.DataFrame:
        """The latest metadata rows of ``features`` and ``targets``, in request order."""
        if targets:
            # A name is a target if any of its versions is, as for stored reads.
            registered = set(self.feature_metadata().filter(pl.col("signal_type") == "target")["feature_name"])
            misregistered = [t for t in targets if t not in registered]
        frame = self.latest_features()
        latest = {row["feature_name"]: i for i, row in enumerate(frame.iter_rows(named=True))}
        if targets and misregistered:
            raise MissingOrMisregisteredSignalsException(
                f"The following target values were not found to be registered as target signal types: {misregistered}"
            )
        names = [n for n in dict.fromkeys([*features, *targets]) if n in latest]
        if not names:
            raise MissingFeaturesException(f"No FeatureTables exist for any of your input features: {features + targets}")
        return frame[[latest[n] for n in names]]

    def _pinned(self, features: Mapping[str, int]) -> pl.DataFrame:
        """The metadata rows of each feature at the version it maps to, in mapping order."""
        metadata = self.feature_metadata()
        index = {key: i for i, key in enumerate(metadata.select("feature_name", "version").rows())}
        missing = [key for key in features.items() if key not in index]
        if missing:
            raise MissingFeaturesException(f"No FeatureTables exist for these feature versions: {missing}")
        return metadata[[index[key] for key in features.items()]]

    def _runtime(self, selected: pl.DataFrame, start, end, filters) -> pl.LazyFrame:
        """A lazy read of the metadata rows ``selected`` that computes the runtime-computed ones.

        Stored features the expressions read are added at their latest versions, as are
        runtime-computed ones, transitively; the result holds the selected names, then the
        added ones in the order found, then the sort keys.
        """
        if not self._runtime_computed_features:
            raise ValueError(
                "Runtime-computed features are disabled. Enable them (runtime_computed_features=True on NativeLake, enable_runtime_computed_features=True on GammaFeatureLake) only when feature metadata is trusted."
            )
        selected = selected.unique("feature_name", keep="first", maintain_order=True)
        rows = list(selected.iter_rows(named=True))
        known = {row["feature_name"] for row in rows}
        latest = None
        expressions = []
        i = 0
        while i < len(rows):
            row = rows[i]
            i += 1
            if row["signal_type"] != _RUNTIME_COMPUTED:
                continue
            expression = deserialize_expression(row["feature_params"])
            expressions.append(expression)
            for root in expression.meta.root_names():
                if root in known or root in self.sort_keys:
                    continue
                if latest is None:
                    latest = {r["feature_name"]: r for r in self.latest_features().iter_rows(named=True)}
                if root not in latest:
                    raise MissingFeaturesException(f"runtime-computed feature {row['feature_name']!r} reads {root!r}, which is not in the lake")
                known.add(root)
                rows.append(latest[root])
        stored = pl.DataFrame([r for r in rows if r["signal_type"] != _RUNTIME_COMPUTED], schema=selected.schema)
        frame = self.scan_selected(stored, start, end, filters)
        output = [r["feature_name"] for r in rows]
        return with_columns_topo(frame, expressions).select(output + self.sort_keys)

    def _lazy(self, schema: pl.Schema, open_stream, start, end, filters) -> pl.LazyFrame:
        """A lazy frame of ``schema`` whose batches come from ``open_stream(names, dnf)``."""
        names = [n for n in schema if n not in self.sort_keys]
        # The row count depends on every table, so a projection without features still
        # reads them all; each table may be longer than the index.
        return _scan(
            schema,
            lambda columns, dnf: open_stream([n for n in names if columns is None or n in columns] or names, dnf),
            self._window(start, end, filters),
            self._index_schema,
            self.sort_keys,
        )

    def index(self, start=None, end=None, filters=None) -> pl.DataFrame:
        """Rows of the index, in no particular order."""
        return _from_stream(self._lake.index(_dnf(self._window(start, end, filters))))

    def scan_index(self, start=None, end=None, filters=None) -> pl.LazyFrame:
        """:meth:`index` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = self._index_schema
        return _scan(schema, lambda _, dnf: self._lake.index(dnf), self._window(start, end, filters), schema, list(schema))

    def feature_metadata(self, filters=None) -> pl.DataFrame:
        schema = _schema(self._lake.feature_metadata_schema())
        return _from_stream(self._lake.feature_metadata(_dnf(_terms(filters, schema))))

    def scan_feature_metadata(self, filters=None) -> pl.LazyFrame:
        """:meth:`feature_metadata` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = _schema(self._lake.feature_metadata_schema())
        return _scan(schema, lambda _, dnf: self._lake.feature_metadata(dnf), _terms(filters, schema), schema, list(schema))

    def table_metadata(self, filters=None) -> pl.DataFrame:
        schema = _schema(self._lake.table_metadata_schema())
        return _from_stream(self._lake.table_metadata(_dnf(_terms(filters, schema))))

    def scan_table_metadata(self, filters=None) -> pl.LazyFrame:
        """:meth:`table_metadata` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = _schema(self._lake.table_metadata_schema())
        return _scan(schema, lambda _, dnf: self._lake.table_metadata(dnf), _terms(filters, schema), schema, list(schema))

    def transaction_log(self, filters=None) -> pl.DataFrame:
        """Rows of the transaction log; none when the lake has no log yet."""
        schema = _schema(self._lake.transaction_log_schema())
        return _from_stream(self._lake.transaction_log(_dnf(_terms(filters, schema))))

    def scan_transaction_log(self, filters=None) -> pl.LazyFrame:
        """:meth:`transaction_log` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = _schema(self._lake.transaction_log_schema())
        return _scan(schema, lambda _, dnf: self._lake.transaction_log(dnf), _terms(filters, schema), schema, list(schema))

    def latest_features(self) -> pl.DataFrame:
        """The latest version of every feature, one metadata row each."""
        return _from_stream(self._lake.latest_features())

    def _add(
        self,
        df,
        signal_type: str,
        owner: str,
        overlap_mode: str,
        feature_params: dict | None,
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        params = json.dumps(feature_params) if feature_params is not None else None
        comments, tags = comments or {}, tags or {}
        names = dict.fromkeys([*comments, *tags])
        annotations = [(name, None, comments.get(name), list(tags[name]) if name in tags else None) for name in names]
        self._lake.add(df, owner=owner, signal_type=signal_type, overlap_mode=overlap_mode, feature_params=params, annotations=annotations)

    def add_features(
        self,
        df,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        feature_params: dict | None = None,
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Adds the non-key columns of ``df`` as features, as ``GammaFeatureLake.add_features`` does.

        ``df`` is a Polars frame or any object implementing ``__arrow_c_stream__``. Rows with a
        null sort key are dropped and the first row of each key is kept. New keys are added to
        the index and every feature and target table is padded so that it stays aligned to the
        index.
        ``overlap_mode`` is ``"copy"`` (write overlapping features to a new table) or
        ``"merge"`` (upsert into the existing one). ``feature_params`` is stored with the new
        versions; without it, existing features keep their previous parameters.
        ``comments`` and ``tags`` map written features to the comment and tags of the versions
        written, as :meth:`annotate` sets them; features left out keep theirs.

        Raises :class:`PartialWriteError` if a table fails to commit after others did.
        """
        self._add(df, "feature", owner, overlap_mode, feature_params, comments, tags)

    def add_targets(
        self,
        df,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        feature_params: dict | None = None,
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """As :meth:`add_features`, registering the columns as targets."""
        self._add(df, "target", owner, overlap_mode, feature_params, comments, tags)

    def add_as_of_features(
        self,
        df,
        params: dict,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """As :meth:`add_features`, for features read with ``join_asof`` on the primary sort key.

        ``params`` holds the ``join_asof`` arguments, such as ``{"strategy": "forward",
        "tolerance": "1h"}``. The table is not padded to the index.
        """
        self._add(df, "as_of_feature", owner, overlap_mode, params, comments, tags)

    def add_sparse_features(
        self,
        df,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """As :meth:`add_features`, for features stored only where they have values and
        left-joined to the index when read."""
        self._add(df, "sparse_feature", owner, overlap_mode, None, comments, tags)

    def add_runtime_computed_features(self, exprs: Sequence[pl.Expr], owner: str = "missing_owner") -> None:
        """Registers each of ``exprs`` as a runtime-computed feature named by its output name.

        Names must be new to the lake. Every column an expression reads must be a feature of
        the lake or another of ``exprs``, other than itself. Requires
        ``runtime_computed_features=True``.
        """
        if not self._runtime_computed_features:
            raise ValueError(
                "Runtime-computed features are disabled. Enable them (runtime_computed_features=True on NativeLake, enable_runtime_computed_features=True on GammaFeatureLake) only when feature metadata is trusted."
            )
        rows = []
        for expr in exprs:
            name, roots = expr.meta.output_name(), expr.meta.root_names()
            if name in roots:
                raise ValueError(
                    f"Invalid input: the expression {expr} provides the same output name as a provided root name. "
                    "Please alias your runtime features uniquely!"
                )
            rows.append((name, expr.meta.serialize(format="json"), [r for r in roots if r not in self.sort_keys]))
        self._lake.add_expressions(rows, owner)

    def annotate(self, annotations: Iterable[tuple[str, str | None, str | None, Sequence[str] | None]]) -> None:
        """Sets the comment and tags of feature versions.

        Each annotation is ``(feature_name, table_addr, comment, tags)``. ``table_addr`` is the
        table of the version to annotate, ``None`` for the latest. A ``None`` comment or tags
        keeps the stored value. Annotations belong to a feature's column in a table, so every
        version stored in that table shares them. Every annotation is checked before any is
        stored; :class:`PartialWriteError` means some were stored before a failure.
        """
        rows = [(name, addr, comment, None if tags is None else list(tags)) for name, addr, comment, tags in annotations]
        self._lake.annotate(rows)

    def annotations(self, features: Iterable[tuple[str, str]]) -> pl.DataFrame:
        """The ``comment`` and ``tags`` of each ``(feature_name, table_addr)``, one row per pair.

        Unannotated features have a null comment and empty tags.
        """
        return _from_stream(self._lake.annotations(list(features)))

    def describe(self, features: Sequence[str] = ()) -> pl.DataFrame:
        """The latest metadata row of each of ``features`` (all features when empty) with its
        ``comment`` and ``tags``."""
        return _from_stream(self._lake.describe(list(features)))

    def restore_to_timestamp(self, timestamp: dt.datetime) -> None:
        """Restores the lake to how it was at ``timestamp``, a timezone-aware datetime.

        The index and metadata tables are restored first, then every table the restored
        feature metadata refers to. Tables with no commit after ``timestamp`` are left as
        they are, and tables created after it are left in place, unreferenced. Restore is not
        atomic across tables: :class:`PartialWriteError` means some tables were restored.
        """
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("restore_to_timestamp requires a timezone-aware datetime")
        micros = (timestamp - _EPOCH) // dt.timedelta(microseconds=1)
        self._lake.restore(micros)

    def consolidate_feature_groups(self, features: Sequence[str]) -> str | None:
        """Moves the latest versions of ``features`` into one new table and returns its
        address, or ``None`` when they already share a table.

        Only features and targets can be consolidated. Each feature gets a new version in the
        new table, with its comment and tags; reads return the same values.
        """
        return self._lake.consolidate(list(features))

    def add_index_rows(self, df) -> None:
        """Adds the sort keys of ``df`` to the index without features; other columns are ignored."""
        self._lake.add_index_rows(df)
