"""Read and write a gamma-lake through the native (Rust) core.

``NativeLake`` reads and writes the same Delta tables as :class:`~gammalake.GammaFeatureLake`,
which is built on it, and returns Polars frames. Feature tables are aligned to the index by
row position, and writes pad every table so that this alignment holds. As-of and sparse
features are stored unpadded and joined to the index when read.

I/O failures that come with an operating system error raise the matching ``OSError``
subclass, such as ``PermissionError``; other storage and engine errors raise
``RuntimeError``. Errors raised while a stream is being consumed (for example a
failed integrity check with ``validate=True``) surface as ``RuntimeError`` from
:meth:`NativeLake.read` whatever their cause, and errors raised while a
:meth:`NativeLake.scan` is collected surface as a Polars ``ComputeError``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
from polars_io_tools.io_sources.base import (
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ErrorNode,
    FunctionNode,
    TernaryNode,
    UnknownNode,
    get_parsed_expr,
)
from polars_io_tools.io_sources.dnf_visitor import convert_expr_to_dnf
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType
from polars_io_tools.io_sources.util import register_io_source_with_is_pure
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gammalake import gammalake as _native
from gammalake._topo import deserialize_expression, with_columns_topo
from gammalake.abstract import MissingFeaturesException

__all__ = ("ConflictError", "DeltaOptions", "NativeLake", "PartialWriteError", "UnsupportedError")

Row = tuple[str, str | None, str, str | None]
"""``(feature_name, table_addr, signal_type, feature_params)``, as the core reads a feature."""

Term = tuple[str, str, pa.Array | None]

_MAX_CLAUSES = 64

_PUSHED_OPS = {
    "=": "==",
    "==": "==",
    "!=": "!=",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "!in": "not_in",
    "in": "in",
    "not in": "not_in",
}


class DeltaOptions(BaseModel):
    """Configuration owned by the default Delta storage backend."""

    compression: str = "zstd"
    target_file_size: int = Field(128 * 1024 * 1024, gt=0)
    storage_options: dict[str, str] = Field(default_factory=dict, repr=False)

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    @field_validator("compression")
    @classmethod
    def validate_compression(cls, value: str) -> str:
        compressions = ("uncompressed", "snappy", "gzip", "brotli", "lz4", "lz4_raw", "zstd")
        if value.lower() not in compressions:
            raise ValueError(f"expected one of {compressions}, got {value!r}")
        return value


class UnsupportedError(NotImplementedError):
    """The lake cannot serve this request, such as a read of a feature whose signal type it
    does not know or a write of a column type it cannot store."""

    def __init__(self, message: str, feature: str | None = None, signal_type: str | None = None):
        super().__init__(message)
        self.feature = feature
        self.signal_type = signal_type


class ConflictError(RuntimeError):
    """Another writer committed to ``table`` after this write read it, on the last of its 10 attempts.

    The lake is unchanged by the failed write, apart from tables it created and left
    unreferenced, so the write can be run again.
    """

    def __init__(self, message: str, table: str):
        super().__init__(message)
        self.table = table


class PartialWriteError(RuntimeError):
    """A write failed after some of its tables committed, and they could not be rolled back.

    ``committed`` names the tables that may hold part of the write; ``phase`` names the step
    that failed. When a lock failed, its exception is the ``__cause__``. A write that is rolled
    back raises the exception it failed with instead, noting the step that failed.
    """

    def __init__(self, message: str, phase: str, committed: list[str]):
        super().__init__(message)
        self.phase = phase
        self.committed = committed


def _coerce(column: str, dtype: pl.DataType, value: Any) -> Any:
    """``value`` as a Python object of ``dtype``, compared as Polars would compare it."""
    if isinstance(value, np.datetime64) and np.datetime_data(value.dtype)[0] in ("ns", "ps", "fs", "as"):
        if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns" and dtype.time_zone is None:
            # Nanoseconds since the epoch, which a nanosecond column compares exactly.
            return int(value.astype("datetime64[ns]").astype(np.int64))
        # ``item`` gives an int for units finer than Python's microseconds. Round down to whole
        # microseconds, as Polars, and so 0.2, compares such a literal with a microsecond column.
        value = value.astype("datetime64[us]")
    if isinstance(value, np.generic):
        value = value.item()
    ok = False
    if isinstance(dtype, pl.Datetime):
        if isinstance(value, dt.datetime):
            literal_tz = pl.Series([value]).dtype.time_zone
            ok = literal_tz == dtype.time_zone and (dtype.time_unit != "ms" or value.microsecond % 1000 == 0)
        elif isinstance(value, dt.date):
            # Polars compares a date with a datetime as midnight, in UTC when it is zoned.
            value, ok = dt.datetime.combine(value, dt.time(), tzinfo=dt.UTC if dtype.time_zone else None), True
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
    except (OverflowError, TypeError, pl.exceptions.PolarsError) as error:
        raise UnsupportedError(f"cannot represent {values!r} as {column!r} ({dtype})") from error
    if dtype == pl.String:
        # The core compares Utf8 literals with any string column.
        return series.to_arrow().cast(pa.string())
    return series.to_arrow()


def _and(left: list[list[Term]], right: list[list[Term]]) -> list[list[Term]]:
    return [a + b for a in left for b in right]


def _pushdown(predicate: pl.Expr | None, schema: pl.Schema, columns: Iterable[str]) -> list[list[Term]]:
    """The parts of ``predicate`` on ``columns`` that the core evaluates, as DNF.

    ``convert_expr_to_dnf`` gives a DNF selecting every row that ``predicate`` selects. Terms
    the core cannot evaluate, or on other or float columns (the core orders NaNs differently
    from Polars), are dropped from their clause, which only widens it, so ``predicate`` must
    still be applied to what is read. Each top-level conjunct is converted on its own, smallest
    first, and those that would push the DNF past ``_MAX_CLAUSES`` are dropped.
    """
    node = _parse(predicate)
    if node is None or _row_aligned(node):
        return [[]]
    pushable = {c for c in columns if not schema[c].is_float()}
    pushed: list[list[Term]] = [[]]
    for conjunct in sorted(_conjuncts(node), key=lambda n: _dnf_size(n)[0]):
        dnf = _pushed_dnf(conjunct, schema, pushable)
        if dnf and len(pushed) * len(dnf) <= _MAX_CLAUSES:
            pushed = _and(pushed, dnf)
    return pushed


def _conjuncts(node: BaseExprNode) -> list[BaseExprNode]:
    match node:
        case BinaryExprNode(op=OperatorType.AND | OperatorType.LOGICAL_AND):
            return _conjuncts(node.left) + _conjuncts(node.right)
        case FunctionNode(function_type=BooleanFunctionType.ALL_HORIZONTAL):
            return [c for n in node.inputs for c in _conjuncts(n)]
    return [node]


def _pushed_dnf(node: BaseExprNode, schema: pl.Schema, pushable: set[str]) -> list[list[Term]] | None:
    """The pushed DNF of ``node``, or ``None`` when it is too large or restricts nothing."""
    if _dnf_size(node)[0] > _MAX_CLAUSES:
        return None
    dnf = convert_expr_to_dnf(node, schema)
    if not dnf or len(dnf) > _MAX_CLAUSES:
        return None
    clauses = [[t for c, op, v in clause if c in pushable and (t := _pushed_term(c, op, v, schema[c])) is not None] for clause in dnf]
    return None if [] in clauses else clauses


def _dnf_size(node: BaseExprNode) -> tuple[int, int]:
    """Upper bounds on the clauses, and on their width, of ``convert_expr_to_dnf(node)``.

    The expansion is exponential: an AND of ORs multiplies their clauses, and a NOT of ``n``
    clauses of width ``w`` gives ``w ** n``, so ``_pushed_dnf`` bounds it before expanding.
    Counts past ``_MAX_CLAUSES`` are capped.
    """
    cap = _MAX_CLAUSES + 1
    match node:
        case AliasNode() | CastNode():
            return _dnf_size(node.input)
        case BinaryExprNode(op=OperatorType.AND | OperatorType.LOGICAL_AND):
            return _dnf_and([_dnf_size(node.left), _dnf_size(node.right)])
        case BinaryExprNode(op=OperatorType.OR | OperatorType.LOGICAL_OR):
            return _dnf_or([_dnf_size(node.left), _dnf_size(node.right)])
        case BinaryExprNode(op=OperatorType.NOT_EQ_VALIDITY):
            return 2, 1
        case FunctionNode(function_type=BooleanFunctionType.ALL_HORIZONTAL):
            return _dnf_and([_dnf_size(n) for n in node.inputs])
        case FunctionNode(function_type=BooleanFunctionType.ANY_HORIZONTAL):
            return _dnf_or([_dnf_size(n) for n in node.inputs])
        case FunctionNode(function_type=BooleanFunctionType.IS_IN):
            return 2, 1
        case FunctionNode(function_type=BooleanFunctionType.IS_BETWEEN):
            return 1, 2
        case FunctionNode(function_type=BooleanFunctionType.NOT, inputs=[inner, *_]):
            clauses, width = _dnf_size(inner)
            return min(width**clauses, cap), clauses
        case TernaryNode():
            # (P & T) | ((~P | P's columns are null) & F)
            (p, pw), (t, tw), (f, fw) = map(_dnf_size, (node.predicate, node.truthy, node.falsy))
            return min(p * t + (min(pw**p, cap) + p * pw) * f, cap), max(pw + tw, p + fw)
    return 1, 1


def _dnf_and(sizes: list[tuple[int, int]]) -> tuple[int, int]:
    return min(math.prod(c for c, _ in sizes), _MAX_CLAUSES + 1), sum(w for _, w in sizes)


def _dnf_or(sizes: list[tuple[int, int]]) -> tuple[int, int]:
    return min(sum(c for c, _ in sizes), _MAX_CLAUSES + 1), max((w for _, w in sizes), default=1)


def _pushed_term(column: str, op: str, value: Any, dtype: pl.DataType) -> Term | None:
    if value is None and op in ("is", "is not"):
        return (column, "is_null" if op == "is" else "is_not_null", None)
    if op not in _PUSHED_OPS or (op in ("not in", "!in") and not len(value)):
        # The core keeps no null rows for ``not in []``, unlike Polars.
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
            # Per-row literals must see the whole frame at once, and Polars 1 splits even an eager
            # filter across threads, so the mask is computed first.
            collected = list(frames)
            frame = pl.concat(collected, rechunk=True) if collected else pl.DataFrame(schema=schema)
            frames, predicate = [frame.filter(frame.select(predicate).to_series())], None
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


def _row(name: str, addr: str | None, signal_type: str | None, params: str | None) -> Row:
    if addr is None and signal_type != "runtime_computed":
        raise UnsupportedError(f"feature {name!r} has no table", name, signal_type)
    return (name, addr, signal_type or "", params)


def _selected_rows(features: pl.DataFrame) -> list[Row]:
    """The rows of a feature metadata frame."""
    params = features["feature_params"] if "feature_params" in features.columns else [None] * features.height
    return [_row(*row) for row in zip(features["feature_name"], features["table_addr"], features["signal_type"], params)]


def _has_runtime(rows: list[Row]) -> bool:
    return any(row[2] == "runtime_computed" for row in rows)


def _by_address(rows: list[Row]) -> dict[str, Row]:
    """``rows`` by name, each read the way the first row at its table is read.

    A table's signal type and params come from the first of its selected rows, so a lazy
    frame that reads a subset of the rows must keep them.
    """
    first: dict[str, Row] = {}
    for row in rows:
        first.setdefault(row[1], row)
    return {name: (name, addr, first[addr][2], first[addr][3]) for name, addr, _, _ in rows}


def _from_stream(stream) -> pl.DataFrame:
    try:
        table = pa.RecordBatchReader.from_stream(stream).read_all()
    except pa.ArrowException as error:
        raise RuntimeError(str(error)) from error
    return pl.from_arrow(table, rechunk=False)


class NativeLake:
    """A gamma-lake read and written through the native core.

    Every call reads the latest version of each table. ``start`` and ``end`` are inclusive
    bounds on the primary sort key and filter the index and every feature table before
    they are aligned.

    ``base_path`` is a local directory or a ``file://``, ``s3://``, ``az://``, ``abfss://`` or
    ``gs://`` URL. ``delta_options.storage_options`` are delta-rs object store options such as
    ``AWS_REGION`` or ``AWS_ENDPOINT_URL``; credentials they omit come from the environment.

    ``lock`` is called with the absolute path or URL of each table a write commits to, and
    the context manager it returns is held around that commit; it serialises commits but does
    not make a write atomic. Each write is logged at ``transaction_log_path``; one that fails
    after committing part of itself is rolled back or raises :class:`PartialWriteError`, and one
    that conflicts with another writer is rolled back and run again.
    See "Concurrent writers" and "Transactions and rollback" in the README.

    By default a read sorts each table it reads in memory. With ``memory_limit`` set, the
    tables of a read share that many bytes for sorting and spill to the system temporary
    directory (``TMPDIR``) beyond it, so reads larger than memory can be streamed with
    :meth:`scan`. As-of and sparse tables are read into memory outside the limit, and so is
    a whole scan filtered by a predicate with a literal per row, such as a Series.

    Runtime-computed features are Polars expressions stored in the feature metadata and
    evaluated when read. Deserializing an expression can run code, so reading or registering
    them requires ``runtime_computed_features=True``. Anyone who can write the feature
    metadata can then run code in every process that reads it, so enable it only where
    write access to the lake is limited to those trusted to run code.
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
        lock: Callable[[str], AbstractContextManager] | None = None,
        memory_limit: int | None = None,
        transaction_log_path: str = "transaction_log",
    ):
        lake = _native.Lake(
            base_path,
            primary_sort_key=primary_sort_key,
            index_path=index_path,
            feature_metadata_path=feature_metadata_path,
            table_metadata_path=table_metadata_path,
            batch_size=batch_size,
            validate=validate,
            table_name_prefix=table_name_prefix,
            delta_options=delta_options,
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
        self._init(_native.Lake(base_path, index_schema=index_schema, **options), runtime_computed_features)
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

    def _window(self, start, end) -> list[list[Term]]:
        bounds = [
            (self.primary_sort_key, op, _encode(self.primary_sort_key, self._index_schema[self.primary_sort_key], [value]))
            for op, value in ((">=", start), ("<=", end))
            if value is not None
        ]
        return [bounds]

    def read_schema(self, features: list[str], targets: list[str] | None = None) -> pl.Schema:
        """Schema of :meth:`read`: resolved names in request order, then the sort keys."""
        return self._scan_rows(self._lake.resolve(features, targets)).collect_schema()

    def read_arrow(self, features: list[str], targets: list[str] | None = None, start=None, end=None):
        """The result of :meth:`read` as an Arrow C stream (``__arrow_c_stream__``), consumable once."""
        return self._read_rows(self._lake.resolve(features, targets), start, end)

    def read(self, features: list[str], targets: list[str] | None = None, start=None, end=None) -> pl.DataFrame:
        """Features and targets aligned to the index, sorted by the sort keys."""
        return _from_stream(self.read_arrow(features, targets, start, end))

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

    def scan(self, features: list[str], targets: list[str] | None = None, start=None, end=None) -> pl.LazyFrame:
        """A lazy :meth:`read`.

        Only the selected features are read. Predicates added to the returned lazy frame are
        pushed into the native scan where they constrain the sort keys, and are applied again
        after it.
        """
        return self._scan_rows(self._lake.resolve(features, targets), start, end)

    def read_versions_schema(self, features: Mapping[str, int]) -> pl.Schema:
        """Schema of :meth:`read_versions`."""
        return self._scan_rows(self._lake.resolve_versions(list(features.items()))).collect_schema()

    def read_versions(self, features: Mapping[str, int], start=None, end=None) -> pl.DataFrame:
        """Each feature at the version it maps to, read as :meth:`read` reads the latest versions.

        The features a runtime-computed feature reads are taken at their latest versions.
        """
        return _from_stream(self._read_rows(self._lake.resolve_versions(list(features.items())), start, end))

    def scan_versions(self, features: Mapping[str, int], start=None, end=None) -> pl.LazyFrame:
        """A lazy :meth:`read_versions`."""
        return self._scan_rows(self._lake.resolve_versions(list(features.items())), start, end)

    def read_selected_schema(self, features: pl.DataFrame) -> pl.Schema:
        """Schema of :meth:`read_selected`."""
        return self._scan_rows(_selected_rows(features)).collect_schema()

    def read_selected(self, features: pl.DataFrame, start=None, end=None) -> pl.DataFrame:
        """The features that rows of :meth:`feature_metadata` describe, one row per name.

        Uses the ``feature_name``, ``table_addr``, ``signal_type`` and ``feature_params``
        columns. The rows are not checked against the stored metadata. Features that
        runtime-computed rows read and the rows do not select are taken at their latest versions.
        """
        return _from_stream(self._read_rows(_selected_rows(features), start, end))

    def scan_selected(self, features: pl.DataFrame, start=None, end=None) -> pl.LazyFrame:
        """A lazy :meth:`read_selected`."""
        return self._scan_rows(_selected_rows(features), start, end)

    def _read_rows(self, rows: list[Row], start, end):
        """An Arrow C stream of the features ``rows`` describe."""
        if _has_runtime(rows):
            return self._runtime(rows, start, end).collect().to_arrow()
        return self._lake.read_selected(rows, _dnf(self._window(start, end)))

    def _scan_rows(self, rows: list[Row], start=None, end=None) -> pl.LazyFrame:
        """A lazy read of the features ``rows`` describe."""
        if _has_runtime(rows):
            return self._runtime(rows, start, end)
        schema = _schema(self._lake.read_selected_schema(rows))
        by_name = _by_address(rows)

        def open_stream(columns: list[str] | None, dnf: list[list[Term]]):
            # The row count depends on every table, so a projection without features still
            # reads them all; each table may be longer than the index.
            wanted = [row for name, row in by_name.items() if columns is None or name in columns]
            return self._lake.read_selected(wanted or list(by_name.values()), dnf)

        return _scan(schema, open_stream, self._window(start, end), self._index_schema, self.sort_keys)

    def _runtime(self, rows: list[Row], start, end) -> pl.LazyFrame:
        """A lazy read of ``rows`` that computes the runtime-computed ones.

        Stored features the expressions read are added at their latest versions, as are
        runtime-computed ones, transitively; the result holds the selected names, then the
        added ones in the order found, then the sort keys.
        """
        self._ensure_runtime_enabled()
        selected: dict[str, Row] = {}
        for row in rows:
            selected.setdefault(row[0], row)
        rows = list(selected.values())
        latest = None
        expressions = []
        # Rows found while iterating are appended to ``rows`` and visited in turn.
        for name, _, signal_type, params in rows:
            if signal_type != "runtime_computed":
                continue
            expression = deserialize_expression(params)
            expressions.append(expression)
            for root in expression.meta.root_names():
                if root in selected or root in self.sort_keys:
                    continue
                if latest is None:
                    latest = {r["feature_name"]: r for r in self.latest_features().iter_rows(named=True)}
                if root not in latest:
                    raise MissingFeaturesException(f"runtime-computed feature {name!r} reads {root!r}, which is not in the lake")
                found = latest[root]
                selected[root] = _row(root, found["table_addr"], found["signal_type"], found["feature_params"])
                rows.append(selected[root])
        frame = self._scan_rows([row for row in rows if row[2] != "runtime_computed"], start, end)
        return with_columns_topo(frame, expressions).select([row[0] for row in rows] + self.sort_keys)

    def _ensure_runtime_enabled(self) -> None:
        if not self._runtime_computed_features:
            raise ValueError(
                "Runtime-computed features are disabled. Enable them (runtime_computed_features=True on NativeLake, "
                "enable_runtime_computed_features=True on GammaFeatureLake) only when feature metadata is trusted."
            )

    def index(self, start=None, end=None) -> pl.DataFrame:
        """Rows of the index, in no particular order."""
        return _from_stream(self._lake.index(_dnf(self._window(start, end))))

    def scan_index(self, start=None, end=None) -> pl.LazyFrame:
        """:meth:`index` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = self._index_schema
        return _scan(schema, lambda _, dnf: self._lake.index(dnf), self._window(start, end), schema, list(schema))

    def feature_metadata(self) -> pl.DataFrame:
        return _from_stream(self._lake.feature_metadata([]))

    def scan_feature_metadata(self) -> pl.LazyFrame:
        """:meth:`feature_metadata` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = _schema(self._lake.feature_metadata_schema())
        return _scan(schema, lambda _, dnf: self._lake.feature_metadata(dnf), [[]], schema, list(schema))

    def table_metadata(self) -> pl.DataFrame:
        return _from_stream(self._lake.table_metadata([]))

    def scan_table_metadata(self) -> pl.LazyFrame:
        """:meth:`table_metadata` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = _schema(self._lake.table_metadata_schema())
        return _scan(schema, lambda _, dnf: self._lake.table_metadata(dnf), [[]], schema, list(schema))

    def transaction_log(self) -> pl.DataFrame:
        """Rows of the transaction log; none when the lake has no log yet."""
        return _from_stream(self._lake.transaction_log([]))

    def scan_transaction_log(self) -> pl.LazyFrame:
        """:meth:`transaction_log` as a ``LazyFrame``. Filters applied to it later are pushed into the scan."""
        schema = _schema(self._lake.transaction_log_schema())
        return _scan(schema, lambda _, dnf: self._lake.transaction_log(dnf), [[]], schema, list(schema))

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
        allow_new_index_values: bool = True,
    ) -> None:
        params = json.dumps(feature_params) if feature_params is not None else None
        comments, tags = comments or {}, tags or {}
        names = dict.fromkeys([*comments, *tags])
        annotations = [(name, None, comments.get(name), list(tags[name]) if name in tags else None) for name in names]
        self._lake.add(
            df,
            owner=owner,
            signal_type=signal_type,
            overlap_mode=overlap_mode,
            feature_params=params,
            annotations=annotations,
            allow_new_index_values=allow_new_index_values,
        )

    def add_features(
        self,
        df,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        feature_params: dict | None = None,
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
        *,
        allow_new_index_values: bool = True,
    ) -> None:
        """Adds the non-key columns of ``df`` as features.

        ``df`` is a Polars frame or any object implementing ``__arrow_c_stream__``. Rows with a
        null sort key are dropped and the first row of each key is kept. New keys are added to
        the index and every feature and target table is padded so that it stays aligned to the
        index; with ``allow_new_index_values=False``, an input with keys not in the index raises
        ``ValueError`` and writes nothing.

        ``overlap_mode`` is ``"copy"`` (write overlapping features to a new table) or
        ``"merge"`` (upsert into the existing one). ``feature_params`` is stored with the new
        versions; without it, existing features keep their previous parameters.
        ``comments`` and ``tags`` map written features to the comment and tags of the versions
        written, as :meth:`annotate` sets them; features left out keep theirs.

        A write that conflicts with another writer's commit is rolled back and run again, up to
        10 times in all, and raises :class:`ConflictError` if every attempt conflicts. Any other
        failure rolls the write back and raises its exception, or :class:`PartialWriteError` if
        the rollback cannot undo every commit.
        """
        self._add(df, "feature", owner, overlap_mode, feature_params, comments, tags, allow_new_index_values)

    def add_targets(
        self,
        df,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        feature_params: dict | None = None,
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
        *,
        allow_new_index_values: bool = True,
    ) -> None:
        """As :meth:`add_features`, registering the columns as targets."""
        self._add(df, "target", owner, overlap_mode, feature_params, comments, tags, allow_new_index_values)

    def add_as_of_features(
        self,
        df,
        params: dict,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
        *,
        allow_new_index_values: bool = True,
    ) -> None:
        """As :meth:`add_features`, for features read with ``join_asof`` on the primary sort key.

        ``params`` holds the ``join_asof`` arguments, such as ``{"strategy": "forward",
        "tolerance": "1h"}``. The table is not padded to the index.
        """
        self._add(df, "as_of_feature", owner, overlap_mode, params, comments, tags, allow_new_index_values)

    def add_sparse_features(
        self,
        df,
        owner: str = "missing_owner",
        overlap_mode: str = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
        *,
        allow_new_index_values: bool = True,
    ) -> None:
        """As :meth:`add_features`, for features stored only where they have values and
        left-joined to the index when read."""
        self._add(df, "sparse_feature", owner, overlap_mode, None, comments, tags, allow_new_index_values)

    def add_runtime_computed_features(self, exprs: Sequence[pl.Expr], owner: str = "missing_owner") -> None:
        """Registers each of ``exprs`` as a runtime-computed feature named by its output name.

        Names must be new to the lake. Every column an expression reads must be a feature of
        the lake or another of ``exprs``, other than itself. Requires
        ``runtime_computed_features=True``.
        """
        self._ensure_runtime_enabled()
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
        atomic across tables: :class:`PartialWriteError` means some tables were restored. Run
        it with no writers in other processes, while the Delta logs and files of
        ``timestamp`` are still retained.
        """
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("restore_to_timestamp requires a timezone-aware datetime")
        micros = (timestamp - dt.datetime(1970, 1, 1, tzinfo=dt.UTC)) // dt.timedelta(microseconds=1)
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
