"""Polars-facing Gamma Lake API."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from functools import singledispatchmethod, wraps
from typing import Literal

import polars as pl
import pyarrow as pa
from ccflow import ArrowSchema, BaseModel
from ccflow.exttypes.polars import PolarsExpression
from packaging import version
from pydantic import ConfigDict, Field, model_validator, validate_call

from gammalake.abstract import BaseFeatureLake, Comparable, UninitializedDeltaLakeException
from gammalake.native import ConflictError, DeltaOptions, NativeLake, PartialWriteError, UnsupportedError

__all__ = ("ConflictError", "DeltaOptions", "GammaFeatureLake", "PartialWriteError", "UnsupportedError")

_DEFAULT_INDEX_SCHEMA = ArrowSchema.make(
    pa.schema(
        [
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("symbol", pa.string()),
        ]
    )
)


class GammaFeatureLake(BaseFeatureLake, BaseModel):
    """
    A feature store implementation based on Delta Lake, which focuses on supporting a sortable index on a polars frame and efficient appends/reads.

    Reads and writes go through the native core (:class:`gammalake.native.NativeLake`). Overlapping uploads either write
    the overlapping features to a new table (``overlap_mode="copy"``) or merge them into the existing one (``"merge"``).
    """

    base_path: str = Field(
        description="A local directory or a file://, s3://, az://, abfss:// or gs:// URL, identifying the root directory for storing the "
        "DeltaTables associated with this class."
    )
    feature_metadata_path: str = Field("feature_metadata", description="The name of the DeltaTable holding feature metadata.")
    table_metadata_path: str = Field("table_metadata", description="The name of the DeltaTable holding table metadata.")
    transaction_log_path: str = Field(
        "transaction_log", description="The name of the DeltaTable holding the transaction log, which tracks the status of every write."
    )
    index_path: str = Field("index", description="The name of the DeltaTable holding the global index.")
    table_name_prefix: str = Field(
        "",
        description="Optional prefix prepended to the address of every data table this lake creates. "
        "Useful for integration tests to ensure each test's data tables are isolated and can be cleaned up by prefix.",
    )
    primary_sort_key: str = Field("timestamp", description="The name of the primary key in the index used for comparisons when inserting features.")
    delta_options: DeltaOptions = Field(default_factory=DeltaOptions, description="Configuration for the default Delta storage backend.")
    memory_limit: int | None = Field(
        None,
        description="Bytes a read may hold while sorting the tables it reads; beyond it the sorts spill to the system temporary "
        "directory. None sorts in memory. The frame read() returns is held in memory either way.",
    )
    enable_runtime_computed_features: bool = Field(
        False,
        description="Enables runtime-computed features. Only enable this when feature metadata is trusted because Polars expressions may execute Python code.",
    )
    lock: Callable[[str], AbstractContextManager] | None = Field(
        None,
        repr=False,
        exclude=True,
        description="Called with the absolute path or URL of each table a write commits to; the context manager it returns is held "
        "around that commit. Writers in several processes on an object store without atomic commits need one. Reads take no lock.",
    )
    _is_initialized: bool = False
    _sort_keys: list[str] = None
    _native: tuple | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, hide_input_in_errors=True)

    @model_validator(mode="after")
    def check(self) -> GammaFeatureLake:
        """Open the lake if its tables exist; :meth:`initialize` creates them."""
        try:
            native = self._native_lake()
        except UninitializedDeltaLakeException:
            self._is_initialized = False
            return self
        self._is_initialized = True
        self._sort_keys = native.sort_keys
        return self

    def ensure_deltalake_is_initialized(func):
        """A decorator that delegates to an overridable method, and verifies that the proper initialization steps took place"""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._is_initialized:
                raise UninitializedDeltaLakeException("Please call '.initialize(...)' on your feature store before using this method!")
            return func(self, *args, **kwargs)

        return wrapper

    @property
    def sort_keys(self) -> list[str]:
        """Returns the index table's columm schema."""
        return self._sort_keys

    @property
    def index(self) -> str:
        """Convenience method to find the correct path for this feature store's index table."""
        return os.path.join(self.base_path, self.index_path)

    @property
    def feature_metadata(self) -> str:
        """Convenience method to find the correct path for this feature store's feature metadata table"""
        return os.path.join(self.base_path, self.feature_metadata_path)

    @property
    def table_metadata(self) -> str:
        """Convenience method to find the correct path for this feature store's table metadata table"""
        return os.path.join(self.base_path, self.table_metadata_path)

    @property
    def transaction_log(self) -> str:
        """Convenience method to find the correct path for this feature store's transaction log table"""
        return os.path.join(self.base_path, self.transaction_log_path)

    def _apply_filters_to_lazy_frame(self, df: pl.LazyFrame, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Apply optional filters to a lazyframe, and return it"""
        if start is not None:
            df = df.filter(pl.col(self.primary_sort_key) >= start)
        if end is not None:
            df = df.filter(pl.col(self.primary_sort_key) <= end)
        return df

    @ensure_deltalake_is_initialized
    def table_metadata_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """This feature store's table metadata table."""
        return self._apply_filters_to_lazy_frame(self._native_lake().scan_table_metadata(), start, end)

    @ensure_deltalake_is_initialized
    def transaction_log_frame(self) -> pl.LazyFrame:
        """This feature store's transaction log. A transaction's current status is its row with the latest ``updated_at``."""
        return self._native_lake().scan_transaction_log()

    @ensure_deltalake_is_initialized
    def feature_metadata_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """This feature store's feature metadata table."""
        return self._apply_filters_to_lazy_frame(self._native_lake().scan_feature_metadata(), start, end)

    @ensure_deltalake_is_initialized
    def index_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """This feature store's index table, between ``start`` and ``end`` inclusive."""
        return self._native_lake().scan_index(start, end)

    def get_path(self, addr) -> str:
        """Convenience method to join a feature store's base path to a provided table address"""
        return os.path.join(self.base_path, addr)

    def gen_table_addr(self) -> str:
        """Generate a feature-table address."""
        return f"{self.table_name_prefix}delta_{uuid.uuid4().hex[:8]}"

    @ensure_deltalake_is_initialized
    def annotate(self, annotations: pl.DataFrame) -> None:
        """Attach comments and tags to features one physical table at a time.

        ``annotations`` matches the feature metadata schema with a string ``comment``
        column and a list-of-strings ``tags`` column added.
        """
        expected_schema = {**self.feature_metadata_frame().collect_schema(), "comment": pl.String, "tags": pl.List(pl.String)}
        if dict(annotations.schema) != expected_schema:
            raise ValueError(
                "annotations schema does not match the expected annotation schema. Pass the feature_metadata frame "
                "(from feature_metadata_frame()) for the features you want to annotate, with two columns added: "
                "a string `comment` column and a list-of-strings `tags` column.\n"
                f"  expected: {expected_schema}\n  received: {dict(annotations.schema)}"
            )
        annotations = (
            annotations.with_columns(pl.col("tags").fill_null([]))
            .filter(pl.col("comment").is_not_null() | (pl.col("tags").list.len() > 0))
            .sort("version", descending=True)
            .unique(subset=["table_addr", "feature_name"], keep="first", maintain_order=True)
        )
        self._native_lake().annotate(annotations.select("feature_name", "table_addr", "comment", "tags").rows())

    @ensure_deltalake_is_initialized
    def describe(self, features_metadata: pl.LazyFrame) -> pl.DataFrame:
        """Return selected feature metadata with physical-table annotations appended.

        Annotations are stored per ``(table_addr, feature_name)``, so feature metadata
        versions sharing a physical table receive the same annotation.
        """
        metadata = features_metadata.collect()
        stored = metadata.filter((pl.col("signal_type") != "runtime_computed") & pl.col("table_addr").is_not_null() & (pl.col("table_addr") != "n/a"))
        pairs = stored.select("feature_name", "table_addr").unique(maintain_order=True).rows()
        annotations = self._native_lake().annotations(pairs)
        described = metadata.join(annotations, on=["feature_name", "table_addr"], how="left")
        return described.with_columns(pl.col("tags").fill_null(pl.lit([], dtype=pl.List(pl.String))))

    def initialize(self, schema: ArrowSchema = _DEFAULT_INDEX_SCHEMA) -> GammaFeatureLake:
        """
        Create the index and metadata tables with the provided index schema, where they do not exist yet.
        """
        if self.primary_sort_key not in schema.schema.names:
            raise ValueError(f"The provided primary sort key: {self.primary_sort_key} does not exist in the schema keys: {schema.schema.names}")
        native = self._native_lake(index_schema=schema.schema)
        self._is_initialized = True
        self._sort_keys = native.sort_keys
        return self

    @ensure_deltalake_is_initialized
    def restore_to_timestamp(self, timestamp: datetime) -> GammaFeatureLake:
        """Revert the entire feature store to its state as of ``timestamp``.

        Every Delta commit after ``timestamp`` is undone across the index,
        metadata, and referenced physical data tables. Choose a cutoff between
        the last good write and the writes to undo.

        The system tables are restored first. Restored feature metadata then
        identifies the physical data tables that existed at the cutoff. Tables
        created later are left unreferenced and can be reclaimed separately.
        Runtime-computed features are skipped because their ``n/a`` table
        address does not refer to a physical table.

        Restore is idempotent per table but is not atomic across tables. Run it
        without concurrent writers. The cutoff must be after initialization
        and within Delta's retention window, with all required data files still
        available.

        Args:
            timestamp: Timezone-aware cutoff; commits after it are reverted.

        Returns:
            This feature store, restored in place.

        """
        self._native_lake().restore_to_timestamp(timestamp)
        return self

    @singledispatchmethod
    @ensure_deltalake_is_initialized
    def read(self, *arg) -> pl.DataFrame:
        """Abstract base dispatch method for sequential reads of feature tables"""
        raise TypeError(
            "Invalid input types! Please pass a list of string feature names OR a filtered feature_metadata_table object, and optional start/end Comparables."
        )

    @read.register
    @ensure_deltalake_is_initialized
    def _(
        self,
        features: pl.DataFrame | None = None,
        start: Comparable | None = None,
        end: Comparable | None = None,
        materialized: bool = True,
    ) -> pl.DataFrame:
        duplicates = features.filter(features["feature_name"].is_duplicated())["feature_name"].unique().to_list()
        if duplicates:
            raise ValueError(
                f"feature_metadata DataFrame contains the same feature at multiple versions: {duplicates}. Make separate read() calls per version."
            )
        native = self._native_lake()
        return native.read_selected(features, start, end) if materialized else native.scan_selected(features, start, end)

    @read.register
    @ensure_deltalake_is_initialized
    def _(
        self,
        features: list,
        targets: list | None = None,
        start: Comparable | None = None,
        end: Comparable | None = None,
        materialized: bool = True,
    ) -> pl.DataFrame:
        """
        Given a list of features, identifies the delta tables with the most up-to-date versions of those features,
        and reads them aligned to the index. Pass ``materialized=False`` to apply downstream filters lazily: filters
        on sort keys are pushed down into the index and every feature table. As-of joins retain right-hand
        carry-forward context, while runtime-computed reads apply downstream filters after computation so
        neighbouring-row semantics are preserved.
        """
        native = self._native_lake()
        targets = targets or []
        return native.read(features, targets, start, end) if materialized else native.scan(features, targets, start, end)

    @ensure_deltalake_is_initialized
    def feature_table(self, features: list[str], targets: list[str] | None = None):
        """The latest versions of ``features`` and ``targets`` as a table for DataFusion SQL.

        Needs the ``datafusion`` extra. See :meth:`NativeLake.feature_table`::

            ctx = datafusion.SessionContext()
            ctx.register_table("features", lake.feature_table(["f1", "f2"]))
            ctx.sql("SELECT * FROM features WHERE timestamp >= '2024-01-01'").to_polars()
        """
        return self._native_lake().feature_table(features, targets or [])

    def __getstate__(self):
        # The native handle is a cache that cannot be pickled.
        state = super().__getstate__()
        private = state.get("__pydantic_private__")
        if private and private.get("_native") is not None:
            state = {**state, "__pydantic_private__": {**private, "_native": None}}
        return state

    def _native_lake(self, index_schema: pa.Schema | None = None) -> NativeLake:
        """The native lake for this lake's configuration, reopened when it changes.

        With ``index_schema``, the lake's tables are created first where missing.
        """
        options = {
            "index_path": self.index_path,
            "feature_metadata_path": self.feature_metadata_path,
            "table_metadata_path": self.table_metadata_path,
            "transaction_log_path": self.transaction_log_path,
            "primary_sort_key": self.primary_sort_key,
            "table_name_prefix": self.table_name_prefix,
            "delta_options": self.delta_options,
            "memory_limit": self.memory_limit,
            "runtime_computed_features": self.enable_runtime_computed_features,
        }
        config = (self.base_path, *options.values(), id(self.lock))
        if index_schema is None and self._native is not None and self._native[0] == config:
            return self._native[1]
        options["lock"] = self.lock
        lake = NativeLake(self.base_path, **options) if index_schema is None else NativeLake.create(self.base_path, index_schema, **options)
        self._native = (config, lake)
        return lake

    def _get_latest_feature_tables(self, feature_columns, feature_metadata: pl.DataFrame):
        pure_feature_columns = pl.Series("feature_name", list(set(feature_columns) - set(self.sort_keys)), dtype=pl.String)
        latest = feature_metadata.sort("version", descending=True, nulls_last=True).unique(subset=["feature_name"], keep="first")
        return latest.join(pure_feature_columns.to_frame(), how="right", on="feature_name")

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def _add(
        self,
        df: pl.DataFrame,
        signal_type,
        owner: str = "missing_owner",
        feature_params: dict | None = None,
        overlap_mode: str = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """
        Onboards a new table of features to our FeatureStore.

        Args:
            overlap_mode: ``"copy"`` (default) writes overlapping features to a new DeltaTable, preserving the
                old table as an immutable snapshot.  ``"merge"`` upserts into the existing table in-place.

        Performance:
            Each call scans the global index once and sends newly discovered index rows through a
            single append. Prefer wide-and-shallow writes over tall-and-skinny writes: batch related
            value columns and all secondary-key values for each primary-sort-key period instead of
            splitting the same period across calls. For backfills, write blocks of complete periods
            sized to fit memory. See the ``Best-Practices`` wiki page.

        """
        annotated = [*(comments or {}), *(tags or {})]
        if unknown := sorted({name for name in annotated if name not in df.columns or name in self.sort_keys}):
            raise ValueError(f"comments and tags name columns that are not written features: {unknown}")
        native = self._native_lake()
        if set(df.columns) <= set(self.sort_keys):
            native.add_index_rows(df)
        else:
            native._add(df, signal_type, owner, overlap_mode, feature_params, comments, tags)

    @ensure_deltalake_is_initialized
    def add_targets(
        self,
        df: pl.DataFrame,
        owner: str = "missing_owner",
        overlap_mode: Literal["copy", "merge"] = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Add target columns to the lake.

        Each call scans and appends to the global index once, so batch related columns and
        complete primary-sort-key periods rather than splitting them across calls. See ``_add``
        and the ``Best-Practices`` wiki page for write-shape and backfill guidance.

        ``comments`` and ``tags`` map written columns to the comment and tags of the versions
        written, as :meth:`annotate` stores them. A feature given a comment but no tags keeps
        the tags stored with its table, and the reverse.
        """
        self._add(df=df, signal_type="target", owner=owner, overlap_mode=overlap_mode, comments=comments, tags=tags)

    @ensure_deltalake_is_initialized
    def add_features(
        self,
        df: pl.DataFrame,
        owner: str = "missing_owner",
        overlap_mode: Literal["copy", "merge"] = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Add feature columns to the lake.

        Each call scans and appends to the global index once, so batch related columns and
        complete primary-sort-key periods rather than splitting them across calls. See ``_add``
        and the ``Best-Practices`` wiki page for write-shape and backfill guidance.

        ``comments`` and ``tags`` map written columns to the comment and tags of the versions
        written, as :meth:`annotate` stores them. A feature given a comment but no tags keeps
        the tags stored with its table, and the reverse.
        """
        self._add(df=df, signal_type="feature", owner=owner, overlap_mode=overlap_mode, comments=comments, tags=tags)

    @ensure_deltalake_is_initialized
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_index_rows(self, df: pl.DataFrame) -> None:
        """Extend the global index without adding features.

        No feature or table metadata is created. Dense feature tables are padded only where
        the new rows overlap their covered range. Non-overlapping dense tables, as-of
        tables, and sparse tables are left untouched.

        Args:
            df: A DataFrame containing the sort-key columns. Other columns are ignored.

        """
        self._native_lake().add_index_rows(df.select(self.sort_keys))

    @ensure_deltalake_is_initialized
    def consolidate_feature_groups(self, features: list[str]) -> str | None:
        """Merge the current FeatureGroup tables for the given features into one table.

        Only ``"feature"`` and ``"target"`` signal types may be consolidated. Each feature
        gets a new version in the new table, with its comment and tags.

        Args:
            features: Feature names to consolidate. They should share an update cadence.

        Returns:
            The new table address, or ``None`` if the features are already consolidated.

        Raises:
            MissingFeaturesException: If any requested feature does not exist.
            ValueError: If any requested feature has a non-consolidatable signal type.
        """
        return self._native_lake().consolidate_feature_groups(features)

    @ensure_deltalake_is_initialized
    def add_as_of_features(
        self,
        df: pl.DataFrame,
        params: dict,
        owner: str = "missing_owner",
        overlap_mode: Literal["copy", "merge"] = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Calls _add for as_of feature signal_types; see :meth:`add_features` for ``comments`` and ``tags``."""
        self._add(df=df, signal_type="as_of_feature", owner=owner, feature_params=params, overlap_mode=overlap_mode, comments=comments, tags=tags)

    @ensure_deltalake_is_initialized
    def add_sparse_features(
        self,
        df: pl.DataFrame,
        owner: str = "missing_owner",
        overlap_mode: Literal["copy", "merge"] = "copy",
        comments: Mapping[str, str] | None = None,
        tags: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Add sparse features to this FeatureLake.

        Sparse features are stored without null sentinel rows — only rows with actual values
        are written to the physical Delta table.  At read time they are left-joined back to the
        global index, so every index row is returned with ``null`` where the sparse feature has
        no observation.

        Use sparse features for data that is genuinely sparse (e.g. corporate actions, earnings
        events, issuer attributes) where storing a null for every missing index row would be
        wasteful.  For slowly-changing data where you want the last known value carried forward,
        use :meth:`add_as_of_features` instead. See :meth:`add_features` for ``comments`` and
        ``tags``.
        """
        self._add(df=df, signal_type="sparse_feature", owner=owner, overlap_mode=overlap_mode, comments=comments, tags=tags)

    @ensure_deltalake_is_initialized
    def add_runtime_computed_features(self, exprs=list[PolarsExpression], owner: str = "missing_owner") -> None:
        """Add (and possibly increment the version) a serialized expression representing an adhoc runtime feature computation"""
        if not self.enable_runtime_computed_features:
            raise ValueError(
                "Runtime-computed features are disabled. Set enable_runtime_computed_features=True only when feature metadata is trusted."
            )
        self._native_lake().add_runtime_computed_features(exprs, owner)

    @ensure_deltalake_is_initialized
    def add_runtime_transforms(self, features: list[str], transforms: list[str], owner: str = "missing_owner") -> None:
        """Register row-local transforms over stored features as runtime features.

        This convenience wrapper over :meth:`add_runtime_computed_features` registers every
        ``(feature, transform)`` pair as ``<feature>__<transform>``. Numeric outputs use
        ``Float64`` and propagate nulls; ``reciprocal`` also returns null for zero. ``missing``,
        ``nan``, and ``nonzero`` return booleans. ``positive`` and ``negative`` clamp to the
        non-negative part of the value and its negation, respectively, rather than returning
        sign predicates. ``sign`` returns ``-1``, ``0``, or ``1``, and ``signed_log1p`` computes
        ``sign(x) * log1p(abs(x))``.

        Args:
            features: Names of registered features to transform. Except when only ``missing`` is
                requested, each feature must be numeric when read.
            transforms: Transform names applied to every feature. Supported names are ``abs``,
                ``missing``, ``nan``, ``negative``, ``nonzero``, ``positive``, ``reciprocal``,
                ``sign``, ``signed_log1p``, and ``squared``.
            owner: Owner recorded for each generated runtime feature.

        Raises:
            ValueError: If either list is empty or contains duplicates, a transform is unknown,
                a generated name conflicts with a sort key, runtime-computed features are
                disabled, a root feature is unregistered, or an output name is already registered.

        """
        for name, values in (("features", features), ("transforms", transforms)):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be a non-empty list without duplicates")

        builders = {
            "abs": lambda column: column.cast(pl.Float64).abs(),
            "missing": lambda column: column.is_null(),
            "nan": lambda column: column.cast(pl.Float64).is_nan(),
            "negative": lambda column: (-column.cast(pl.Float64)).clip(lower_bound=0),
            "nonzero": lambda column: column.cast(pl.Float64) != 0,
            "positive": lambda column: column.cast(pl.Float64).clip(lower_bound=0),
            "reciprocal": lambda column: pl.when(column.cast(pl.Float64) != 0).then(1 / column.cast(pl.Float64)),
            "sign": lambda column: column.cast(pl.Float64).sign(),
            "signed_log1p": lambda column: column.cast(pl.Float64).sign() * column.cast(pl.Float64).abs().log1p(),
            "squared": lambda column: column.cast(pl.Float64).pow(2),
        }
        if unknown := sorted(set(transforms) - set(builders)):
            raise ValueError(f"Unknown runtime transforms: {unknown}. Supported transforms: {sorted(builders)}")

        generated_names = {f"{feature}__{transform}" for feature in features for transform in transforms}
        if conflicts := sorted(generated_names & set(self.sort_keys)):
            raise ValueError(f"Runtime transform feature names conflict with sort keys: {conflicts}")

        expressions = [builders[transform](pl.col(feature)).alias(f"{feature}__{transform}") for feature in features for transform in transforms]
        self.add_runtime_computed_features(expressions, owner=owner)

    @staticmethod
    def merge(
        left: GammaFeatureLake,
        right: GammaFeatureLake,
        left_features: list | pl.DataFrame,
        right_features: list | pl.DataFrame,
        left_targets: list | None = None,
        right_targets: list | None = None,
        start: Comparable | None = None,
        end: Comparable | None = None,
        suffix: str = "_right",
    ) -> pl.DataFrame:
        """Read from two GammaFeatureLakes with the same sort keys and full-outer-join their outputs.

        Both lakes must share identical ``sort_keys`` (i.e. the same ``primary_sort_key`` and
        secondary key columns).  Features present in only one lake will appear with nulls in the
        rows sourced from the other lake.

        The join exploits the fact that both ``read()`` calls return frames that are already sorted
        on ``sort_keys``.  Rather than building a hash table (the default eager join strategy),
        we route through Polars' streaming execution engine.  On Polars ≥
        ``_STREAMING_MERGE_JOIN_MIN_VERSION`` the ``primary_sort_key`` column is declared sorted
        (via ``set_sorted``) so the optimizer promotes the join to an O(n+m) sort-merge join,
        which is typically **3–18× faster** on large frames.  On older Polars versions the same
        streaming engine is used but falls back to a hash join (correct, just without the
        sort-merge optimisation).

        Args:
            left: First GammaFeatureLake instance.
            right: Second GammaFeatureLake instance.  Must have the same ``sort_keys`` as *left*.
            left_features: Feature names to read from *left*, either as a list of strings or a
                pre-filtered ``feature_metadata`` DataFrame (e.g. to pin a specific version).
            right_features: Feature names to read from *right*, either as a list of strings or a
                pre-filtered ``feature_metadata`` DataFrame.
            left_targets: Target names to read from *left*.  Ignored when ``left_features`` is a
                DataFrame (targets are encoded in the metadata frame).  Defaults to ``[]``.
            right_targets: Target names to read from *right*.  Ignored when ``right_features`` is a
                DataFrame.  Defaults to ``[]``.
            start: Inclusive lower bound on ``primary_sort_key``.  Applied to both lakes.
            end: Inclusive upper bound on ``primary_sort_key``.  Applied to both lakes.
            suffix: Suffix appended to non-key columns whose names clash between the two lakes.

        Returns:
            A ``pl.DataFrame`` containing all rows from both lakes, sorted on ``sort_keys``.
            Rows present in only one lake have nulls in the columns sourced from the other.

        Raises:
            ValueError: If the two lakes have different ``sort_keys``.
        """
        if left.sort_keys != right.sort_keys:
            raise ValueError(f"Cannot merge two GammaFeatureLakes with different sort_keys: {left.sort_keys!r} vs {right.sort_keys!r}")

        def _read_lazy(lake: GammaFeatureLake, features: list | pl.DataFrame, targets: list | None) -> pl.LazyFrame:
            if isinstance(features, pl.DataFrame):
                return lake.read(features, start=start, end=end, materialized=False)
            return lake.read(features, targets or [], start=start, end=end, materialized=False)

        on = left.sort_keys
        left_lf = _read_lazy(left, left_features, left_targets)
        right_lf = _read_lazy(right, right_features, right_targets)

        # Polars ≥ 1.37.0: hint that both frames are sorted so the streaming engine
        # uses an O(n+m) sort-merge join instead of a hash join.
        if version.parse(pl.__version__) >= version.parse("1.37.0"):
            left_lf = left_lf.set_sorted(left.primary_sort_key)
            right_lf = right_lf.set_sorted(right.primary_sort_key)

        return left_lf.join(right_lf, on=on, how="full", coalesce=True, suffix=suffix).collect(engine="streaming")
