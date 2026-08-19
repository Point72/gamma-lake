"""Shared test logic for GammaFeatureLake.

Each ``_test_*`` method receives a fully initialized ``GammaFeatureLake``
instance and performs all assertions through ``fs.io.*`` so the scenarios
remain backend agnostic.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from ccflow import ArrowSchema
from packaging import version
from polars.testing import assert_frame_equal

from gammalake import (
    GammaFeatureLake,
    MissingFeaturesException,
    MissingOrMisregisteredSignalsException,
    UninitializedDeltaLakeException,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def generate_test_data(
    feature_ids_start=0,
    symbols_id_start=0,
    n_symbols=5,
    n_features=3,
    n_days=30,
    start_date=datetime(2020, 6, 20, tzinfo=UTC),
    feature_suffix="feature",
):
    """Generate test data directly in polars."""
    feature_expressions = [
        pl.lit(np.random.normal(loc=100, scale=15, size=n_days)).alias(f"{feature_suffix}_{i}")
        for i in range(feature_ids_start, feature_ids_start + n_features)
    ]

    dfs = []
    for symbol in range(symbols_id_start, symbols_id_start + n_symbols):
        dfs.append(
            pl.DataFrame()
            .with_columns(
                pl.date_range(
                    start=start_date,
                    end=start_date + timedelta(days=n_days),
                    closed="right",
                    interval="1d",
                ).alias("timestamp"),
                pl.lit(f"Symbol_{symbol}").alias("symbol"),
            )
            .with_columns(feature_expressions)
            .with_columns(pl.col("timestamp").cast(pl.Datetime).dt.replace_time_zone("UTC").alias("timestamp"))
        )
    df = pl.concat(dfs)
    duplicates = df.sample(fraction=0.20, with_replacement=True)
    null_symbol = df.sample(fraction=0.20, with_replacement=True).with_columns(symbol=pl.lit(None))
    null_timestamp = df.sample(fraction=0.20, with_replacement=True).with_columns(timestamp=pl.lit(None))
    return pl.concat([df, duplicates, null_symbol, null_timestamp], how="vertical")


def clean(df: pl.DataFrame, sort_keys: list) -> pl.DataFrame:
    """Drop nulls and duplicates on sort keys — mirrors the unit-test clean-up."""
    return df.unique(subset=sort_keys).filter(pl.all_horizontal(pl.col(k).is_not_null() for k in sort_keys))


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class GammaFeatureLakeTestsMixin:
    """Backend-agnostic test logic.  Subclass this together with a test
    framework class (unittest.TestCase or a plain pytest class).

    Each ``_test_*`` method accepts a ready-to-use ``GammaFeatureLake``
    instance. Concrete subclasses must provide an ``fs`` pytest fixture
    that returns an initialized ``GammaFeatureLake``.

    ``test_*`` wrappers are generated automatically for every ``_test_*``
    method whose only required parameter (after ``self``) is ``fs``.
    Override any generated wrapper in the subclass to customise behaviour
    (e.g. add parametrize markers, supply a different fixture).
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for attr_name in dir(GammaFeatureLakeTestsMixin):
            if not attr_name.startswith("_test_"):
                continue
            test_name = attr_name[1:]  # strip leading _
            if test_name in cls.__dict__:
                continue  # explicit override wins
            method = getattr(GammaFeatureLakeTestsMixin, attr_name)
            sig = inspect.signature(method)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            # Auto-wire only when fs is the first (and only required) parameter.
            if not params or params[0].name != "fs":
                continue
            required = [p for p in params if p.default is inspect.Parameter.empty]
            if len(required) != 1:
                continue  # has required params beyond fs — needs explicit wrapper

            def _make(m: object, name: str) -> object:
                def test_fn(self, fs: GammaFeatureLake) -> None:
                    m(self, fs)

                test_fn.__name__ = name
                return test_fn

            setattr(cls, test_name, _make(method, test_name))

    # ------------------------------------------------------------------
    # Shared assertion helpers
    # ------------------------------------------------------------------

    def assertExpected(
        self,
        fs: GammaFeatureLake,
        feature_names,
        feature_metadata_height,
        table_metadata_height,
        index_height,
        n_unique_deltatable,
        n_unique_versions,
    ):
        fm = fs.io.scan_delta(fs.feature_metadata).collect()
        tm = fs.io.scan_delta(fs.table_metadata).collect()
        idx = fs.io.scan_delta(fs.index).collect()

        assert sorted(fm["feature_name"].unique().to_list()) == sorted(feature_names)
        assert fm.height == feature_metadata_height
        assert fm["version"].drop_nulls().n_unique() == n_unique_versions
        assert tm.height == table_metadata_height
        assert idx.height == index_height
        assert len(tm["table_addr"].unique()) == n_unique_deltatable

    def verify_index_alignment(self, fs: GammaFeatureLake):
        """Verify that all physical feature tables and read() outputs are consistent
        with the global index.  Three invariants are checked:

        1. **Physical subset**: every sort-key pair in each physical feature table
           exists in the global index (no orphan rows), and each pair is unique within
           that table.

        2. **Read-path subset**: ``fs.read()`` for each table's features returns only
           sort-key pairs that exist in the global index, with no duplicates.

        3. **Full coverage**: reading *all* non-runtime features together returns
           *exactly* the full global index — every (ts, sym) pair is present, no extras,
           no reordering.  This is guaranteed by the ``table_addr is None`` new-table path
           which always writes null sentinel rows for every pre-existing global index
           entry, ensuring the newest table always spans the complete index.
        """
        index = fs.io.scan_delta(fs.index).collect().sort(fs.sort_keys)
        index_keys = index.select(fs.sort_keys)

        # --- Invariant 1: physical tables are subsets of the global index with no dupes ---
        for table_addr in fs.io.scan_delta(fs.table_metadata).collect()["table_addr"].unique():
            physical = fs.io.scan_delta(fs.get_path(table_addr)).collect().select(fs.sort_keys).sort(fs.sort_keys)
            orphans = physical.join(index_keys, on=fs.sort_keys, how="anti")
            assert orphans.height == 0, (
                f"Physical table {table_addr} has {orphans.height} row(s) whose sort keys are absent from the global index:\n{orphans}"
            )
            dupes = physical.filter(physical.is_duplicated())
            assert dupes.height == 0, f"Physical table {table_addr} has {dupes.height} duplicate sort-key row(s):\n{dupes}"

        features_metadata = (
            fs.feature_metadata_frame().collect().filter((pl.col("table_addr") != "n/a") & (pl.col("signal_type") != "runtime_computed"))
        )

        # --- Invariant 2: read() per-table returns only valid, non-duplicate sort keys ---
        for key, frame in features_metadata.group_by("table_addr"):
            feature_names = frame["feature_name"].unique().to_list()
            read_keys = fs.read(feature_names, debug=True).select(fs.sort_keys).sort(fs.sort_keys)

            orphans = read_keys.join(index_keys, on=fs.sort_keys, how="anti")
            assert orphans.height == 0, (
                f"read({feature_names}) returned {orphans.height} row(s) with sort keys absent from the global index:\n{orphans}"
            )
            dupes = read_keys.filter(read_keys.is_duplicated())
            assert dupes.height == 0, f"read({feature_names}) returned {dupes.height} duplicate sort-key row(s):\n{dupes}"

        # --- Invariant 3: reading all features together reproduces the complete global index ---
        all_features = features_metadata["feature_name"].unique().to_list()
        if all_features:
            full_read_keys = fs.read(all_features, debug=True).select(fs.sort_keys).sort(fs.sort_keys)
            assert_frame_equal(full_read_keys, index_keys, check_row_order=True)

    def verify_delta_version_alignment(self, fs: GammaFeatureLake):
        """For each unique table_addr currently referenced in feature_metadata, verify that
        scanning the table at latest returns sort keys that match the global index on the
        table's own [pk_min, pk_max] range.

        ``delta_version`` has been removed from ``feature_metadata``; all tables are always
        read at their latest physical version, which is kept aligned by the write path.

        sparse_feature tables are excluded: they intentionally store only a subset of index rows.
        """
        pk = fs.primary_sort_key
        index = fs.io.scan_delta(fs.index).collect().sort(fs.sort_keys)

        fm = fs.feature_metadata_frame().collect()
        sparse_addrs = fm.filter(pl.col("signal_type") == "sparse_feature")["table_addr"].unique().to_list()
        current_addrs = (
            fm.drop_nulls(subset=["table_addr"])
            .filter(~pl.col("table_addr").is_in(sparse_addrs))
            .sort("version", descending=True, nulls_last=True)
            .group_by("feature_name")
            .first()["table_addr"]
            .unique()
        )
        for table_addr in current_addrs:
            table_latest = fs.io.scan_delta(fs.get_path(table_addr)).collect().select(fs.sort_keys).sort(fs.sort_keys)
            if table_latest.height == 0:
                continue
            pk_min = table_latest[pk].min()
            pk_max = table_latest[pk].max()
            index_on_range = index.filter((pl.col(pk) >= pk_min) & (pl.col(pk) <= pk_max)).select(fs.sort_keys).sort(fs.sort_keys)
            assert_frame_equal(table_latest, index_on_range, check_row_order=True, check_column_order=False)

    # ------------------------------------------------------------------
    # Test scenarios
    # ------------------------------------------------------------------

    def _test_exceptions_uninitialized(self, fs: GammaFeatureLake):
        """Requires an *uninitialized* GammaFeatureLake instance."""
        with pytest.raises(UninitializedDeltaLakeException):
            fs._get_latest_feature_tables(...)
        with pytest.raises(UninitializedDeltaLakeException):
            fs.read(...)
        with pytest.raises(UninitializedDeltaLakeException):
            fs._write_new_feature_table(...)
        with pytest.raises(UninitializedDeltaLakeException):
            fs._get_tables_to_update(...)
        with pytest.raises(UninitializedDeltaLakeException):
            fs.add_features(...)

    def _test_exceptions(self, fs: GammaFeatureLake):
        """Requires an already-initialized GammaFeatureLake instance."""
        df = generate_test_data(n_features=2, n_symbols=3, n_days=5)
        fs.add_features(df, owner="test-owner")

        with pytest.raises(MissingFeaturesException):
            fs.read(["feature_99_does_not_exist"])

        with pytest.raises(MissingOrMisregisteredSignalsException):
            fs.read(["feature_0", "feature_1"], targets=["feature_2"])

        with pytest.raises(ValueError):
            fs.add_features(pl.DataFrame())

    def _test_simple_operations(self, fs: GammaFeatureLake, use_remote_data=False, run_on_ray_cluster=False):
        n_features = 3
        n_symbols = 10
        n_days = 30
        features = [f"feature_{i}" for i in range(n_features)] + ["event"]
        df = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days)
        event_df = (
            df.unique(fs.sort_keys)
            .sample(fraction=0.05, with_replacement=False)
            .with_columns(pl.lit(0).alias("event"))
            .select(["event"] + fs.sort_keys)
        )
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df))
        else:
            fs.add_features(df, owner="test-owner")
        fs.add_as_of_features(event_df.select(fs.sort_keys + ["event"]), params={"tolerance": "1h"})
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=len(features),
            table_metadata_height=2,
            n_unique_deltatable=2,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)
        with pytest.raises(MissingFeaturesException):
            fs.read(["feature_does_not_exist"])

        with pytest.raises(MissingOrMisregisteredSignalsException):
            fs.read(["feature_0", "feature_1"], targets=["feature_2"])

        with pytest.raises(ValueError):
            fs.add_features(pl.DataFrame())

        with pytest.raises(ValueError):
            fs.add_targets(pl.DataFrame())

    def _test_adding_days(self, fs: GammaFeatureLake, use_remote_data=False, run_on_ray_cluster=False):
        # Step 1: Add 10 days worth of features, starting on day 0.
        # Step 2: Add 20 days worth of features, starting on day 0.
        # Step 3: Add 20 days worth of features, starting on day 50.
        # Step 4: Add 20 days worth of features, starting on day 40.
        n_features = 3
        n_symbols = 10
        n_days = 10
        features = [f"feature_{i}" for i in range(n_features)]
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        # Step 1
        df = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df))
        else:
            fs.add_features(df)
        df = clean(df, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=len(features),
            table_metadata_height=1,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )
        assert df.sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        last_updates = fs.io.scan_delta(fs.table_metadata).collect().group_by("table_addr").agg(pl.col("last_updated").max())
        assert sorted(last_updates["last_updated"]) == [start_date + timedelta(days=n_days)]
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 2
        df2 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date)
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df2), overlap_mode="merge")
        else:
            fs.add_features(df2, overlap_mode="merge")
        df2 = clean(df2, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=2 * len(features),
            table_metadata_height=2,
            n_unique_deltatable=1,
            n_unique_versions=2,
            index_height=2 * (n_symbols * n_days),
        )
        assert df2.sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        last_updates = fs.io.scan_delta(fs.table_metadata).collect().group_by("table_addr").agg(pl.col("last_updated").max())
        assert sorted(last_updates["last_updated"]) == [start_date + timedelta(days=n_days * 2)]
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 3
        df3 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date + timedelta(days=50))
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df3))
        else:
            fs.add_features(df3)
        df3 = clean(df3, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=2 * len(features),
            table_metadata_height=3,
            n_unique_deltatable=1,
            n_unique_versions=2,
            index_height=2 * (n_symbols * n_days) + 2 * (n_symbols * n_days),
        )
        assert pl.concat([df2.sort(fs.sort_keys), df3.sort(fs.sort_keys)]).select(features).equals(fs.read(features).select(features))
        last_updates = fs.io.scan_delta(fs.table_metadata).collect().group_by("table_addr").agg(pl.col("last_updated").max())
        assert sorted(last_updates["last_updated"]) == [start_date + timedelta(days=50) + timedelta(days=n_days * 2)]
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 4
        df4 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date + timedelta(days=40))
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df4), overlap_mode="merge")
        else:
            fs.add_features(df4, overlap_mode="merge")
        df4 = clean(df4, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=3 * len(features),
            table_metadata_height=4,
            n_unique_deltatable=1,
            n_unique_versions=3,
            index_height=2 * (n_symbols * n_days) + 2 * (n_symbols * n_days) + (n_days * n_symbols),
        )
        df3_remainder = df3.filter(pl.col("timestamp") > df4["timestamp"].max())
        expected = pl.concat([df2.sort(fs.sort_keys), df4.sort(fs.sort_keys), df3_remainder.sort(fs.sort_keys)]).sort(fs.sort_keys)
        assert expected.select(features).equals(fs.read(features).select(features))
        last_updates = fs.io.scan_delta(fs.table_metadata).collect().group_by("table_addr").agg(pl.col("last_updated").max())
        assert sorted(last_updates["last_updated"]) == [start_date + timedelta(days=50) + timedelta(days=n_days * 2)]
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

    def _test_adding_symbols(self, fs: GammaFeatureLake, use_remote_data=False, run_on_ray_cluster=False):
        # Step 1: Add 10 days worth of features for 5 symbols
        # Step 2: Add 10 days worth of the same features, for the original 5 symbols + 5 new symbols
        # Step 3: Add 10 days worth of the same features, for 10 new symbols
        # Step 4: Add 10 days worth of the same features, 20 days in the future, for 20 new symbols
        # Step 5: Add 10 days worth of new features, 20 days in the future, for 20 new symbols
        # Step 6: Add 10 days worth of new features, 20 days in the future, for 5 symbols + 5 new symbols
        n_features = 3
        n_symbols = 5
        n_days = 10
        features = [f"feature_{i}" for i in range(n_features)]
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        # Step 1
        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df1), overlap_mode="merge")
        else:
            fs.add_features(df1, overlap_mode="merge")
        df1 = clean(df1, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=len(features),
            table_metadata_height=1,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )
        assert df1.sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 2
        df2 = generate_test_data(n_features=n_features, n_symbols=2 * n_symbols, n_days=n_days, start_date=start_date)
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df2), overlap_mode="merge")
        else:
            fs.add_features(df2, overlap_mode="merge")
        df2 = clean(df2, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=2 * len(features),
            table_metadata_height=2,
            n_unique_deltatable=1,
            n_unique_versions=2,
            index_height=100,
        )
        assert df2.sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 3
        df3 = generate_test_data(n_features=n_features, n_symbols=10, symbols_id_start=10, n_days=n_days, start_date=start_date)
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df3), overlap_mode="merge")
        else:
            fs.add_features(df3, overlap_mode="merge")
        df3 = clean(df3, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=2 * len(features),
            table_metadata_height=3,
            n_unique_deltatable=1,
            n_unique_versions=2,
            index_height=200,
        )
        assert pl.concat([df2, df3]).sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 4
        df4 = generate_test_data(n_features=n_features, n_symbols=20, symbols_id_start=20, n_days=n_days, start_date=start_date + timedelta(days=20))
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df4), overlap_mode="merge")
        else:
            fs.add_features(df4, overlap_mode="merge")
        df4 = clean(df4, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=2 * len(features),
            table_metadata_height=4,
            n_unique_deltatable=1,
            n_unique_versions=2,
            index_height=400,
        )
        assert pl.concat([df2, df3, df4]).sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 5
        df5 = generate_test_data(
            n_features=n_features,
            n_symbols=20,
            symbols_id_start=40,
            feature_ids_start=n_features,
            n_days=n_days,
            start_date=start_date + timedelta(days=20),
        )
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df5), overlap_mode="merge")
        else:
            fs.add_features(df5, overlap_mode="merge")
        df5 = clean(df5, fs.sort_keys)
        new_features = features + [f"feature_{i}" for i in range(n_features, 2 * n_features)]
        self.assertExpected(
            fs,
            feature_names=new_features,
            feature_metadata_height=3 * n_features,
            table_metadata_height=5,
            n_unique_deltatable=2,
            n_unique_versions=2,
            index_height=600,
        )
        assert (
            pl.concat([df2, df3, df4, df5], how="diagonal").sort(fs.sort_keys).select(new_features).equals(fs.read(new_features).select(new_features))
        )
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 6
        df6 = generate_test_data(
            n_features=n_features,
            n_symbols=2 * n_symbols,
            symbols_id_start=55,
            feature_ids_start=2 * n_features,
            n_days=n_days,
            start_date=start_date + timedelta(days=20),
        )
        new_features = features + [f"feature_{i}" for i in range(2 * n_features, 3 * n_features)]
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df6), overlap_mode="merge")
        else:
            fs.add_features(df6, overlap_mode="merge")
        df6 = clean(df6, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=[f"feature_{i}" for i in range(3 * n_features)],
            feature_metadata_height=4 * n_features,
            table_metadata_height=6,
            n_unique_deltatable=3,
            n_unique_versions=2,
            index_height=650,
        )
        assert sorted(fs.io.scan_delta(fs.feature_metadata).collect()["feature_name"].unique().to_list()) == [
            f"feature_{i}" for i in range(3 * n_features)
        ]
        step5_and_6_index_values = pl.concat([df5, df6], how="diagonal").select(fs.sort_keys).unique()
        new_values_added = step5_and_6_index_values.join(df5, how="left", on=fs.sort_keys).join(df6, how="left", on=fs.sort_keys)
        assert (
            pl.concat([df2, df3, df4, new_values_added], how="diagonal")
            .sort(fs.sort_keys)
            .select(new_features)
            .equals(fs.read(new_features).select(new_features))
        )

        # After steps 5 and 6, there are modifications to the reading of the first set of features
        assert (
            pl.concat([df2, df3, df4, step5_and_6_index_values], how="diagonal")
            .sort(fs.sort_keys)
            .select(features)
            .equals(fs.read(features).select(features))
        )
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

    def _test_appending_new_features(self, fs: GammaFeatureLake, use_remote_data=False, run_on_ray_cluster=False):
        # Step 1: Add 10 days worth of 3 features (f0, f1, f2) for 5 symbols
        # Step 2: Add 10 days worth of 3 old features and 2 new features (f3, f4) for 5 symbols.
        n_symbols = 5
        n_days = 10

        # Step 1
        df = generate_test_data(n_features=3, n_symbols=5, n_days=10)
        features = [f"feature_{i}" for i in range(3)]
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df))
        else:
            fs.add_features(df)
        df = clean(df, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=len(features),
            table_metadata_height=1,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )
        assert df.sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 2
        df = generate_test_data(n_features=5, n_symbols=5, n_days=10)
        features = [f"feature_{i}" for i in range(5)]
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df), overlap_mode="merge")
        else:
            fs.add_features(df, overlap_mode="merge")
        df = clean(df, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=3 + 3 + 2,
            table_metadata_height=3,
            n_unique_deltatable=2,
            n_unique_versions=2,
            index_height=n_symbols * n_days,
        )
        assert df.sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        assert df.sort(fs.sort_keys).select(["feature_3", "feature_4"]).equals(fs.read(features).select(["feature_3", "feature_4"]))
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

    def _test_features_with_holes(self, fs: GammaFeatureLake, use_remote_data=False, run_on_ray_cluster=False):
        # Step 1: Add 10 days worth of 5 features, starting on day 0.
        # Step 2: Add 10 days worth of 3 features, starting on day 10
        # Step 3: Add 10 days worth of 5 features, starting on day 20.
        n_symbols = 10
        n_days = 10
        n_features = 5
        features = [f"feature_{i}" for i in range(n_features)]
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        # Step 1
        df = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df))
        else:
            fs.add_features(df)
        df = clean(df, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=len(features),
            table_metadata_height=1,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )
        assert df.sort(fs.sort_keys).select(features).equals(fs.read(features).select(features))
        last_updates = fs.io.scan_delta(fs.table_metadata).collect().group_by("table_addr").agg(pl.col("last_updated").max())
        assert sorted(last_updates["last_updated"]) == [start_date + timedelta(days=n_days)]
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 2
        df2 = generate_test_data(n_features=n_features - 2, n_symbols=n_symbols, n_days=n_days, start_date=start_date + timedelta(days=10))
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df2))
        else:
            fs.add_features(df2)
        df2 = clean(df2, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=len(features),
            table_metadata_height=2,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=2 * (n_symbols * n_days),
        )
        last_updates = fs.io.scan_delta(fs.table_metadata).collect().group_by("table_addr").agg(pl.col("last_updated").max())
        assert sorted(last_updates["last_updated"]) == [start_date + timedelta(days=n_days * 2)]
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 3
        df3 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date + timedelta(days=20))
        if use_remote_data:
            import ray

            fs.add_features(ray.put(df3))
        else:
            fs.add_features(df3)
        df3 = clean(df3, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=features,
            feature_metadata_height=len(features),
            table_metadata_height=3,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=3 * (n_symbols * n_days),
        )
        last_updates = fs.io.scan_delta(fs.table_metadata).collect().group_by("table_addr").agg(pl.col("last_updated").max())
        assert sorted(last_updates["last_updated"]) == [start_date + timedelta(days=n_days * 3)]
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

    def _test_updating_old_features(self, fs: GammaFeatureLake):
        # Step 1: Add 10 days worth of F1 features, starting on day 0.
        # Step 2: Add 10 days worth of F2 features, starting on day 0
        # Step 3: Add 20 days worth of F1 features, starting on day 10
        # Step 4: Add 10 days worth of F2 features, starting on day 20.
        n_symbols = 10
        n_days = 10
        n_features = 5
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        # Step 1
        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_suffix="F1")
        fs.add_features(df1)
        df1 = clean(df1, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)],
            feature_metadata_height=n_features,
            table_metadata_height=1,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )

        # Step 2
        df2 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_suffix="F2")
        fs.add_features(df2)
        df2 = clean(df2, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)] + [f"F2_{i}" for i in range(n_features)],
            feature_metadata_height=2 * n_features,
            table_metadata_height=2,
            n_unique_deltatable=2,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )

        # Step 3
        df3 = generate_test_data(
            n_features=n_features, n_symbols=n_symbols, n_days=2 * n_days, start_date=start_date + timedelta(days=10), feature_suffix="F1"
        )
        fs.add_features(df3)
        df3 = clean(df3, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)] + [f"F2_{i}" for i in range(n_features)],
            feature_metadata_height=2 * n_features,
            table_metadata_height=3,
            n_unique_deltatable=2,
            n_unique_versions=1,
            index_height=n_symbols * n_days + (2 * n_days * n_symbols),
        )

        # Step 4
        df4 = generate_test_data(
            n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date + timedelta(days=20), feature_suffix="F2"
        )
        fs.add_features(df4)
        df4 = clean(df4, fs.sort_keys)
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)] + [f"F2_{i}" for i in range(n_features)],
            feature_metadata_height=2 * n_features,
            table_metadata_height=4,
            n_unique_deltatable=2,
            n_unique_versions=1,
            index_height=n_symbols * n_days + (2 * n_days * n_symbols),
        )
        features = [f"F1_{i}" for i in range(n_features)] + [f"F2_{i}" for i in range(n_features)]
        tbl = fs.read(features)
        f1s = pl.concat([df1, df3])
        assert f1s.height == 300
        f2s = pl.concat([df2, df4])
        assert f2s.height == 200
        missing_index_rows_from_f2 = f1s.join(f2s, on=fs.sort_keys, how="anti").select(fs.sort_keys)
        f2s = pl.concat([f2s, missing_index_rows_from_f2], how="diagonal")
        local_table = pl.concat(
            [
                f1s.sort(fs.sort_keys).select([f"F1_{i}" for i in range(n_features)]),
                f2s.sort(fs.sort_keys).select([f"F2_{i}" for i in range(n_features)]),
            ],
            how="horizontal",
        )
        assert tbl.select(features).equals(local_table)

    def _test_non_unique_index_data(self, fs: GammaFeatureLake):
        # Step 1: Upload and insert some data.
        # Step 2: Upload and insert the same DataFrame, concatenated with itself.
        n_symbols = 10
        n_days = 10
        n_features = 5
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        # Step 1
        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_suffix="F1")
        fs.add_features(df1)
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)],
            feature_metadata_height=n_features,
            table_metadata_height=1,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )

        # Step 2
        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_suffix="F1")
        fs.add_features(pl.concat([df1, df1], how="vertical"), overlap_mode="merge")
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)],
            feature_metadata_height=2 * n_features,
            table_metadata_height=2,
            n_unique_deltatable=1,
            n_unique_versions=2,
            index_height=n_symbols * n_days,
        )

    def _test_merge_extends_date_range(self, fs: GammaFeatureLake):
        """After an overlap merge, the table has more rows and the same table_addr is reused."""
        n_features = 3
        n_symbols = 5
        n_days = 10
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        # Step 1: initial upload
        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        fs.add_features(df1)

        table_addr = fs.io.scan_delta(fs.feature_metadata).collect().filter(pl.col("table_addr") != "n/a")["table_addr"].unique().item()
        table_path = fs.get_path(table_addr)
        rows_after_step1 = fs.io.scan_delta(table_path).collect().height

        # Step 2: overlapping upload triggers merge
        df2 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date)
        fs.add_features(df2, overlap_mode="merge")
        df2 = clean(df2, fs.sort_keys)

        current_data = fs.io.scan_delta(table_path).collect().sort(fs.sort_keys)
        assert current_data.height > rows_after_step1
        assert rows_after_step1 == n_symbols * n_days
        assert current_data.height == n_symbols * n_days * 2

    def _test_multiple_sequential_merges(self, fs: GammaFeatureLake):
        """Multiple sequential merges accumulate correctly; final state reflects all updates."""
        n_features = 3
        n_symbols = 5
        n_days = 10
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        fs.add_features(df1)

        table_addr = fs.io.scan_delta(fs.feature_metadata).collect().filter(pl.col("table_addr") != "n/a")["table_addr"].unique().item()
        table_path = fs.get_path(table_addr)

        # Steps 2-4: each overlaps with previous, triggering merge; row count grows each time
        prev_height = fs.io.scan_delta(table_path).collect().height
        for step in range(3):
            df_new = generate_test_data(
                n_features=n_features,
                n_symbols=n_symbols,
                n_days=n_days * (step + 2),
                start_date=start_date,
            )
            fs.add_features(df_new, overlap_mode="merge")
            current_height = fs.io.scan_delta(table_path).collect().height
            assert current_height > prev_height
            prev_height = current_height

        # Final state has the largest date range
        final_data = fs.io.scan_delta(table_path).collect()
        assert final_data.height == n_symbols * n_days * 4

    def _test_merge_updates_overlap_values(self, fs: GammaFeatureLake):
        """After a merge, the overlapping rows in the final table reflect the latest upload's values."""
        n_features = 3
        n_symbols = 5
        n_days = 10
        features = [f"feature_{i}" for i in range(n_features)]
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        fs.add_features(df1)
        df1_clean = clean(df1, fs.sort_keys)

        table_addr = fs.io.scan_delta(fs.feature_metadata).collect().filter(pl.col("table_addr") != "n/a")["table_addr"].unique().item()
        table_path = fs.get_path(table_addr)

        # Step 2: overlap merge with different random data
        df2 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date)
        fs.add_features(df2, overlap_mode="merge")
        df2_clean = clean(df2, fs.sort_keys)

        final_table = fs.io.scan_delta(table_path).collect().sort(fs.sort_keys)
        assert_frame_equal(final_table.select(features), df2_clean.sort(fs.sort_keys).select(features))

        # The overlap rows (step 1 range) reflect df2's values, not df1's
        overlap_range = final_table.filter(pl.col("timestamp") <= df1_clean["timestamp"].max())
        df2_overlap = df2_clean.filter(pl.col("timestamp") <= df1_clean["timestamp"].max()).sort(fs.sort_keys)
        assert_frame_equal(overlap_range.select(features), df2_overlap.select(features))

    def _test_merge_adds_symbols(self, fs: GammaFeatureLake):
        """After a merge that introduces new symbols, the final table contains all symbols."""
        n_features = 3
        n_symbols = 5
        n_days = 10
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        fs.add_features(df1)
        df1 = clean(df1, fs.sort_keys)

        table_addr = fs.io.scan_delta(fs.feature_metadata).collect().filter(pl.col("table_addr") != "n/a")["table_addr"].unique().item()
        table_path = fs.get_path(table_addr)

        # Step 2: 10 symbols (5 original + 5 new), same date range → merge
        df2 = generate_test_data(n_features=n_features, n_symbols=2 * n_symbols, n_days=n_days, start_date=start_date)
        fs.add_features(df2, overlap_mode="merge")
        df2 = clean(df2, fs.sort_keys)

        final_data = fs.io.scan_delta(table_path).collect()
        assert len(final_data["symbol"].unique()) == 2 * n_symbols
        assert set(df1["symbol"].unique().to_list()).issubset(set(final_data["symbol"].unique().to_list()))

    def _test_merge_preserves_non_overlapping_rows(self, fs: GammaFeatureLake):
        """After a partial-overlap merge, the final table contains both original and new rows."""
        n_features = 3
        n_symbols = 5
        n_days = 20
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        fs.add_features(df1)

        table_addr = fs.io.scan_delta(fs.feature_metadata).collect().filter(pl.col("table_addr") != "n/a")["table_addr"].unique().item()
        table_path = fs.get_path(table_addr)
        rows_before = fs.io.scan_delta(table_path).collect().height
        assert rows_before == n_symbols * n_days

        # Step 2: overlap with first n_days + extend by another n_days
        df2 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date)
        fs.add_features(df2, overlap_mode="merge")
        df2_clean = clean(df2, fs.sort_keys)

        new_data = fs.io.scan_delta(table_path).collect()
        assert new_data.height == df2_clean.height

    def _test_read_with_feature_metadata_dataframe(self, fs: GammaFeatureLake):
        """Verify that fs.read(feature_metadata_df) correctly reads the latest version of each feature."""
        n_features = 3
        n_symbols = 5
        n_days = 10
        features = [f"feature_{i}" for i in range(n_features)]
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date)
        fs.add_features(df1)
        df2 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date)
        fs.add_features(df2, overlap_mode="merge")
        df2_clean = clean(df2, fs.sort_keys).sort(fs.sort_keys)

        meta = fs.feature_metadata_frame().collect()
        assert "delta_version" not in meta.columns

        # Reading via the latest feature_metadata rows returns the latest data.
        latest_meta = meta.sort("version", descending=True, nulls_last=True).group_by("feature_name").first()
        result = fs.read(latest_meta).sort(fs.sort_keys)
        assert result.height == df2_clean.height
        assert_frame_equal(result.select(features), df2_clean.select(features))

        # Same feature at two different version rows → raises ValueError.
        duplicate_meta = pl.concat([meta.filter(pl.col("version") == 0).head(1), meta.filter(pl.col("version") == 1).head(1)]).filter(
            pl.col("feature_name") == features[0]
        )
        if duplicate_meta.height > 1:
            with pytest.raises(ValueError, match="multiple versions"):
                fs.read(duplicate_meta)

    def _test_merge_cross_table_alignment(self, fs: GammaFeatureLake):
        """Verifies that alignment appends to table B are recorded in feature_metadata so that
        reads via feature_metadata produce correctly aligned frames across tables.

        Scenario: A and B initially lack a mid-range day. A is re-inserted including that day,
        triggering a merge on A and an alignment append on B. The fix ensures align_feature_tables
        writes a new feature_metadata row for B so reads see the aligned version.
        """
        n_symbols = 3
        n_days = 10
        start_date = datetime(2020, 6, 20, tzinfo=UTC)
        missing_day = start_date + timedelta(days=5)

        def make_df(feature_ids_start, skip_day=None):
            df = clean(
                generate_test_data(
                    n_features=1,
                    n_symbols=n_symbols,
                    n_days=n_days,
                    start_date=start_date,
                    feature_ids_start=feature_ids_start,
                ),
                fs.sort_keys,
            )
            if skip_day is not None:
                df = df.filter(pl.col("timestamp") != skip_day)
            return df

        # Step 1: insert feature_0 (table A) and feature_3 (table B) for all days except missing_day.
        fs.add_features(make_df(feature_ids_start=0, skip_day=missing_day))
        fs.add_features(make_df(feature_ids_start=3, skip_day=missing_day))

        fm_initial = fs.io.scan_delta(fs.feature_metadata).collect()
        table_addr_b = fm_initial.filter(pl.col("feature_name") == "feature_3")["table_addr"].unique().item()
        rows_b_before = fs.io.scan_delta(fs.get_path(table_addr_b)).collect().height

        # Step 2: re-insert feature_0 for ALL days including missing_day.
        fs.add_features(make_df(feature_ids_start=0, skip_day=None), overlap_mode="merge")

        expected_total_rows = n_symbols * n_days
        assert missing_day in fs.io.scan_delta(fs.index).collect()["timestamp"].to_list()

        # B's Delta table at latest version has the alignment rows.
        b_latest_delta = fs.io.scan_delta(fs.get_path(table_addr_b)).collect()
        assert b_latest_delta.height == expected_total_rows
        assert b_latest_delta.height > rows_b_before

        # align_feature_tables no longer writes a new feature_metadata row (version counter
        # is not affected by alignment appends — the table_addr is unchanged and reads always
        # use the latest physical version).
        fm_after = fs.io.scan_delta(fs.feature_metadata).collect()
        b_meta_versions = fm_after.filter(pl.col("feature_name") == "feature_3")["version"].to_list()
        assert len(b_meta_versions) == 1, (
            f"feature_metadata for feature_3 should still have 1 version row after alignment append, got {b_meta_versions}"
        )

        self.verify_delta_version_alignment(fs)

        # Reading via the latest feature_metadata rows returns aligned data.
        meta_a_latest = fm_after.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)
        meta_b_latest = fm_after.filter(pl.col("feature_name") == "feature_3").sort("version", descending=True, nulls_last=True).head(1)
        mixed_meta = pl.concat([meta_a_latest, meta_b_latest])

        result = fs.read(mixed_meta).sort(fs.sort_keys)

        # Assert 1: correct row count — no null-padding from mismatched heights.
        assert result.height == expected_total_rows, f"Time-travel read returned {result.height} rows but expected {expected_total_rows}."

        # Assert 2: sort keys and feature_3 values must match B's latest Delta table exactly.
        # Misalignment would cause B's post-gap values to appear under pre-gap sort keys.
        b_direct = b_latest_delta.sort(fs.sort_keys)
        assert_frame_equal(
            result.select(fs.sort_keys + ["feature_3"]),
            b_direct.select(fs.sort_keys + ["feature_3"]),
            check_row_order=True,
        )

    def _test_merge_with_new_mid_range_index_row(self, fs: GammaFeatureLake):
        """Verify that a merge insert which includes a brand-new mid-range index row (a
        (timestamp, symbol) pair that was never in the index) correctly:

          1. Inserts real feature values for the new row into the target feature table.
          2. Propagates a null-padded alignment row for the new index row to every other
             feature table whose last_updated date covers that row.
          3. Adds the new row to the global index.
        """
        n_symbols = 5
        n_days = 10
        start_date = datetime(2020, 6, 20, tzinfo=UTC)
        # This date falls in the middle of the range but is intentionally absent from the
        # initial inserts so it never enters the index.
        missing_day = start_date + timedelta(days=5)  # 2020-06-25
        features_a = [f"feature_{i}" for i in range(3)]
        features_b = [f"feature_{i}" for i in range(3, 6)]

        # Step 1: Insert feature group A for the full range, skipping the missing day.
        df_a_partial = clean(
            generate_test_data(n_features=3, n_symbols=n_symbols, n_days=n_days, start_date=start_date),
            fs.sort_keys,
        ).filter(pl.col("timestamp") != missing_day)
        fs.add_features(df_a_partial)

        # Step 2: Insert a second feature group B for the same (gapped) range.
        df_b_partial = clean(
            generate_test_data(n_features=3, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_ids_start=3),
            fs.sort_keys,
        ).filter(pl.col("timestamp") != missing_day)
        fs.add_features(df_b_partial)

        # Sanity-check: the missing day must not yet be in the index.
        index_before = fs.io.scan_delta(fs.index).collect()
        assert index_before.height == (n_days - 1) * n_symbols
        assert missing_day not in index_before["timestamp"].to_list()
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Step 3: Re-insert feature group A for the FULL range, including the missing day.
        # This is an overlapping merge that also introduces a new index row.
        df_a_full = clean(
            generate_test_data(n_features=3, n_symbols=n_symbols, n_days=n_days, start_date=start_date),
            fs.sort_keys,
        )
        fs.add_features(df_a_full, overlap_mode="merge")

        # The global index must now contain the previously missing day.
        index_after = fs.io.scan_delta(fs.index).collect()
        assert index_after.height == n_days * n_symbols
        assert missing_day in index_after["timestamp"].to_list()

        # Feature table A must have real (non-null) feature values for the new day.
        result_a = fs.read(features_a).filter(pl.col("timestamp") == missing_day)
        assert result_a.height == n_symbols, f"Expected {n_symbols} rows for missing day in table A, got {result_a.height}"
        for feat in features_a:
            assert result_a[feat].is_not_null().all(), f"Expected non-null values for {feat} on missing day in table A"

        # Feature table B must have null-padded alignment rows for the new day.
        result_b = fs.read(features_b).filter(pl.col("timestamp") == missing_day)
        assert result_b.height == n_symbols, f"Expected {n_symbols} alignment rows for missing day in table B, got {result_b.height}"
        for feat in features_b:
            assert result_b[feat].is_null().all(), f"Expected null alignment values for {feat} on missing day in table B"

        # Full alignment check: every feature table must have the same index as the global index.
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

    def _test_selective_version_update(self, fs: GammaFeatureLake):
        """Verify that alignment appends and feature_metadata version recording only happen for
        feature tables whose range covers the new index rows.

        Setup:
          - Table A (feature_0): wide date range Jan 1 – Apr 10, 2020, skipping missing_day.
          - Table B (feature_1): overlapping range Jan 21 – Apr 10, 2020, skipping missing_day.
            missing_day (Feb 20) falls within B's range so B should receive alignment rows.
          - Table C (feature_2): early range Jan 2 – Jan 31, 2020.
            missing_day (Feb 20) is AFTER C's last_updated so C must NOT be touched.

        After re-inserting feature_0 with missing_day included:
          - A: merge fires → new feature_metadata version row.
          - B: alignment append fires → new feature_metadata version row.
          - C: untouched → Delta version and feature_metadata row count unchanged.
        """
        n_symbols = 3
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        missing_day = start_date + timedelta(days=50)  # Feb 20, 2020

        # C's range ends at day 30 (Jan 31), which is before missing_day (day 50).
        n_days_c = 30  # last_updated = Jan 31
        # A and B share the same end date (day 100 from their respective starts) so both
        # last_updated values land on Apr 10, well after missing_day.
        n_days_a = 100  # last_updated = start_date + 100d = Apr 10
        n_days_b = 80  # start offset 20d + 80d = Apr 10

        def make_feature_df(feature_id, n_days, start, skip_day=None):
            df = clean(
                generate_test_data(
                    n_features=1,
                    n_symbols=n_symbols,
                    n_days=n_days,
                    start_date=start,
                    feature_ids_start=feature_id,
                ),
                fs.sort_keys,
            )
            if skip_day is not None:
                df = df.filter(pl.col("timestamp") != skip_day)
            return df

        # Step 1: insert all three feature groups — missing_day absent from A and B but
        # irrelevant for C (it's outside C's date range entirely).
        df_a = make_feature_df(0, n_days_a, start_date, skip_day=missing_day)
        df_b = make_feature_df(1, n_days_b, start_date + timedelta(days=20), skip_day=missing_day)
        df_c = make_feature_df(2, n_days_c, start_date)
        fs.add_features(df_a)
        fs.add_features(df_b)
        fs.add_features(df_c)

        fm_initial = fs.io.scan_delta(fs.feature_metadata).collect()
        table_addr_b = fm_initial.filter(pl.col("feature_name") == "feature_1")["table_addr"].unique().item()
        table_addr_c = fm_initial.filter(pl.col("feature_name") == "feature_2")["table_addr"].unique().item()

        rows_b_before = fs.io.scan_delta(fs.get_path(table_addr_b)).collect().height
        rows_c_before = fs.io.scan_delta(fs.get_path(table_addr_c)).collect().height

        # Confirm C's last_updated is strictly before missing_day.
        table_meta = fs.io.scan_delta(fs.table_metadata).collect()
        c_last_updated = table_meta.filter(pl.col("table_addr") == table_addr_c)["last_updated"].max()
        assert c_last_updated < missing_day, f"Test setup error: C last_updated ({c_last_updated}) should be before missing_day ({missing_day})"
        b_last_updated = table_meta.filter(pl.col("table_addr") == table_addr_b)["last_updated"].max()
        assert b_last_updated >= missing_day, f"Test setup error: B last_updated ({b_last_updated}) should cover missing_day ({missing_day})"

        # Step 2: re-insert feature_0 with missing_day included → triggers merge on A,
        # alignment on B (covers missing_day), and nothing on C (does not cover missing_day).
        df_a_full = make_feature_df(0, n_days_a, start_date, skip_day=None)
        fs.add_features(df_a_full, overlap_mode="merge")

        fm_after = fs.io.scan_delta(fs.feature_metadata).collect()

        # B must have received an alignment append (row count increased).
        rows_b_after = fs.io.scan_delta(fs.get_path(table_addr_b)).collect().height
        assert rows_b_after > rows_b_before, "Table B row count should have increased after alignment append"

        # align_feature_tables no longer writes a new feature_metadata row — B still has 1.
        b_version_rows = fm_after.filter(pl.col("feature_name") == "feature_1")["version"].to_list()
        assert len(b_version_rows) == 1, (
            f"feature_metadata for feature_1 (table B) should still have 1 version row after alignment, got {b_version_rows}"
        )

        # C must be completely untouched: same row count, no new feature_metadata rows.
        rows_c_after = fs.io.scan_delta(fs.get_path(table_addr_c)).collect().height
        assert rows_c_after == rows_c_before, (
            f"Table C row count should be unchanged (C's range does not cover missing_day): before={rows_c_before}, after={rows_c_after}"
        )
        c_version_rows = fm_after.filter(pl.col("feature_name") == "feature_2")["version"].to_list()
        assert len(c_version_rows) == 1, (
            f"feature_metadata for feature_2 (table C) should have 1 version row (C was not updated), got {c_version_rows}"
        )

        # End-to-end: reading all three features should be correctly aligned.
        result = fs.read(["feature_0", "feature_1", "feature_2"]).sort(fs.sort_keys)
        assert result.height > 0
        # missing_day rows should exist for feature_0 (real values) and feature_1 (null alignment).
        rows_on_missing_day = result.filter(pl.col("timestamp") == missing_day)
        assert rows_on_missing_day.height == n_symbols, f"Expected {n_symbols} rows for missing_day after re-insert, got {rows_on_missing_day.height}"
        assert rows_on_missing_day["feature_0"].is_not_null().all(), "feature_0 values on missing_day should be non-null (real data)"
        assert rows_on_missing_day["feature_1"].is_null().all(), "feature_1 values on missing_day should be null (alignment rows)"
        # feature_2 has no rows for missing_day (C's range doesn't cover it) — that's correct.
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

    def _test_feature_params_preserved_on_remerge(self, fs: GammaFeatureLake):
        """Verify that when a feature is re-added (triggering the merge path) with new
        feature_params, the new version row in feature_metadata records the updated params.

        Old bug: the merge path wrote feature_df.with_columns(version=..., owner=...) only,
        carrying forward the old feature_params silently.
        """
        import json as _json

        n_symbols = 3
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        old_params = {"strategy": "linear", "window": 5}
        new_params = {"strategy": "exponential", "window": 10}

        df = clean(
            generate_test_data(n_features=1, n_symbols=n_symbols, n_days=n_days, start_date=start_date),
            fs.sort_keys,
        )

        # First insert: stores old_params.
        fs.add_as_of_features(df, params=old_params)

        fm_v0 = fs.io.scan_delta(fs.feature_metadata).collect()
        assert fm_v0.filter(pl.col("feature_name") == "feature_0")["feature_params"].first() == _json.dumps(old_params)

        # Second insert overlaps → merge path fires. Should record new_params.
        df2 = clean(
            generate_test_data(n_features=1, n_symbols=n_symbols, n_days=n_days, start_date=start_date),
            fs.sort_keys,
        )
        fs.add_as_of_features(df2, params=new_params, overlap_mode="merge")

        fm_after = fs.io.scan_delta(fs.feature_metadata).collect()
        # Get the row with the highest version for feature_0.
        latest_row = fm_after.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)
        assert latest_row["feature_params"].first() == _json.dumps(new_params), (
            f"Expected updated feature_params {_json.dumps(new_params)} in latest version row, got {latest_row['feature_params'].first()}"
        )
        # Old params row must still be there (append-only metadata).
        v0_row = fm_after.filter(pl.col("feature_name") == "feature_0").sort("version").head(1)
        assert v0_row["feature_params"].first() == _json.dumps(old_params), "v0 row should retain original params"

    def _test_as_of_feature_alignment_skipped(self, fs: GammaFeatureLake):
        """Verify that align_feature_tables never injects null sentinels into as_of_feature tables.

        as_of features are read via join_asof(strategy="backward") against the global index —
        not as a direct scan. join_asof naturally carries the last known observation forward
        to any date that lacks an entry in the as_of table. Injecting null sentinels would
        corrupt that carry-forward semantics.

        Two scenarios are tested:
          A) New index row falls WITHIN the as_of table's existing range (mid-range gap).
          B) New index row falls BEYOND the as_of table's last_updated.

        In both cases the as_of physical table must remain unchanged, and read() must return
        the correctly carried-forward value (not null).
        """
        n_symbols = 3
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        mid_gap = start_date + timedelta(days=5)  # T5: within range of as_of table
        t11 = start_date + timedelta(days=11)  # T11: beyond as_of last_updated

        # as_of feature_1 covers T1–T10 skipping T5 (T5 never enters the index).
        df_b = clean(
            generate_test_data(n_features=1, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_ids_start=1),
            fs.sort_keys,
        ).filter(pl.col("timestamp") != mid_gap)
        fs.add_as_of_features(df_b, params={"strategy": "backward"})

        fm = fs.io.scan_delta(fs.feature_metadata).collect()
        table_addr_b = fm.filter(pl.col("feature_name") == "feature_1")["table_addr"].unique().item()
        rows_before = fs.io.scan_delta(fs.get_path(table_addr_b)).collect().height

        assert mid_gap not in fs.io.scan_delta(fs.index).collect()["timestamp"].to_list()

        # Scenario A: add normal feature_0 for T1–T10 INCLUDING T5 → T5 is new mid-range index row.
        df_a = clean(
            generate_test_data(n_features=1, n_symbols=n_symbols, n_days=n_days, start_date=start_date),
            fs.sort_keys,
        )
        fs.add_features(df_a, overlap_mode="merge")
        assert mid_gap in fs.io.scan_delta(fs.index).collect()["timestamp"].to_list()

        # as_of physical table must be unchanged — no null sentinels injected.
        rows_after_a = fs.io.scan_delta(fs.get_path(table_addr_b)).collect().height
        assert rows_after_a == rows_before, f"align_feature_tables must not touch as_of tables: expected {rows_before} rows, got {rows_after_a}"

        # read() must return the carry-forward value for T5, NOT null.
        result_a = fs.read(["feature_1"])
        t5_result = result_a.filter(pl.col("timestamp") == mid_gap)
        assert t5_result.height == n_symbols
        assert t5_result["feature_1"].is_not_null().all(), (
            "join_asof carry-forward must supply T4's value for T5 — null here means sentinels were incorrectly injected"
        )

        # Scenario B: add normal feature_0 for T1–T11 → T11 is new beyond-last_updated index row.
        df_c = clean(
            generate_test_data(n_features=1, n_symbols=n_symbols, n_days=11, start_date=start_date),
            fs.sort_keys,
        )
        fs.add_features(df_c, overlap_mode="merge")
        assert t11 in fs.io.scan_delta(fs.index).collect()["timestamp"].to_list()

        rows_after_b = fs.io.scan_delta(fs.get_path(table_addr_b)).collect().height
        assert rows_after_b == rows_before, (
            f"align_feature_tables must not touch as_of tables for beyond-last_updated rows: expected {rows_before}, got {rows_after_b}"
        )

        # read() must return T10's carried-forward value for T11.
        result_b = fs.read(["feature_1"])
        t11_result = result_b.filter(pl.col("timestamp") == t11)
        assert t11_result.height == n_symbols
        assert t11_result["feature_1"].is_not_null().all(), (
            "join_asof carry-forward must supply T10's value for T11 — null here means sentinels were incorrectly injected"
        )

    def _test_read_validation_errors(self, fs: GammaFeatureLake):
        """Verify that read(feature_metadata_df) raises clear errors for invalid inputs:
        1. Duplicate feature names in the supplied DataFrame.
        """
        n_symbols = 3
        n_days = 5
        start_date = datetime(2020, 1, 1, tzinfo=UTC)

        df = clean(
            generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, start_date=start_date),
            fs.sort_keys,
        )
        fs.add_features(df)
        fs.add_features(
            clean(generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, start_date=start_date), fs.sort_keys),
            overlap_mode="merge",
        )
        fs.add_features(
            clean(generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days * 2, start_date=start_date), fs.sort_keys),
            overlap_mode="merge",
        )

        fm = fs.io.scan_delta(fs.feature_metadata).collect()

        # Case 1: duplicate feature name in the passed DataFrame.
        row_f0 = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)
        dup_df = pl.concat([row_f0, row_f0])
        with pytest.raises(ValueError, match="multiple versions"):
            fs.read(dup_df)

    def _test_local_vs_parallel_read(self, fs: GammaFeatureLake):
        """Verify serial (run_on_ray_cluster=False) and parallel (run_on_ray_cluster=True) reads
        return identical results from the same underlying data.
        """
        n_symbols = 3
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)

        df = clean(
            generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, start_date=start_date),
            fs.sort_keys,
        )
        fs.add_features(df)

        feature_names = ["feature_0", "feature_1"]
        serial_fs = fs.model_copy(update={"run_on_ray_cluster": False})
        parallel_fs = fs.model_copy(update={"run_on_ray_cluster": True})

        def assert_serial_parallel_equal(start=None, end=None):
            serial = serial_fs.read(feature_names, start=start, end=end).sort(fs.sort_keys)
            parallel = parallel_fs.read(feature_names, start=start, end=end).sort(fs.sort_keys)
            assert_frame_equal(serial, parallel, check_row_order=True, check_column_order=False)

        mid_start = start_date + timedelta(days=3)
        mid_end = start_date + timedelta(days=7)

        assert_serial_parallel_equal()
        assert_serial_parallel_equal(start=mid_start)
        assert_serial_parallel_equal(end=mid_end)
        assert_serial_parallel_equal(start=mid_start, end=mid_end)

    def _test_partial_symbol_overlap_nulls_missing_symbols(self, fs: GammaFeatureLake):
        """Verify that re-adding a feature for a subset of symbols in an overlapping date range
        nulls out the rows for symbols absent from the re-add.

        ``add_features`` has "complete replacement" semantics for the supplied date range: any
        (timestamp, symbol) pair in the existing table that falls within [input_min, input_max]
        but is absent from the new input is treated as having no value and is set to null.

        Setup:
          - Add feature_0 for T1–T10, symbols S1 and S2.
          - Re-add feature_0 for T5–T10 with S1 only (S2 absent).

        Expected:
          - T1–T4: S1 and S2 values preserved (outside the re-add range).
          - T5–T10: S1 updated to new values, S2 nulled (absent from re-add input).
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)  # T5

        # Initial add: S1 + S2 for all 10 days.
        initial = pl.DataFrame(
            {
                "timestamp": timestamps * 2,
                "symbol": ["S1"] * n_days + ["S2"] * n_days,
                "feature_0": list(range(n_days)) + list(range(100, 100 + n_days)),
            }
        )
        fs.add_features(initial)

        # Re-add: S1 only for T5–T10, with distinct sentinel values.
        s1_only = pl.DataFrame(
            {
                "timestamp": [overlap_start + timedelta(days=i) for i in range(6)],
                "symbol": ["S1"] * 6,
                "feature_0": [999] * 6,
            }
        )
        fs.add_features(s1_only, overlap_mode="merge")

        result = fs.read(["feature_0"]).sort(["timestamp", "symbol"])

        # T1–T4: both S1 and S2 unchanged.
        before_overlap = result.filter(pl.col("timestamp") < overlap_start)
        assert before_overlap["feature_0"].is_not_null().all(), "feature_0 values before the overlap range must be preserved"

        # T5–T10, S1: updated to sentinel value 999.
        s1_overlap = result.filter((pl.col("timestamp") >= overlap_start) & (pl.col("symbol") == "S1"))
        assert (s1_overlap["feature_0"] == 999).all(), "S1 values in overlap range must be updated"

        # T5–T10, S2: nulled because S2 was absent from the re-add input.
        s2_overlap = result.filter((pl.col("timestamp") >= overlap_start) & (pl.col("symbol") == "S2"))
        assert s2_overlap["feature_0"].is_null().all(), "S2 values in overlap range must be null (complete-replacement semantics)"

    def _test_beyond_range_index_entries_null_padded(self, fs: GammaFeatureLake):
        """Verify that global-index entries introduced by a *different* feature table, which
        lie beyond the current table's last_updated, are read as null for the current table.

        Polars ``concat(how="horizontal")`` null-pads shorter frames, so no physical null rows
        need to be written for index entries that post-date a table's last_updated.

        Setup:
          - Add feature_A for T1–T10 (S1 only).  Global index: T1–T10.
          - Add feature_B for T1–T15 (S1 only).  Global index grows to T1–T15.
          - Re-add feature_A for T8–T12 only.  T13–T15 are in the global index but NOT
            provided in the re-add input.

        Expected:
          - Reading feature_A: 15 rows total (T1–T15).
          - feature_A values for T13–T15 are null (index rows that exceed feature_A's physical
            range are automatically null-padded during horizontal concat).
          - feature_A values for T8–T12 are the re-add sentinel values.
          - feature_A values for T1–T7 are the original values.
        """
        n_days_short = 10
        n_days_long = 15
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days_long)]
        readd_end = start_date + timedelta(days=11)  # T12 inclusive (0-indexed day 11)

        df_a = pl.DataFrame(
            {
                "timestamp": timestamps[:n_days_short],
                "symbol": ["S1"] * n_days_short,
                "feature_A": list(range(n_days_short)),
            }
        )
        fs.add_features(df_a)

        df_b = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["S1"] * n_days_long,
                "feature_B": list(range(n_days_long)),
            }
        )
        fs.add_features(df_b)

        # Re-add feature_A for T8–T12 only; T13–T15 remain beyond feature_A's range.
        readd_start = start_date + timedelta(days=7)  # T8 (0-indexed day 7)
        df_a2 = pl.DataFrame(
            {
                "timestamp": [readd_start + timedelta(days=i) for i in range(5)],
                "symbol": ["S1"] * 5,
                "feature_A": [999] * 5,
            }
        )
        fs.add_features(df_a2, overlap_mode="merge")

        result = fs.read(["feature_A", "feature_B"]).sort("timestamp")

        assert result.height == n_days_long, f"Expected {n_days_long} rows, got {result.height}"

        # T1–T7: original feature_A values (0–6).
        original_range = result.filter(pl.col("timestamp") < readd_start)
        assert (original_range["feature_A"] == pl.Series(list(range(7)))).all(), (
            "feature_A values before the re-add range must be the original values"
        )

        # T8–T12: feature_A updated to sentinel 999.
        readd_range = result.filter((pl.col("timestamp") >= readd_start) & (pl.col("timestamp") <= readd_end))
        assert (readd_range["feature_A"] == 999).all(), "feature_A values in re-add range must be 999"

        # T13–T15: feature_A null (index entries beyond feature_A's physical range).
        beyond_range = result.filter(pl.col("timestamp") > readd_end)
        assert beyond_range.height == 3, f"Expected 3 rows beyond re-add range, got {beyond_range.height}"
        assert beyond_range["feature_A"].is_null().all(), "feature_A must be null for global-index entries beyond the table's physical range"

    def _test_colocated_feature_preserved_on_partial_remerge(self, fs: GammaFeatureLake):
        """Verify that when two features share a physical table and only one is re-added with
        an overlap, the other feature's values are preserved for all rows.

        Old code used write_delta(mode="overwrite") in the overlap path which replaced the
        entire physical table, silently dropping columns (and data) for co-located features
        not present in the re-add input.  The new merge_delta path only touches columns
        present in the source, so the sibling feature must be fully preserved.

        Setup:
          - Add feature_0 and feature_1 together in one call → they share a table_addr.
          - Re-add feature_0 only for T5–T10 (overlapping range).

        Expected:
          - feature_0 updated for T5–T10.
          - feature_1 values unchanged for ALL timestamps (T1–T10).
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)  # T5

        initial = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["S1"] * n_days,
                "feature_0": list(range(n_days)),
                "feature_1": list(range(100, 100 + n_days)),
            }
        )
        fs.add_features(initial)

        # Confirm both features share the same physical table.
        meta = fs.feature_metadata_frame().collect()
        addrs = meta.filter(pl.col("feature_name").is_in(["feature_0", "feature_1"]))["table_addr"].unique().to_list()
        assert len(addrs) == 1, f"Expected co-location in one table, got addresses: {addrs}"

        # Re-add feature_0 only with overlap.
        df0_overlap = pl.DataFrame(
            {
                "timestamp": [overlap_start + timedelta(days=i) for i in range(6)],
                "symbol": ["S1"] * 6,
                "feature_0": [999] * 6,
            }
        )
        fs.add_features(df0_overlap, overlap_mode="merge")

        result = fs.read(["feature_0", "feature_1"]).sort("timestamp")

        # feature_0: T1–T4 original, T5–T10 updated to 999.
        f0_before = result.filter(pl.col("timestamp") < overlap_start)["feature_0"].to_list()
        assert f0_before == list(range(4)), f"feature_0 before overlap should be 0–3, got {f0_before}"

        f0_overlap_vals = result.filter(pl.col("timestamp") >= overlap_start)["feature_0"].to_list()
        assert f0_overlap_vals == [999] * 6, f"feature_0 in overlap range must all be 999, got {f0_overlap_vals}"

        # feature_1: ALL values must be fully preserved (T1–T10).
        f1_vals = result["feature_1"].to_list()
        assert f1_vals == list(range(100, 110)), f"feature_1 must be entirely preserved after co-located re-add, got {f1_vals}"

    def _test_user_metadata_overlap_uses_merge(self, fs: GammaFeatureLake):
        """Verify that add_features(df, metadata=custom_df) still performs a correct in-place
        merge when there is an overlap, now that the old overwrite/copy-on-overlap path has
        been replaced with merge_delta.

        The ``metadata`` parameter now only affects _get_tables_to_update (Step 2 alignment
        exclusions); the Step 1 write path always uses merge_delta regardless.

        Setup:
          - Add feature_0 for T1–T10.
          - Re-add feature_0 for T5–T10 via add_features(df, metadata=custom_metadata_df).
          - custom_metadata_df is a valid feature_metadata slice (the current version row).

        Expected:
          - feature_0 values updated for T5–T10.
          - T1–T4 values preserved.
          - No orphaned tables created (table_addr unchanged).
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["S1"] * n_days,
                "feature_0": list(range(n_days)),
            }
        )
        fs.add_features(df)

        original_addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        # Build a metadata df that the caller "owns" (mimics an Airflow/ETL use case).
        custom_meta = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0").sort("version", descending=True).head(1)

        df2 = pl.DataFrame(
            {
                "timestamp": [overlap_start + timedelta(days=i) for i in range(6)],
                "symbol": ["S1"] * 6,
                "feature_0": [999] * 6,
            }
        )
        fs.add_features(df2, metadata=custom_meta, overlap_mode="merge")

        result = fs.read(["feature_0"]).sort("timestamp")

        # Values correct.
        assert result.filter(pl.col("timestamp") < overlap_start)["feature_0"].to_list() == list(range(4))
        assert result.filter(pl.col("timestamp") >= overlap_start)["feature_0"].to_list() == [999] * 6

        # table_addr must not have changed (still in-place merge, not copy-to-new-path).
        new_addr = (
            fs.feature_metadata_frame()
            .collect()
            .filter(pl.col("feature_name") == "feature_0")
            .sort("version", descending=True)
            .head(1)["table_addr"]
            .item()
        )
        assert new_addr == original_addr, f"table_addr must not change after merge-based overlap (was {original_addr}, got {new_addr})"

    def _test_as_of_feature_overlap_remerge(self, fs: GammaFeatureLake):
        """Verify that re-adding an as_of feature with an overlapping date range correctly
        updates values and does NOT null out existing as_of rows within the input range.

        as_of tables are excluded from when_not_matched_by_source_update (not_matched_predicate
        is set to None), so existing rows outside the input must be preserved.

        Setup:
          - Add regular feature_reg for T1–T10 (populates global index).
          - Add as_of feature_asof for T1–T8 with values 0–7.
          - Re-add feature_asof for T5–T10 with sentinel value 999.

        Expected:
          - T1–T4: original values 0–3 preserved.
          - T5–T10: updated to 999.
          - Reading via join_asof still returns non-null values for all T1–T10.
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)  # T5

        reg = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["S1"] * n_days,
                "feature_reg": list(range(n_days)),
            }
        )
        fs.add_features(reg)

        asof_initial = pl.DataFrame(
            {
                "timestamp": timestamps[:8],
                "symbol": ["S1"] * 8,
                "feature_asof": list(range(8)),
            }
        )
        fs.add_as_of_features(asof_initial, params={"strategy": "backward"})

        # Re-add as_of with overlap.
        asof_overlap = pl.DataFrame(
            {
                "timestamp": [overlap_start + timedelta(days=i) for i in range(6)],
                "symbol": ["S1"] * 6,
                "feature_asof": [999] * 6,
            }
        )
        fs.add_as_of_features(asof_overlap, params={"strategy": "backward"}, overlap_mode="merge")

        result = fs.read(["feature_reg", "feature_asof"]).sort("timestamp")

        # No nulls: as_of carry-forward means all rows have a value.
        assert result["feature_asof"].is_not_null().all(), "feature_asof must be non-null for all rows (join_asof carry-forward)"

        # T1–T4: original values.
        before = result.filter(pl.col("timestamp") < overlap_start)["feature_asof"].to_list()
        assert before == list(range(4)), f"as_of values before overlap must be 0–3, got {before}"

        # T5–T10: updated values.
        after = result.filter(pl.col("timestamp") >= overlap_start)["feature_asof"].to_list()
        assert after == [999] * 6, f"as_of values in overlap range must all be 999, got {after}"

    def _test_feature_params_json_on_overlap_remerge(self, fs: GammaFeatureLake):
        """Verify that re-adding an as_of feature with updated params via an overlapping
        merge correctly JSON-serializes the new params and that read_table can deserialize
        them via json.loads to drive join_asof.

        Old code used pl.lit(feature_params) in the overlap write path, which stored a
        Python dict repr (e.g. \"{'strategy': 'backward'}\") instead of valid JSON.  New
        code uses json.dumps consistently.

        Setup:
          - Add regular feature_reg for T1–T10 (populates index).
          - Add as_of feature_asof with params={'strategy': 'backward'}.
          - Re-add feature_asof with overlap and params={'strategy': 'backward', 'tolerance': '1d'}.

        Expected:
          - feature_metadata records the new params as valid JSON.
          - read(['feature_asof']) succeeds (would raise json.JSONDecodeError with old code).
          - The new params are retrievable via json.loads from feature_metadata.
        """
        import json as _json

        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        reg = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["S1"] * n_days,
                "feature_reg": list(range(n_days)),
            }
        )
        fs.add_features(reg)

        asof_initial = pl.DataFrame(
            {
                "timestamp": timestamps[:8],
                "symbol": ["S1"] * 8,
                "feature_asof": list(range(8)),
            }
        )
        fs.add_as_of_features(asof_initial, params={"strategy": "backward"})

        # Re-add with overlap and UPDATED params.
        new_params = {"strategy": "backward", "tolerance": "1d"}
        asof_overlap = pl.DataFrame(
            {
                "timestamp": [overlap_start + timedelta(days=i) for i in range(6)],
                "symbol": ["S1"] * 6,
                "feature_asof": [999] * 6,
            }
        )
        fs.add_as_of_features(asof_overlap, params=new_params, overlap_mode="merge")

        # The latest feature_params in feature_metadata must be valid JSON.
        meta = fs.feature_metadata_frame().collect()
        latest_params_str = (
            meta.filter(pl.col("feature_name") == "feature_asof").sort("version", descending=True, nulls_last=True).head(1)["feature_params"].item()
        )
        try:
            parsed = _json.loads(latest_params_str)
        except (_json.JSONDecodeError, TypeError) as exc:
            raise AssertionError(f"feature_params must be valid JSON after overlap re-add, got: {latest_params_str!r}") from exc

        assert parsed == new_params, f"Parsed params {parsed} != expected {new_params}"

        # read() must succeed (exercises json.loads inside read_table for as_of).
        result = fs.read(["feature_reg", "feature_asof"]).sort("timestamp")
        assert result.height == n_days
        assert result["feature_asof"].is_not_null().all()

    # ------------------------------------------------------------------
    # Tests for overlap_mode="copy" and mixed-mode scenarios
    # ------------------------------------------------------------------

    def _test_copy_mode_creates_new_table(self, fs: GammaFeatureLake):
        """Verify that overlap_mode='copy' (default) creates a new table_addr on overlap and
        that reads return correct values.
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        initial = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["S1"] * n_days,
                "feature_0": list(range(n_days)),
            }
        )
        fs.add_features(initial)  # default overlap_mode="copy"

        original_addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        df2 = pl.DataFrame(
            {
                "timestamp": [overlap_start + timedelta(days=i) for i in range(6)],
                "symbol": ["S1"] * 6,
                "feature_0": [999] * 6,
            }
        )
        fs.add_features(df2)  # default overlap_mode="copy" → new table

        fm = fs.feature_metadata_frame().collect()
        latest_addr = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert latest_addr != original_addr, "copy mode must create a new table_addr on overlap"

        # delta_version column no longer exists in feature_metadata
        assert "delta_version" not in fm.columns, "delta_version must not be present in feature_metadata"

        # Read values should be correct: pre-overlap original, overlap updated
        result = fs.read(["feature_0"]).sort("timestamp")
        before_overlap = result.filter(pl.col("timestamp") < overlap_start)
        assert before_overlap["feature_0"].to_list() == list(range(4))
        in_overlap = result.filter(pl.col("timestamp") >= overlap_start)
        assert in_overlap["feature_0"].to_list() == [999] * 6

    def _test_merge_mode_reuses_table(self, fs: GammaFeatureLake):
        """Verify that overlap_mode='merge' reuses the same table_addr and upserts data."""
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        initial = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["S1"] * n_days,
                "feature_0": list(range(n_days)),
            }
        )
        fs.add_features(initial, overlap_mode="merge")

        original_addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        df2 = pl.DataFrame(
            {
                "timestamp": [overlap_start + timedelta(days=i) for i in range(6)],
                "symbol": ["S1"] * 6,
                "feature_0": [999] * 6,
            }
        )
        fs.add_features(df2, overlap_mode="merge")

        fm = fs.feature_metadata_frame().collect()
        all_addrs = fm.filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().to_list()
        assert all_addrs == [original_addr], f"merge mode must keep the same table_addr, got {all_addrs}"

        # Reads correct
        result = fs.read(["feature_0"]).sort("timestamp")
        assert (result.filter(pl.col("timestamp") >= overlap_start)["feature_0"] == 999).all()

    def _test_mixed_overlap_modes(self, fs: GammaFeatureLake):
        """Verify that copy and merge modes can be mixed across different features and across
        re-adds of the same feature.  Each mode must produce correct reads independently.
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        # F1: add with copy mode, re-add with copy mode → 2 distinct table_addrs
        df_f1 = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_F1": list(range(n_days))})
        fs.add_features(df_f1)
        df_f1b = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(6)], "symbol": ["S1"] * 6, "feature_F1": [100] * 6})
        fs.add_features(df_f1b)  # copy (default)

        fm = fs.feature_metadata_frame().collect()
        f1_addrs = fm.filter(pl.col("feature_name") == "feature_F1")["table_addr"].unique().to_list()
        assert len(f1_addrs) == 2, f"F1 copy+copy should have 2 distinct table_addrs, got {f1_addrs}"

        # F2: add with merge mode, re-add with merge mode → 1 table_addr
        df_f2 = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_F2": list(range(100, 100 + n_days))})
        fs.add_features(df_f2, overlap_mode="merge")
        df_f2b = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(6)], "symbol": ["S1"] * 6, "feature_F2": [200] * 6})
        fs.add_features(df_f2b, overlap_mode="merge")

        fm = fs.feature_metadata_frame().collect()
        f2_addrs = fm.filter(pl.col("feature_name") == "feature_F2")["table_addr"].unique().to_list()
        assert len(f2_addrs) == 1, f"F2 merge+merge should have 1 table_addr, got {f2_addrs}"

        # Both features should read correctly
        result = fs.read(["feature_F1", "feature_F2"]).sort("timestamp")
        f1_overlap = result.filter(pl.col("timestamp") >= overlap_start)["feature_F1"].to_list()
        assert f1_overlap == [100] * 6, f"F1 overlap values wrong: {f1_overlap}"
        f2_overlap = result.filter(pl.col("timestamp") >= overlap_start)["feature_F2"].to_list()
        assert f2_overlap == [200] * 6, f"F2 overlap values wrong: {f2_overlap}"

    def _test_copy_then_merge_same_feature(self, fs: GammaFeatureLake):
        """Verify that a feature can be first added with copy mode and then re-added with merge
        mode (and vice versa).  The latest version always wins for reads regardless of the
        mode used for each individual write.
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        # Write 1: copy mode (initial)
        df1 = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_0": [0] * n_days})
        fs.add_features(df1)  # copy
        addr_v0 = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        # Write 2: copy mode (overlap → new table)
        df2 = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(6)], "symbol": ["S1"] * 6, "feature_0": [2] * 6})
        fs.add_features(df2)  # copy
        fm = fs.feature_metadata_frame().collect()
        addr_v1 = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert addr_v1 != addr_v0

        # Write 3: merge mode (overlap on the copy table → upsert in-place in v1's table)
        df3 = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(6)], "symbol": ["S1"] * 6, "feature_0": [3] * 6})
        fs.add_features(df3, overlap_mode="merge")
        fm = fs.feature_metadata_frame().collect()
        addr_v2 = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()

        # addr_v2 should be the same as addr_v1 (merge reuses the latest table)
        assert addr_v2 == addr_v1, f"Write 3 (merge) must target the latest table ({addr_v1}), got {addr_v2}"

        # Read must return write 3's values for the overlap range
        result = fs.read(["feature_0"]).sort("timestamp")
        overlap_vals = result.filter(pl.col("timestamp") >= overlap_start)["feature_0"].to_list()
        assert overlap_vals == [3] * 6, f"Expected [3]*6 after merge on top of copy, got {overlap_vals}"

    def _test_parametrized_overlap_modes_read_correctness(self, fs: GammaFeatureLake, overlap_mode: str):
        """Verify that both overlap modes produce identical read() output for a simple
        add-then-overlap scenario.  The feature values returned by read() should be the
        same regardless of whether copy or merge is used.
        """
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        df1 = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_0": list(range(n_days))})
        fs.add_features(df1, overlap_mode=overlap_mode)

        df2 = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(6)], "symbol": ["S1"] * 6, "feature_0": [999] * 6})
        fs.add_features(df2, overlap_mode=overlap_mode)

        result = fs.read(["feature_0"]).sort("timestamp")
        assert result.height == n_days
        # T1–T4: original values
        assert result.filter(pl.col("timestamp") < overlap_start)["feature_0"].to_list() == list(range(4))
        # T5–T10: updated to 999
        assert result.filter(pl.col("timestamp") >= overlap_start)["feature_0"].to_list() == [999] * 6

    def _test_index_alignment_invariant_copy(self, fs: GammaFeatureLake):
        """Verify the alignment invariant for copy mode.

        After adding new index rows beyond an existing table's range, every
        table_addr recorded in feature_metadata must scan (at latest physical version)
        to exactly the global index on its pk range.
        """
        n_days = 5
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        extra_days = [start_date + timedelta(days=n_days + i) for i in range(3)]

        df_a = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_A": list(range(n_days))})
        df_b = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_B": [10] * n_days})
        fs.add_features(df_a)
        fs.add_features(df_b)
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Extend feature_A with 3 new days beyond the initial range (pure append, no overlap).
        df_a2 = pl.DataFrame({"timestamp": extra_days, "symbol": ["S1"] * 3, "feature_A": [99] * 3})
        fs.add_features(df_a2)
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # feature_B must be null for the extra days (beyond its range — null-padded by horizontal concat).
        result = fs.read(["feature_A", "feature_B"]).sort("timestamp")
        assert result.height == n_days + 3, f"Expected {n_days + 3} rows, got {result.height}"
        beyond = result.filter(pl.col("timestamp").is_in(extra_days))
        assert beyond["feature_B"].is_null().all(), "feature_B must be null for days beyond its range"
        assert (beyond["feature_A"] == 99).all()

    def _test_index_alignment_invariant_merge(self, fs: GammaFeatureLake):
        """Verify the alignment invariant for merge mode.

        After a merge-mode write introduces a previously-missing index row, every
        table_addr in feature_metadata must scan (at latest physical version) to exactly
        the global index on its pk range — including the newly aligned tables updated
        by align_feature_tables.
        """
        n_days = 5
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        missing_day = timestamps[2]

        def without_missing(df):
            return df.filter(pl.col("timestamp") != missing_day)

        df_a = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_A": list(range(n_days))})
        df_b = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_B": [10] * n_days})
        fs.add_features(without_missing(df_a))
        fs.add_features(without_missing(df_b))
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Merge introduces missing_day; align_feature_tables must update feature_B's delta_version.
        fs.add_features(df_a, overlap_mode="merge")
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)
        self.verify_delta_version_alignment(fs)

        result = fs.read(["feature_A", "feature_B"]).sort("timestamp")
        assert result.height == n_days
        missing_row = result.filter(pl.col("timestamp") == missing_day)
        assert missing_row.height == 1
        assert missing_row["feature_A"].item() == 2
        assert missing_row["feature_B"].item() is None, "feature_B must be null at missing_day (alignment row)"

    def _test_index_alignment_invariant_mixed_modes(self, fs: GammaFeatureLake):
        """Verify the alignment invariant with mixed copy/merge table histories.

        feature_B is copy-mode. After a merge-mode write on feature_A introduces a new
        index row, align_feature_tables appends to feature_B and records a new
        feature_metadata entry. Every table_addr must still scan (at latest physical
        version) to exactly the global index on its pk range.
        """
        n_days = 5
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        new_day = start_date + timedelta(days=n_days)

        df_a = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_A": list(range(n_days))})
        df_b = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_B": [10] * n_days})
        fs.add_features(df_a, overlap_mode="merge")
        fs.add_features(df_b)
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)
        self.verify_delta_version_alignment(fs)

        df_a2 = pl.DataFrame({"timestamp": timestamps + [new_day], "symbol": ["S1"] * (n_days + 1), "feature_A": list(range(n_days)) + [99]})
        fs.add_features(df_a2, overlap_mode="merge")
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)
        self.verify_delta_version_alignment(fs)

        result = fs.read(["feature_A", "feature_B"]).sort("timestamp")
        assert result.height == n_days + 1
        new_row = result.filter(pl.col("timestamp") == new_day)
        assert new_row["feature_A"].item() == 99
        assert new_row["feature_B"].item() is None

    def _test_overlap_copy_creates_new_table(self, fs: GammaFeatureLake):
        """Verify copy mode creates a new table_addr on overlap, and reads are correct."""
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]

        df1 = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_0": list(range(n_days))})
        fs.add_features(df1)
        original_addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        # Re-add with overlap (copy mode, which is the default).
        df2 = pl.DataFrame({"timestamp": timestamps[5:], "symbol": ["S1"] * 5, "feature_0": [999] * 5})
        fs.add_features(df2, overlap_mode="copy")

        fm = fs.feature_metadata_frame().collect()
        latest_addr = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert latest_addr != original_addr, "copy mode must create a new table_addr on overlap"

        # Read must return correct values (original before overlap, 999 in overlap).
        result = fs.read(["feature_0"]).sort("timestamp")
        assert result.height == n_days
        before = result.filter(pl.col("timestamp") < timestamps[5])["feature_0"].to_list()
        assert before == list(range(5)), f"values before overlap must be original, got {before}"
        after = result.filter(pl.col("timestamp") >= timestamps[5])["feature_0"].to_list()
        assert after == [999] * 5, f"values in overlap range must be 999, got {after}"

    def _test_overlap_merge_reuses_table(self, fs: GammaFeatureLake):
        """Verify merge mode reuses the same table_addr on overlap, and reads are correct."""
        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]

        df1 = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_0": list(range(n_days))})
        fs.add_features(df1)
        original_addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        # Re-add with overlap (merge mode).
        df2 = pl.DataFrame({"timestamp": timestamps[5:], "symbol": ["S1"] * 5, "feature_0": [999] * 5})
        fs.add_features(df2, overlap_mode="merge")

        fm = fs.feature_metadata_frame().collect()
        latest_row = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)
        assert latest_row["table_addr"].item() == original_addr, "merge mode must reuse the same table_addr"

        # Read must return correct values.
        result = fs.read(["feature_0"]).sort("timestamp")
        assert result.height == n_days
        before = result.filter(pl.col("timestamp") < timestamps[5])["feature_0"].to_list()
        assert before == list(range(5)), f"values before overlap must be original, got {before}"
        after = result.filter(pl.col("timestamp") >= timestamps[5])["feature_0"].to_list()
        assert after == [999] * 5, f"values in overlap range must be 999, got {after}"

    def _test_copy_default_behavior(self, fs: GammaFeatureLake):
        """Verify the default add_features call uses copy mode (new table_addr on overlap)."""
        n_days = 5
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]

        df = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_0": list(range(n_days))})
        fs.add_features(df)
        addr_before = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        # Default overlap_mode="copy" should create a new table_addr.
        df2 = pl.DataFrame({"timestamp": timestamps[2:], "symbol": ["S1"] * 3, "feature_0": [99] * 3})
        fs.add_features(df2)  # no overlap_mode → defaults to "copy"

        fm = fs.feature_metadata_frame().collect()
        latest_addr = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert latest_addr != addr_before, "default (copy) mode must create a new table_addr on overlap"

    def _test_add_targets_overlap_modes(self, fs: GammaFeatureLake):
        """Verify overlap_mode works correctly for add_targets (copy and merge)."""
        n_days = 8
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        df = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "target_0": list(range(n_days))})
        fs.add_targets(df)
        addr_v0 = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "target_0")["table_addr"].unique().item()

        # copy mode: overlap → new table
        df2 = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(4)], "symbol": ["S1"] * 4, "target_0": [99] * 4})
        fs.add_targets(df2)  # default copy
        fm = fs.feature_metadata_frame().collect()
        addr_v1 = fm.filter(pl.col("feature_name") == "target_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert addr_v1 != addr_v0, "add_targets copy mode must create new table_addr"

        result = fs.read(["target_0"]).sort("timestamp")
        assert (result.filter(pl.col("timestamp") >= overlap_start)["target_0"] == 99).all()

        # merge mode: overlap → same table, delta_version recorded
        df3 = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(4)], "symbol": ["S1"] * 4, "target_0": [777] * 4})
        fs.add_targets(df3, overlap_mode="merge")
        fm = fs.feature_metadata_frame().collect()
        latest_addr = fm.filter(pl.col("feature_name") == "target_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert latest_addr == addr_v1, "add_targets merge mode must reuse the latest table_addr"

        result = fs.read(["target_0"]).sort("timestamp")
        assert (result.filter(pl.col("timestamp") >= overlap_start)["target_0"] == 777).all()

    def _test_add_as_of_features_copy_mode(self, fs: GammaFeatureLake):
        """Verify overlap_mode='copy' works correctly for add_as_of_features."""
        n_days = 8
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_start = start_date + timedelta(days=4)

        df = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "asof_0": list(range(n_days))})
        fs.add_as_of_features(df, params={})
        addr_v0 = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "asof_0")["table_addr"].unique().item()

        df2 = pl.DataFrame({"timestamp": [overlap_start + timedelta(days=i) for i in range(4)], "symbol": ["S1"] * 4, "asof_0": [99] * 4})
        fs.add_as_of_features(df2, params={})  # default copy
        fm = fs.feature_metadata_frame().collect()
        addr_v1 = fm.filter(pl.col("feature_name") == "asof_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert addr_v1 != addr_v0, "add_as_of_features copy mode must create new table_addr"

        result = fs.read(["asof_0"]).sort("timestamp")
        assert result.height > 0
        assert (result.filter(pl.col("timestamp") >= overlap_start)["asof_0"] == 99).all()

    def _test_mixed_signal_type_overlap_modes(self, fs: GammaFeatureLake):
        """Verify that features, targets and as_of_features can independently use copy or merge
        in the same store, with no cross-contamination of table_addr."""
        n_days = 8
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap_ts = [start_date + timedelta(days=i) for i in range(4, 8)]

        df_feat = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feat_0": list(range(n_days))})
        df_tgt = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "tgt_0": list(range(100, 100 + n_days))})
        df_asof = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "asof_0": list(range(200, 200 + n_days))})

        fs.add_features(df_feat)
        fs.add_targets(df_tgt)
        fs.add_as_of_features(df_asof, params={})

        # features: copy mode (default)
        fs.add_features(pl.DataFrame({"timestamp": overlap_ts, "symbol": ["S1"] * 4, "feat_0": [1] * 4}))
        # targets: merge mode
        fs.add_targets(pl.DataFrame({"timestamp": overlap_ts, "symbol": ["S1"] * 4, "tgt_0": [2] * 4}), overlap_mode="merge")
        # as_of: merge mode
        fs.add_as_of_features(pl.DataFrame({"timestamp": overlap_ts, "symbol": ["S1"] * 4, "asof_0": [3] * 4}), params={}, overlap_mode="merge")

        fm = fs.feature_metadata_frame().collect()

        # feat_0: copy → 2 distinct table_addrs
        feat_addrs = fm.filter(pl.col("feature_name") == "feat_0")["table_addr"].unique().len()
        assert feat_addrs == 2, f"feat_0 copy should have 2 addrs, got {feat_addrs}"

        # tgt_0: merge → 1 table_addr
        tgt_addrs = fm.filter(pl.col("feature_name") == "tgt_0")["table_addr"].unique().len()
        assert tgt_addrs == 1, f"tgt_0 merge should have 1 addr, got {tgt_addrs}"

        # asof_0: merge → 1 table_addr
        asof_addrs = fm.filter(pl.col("feature_name") == "asof_0")["table_addr"].unique().len()
        assert asof_addrs == 1, f"asof_0 merge should have 1 addr, got {asof_addrs}"

        # reads correct across all three
        result = fs.read(["feat_0", "tgt_0", "asof_0"]).sort("timestamp")
        overlap_rows = result.filter(pl.col("timestamp") >= start_date + timedelta(days=4))
        assert (overlap_rows["feat_0"] == 1).all()
        assert (overlap_rows["tgt_0"] == 2).all()
        assert (overlap_rows["asof_0"] == 3).all()

    def _test_read_equivalence_across_overlap_modes(self, fs_factory: Callable[[], GammaFeatureLake], modes: list[str]):
        """Verify that read() returns identical results regardless of which overlap_mode is used.

        Runs the same write sequence (initial insert + two overlapping updates) on independent
        feature stores — one per mode in ``modes`` — then asserts the final read() output is
        identical across all of them.

        This is the core correctness guarantee: overlap_mode affects storage structure
        (table_addr reuse vs copy), but must never affect the data returned
        by read().

        Args:
            fs_factory: zero-arg callable returning a fresh initialized GammaFeatureLake
            modes: list of overlap_mode strings to compare (e.g. ["copy", "merge"])
        """
        from polars.testing import assert_frame_equal

        n_days = 10
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        overlap1_ts = [start_date + timedelta(days=i) for i in range(3, 8)]
        overlap2_ts = [start_date + timedelta(days=i) for i in range(6, 10)]

        # Deterministic data — same values written to every store
        df_init = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * n_days, "feature_0": list(range(n_days))})
        df_overlap1 = pl.DataFrame({"timestamp": overlap1_ts, "symbol": ["S1"] * len(overlap1_ts), "feature_0": [100] * len(overlap1_ts)})
        df_overlap2 = pl.DataFrame({"timestamp": overlap2_ts, "symbol": ["S1"] * len(overlap2_ts), "feature_0": [200] * len(overlap2_ts)})

        results = {}
        for mode in modes:
            fs = fs_factory()
            fs.add_features(df_init, overlap_mode=mode)
            fs.add_features(df_overlap1, overlap_mode=mode)
            fs.add_features(df_overlap2, overlap_mode=mode)
            results[mode] = fs.read(["feature_0"]).sort("timestamp")

        reference_mode = modes[0]
        for mode in modes[1:]:
            (
                assert_frame_equal(
                    results[reference_mode],
                    results[mode],
                    check_row_order=True,
                    check_column_order=False,
                ),
                f"read() mismatch between overlap_mode='{reference_mode}' and '{mode}'",
            )

    def _test_read_equivalence_multi_symbol(self, fs_factory: Callable[[], GammaFeatureLake], modes: list[str]):
        """Same as _test_read_equivalence_across_overlap_modes but with multiple symbols and
        features to catch any symbol-level grouping differences between modes."""
        from polars.testing import assert_frame_equal

        n_days = 8
        n_symbols = 4
        n_features = 3
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        symbols = [f"S{i}" for i in range(n_symbols)]
        feature_cols = [f"feature_{i}" for i in range(n_features)]

        def make_df(val_offset: int, day_range: range) -> pl.DataFrame:
            rows = []
            for sym in symbols:
                for d in day_range:
                    row = {"timestamp": start_date + timedelta(days=d), "symbol": sym}
                    for fi in range(n_features):
                        row[f"feature_{fi}"] = val_offset + d + fi
                    rows.append(row)
            return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        df_init = make_df(0, range(n_days))
        df_overlap = make_df(1000, range(4, n_days))

        results = {}
        for mode in modes:
            fs = fs_factory()
            fs.add_features(df_init, overlap_mode=mode)
            fs.add_features(df_overlap, overlap_mode=mode)
            results[mode] = fs.read(feature_cols).sort(fs.sort_keys)

        reference_mode = modes[0]
        for mode in modes[1:]:
            (
                assert_frame_equal(
                    results[reference_mode],
                    results[mode],
                    check_row_order=True,
                    check_column_order=False,
                ),
                f"multi-symbol read() mismatch between overlap_mode='{reference_mode}' and '{mode}'",
            )

    def _test_last_updated_cutoff_boundary(self, fs: GammaFeatureLake):
        """Verify alignment when a merge write introduces a missing index row.

        When feature_C is written for days 5-9, missing_input_rows fills in null sentinel
        rows for days 0-4 (all existing index rows below its pk_max). So table C physically
        spans day0-day9 from the start, with nulls for days 0-4.

        When a merge on feature_A later introduces the previously-missing day2, align_feature_tables
        must append day2 to both B (which has real data in that range) and C (which already
        has null sentinels spanning that range). Both must pass verify_delta_version_alignment.
        """
        start = datetime(2020, 1, 1, tzinfo=UTC)
        days = [start + timedelta(days=i) for i in range(10)]
        missing_day = days[2]
        symbols = ["S1", "S2"]

        def make_df(feature: str, day_indices: list[int]) -> pl.DataFrame:
            rows = [{"timestamp": days[i], "symbol": s, feature: float(i)} for i in day_indices for s in symbols]
            return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        fs.add_features(make_df("feature_A", [0, 1, 3, 4]))  # skip day2
        fs.add_features(make_df("feature_B", [0, 1, 3, 4]))  # skip day2
        fs.add_features(make_df("feature_C", [5, 6, 7, 8, 9]))  # gets null sentinels for days 0-4

        fm = fs.feature_metadata_frame().collect()
        table_addr_c = fm.filter(pl.col("feature_name") == "feature_C")["table_addr"].unique().item()

        # C must already span day0-day9 due to missing_input_rows filling in null sentinels.
        c_before = fs.io.scan_delta(fs.get_path(table_addr_c)).collect()
        assert c_before[fs.primary_sort_key].min() == days[0], "feature_C pk_min must be day0 (null sentinels filled by missing_input_rows)"

        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Merge feature_A with day2 included → introduces day2 to the global index.
        fs.add_features(make_df("feature_A", [0, 1, 2, 3, 4]), overlap_mode="merge")

        # Both B and C must receive day2 alignment rows (day2 is within both tables' pk ranges).
        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        result = fs.read(["feature_A", "feature_B", "feature_C"]).sort("timestamp", "symbol")
        day2_rows = result.filter(pl.col("timestamp") == missing_day)
        assert day2_rows.height == len(symbols)
        assert (day2_rows["feature_A"] == 2.0).all()
        assert day2_rows["feature_B"].is_null().all()
        assert day2_rows["feature_C"].is_null().all()

    def _test_new_symbol_via_merge_aligned(self, fs: GammaFeatureLake):
        """Pitfall 2: verify that introducing a new symbol via merge aligns all co-resident tables.

        Initially both feature_A and feature_B exist for symbols {S1, S2}.
        A merge write on feature_A includes symbol S3 (new). The global index gains S3 rows.
        feature_B must then have S3 null-sentinel rows appended for every date in its range,
        and verify_delta_version_alignment must pass.
        """
        start = datetime(2020, 1, 1, tzinfo=UTC)
        days = [start + timedelta(days=i) for i in range(5)]

        def make_df(feature: str, syms: list[str], day_list=None) -> pl.DataFrame:
            d = days if day_list is None else day_list
            rows = [{"timestamp": t, "symbol": s, feature: 1.0} for t in d for s in syms]
            return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        fs.add_features(make_df("feature_A", ["S1", "S2"]))
        fs.add_features(make_df("feature_B", ["S1", "S2"]))

        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # Merge feature_A for all original days but now including new symbol S3.
        fs.add_features(make_df("feature_A", ["S1", "S2", "S3"]), overlap_mode="merge")

        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

        # feature_B must now have S3 null-sentinel rows for each day.
        fm = fs.feature_metadata_frame().collect()
        table_addr_b = fm.filter(pl.col("feature_name") == "feature_B")["table_addr"].unique().item()
        b_latest = fs.io.scan_delta(fs.get_path(table_addr_b)).collect()
        b_s3 = b_latest.filter(pl.col("symbol") == "S3")
        assert b_s3.height == len(days), f"feature_B must have {len(days)} S3 sentinel rows after new-symbol merge, got {b_s3.height}"
        assert b_s3["feature_B"].is_null().all(), "S3 sentinel rows in feature_B must be null"

        # End-to-end read must include S3 rows.
        result = fs.read(["feature_A", "feature_B"])
        s3_rows = result.filter(pl.col("symbol") == "S3")
        assert s3_rows.height == len(days), f"read() must return {len(days)} S3 rows, got {s3_rows.height}"
        assert (s3_rows["feature_A"] == 1.0).all(), "feature_A S3 values must be non-null"
        assert s3_rows["feature_B"].is_null().all(), "feature_B S3 values must be null (no data written for S3)"

    def _test_copy_mode_zero_overlap_reuses_table(self, fs: GammaFeatureLake):
        """Pitfall 4: copy mode with entirely new data (no pk overlap) must reuse the same table_addr.

        'copy' semantics mean 'on overlap, create a new snapshot'. When there is no overlap
        the same table should simply be extended (append-only), regardless of overlap_mode.
        """
        start = datetime(2020, 1, 1, tzinfo=UTC)
        days_1 = [start + timedelta(days=i) for i in range(5)]
        days_2 = [start + timedelta(days=i) for i in range(5, 10)]

        df1 = pl.DataFrame({"timestamp": days_1, "symbol": ["S1"] * 5, "feature_0": list(range(5))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df2 = pl.DataFrame({"timestamp": days_2, "symbol": ["S1"] * 5, "feature_0": list(range(5, 10))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        fs.add_features(df1)
        addr_v1 = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"].unique().item()

        # Pure extension: days 6-10 don't exist in the index yet — no overlap.
        fs.add_features(df2, overlap_mode="copy")

        fm = fs.feature_metadata_frame().collect()
        addr_v2 = fm.filter(pl.col("feature_name") == "feature_0").sort("version", descending=True, nulls_last=True).head(1)["table_addr"].item()
        assert addr_v2 == addr_v1, f"copy mode with zero overlap must reuse the same table_addr, got new addr {addr_v2}"

        result = fs.read(["feature_0"]).sort("timestamp")
        assert result.height == 10, f"Expected 10 rows after two non-overlapping writes, got {result.height}"
        assert result["feature_0"].to_list() == list(range(10))

        self.verify_index_alignment(fs)
        self.verify_delta_version_alignment(fs)

    def _test_merge_feature_asof_colocated_not_aligned(self, fs: GammaFeatureLake):
        """Pitfall 6: a merge write on a regular feature must NOT inject null sentinels into an as-of table.

        as_of_feature tables are excluded from align_feature_tables because they are read
        via join_asof carry-forward, which naturally handles gaps. Injecting nulls would corrupt
        that semantics. This test verifies:
        1. The as-of table's physical Delta version is unchanged after the merge write.
        2. The as-of table contains no null-sentinel rows for the new index row.
        3. read() on the as-of feature still returns correct carry-forward values.
        """
        start = datetime(2020, 1, 1, tzinfo=UTC)
        days = [start + timedelta(days=i) for i in range(5)]
        missing_day = days[2]

        def df_no_missing(feature: str, value: float) -> pl.DataFrame:
            rows = [{"timestamp": t, "symbol": "S1", feature: value} for t in days if t != missing_day]
            return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        # Write regular feature without missing_day.
        fs.add_features(df_no_missing("feature_A", 1.0))
        # Write as-of feature without missing_day.
        fs.add_as_of_features(df_no_missing("asof_0", 99.0), params={})

        fm = fs.feature_metadata_frame().collect()
        asof_addr = fm.filter(pl.col("feature_name") == "asof_0")["table_addr"].unique().item()
        asof_rows_before = fs.io.scan_delta(fs.get_path(asof_addr)).collect().height

        # Merge write on feature_A including missing_day → introduces missing_day to global index.
        full_df = pl.DataFrame([{"timestamp": t, "symbol": "S1", "feature_A": 1.0} for t in days]).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        fs.add_features(full_df, overlap_mode="merge")

        # as-of table must be completely untouched.
        asof_rows_after = fs.io.scan_delta(fs.get_path(asof_addr)).collect().height
        assert asof_rows_after == asof_rows_before, f"as-of table row count must not change (before={asof_rows_before}, after={asof_rows_after})"

        # read() on asof_0 must carry forward the last known value across missing_day.
        result = fs.read(["asof_0"]).sort("timestamp")
        missing_rows = result.filter(pl.col("timestamp") == missing_day)
        # join_asof carry-forward: missing_day should get the value from the preceding day.
        assert missing_rows.height == 1, f"Expected 1 row for missing_day in asof read, got {missing_rows.height}"
        assert missing_rows["asof_0"].item() == 99.0, f"join_asof carry-forward must supply 99.0 for missing_day, got {missing_rows['asof_0'].item()}"

    # ------------------------------------------------------------------
    # GammaFeatureLake.merge() tests
    # ------------------------------------------------------------------

    def _test_merge_static_identical_keys(self, fs: GammaFeatureLake):
        """merge() on two lakes with exactly the same (timestamp, symbol) index → no nulls."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(5)]
        df_left = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 5, "feat_a": list(range(5))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 5, "feat_b": list(range(100, 105))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_ml", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_mr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        result = GammaFeatureLake.merge(lake_left, lake_right, ["feat_a"], ["feat_b"])
        assert result["feat_a"].null_count() == 0
        assert result["feat_b"].null_count() == 0
        assert len(result) == 5

    def _test_merge_static_disjoint_keys(self, fs: GammaFeatureLake):
        """merge() on two lakes with no common (timestamp, symbol) pairs → each side has nulls for the other."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        left_timestamps = [start_date + timedelta(days=i) for i in range(3)]
        right_timestamps = [start_date + timedelta(days=i + 10) for i in range(3)]

        df_left = pl.DataFrame({"timestamp": left_timestamps, "symbol": ["S1"] * 3, "feat_a": [1.0, 2.0, 3.0]}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": right_timestamps, "symbol": ["S1"] * 3, "feat_b": [10.0, 20.0, 30.0]}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_dl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_dr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        result = GammaFeatureLake.merge(lake_left, lake_right, ["feat_a"], ["feat_b"])
        assert len(result) == 6
        assert result["feat_a"].null_count() == 3  # right-only rows have null feat_a
        assert result["feat_b"].null_count() == 3  # left-only rows have null feat_b

    def _test_merge_static_partial_overlap(self, fs: GammaFeatureLake):
        """merge() with partial key overlap: matched rows are full, unmatched have nulls."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        shared = [start_date + timedelta(days=i) for i in range(3)]
        left_only = [start_date + timedelta(days=i + 10) for i in range(2)]
        right_only = [start_date + timedelta(days=i + 20) for i in range(2)]

        df_left = pl.DataFrame({"timestamp": shared + left_only, "symbol": ["S1"] * 5, "feat_a": list(range(5))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": shared + right_only, "symbol": ["S1"] * 5, "feat_b": list(range(5))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_pol", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_por", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        result = GammaFeatureLake.merge(lake_left, lake_right, ["feat_a"], ["feat_b"])
        assert len(result) == 7  # 3 shared + 2 left-only + 2 right-only
        assert result["feat_a"].null_count() == 2  # right-only rows
        assert result["feat_b"].null_count() == 2  # left-only rows

    def _test_merge_static_multiple_symbols(self, fs: GammaFeatureLake):
        """merge() correctly aligns rows across multiple symbols."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(4)]

        # left lake has S1 and S2; right lake has S2 and S3
        left_rows = [{"timestamp": t, "symbol": s, "feat_a": float(i)} for i, (t, s) in enumerate((t, s) for t in timestamps for s in ["S1", "S2"])]
        right_rows = [{"timestamp": t, "symbol": s, "feat_b": float(i)} for i, (t, s) in enumerate((t, s) for t in timestamps for s in ["S2", "S3"])]

        df_left = pl.DataFrame(left_rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        df_right = pl.DataFrame(right_rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_msl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_msr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        result = GammaFeatureLake.merge(lake_left, lake_right, ["feat_a"], ["feat_b"])
        # S1: 4 rows (left only), S2: 4 rows (both), S3: 4 rows (right only)
        assert len(result) == 12
        s1 = result.filter(pl.col("symbol") == "S1")
        s2 = result.filter(pl.col("symbol") == "S2")
        s3 = result.filter(pl.col("symbol") == "S3")
        assert s1["feat_b"].null_count() == 4  # S1 has no right-side data
        assert s2["feat_a"].null_count() == 0
        assert s2["feat_b"].null_count() == 0
        assert s3["feat_a"].null_count() == 4  # S3 has no left-side data

    def _test_merge_static_result_matches_polars_join(self, fs: GammaFeatureLake):
        """merge() result equals a plain polars full outer join on the same data."""
        n_days, n_symbols = 10, 4
        df_left = clean(generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, feature_suffix="L"), ["timestamp", "symbol"])
        df_right = clean(
            generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, feature_suffix="R", symbols_id_start=2), ["timestamp", "symbol"]
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_jl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_jr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        left_read = lake_left.read(["L_0", "L_1"])
        right_read = lake_right.read(["R_0", "R_1"])
        expected = left_read.join(right_read, on=lake_left.sort_keys, how="full", coalesce=True).sort(lake_left.sort_keys)
        result = GammaFeatureLake.merge(lake_left, lake_right, ["L_0", "L_1"], ["R_0", "R_1"]).sort(lake_left.sort_keys)

        assert_frame_equal(result, expected, check_column_order=False)

    def _test_merge_static_date_range_filter(self, fs: GammaFeatureLake):
        """start/end parameters are applied to both lakes before merging."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(10)]
        df_left = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 10, "feat_a": list(range(10))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 10, "feat_b": list(range(100, 110))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_drl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_drr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        window_start = start_date + timedelta(days=3)
        window_end = start_date + timedelta(days=6)
        result = GammaFeatureLake.merge(lake_left, lake_right, ["feat_a"], ["feat_b"], start=window_start, end=window_end)

        # Only days 3, 4, 5, 6 (inclusive) should be present
        assert len(result) == 4
        assert result["timestamp"].min() >= window_start
        assert result["timestamp"].max() <= window_end

    def _test_merge_static_suffix_collision(self, fs: GammaFeatureLake):
        """Non-key columns with the same name in both lakes get the suffix applied."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(3)]
        df_left = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 3, "val": [1.0, 2.0, 3.0]}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 3, "val": [10.0, 20.0, 30.0]}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_scl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_scr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        result = GammaFeatureLake.merge(lake_left, lake_right, ["val"], ["val"], suffix="_right")
        assert "val" in result.columns
        assert "val_right" in result.columns
        assert result.sort("timestamp")["val"].to_list() == [1.0, 2.0, 3.0]
        assert result.sort("timestamp")["val_right"].to_list() == [10.0, 20.0, 30.0]

    def _test_merge_static_mismatched_sort_keys_raises(self, fs: GammaFeatureLake):
        """merge() raises ValueError when the two lakes have different sort_keys."""
        lake_a = GammaFeatureLake(base_path=fs.base_path + "_mka", run_on_ray_cluster=False).initialize(
            schema=ArrowSchema.make(pa.schema([("timestamp", pa.timestamp("us", tz="UTC")), ("symbol", pa.string())]))
        )
        # Second lake initialized with a different secondary key column name
        lake_b = GammaFeatureLake(base_path=fs.base_path + "_mkb", run_on_ray_cluster=False, primary_sort_key="timestamp").initialize(
            schema=ArrowSchema.make(pa.schema([("timestamp", pa.timestamp("us", tz="UTC")), ("PointID", pa.int64())]))
        )

        with pytest.raises(ValueError, match="sort_keys"):
            GammaFeatureLake.merge(lake_a, lake_b, [], [])

    def _test_merge_static_targets_included(self, fs: GammaFeatureLake):
        """merge() correctly passes through target columns from each lake."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(5)]
        df_left = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 5, "feat_a": list(range(5)), "tgt_a": list(range(10, 15))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 5, "feat_b": list(range(5)), "tgt_b": list(range(20, 25))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_tl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_tr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left.select(["timestamp", "symbol", "feat_a"]))
        lake_left.add_targets(df_left.select(["timestamp", "symbol", "tgt_a"]))
        lake_right.add_features(df_right.select(["timestamp", "symbol", "feat_b"]))
        lake_right.add_targets(df_right.select(["timestamp", "symbol", "tgt_b"]))

        result = GammaFeatureLake.merge(lake_left, lake_right, ["feat_a"], ["feat_b"], left_targets=["tgt_a"], right_targets=["tgt_b"])
        assert set(result.columns) == {"timestamp", "symbol", "feat_a", "tgt_a", "feat_b", "tgt_b"}
        assert result["tgt_a"].null_count() == 0
        assert result["tgt_b"].null_count() == 0

    def _test_merge_serial_parallel_equivalence(self, fs: GammaFeatureLake):
        """merge() with run_on_ray_cluster=True and run_on_ray_cluster=False return identical results."""
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(10)]
        df_left = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 10, "feat_a": list(range(10))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 10, "feat_b": list(range(100, 110))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_mspl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_mspr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        serial_left = lake_left.model_copy(update={"run_on_ray_cluster": False})
        serial_right = lake_right.model_copy(update={"run_on_ray_cluster": False})
        parallel_left = lake_left.model_copy(update={"run_on_ray_cluster": True})
        parallel_right = lake_right.model_copy(update={"run_on_ray_cluster": True})

        serial_result = GammaFeatureLake.merge(serial_left, serial_right, ["feat_a"], ["feat_b"]).sort(fs.sort_keys)
        parallel_result = GammaFeatureLake.merge(parallel_left, parallel_right, ["feat_a"], ["feat_b"]).sort(fs.sort_keys)

        assert_frame_equal(serial_result, parallel_result, check_row_order=True, check_column_order=False)

    def _test_merge_static_large_frame_performance(self, fs: GammaFeatureLake):
        """merge() on large sorted frames should complete without error (smoke test)."""
        import time

        n_days, n_symbols = 252, 100
        df_left = clean(generate_test_data(n_features=5, n_symbols=n_symbols, n_days=n_days, feature_suffix="L"), ["timestamp", "symbol"])
        df_right = clean(
            generate_test_data(n_features=5, n_symbols=n_symbols, n_days=n_days, feature_suffix="R", symbols_id_start=50), ["timestamp", "symbol"]
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_pefl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_pefr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        t0 = time.perf_counter()
        result = GammaFeatureLake.merge(lake_left, lake_right, [f"L_{i}" for i in range(5)], [f"R_{i}" for i in range(5)])
        elapsed = time.perf_counter() - t0

        assert len(result) > 0
        assert elapsed < 30.0, f"merge() took {elapsed:.1f}s — unexpectedly slow"

    def _test_merge_static_streaming_sort_merge(self, fs: GammaFeatureLake):
        """merge() uses streaming sort-merge join on polars >= _STREAMING_MERGE_JOIN_MIN_VERSION.

        Two correctness checks:
        1. The result matches a plain polars full-outer join (functional correctness is
           identical regardless of which engine path is taken).
        2. When the installed polars version supports streaming sort-merge joins, the lazy
           plan produced by merge() carries a ``hint.sorted`` annotation on the primary sort
           key of both sides, confirming the optimizer will promote to a sort-merge join.
        """
        n_days, n_symbols = 8, 3
        df_left = clean(generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, feature_suffix="L"), ["timestamp", "symbol"])
        df_right = clean(
            generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, feature_suffix="R", symbols_id_start=1), ["timestamp", "symbol"]
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_smjl", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_smjr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        # --- functional correctness ---
        result = GammaFeatureLake.merge(lake_left, lake_right, ["L_0", "L_1"], ["R_0", "R_1"])
        left_read = lake_left.read(["L_0", "L_1"])
        right_read = lake_right.read(["R_0", "R_1"])
        expected = left_read.join(right_read, on=lake_left.sort_keys, how="full", coalesce=True).sort(lake_left.sort_keys)
        assert_frame_equal(result.sort(lake_left.sort_keys), expected, check_column_order=False)

        # --- sort hint in the query plan (version-gated) ---
        if version.parse(pl.__version__) >= version.parse("1.37.0"):
            left_lf = lake_left.read(["L_0", "L_1"], materialized=False).set_sorted(lake_left.primary_sort_key)
            right_lf = lake_right.read(["R_0", "R_1"], materialized=False).set_sorted(lake_right.primary_sort_key)
            plan = left_lf.join(right_lf, on=lake_left.sort_keys, how="full", coalesce=True).explain(engine="streaming")
            pk = lake_left.primary_sort_key
            assert f"hint.sorted('{pk}'" in plan, f"Expected sort hint in streaming plan for polars {pl.__version__}, got:\n{plan}"

    def _test_merge_static_with_metadata_frame(self, fs: GammaFeatureLake):
        """merge() accepts pre-filtered metadata DataFrames in addition to feature name lists.

        Passing a filtered ``feature_metadata`` DataFrame lets callers pin a specific version
        or read a hand-curated subset — the same flexibility as the ``read(pl.DataFrame, ...)``
        dispatch.  The result must be identical to the equivalent list-based call.
        """
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [start_date + timedelta(days=i) for i in range(5)]
        df_left = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 5, "feat_a": list(range(5))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        df_right = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 5, "feat_b": list(range(100, 105))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )

        lake_left = GammaFeatureLake(base_path=fs.base_path + "_mml", run_on_ray_cluster=False).initialize()
        lake_right = GammaFeatureLake(base_path=fs.base_path + "_mmr", run_on_ray_cluster=False).initialize()
        lake_left.add_features(df_left)
        lake_right.add_features(df_right)

        # Pre-filtered metadata frames — same dispatch path as read(pl.DataFrame, ...)
        left_meta = lake_left.feature_metadata_frame().filter(pl.col("feature_name") == "feat_a").collect()
        right_meta = lake_right.feature_metadata_frame().filter(pl.col("feature_name") == "feat_b").collect()

        result_meta = GammaFeatureLake.merge(lake_left, lake_right, left_meta, right_meta)
        result_list = GammaFeatureLake.merge(lake_left, lake_right, ["feat_a"], ["feat_b"])

        assert result_meta["feat_a"].null_count() == 0
        assert result_meta["feat_b"].null_count() == 0
        assert len(result_meta) == 5
        assert_frame_equal(result_meta.sort(lake_left.sort_keys), result_list.sort(lake_left.sort_keys), check_column_order=False)

    def _test_sparse_features(self, fs: GammaFeatureLake):
        """Verify the behaviour of sparse features end-to-end.

        Four invariants are checked:

        1. **No null sentinels in physical storage**: the sparse physical table stores only the
           explicitly-uploaded rows; it is never padded with null sentinel rows.

        2. **Alignment skipped on subsequent dense uploads**: after adding new dense features
           (which grow the global index), the sparse physical table must remain unchanged.

        3. **Read path — nulls for missing rows**: ``fs.read()`` of a sparse feature returns every
           row in the bounded index with ``null`` where the sparse feature has no observation.

        4. **Read path — bounded index scan**: the returned rows are bounded by the max sparse
           date, so rows in the global index beyond the last sparse observation are not returned.
        """
        start_date = datetime(2020, 1, 1, tzinfo=UTC)
        n_symbols, n_days = 4, 20

        # --- upload dense features that populate the global index ---
        dense_df = clean(generate_test_data(n_features=2, n_symbols=n_symbols, n_days=n_days, start_date=start_date), fs.sort_keys)
        fs.add_features(dense_df, owner="test")

        # --- build a sparse dataset covering ~30 % of index rows, limited to first 10 days ---
        all_index = fs.index_frame().collect()
        sparse_cutoff = start_date + timedelta(days=10)
        sparse_pool = all_index.filter(pl.col(fs.primary_sort_key) <= sparse_cutoff)
        sparse_df = sparse_pool.sample(fraction=0.3, seed=42).with_columns(pl.lit(99.0).alias("sparse_val"))

        fs.add_sparse_features(sparse_df, owner="test")

        # --- Invariant 1: physical sparse table stores only the uploaded rows ---
        sparse_meta = fs.feature_metadata_frame().collect().filter(pl.col("signal_type") == "sparse_feature")
        assert sparse_meta.height == 1, f"Expected 1 sparse feature metadata row, got {sparse_meta.height}"
        sparse_table_addr = sparse_meta["table_addr"][0]
        physical_sparse = fs.io.scan_delta(fs.get_path(sparse_table_addr)).collect()
        assert physical_sparse.height == sparse_df.height, (
            f"Sparse physical table has {physical_sparse.height} rows; expected {sparse_df.height} (no null sentinels)"
        )
        assert physical_sparse["sparse_val"].null_count() == 0, "Sparse physical table must contain no null values"

        rows_before = physical_sparse.height

        # --- Invariant 2: adding new dense features does not touch the sparse table ---
        start_date2 = start_date + timedelta(days=n_days + 1)
        dense_df2 = clean(generate_test_data(n_features=1, n_symbols=n_symbols, n_days=5, start_date=start_date2, feature_suffix="new"), fs.sort_keys)
        fs.add_features(dense_df2, owner="test")
        rows_after = fs.io.scan_delta(fs.get_path(sparse_table_addr)).collect().height
        assert rows_after == rows_before, (
            f"Sparse physical table grew from {rows_before} to {rows_after} rows after dense upload — "
            "align_feature_tables must not inject null sentinels into sparse tables"
        )

        # --- Invariant 3: read() returns nulls for index rows where sparse data is absent ---
        result = fs.read(["sparse_val"])
        # Rows within the sparse date range: every index row is present, sparse value may be null
        result_in_range = result.filter(pl.col(fs.primary_sort_key) <= sparse_cutoff)
        non_null_vals = result_in_range.filter(pl.col("sparse_val").is_not_null())
        assert_frame_equal(
            non_null_vals.select(fs.sort_keys).sort(fs.sort_keys),
            sparse_df.select(fs.sort_keys).sort(fs.sort_keys),
        )
        # Rows with null values must correspond to index rows that were not in sparse_df
        null_keys = result_in_range.filter(pl.col("sparse_val").is_null()).select(fs.sort_keys)
        expected_null_keys = (
            all_index.filter(pl.col(fs.primary_sort_key) <= sparse_cutoff)
            .join(sparse_df.select(fs.sort_keys), on=fs.sort_keys, how="anti")
            .select(fs.sort_keys)
        )
        assert_frame_equal(null_keys.sort(fs.sort_keys), expected_null_keys.sort(fs.sort_keys))

        # --- Invariant 4: rows beyond max(sparse date) are not returned ---
        beyond_cutoff = result.filter(pl.col(fs.primary_sort_key) > sparse_cutoff)
        assert beyond_cutoff.height == 0, f"read() returned {beyond_cutoff.height} rows beyond the sparse cutoff date — index scan must be bounded"

    def _test_read_with_filtered_metadata_resolves_runtime_upstream_features(self, fs: GammaFeatureLake):
        """When passing a filtered feature_metadata DataFrame to read(), runtime features whose upstream
        root features are absent from the filter must still be computed correctly."""
        n_features = 3
        n_symbols = 5
        n_days = 10
        df = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days)
        fs.add_features(df, owner="test-owner")

        runtime_expr = (pl.col("feature_0") + pl.col("feature_1")).alias("runtime_sum")
        fs.add_runtime_computed_features([runtime_expr])

        # Build a filtered metadata frame that contains ONLY the runtime feature — not its upstream roots.
        # Before the fix, this caused the roots to be absent from the load and the computation to fail.
        runtime_only_metadata = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "runtime_sum")
        assert runtime_only_metadata.height > 0, "runtime_sum should be registered in metadata"

        result = fs.read(runtime_only_metadata)

        assert "runtime_sum" in result.columns
        assert "feature_0" in result.columns
        assert "feature_1" in result.columns
        expected = (
            df.drop_nulls()
            .unique(fs.sort_keys)
            .with_columns(runtime_expr)
            .select(["runtime_sum", "feature_1", "feature_0"] + fs.sort_keys)
            .sort(fs.sort_keys)
        )
        assert_frame_equal(result.sort(fs.sort_keys), expected, check_column_order=False)
