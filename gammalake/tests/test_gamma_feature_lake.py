import os
import shutil
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import polars as pl
import pyarrow as pa
import pytest
from ccflow import ArrowSchema
from deltalake import DeltaTable
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gammalake import DeltaOptions, GammaFeatureLake, PartialWriteError
from gammalake.abstract import BaseFeatureLake
from gammalake.tests._model import frame, keys, ts
from gammalake.tests.base import GammaFeatureLakeTestsMixin, generate_test_data


class TestGammaFeatureLake(GammaFeatureLakeTestsMixin):
    """Unit test suite for GammaFeatureLake backed by local Delta tables.

    ``test_*`` wrappers for every ``_test_*(self, fs)`` method in the mixin are
    generated automatically via ``__init_subclass__``.  The ``fs`` fixture below
    provides each auto-wired test with a fresh, local GammaFeatureLake.

    Tests that need an uninitialised instance, extra parameters or ``fs_factory``
    are defined explicitly here.
    """

    @pytest.fixture
    def fs(self, tmp_path) -> GammaFeatureLake:
        return GammaFeatureLake(base_path=str(tmp_path)).initialize()

    # --- special cases ---

    def test_gen_table_addr_uses_prefix(self, tmp_path):
        lake = GammaFeatureLake(base_path=str(tmp_path), table_name_prefix="tables/")
        first = lake.gen_table_addr()
        second = lake.gen_table_addr()
        assert first.startswith("tables/delta_")
        assert len(first.removeprefix("tables/delta_")) == 8
        assert first != second

    def test_feature_tables_limit_stats_to_sort_keys(self, fs):
        fs.add_features(generate_test_data(n_features=4, n_symbols=3, n_days=5))
        addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"][0]
        table = DeltaTable(fs.get_path(addr))
        assert table.metadata().configuration.get("delta.dataSkippingStatsColumns") == ",".join(fs.sort_keys)
        adds = pl.from_arrow(table.get_add_actions(flatten=True))
        if isinstance(adds, pl.Series):
            adds = adds.struct.unnest()
        assert {column.removeprefix("min.") for column in adds.columns if column.startswith("min.")} == set(fs.sort_keys)

        for system_table in (fs.index, fs.feature_metadata, fs.table_metadata):
            assert "delta.dataSkippingStatsColumns" not in DeltaTable(system_table).metadata().configuration

    def test_copy_replacement_limits_stats_to_sort_keys(self, fs):
        timestamps = [datetime(2020, 1, day, tzinfo=UTC) for day in range(1, 4)]
        initial = pl.DataFrame({"timestamp": timestamps, "symbol": ["S1"] * 3, "feature_0": [1] * 3})
        fs.add_features(initial)
        old_addr = fs.feature_metadata_frame().collect()["table_addr"][0]

        fs.add_features(initial.with_columns(feature_0=pl.lit(2)))
        latest = fs.feature_metadata_frame().collect().sort("version", descending=True).row(0, named=True)

        assert latest["table_addr"] != old_addr
        configuration = DeltaTable(fs.get_path(latest["table_addr"])).metadata().configuration
        assert configuration.get("delta.dataSkippingStatsColumns") == ",".join(fs.sort_keys)

    def test_existing_feature_table_append_does_not_reset_configuration(self, fs):
        initial = generate_test_data(n_features=1, n_symbols=1, n_days=3)
        fs.add_features(initial)
        addr = fs.feature_metadata_frame().collect()["table_addr"][0]
        table_path = fs.get_path(addr)
        later = generate_test_data(n_features=1, n_symbols=1, n_days=3, start_date=datetime(2021, 1, 1, tzinfo=UTC))

        table = DeltaTable(table_path)
        version, configuration = table.version(), table.metadata().configuration

        fs.add_features(later)

        table = DeltaTable(table_path)
        assert fs.feature_metadata_frame().collect()["table_addr"].unique().to_list() == [addr]
        assert table.version() > version
        assert table.metadata().configuration == configuration

    def test_exceptions(self, tmp_path):
        fs_uninit = GammaFeatureLake(base_path=str(tmp_path))
        self._test_exceptions_uninitialized(fs_uninit)
        fs = fs_uninit.initialize()
        self._test_exceptions(fs)

    def test_exceptions_uninitialized(self, tmp_path):
        self._test_exceptions_uninitialized(GammaFeatureLake(base_path=str(tmp_path)))

    def test_read_with_filtered_metadata_resolves_runtime_upstream_features(self, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), enable_runtime_computed_features=True).initialize()
        self._test_read_with_filtered_metadata_resolves_runtime_upstream_features(fs)

    @pytest.mark.parametrize("overlap_mode", ["copy", "merge"])
    def test_parametrized_overlap_modes_read_correctness(self, overlap_mode, fs):
        self._test_parametrized_overlap_modes_read_correctness(fs, overlap_mode=overlap_mode)

    @staticmethod
    def _clean_frame(symbol: str, *feature_ids: int) -> pl.DataFrame:
        timestamps = pl.datetime_range(
            datetime(2020, 6, 20, tzinfo=UTC), datetime(2020, 6, 24, tzinfo=UTC), interval="1d", eager=True, time_zone="UTC"
        )
        frame = pl.DataFrame({"timestamp": timestamps, "symbol": symbol})
        return frame.with_columns(*(pl.arange(0, frame.height).cast(pl.Float64).alias(f"feature_{i}") for i in feature_ids))

    def test_as_of_read_requires_feature_params(self, fs):
        fs.add_as_of_features(self._clean_frame("A", 0), params={})
        metadata = fs.feature_metadata_frame().collect().with_columns(pl.lit(None, dtype=pl.String).alias("feature_params"))

        with pytest.raises(ValueError, match='as-of feature "feature_0" has no feature_params'):
            fs.read(metadata)

    @pytest.mark.parametrize(
        ("params", "observation_indices", "values", "start_index", "end_index", "expected"),
        [
            ({}, [0], [10.0], 2, 4, [10.0, 10.0, 10.0]),
            ({"strategy": "forward"}, [4], [20.0], 0, 2, [20.0, 20.0, 20.0]),
            ({"strategy": "nearest"}, [0, 4], [10.0, 20.0], 3, 3, [20.0]),
        ],
    )
    def test_as_of_range_preserves_out_of_range_context(
        self,
        fs,
        params,
        observation_indices,
        values,
        start_index,
        end_index,
        expected,
    ):
        days = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(5)]
        fs.add_as_of_features(
            pl.DataFrame(
                {
                    "timestamp": [days[index] for index in observation_indices],
                    "symbol": ["A"] * len(observation_indices),
                    "fundamental": values,
                }
            ),
            params=params,
        )
        fs.add_features(pl.DataFrame({"timestamp": days, "symbol": ["A"] * len(days), "dummy": [0] * len(days)}))

        result = fs.read(["fundamental"], start=days[start_index], end=days[end_index])

        assert result["fundamental"].to_list() == expected

    def test_multi_table_read_preserves_horizontal_alignment(self, fs):
        days = pl.datetime_range(
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 3, tzinfo=UTC),
            interval="1d",
            eager=True,
            time_zone="UTC",
        )
        keys = pl.DataFrame({"timestamp": [day for day in days for _ in ("S0", "S1")], "symbol": ["S0", "S1"] * len(days)})
        feature_0 = keys.with_columns(pl.Series("feature_0", [10.0, 11.0, 20.0, 21.0, 30.0, 31.0]))
        feature_1 = keys.with_columns(pl.Series("feature_1", [100.0, 101.0, 200.0, 201.0, 300.0, 301.0]))
        fs.add_features(feature_0)
        fs.add_features(feature_1)

        expected = feature_0.join(feature_1, on=fs.sort_keys).select(["feature_0", "feature_1"] + fs.sort_keys)

        assert_frame_equal(fs.read(["feature_0", "feature_1"]), expected, check_column_order=False)

    def test_multi_source_filter_plan_and_exact_equivalence(self, fs):
        fs.add_features(generate_test_data(n_features=1, n_symbols=3, n_days=5, feature_ids_start=0))
        fs.add_features(generate_test_data(n_features=2, n_symbols=3, n_days=5, feature_ids_start=1))
        user_filter = pl.col("symbol") == "Symbol_1"
        features = ["feature_0", "feature_1", "feature_2"]
        expected = fs.read(features).filter(user_filter).select("feature_0", *fs.sort_keys)

        lazy = fs.read(features, materialized=False).filter(user_filter).select("feature_0", *fs.sort_keys)
        outer_plan = lazy.explain(optimized=True)
        first = lazy.collect()
        second = lazy.collect()

        assert_frame_equal(first, expected)
        assert_frame_equal(second, expected)
        assert "PYTHON" in outer_plan and "SCAN" in outer_plan
        assert 'col("symbol")' in outer_plan
        assert '"Symbol_1"' in outer_plan

    def test_feature_filter_matches_filtering_unmaterialized_result(self, fs):
        fs.add_features(generate_test_data(n_features=2, n_symbols=3, n_days=5))
        lazy = fs.read(["feature_0", "feature_1"], materialized=False)
        user_filter = pl.col("feature_0") > pl.col("feature_0").median()

        expected = lazy.collect().filter(user_filter)
        result = lazy.filter(user_filter).collect()

        assert_frame_equal(result, expected)

    def test_projection_only_aggregation_remains_correct(self, fs):
        fs.add_features(generate_test_data(n_features=1, n_symbols=3, n_days=5, feature_ids_start=0))
        fs.add_features(generate_test_data(n_features=1, n_symbols=3, n_days=5, feature_ids_start=1))

        result = fs.read(["feature_0", "feature_1"], materialized=False).select(pl.len()).collect()

        assert result.item() == fs.index_frame().select(pl.len()).collect().item()

    def test_runtime_features_compute_before_lazy_filter(self, tmp_path):
        fs = GammaFeatureLake(
            base_path=str(tmp_path),
            enable_runtime_computed_features=True,
        ).initialize()
        days = [datetime(2020, 1, day, tzinfo=UTC) for day in range(1, 6)]
        fs.add_features(pl.DataFrame({"timestamp": days, "symbol": ["S0"] * len(days), "feature_0": [10.0, 20.0, 30.0, 40.0, 50.0]}))
        fs.add_runtime_computed_features([pl.col("feature_0").shift(1).alias("feature_0_lagged")])
        user_filter = pl.col("timestamp") == days[2]

        expected = fs.read(["feature_0_lagged"]).filter(user_filter)
        result = fs.read(["feature_0_lagged"], materialized=False).filter(user_filter).collect()

        assert_frame_equal(result, expected)
        assert result["feature_0_lagged"].item() == 20.0

    def test_as_of_features_compute_before_lazy_filter(self, fs):
        days = [datetime(2020, 1, day, tzinfo=UTC) for day in range(1, 5)]
        fs.add_as_of_features(
            pl.DataFrame({"timestamp": [days[0], days[2]], "symbol": ["S0", "S0"], "as_of": [10.0, 30.0]}),
            params={"strategy": "backward"},
        )
        fs.add_features(pl.DataFrame({"timestamp": days, "symbol": ["S0"] * len(days), "dense": [1.0, 2.0, 3.0, 4.0]}))
        user_filter = pl.col("timestamp") == days[1]

        expected = fs.read(["as_of"]).filter(user_filter)
        result = fs.read(["as_of"], materialized=False).filter(user_filter).collect()

        assert_frame_equal(result, expected)
        assert result["as_of"].item() == 10.0

    def test_mixed_dense_sparse_read_aligns_values_to_keys(self, fs):
        start = datetime(2020, 1, 1, tzinfo=UTC)
        days = [start + timedelta(days=i) for i in range(6)]
        dense = pl.DataFrame(
            {
                "timestamp": [day for day in days for _ in ("S0", "S1")],
                "symbol": ["S0", "S1"] * len(days),
                "dense_0": [float(i) for i in range(2 * len(days))],
            }
        )
        fs.add_features(dense)
        sparse = pl.DataFrame({"timestamp": [days[1], days[3]], "symbol": ["S1", "S0"], "sparse_val": [111.0, 333.0]})
        fs.add_sparse_features(sparse)

        result = fs.read(["dense_0", "sparse_val"])

        assert_frame_equal(result.select(fs.sort_keys + ["dense_0"]), dense.select(fs.sort_keys + ["dense_0"]), check_dtypes=False)
        non_null = result.filter(pl.col("sparse_val").is_not_null())
        assert_frame_equal(non_null.select(fs.sort_keys + ["sparse_val"]).sort(fs.sort_keys), sparse.sort(fs.sort_keys), check_dtypes=False)

    def test_multi_sparse_read_aligns_when_tails_differ(self, fs):
        start = datetime(2020, 1, 1, tzinfo=UTC)
        days = [start + timedelta(days=i) for i in range(5)]
        fs.add_features(pl.DataFrame({"timestamp": days, "symbol": ["S0"] * len(days), "dense_0": [1.0, 2.0, 3.0, 4.0, 5.0]}))
        fs.add_sparse_features(pl.DataFrame({"timestamp": [days[0], days[1]], "symbol": ["S0", "S0"], "early_val": [10.0, 20.0]}))
        fs.add_sparse_features(pl.DataFrame({"timestamp": [days[0], days[3]], "symbol": ["S0", "S0"], "late_val": [30.0, 40.0]}))

        result = fs.read(["early_val", "late_val"])
        by_day = {row["timestamp"]: row for row in result.iter_rows(named=True)}

        assert by_day[days[0]]["early_val"] == 10.0
        assert by_day[days[0]]["late_val"] == 30.0
        assert by_day[days[1]]["early_val"] == 20.0
        assert by_day[days[1]]["late_val"] is None
        assert by_day[days[3]]["early_val"] is None
        assert by_day[days[3]]["late_val"] == 40.0
        assert by_day[days[4]]["early_val"] is None
        assert by_day[days[4]]["late_val"] is None

    def test_sparse_read_with_no_rows_in_window_stays_bounded_to_end(self, fs):
        start = datetime(2020, 1, 1, tzinfo=UTC)
        days = [start + timedelta(days=i) for i in range(6)]
        fs.add_features(pl.DataFrame({"timestamp": days, "symbol": ["S0"] * len(days), "dense_0": [float(i) for i in range(len(days))]}))
        fs.add_sparse_features(pl.DataFrame({"timestamp": [days[4], days[5]], "symbol": ["S0", "S0"], "sparse_val": [40.0, 50.0]}))

        result = fs.read(["dense_0", "sparse_val"], [], days[0], days[2])

        assert result.height == 3
        assert result["timestamp"].to_list() == days[:3]
        assert result.filter(pl.col("timestamp").is_null()).is_empty()
        assert result["sparse_val"].to_list() == [None, None, None]

    def test_non_leading_primary_sort_key_preserves_alignment_and_bounds(self, tmp_path):
        schema = ArrowSchema.make(pa.schema([("symbol", pa.string()), ("timestamp", pa.timestamp("us", tz="UTC"))]))
        fs = GammaFeatureLake(base_path=str(tmp_path), primary_sort_key="timestamp").initialize(schema=schema)
        days = [datetime(2020, 1, day, tzinfo=UTC) for day in range(1, 4)]
        full = pl.DataFrame(
            {
                "symbol": ["S0"] * 3 + ["S1"] * 3,
                "timestamp": days * 2,
                "feature_0": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
            }
        )
        partial = pl.DataFrame({"symbol": ["S0", "S0"], "timestamp": days[:2], "feature_1": [100.0, 200.0]})
        fs.add_features(full)
        fs.add_features(partial)
        lazy = fs.read(["feature_0", "feature_1"], materialized=False)
        user_filter = pl.col("timestamp") == days[1]

        unfiltered = lazy.collect()
        filtered = lazy.filter(user_filter).collect()
        bounded = fs.read(["feature_0", "feature_1"], start=days[1], end=days[1])

        assert_frame_equal(filtered, unfiltered.filter(user_filter))
        assert_frame_equal(bounded, filtered, check_column_order=False)
        assert filtered["symbol"].to_list() == ["S0", "S1"]
        assert filtered["feature_1"].to_list() == [200.0, None]

    @pytest.mark.parametrize("modes", [["copy", "merge"], ["merge", "copy"], ["copy"], ["merge"]])
    def test_read_equivalence_across_overlap_modes(self, modes, tmp_path):
        def fs_factory():
            p = tmp_path / uuid.uuid4().hex
            p.mkdir()
            return GammaFeatureLake(base_path=str(p)).initialize()

        self._test_read_equivalence_across_overlap_modes(fs_factory, modes)

    @pytest.mark.parametrize("modes", [["copy", "merge"], ["merge", "copy"]])
    def test_read_equivalence_multi_symbol(self, modes, tmp_path):
        def fs_factory():
            p = tmp_path / uuid.uuid4().hex
            p.mkdir()
            return GammaFeatureLake(base_path=str(p)).initialize()

        self._test_read_equivalence_multi_symbol(fs_factory, modes)

    def test_runtime_transform_rejects_sort_key_collision(self, tmp_path):
        fs = GammaFeatureLake(
            base_path=str(tmp_path),
            primary_sort_key="value__abs",
            enable_runtime_computed_features=True,
        ).initialize(ArrowSchema.make(pa.schema([("value__abs", pa.int64()), ("symbol", pa.string())])))
        fs.add_features(pl.DataFrame({"value__abs": [1], "symbol": ["A"], "value": [2.0]}))

        with pytest.raises(ValueError, match="conflict with sort keys"):
            fs.add_runtime_transforms(features=["value"], transforms=["abs"])

    def test_runtime_computed_features_require_opt_in(self, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
        df = generate_test_data(n_features=2, n_symbols=2, n_days=2)
        fs.add_features(df, owner="test-owner")
        runtime_expr = (pl.col("feature_0") + pl.col("feature_1")).alias("runtime_feature")

        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.add_runtime_computed_features([runtime_expr])
        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.add_runtime_transforms(["feature_0"], ["abs"])

        enabled_fs = GammaFeatureLake(base_path=str(tmp_path), enable_runtime_computed_features=True)
        enabled_fs.add_runtime_computed_features([runtime_expr])
        enabled_fs.add_runtime_transforms(["feature_0"], ["abs"])
        runtime_metadata = enabled_fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "runtime_feature")
        transform_metadata = enabled_fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0__abs")

        with pytest.raises(ValueError, match="Runtime-computed features are disabled"):
            fs.read(["runtime_feature"])
        with pytest.raises(ValueError, match="Runtime-computed features are disabled"):
            fs.read(runtime_metadata)
        with pytest.raises(ValueError, match="Runtime-computed features are disabled"):
            fs.read(["feature_0__abs"])
        with pytest.raises(ValueError, match="Runtime-computed features are disabled"):
            fs.read(transform_metadata)

    # --- inline test not in mixin: tests Polars compression field ---

    def test_updating_old_features_while_changing_write_compression_level(self, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
        n_symbols = 10
        n_days = 10
        n_features = 5
        start_date = datetime(2020, 6, 20, tzinfo=UTC)

        df1 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_suffix="F1")
        fs.add_features(df1)
        df1 = df1.unique(subset=fs.sort_keys).filter(pl.all_horizontal(pl.col(key).is_not_null() for key in fs.sort_keys))
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)],
            feature_metadata_height=n_features,
            table_metadata_height=1,
            n_unique_deltatable=1,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )

        df2 = generate_test_data(n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date, feature_suffix="F2")
        fs = GammaFeatureLake(base_path=str(tmp_path), delta_options=DeltaOptions(compression="gzip"))
        fs.add_features(df2)
        df2 = df2.unique(subset=fs.sort_keys).filter(pl.all_horizontal(pl.col(key).is_not_null() for key in fs.sort_keys))
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)] + [f"F2_{i}" for i in range(n_features)],
            feature_metadata_height=2 * n_features,
            table_metadata_height=2,
            n_unique_deltatable=2,
            n_unique_versions=1,
            index_height=n_symbols * n_days,
        )

        df3 = generate_test_data(
            n_features=n_features, n_symbols=n_symbols, n_days=2 * n_days, start_date=start_date + timedelta(days=10), feature_suffix="F1"
        )
        fs = GammaFeatureLake(base_path=str(tmp_path), delta_options=DeltaOptions(compression="lz4"))
        fs.add_features(df3)
        df3 = df3.unique(subset=fs.sort_keys).filter(pl.all_horizontal(pl.col(key).is_not_null() for key in fs.sort_keys))
        self.assertExpected(
            fs,
            feature_names=[f"F1_{i}" for i in range(n_features)] + [f"F2_{i}" for i in range(n_features)],
            feature_metadata_height=2 * n_features,
            table_metadata_height=3,
            n_unique_deltatable=2,
            n_unique_versions=1,
            index_height=n_symbols * n_days + (2 * n_days * n_symbols),
        )

        df4 = generate_test_data(
            n_features=n_features, n_symbols=n_symbols, n_days=n_days, start_date=start_date + timedelta(days=20), feature_suffix="F2"
        )
        fs = GammaFeatureLake(base_path=str(tmp_path), delta_options=DeltaOptions(compression="SNAPPY"))
        fs.add_features(df4)
        df4 = df4.unique(subset=fs.sort_keys).filter(pl.all_horizontal(pl.col(key).is_not_null() for key in fs.sort_keys))
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

        with pytest.raises(ValidationError):
            GammaFeatureLake(base_path=str(tmp_path), delta_options={"compression": "MIDDLE_OUT"})

    def test_consolidate_features_and_targets(self, fs):
        features = generate_test_data(n_features=2, n_symbols=3, n_days=5)
        targets = generate_test_data(n_features=1, n_symbols=3, n_days=5, feature_suffix="target")
        fs.add_features(features)
        fs.add_targets(targets)

        expected = fs.read(["feature_0", "feature_1"], targets=["target_0"])
        new_addr = fs.consolidate_feature_groups(["feature_0", "feature_1", "target_0"])

        assert new_addr is not None
        assert DeltaTable(fs.get_path(new_addr)).metadata().configuration.get("delta.dataSkippingStatsColumns") == ",".join(fs.sort_keys)
        assert_frame_equal(
            expected.sort(fs.sort_keys),
            fs.read(["feature_0", "feature_1"], targets=["target_0"]).sort(fs.sort_keys),
            check_column_order=False,
        )

    def test_appending_existing_feature_pads_missing_index_rows(self, fs):
        fs.add_features(frame(keys([1], "A"), x=1))
        fs.add_features(frame(keys([1, 2, 3, 4], "A"), y=10))
        fs.add_features(frame(keys([5], "A"), x=1))

        expected = pl.DataFrame(
            {
                "timestamp": [ts(day) for day in range(1, 6)],
                "symbol": ["A"] * 5,
                "x": [1.0, None, None, None, 5.0],
                "y": [10.0, 20.0, 30.0, 40.0, None],
            }
        ).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))

        assert_frame_equal(
            fs.read(["x", "y"]).sort(fs.sort_keys),
            expected.sort(fs.sort_keys),
            check_column_order=False,
        )

    def test_write_at_last_updated_timestamp_keeps_feature_rows_aligned(self, fs):
        fs.add_features(frame([(1, "A"), (2, "A"), (3, "A")], x=1))
        fs.add_features(frame([(3, "B")], y=10))
        fs.add_features(frame([(4, "A")], x=1))

        result = fs.read(["x", "y"]).select("timestamp", "symbol", "x")
        expected = pl.DataFrame(
            {
                "timestamp": [ts(day) for day in (1, 2, 3, 3, 4)],
                "symbol": ["A", "A", "A", "B", "A"],
                "x": [1.0, 2.0, 3.0, None, 4.0],
            }
        ).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
        assert_frame_equal(result, expected)

    def test_partial_future_feature_updates(self, fs):
        fs.add_features(frame(keys([1, 2, 3], "A"), x=1, y=10))  # x and y share one feature table
        fs.add_features(frame(keys([1, 2, 3, 4, 5], "A"), z=100))  # the index now runs to day 5
        fs.add_features(frame(keys([4], "A"), x=1), overlap_mode="merge")  # write x alone for day 4

        expected = pl.DataFrame(
            {
                "timestamp": [ts(d) for d in range(1, 6)],
                "symbol": ["A"] * 5,
                "x": [1.0, 2.0, 3.0, 4.0, None],
                "y": [10.0, 20.0, 30.0, None, None],
                "z": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        ).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
        assert_frame_equal(fs.read(["x", "y", "z"]).sort(fs.sort_keys), expected, check_column_order=False)


def test_primary_sort_key_not_in_schema_raises(tmp_path):
    """initialize() raises ValueError when primary_sort_key is absent from the provided schema."""
    schema = ArrowSchema.make(pa.schema([("timestamp", pa.timestamp("us", tz="UTC")), ("symbol", pa.string())]))
    fs = GammaFeatureLake(base_path=str(tmp_path), primary_sort_key="not_a_column")
    with pytest.raises(ValueError, match="not_a_column"):
        fs.initialize(schema=schema)


def test_primary_sort_key_not_in_existing_index_raises(tmp_path):
    """Re-opening an existing lake with a primary_sort_key absent from the on-disk index raises ValidationError."""
    GammaFeatureLake(base_path=str(tmp_path)).initialize()
    with pytest.raises(ValidationError, match="not_a_column"):
        GammaFeatureLake(base_path=str(tmp_path), primary_sort_key="not_a_column")


def test_adds_take_data_frames_and_return_nothing(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    df = generate_test_data(n_features=1, n_symbols=2, n_days=3)

    assert fs.add_features(df) is None
    assert fs.add_targets(df.rename({"feature_0": "target_0"})) is None
    assert fs.add_index_rows(df.select(fs.sort_keys)) is None
    for add in (fs.add_features, fs.add_targets, fs.add_index_rows, fs.add_sparse_features):
        with pytest.raises(ValidationError):
            add(df.lazy())
    with pytest.raises(ValidationError):
        fs.add_as_of_features(df.to_dict(), params={})


def test_add_features_empty_table_no_crash(tmp_path):
    """_get_latest_feature_tables must return an empty frame when feature_columns contains only sort keys."""
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    df = generate_test_data(n_symbols=3, n_days=5).select(["timestamp", "symbol"])
    fs.add_features(df, owner="test")
    assert fs._get_latest_feature_tables(fs.sort_keys, fs.feature_metadata_frame().collect()).is_empty()
    assert fs.feature_metadata_frame().collect().is_empty()


def test_optional_backend_extensions_remain_compatible():
    assert "add_index_rows" not in BaseFeatureLake.__abstractmethods__


def test_add_features_single_commit_per_metadata_table(tmp_path):
    """add_features must batch metadata rows into one commit per non-empty metadata table."""
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    df = generate_test_data(n_symbols=5, n_days=3)
    before = {path: DeltaTable(path).version() for path in (fs.table_metadata, fs.feature_metadata)}

    fs.add_features(df, owner="test")

    assert {path: DeltaTable(path).version() for path in before} == {path: version + 1 for path, version in before.items()}


def _frame(days, value: float) -> pl.DataFrame:
    """One feature ``f`` for symbols ``a`` and ``b`` on each of ``days`` (day-of-January offsets)."""
    timestamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day) for day in days for _ in ("a", "b")]
    return pl.DataFrame({"timestamp": timestamps, "symbol": ["a", "b"] * len(days), "f": [float(value)] * len(timestamps)}).with_columns(
        pl.col("timestamp").dt.cast_time_unit("us")
    )


def _failing_lock(fails):
    """A lock factory raising on the ``n``-th lock of table ``name`` when ``fails(name, n)``."""
    counts = Counter()

    @contextmanager
    def lock(path):
        name = os.path.basename(path)
        counts[name] += 1
        if fails(name, counts[name]):
            raise RuntimeError(f"simulated failure locking {name}")
        yield

    return lock


def _log(fs: GammaFeatureLake) -> pl.DataFrame:
    return fs.transaction_log_frame().sort("updated_at").collect()


def test_transaction_log_records_pending_then_success(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path), transaction_log_path="writes").initialize()
    assert _log(fs).is_empty()

    fs.add_features(_frame(range(1, 3), 1), owner="me")

    log = _log(fs)
    assert log["status"].to_list() == ["pending", "success"]
    assert log["transaction_id"].n_unique() == 1
    assert log["owner"].to_list() == ["me", "me"]
    assert log["signal_type"].to_list() == ["feature", "feature"]
    assert DeltaTable(str(tmp_path / "writes")).version() == 2
    assert not (tmp_path / "transaction_log").exists()


def test_rejected_input_is_not_logged_as_a_transaction(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()

    with pytest.raises(ValueError, match="no rows"):
        fs.add_features(_frame([1], 1).clear())

    assert _log(fs).is_empty()


def test_new_index_values_can_be_disallowed(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_features(_frame(range(1, 4), 1), owner="me")
    index, features, log = fs.index_frame().collect(), fs.feature_metadata_frame().collect(), _log(fs)

    for df in (_frame([3, 4], 2).rename({"f": "g"}), _frame([4], 2).drop("f")):
        with pytest.raises(ValueError, match="2 input keys are not in the index"):
            fs.add_features(df, owner="me", allow_new_index_values=False)
    assert_frame_equal(fs.index_frame().collect(), index)
    assert_frame_equal(fs.feature_metadata_frame().collect(), features)
    assert_frame_equal(_log(fs), log)

    fs.add_features(_frame([2], 2).rename({"f": "g"}), owner="me", allow_new_index_values=False)
    assert_frame_equal(fs.index_frame().collect(), index)
    assert fs.read(["g"]).sort("timestamp", "symbol")["g"].to_list() == [None, None, 2.0, 2.0, None, None]


def test_failed_add_is_rolled_back_and_later_writes_do_not_see_it(tmp_path):
    """Regression test for upstream issue #52: feature tables must not keep a failed write that the index never received."""
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_features(_frame(range(1, 6), 1), owner="me")
    index_before = fs.index_frame().collect()
    (table,) = fs.table_metadata_frame().collect()["table_addr"]
    version_before = DeltaTable(fs.get_path(table)).version()

    failing = GammaFeatureLake(base_path=str(tmp_path), lock=_failing_lock(lambda name, _: name == "index"))
    with pytest.raises(RuntimeError, match="simulated failure locking index"):
        failing.add_features(_frame([6, 7], 2), owner="me")

    assert DeltaTable(fs.get_path(table)).version() == version_before + 2
    assert_frame_equal(fs.index_frame().collect(), index_before)
    assert fs.read(["f"]).sort("timestamp", "symbol")["f"].to_list() == [1.0] * 10
    log = _log(fs)
    assert log["status"].to_list()[-3:] == ["pending", "failed", "rolled_back"]
    assert "simulated failure locking index" in log["error"][-1]

    fs.add_features(_frame([8, 9], 3), owner="me")

    assert fs.read(["f"]).sort("timestamp", "symbol")["f"].to_list() == [1.0] * 10 + [3.0] * 4


def test_failed_rollback_leaves_transaction_failed(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_features(_frame(range(1, 6), 1), owner="me")
    (table,) = fs.table_metadata_frame().collect()["table_addr"]

    failing = GammaFeatureLake(base_path=str(tmp_path), lock=_failing_lock(lambda name, n: name == "index" or (name == table and n == 2)))
    with pytest.raises(PartialWriteError) as raised:
        failing.add_features(_frame([6, 7], 2), owner="me")

    assert raised.value.committed == [table]
    assert str(raised.value.__cause__) == f"simulated failure locking {table}"
    log = _log(fs)
    assert log["status"].to_list()[-2:] == ["pending", "failed"]
    assert "simulated failure locking index" in log["error"][-1]


def test_lake_created_before_the_transaction_log_creates_it_on_first_write(tmp_path):
    shutil.rmtree(GammaFeatureLake(base_path=str(tmp_path)).initialize().transaction_log)
    fs = GammaFeatureLake(base_path=str(tmp_path))

    fs.add_features(_frame(range(1, 3), 1), owner="me")

    assert _log(fs)["status"].to_list() == ["pending", "success"]
