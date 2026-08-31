import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import polars as pl
import pyarrow as pa
import pytest
from ccflow import ArrowSchema
from deltalake import DeltaTable
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gammalake import GammaFeatureLake
from gammalake._multi_source import scan_aligned_sources as original_scan_aligned_sources
from gammalake.abstract import BaseFeatureLake
from gammalake.gamma_feature_lake import write_metadata
from gammalake.io import FrameIO, PolarsIO
from gammalake.tests.base import GammaFeatureLakeTestsMixin, generate_test_data


@pytest.mark.usefixtures("barebones_ray_cluster")
class TestGammaFeatureLake(GammaFeatureLakeTestsMixin):
    """Unit test suite for GammaFeatureLake backed by local Delta tables.

    ``test_*`` wrappers for every ``_test_*(self, fs)`` method in the mixin are
    generated automatically via ``__init_subclass__``.  The ``fs`` fixture below
    provides each auto-wired test with a fresh, local (non-Ray) GammaFeatureLake.

    Tests that require Ray, parametrize over ``use_remote_data`` / ``use_ray_cluster``,
    need an uninitialised instance, or need ``fs_factory`` are defined explicitly here.
    """

    @pytest.fixture
    def fs(self, tmp_path) -> GammaFeatureLake:
        return GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()

    # --- special cases ---

    def test_feature_tables_limit_stats_to_sort_keys(self, fs):
        fs.add_features(generate_test_data(n_features=4, n_symbols=3, n_days=5))
        addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0")["table_addr"][0]
        table = DeltaTable(fs.get_path(addr))
        assert table.metadata().configuration.get("delta.dataSkippingStatsColumns") == ",".join(fs.sort_keys)
        adds = pl.from_arrow(table.get_add_actions(flatten=True))
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

        with patch.object(fs.io, "write_delta", wraps=fs.io.write_delta) as mock_write:
            fs.add_features(later)

        feature_writes = [call for call in mock_write.call_args_list if call.args[1] == table_path]
        assert feature_writes
        assert all("configuration" not in call.kwargs["delta_write_options"] for call in feature_writes)

    def test_exceptions(self, tmp_path):
        fs_uninit = GammaFeatureLake(base_path=str(tmp_path))
        self._test_exceptions_uninitialized(fs_uninit)
        fs = fs_uninit.initialize()
        self._test_exceptions(fs)

    def test_exceptions_uninitialized(self, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path))
        self._test_exceptions_uninitialized(fs)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_simple_operations(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_simple_operations(fs, use_remote_data=use_remote_data, run_on_ray_cluster=use_ray_cluster)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_adding_days(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_adding_days(fs, use_remote_data=use_remote_data, run_on_ray_cluster=use_ray_cluster)

    @pytest.mark.parametrize("use_ray_cluster", [True, False])
    def test_read_with_filtered_metadata_resolves_runtime_upstream_features(self, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster, enable_runtime_computed_features=True).initialize()
        self._test_read_with_filtered_metadata_resolves_runtime_upstream_features(fs)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_adding_symbols(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_adding_symbols(fs, use_remote_data=use_remote_data, run_on_ray_cluster=use_ray_cluster)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_appending_new_features(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_appending_new_features(fs, use_remote_data=use_remote_data, run_on_ray_cluster=use_ray_cluster)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_features_with_holes(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_features_with_holes(fs, use_remote_data=use_remote_data, run_on_ray_cluster=use_ray_cluster)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_updating_old_features(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_updating_old_features(fs)

    @pytest.mark.parametrize("use_ray_cluster", [True, False])
    def test_non_unique_index_data(self, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_non_unique_index_data(fs)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_add_index_rows(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_add_index_rows(fs, use_remote_data=use_remote_data)

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

    def test_feature_metadata_opened_once_per_read(self, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False, enable_runtime_computed_features=True).initialize()
        fs.add_as_of_features(self._clean_frame("A", 0), params={"strategy": "backward"})
        fs.add_targets(self._clean_frame("A", 1))
        fs.add_runtime_computed_features([(pl.col("feature_0") + 1).alias("runtime_feature")])

        with patch.object(type(fs), "feature_metadata_frame", autospec=True, side_effect=type(fs).feature_metadata_frame) as spy:
            fs.read(["runtime_feature"], targets=["feature_1"])
            fs.read(["runtime_feature"], targets=["feature_1"])

        assert spy.call_count == 2, "feature_metadata Delta table should be opened exactly once per read without a cross-call cache"

    def test_as_of_read_requires_feature_params(self, fs):
        fs.add_as_of_features(self._clean_frame("A", 0), params={})
        metadata = fs.feature_metadata_frame().collect().with_columns(pl.lit(None, dtype=pl.String).alias("feature_params"))

        with pytest.raises(ValueError, match="as_of_feature table .* requires feature_params"):
            fs.read(metadata)

    def test_parallel_read_passes_builtin_metadata_payload(self, fs):
        fs.add_features(self._clean_frame("A", 0, 1))
        tables = fs.feature_metadata_frame().collect()
        calls = []

        def switch(_self, function, **_kwargs):
            def capture(*args):
                calls.append(args)
                return function(*args)

            return capture

        with patch.object(type(fs), "switch", autospec=True, side_effect=switch):
            fs._load_features_from_tables_in_parallel(tables).collect()

        _, table_meta, features, *_ = calls[0]
        assert isinstance(table_meta, dict)
        assert isinstance(features, list)
        assert {"table_addr", "signal_type", "feature_params"} <= table_meta.keys()

    @pytest.mark.parametrize("run_on_ray_cluster", [False, True])
    def test_multi_table_read_preserves_horizontal_alignment(self, run_on_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
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
        reader = fs.model_copy(update={"run_on_ray_cluster": run_on_ray_cluster})

        expected = feature_0.join(feature_1, on=fs.sort_keys).select(["feature_0", "feature_1"] + fs.sort_keys)

        assert_frame_equal(reader.read(["feature_0", "feature_1"]), expected, check_column_order=False)

    def test_multi_source_filter_plan_and_exact_equivalence(self, fs):
        fs.add_features(generate_test_data(n_features=1, n_symbols=3, n_days=5, feature_ids_start=0))
        fs.add_features(generate_test_data(n_features=1, n_symbols=3, n_days=5, feature_ids_start=1))
        user_filter = pl.col("symbol") == "Symbol_1"
        expected = fs.read(["feature_0", "feature_1"], materialized=False).collect().filter(user_filter).select("feature_0", *fs.sort_keys)
        source_calls = {}
        source_plans = {}
        source_schemas = {}

        def tracked_source(name, frame):
            source_schemas[name] = frame.collect_schema()

            def source_generator(with_columns, predicate, n_rows, batch_size):
                source_calls.setdefault(name, []).append((with_columns, predicate))
                query = frame.filter(predicate) if predicate is not None else frame
                source_plans[name] = query.explain(optimized=True)
                if n_rows is not None:
                    query = query.head(n_rows)
                if with_columns is not None:
                    query = query.select(with_columns)
                result = query.collect()
                if batch_size is None or result.is_empty():
                    yield result
                else:
                    yield from result.iter_slices(n_rows=batch_size)

            return pl.io.plugins.register_io_source(source_generator, schema=frame.collect_schema(), is_pure=True)

        def capture_sources(sources, **kwargs):
            tracked = {name: tracked_source(name, frame) for name, frame in sources.items()}
            return original_scan_aligned_sources(tracked, **kwargs)

        with patch("gammalake.gamma_feature_lake.scan_aligned_sources", side_effect=capture_sources):
            lazy = fs.read(["feature_0", "feature_1"], materialized=False).filter(user_filter).select("feature_0", *fs.sort_keys)
            outer_plan = lazy.explain(optimized=True)
            first = lazy.collect()
            second = lazy.collect()

        assert_frame_equal(first, expected)
        assert_frame_equal(second, expected)
        assert "PYTHON SCAN" in outer_plan
        assert 'col("symbol")' in outer_plan
        assert '"Symbol_1"' in outer_plan
        assert len(source_calls) == 3
        assert all(len(calls) == 2 for calls in source_calls.values())
        assert all(calls[0][1] is not None and calls[0][1].meta.root_names() == ["symbol"] for calls in source_calls.values())
        assert all('col("symbol")' in plan and '"Symbol_1"' in plan for plan in source_plans.values())
        unused_source = next(name for name, schema in source_schemas.items() if "feature_1" in schema)
        assert source_calls[unused_source][0][0] == fs.sort_keys

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
            run_on_ray_cluster=False,
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
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False, primary_sort_key="timestamp").initialize(schema=schema)
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
            return GammaFeatureLake(base_path=str(p), run_on_ray_cluster=False).initialize()

        self._test_read_equivalence_across_overlap_modes(fs_factory, modes)

    @pytest.mark.parametrize("modes", [["copy", "merge"], ["merge", "copy"]])
    def test_read_equivalence_multi_symbol(self, modes, tmp_path):
        def fs_factory():
            p = tmp_path / uuid.uuid4().hex
            p.mkdir()
            return GammaFeatureLake(base_path=str(p), run_on_ray_cluster=False).initialize()

        self._test_read_equivalence_multi_symbol(fs_factory, modes)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_runtime_computed_features(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster, enable_runtime_computed_features=True).initialize()
        self._test_runtime_computed_features(fs, use_remote_data=use_remote_data)

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_runtime_transforms(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster, enable_runtime_computed_features=True).initialize()
        self._test_runtime_transforms(fs, use_remote_data=use_remote_data)

    def test_runtime_transform_rejects_sort_key_collision(self, tmp_path):
        fs = GammaFeatureLake(
            base_path=str(tmp_path),
            run_on_ray_cluster=False,
            primary_sort_key="value__abs",
            enable_runtime_computed_features=True,
        ).initialize(ArrowSchema.make(pa.schema([("value__abs", pa.int64()), ("symbol", pa.string())])))
        fs.add_features(pl.DataFrame({"value__abs": [1], "symbol": ["A"], "value": [2.0]}))

        with pytest.raises(ValueError, match="conflict with sort keys"):
            fs.add_runtime_transforms(features=["value"], transforms=["abs"])

    def test_runtime_computed_features_require_opt_in(self, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
        df = generate_test_data(n_features=2, n_symbols=2, n_days=2)
        fs.add_features(df, owner="test-owner")
        runtime_expr = (pl.col("feature_0") + pl.col("feature_1")).alias("runtime_feature")

        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.add_runtime_computed_features([runtime_expr])
        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.add_runtime_transforms(["feature_0"], ["abs"])

        enabled_fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False, enable_runtime_computed_features=True)
        enabled_fs.add_runtime_computed_features([runtime_expr])
        enabled_fs.add_runtime_transforms(["feature_0"], ["abs"])
        runtime_metadata = enabled_fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "runtime_feature")
        transform_metadata = enabled_fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "feature_0__abs")

        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.read(["runtime_feature"])
        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.read(runtime_metadata)
        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.read(["feature_0__abs"])
        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.read(transform_metadata)

    # --- inline test not in mixin: tests Polars compression field ---

    @pytest.mark.parametrize("use_remote_data,use_ray_cluster", [(True, True), (False, True), (False, False)])
    def test_updating_old_features_while_changing_write_compression_level(self, use_remote_data, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
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
        fs = GammaFeatureLake(base_path=str(tmp_path), compression="gzip", run_on_ray_cluster=use_ray_cluster)
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
        fs = GammaFeatureLake(base_path=str(tmp_path), compression="lz4", run_on_ray_cluster=use_ray_cluster)
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
        fs = GammaFeatureLake(base_path=str(tmp_path), compression="SNAPPY", run_on_ray_cluster=use_ray_cluster)
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
            GammaFeatureLake(base_path=str(tmp_path), compression="MIDDLE_OUT", run_on_ray_cluster=use_ray_cluster)

    @pytest.mark.parametrize("use_ray_cluster", [True, False])
    def test_consolidate_feature_groups(self, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_consolidate_feature_groups(fs)

    @pytest.mark.parametrize("use_ray_cluster", [True, False])
    def test_consolidate_independent_groups_stay_separate(self, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
        self._test_consolidate_independent_groups_stay_separate(fs)

    @pytest.mark.parametrize("use_ray_cluster", [True, False])
    def test_consolidate_rejects_non_consolidatable_signal_types(self, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(
            base_path=str(tmp_path),
            run_on_ray_cluster=use_ray_cluster,
            enable_runtime_computed_features=True,
        ).initialize()
        self._test_consolidate_rejects_non_consolidatable_signal_types(fs)

    @pytest.mark.parametrize("use_ray_cluster", [True, False])
    def test_consolidate_features_and_targets(self, use_ray_cluster, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=use_ray_cluster).initialize()
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


def test_primary_sort_key_not_in_schema_raises(tmp_path):
    """initialize() raises ValueError when primary_sort_key is absent from the provided schema."""
    schema = ArrowSchema.make(pa.schema([("timestamp", pa.timestamp("us", tz="UTC")), ("symbol", pa.string())]))
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False, primary_sort_key="not_a_column")
    with pytest.raises(ValueError, match="not_a_column"):
        fs.initialize(schema=schema)


def test_primary_sort_key_not_in_existing_index_raises(tmp_path):
    """Re-opening an existing lake with a primary_sort_key absent from the on-disk index raises ValidationError."""
    GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    with pytest.raises(ValidationError, match="not_a_column"):
        GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False, primary_sort_key="not_a_column")


def test_failed_write_does_not_corrupt_feature_metadata(tmp_path):
    """Regression test for primary sort key validation.

    If a feature write fails after the feature data is written but before table_metadata
    is written, a subsequent retry must not crash with a TypeError from comparing
    input_comparable_min against a null last_updated.

    We simulate the failure by patching write_delta to raise on the table_metadata write
    (the second write_delta call in the new-table path).
    """
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    df = generate_test_data(n_features=2, n_symbols=3, n_days=5)

    real_write_delta = fs.io.write_delta
    call_count = {"n": 0}

    def fail_on_table_metadata(frame, path, **kwargs):
        call_count["n"] += 1
        # The table_metadata write is the second write_delta call in the new-table path
        # (after the feature-data write). Raise here to simulate a partial failure.
        if call_count["n"] == 2:
            raise RuntimeError("simulated table_metadata write failure")
        return real_write_delta(frame, path, **kwargs)

    with patch.object(fs.io, "write_delta", side_effect=fail_on_table_metadata), pytest.raises(RuntimeError, match="simulated"):
        fs.add_features(df, owner="test-owner")

    # feature_metadata must be empty: its write was deferred past the failing call.
    assert fs.feature_metadata_frame().collect().is_empty()

    # A retry must succeed without crashing on a null last_updated comparison.
    fs.add_features(df, owner="test-owner")
    assert fs.read([f"feature_{i}" for i in range(2)]).height > 0


def test_write_metadata_single_commit(tmp_path):
    """write_metadata must issue exactly one write_delta call for any number of rows."""
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    row1 = pa.table({"col_a": [1], "col_b": ["x"]})
    row2 = pa.table({"col_a": [2], "col_b": ["y"]})
    with patch.object(fs.io, "write_delta") as mock_write:
        write_metadata(fs, fs.table_metadata, row1, None, row2)
        assert mock_write.call_count == 1


def test_add_features_empty_table_no_crash(tmp_path):
    """_get_latest_feature_tables must return an empty frame when feature_columns contains only sort keys."""
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    df = generate_test_data(n_symbols=3, n_days=5).select(["timestamp", "symbol"])
    fs.add_features(df, owner="test")
    assert fs._get_latest_feature_tables(fs.sort_keys, fs.feature_metadata_frame().collect()).is_empty()
    assert fs.feature_metadata_frame().collect().is_empty()


def test_add_features_scans_index_and_metadata_once(tmp_path):
    """Each add must reuse one snapshot of the index and metadata tables."""
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    df = generate_test_data(n_features=2, n_symbols=3, n_days=5)
    fs.add_features(df, owner="test")

    with patch.object(fs.io, "scan_delta", wraps=fs.io.scan_delta) as mock_scan:
        fs.add_features(df.with_columns(pl.col("feature_0") + 1), owner="test")

    scanned_paths = [call.args[0] for call in mock_scan.call_args_list]
    assert scanned_paths.count(fs.index) == 1
    assert scanned_paths.count(fs.feature_metadata) == 1
    assert scanned_paths.count(fs.table_metadata) == 1


def test_write_metadata_all_none_skips_write(tmp_path):
    """write_metadata must not call write_delta when every row is None."""
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    with patch.object(fs.io, "write_delta") as mock_write:
        write_metadata(fs, fs.table_metadata, None, None)
        assert mock_write.call_count == 0


def test_write_metadata_schema_mode_forwarded(tmp_path):
    """write_metadata must pass schema_mode through to delta_write_options."""
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    row = pa.table({"col_a": [1], "col_b": ["x"]})
    with patch.object(fs.io, "write_delta") as mock_write:
        write_metadata(fs, fs.table_metadata, row, schema_mode="merge")
        assert mock_write.call_count == 1
        _, kwargs = mock_write.call_args
        assert kwargs["delta_write_options"].get("schema_mode") == "merge"


def test_failed_index_write_does_not_publish_metadata(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    df = generate_test_data(n_features=2, n_symbols=3, n_days=5)

    with (
        patch("gammalake.gamma_feature_lake.update_index", side_effect=RuntimeError("simulated index failure")),
        pytest.raises(RuntimeError, match="simulated index failure"),
    ):
        fs.add_features(df, owner="test-owner")

    assert fs.index_frame().collect().is_empty()
    assert fs.table_metadata_frame().collect().is_empty()
    assert fs.feature_metadata_frame().collect().is_empty()

    fs.add_features(df, owner="test-owner")
    assert fs.read(["feature_0", "feature_1"]).height == fs.index_frame().collect().height


def test_optional_backend_extensions_remain_compatible():
    assert {"annotate_table", "describe_table", "restore_to_timestamp"}.isdisjoint(FrameIO.__abstractmethods__)
    assert "add_index_rows" not in BaseFeatureLake.__abstractmethods__


def test_polars_io_forwards_delta_configuration():
    configuration = {"delta.dataSkippingStatsColumns": "timestamp,symbol"}
    with patch.object(pl.DataFrame, "write_delta") as mock_write:
        PolarsIO().write_delta(pl.DataFrame({"value": [1]}), "unused", delta_write_options={"configuration": configuration})

    mock_write.assert_called_once_with("unused", delta_write_options={"configuration": configuration})


def test_add_features_single_commit_per_metadata_table(tmp_path):
    """add_features must batch metadata rows into one commit per non-empty metadata table."""
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    df = generate_test_data(n_symbols=5, n_days=3)
    with patch.object(fs.io, "write_delta") as mock_write:
        fs.add_features(df, owner="test")
        metadata_paths = [call.args[1] for call in mock_write.call_args_list if call.args[1] in (fs.table_metadata, fs.feature_metadata)]
        assert metadata_paths.count(fs.table_metadata) == 1
        assert metadata_paths.count(fs.feature_metadata) == 1
