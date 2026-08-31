import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import polars as pl
import pyarrow as pa
import pytest
from ccflow import ArrowSchema
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gammalake import GammaFeatureLake
from gammalake.gamma_feature_lake import write_metadata
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

    def test_runtime_computed_features_require_opt_in(self, tmp_path):
        fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
        df = generate_test_data(n_features=2, n_symbols=2, n_days=2)
        fs.add_features(df, owner="test-owner")
        runtime_expr = (pl.col("feature_0") + pl.col("feature_1")).alias("runtime_feature")

        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.add_runtime_computed_features([runtime_expr])

        enabled_fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False, enable_runtime_computed_features=True)
        enabled_fs.add_runtime_computed_features([runtime_expr])
        runtime_metadata = enabled_fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "runtime_feature")

        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.read(["runtime_feature"])
        with pytest.raises(ValueError, match="enable_runtime_computed_features=True"):
            fs.read(runtime_metadata)

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


def test_add_features_single_commit_per_metadata_table(tmp_path):
    """add_features must batch metadata rows into one commit per non-empty metadata table."""
    fs = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    df = generate_test_data(n_symbols=5, n_days=3)
    with patch.object(fs.io, "write_delta") as mock_write:
        fs.add_features(df, owner="test")
        metadata_paths = [call.args[1] for call in mock_write.call_args_list if call.args[1] in (fs.table_metadata, fs.feature_metadata)]
        assert metadata_paths.count(fs.table_metadata) == 1
        assert metadata_paths.count(fs.feature_metadata) == 1
