import datetime as dt
import gc
import itertools
import json
import random
import shutil
import threading
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow as pa
import pytest
from deltalake import write_deltalake
from polars.testing import assert_frame_equal
from pydantic import ValidationError

import gammalake
from gammalake import DeltaOptions, GammaFeatureLake, UnsupportedError, gammalake as native
from gammalake.abstract import MissingFeaturesException, MissingOrMisregisteredSignalsException, UninitializedDeltaLakeException
from gammalake.gamma_feature_lake import ArrowSchema
from gammalake.native import NativeLake

UTC = dt.UTC
KEYS = ["timestamp", "symbol"]
SYMBOLS = ["A", "B", "C"]


def test_native_version_matches_package():
    assert native.version() == gammalake.__version__


def ts(day: int) -> dt.datetime:
    return dt.datetime(2024, 1, day, tzinfo=UTC)


def frame(days, symbols, **features) -> pl.DataFrame:
    rows = list(itertools.product(days, symbols))
    data = {"timestamp": [ts(d) for d, _ in rows], "symbol": [s for _, s in rows]}
    for name, scale in features.items():
        data[name] = [float(scale * d + (ord(s) - ord("A")) / 10) for d, s in rows]
    return pl.DataFrame(data).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))


@pytest.fixture(scope="module")
def lake_path(tmp_path_factory) -> str:
    """Several tables, a target, tail rows missing from older tables and a sparse new symbol."""
    path = str(tmp_path_factory.mktemp("native_lake"))
    fs = GammaFeatureLake(base_path=path).initialize()
    fs.add_features(frame(range(1, 7), SYMBOLS, f1=1, f2=2))
    fs.add_targets(frame(range(1, 7), SYMBOLS, t1=-1))
    fs.add_features(frame(range(4, 10), SYMBOLS, f3=3))
    fs.add_features(frame([3, 6], ["D"], f4=4))
    return path


@pytest.fixture
def lake(lake_path) -> GammaFeatureLake:
    return GammaFeatureLake(base_path=lake_path).initialize()


def utc(bound):
    if bound is None or isinstance(bound, dt.datetime):
        return bound and bound.astimezone(UTC)
    return dt.datetime.combine(bound, dt.time(), UTC)


def latest(fs: GammaFeatureLake, names: list[str]) -> pl.DataFrame:
    metadata = pl.read_delta(fs.feature_metadata).sort("version", descending=True, nulls_last=True)
    return pl.concat([metadata.filter(pl.col("feature_name") == name).head(1) for name in names])


def reference_read(fs: GammaFeatureLake, selected: pl.DataFrame, start=None, end=None) -> pl.DataFrame:
    """Reads feature metadata rows from the raw Delta tables with Polars joins.

    Bounds are inclusive. Dense and sparse tables are left-joined to the index; as-of tables
    are ``join_asof``-ed with their stored parameters, keeping rows before ``start`` for
    backward joins and after ``end`` for forward joins.
    """
    start, end = utc(start), utc(end)
    after = pl.col("timestamp") >= start if start else pl.lit(True)
    before = pl.col("timestamp") <= end if end else pl.lit(True)
    index = pl.read_delta(fs.index).select(KEYS).filter(after & before)
    result = index
    for row in selected.iter_rows(named=True):
        name = row["feature_name"]
        table = pl.read_delta(fs.get_path(row["table_addr"])).select(*KEYS, name)
        if row["signal_type"] == "as_of_feature":
            params = json.loads(row["feature_params"])
            strategy = params.get("strategy", "backward")
            table = table.filter(before if strategy == "backward" else after if strategy == "forward" else pl.lit(True))
            table = index.sort("symbol", "timestamp").join_asof(table.sort("symbol", "timestamp"), on="timestamp", by="symbol", **params)
        result = result.join(table, on=KEYS, how="left")
    return result.select(*selected["feature_name"], *KEYS).sort(KEYS)


def by_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return frame.select(columns).sort(["timestamp", "symbol"])


# --------------------------------------------------------------------- streams


def test_stream_is_consumed_once(lake_path):
    stream = native.Lake(lake_path).read(["f1"])
    assert not stream.consumed
    assert pa.schema(stream.schema).names == ["f1", "timestamp", "symbol"]
    assert pa.RecordBatchReader.from_stream(stream).read_all().num_rows == 29
    assert stream.consumed
    with pytest.raises(RuntimeError, match="already consumed"):
        stream.__arrow_c_stream__()


def test_dropped_streams_are_released(lake_path):
    gc.collect()
    before = native._live_streams()
    lake = native.Lake(lake_path)
    streams = [lake.read(["f1", "f3"]) for _ in range(3)]
    reader = pa.RecordBatchReader.from_stream(streams[0])
    assert native._live_streams() == before + 3
    del streams, reader
    gc.collect()
    assert native._live_streams() == before


def test_predicate_import_errors(lake_path):
    lake = native.Lake(lake_path)
    with pytest.raises(TypeError, match="__arrow_c_array__"):
        lake.read(["f1"], None, [[("symbol", "==", ["A"])]])
    with pytest.raises(ValueError):
        lake.read(["f1"], None, [[("symbol", "~", pa.array(["A"]))]])
    with pytest.raises(ValueError):
        lake.read(["f1"], None, [[("f1", "==", pa.array([1.0]))]])


def test_reads_release_the_gil(lake_path):
    lake = NativeLake(lake_path)
    expected = lake.read(["f1", "f3"])
    results, errors = [], []

    def work():
        try:
            for _ in range(5):
                results.append(lake.read(["f1", "f3"]))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors and len(results) == 20
    for result in results:
        assert_frame_equal(result, expected)


# --------------------------------------------------------------------- errors


def test_error_mapping(lake_path, tmp_path):
    lake = NativeLake(lake_path)
    with pytest.raises(MissingFeaturesException):
        lake.read(["nope"])
    with pytest.raises(MissingOrMisregisteredSignalsException):
        lake.read(["f1"], ["f2"])
    with pytest.raises(UninitializedDeltaLakeException):
        NativeLake(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="unknown filter column"):
        lake.read(["f1"], filters=[("nope", "=", 1)])
    with pytest.raises(UnsupportedError):
        lake.read(["f1"], start="2024-01-02")


@pytest.mark.parametrize("table", ["index", "feature_metadata", "table_metadata"])
def test_initialize_repairs_any_missing_required_table(tmp_path, table):
    GammaFeatureLake(base_path=str(tmp_path)).initialize()
    shutil.rmtree(tmp_path / table)

    with pytest.raises(UninitializedDeltaLakeException):
        NativeLake(str(tmp_path))

    repaired = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    assert (tmp_path / table / "_delta_log").is_dir()
    assert repaired.index_frame().collect().is_empty()


def test_validate_reports_misaligned_tables(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_features(frame([1], ["A"], x=1))
    fs.add_features(frame(range(1, 5), ["A"], y=10))
    addr = fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "x")["table_addr"].item()
    write_deltalake(fs.get_path(addr), frame([3], ["A"], x=1).to_arrow(), mode="append")
    assert NativeLake(str(tmp_path)).read(["x", "y"]).height == 4
    with pytest.raises(RuntimeError, match="integrity check failed for table"):
        NativeLake(str(tmp_path), validate=True).read(["x", "y"])
    keys = NativeLake(str(tmp_path)).scan(["x", "y"]).select("timestamp", "symbol")
    assert keys.collect().height == 4
    with pytest.raises(Exception, match="integrity check failed for table"):
        NativeLake(str(tmp_path), validate=True).scan(["x", "y"]).select("symbol").collect()


# --------------------------------------------------------------------- literals


def test_date_bounds_compare_as_midnight_utc(lake):
    result = lake.read(["f1"], start=dt.date(2024, 1, 2), end=dt.date(2024, 1, 4))
    assert_frame_equal(by_columns(result, result.columns), reference_read(lake, latest(lake, ["f1"]), ts(2), ts(4)))
    assert result["timestamp"].min() == ts(2) and result["timestamp"].max() == ts(4)


def test_literals_in_another_time_zone_are_unsupported(lake):
    start = dt.datetime(2024, 1, 2, tzinfo=ZoneInfo("America/New_York"))
    for read in (lambda: NativeLake(lake.base_path).read(["f1"], start=start), lambda: lake.read(["f1"], start=start)):
        with pytest.raises(UnsupportedError, match="convert the value"):
            read()


# --------------------------------------------------------------------- runtime-computed


RC_EXPRESSIONS = [
    (pl.col("f1") + pl.col("f2")).alias("s"),
    pl.col("s").shift(1).over("symbol").alias("lag_s"),
    pl.col("f2").rolling_mean(2).over("symbol").alias("roll"),
]


@pytest.fixture
def rc_fs(tmp_path) -> GammaFeatureLake:
    """Two versions of f1, a target and runtime-computed features, one reading another."""
    fs = GammaFeatureLake(base_path=str(tmp_path), enable_runtime_computed_features=True).initialize()
    fs.add_features(frame(range(1, 7), SYMBOLS, f1=1, f2=2))
    fs.add_targets(frame(range(1, 7), SYMBOLS, t1=-1))
    fs.add_features(frame(range(4, 9), SYMBOLS, f1=10))
    fs.add_runtime_computed_features(RC_EXPRESSIONS)
    return fs


def rc_reference(fs: GammaFeatureLake, roots: pl.DataFrame, start=None, end=None) -> pl.DataFrame:
    """Runtime-computed features over a reference read of their roots, computed after the bounds apply."""
    result = reference_read(fs, roots, start, end)
    for expression in RC_EXPRESSIONS:
        result = result.with_columns(expression)
    return result


RC_READS = [
    (["s"], {}),
    (["roll", "f1"], {"targets": ["t1"], "start": ts(2), "end": ts(7)}),
    (["f2", "s"], {"start": ts(5)}),
]


@pytest.mark.parametrize("materialized", [True, False])
@pytest.mark.parametrize("features,kwargs", RC_READS)
def test_runtime_computed_reads_match_polars(rc_fs, features, kwargs, materialized):
    got = rc_fs.read(features, materialized=materialized, **kwargs)
    got = got if materialized else got.collect()
    bounds = {key: kwargs[key] for key in ("start", "end") if key in kwargs}
    expected = rc_reference(rc_fs, latest(rc_fs, ["f1", "f2", *kwargs.get("targets", [])]), **bounds)
    assert_frame_equal(got, expected.select(got.columns))
    assert got.columns[: len(features)] == features
    assert got.columns[-2:] == ["timestamp", "symbol"]


def test_runtime_computed_scans_filter_after_computing(rc_fs):
    columns = ["lag_s", "s", "f1", "f2", "timestamp", "symbol"]
    later = pl.col("timestamp") >= ts(3)
    expected = rc_reference(rc_fs, latest(rc_fs, ["f1", "f2"]))
    got = rc_fs.read(["lag_s"], materialized=False).filter(later).collect()
    assert got.columns == columns
    assert_frame_equal(got, expected.filter(later).select(columns))
    assert got.filter(pl.col("timestamp") == ts(3))["lag_s"].null_count() == 0
    one = rc_fs.read(["s"], materialized=False).filter(pl.col("symbol") == "B").collect()
    assert_frame_equal(one, expected.filter(pl.col("symbol") == "B").select(one.columns))


def test_runtime_computed_features_read_pinned_versions(rc_fs):
    metadata = rc_fs.feature_metadata_frame().collect()
    selected = metadata.filter(pl.col("feature_name").is_in(["s", "f1"]) & (pl.col("version") == 0))
    expected = rc_reference(rc_fs, pl.concat([selected.filter(pl.col("feature_name") == "f1"), latest(rc_fs, ["f2"])]))
    for materialized in (True, False):
        got = rc_fs.read(selected, materialized=materialized)
        got = got if materialized else got.collect()
        assert_frame_equal(got, expected.select(got.columns))
    assert got["s"].to_list()[:3] == pytest.approx([3.0, 3.2, 3.4])
    lake = rc_fs._native_lake()
    versions = lake.read_versions({"s": 0, "f1": 0})
    assert_frame_equal(versions, got, check_column_order=False)
    assert lake.read_versions_schema({"s": 0, "f1": 0}) == versions.schema
    assert lake.scan_versions({"s": 0, "f1": 0}).collect().equals(versions)
    assert lake.read_selected_schema(selected) == lake.read_selected(selected).schema


def test_runtime_computed_features_read_their_roots_transitively(rc_fs):
    lake = rc_fs._native_lake()
    got = lake.read(["lag_s"])
    assert got.columns == ["lag_s", "s", "f1", "f2", "timestamp", "symbol"]
    assert lake.read_schema(["lag_s"]) == got.schema
    assert lake.scan(["lag_s"]).collect().equals(got)
    expected = got.with_columns(pl.col("s").shift(1).over("symbol").alias("lag_s"))
    assert_frame_equal(got, expected)
    arrow = pa.table(lake.read_arrow(["lag_s"]))
    assert pl.from_arrow(arrow).equals(got)


def test_runtime_computed_reads_accept_targets_registered_at_any_version(rc_fs):
    rc_fs.add_features(frame(range(1, 3), SYMBOLS, t1=5))
    got = rc_fs.read(["s"], ["t1"])
    assert_frame_equal(got, rc_reference(rc_fs, latest(rc_fs, ["f1", "f2", "t1"])).select(got.columns))
    assert got.filter(pl.col("timestamp") == ts(1))["t1"].to_list() == [5.0, 5.1, 5.2]
    with pytest.raises(MissingOrMisregisteredSignalsException):
        rc_fs.read(["s"], ["f2"])


def test_runtime_computed_features_need_the_trust_gate(rc_fs):
    untrusted = NativeLake(rc_fs.base_path)
    with pytest.raises(ValueError, match="disabled"):
        untrusted.read(["s"])
    with pytest.raises(ValueError, match="disabled"):
        untrusted.scan(["f1", "s"])
    with pytest.raises(ValueError, match="disabled"):
        untrusted.add_runtime_computed_features([pl.col("f1").alias("g")])
    assert untrusted.read(["f1"]).columns == ["f1", "timestamp", "symbol"]
    with pytest.raises(ValueError, match="disabled"):
        rc_fs.model_copy(update={"enable_runtime_computed_features": False}).read(["s"])


def test_registration_is_checked_natively(rc_fs):
    native_fs = rc_fs
    before = native_fs.feature_metadata_frame().collect()
    for exprs in (
        [(pl.col("f1") * 2).alias("s")],
        [pl.col("missing").alias("g")],
        [(pl.col("f1") + 1).alias("f1")],
        [pl.col("h").alias("g"), pl.col("g").alias("h")],
    ):
        with pytest.raises(ValueError):
            native_fs.add_runtime_computed_features(exprs)
    assert native_fs.feature_metadata_frame().collect().equals(before)
    native_fs.add_runtime_computed_features([pl.col("timestamp").dt.day().alias("day")], owner="me")
    row = native_fs.feature_metadata_frame().collect().filter(pl.col("feature_name") == "day")
    assert row.select("version", "table_addr", "owner", "signal_type").row(0) == (0, "n/a", "me", "runtime_computed")
    assert native_fs.read(["day"])["day"].to_list()[:4] == [1, 1, 1, 2]


def test_registrations_written_by_polars_read_and_extend(tmp_path):
    old = {
        "feature_name": pl.String,
        "version": pl.Int64,
        "table_addr": pl.String,
        "owner": pl.String,
        "signal_type": pl.String,
        "feature_params": pl.String,
    }
    pl.DataFrame(schema=old).write_delta(str(tmp_path / "feature_metadata"))
    fs = GammaFeatureLake(base_path=str(tmp_path), enable_runtime_computed_features=True).initialize()
    fs.add_features(frame(range(1, 4), SYMBOLS, f1=1))
    legacy = pl.DataFrame(
        {"feature_name": ["a"], "version": [0], "table_addr": ["n/a"], "owner": ["x"], "signal_type": ["runtime_computed"]},
        schema_overrides={"version": pl.Int64},
    ).with_columns(feature_params=pl.lit((pl.col("f1") + 1).alias("a").meta.serialize(format="json")))
    legacy.write_delta(str(tmp_path / "feature_metadata"), mode="append")
    assert fs.read(["a"])["a"].to_list()[:2] == [2.0, 2.1]
    fs.add_runtime_computed_features([(pl.col("a") * 2).alias("b")])
    got = fs.read(["b"])
    assert got.columns == ["b", "a", "f1", "timestamp", "symbol"]
    assert got["b"].to_list()[:2] == [4.0, 4.2]


# --------------------------------------------------------------------- parity


READS = [
    (["f1"], {}),
    (["f3", "f1", "f4"], {"targets": ["t1"]}),
    (["f2", "f3"], {"start": ts(3), "end": ts(5)}),
    (["f4", "f1"], {"start": ts(6)}),
    (["f3"], {"end": dt.date(2024, 1, 4)}),
    (["f1"], {"start": dt.datetime(2024, 1, 3, 1, tzinfo=dt.timezone(dt.timedelta(hours=2)))}),
    (["f1", "f2", "f3", "f4"], {"targets": ["t1"], "start": ts(2), "end": ts(8)}),
]


@pytest.mark.parametrize("materialized", [True, False])
@pytest.mark.parametrize("features,kwargs", READS)
def test_reads_match_polars(lake, features, kwargs, materialized):
    bounds = {key: kwargs[key] for key in ("start", "end") if key in kwargs}
    expected = reference_read(lake, latest(lake, [*features, *kwargs.get("targets", [])]), **bounds)
    result = lake.read(features, **kwargs, materialized=materialized)
    result = result if materialized else result.collect()
    assert set(result.columns) == set(expected.columns)
    assert_frame_equal(by_columns(result, expected.columns), expected)


SCANS = [
    lambda f: f.filter(pl.col("symbol") == "B"),
    lambda f: f.filter(pl.col("symbol").is_in(["A", "D"])).select("f4", "symbol"),
    lambda f: f.filter(pl.col("f3") > 6.0),
    lambda f: f.filter((pl.col("symbol") == "C") & (pl.col("timestamp") >= ts(5))),
    lambda f: f.select(pl.len()),
    lambda f: f.select("timestamp", "symbol"),
    lambda f: f.sort("timestamp", "symbol").head(4),
]


@pytest.mark.parametrize("query", SCANS)
def test_scan_queries_match_polars(lake, query):
    features = ["f1", "f3", "f4"]
    expected = query(reference_read(lake, latest(lake, features)).lazy()).collect()
    result = query(lake.read(features, materialized=False)).collect()
    if "symbol" in expected.columns and "timestamp" in expected.columns:
        expected, result = expected.sort("timestamp", "symbol"), result.sort("timestamp", "symbol")
    assert_frame_equal(result.select(expected.columns), expected)


class Spy:
    """Records the predicate of every read of a native lake."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def __getattr__(self, name):
        method = getattr(self._inner, name)
        if name not in ("read_selected", "index", "feature_metadata", "table_metadata"):
            return method

        def call(*args):
            self.calls.append((name, args))
            return method(*args)

        return call


def spied(lake: NativeLake) -> Spy:
    lake._lake = Spy(lake._lake)
    return lake._lake


def pushed(spy: Spy) -> list[list[tuple[str, str]]]:
    """The ``(column, op)`` terms of the only read's predicate, sorted within each clause."""
    ((_, args),) = spy.calls
    return [sorted((c, op) for c, op, _ in clause) for clause in args[-1]]


EQ, NE, LT, LE, GT, GE, IN, NOT_IN, IS_NULL = "==", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_null"
PUSHDOWNS = [
    ((pl.col("symbol") == "B") & (pl.col("f3") > 6.0), [[("symbol", EQ)]]),
    (pl.col("timestamp").is_between(ts(3), ts(5)), [[("timestamp", GE), ("timestamp", LE)]]),
    (pl.col("timestamp").is_between(ts(3), ts(5), closed="none"), [[("timestamp", GT), ("timestamp", LT)]]),
    ((pl.col("timestamp") > ts(2)) & pl.col("symbol").is_in(["A", "C"]), [[("timestamp", GT), ("symbol", IN)]]),
    ((pl.col("symbol") == "A") | (pl.col("timestamp") >= ts(8)), [[("symbol", EQ)], [("timestamp", GE)]]),
    (pl.lit(ts(4)) >= pl.col("timestamp"), [[("timestamp", LE)]]),
    (~pl.col("symbol").is_in(["A"]), [[("symbol", NOT_IN)]]),
    (~pl.col("timestamp").is_between(ts(3), ts(5)), [[("timestamp", LT)], [("timestamp", GT)]]),
    (~((pl.col("symbol") == "A") | (pl.col("f3") > 6.0)), [[("symbol", NE)]]),
    (pl.col("symbol").is_null(), [[("symbol", IS_NULL)]]),
    (pl.col("symbol").ne_missing("A"), [[("symbol", NE)], [("symbol", IS_NULL)]]),
    # Pushing a part of these would drop rows they select.
    (~((pl.col("symbol") == "A") & (pl.col("f3") > 6.0)), []),
    ((pl.col("symbol") == "A") | (pl.col("f3") > 6.0), []),
    (pl.col("timestamp").dt.day() == 3, []),
]


@pytest.mark.parametrize("predicate,expected", PUSHDOWNS)
def test_scan_pushes_sort_key_predicates(lake_path, predicate, expected):
    lake = NativeLake(lake_path)
    reference = lake.read(["f1", "f3"]).filter(predicate).select("f3", "timestamp", "symbol")
    spy = spied(lake)
    out = lake.scan(["f1", "f3"]).filter(predicate).select("f3", "timestamp", "symbol").collect()
    assert_frame_equal(out.sort("timestamp", "symbol"), reference.sort("timestamp", "symbol"))
    assert [row[0] for row in spy.calls[0][1][0]] == ["f3"]
    assert pushed(spy) == [sorted(clause) for clause in expected]


def test_index_and_metadata_frames_push_predicates(lake_path):
    fs = GammaFeatureLake(base_path=lake_path).initialize()
    native_lake = fs._native_lake()
    window = pl.col("timestamp").is_between(ts(3), ts(5)) & (pl.col("symbol") != "B")
    spy = spied(native_lake)
    index = fs.index_frame().filter(window).collect()
    assert pushed(spy) == [[("symbol", NE), ("timestamp", LE), ("timestamp", GE)]]
    assert_frame_equal(index.sort("timestamp", "symbol"), fs._native_lake().index().filter(window).sort("timestamp", "symbol"))
    spy.calls.clear()
    names = fs.feature_metadata_frame().filter(pl.col("feature_name").is_in(["f1", "t1"])).collect()["feature_name"]
    assert pushed(spy) == [[("feature_name", IN)]]
    assert sorted(names) == ["f1", "t1"]
    addr = native_lake.table_metadata()["table_addr"][0]
    spy.calls.clear()
    tables = fs.table_metadata_frame().filter(pl.col("table_addr") == addr).collect()
    assert pushed(spy) == [[("table_addr", EQ)]]
    assert tables["table_addr"].to_list() == [addr]


PUSHDOWN_SCHEMA = pl.Schema({"x": pl.Int64, "i": pl.Int32, "f": pl.Float64})


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (pl.col("i").is_in([1, 2]), [[("i", IN)]]),
        (pl.col("x").cast(pl.Int8, wrap_numerical=True) == 1, []),
        (pl.col("x").cast(pl.UInt64, strict=False).is_null(), []),
        (~pl.col("x").is_in([1], nulls_equal=True), []),
        (~pl.col("x").is_in([1]), [[("x", NOT_IN)]]),
        (pl.col("f") >= 1.0, []),
    ],
)
def test_pushdown_keeps_only_parts_that_select_a_superset(predicate, expected):
    def source(with_columns, pushed_predicate, n_rows, batch_size):
        dnf = gammalake.native._dnf(gammalake.native._pushdown(pushed_predicate, PUSHDOWN_SCHEMA, list(PUSHDOWN_SCHEMA)))
        seen.append([sorted((c, op) for c, op, _ in clause) for clause in dnf])
        yield pl.DataFrame(schema=PUSHDOWN_SCHEMA)

    seen = []
    gammalake.native.register_io_source_with_is_pure(source, schema=PUSHDOWN_SCHEMA).filter(predicate).collect()
    assert seen == [expected]


def test_scan_slices_before_filtering(lake_path):
    lake = NativeLake(lake_path)
    predicate = pl.col("symbol") == "B"
    expected = lake.read(["f1"]).head(4).filter(predicate)
    assert_frame_equal(lake.scan(["f1"]).head(4).filter(predicate).collect(), expected)
    metadata = lake.feature_metadata()
    other = pl.col("feature_name") == metadata["feature_name"][1]
    assert_frame_equal(lake.scan_feature_metadata().head(1).filter(other).collect(), metadata.head(1).filter(other))


def test_scans_take_row_aligned_predicates(lake_path):
    lake = NativeLake(lake_path)
    metadata = lake.feature_metadata()
    first = metadata["feature_name"][0]
    for predicate in [
        pl.col("feature_name").is_in(pl.Series([[name] for name in metadata["feature_name"]])),
        (pl.col("feature_name") == first) & pl.col("feature_name").is_in(pl.Series([[first]] * metadata.height)),
        (pl.col("feature_name") != first) & pl.col("feature_name").is_in(pl.Series([[name] for name in metadata["feature_name"]])),
    ]:
        assert_frame_equal(lake.scan_feature_metadata().filter(predicate).collect(), metadata.filter(metadata.select(predicate).to_series()))
    index = lake.index()
    predicate = (pl.col("symbol") != "A") & pl.col("symbol").is_in(pl.Series([[symbol] for symbol in index["symbol"]]))
    assert_frame_equal(lake.scan_index().filter(predicate).collect(), index.filter(index.select(predicate).to_series()))


JOIN_PARAMS = [
    {},
    {"strategy": "forward"},
    {"strategy": "nearest"},
    {"strategy": "backward", "tolerance": "2d"},
    {"strategy": "forward", "tolerance": "1d"},
    {"strategy": "nearest", "tolerance": "36h"},
]


def rows_frame(rows, name: str, scale: float) -> pl.DataFrame:
    data = {"timestamp": [ts(d) for d, _ in rows], "symbol": [s for _, s in rows], name: [scale * d + ord(s) for d, s in rows]}
    return pl.DataFrame(data).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))


def joined_writes(seed: int):
    """Dense, as-of and sparse writes in a random order; no write overlaps an older one of the same feature."""
    rng = random.Random(seed)
    keys = list(itertools.product(range(1, 12), SYMBOLS))
    writes = [("add_features", frame(range(1, 6), SYMBOLS, d1=1), {})]
    for i, params in enumerate(JOIN_PARAMS):
        writes.append(("add_as_of_features", rows_frame(rng.sample(keys, rng.randint(1, 8)), f"a{i}", i + 1), {"params": params}))
    writes.append(("add_sparse_features", rows_frame(rng.sample(keys, 6), "s0", 10), {}))
    rng.shuffle(writes)
    writes.append(("add_features", frame(range(6, 12), SYMBOLS, d1=1), {}))
    writes.append(("add_as_of_features", rows_frame(list(itertools.product([12, 13], SYMBOLS)), "a0", 1), {"params": {}}))
    writes.append(("add_sparse_features", rows_frame([(12, "B")], "s0", 10), {}))
    return writes


JOINED_NAMES = ["d1", "s0", *(f"a{i}" for i in range(len(JOIN_PARAMS)))]
JOINED_QUERIES = [
    lambda f: f,
    lambda f: f.filter(pl.col("symbol") == "B"),
    lambda f: f.filter(pl.col("timestamp") >= ts(5)),
    lambda f: f.filter(pl.col("symbol").is_in(["A", "C"]) & (pl.col("timestamp") < ts(7))),
]


@pytest.mark.parametrize("seed", range(4))
def test_as_of_and_sparse_match_polars(tmp_path, seed):
    lake = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    for method, df, kwargs in joined_writes(seed):
        getattr(lake, method)(df, **kwargs)
    selected = latest(lake, JOINED_NAMES)
    assert selected["signal_type"].to_list() == ["feature", "sparse_feature", *["as_of_feature"] * len(JOIN_PARAMS)]
    for window in [{}, {"start": ts(3)}, {"end": ts(6)}, {"start": ts(4), "end": ts(9)}]:
        expected = reference_read(lake, selected, **window)
        assert_frame_equal(lake.read(JOINED_NAMES, **window).sort(KEYS).select(expected.columns), expected)
        for query in JOINED_QUERIES:
            result = query(lake.read(JOINED_NAMES, **window, materialized=False)).collect().sort(KEYS)
            assert_frame_equal(result.select(expected.columns), query(expected.lazy()).collect())


@pytest.mark.parametrize("seed", range(2))
def test_memory_limit_reads_the_same(tmp_path, seed):
    lake = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    for method, df, kwargs in joined_writes(seed):
        getattr(lake, method)(df, **kwargs)
    expected = lake.read(JOINED_NAMES).sort(KEYS)
    bounded = GammaFeatureLake(base_path=str(tmp_path), memory_limit=1 << 20).initialize()
    assert_frame_equal(bounded.read(JOINED_NAMES).sort(KEYS), expected)
    scan = NativeLake(str(tmp_path), batch_size=3, memory_limit=1 << 20).scan(JOINED_NAMES)
    assert_frame_equal(scan.collect().sort(KEYS).select(expected.columns), expected)


def test_read_frame_pins_versions(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_features(frame(range(1, 5), SYMBOLS, f1=1, f2=2))
    fs.add_as_of_features(rows_frame([(2, "A"), (3, "B")], "a", 1), params={"strategy": "backward"})
    fs.add_features(frame(range(3, 7), SYMBOLS, f1=10), overlap_mode="copy")
    fs.add_as_of_features(rows_frame([(5, "A")], "a", 7), params={"strategy": "forward"})
    metadata = fs.feature_metadata_frame().collect()
    old = metadata.filter((pl.col("version") == 0) & pl.col("feature_name").is_in(["f1", "a", "f2"]))
    assert old.height == 3
    result = fs.read(old, ts(2), ts(5))
    expected = reference_read(fs, old, ts(2), ts(5))
    assert result.columns == expected.columns
    assert_frame_equal(result.sort(KEYS), expected)
    assert_frame_equal(fs.read(old, materialized=False).collect().sort(KEYS), reference_read(fs, old))
    pinned = NativeLake(str(tmp_path)).read_versions({"a": 0, "f1": 0, "f2": 0}, ts(2), ts(5))
    assert_frame_equal(pinned.sort(KEYS), expected.select(pinned.columns))
    with pytest.raises(MissingFeaturesException, match="f2 version 1"):
        NativeLake(str(tmp_path)).read_versions({"f2": 1})
    with pytest.raises(ValueError, match="multiple versions"):
        fs.read(metadata.filter(pl.col("feature_name") == "f1"))


def test_lazy_projection_keeps_how_a_table_is_read(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_features(frame(range(1, 6), SYMBOLS, d=1))
    fs.add_as_of_features(rows_frame([(1, "A"), (3, "B")], "a", 1).with_columns(b=pl.col("a") * 2), params={"strategy": "backward"})
    fs.add_features(rows_frame([(2, "A")], "b", 5), overlap_mode="merge")
    metadata = fs.feature_metadata_frame().collect()
    rows = pl.concat([metadata.filter(pl.col("feature_name") == "a"), metadata.filter((pl.col("feature_name") == "b") & (pl.col("version") == 1))])
    assert rows["table_addr"].n_unique() == 1 and rows["signal_type"].to_list() == ["as_of_feature", "feature"]
    keys = ["timestamp", "symbol"]
    native = NativeLake(str(tmp_path))
    eager = native.read_selected(rows).select("b", *keys).sort(keys)
    assert not eager.equals(native.read_selected(rows[1:]).select("b", *keys).sort(keys))
    assert_frame_equal(native.scan_selected(rows).select("b", *keys).collect().sort(keys), eager)
    assert_frame_equal(fs.read(rows, materialized=False).select("b", *keys).collect().sort(keys), eager)
    pinned = {"a": 0, "b": 1}
    assert_frame_equal(native.scan_versions(pinned).select("b", *keys).collect().sort(keys), eager)
    assert_frame_equal(native.scan(["a", "b"]).select("b", *keys).collect().sort(keys), native.read(["a", "b"]).select("b", *keys).sort(keys))


def test_an_empty_selection_reads_the_index(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_features(frame(range(1, 4), SYMBOLS, d=1))
    empty = fs.feature_metadata_frame().collect().head(0)
    keys = ["timestamp", "symbol"]
    expected = pl.read_delta(fs.index).select(keys).sort(keys)
    assert_frame_equal(fs.read(empty).sort(keys), expected)
    assert_frame_equal(fs.read(empty, materialized=False).collect().sort(keys), expected)


def test_as_of_on_a_string_key_is_unsupported(tmp_path):
    lake = NativeLake.create(str(tmp_path), pa.schema([("x", pa.string())]), primary_sort_key="x")
    with pytest.raises(UnsupportedError, match="integer or temporal primary sort key"):
        lake.add_as_of_features(pl.DataFrame({"x": ["1", "2"], "a": [1, 2]}), params={})
    assert lake.index().is_empty() and lake.latest_features().is_empty()


def test_a_float_sort_key_is_rejected(tmp_path):
    schema = pa.schema([("t", pa.int64()), ("x", pa.float64())])
    with pytest.raises(ValueError, match="invalid argument"):
        GammaFeatureLake(base_path=str(tmp_path / "lake"), primary_sort_key="t").initialize(ArrowSchema.make(schema))
    with pytest.raises(ValueError, match="invalid argument"):
        NativeLake.create(str(tmp_path / "native"), schema, primary_sort_key="t")
    assert not (tmp_path / "lake" / "index").exists() and not (tmp_path / "native" / "index").exists()


def test_a_table_once_as_of_is_never_padded(tmp_path):
    fs = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    fs.add_as_of_features(rows_frame([(1, "A"), (3, "B")], "x", 1), params={"strategy": "backward"})
    fs.add_features(rows_frame([(2, "A"), (3, "B")], "x", 2), overlap_mode="merge")
    fs.add_features(frame(range(4, 7), SYMBOLS, d=1))
    metadata = fs.feature_metadata_frame().collect()
    assert metadata.filter(pl.col("feature_name") == "x")["table_addr"].n_unique() == 1
    version_0 = metadata.filter((pl.col("feature_name") == "x") & (pl.col("version") == 0))
    assert pl.read_delta(fs.get_path(version_0["table_addr"].item())).height == 3
    assert_frame_equal(fs.read(version_0).sort(KEYS), reference_read(fs, version_0))
    assert_frame_equal(fs.read(["x", "d"]).sort(KEYS), reference_read(fs, latest(fs, ["x", "d"])))


# --------------------------------------------------------------------- engine


def test_native_handle_follows_lake_location(lake_path, tmp_path):
    fs = GammaFeatureLake(base_path=lake_path).initialize()
    first = fs._native_lake()
    assert fs._native_lake() is first
    other = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    other.add_features(frame([1, 2], ["A"], g1=1))
    moved = fs.model_copy(update={"base_path": str(tmp_path)})
    assert moved._native_lake() is not first
    assert moved.read(["g1"]).height == 2


@pytest.mark.parametrize("update", [{"base_path": "s3://bucket/lake%41"}, {"index_path": "../index"}, {"feature_metadata_path": "/abs"}])
def test_unaddressable_lakes_are_rejected(update, lake):
    with pytest.raises(ValueError, match="not supported|must be a relative path inside the lake"):
        lake.model_copy(update=update).read(["f1"])


def test_delta_options_reject_unknown_fields():
    with pytest.raises(ValidationError, match="compresion"):
        DeltaOptions(compresion="gzip")


@pytest.mark.parametrize("bad", [{"engine": "python"}, {"compression": "nope"}])
def test_validation_errors_hide_storage_options(tmp_path, bad):
    secret = {"AWS_SECRET_ACCESS_KEY": "hunter2"}
    with pytest.raises(ValueError) as error:
        GammaFeatureLake(base_path=str(tmp_path), delta_options=DeltaOptions(storage_options=secret), **bad)
    assert "hunter2" not in str(error.value)
