import ctypes
import datetime as dt
import gc
import itertools

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from gammalake import GammaFeatureLake
from gammalake.abstract import MissingFeaturesException

datafusion = pytest.importorskip("datafusion")

UTC = dt.UTC
ORDER = "ORDER BY timestamp NULLS FIRST, symbol NULLS FIRST"


def frame(days, symbols, **features) -> pl.DataFrame:
    rows = list(itertools.product(days, symbols))
    data = {"timestamp": [dt.datetime(2024, 1, d, tzinfo=UTC) for d, _ in rows], "symbol": [s for _, s in rows]}
    for name, scale in features.items():
        data[name] = [float(scale * d + (ord(s) - ord("A")) / 10) for d, s in rows]
    return pl.DataFrame(data).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))


@pytest.fixture(scope="module")
def lake(tmp_path_factory) -> GammaFeatureLake:
    fs = GammaFeatureLake(base_path=str(tmp_path_factory.mktemp("datafusion_lake"))).initialize()
    fs.add_features(frame(range(1, 7), "ABC", f1=1, f2=2))
    fs.add_targets(frame(range(1, 7), "ABC", t1=-1))
    fs.add_features(frame(range(4, 10), "ABC", f3=3))
    return fs


def query(lake: GammaFeatureLake, sql: str, features=("f1", "f3"), targets=("t1",)) -> pl.DataFrame:
    ctx = datafusion.SessionContext()
    ctx.register_table("t", lake.feature_table(list(features), list(targets)))
    return ctx.sql(sql).to_polars()


def test_select_star_equals_read(lake):
    table = lake.feature_table(["f1", "f3"], ["t1"])
    assert table.features == ["f1", "f3", "t1"]
    expected = lake.read(["f1", "f3"], ["t1"])
    assert pl.from_arrow(pa.schema(table.schema).empty_table()).schema == expected.schema
    assert_frame_equal(query(lake, f"SELECT * FROM t {ORDER}"), expected)


def explain(lake: GammaFeatureLake, sql: str) -> str:
    return query(lake, f"EXPLAIN {sql}")["plan"].str.join("\n").item()


def test_sort_key_filters_are_pushed_down(lake):
    sql = "SELECT * FROM t WHERE timestamp >= '2024-01-05T00:00:00Z' AND symbol IN ('A', 'C')"
    assert "partial_filters=[t.timestamp >= " in explain(lake, sql)
    expected = lake.read(["f1", "f3"], ["t1"]).filter(pl.col("timestamp") >= dt.datetime(2024, 1, 5, tzinfo=UTC), pl.col("symbol").is_in(["A", "C"]))
    assert_frame_equal(query(lake, f"{sql} {ORDER}"), expected)


def test_python_udf_filters_run_in_datafusion(lake):
    def big(values: pa.Array) -> pa.Array:
        return pa.array([v is not None and v > 5 for v in values.to_pylist()])

    ctx = datafusion.SessionContext()
    ctx.register_udf(datafusion.udf(big, [pa.float64()], pa.bool_(), "stable", name="big"))
    ctx.register_table("t", lake.feature_table(["f1", "f3"]))
    assert "partial_filters" not in ctx.sql("EXPLAIN SELECT * FROM t WHERE big(f3)").to_polars()["plan"].str.join("\n").item()
    result = ctx.sql(f"SELECT * FROM t WHERE big(f3) {ORDER}").to_polars()
    assert_frame_equal(result, lake.read(["f1", "f3"]).filter(pl.col("f3") > 5))


def test_aggregates_and_projection(lake):
    result = query(lake, "SELECT symbol, count(*) AS n, count(f3) AS n3 FROM t GROUP BY symbol ORDER BY symbol")
    expected = lake.read(["f1", "f3"]).group_by("symbol").agg(pl.len().alias("n"), pl.col("f3").count().alias("n3")).sort("symbol")
    assert_frame_equal(result, expected, check_dtypes=False)


def test_missing_features_raise(lake):
    with pytest.raises(MissingFeaturesException):
        lake.feature_table(["nope"])


def test_table_outlives_the_lake(lake):
    ctx = datafusion.SessionContext()
    ctx.register_table("t", GammaFeatureLake(base_path=lake.base_path).initialize().feature_table(["f1"]))
    gc.collect()
    assert ctx.sql("SELECT count(*) AS n FROM t").to_polars()["n"].item() == len(lake.read(["f1"]))


def test_session_codec_capsule_is_checked(lake):
    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    name = ctypes.create_string_buffer(b"something_else")

    class Session:
        def __datafusion_logical_extension_codec__(self):
            return new_capsule(ctypes.addressof(name), name, None)

    with pytest.raises(ValueError, match="datafusion_logical_extension_codec"):
        lake.feature_table(["f1"]).__datafusion_table_provider__(Session())
