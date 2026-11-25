import gc

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from gammalake import GammaFeatureLake
from gammalake.abstract import MissingFeaturesException
from gammalake.tests._model import frame, keys, ts

datafusion = pytest.importorskip("datafusion")

ORDER = "ORDER BY timestamp NULLS FIRST, symbol NULLS FIRST"


@pytest.fixture(scope="module")
def lake(tmp_path_factory) -> GammaFeatureLake:
    fs = GammaFeatureLake(base_path=str(tmp_path_factory.mktemp("datafusion_lake"))).initialize()
    fs.add_features(frame(keys(range(1, 7), "ABC"), f1=1, f2=2))
    fs.add_targets(frame(keys(range(1, 7), "ABC"), t1=-1))
    fs.add_features(frame(keys(range(4, 10), "ABC"), f3=3))
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


def test_sort_key_filters_are_pushed_down(lake):
    sql = "SELECT * FROM t WHERE timestamp >= '2024-01-05T00:00:00Z' AND symbol IN ('A', 'C')"
    assert "partial_filters=[t.timestamp >= " in query(lake, f"EXPLAIN {sql}")["plan"].str.join("\n").item()
    expected = lake.read(["f1", "f3"], ["t1"]).filter(pl.col("timestamp") >= ts(5), pl.col("symbol").is_in(["A", "C"]))
    assert_frame_equal(query(lake, f"{sql} {ORDER}"), expected)


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
    class Session:
        def __datafusion_logical_extension_codec__(self):
            return pa.schema([]).__arrow_c_schema__()

    with pytest.raises(ValueError, match="datafusion_logical_extension_codec"):
        lake.feature_table(["f1"]).__datafusion_table_provider__(Session())
