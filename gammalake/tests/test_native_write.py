"""Writes through the native engine, checked against a reference model."""

import os
import subprocess
import sys
import textwrap

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from gammalake import GammaFeatureLake, UnsupportedError
from gammalake.native import NativeLake
from gammalake.tests._model import KEYS, SCENARIOS, Model, frame, keys


def lake_at(path) -> GammaFeatureLake:
    return GammaFeatureLake(base_path=str(path)).initialize()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_writes_match_the_model(tmp_path, scenario):
    lake, model = lake_at(tmp_path), Model()
    for rows, mode, scales in SCENARIOS[scenario]:
        lake.add_features(frame(rows, **scales), overlap_mode=mode)
        model.add(rows, **scales)
    names = sorted(model.values)
    assert_frame_equal(lake.read(names).select(*names, *KEYS).sort(KEYS), model.read(names), check_column_order=False)


def test_metadata_of_mixed_writes(tmp_path):
    lake = lake_at(tmp_path)
    model = Model()
    lake.add_features(frame(keys(range(1, 4), "ab"), f=1, g=2), owner="alice")
    model.add(keys(range(1, 4), "ab"), f=1, g=2)
    lake.add_targets(frame(keys(range(1, 4), "ab"), y=1), owner="bob")
    model.add(keys(range(1, 4), "ab"), y=1)
    lake.add_features(frame(keys(range(2, 5), "ab"), f=3), owner="carol", overlap_mode="merge")
    model.add(keys(range(2, 5), "ab"), f=3)
    lake.add_index_rows(frame(keys([6], "abc")))
    model.add(keys([6], "abc"))

    columns = ["feature_name", "version", "owner", "signal_type", "feature_params"]
    assert lake.feature_metadata_frame().collect().select(columns).sort("feature_name", "version").rows() == [
        ("f", 0, "alice", "feature", "{}"),
        ("f", 1, "carol", "feature", "{}"),
        ("g", 0, "alice", "feature", "{}"),
        ("y", 0, "bob", "target", "{}"),
    ]
    assert_frame_equal(lake.index_frame().collect().sort(KEYS), model.read([]).select(KEYS))
    assert_frame_equal(lake.read(["f", "g"], targets=["y"]).select("f", "g", "y", *KEYS).sort(KEYS), model.read(["f", "g", "y"]))


@pytest.mark.parametrize(
    "column",
    [
        pa.array([[1, 2], [3], None], pa.list_(pa.int64())),
        pa.array([[1.5], [], [None]], pa.large_list(pa.float64())),
        pa.array([{"x": 1, "y": "a"}, None, {"x": 3, "y": None}], pa.struct([("x", pa.int32()), ("y", pa.string())])),
        pa.array([[{"x": [1]}], None, []], pa.list_(pa.struct([("x", pa.large_list(pa.int8()))]))),
    ],
    ids=["list", "large_list", "struct", "list_of_struct"],
)
def test_nested_columns_round_trip(tmp_path, column):
    lake = lake_at(tmp_path)
    written = frame(keys(range(1, 4), "a")).with_columns(n=pl.from_arrow(column))
    lake.add_features(written)
    lake.add_features(frame(keys(range(1, 5), "a"), f=1))
    got = lake.read(["n", "f"]).sort(KEYS)
    expected = pl.concat([written, frame(keys([4], "a"))], how="diagonal").with_columns(f=pl.col("timestamp").dt.day().cast(pl.Float64))
    assert_frame_equal(got, expected, check_column_order=False)


SCHEMA = pa.schema([("timestamp", pa.timestamp("us", tz="UTC")), ("symbol", pa.string())])


def test_create_add_and_read(tmp_path):
    lake = NativeLake.create(str(tmp_path), SCHEMA)
    assert lake.sort_keys == KEYS
    lake.add_features(frame(keys(range(1, 4), "ab"), f=1), owner="alice", feature_params={"window": 5})
    lake.add_targets(pa.table(frame(keys(range(1, 4), "ab"), y=2).to_arrow()))
    lake.add_index_rows(frame(keys([4], "ab")).with_columns(ignored=pl.lit(1)))
    got = lake.read(["f"], ["y"])
    assert got.height == 8
    assert got["f"].null_count() == got["y"].null_count() == 2
    latest = lake.latest_features().sort("feature_name")
    assert latest.select("feature_name", "owner", "signal_type", "feature_params").rows() == [
        ("f", "alice", "feature", '{"window": 5}'),
        ("y", "missing_owner", "target", "{}"),
    ]
    again = NativeLake.create(str(tmp_path), SCHEMA)
    assert again.index().height == 8


def test_write_errors(tmp_path):
    lake = NativeLake.create(str(tmp_path), SCHEMA)
    with pytest.raises(ValueError, match="no rows"):
        lake.add_features(frame([]))
    with pytest.raises(ValueError, match="overlap_mode"):
        lake.add_features(frame(keys([1], "a"), f=1), overlap_mode="replace")
    with pytest.raises(ValueError, match="sort key"):
        lake.add_features(pl.DataFrame({"f": [1.0]}))
    with pytest.raises(UnsupportedError, match="signal type"):
        lake._add(frame(keys([1], "a"), f=1), "runtime_computed", "me", "copy", None)
    with pytest.raises(UnsupportedError, match="(?i)duration"):
        lake.add_features(frame(keys([1], "a")).with_columns(d=pl.duration(days=1)))
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        lake.add_features([1, 2])
    with pytest.raises(ValueError, match="primary sort key"):
        NativeLake.create(str(tmp_path / "other"), pa.schema([("symbol", pa.string())]))

    class Consumed:
        def __arrow_c_schema__(self):
            capsule = SCHEMA.__arrow_c_schema__()
            pa.Schema._import_from_c_capsule(capsule)
            return capsule

    with pytest.raises(ValueError, match="already been consumed"):
        NativeLake.create(str(tmp_path / "consumed"), Consumed())
    assert not (tmp_path / "consumed").exists()
    assert lake.index().height == 0


def described(lake) -> dict[str, tuple]:
    rows = lake.describe(lake.feature_metadata_frame()).sort("version").iter_rows(named=True)
    return {row["feature_name"]: (row["comment"], row["tags"]) for row in rows}


def test_add_comments_and_tags(tmp_path):
    lake = lake_at(tmp_path)
    lake.add_features(frame(keys(range(1, 3), "a"), f=1, g=2), comments={"f": "first"}, tags={"f": ["x"], "g": ["y"]})
    assert described(lake) == {"f": ("first", ["x"]), "g": (None, ["y"])}

    lake.add_features(frame(keys(range(1, 3), "a"), f=3), comments={"f": "second"})
    assert described(lake)["f"] == ("second", [])

    lake.add_features(frame(keys(range(1, 3), "a"), g=4), overlap_mode="merge", tags={"g": []})
    assert described(lake)["g"] == (None, [])

    before = lake.feature_metadata_frame().collect()
    with pytest.raises(ValueError, match="not written features"):
        lake.add_features(frame(keys(range(3, 4), "a"), h=1), comments={"symbol": "no"})
    with pytest.raises(ValueError, match="not written features"):
        lake.add_features(frame(keys(range(3, 4), "a"), h=1), tags={"missing": ["no"]})
    assert_frame_equal(lake.feature_metadata_frame().collect(), before)


def test_appending_some_columns_keeps_annotations(tmp_path):
    lake = lake_at(tmp_path)
    lake.add_features(frame(keys(range(1, 3), "a"), f=1, g=2), comments={"f": "first"}, tags={"f": ["x"], "g": ["y"]})
    lake.add_features(frame(keys(range(3, 5), "a"), f=1), comments={"f": "second"})
    assert described(lake) == {"f": ("second", ["x"]), "g": (None, ["y"])}


def test_annotate_by_version(tmp_path):
    lake = lake_at(tmp_path)
    lake.add_features(frame(keys(range(1, 4), "ab"), f=1, g=2, h=3))
    lake.add_features(frame(keys(range(1, 4), "ab"), g=5))
    metadata = lake.feature_metadata_frame().collect().sort("feature_name", "version")
    lake.annotate(
        metadata.with_columns(
            comment=pl.Series(["f", "g1", "g2", None]),
            tags=pl.Series([["a"], None, ["b", "c"], []], dtype=pl.List(pl.String)),
        )
    )
    rows = lake.describe(lake.feature_metadata_frame()).sort("feature_name", "version")
    assert rows.select("feature_name", "version", "comment", "tags").rows() == [
        ("f", 0, "f", ["a"]),
        ("g", 0, "g1", []),
        ("g", 1, "g2", ["b", "c"]),
        ("h", 0, None, []),
    ]


def test_describe_gives_runtime_computed_features_no_annotations(tmp_path):
    lake = GammaFeatureLake(base_path=str(tmp_path), enable_runtime_computed_features=True).initialize()
    lake.add_features(frame(keys(range(1, 3), "a"), f=1), tags={"f": ["x"]})
    lake.add_runtime_computed_features([(pl.col("f") * 2).alias("rc")])
    assert described(lake) == {"f": (None, ["x"]), "rc": (None, [])}


def test_single_thread_matches_default(tmp_path):
    """Writes, merges and reads finish and agree with ``GAMMA_LAKE_THREADS=1``."""
    script = textwrap.dedent(
        """
        import sys
        from datetime import UTC, datetime, timedelta

        import polars as pl

        from gammalake import GammaFeatureLake

        lake = GammaFeatureLake(base_path=sys.argv[1]).initialize()
        days = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(6)]

        def frame(days, offset):
            rows = [(d, s) for d in days for s in "abc"]
            data = {"timestamp": [d for d, _ in rows], "symbol": [s for _, s in rows]}
            data |= {f"f{i}": [float(i + offset + d.day) for d, _ in rows] for i in range(40)}
            return pl.DataFrame(data).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        lake.add_features(frame(days[:3], 0))
        lake.add_features(frame(days[3:], 0))
        lake.add_features(frame(days[2:4], 1))
        lake.add_features(frame(days[4:], 2), overlap_mode="merge")
        names = [f"f{i}" for i in range(40)]
        full = lake.read(names)
        one = lake.read(names, materialized=False).filter(pl.col("symbol") == "b").collect()
        print(full.height, full.hash_rows().sum(), one.height, one.hash_rows().sum())
        """
    )
    outputs = []
    for threads in ("1", None):
        env = {k: v for k, v in os.environ.items() if k != "GAMMA_LAKE_THREADS"}
        if threads:
            env["GAMMA_LAKE_THREADS"] = threads
        path = tmp_path / f"threads-{threads}"
        run = subprocess.run([sys.executable, "-c", script, str(path)], check=False, env=env, capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr
        outputs.append(run.stdout)
    assert outputs[0] == outputs[1]


def test_feature_files_hold_one_sorted_row_group_up_to_a_million_rows(tmp_path):
    """Files hold rows in key order, in row groups as large as the Python ``deltalake`` writer's."""
    lake = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    lake.add_features(frame(keys(range(1, 8), [f"s{i:05}" for i in range(20_000)]), f=1))
    files = [f for f in tmp_path.glob("delta_*/*.parquet")]
    assert files
    assert [pq.ParquetFile(f).metadata.num_row_groups for f in files] == [1] * len(files)
    rows = pl.from_arrow(pq.read_table(files[0], columns=KEYS))
    assert rows.equals(rows.sort(KEYS))
