"""Writes through the native engine, checked against a reference model."""

import datetime as dt
import os
import stat
import subprocess
import sys
import textwrap

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from gammalake import GammaFeatureLake, PartialWriteError, UnsupportedError
from gammalake.native import NativeLake

UTC = dt.UTC
KEYS = ["timestamp", "symbol"]


def ts(day: int) -> dt.datetime:
    return dt.datetime(2024, 1, day, tzinfo=UTC)


def keys(days, symbols) -> list[tuple[int, str]]:
    return [(d, s) for d in days for s in symbols]


def frame(rows, **scales) -> pl.DataFrame:
    """One row per ``(day, symbol)`` key; feature ``name`` holds ``scale * day``."""
    data = {"timestamp": [ts(d) for d, _ in rows], "symbol": [s for _, s in rows]}
    for name, scale in scales.items():
        data[name] = [float(scale * d) for d, _ in rows]
    schema = {"timestamp": pl.Datetime("us", "UTC"), "symbol": pl.String, **dict.fromkeys(scales, pl.Float64)}
    return pl.DataFrame(data, schema=schema)


class Model:
    """Expected reads.

    An input holding only new keys appends them. Otherwise it replaces its features over
    its primary key window, so keys of the window missing from the input read as null.
    """

    def __init__(self):
        self.index: set[tuple[int, str]] = set()
        self.values: dict[str, dict[tuple[int, str], float | None]] = {}

    def add(self, rows, **scales) -> None:
        given = set(rows)
        overlaps = bool(given & self.index)
        self.index |= given
        low, high = min(d for d, _ in rows), max(d for d, _ in rows)
        for name, scale in scales.items():
            column = self.values.setdefault(name, {})
            for key in self.index if overlaps else given:
                if low <= key[0] <= high:
                    column[key] = float(scale * key[0]) if key in given else None

    def read(self, features) -> pl.DataFrame:
        index = sorted(self.index)
        data = {name: [self.values[name].get(k) for k in index] for name in features}
        data |= {"timestamp": [ts(d) for d, _ in index], "symbol": [s for _, s in index]}
        return pl.DataFrame(data, schema={**dict.fromkeys(features, pl.Float64), "timestamp": pl.Datetime("us", "UTC"), "symbol": pl.String})


# Each step is (rows, overlap_mode, scales).
SCENARIOS = {
    "append": [
        (keys(range(1, 4), "ab"), "copy", {"f": 1}),
        (keys(range(4, 6), "ab"), "copy", {"f": 1}),
    ],
    "new_feature_on_new_keys": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        (keys(range(2, 6), "ab"), "copy", {"g": 2}),
    ],
    # Point72/gamma-lake#45: appending only new keys skips padding the tail of f.
    "all_new_keys_after_coverage": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        (keys(range(1, 6), "a"), "copy", {"g": 2}),
        (keys([6], "a"), "copy", {"f": 1}),
    ],
    # New keys inside the coverage of g are padded into it.
    "all_new_keys_inside_coverage": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        (keys(range(1, 4), "a"), "copy", {"g": 2}),
        ([(2, "b"), (3, "b")], "copy", {"f": 1}),
    ],
    # Point72/gamma-lake#46: copy mode drops the rows of f after the window.
    "overlap_copy": [
        (keys(range(1, 7), "a"), "copy", {"f": 1}),
        (keys(range(1, 7), "ab"), "copy", {"g": 2}),
        (keys(range(2, 4), "a"), "copy", {"f": 10}),
    ],
    "overlap_merge": [
        (keys(range(1, 5), "ab"), "copy", {"f": 1, "h": 3}),
        ([(2, "a"), (3, "a"), (5, "a")], "merge", {"f": 10}),
    ],
    # Tables that a merge left untouched are still padded by the next write.
    "write_after_overlap_merge": [
        (keys(range(1, 5), "ab"), "copy", {"f": 1, "h": 3}),
        (keys(range(2, 4), "a"), "merge", {"f": 10}),
        (keys(range(5, 6), "ab"), "copy", {"g": 1}),
    ],
    # A new key whose primary key equals the coverage of f must be padded into f.
    "new_key_at_coverage_end": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        ([(3, "b")], "copy", {"g": 1}),
        (keys([4], "a"), "copy", {"f": 1}),
    ],
}


def run(lake: GammaFeatureLake, steps) -> Model:
    model = Model()
    for rows, mode, scales in steps:
        lake.add_features(frame(rows, **scales), overlap_mode=mode)
        model.add(rows, **scales)
        assert_physically_aligned(lake.base_path)
    return model


def assert_physically_aligned(path: str) -> None:
    """Each table holds exactly the index keys up to its coverage.

    Reads cannot check this: a table that ends early reads as trailing nulls.
    """
    index = pl.read_delta(f"{path}/index").select(KEYS).sort(KEYS)
    coverage = pl.read_delta(f"{path}/table_metadata").group_by("table_addr").agg(pl.col("last_updated").max())
    for addr, last in coverage.iter_rows():
        table = pl.read_delta(f"{path}/{addr}").select(KEYS).sort(KEYS)
        assert_frame_equal(table, index.filter(pl.col("timestamp") <= last))


def features(steps) -> list[str]:
    return sorted({name for *_, scales in steps for name in scales})


def check(lake: GammaFeatureLake, model: Model, names) -> None:
    got = lake.read(names).select(*names, *KEYS).sort(KEYS)
    assert_frame_equal(got, model.read(names), check_column_order=False)


def lake_at(path) -> GammaFeatureLake:
    return GammaFeatureLake(base_path=str(path)).initialize()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_writes_match_the_model(tmp_path, scenario):
    steps = SCENARIOS[scenario]
    lake = lake_at(tmp_path)
    model = run(lake, steps)
    check(lake, model, features(steps))


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


@pytest.mark.parametrize(
    "first,second,widened",
    [
        (pl.Int32, pl.Float64, pl.Float64),
        (pl.Int8, pl.Int32, pl.Int32),
        (pl.Int16, pl.Float32, pl.Float32),
        (pl.Int64, pl.Float32, pl.Float64),
    ],
)
def test_copy_mode_widens_types(tmp_path, first, second, widened):
    lake = lake_at(tmp_path)
    model = Model()
    lake.add_features(frame(keys(range(1, 4), "a"), f=1).with_columns(pl.col("f").cast(first)))
    model.add(keys(range(1, 4), "a"), f=1)
    lake.add_features(frame(keys(range(2, 4), "a"), f=2).with_columns(pl.col("f").cast(second)))
    model.add(keys(range(2, 4), "a"), f=2)
    got = lake.read(["f"]).sort(KEYS)
    assert got.schema["f"] == widened
    assert_frame_equal(got, model.read(["f"]).with_columns(pl.col("f").cast(widened)), check_column_order=False)


# ---------------------------------------------------------------- NativeLake


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


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_a_failed_commit_names_the_committed_tables(tmp_path):
    lake = NativeLake.create(str(tmp_path), SCHEMA)
    lake.add_features(frame(keys([1], "a"), f=1))
    log = tmp_path / "table_metadata" / "_delta_log"
    mode = log.stat().st_mode
    log.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(PartialWriteError) as raised:
            lake.add_features(frame(keys([2], "a"), f=1))
    finally:
        log.chmod(mode)
    assert raised.value.phase == "writing table table_metadata"
    assert "index" in raised.value.committed


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
    described = lake.describe(lake.feature_metadata_frame()).sort("feature_name", "version")
    assert described.select("feature_name", "version", "comment", "tags").rows() == [
        ("f", 0, "f", ["a"]),
        ("g", 0, "g1", []),
        ("g", 1, "g2", ["b", "c"]),
        ("h", 0, None, []),
    ]


def test_restore_leaves_tables_without_later_commits(tmp_path):
    import time

    lake = lake_at(tmp_path)
    lake.add_features(frame(keys(range(1, 3), "a"), f=1))
    time.sleep(0.05)
    cutoff = dt.datetime.now(UTC)
    time.sleep(0.05)
    lake.add_features(frame(keys(range(1, 3), "a"), g=1))
    lake.restore_to_timestamp(cutoff)
    assert lake.feature_metadata_frame().collect()["feature_name"].to_list() == ["f"]
    assert_frame_equal(lake.read(["f"]), frame(keys(range(1, 3), "a"), f=1), check_column_order=False)


def test_describe_gives_runtime_computed_features_no_annotations(tmp_path):
    lake = GammaFeatureLake(base_path=str(tmp_path), enable_runtime_computed_features=True).initialize()
    lake.add_features(frame(keys(range(1, 3), "a"), f=1), tags={"f": ["x"]})
    lake.add_runtime_computed_features([(pl.col("f") * 2).alias("rc")])
    assert described(lake) == {"f": (None, ["x"]), "rc": (None, [])}


_THREADS_SCRIPT = textwrap.dedent(
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


def test_single_thread_matches_default(tmp_path):
    """Writes, merges and reads finish and agree with ``GAMMA_LAKE_THREADS=1``."""
    outputs = []
    for threads in ("1", None):
        env = {k: v for k, v in os.environ.items() if k != "GAMMA_LAKE_THREADS"}
        if threads:
            env["GAMMA_LAKE_THREADS"] = threads
        path = tmp_path / f"threads-{threads}"
        run = subprocess.run([sys.executable, "-c", _THREADS_SCRIPT, str(path)], check=False, env=env, capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr
        outputs.append(run.stdout)
    assert outputs[0] == outputs[1]


def test_feature_files_hold_one_sorted_row_group_up_to_a_million_rows(tmp_path):
    """Files hold rows in key order, in row groups as large as the Python ``deltalake`` writer's."""
    import pyarrow.parquet as pq

    lake = GammaFeatureLake(base_path=str(tmp_path)).initialize()
    lake.add_features(frame(keys(range(1, 8), [f"s{i:05}" for i in range(20_000)]), f=1))
    files = [f for f in tmp_path.glob("delta_*/*.parquet")]
    assert files
    assert [pq.ParquetFile(f).metadata.num_row_groups for f in files] == [1] * len(files)
    rows = pl.from_arrow(pq.read_table(files[0], columns=KEYS))
    assert rows.equals(rows.sort(KEYS))
