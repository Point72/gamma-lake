"""Compatibility with the pure-Python gamma-lake on the same lake.

These tests need the source of an earlier gamma-lake (the ``gammalake`` package directory's
parent) in ``GAMMA_LAKE_OLD_SRC``; ``make test-compat`` extracts one with ``git archive``.
Each step runs with one engine; after every step both engines read the lake and must agree.
Old-engine steps run in a subprocess with the old source first on ``PYTHONPATH``.
"""

import datetime as dt
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from gammalake import GammaFeatureLake
from gammalake.tests.test_native_write import KEYS, SCENARIOS, Model, frame, keys

OLD_SRC = os.environ.get("GAMMA_LAKE_OLD_SRC")
pytestmark = pytest.mark.skipif(not OLD_SRC, reason="GAMMA_LAKE_OLD_SRC is not set")

# Old-engine bugs the model does not reproduce (Point72/gamma-lake#45, #46 and #48). Scenarios
# that exercise them are compared between engines only.
OLD_ENGINE_BUGS = {"all_new_keys_after_coverage", "overlap_copy", "new_key_at_coverage_end"}

_RUNNER = textwrap.dedent(
    """
    import json, sys
    import polars as pl
    import gammalake
    from gammalake import GammaFeatureLake

    path, ops, src = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
    assert gammalake.__file__.startswith(src), gammalake.__file__
    for op in ops:
        lake = GammaFeatureLake(base_path=path)
        if op["method"] == "initialize":
            lake.initialize()
            continue
        args = [pl.read_parquet(op["frame"])] if "frame" in op else op.get("args", [])
        result = getattr(lake, op["method"])(*args, **op.get("kwargs", {}))
        if "out" in op:
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            result.write_parquet(op["out"])
    """
)


class Engines:
    """Runs the same operations with either engine on one lake."""

    def __init__(self, path: Path):
        self.path = path
        self.files = path.parent / f"{path.name}-files"
        self.files.mkdir()
        self.count = 0

    def _file(self) -> str:
        self.count += 1
        return str(self.files / f"{self.count}.parquet")

    def run(self, engine: str, method: str, df: pl.DataFrame | None = None, out: bool = False, args=(), **kwargs):
        op = {"method": method, "kwargs": kwargs, "args": list(args)}
        if df is not None:
            op["frame"] = self._file()
            df.write_parquet(op["frame"])
        if out:
            op["out"] = self._file()
        if engine == "old":
            # -P keeps the working directory, which may hold the new package, off sys.path.
            done = subprocess.run(
                [sys.executable, "-P", "-W", "ignore", "-c", _RUNNER, str(self.path), json.dumps([op]), OLD_SRC],
                env={**os.environ, "PYTHONPATH": OLD_SRC},
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            assert done.returncode == 0, done.stderr[-4000:]
        else:
            lake = GammaFeatureLake(base_path=str(self.path))
            result = getattr(lake, method)(*([df] if df is not None else list(args)), **kwargs)
            if out:
                (result.collect() if isinstance(result, pl.LazyFrame) else result).write_parquet(op["out"])
        return pl.read_parquet(op["out"]) if out else None

    def initialize(self, engine: str) -> None:
        if engine == "old":
            self.run("old", "initialize")
        else:
            GammaFeatureLake(base_path=str(self.path)).initialize()

    def read(self, engine: str, names) -> pl.DataFrame:
        return canonical(self.run(engine, "read", out=True, args=[list(names)]))

    def agree(self, names) -> pl.DataFrame:
        """Both engines read ``names`` the same way; returns the read."""
        old, new = self.read("old", names), self.read("new", names)
        assert_frame_equal(new, old)
        return new

    def metadata_agrees(self) -> None:
        for method in ("feature_metadata_frame", "table_metadata_frame", "index_frame"):
            old = self.run("old", method, out=True)
            new = self.run("new", method, out=True)
            assert_frame_equal(canonical(new), canonical(old), check_dtypes=False)


def canonical(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(sorted(df.columns))
    keys = [c for c in KEYS if c in df.columns]
    if not keys:
        keys = [c for c in df.columns if df.schema[c].is_temporal() or df.schema[c] in (pl.String, pl.Int64, pl.Float64)]
    return df.sort(keys, nulls_last=False, maintain_order=True)


def engines_for(pattern: str, n: int) -> list[str]:
    first, second = ("old", "new") if pattern == "old-first" else ("new", "old")
    return [first if i % 2 == 0 else second for i in range(n)]


@pytest.mark.parametrize("pattern", ["old-first", "new-first"])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_engines_take_turns_writing(tmp_path, scenario, pattern):
    steps = SCENARIOS[scenario]
    lake = Engines(tmp_path / "lake")
    lake.initialize(engines_for(pattern, 1)[0])
    model, written = Model(), set()
    for engine, (rows, mode, scales) in zip(engines_for(pattern, len(steps)), steps, strict=True):
        lake.run(engine, "add_features", frame(rows, **scales), owner="compat", overlap_mode=mode)
        model.add(rows, **scales)
        written |= set(scales)
        read = lake.agree(sorted(written))
        if scenario not in OLD_ENGINE_BUGS:
            assert_frame_equal(read, canonical(model.read(sorted(written))))
    lake.metadata_agrees()


def _as_of_frame(days, value: float) -> pl.DataFrame:
    return frame(keys(days, "ab"), a=value).rename({"a": "carried"})


@pytest.mark.parametrize("writer", ["old", "new"])
def test_signal_types_written_by_one_engine_read_by_the_other(tmp_path, writer):
    lake = Engines(tmp_path / "lake")
    lake.initialize(writer)
    lake.run(writer, "add_features", frame(keys(range(1, 8), "ab"), f=1), owner="compat")
    lake.run(writer, "add_targets", frame(keys(range(1, 6), "ab"), y=2), owner="compat")
    params = {"strategy": "backward", "tolerance": None, "allow_exact_matches": True}
    lake.run(writer, "add_as_of_features", _as_of_frame([1, 4], 5.0), params=params, owner="compat")
    lake.run(writer, "add_sparse_features", frame([(3, "a"), (6, "b")], s=3), owner="compat")
    lake.agree(["f", "carried", "s"])
    lake.run(writer, "add_features", frame(keys(range(8, 10), "ab"), f=1), owner="compat")
    lake.run(writer, "add_sparse_features", frame([(9, "a")], s=3), owner="compat")
    lake.agree(["f", "carried", "s"])
    lake.metadata_agrees()


def test_core_maintenance_is_readable_by_the_old_engine(tmp_path):
    lake = Engines(tmp_path / "lake")
    lake.initialize("old")
    lake.run("old", "add_features", frame(keys(range(1, 5), "ab"), f=1), owner="compat")
    lake.run("old", "add_features", frame(keys(range(1, 5), "ab"), g=2), owner="compat")
    before = lake.agree(["f", "g"])
    time.sleep(0.05)
    cutoff = dt.datetime.now(dt.UTC)
    time.sleep(0.05)
    lake.run("new", "consolidate_feature_groups", args=[["f", "g"]])
    assert_frame_equal(lake.agree(["f", "g"]), before)
    lake.run("new", "add_features", frame(keys(range(5, 7), "ab"), f=1, g=2), owner="compat")
    lake.agree(["f", "g"])
    lake.metadata_agrees()
    lake.run("new", "restore_to_timestamp", args=[cutoff])
    assert_frame_equal(lake.agree(["f", "g"]), before)
    lake.run("old", "add_features", frame(keys(range(5, 7), "ab"), f=1, g=2), owner="compat")
    lake.agree(["f", "g"])


def _file(path: Path, merged: bool) -> tuple:
    """The parquet settings, Arrow schema, row groups and row order of one file.

    Row order is left out for files a merge wrote.

    The ``created_by`` string is left out: it names the parquet library version each engine links.
    """
    parquet = pq.ParquetFile(path).metadata
    columns = []
    for i in range(parquet.num_row_groups):
        group = parquet.row_group(i)
        for j in range(group.num_columns):
            c, stat = group.column(j), group.column(j).statistics
            columns.append(
                (
                    c.path_in_schema,
                    c.physical_type,
                    str(parquet.schema.column(j).logical_type),
                    c.compression,
                    tuple(sorted(c.encodings)),
                    c.has_dictionary_page,
                    None if stat is None else (stat.has_min_max, stat.has_null_count, stat.has_distinct_count),
                    c.has_column_index,
                    c.has_offset_index,
                )
            )
    arrow = parquet.schema.to_arrow_schema()
    in_order = None
    if not merged and set(KEYS) <= set(arrow.names):
        rows = pl.from_arrow(pq.read_table(path, columns=KEYS))
        in_order = rows.equals(rows.sort(KEYS))
    return (
        parquet.format_version,
        tuple(sorted(k.decode() for k in parquet.metadata or {})),
        tuple(f"{f.name}: {f.type}" for f in arrow),
        tuple(parquet.row_group(i).num_rows for i in range(parquet.num_row_groups)),
        tuple(columns),
        in_order,
    )


def _settings(path: Path) -> dict[str, list[tuple]]:
    """Each table's commits in version order, with the Delta and parquet settings of every file.

    Feature tables are keyed by the features they hold, since their names are random. The
    commit info is reduced to the operation, because its parameters and metrics name the
    Delta library version.
    """
    tables: dict[str, list[tuple]] = {}
    for table in sorted(p for p in path.iterdir() if (p / "_delta_log").is_dir()):
        commits = []
        fields: list[str] = []
        for log in sorted((table / "_delta_log").glob("*.json")):
            actions = []
            lines = list(map(json.loads, log.read_text().splitlines()))
            # Neither engine keeps key order through a merge's join.
            merge = any(a.get("commitInfo", {}).get("operation") == "MERGE" for a in lines)
            for action in lines:
                if "protocol" in action:
                    actions.append(("protocol", json.dumps(action["protocol"], sort_keys=True)))
                elif "metaData" in action:
                    meta = action["metaData"]
                    fields = [f["name"] for f in json.loads(meta["schemaString"])["fields"]]
                    actions.append(
                        (
                            "metadata",
                            meta["schemaString"],
                            json.dumps(meta.get("configuration"), sort_keys=True),
                            json.dumps(meta["partitionColumns"]),
                        )
                    )
                elif "commitInfo" in action:
                    actions.append(("commit", action["commitInfo"].get("operation")))
                elif "add" in action:
                    add = action["add"]
                    stats = json.loads(add.get("stats") or "{}")
                    actions.append(
                        (
                            "add",
                            tuple(sorted(add)),
                            tuple(sorted(stats)),
                            tuple(sorted(stats.get("minValues", {}))),
                            _file(table / add["path"], merge),
                        )
                    )
                else:
                    actions.append((next(iter(action)),))
            commits.append(sorted(actions, key=repr))
        key = table.name if not table.name.startswith("delta_") else ",".join(f for f in fields if f not in KEYS)
        assert key not in tables, f"two tables hold {key}"
        tables[key] = commits
    return tables


def _nested_frame(days) -> pl.DataFrame:
    rows = keys(days, "ab")
    return frame(rows).with_columns(
        tags=pl.Series([[s, str(d)] for d, s in rows]),
        rec=pl.Series([{"s": s, "b": s.encode(), "v": float(d)} for d, s in rows]),
    )


def test_both_engines_write_files_with_the_same_settings(tmp_path):
    many = [f"s{i:05}" for i in range(20_000)]
    params = {"strategy": "backward", "tolerance": None, "allow_exact_matches": True}
    for writer in ("old", "new"):
        lake = Engines(tmp_path / writer)
        lake.initialize(writer)
        lake.run(writer, "add_features", frame(keys(range(1, 6), "ab"), f=1, g=2), owner="compat")
        lake.run(writer, "add_features", frame(keys(range(6, 9), "ab"), f=1, g=2), owner="compat")
        lake.run(writer, "add_features", frame(keys(range(7, 10), "ab"), f=3), owner="compat", overlap_mode="merge")
        lake.run(writer, "add_features", frame(keys(range(8, 11), "ab"), g=5), owner="compat", overlap_mode="copy")
        lake.run(writer, "add_targets", frame(keys(range(1, 6), "ab"), y=2), owner="compat")
        lake.run(writer, "add_as_of_features", _as_of_frame([1, 4], 5.0), params=params, owner="compat")
        lake.run(writer, "add_sparse_features", frame([(3, "a"), (6, "b")], s=3), owner="compat")
        lake.run(writer, "add_features", _nested_frame(range(1, 6)), owner="compat")
        lake.run(writer, "add_features", _nested_frame(range(4, 7)), owner="compat", overlap_mode="merge")
        # More rows than one 128Ki-row group holds, written by several threads.
        lake.run(writer, "add_features", frame(keys(range(11, 18), many), h=1), owner="compat")
    old, new = _settings(tmp_path / "old"), _settings(tmp_path / "new")
    assert old.keys() == new.keys()
    for table in old:
        assert len(new[table]) == len(old[table]), (table, len(old[table]), len(new[table]))
        for version, (was, now) in enumerate(zip(old[table], new[table])):
            assert now == was, (table, version, [a for a in was if a not in now], [a for a in now if a not in was])
