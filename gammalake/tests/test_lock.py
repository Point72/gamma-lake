"""The lock factory held around every commit to a table."""

import datetime as dt
import os
import stat
import threading
import time

import polars as pl
import pyarrow as pa
import pytest
from deltalake import DeltaTable

from gammalake import ConflictError, GammaFeatureLake, PartialWriteError
from gammalake.native import NativeLake

UTC = dt.UTC
LAKE_TABLES = {"index", "table_metadata", "feature_metadata"}


def frame(days, symbols, **scales) -> pl.DataFrame:
    rows = [(d, s) for d in days for s in symbols]
    data = {"timestamp": [dt.datetime(2024, 1, d, tzinfo=UTC) for d, _ in rows], "symbol": [s for _, s in rows]}
    for name, scale in scales.items():
        data[name] = [float(scale * d) for d, _ in rows]
    return pl.DataFrame(data).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))


def version(path: str) -> int | None:
    return DeltaTable(path).version() if DeltaTable.is_deltatable(path) else None


def table_versions(base: str) -> dict[str, int]:
    return {name: version(os.path.join(base, name)) for name in os.listdir(base) if DeltaTable.is_deltatable(os.path.join(base, name))}


class Lock:
    def __init__(self, recorder: "Recorder", path: str):
        self.recorder, self.path = recorder, path

    def __enter__(self):
        if self.recorder.on_enter:
            self.recorder.on_enter(self.path)
        self.entered = version(self.path)

    def __exit__(self, *exc_info):
        self.recorder.held.append((os.path.basename(self.path), self.entered, version(self.path), exc_info))
        if self.recorder.on_exit:
            self.recorder.on_exit(self.path)


class Recorder:
    """A lock factory recording each table locked and its Delta version on entry and exit."""

    def __init__(self, on_lock=None, on_enter=None, on_exit=None):
        self.locked: list[str] = []
        self.held: list[tuple[str, int | None, int | None, tuple]] = []
        self.on_lock, self.on_enter, self.on_exit = on_lock, on_enter, on_exit

    def __call__(self, path: str) -> Lock:
        self.locked.append(path)
        if self.on_lock:
            self.on_lock(path)
        return Lock(self, path)

    def names(self) -> list[str]:
        return sorted(os.path.basename(path) for path in self.locked)


def populated(path) -> GammaFeatureLake:
    lake = GammaFeatureLake(base_path=str(path), enable_runtime_computed_features=True).initialize()
    lake.add_features(frame(range(1, 4), "ab", f=1, g=2))
    lake.add_targets(frame(range(1, 4), "ab", y=1))
    return lake


def with_lock(lake: GammaFeatureLake, lock) -> GammaFeatureLake:
    return GammaFeatureLake(base_path=lake.base_path, enable_runtime_computed_features=True, lock=lock)


def annotate(lake: GammaFeatureLake) -> None:
    metadata = lake.feature_metadata_frame().collect().filter(pl.col("feature_name") == "f")
    lake.annotate(metadata.with_columns(comment=pl.lit("note"), tags=pl.lit(["x"])))


# The tables each operation changes: "features" and "targets" are the tables of f and y, "new" a table it creates.
OPERATIONS = {
    "add_features": (lambda lake: lake.add_features(frame(range(1, 3), "ab", h=1)), ["feature_metadata", "new", "table_metadata"]),
    "add_features_on_new_keys": (
        lambda lake: lake.add_features(frame(range(4, 6), "ab", f=1, g=2)),
        ["features", "index", "table_metadata"],
    ),
    "add_targets": (lambda lake: lake.add_targets(frame(range(1, 3), "ab", z=1)), ["feature_metadata", "new", "table_metadata"]),
    "add_index_rows": (lambda lake: lake.add_index_rows(frame([5], "ab")), ["index"]),
    "add_index_rows_inside_coverage": (lambda lake: lake.add_index_rows(frame([2], "c")), ["features", "index", "targets"]),
    "annotate": (annotate, ["features"]),
    "consolidate_feature_groups": (
        lambda lake: lake.consolidate_feature_groups(["f", "y"]),
        ["feature_metadata", "new", "table_metadata"],
    ),
    "add_runtime_computed_features": (lambda lake: lake.add_runtime_computed_features([(pl.col("f") * 2).alias("rc")]), ["feature_metadata"]),
}


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_writes_lock_exactly_the_tables_they_change(tmp_path, operation):
    run, expected = OPERATIONS[operation]
    lake = populated(tmp_path)
    metadata = lake.feature_metadata_frame().collect()
    roles = {metadata.filter(pl.col("feature_name") == name)["table_addr"].item(): role for name, role in (("f", "features"), ("y", "targets"))}
    before = table_versions(lake.base_path)
    recorder = Recorder()

    run(with_lock(lake, recorder))

    after = table_versions(lake.base_path)
    changed = sorted(name for name in after if before.get(name) != after[name])
    assert recorder.names() == changed
    assert sorted(roles.get(name, name if name in before else "new") for name in changed) == expected
    assert recorder.locked == [os.path.join(lake.base_path, os.path.basename(path)) for path in recorder.locked]


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_commits_happen_while_the_lock_is_held(tmp_path, operation):
    lake = populated(tmp_path)
    recorder = Recorder()
    OPERATIONS[operation][0](with_lock(lake, recorder))
    assert len(recorder.held) == len(recorder.locked)
    for name, entered, exited, exc_info in recorder.held:
        assert exited is not None and (entered is None or exited > entered), name
        assert exc_info == (None, None, None)


def test_initialize_locks_the_tables_it_creates(tmp_path):
    recorder = Recorder()
    lake = GammaFeatureLake(base_path=str(tmp_path), lock=recorder).initialize()
    assert recorder.names() == sorted(LAKE_TABLES)
    assert sorted((name, entered, exited) for name, entered, exited, _ in recorder.held) == [(name, None, 0) for name in sorted(LAKE_TABLES)]
    recorder.locked.clear()
    lake.initialize()
    assert recorder.locked == []


def test_restore_locks_every_table_it_may_restore(tmp_path):
    lake = populated(tmp_path)
    referenced = set(lake.feature_metadata_frame().collect()["table_addr"])
    time.sleep(0.05)
    cutoff = dt.datetime.now(UTC)
    time.sleep(0.05)
    lake.add_features(frame(range(1, 3), "ab", h=1))
    annotate(lake)
    before = table_versions(lake.base_path)
    recorder = Recorder()

    with_lock(lake, recorder).restore_to_timestamp(cutoff)

    after = table_versions(lake.base_path)
    changed = {name for name in after if before[name] != after[name]}
    assert recorder.names() == sorted(LAKE_TABLES | referenced)
    assert changed == {"table_metadata", "feature_metadata", *(name for name in referenced if before[name] > 0)}
    for name, entered, exited, _ in recorder.held:
        assert exited == (entered + 1 if name in changed else entered), name


def test_reads_lock_nothing(tmp_path):
    recorder = Recorder()
    lake = with_lock(populated(tmp_path), recorder)
    lake.read(["f", "g"], targets=["y"])
    lake.read(["f"], materialized=False).collect()
    lake.read(lake.feature_metadata_frame().collect().filter(pl.col("feature_name") == "g"))
    lake.index_frame().collect()
    lake.table_metadata_frame().collect()
    lake.describe(lake.feature_metadata_frame())
    NativeLake(lake.base_path, lock=recorder).scan(["f"]).collect()
    assert recorder.locked == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_a_failed_commit_releases_its_lock(tmp_path):
    lake = populated(tmp_path)
    log = tmp_path / "table_metadata" / "_delta_log"
    mode = log.stat().st_mode

    def deny(path):
        if os.path.basename(path) == "table_metadata":
            log.chmod(stat.S_IRUSR | stat.S_IXUSR)

    recorder = Recorder(on_enter=deny)
    try:
        with pytest.raises(PartialWriteError):
            with_lock(lake, recorder).add_features(frame(range(3, 5), "ab", f=1))
    finally:
        log.chmod(mode)
    assert recorder.names() == sorted(os.path.basename(name) for name, *_ in recorder.held)
    assert ("table_metadata", 2, 2, (None, None, None)) in recorder.held


def is_data_table(path: str) -> bool:
    return os.path.basename(path) not in LAKE_TABLES


class FalsyError(Exception):
    def __bool__(self):
        return False


# Arrow exceptions and falsy exceptions must not be mistaken for errors of the core.
LOCK_ERRORS = [OSError, pa.ArrowInvalid, FalsyError]


@pytest.mark.parametrize("error_type", LOCK_ERRORS)
@pytest.mark.parametrize("raised_by", ["factory", "enter"])
def test_a_lock_error_before_any_commit_propagates_unchanged(tmp_path, raised_by, error_type):
    lake = populated(tmp_path)
    error = error_type("lock service unavailable")

    def fail(path):
        if is_data_table(path):
            raise error

    before = table_versions(lake.base_path)
    recorder = Recorder(**{"on_lock" if raised_by == "factory" else "on_enter": fail})
    with pytest.raises(error_type) as raised:
        with_lock(lake, recorder).add_features(frame(range(3, 5), "ab", f=1))
    assert raised.value is error
    assert table_versions(lake.base_path) == before
    assert recorder.names() == [os.path.basename(path) for path in recorder.locked if is_data_table(path)]


@pytest.mark.parametrize("error_type", LOCK_ERRORS)
@pytest.mark.parametrize("raised_by", ["factory", "enter"])
def test_a_lock_error_after_a_commit_is_a_partial_write(tmp_path, raised_by, error_type):
    lake = populated(tmp_path)
    error = error_type("lock service unavailable")

    def fail(path):
        if os.path.basename(path) == "feature_metadata":
            raise error

    before = table_versions(lake.base_path)
    recorder = Recorder(**{"on_lock" if raised_by == "factory" else "on_enter": fail})
    with pytest.raises(PartialWriteError) as raised:
        with_lock(lake, recorder).add_features(frame(range(3, 5), "ab", h=1))
    assert raised.value.__cause__ is error
    after = table_versions(lake.base_path)
    created = [name for name in after if name not in before]
    assert len(created) == 1
    assert sorted(raised.value.committed) == sorted([*created, "index", "table_metadata"])
    assert after["feature_metadata"] == before["feature_metadata"]
    assert "h" not in lake.feature_metadata_frame().collect()["feature_name"]


@pytest.mark.parametrize("failing", ["new table", "index"])
def test_a_failed_release_after_a_commit_counts_the_table_as_committed(tmp_path, failing):
    lake = populated(tmp_path)
    before = table_versions(lake.base_path)
    error = pa.ArrowInvalid("cannot unlock")

    def fail(path):
        name = os.path.basename(path)
        if (name not in before) if failing == "new table" else (name == "index"):
            raise error

    with pytest.raises(PartialWriteError) as raised:
        with_lock(lake, Recorder(on_exit=fail)).add_features(frame(range(3, 5), "ab", h=1))
    assert raised.value.__cause__ is error
    created = [name for name in table_versions(lake.base_path) if name not in before]
    assert len(created) == 1
    expected = created if failing == "new table" else [*created, "index"]
    assert sorted(raised.value.committed) == sorted(expected)


@pytest.mark.parametrize("raced", ["feature table", "index"])
def test_a_table_changed_after_the_write_read_it_is_not_committed_to(tmp_path, raced):
    lake = populated(tmp_path)
    before = table_versions(lake.base_path)
    changed = []

    def race(path):
        name = os.path.basename(path)
        if not changed and (name == "index" if raced == "index" else name not in LAKE_TABLES):
            DeltaTable(path).alter.set_table_description("changed by another writer")
            changed.append(name)

    expected = ConflictError if raced == "feature table" else PartialWriteError
    with pytest.raises(expected) as raised:
        with_lock(lake, Recorder(on_enter=race)).add_features(frame(range(4, 5), "ab", f=1, g=2))
    assert "changed after this write read it" in str(raised.value)
    after = table_versions(lake.base_path)
    assert after[changed[0]] == before[changed[0]] + 1
    if raced == "feature table":
        assert raised.value.table == changed[0]
        assert after == {**before, changed[0]: before[changed[0]] + 1}
    else:
        assert len(raised.value.committed) == 1
        assert after["index"] == before["index"] + 1
    assert lake.index_frame().collect().height == 6


def test_a_thread_affine_lock_works(tmp_path):
    lake = populated(tmp_path)
    shared = threading.RLock()
    locked = with_lock(lake, lambda path: shared)
    locked.add_features(frame(range(3, 5), "ab", f=1, h=2))
    assert shared.acquire(blocking=False)
    shared.release()
    assert locked.read(["f", "h"]).height == 8


def test_lock_keys_are_canonical_table_paths(tmp_path):
    real = tmp_path / "lake"
    GammaFeatureLake(base_path=str(real)).initialize()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    keys = []
    for base in (str(real), str(real) + "/", str(link), str(link) + "/"):
        recorder = Recorder()
        GammaFeatureLake(base_path=base, lock=recorder).add_index_rows(frame([len(keys) + 1], "a"))
        keys.append(recorder.locked)
    recorder = Recorder()
    NativeLake(str(link) + "/", lock=recorder).add_index_rows(frame([9], "a"))
    keys.append(recorder.locked)
    assert keys == [[os.path.join(os.path.realpath(real), "index")]] * 5
