import asyncio
from datetime import UTC, datetime

import polars as pl
import pytest
import ray
from deltalake import DeltaTable
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gammalake import GammaFeatureLake, PolarsIO, RayIOWriter


def frame(name, days, values=None, symbols=None):
    return pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, day, tzinfo=UTC) for day in days],
            "symbol": symbols or ["A"] * len(days),
            name: values or days,
        }
    )


class WriteBarrier:
    def __init__(self):
        self.arrivals = 0
        self.ready = asyncio.Event()

    async def arrive(self):
        self.arrivals += 1
        if self.arrivals >= 2:
            self.ready.set()
        await asyncio.wait_for(self.ready.wait(), timeout=30)

    def count(self):
        return self.arrivals


class BarrierIO(PolarsIO):
    def __init__(self, barrier):
        self.barrier = barrier

    def write_delta(self, df, target, **kwargs):
        columns = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        if "left" in columns or "right" in columns:
            ray.get(self.barrier.arrive.remote(), timeout=40)
        return super().write_delta(df, target, **kwargs)


class FailingIO(PolarsIO):
    def __init__(self, stage="feature"):
        self.stage = stage

    def write_delta(self, df, target, **kwargs):
        columns = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        if (self.stage == "feature" and "fail" in columns) or target.rsplit("/", 1)[-1] == self.stage:
            raise RuntimeError(f"injected {self.stage} write failure")
        return super().write_delta(df, target, **kwargs)


class WriteGate:
    def __init__(self):
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self.events = {}

    async def hold(self):
        self.started.set()
        await asyncio.wait_for(self.released.wait(), timeout=60)

    async def wait_started(self):
        await asyncio.wait_for(self.started.wait(), timeout=30)

    def release(self):
        self.released.set()

    def record(self, name):
        self.events.setdefault(name, asyncio.Event()).set()

    async def wait_for(self, name):
        await asyncio.wait_for(self.events.setdefault(name, asyncio.Event()).wait(), timeout=30)


class GatedIO(PolarsIO):
    def __init__(self, gate, blocked="slow"):
        self.gate = gate
        self.blocked = blocked

    def write_delta(self, df, target, **kwargs):
        columns = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        if self.blocked in columns or target.rsplit("/", 1)[-1] == self.blocked:
            ray.get(self.gate.hold.remote(), timeout=70)
        result = super().write_delta(df, target, **kwargs)
        for name in columns:
            ray.get(self.gate.record.remote(name))
        return result


@pytest.fixture
def lake(tmp_path, barebones_ray_cluster):
    GammaFeatureLake(base_path=str(tmp_path)).initialize()
    lake = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=True, coordinate_writes=True)
    yield lake
    ray.kill(RayIOWriter.connect(lake))


def test_ray_only(tmp_path):
    with pytest.raises(ValidationError, match="requires run_on_ray_cluster"):
        GammaFeatureLake(base_path=str(tmp_path), coordinate_writes=True)


def test_next_index_and_independent_write_finish_before_slow_call(lake):
    gate = ray.remote(WriteGate).options(num_cpus=0).remote()
    lake.io = GatedIO(gate)
    writer = RayIOWriter.connect(lake)
    slow = writer.submit.remote(lake, frame("slow", [1, 3]))
    try:
        ray.get(gate.wait_started.remote(), timeout=40)
        fast = writer.submit.remote(lake, frame("fast", [4]))
        assert ray.get(fast, timeout=30) == []
        assert ray.wait([slow], timeout=0)[0] == []
        assert lake.index_frame().collect().height == 3
        # Existing-index calls also take the hot path while the slow table is pending.
        assert ray.get(writer.submit.remote(lake, frame("existing_keys", [1, 3])), timeout=30) == []
    finally:
        ray.get(gate.release.remote())
    ray.get(slow, timeout=30)
    expected = frame("slow", [1, 3, 4], [1, 3, None]).with_columns(fast=pl.Series([None, None, 4], dtype=pl.Int64))
    assert_frame_equal(lake.read(["slow", "fast"]), expected, check_column_order=False)
    ray.kill(gate)


def test_pending_creation_receives_later_alignment(lake):
    gate = ray.remote(WriteGate).options(num_cpus=0).remote()
    lake.io = GatedIO(gate)
    writer = RayIOWriter.connect(lake)
    slow = writer.submit.remote(lake, frame("slow", [1, 3]))
    try:
        ray.get(gate.wait_started.remote(), timeout=40)
        later = writer.submit.remote(lake, frame("fast", [2]))
        ray.get(gate.wait_for.remote("fast"), timeout=40)
        assert lake.index_frame().collect().height == 3
        # The fast table is already written; only its alignment dependency is blocked.
        assert ray.wait([later], timeout=0)[0] == []
    finally:
        ray.get(gate.release.remote())
    ray.get([slow, later], timeout=40)
    expected = frame("slow", [1, 2, 3], [1, None, 3]).with_columns(fast=pl.Series([None, 2, None], dtype=pl.Int64))
    assert_frame_equal(lake.read(["slow", "fast"]), expected, check_column_order=False)
    ray.kill(gate)


@pytest.mark.parametrize("overlap_mode", ["copy", "merge"])
def test_conflicting_pending_writes_do_not_block_unrelated_tables(lake, overlap_mode):
    gate = ray.remote(WriteGate).options(num_cpus=0).remote()
    lake.io = GatedIO(gate)
    writer = RayIOWriter.connect(lake)
    first = writer.submit.remote(lake, frame("slow", [1, 3]))
    try:
        ray.get(gate.wait_started.remote(), timeout=40)
        second = writer.submit.remote(lake, frame("slow", [1, 3], [10, 30]), overlap_mode=overlap_mode)
        ray.get(writer.submit.remote(lake, frame("fast", [4])), timeout=30)
        assert ray.wait([first, second], timeout=0)[0] == []
    finally:
        ray.get(gate.release.remote())
    ray.get([first, second], timeout=40)
    metadata = lake.feature_metadata_frame().collect().filter(pl.col("feature_name") == "slow")
    assert sorted(metadata["version"]) == [0, 1]
    assert lake.read(["slow"])["slow"].to_list() == [10, 30, None]
    ray.kill(gate)


def test_metadata_publication_does_not_hold_index_lane(lake):
    gate = ray.remote(WriteGate).options(num_cpus=0).remote()
    lake.io = GatedIO(gate, "table_metadata")
    writer = RayIOWriter.connect(lake)
    first = writer.submit.remote(lake, frame("a", [1]))
    try:
        ray.get(gate.wait_started.remote(), timeout=40)
        second = writer.submit.remote(lake, frame("fast", [2]))
        ray.get(gate.wait_for.remote("fast"), timeout=40)
        assert lake.index_frame().collect().height == 2
        assert ray.wait([first, second], timeout=0)[0] == []
    finally:
        ray.get(gate.release.remote())
    ray.get([first, second], timeout=40)
    assert lake.feature_metadata_frame().collect().height == 2
    ray.kill(gate)


def test_pending_copy_source_and_destination_both_get_alignment(lake):
    seed = lake.model_copy(update={"coordinate_writes": False, "run_on_ray_cluster": False})
    seed.add_features(frame("slow", [1, 3, 5]))
    gate = ray.remote(WriteGate).options(num_cpus=0).remote()
    lake.io = GatedIO(gate)
    writer = RayIOWriter.connect(lake)
    copy = writer.submit.remote(lake, frame("slow", [3, 5], [30, 50]))
    try:
        ray.get(gate.wait_started.remote(), timeout=40)
        later = writer.submit.remote(lake, frame("fast", [2, 4]))
        ray.get(gate.wait_for.remote("fast"), timeout=40)
        # A third, unrelated call need not wait for either copy source or destination.
        ray.get(writer.submit.remote(lake, frame("tail", [6])), timeout=30)
    finally:
        ray.get(gate.release.remote())
    ray.get([copy, later], timeout=40)
    result = lake.read(["slow", "fast"])
    assert result["slow"].to_list() == [1, None, 30, None, 50, None]
    assert result["fast"].to_list() == [None, 2, None, 4, None, None]
    for addr in lake.feature_metadata_frame().collect().filter(pl.col("feature_name") == "slow")["table_addr"]:
        keys = lake.io.scan_delta(lake.get_path(addr)).select(lake.sort_keys).collect().sort(lake.sort_keys)
        assert_frame_equal(keys, frame("unused", [1, 2, 3, 4, 5]).select(lake.sort_keys))
    ray.kill(gate)


def test_nonrecursive_cancellation_does_not_drop_pending_table(lake):
    gate = ray.remote(WriteGate).options(num_cpus=0).remote()
    lake.io = GatedIO(gate)
    writer = RayIOWriter.connect(lake)
    first = writer.submit.remote(lake, frame("slow", [1, 3]))
    try:
        ray.get(gate.wait_started.remote(), timeout=40)
        ray.cancel(first, recursive=False)
        second = writer.submit.remote(lake, frame("slow", [1, 3], [10, 30]), overlap_mode="merge")
        ray.get(writer.submit.remote(lake, frame("tail", [4])), timeout=30)
    finally:
        ray.get(gate.release.remote())
    ray.get(second, timeout=40)
    assert lake.read(["slow"])["slow"].to_list() == [10, 30, None]
    assert sorted(lake.feature_metadata_frame().collect().filter(pl.col("feature_name") == "slow")["version"]) == [0, 1]
    ray.kill(gate)


def test_existing_keys_skip_index_commit(lake):
    lake.add_features(frame("a", [1, 2]))
    before = DeltaTable(lake.index).version()
    lake.add_features(frame("b", [1, 2]))
    assert DeltaTable(lake.index).version() == before


def test_shared_actor_and_parallel_calls(lake):
    barrier = ray.remote(WriteBarrier).options(num_cpus=0).remote()
    lake.io = BarrierIO(barrier)
    coordinator = RayIOWriter.connect(lake)
    assert coordinator == RayIOWriter.connect(lake.model_copy())
    frames = [frame("left", [1, 3]), frame("right", [2, 3])]
    assert ray.get(coordinator.submit_batch.remote(lake, frames), timeout=90) == [[], []]
    assert ray.get(barrier.count.remote()) == 2
    assert lake.index_frame().collect().height == 3
    expected = frame("left", [1, 2, 3], [1, None, 3]).with_columns(right=pl.Series([None, 2, 3], dtype=pl.Int64))
    assert_frame_equal(lake.read(["left", "right"]), expected, check_column_order=False)
    ray.kill(barrier)


@pytest.mark.parametrize("overlap_mode", ["copy", "merge"])
def test_backfill_alignment_and_conflicting_groups(lake, overlap_mode):
    actor = RayIOWriter.connect(lake)
    lake.add_features(frame("a", [1, 3]))
    lake.add_features(frame("a", [4]))
    frames = [frame("a", [2, 3], [20, 30]), frame("b", [2, 3], [200, 300])]
    ray.get(actor.submit_batch.remote(lake, frames, overlap_mode=overlap_mode), timeout=90)
    expected = frame("a", [1, 2, 3, 4], [1, 20, 30, None] if overlap_mode == "copy" else [1, 20, 30, 4])
    expected = expected.with_columns(b=pl.Series([None, 200, 300, None], dtype=pl.Int64))
    assert_frame_equal(lake.read(["a", "b"]), expected, check_column_order=False)
    index = lake.index_frame().collect()
    assert index.height == index.unique().height == 4
    # All historical dense tables must remain aligned, not just current versions.
    for addr in lake.feature_metadata_frame().collect()["table_addr"].unique():
        physical = lake.io.scan_delta(lake.get_path(addr)).collect().sort(lake.sort_keys)
        keys = physical.select(lake.sort_keys)
        assert keys.height == keys.unique().height
        assert_frame_equal(keys, index.filter(pl.col("timestamp") <= keys["timestamp"].max()).sort(lake.sort_keys))
    # These requests share a feature name, so the second must use the first's metadata.
    ray.get(actor.submit_batch.remote(lake, [frame("a", [3], [31]), frame("a", [3], [32])], overlap_mode=overlap_mode), timeout=90)
    assert lake.read(["a"], start=datetime(2020, 1, 3, tzinfo=UTC), end=datetime(2020, 1, 3, tzinfo=UTC))["a"].item() == 32
    metadata = lake.feature_metadata_frame().collect().filter(pl.col("feature_name") == "a")
    assert metadata["version"].n_unique() == metadata.height


def test_same_timestamp_new_symbol_and_colocated_columns(lake):
    lake.add_features(frame("a", [1]).with_columns(b=pl.lit(10, dtype=pl.Int64)))
    lake.add_features(frame("other", [1]))
    actor = RayIOWriter.connect(lake)
    ray.get(
        actor.submit_batch.remote(lake, [frame("a", [1], [2], ["B"]), frame("b", [1], [20], ["B"])], overlap_mode="merge"),
        timeout=90,
    )
    expected = frame("a", [1, 1], [1, 2], ["A", "B"]).with_columns(
        b=pl.Series([None, 20], dtype=pl.Int64), other=pl.Series([1, None], dtype=pl.Int64)
    )
    assert_frame_equal(lake.read(["a", "b", "other"]), expected, check_column_order=False)
    keys = lake.index_frame().collect()
    assert keys.height == keys.unique().height == 2


@pytest.mark.parametrize("overlap_mode", ["copy", "merge"])
def test_new_keys_preserve_existing_values_and_versions(lake, overlap_mode):
    lake.add_features(frame("a", [1, 3]))
    actor = RayIOWriter.connect(lake)
    ray.get(actor.submit_batch.remote(lake, [frame("a", [2], [20]), frame("b", [2], [200])], overlap_mode=overlap_mode), timeout=90)
    lake.add_features(frame("a", [1], [10], ["B"]), overlap_mode=overlap_mode)
    expected = frame("a", [1, 1, 2, 3], [1, 10, 20, 3], ["A", "B", "A", "A"])
    assert_frame_equal(lake.read(["a"]), expected, check_column_order=False)
    metadata = lake.feature_metadata_frame().collect().filter(pl.col("feature_name") == "a")
    assert metadata["version"].to_list() == [0]


def test_merge_backfill_preserves_physical_high_water_mark(lake):
    lake.add_features(frame("a", [1, 3, 4]))
    lake.add_features(frame("a", [2, 3], [20, 30]), overlap_mode="merge")
    lake.add_features(frame("b", [4], [400], ["B"]))
    lake.add_features(frame("a", [4], [40]), overlap_mode="merge")
    expected = frame("a", [1, 2, 3, 4, 4], [1, 20, 30, 40, None], ["A", "A", "A", "A", "B"])
    expected = expected.with_columns(b=pl.Series([None, None, None, None, 400], dtype=pl.Int64))
    assert_frame_equal(lake.read(["a", "b"]), expected, check_column_order=False)
    metadata = lake.feature_metadata_frame().collect()
    for addr in metadata["table_addr"].unique():
        physical = lake.io.scan_delta(lake.get_path(addr)).collect()
        assert physical.height == physical.select(lake.sort_keys).unique().height == 5


@pytest.mark.parametrize("stage", ["index", "feature", "table_metadata", "feature_metadata"])
def test_failure_stops_further_writes(lake, stage):
    lake.io = FailingIO(stage)
    actor = RayIOWriter.connect(lake)
    with pytest.raises(ray.exceptions.RayTaskError, match=f"injected {stage} write failure"):
        ray.get(actor.submit.remote(lake, frame("fail", [1, 2])), timeout=90)
    assert lake.feature_metadata_frame().collect().is_empty()
    with pytest.raises(ray.exceptions.RayTaskError, match="stopped after a failed write"):
        lake.add_features(frame("later", [3]))
    assert lake.index_frame().collect().height == (0 if stage == "index" else 2)


def test_object_refs_and_index_only(lake):
    lake.add_index_rows(ray.put(frame("discard", [1]).select(lake.sort_keys)))
    lake.add_targets(ray.put(frame("target", [2])))
    assert lake.index_frame().collect().height == 2
    assert lake.feature_metadata_frame().collect()["signal_type"].to_list() == ["target"]


@pytest.mark.parametrize("signal_type", ["as_of_feature", "sparse_feature"])
@pytest.mark.parametrize("overlap_mode", ["copy", "merge"])
def test_sparse_observations_are_not_padded(lake, signal_type, overlap_mode):
    actor = RayIOWriter.connect(lake)
    options = {"signal_type": signal_type, "feature_params": {"strategy": "backward"}, "overlap_mode": overlap_mode}
    ray.get(actor.submit_batch.remote(lake, [frame("a", [1]), frame("b", [2])], **options), timeout=90)
    ray.get(actor.submit_batch.remote(lake, [frame("a", [4]), frame("b", [3])], **options), timeout=90)
    result = lake.read(["a"]).sort(lake.sort_keys)
    assert result["a"].to_list() == ([1, 1, 1, 4] if signal_type == "as_of_feature" else [1, None, None, 4])
    ray.get(actor.submit.remote(lake, frame("a", [1, 4], [10, 40]), **options), timeout=90)
    result = lake.read(["a"]).sort(lake.sort_keys)
    assert result["a"].to_list() == ([10, 10, 10, 40] if signal_type == "as_of_feature" else [10, None, None, 40])
    for addr in lake.feature_metadata_frame().collect()["table_addr"].unique():
        physical = lake.io.scan_delta(lake.get_path(addr)).collect()
        values = [name for name in physical.columns if name not in lake.sort_keys]
        assert not physical.select(pl.any_horizontal(pl.col(values).is_null()).any()).item()


def test_mismatched_configuration_rejected(lake):
    actor = RayIOWriter.connect(lake)
    with pytest.raises(ray.exceptions.RayTaskError, match="same lake configuration"):
        ray.get(actor.submit.remote(lake.model_copy(update={"compression": "snappy"}), frame("a", [1])), timeout=30)
    assert lake.index_frame().collect().is_empty()


def test_independent_producers_share_coordinator(lake):
    @ray.remote(num_cpus=0, max_retries=0)
    def produce(config, index):
        writer = GammaFeatureLake(**config)
        writer.add_features(frame(f"feature_{index}", [1, index + 2]))

    config = lake.model_dump(exclude={"io"})
    ray.get([produce.remote(config, index) for index in range(4)], timeout=120)
    keys = lake.index_frame().collect()
    assert keys.height == keys.unique().height == 5
    result = lake.read([f"feature_{index}" for index in range(4)])
    for index in range(4):
        assert result[f"feature_{index}"].drop_nulls().to_list() == [1, index + 2]


def test_reject_unsupported_operations(lake):
    with pytest.raises(ValueError, match="Maintenance"):
        lake.initialize()
    with pytest.raises(ValueError, match="Maintenance"):
        lake.consolidate_feature_groups(["a"])
    with pytest.raises(ray.exceptions.RayTaskError, match="metadata overrides"):
        lake.add_features(frame("a", [1]), metadata=pl.DataFrame())
    assert lake.index_frame().collect().is_empty()
    lake.add_features(frame("a", [1]))
