from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import BaseModel, ConfigDict

from gammalake._ray import remote, require_ray
from gammalake._types import RayObjectReference

if TYPE_CHECKING:
    from gammalake.gamma_feature_lake import GammaFeatureLake

__all__ = ("RayIOWriter",)

logger = logging.getLogger(__name__)


class _WritePlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source: str | None
    target: str
    feature_metadata: pl.DataFrame
    table_metadata: pl.DataFrame
    publication: pl.DataFrame | None
    options: dict


def _write_features(lake: GammaFeatureLake, plan: _WritePlan, deltas: tuple, *dependencies) -> tuple:
    from gammalake.gamma_feature_lake import update_feature_tables

    inner, maximum, minimum, length, columns, new, _, missing = deltas
    if plan.options["signal_type"] in ("as_of_feature", "sparse_feature"):
        missing = missing.head(0)
    return update_feature_tables(
        lake,
        plan.source,
        length,
        inner,
        new,
        missing,
        maximum,
        minimum,
        columns,
        feature_metadata=plan.feature_metadata,
        table_metadata=plan.table_metadata,
        _planned_write=(plan.target, plan.publication),
        **plan.options,
    )


def _align(lake: GammaFeatureLake, new: pl.DataFrame, row: dict, minimum, *dependencies) -> None:
    from gammalake.gamma_feature_lake import align_feature_tables

    align_feature_tables(lake, new, row, minimum)


def _publish(lake: GammaFeatureLake, results: list[tuple]) -> None:
    from gammalake.gamma_feature_lake import write_metadata

    write_metadata(lake, lake.table_metadata, *(table for table, _ in results))
    write_metadata(lake, lake.feature_metadata, *(features for _, features in results), schema_mode="merge")


class RayIOWriter:
    """Ray IO writer with one index lane and independent physical-table lanes.

    Each call reserves destinations and versions, commits its index additions,
    then releases the index lane without waiting for feature writes or padding.
    Table dependencies include pending creations and historical copy sources.
    Metadata publication uses a separate short critical section. All producers
    must use the same actor and storage configuration; readers are not isolated.
    """

    def __init__(self, lake: GammaFeatureLake):
        self._configuration = lake.model_dump(exclude={"io"})
        self._io_type = type(lake.io)
        self._io_configuration = vars(lake.io)
        self._lake = lake.model_copy(update={"coordinate_writes": False, "run_on_ray_cluster": False})
        self._features = lake.feature_metadata_frame().collect()
        self._tables = lake.table_metadata_frame().collect()
        self._index_lock = asyncio.Lock()
        self._publication_lock = asyncio.Lock()
        self._tails: dict[str, RayObjectReference] = {}
        self._active: set[asyncio.Task] = set()
        self._failure: str | None = None

    @classmethod
    def connect(cls, lake: GammaFeatureLake) -> Any:
        """Get the shared writer for a lake in the current Ray cluster.

        The named actor outlives submitting drivers but never restarts or
        automatically replays writes. Initialize the lake and Ray beforehand.
        """
        ray = require_ray()
        if not lake.run_on_ray_cluster or not lake.coordinate_writes:
            raise ValueError("RayIOWriter requires coordinate_writes=True and run_on_ray_cluster=True")
        if not lake._is_initialized:
            raise ValueError("Initialize the lake before enabling coordinated writes")
        if not ray.is_initialized():
            raise RuntimeError("Initialize Ray explicitly before using coordinated writes")
        root = lake.base_path.rstrip("/") if "://" in lake.base_path else os.path.realpath(lake.base_path)
        name = "gamma-lake-" + hashlib.sha256(root.encode()).hexdigest()
        return (
            ray.remote(cls)
            .options(
                name=name,
                namespace="gammalake-writes",
                get_if_exists=True,
                lifetime="detached",
                num_cpus=0,
                max_restarts=0,
                max_task_retries=0,
            )
            .remote(lake)
        )

    def _check_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)

    async def submit(
        self,
        lake: GammaFeatureLake,
        df: pl.DataFrame | RayObjectReference[pl.DataFrame],
        *,
        signal_type: str = "feature",
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
        feature_params: dict | None = None,
        overlap_mode: str = "copy",
    ) -> list:
        """Plan one add operation and await only its own write/publication work."""
        from gammalake.gamma_feature_lake import preprocess_df

        if lake.model_dump(exclude={"io"}) != self._configuration or type(lake.io) is not self._io_type or vars(lake.io) != self._io_configuration:
            raise ValueError("All coordinated writers must use the same lake configuration and IO implementation")
        self._check_failure()
        if not isinstance(df, pl.DataFrame):
            df = await df
        df = preprocess_df(self._lake, df)
        if df.is_empty():
            raise ValueError("No valid index rows remain after preprocessing")
        if metadata is not None:
            raise ValueError("Explicit metadata overrides are not supported by coordinated writes")
        if signal_type not in ("feature", "target", "as_of_feature", "sparse_feature") or overlap_mode not in ("copy", "merge"):
            raise ValueError("Invalid signal_type or overlap_mode")
        options = {"signal_type": signal_type, "owner": owner, "feature_params": feature_params, "overlap_mode": overlap_mode}
        # Keep admitted writes alive if a submitting actor method is cancelled.
        task = asyncio.create_task(self._add(df, options))
        self._active.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    def _finished(self, task: asyncio.Task) -> None:
        self._active.discard(task)
        if not task.cancelled():
            task.exception()

    async def submit_batch(self, lake: GammaFeatureLake, frames: list[pl.DataFrame | RayObjectReference[pl.DataFrame]], **options) -> list[list]:
        """Submit several independent calls; no batch barrier or union index is used."""
        return await asyncio.gather(*(self.submit(lake, frame, **options) for frame in frames))

    async def _add(self, df: pl.DataFrame, options: dict) -> list:
        from gammalake.gamma_feature_lake import _plan_feature_table, compute_index_deltas, update_index

        try:
            async with self._index_lock:
                self._check_failure()
                deltas = await remote(compute_index_deltas, max_retries=0)(self._lake, df)
                self._check_failure()
                _, maximum, minimum, length, columns, new, earliest, _ = deltas
                features, tables = self._features, self._tables
                current = self._lake._get_latest_feature_tables(columns, features)
                plans = []
                for (source,), _ in current.group_by("table_addr", maintain_order=True):
                    target, publication = _plan_feature_table(
                        self._lake,
                        source,
                        columns,
                        length,
                        new.height,
                        minimum,
                        feature_metadata=features,
                        table_metadata=tables,
                        **options,
                    )
                    plans.append(
                        _WritePlan(
                            source=source,
                            target=target,
                            feature_metadata=features,
                            table_metadata=tables,
                            publication=publication,
                            options=options,
                        )
                    )
                alignment = self._lake._get_tables_to_update(
                    columns,
                    earliest,
                    feature_metadata=features,
                    table_metadata=tables,
                )
                # No feature IO or metadata publication is awaited in this lane.
                if not new.is_empty():
                    await remote(update_index, max_retries=0)(self._lake, new)
                self._check_failure()
                feature_tasks = []
                tasks = []
                delta_ref = require_ray().put(deltas)
                for plan in plans:
                    addresses = {plan.target}
                    if plan.source is not None:
                        addresses.add(plan.source)
                    dependencies = [self._tails[addr] for addr in sorted(addresses) if addr in self._tails]
                    task = remote(_write_features, max_retries=0)(self._lake, plan, delta_ref, *dependencies)
                    for addr in addresses:
                        self._tails[addr] = task
                    feature_tasks.append(task)
                    if plan.publication is not None:
                        self._features = pl.concat([self._features, plan.publication], how="diagonal_relaxed")
                    self._tables = pl.concat(
                        [self._tables, pl.DataFrame({"table_addr": [plan.target], "last_updated": [maximum]})],
                        how="diagonal_relaxed",
                    )
                for row in alignment.iter_rows(named=True):
                    addr = row["table_addr"]
                    dependencies = [self._tails[addr]] if addr in self._tails else []
                    task = remote(_align, max_retries=0)(self._lake, new, row, minimum, *dependencies)
                    self._tails[addr] = task
                    tasks.append(task)
            results = await asyncio.gather(*feature_tasks, *tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            async with self._publication_lock:
                self._check_failure()
                await remote(_publish, max_retries=0)(self._lake, results[: len(feature_tasks)])
            completed = set(feature_tasks + tasks)
            self._tails = {addr: tail for addr, tail in self._tails.items() if tail not in completed}
            return []
        except Exception as exc:
            logger.exception("Coordinated write failed; further admission is disabled")
            self._failure = self._failure or f"RayIOWriter stopped after a failed write; quiesce writers and repair the lake before recovery: {exc}"
            raise RuntimeError(self._failure) from exc
