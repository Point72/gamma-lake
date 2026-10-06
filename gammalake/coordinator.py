from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any

import polars as pl

from gammalake._ray import remote, require_ray
from gammalake._types import RayObjectReference

if TYPE_CHECKING:
    from gammalake.gamma_feature_lake import GammaFeatureLake

__all__ = ("WriteCoordinator",)

logger = logging.getLogger(__name__)


def _prepare_index(lake: GammaFeatureLake, frames: list[pl.DataFrame]) -> list[bool]:
    keys = pl.concat([frame.select(lake.sort_keys) for frame in frames]).unique()
    index = lake.index_frame().filter(pl.col(lake.primary_sort_key) <= keys[lake.primary_sort_key].max()).collect()
    new_only = [frame.select(lake.sort_keys).join(index, on=lake.sort_keys, how="inner").is_empty() for frame in frames]
    lake.add_index_rows(keys)
    return new_only


def _write_features(lake: GammaFeatureLake, df: pl.DataFrame, options: dict, index_was_new: bool) -> tuple[list, list]:
    return lake._add(df, **options, _defer_metadata=True, _index_was_new=index_was_new)


def _publish(lake: GammaFeatureLake, results: list[tuple[list, list]]) -> None:
    from gammalake.gamma_feature_lake import write_metadata

    write_metadata(lake, lake.table_metadata, *(row for tables, _ in results for row in tables))
    write_metadata(lake, lake.feature_metadata, *(row for _, features in results for row in features), schema_mode="merge")


class WriteCoordinator:
    """Ray actor coordinating batches of writes to one lake.

    Use :meth:`connect` to obtain the shared actor, or enable
    ``GammaFeatureLake.coordinate_writes`` to route ``add_*`` calls automatically.
    All writers must use the same Ray cluster and coordinator configuration.
    Readers are not isolated from in-progress batches.

    New index keys are unioned and aligned before independent physical feature
    tables are written concurrently. Metadata is published only after every
    feature task succeeds. Any batch failure stops the actor from admitting
    further writes; recovery requires quiescing workers and repairing the lake.
    """

    def __init__(self, lake: GammaFeatureLake):
        self._configuration = lake.model_dump(exclude={"io"})
        self._io_type = type(lake.io)
        self._io_configuration = vars(lake.io)
        self._lake = lake.model_copy(update={"coordinate_writes": False})
        self._lake._coordinated_worker = True
        self._pending: list[tuple[pl.DataFrame, dict, asyncio.Future]] = []
        self._draining = False
        self._failure: str | None = None

    @classmethod
    def connect(cls, lake: GammaFeatureLake) -> Any:
        """Get or create the cluster-wide actor for this lake.

        The actor survives individual submitting drivers, but not the cluster.
        It never restarts or retries tasks automatically after a failure.
        """
        ray = require_ray()
        if not lake.run_on_ray_cluster or not lake.coordinate_writes:
            raise ValueError("WriteCoordinator requires coordinate_writes=True and run_on_ray_cluster=True")
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

    async def submit(self, lake: GammaFeatureLake, df: pl.DataFrame | RayObjectReference[pl.DataFrame], **options) -> list:
        """Queue one ``_add`` request and wait for its metadata publication.

        Requests sharing feature names or physical tables execute in separate
        batches. ``df`` may be a top-level Ray object reference. Successful calls
        return an empty list; per-task results are internal to the coordinator.
        """
        from gammalake.gamma_feature_lake import preprocess_df

        if lake.model_dump(exclude={"io"}) != self._configuration or type(lake.io) is not self._io_type or vars(lake.io) != self._io_configuration:
            raise ValueError("All coordinated writers must use the same lake configuration and IO implementation")
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if not isinstance(df, pl.DataFrame):
            df = await df
            if self._failure is not None:
                raise RuntimeError(self._failure)
        df = preprocess_df(self._lake, df)
        if df.is_empty():
            raise ValueError("No valid index rows remain after preprocessing")
        if options.get("metadata") is not None:
            raise ValueError("Explicit metadata overrides are not supported by coordinated writes")
        options.setdefault("signal_type", "feature")
        future = asyncio.get_running_loop().create_future()
        self._pending.append((df, options, future))
        if not self._draining:
            self._draining = True
            asyncio.create_task(self._drain())
        return await asyncio.shield(future)

    async def submit_batch(self, lake: GammaFeatureLake, frames: list[pl.DataFrame | RayObjectReference[pl.DataFrame]], **options) -> list[list]:
        """Submit a group of frames together, using the same ``add_*`` options.

        This allows new index keys from all independent frames to be planned in
        one batch even when the caller has not started separate producer tasks.
        Conflicting frames retain submission order and execute in later batches.
        """
        return await asyncio.gather(*(self.submit(lake, frame, **options) for frame in frames))

    async def _drain(self) -> None:
        batch = []
        try:
            # Let already-arriving submissions join the same index-planning batch.
            await asyncio.sleep(0)
            while self._pending:
                metadata = self._lake.feature_metadata_frame().collect()
                resources: set[tuple[str, str]] = set()
                batch = []
                for frame, options, future in self._pending:
                    names = set(frame.columns) - set(self._lake.sort_keys)
                    tables = self._lake._get_latest_feature_tables(list(names), metadata)["table_addr"].drop_nulls()
                    requested = {("feature", name) for name in names} | {("table", table) for table in tables}
                    if requested & resources:
                        break
                    resources.update(requested)
                    batch.append((frame, options, future))
                del self._pending[: len(batch)]
                new_only = await remote(_prepare_index, num_cpus=0, max_retries=0)(self._lake, [frame for frame, _, _ in batch])
                tasks = [
                    remote(_write_features, num_cpus=0, max_retries=0)(self._lake, frame, options, index_was_new)
                    for (frame, options, _), index_was_new in zip(batch, new_only, strict=True)
                ]
                # Drain all writers even when one fails; never start another batch
                # while tasks from a failed batch can still modify the lake.
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                await remote(_publish, num_cpus=0, max_retries=0)(self._lake, results)
                for _, _, future in batch:
                    future.set_result([])
                batch = []
        except Exception as exc:
            logger.exception("Coordinated batch failed; further writes are disabled")
            self._failure = f"WriteCoordinator stopped after a failed batch; quiesce writers and repair the lake before recovery: {exc}"
            for _, _, future in [*batch, *self._pending]:
                if not future.done():
                    future.set_exception(RuntimeError(self._failure))
            self._pending.clear()
        finally:
            self._draining = False
