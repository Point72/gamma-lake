from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import polars as pl
from deltalake import DeltaTable

__all__ = (
    "FrameIO",
    "PolarsIO",
)


class FrameIO(ABC):
    """
    An abstraction for polars DataFrame reading/writing, allows for better interaction with a variety of storage solutions.
    """

    @abstractmethod
    def scan_delta(self, target: str, **kwargs: Any) -> pl.LazyFrame:
        """An implementation of scan_delta, which mimics the signature of polars.scan_delta

        Args:
            target: the string target pointing to a valid delta table target

        Returns:
            a polars LazyFrame

        """
        ...

    @abstractmethod
    def write_delta(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        target: str,
        *,
        delta_write_options: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        """An implementation of write_delta, which mimics the signature of polars.DataFrame.write_delta

        Args:
            df: a dataframe (either lazy or materialized) to write
            target: the target destination of the polars Dataframe
            delta_write_options: options passed through to the underlying Delta writer
                (e.g. WriterProperties, configuration, schema_mode, target_file_size).

        Returns:
            An Any type, typically None

        """
        ...

    @abstractmethod
    def annotate_table(self, table_addr: str, annotations: pl.DataFrame) -> None:
        """Persist comment-level annotations and string tags for features in one table."""
        ...

    @abstractmethod
    def describe_table(self, table_addr: str, feature_names: list[str]) -> pl.DataFrame:
        """Fetch comments and tags for the named features in one table."""
        ...

    @abstractmethod
    def is_deltatable(self, target: str) -> bool:
        """A convenience method, checking to see if a DeltaTable exists at a specified path"""
        ...

    @abstractmethod
    def merge_delta(
        self,
        df: pl.DataFrame,
        target: str,
        on: list[str],
        matched_predicate: str | None = None,
        when_matched_update: dict[str, str] | None = None,
        when_not_matched_insert: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Merge (upsert) a DataFrame into an existing Delta table.

        Matched rows (by ``on`` keys) are updated; unmatched source rows are inserted.

        Args:
            df: source dataframe for the merge
            target: path to the target delta table
            on: column names used to build the merge predicate
            matched_predicate: optional SQL predicate restricting which matched rows are updated
            when_matched_update: optional explicit column->expression mapping for matched updates.
                When None (default), all source columns are updated (``UPDATE SET *`` equivalent).
            when_not_matched_insert: optional explicit column->expression mapping for inserts.
                When None (default), all source columns are inserted (``INSERT *`` equivalent).

        Returns:
            merge metrics dict

        """
        ...

    @abstractmethod
    def restore_to_timestamp(self, target: str, timestamp: datetime) -> None:
        """Restore a Delta table to its state as of ``timestamp``.

        Args:
            target: Path of the Delta table to restore.
            timestamp: Timezone-aware cutoff; commits after it are reverted.

        """
        ...


class PolarsIO(FrameIO):
    """An implementation of FrameIO using native polars Delta table I/O."""

    def write_delta(self, df, target, *, delta_write_options=None, **kwargs):
        delta_write_options = delta_write_options or {}
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df.write_delta(target, delta_write_options=delta_write_options, **kwargs)

    def scan_delta(self, target, **kwargs):
        return pl.scan_delta(target, **kwargs)

    def restore_to_timestamp(self, target: str, timestamp: datetime) -> None:
        """Restore a Delta table to ``timestamp`` using deltalake."""
        DeltaTable(target).restore(timestamp)

    def annotate_table(self, table_addr: str, annotations: pl.DataFrame) -> None:
        """Persist comments and tags in Delta column metadata for one table."""
        dt = DeltaTable(table_addr)
        for row in annotations.iter_rows(named=True):
            metadata = {key: value for key, value in {"comment": row["comment"], "tags": json.dumps(row["tags"])}.items() if value is not None}
            dt.alter.set_column_metadata(row["feature_name"], metadata)

    def describe_table(self, table_addr: str, feature_names: list[str]) -> pl.DataFrame:
        """Read comments and tags from Delta column metadata for one table."""
        metadata = {field.name: dict(field.metadata) for field in DeltaTable(table_addr).schema().fields}
        return pl.DataFrame(
            {
                "feature_name": feature_names,
                "comment": [metadata.get(name, {}).get("comment") for name in feature_names],
                "tags": [json.loads(metadata.get(name, {}).get("tags", "[]")) for name in feature_names],
            },
            schema={"feature_name": pl.String, "comment": pl.String, "tags": pl.List(pl.String)},
        )

    def is_deltatable(self, target):
        from deltalake.exceptions import TableNotFoundError

        try:
            DeltaTable(target)
            return True
        except TableNotFoundError:
            return False
        except FileNotFoundError:
            return False
        except OSError as e:
            raise PermissionError("Please verify you can access the DeltaTables in your base path — your credentials may be incorrect.") from e

    def merge_delta(
        self,
        df,
        target,
        on,
        matched_predicate=None,
        when_matched_update=None,
        when_not_matched_insert=None,
        **kwargs,
    ):
        predicate = " AND ".join(f'target."{col}" = source."{col}"' for col in on)
        dt = DeltaTable(target)
        merger = dt.merge(source=df.to_arrow(), predicate=predicate, source_alias="source", target_alias="target", **kwargs)
        merger = (
            merger.when_matched_update(updates=when_matched_update, predicate=matched_predicate)
            if when_matched_update is not None
            else merger.when_matched_update_all(predicate=matched_predicate)
        )
        merger = (
            merger.when_not_matched_insert(updates=when_not_matched_insert)
            if when_not_matched_insert is not None
            else merger.when_not_matched_insert_all()
        )
        return merger.execute()
