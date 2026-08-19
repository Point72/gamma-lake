from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

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
                (e.g. WriterProperties, schema_mode, target_file_size).

        Returns:
            An Any type, typically None

        """
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


class PolarsIO(FrameIO):
    """An implementation of FrameIO using native polars Delta table I/O."""

    def write_delta(self, df, target, *, delta_write_options=None, **kwargs):
        delta_write_options = delta_write_options or {}
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df.write_delta(target, delta_write_options=delta_write_options, **kwargs)

    def scan_delta(self, target, **kwargs):
        return pl.scan_delta(target, **kwargs)

    def is_deltatable(self, target):
        from deltalake import DeltaTable
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
        from deltalake import DeltaTable

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
