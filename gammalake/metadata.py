"""Definitions of the Delta tables that Gamma Lake keeps next to the feature tables.

This module owns what each system table looks like: its schema, how an empty table is built, and how rows for it are
constructed. ``GammaFeatureLake`` decides when to create and write them.

- ``index``: the sort-key tuples of every row. Its schema is supplied by the user; :data:`DEFAULT_INDEX_SCHEMA` is the default.
- ``feature_metadata``: one row per feature version and the physical table that holds it.
- ``table_metadata``: one row per update of a physical table.
- ``transaction_log``: an append-only event log of every tracked write. See :class:`TransactionStatus`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import polars as pl
import pyarrow as pa
from ccflow import ArrowSchema

DEFAULT_INDEX_SCHEMA = ArrowSchema.make(
    pa.schema(
        [
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("symbol", pa.string()),
        ]
    )
)

FEATURE_METADATA_SCHEMA = pa.schema(
    [
        ("feature_name", pa.string()),
        ("version", pa.int64()),
        ("table_addr", pa.string()),
        ("owner", pa.string()),
        ("signal_type", pa.string()),
        ("feature_params", pa.string()),
    ]
)

TRANSACTION_LOG_SCHEMA = pa.schema(
    [
        ("transaction_id", pa.string()),
        ("status", pa.string()),
        ("owner", pa.string()),
        ("signal_type", pa.string()),
        ("started_at", pa.timestamp("us", tz="UTC")),
        ("updated_at", pa.timestamp("us", tz="UTC")),
        ("error", pa.string()),
    ]
)

_MAX_LOGGED_ERROR_LENGTH = 2000


def table_metadata_schema(primary_sort_key_type: pa.DataType) -> pa.Schema:
    """The ``table_metadata`` schema; ``last_updated`` takes the type of the index's primary sort key."""
    return pa.schema(
        [
            ("table_addr", pa.string()),
            ("last_updated", primary_sort_key_type),
            ("update_timestamp", pa.timestamp("us", tz="UTC")),
        ]
    )


def empty_feature_metadata() -> pl.DataFrame:
    return pl.from_arrow(FEATURE_METADATA_SCHEMA.empty_table())


def empty_table_metadata(primary_sort_key_type: pa.DataType) -> pl.DataFrame:
    return pl.from_arrow(table_metadata_schema(primary_sort_key_type).empty_table())


def empty_transaction_log() -> pl.DataFrame:
    return pl.from_arrow(TRANSACTION_LOG_SCHEMA.empty_table())


def table_metadata_row(table_addr: str, last_updated) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "table_addr": [table_addr],
            "last_updated": [last_updated],
            "update_timestamp": [datetime.now(UTC)],
        }
    )


class TransactionStatus(StrEnum):
    """Lifecycle states recorded in the transaction log for each write."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class Transaction:
    """Identity of one tracked write; ``started_at`` is the cutoff a rollback restores to."""

    owner: str
    signal_type: str
    transaction_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def transaction_log_row(transaction: Transaction, status: TransactionStatus, error: str | None = None) -> pa.Table:
    """One ``transaction_log`` row recording ``status`` for ``transaction``; ``error`` is truncated to a bounded length."""
    return pa.Table.from_pylist(
        [
            {
                "transaction_id": transaction.transaction_id,
                "status": str(status),
                "owner": transaction.owner,
                "signal_type": transaction.signal_type,
                "started_at": transaction.started_at,
                "updated_at": datetime.now(UTC),
                "error": error[:_MAX_LOGGED_ERROR_LENGTH] if error is not None else None,
            }
        ],
        schema=TRANSACTION_LOG_SCHEMA,
    )
