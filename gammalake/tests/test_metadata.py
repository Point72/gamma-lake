from datetime import UTC, datetime

import polars as pl
import pyarrow as pa

import gammalake
from gammalake.metadata import (
    DEFAULT_INDEX_SCHEMA,
    Transaction,
    TransactionStatus,
    empty_feature_metadata,
    empty_table_metadata,
    empty_transaction_log,
    table_metadata_row,
    transaction_log_row,
)


def test_empty_feature_metadata_schema():
    assert dict(empty_feature_metadata().schema) == {
        "feature_name": pl.String,
        "version": pl.Int64,
        "table_addr": pl.String,
        "owner": pl.String,
        "signal_type": pl.String,
        "feature_params": pl.String,
    }


def test_empty_table_metadata_takes_the_primary_sort_key_type():
    frame = empty_table_metadata(DEFAULT_INDEX_SCHEMA.schema.field("timestamp").type)
    assert frame.is_empty()
    assert dict(frame.schema) == {"table_addr": pl.String, "last_updated": pl.Datetime("us", "UTC"), "update_timestamp": pl.Datetime("us", "UTC")}
    assert empty_table_metadata(pa.int64()).schema["last_updated"] == pl.Int64


def test_table_metadata_row_matches_the_empty_table_schema():
    row = table_metadata_row("addr", datetime(2024, 1, 1, tzinfo=UTC))
    empty = empty_table_metadata(pa.timestamp("us", tz="UTC"))
    assert row.height == 1
    assert row.columns == empty.columns
    assert row.cast(dict(empty.schema)).schema == empty.schema


def test_transaction_status_values_and_package_export():
    assert [status.value for status in TransactionStatus] == ["pending", "success", "failed", "rolled_back"]
    assert gammalake.TransactionStatus is TransactionStatus


def test_transaction_log_row_matches_the_empty_log_schema():
    transaction = Transaction("me", "feature")
    row = pl.from_arrow(transaction_log_row(transaction, TransactionStatus.PENDING))

    assert dict(row.schema) == dict(empty_transaction_log().schema)
    assert row.row(0, named=True) | {"updated_at": None} == {
        "transaction_id": transaction.transaction_id,
        "status": "pending",
        "owner": "me",
        "signal_type": "feature",
        "started_at": transaction.started_at,
        "updated_at": None,
        "error": None,
    }


def test_transaction_log_row_truncates_long_errors():
    row = transaction_log_row(Transaction("me", "feature"), TransactionStatus.FAILED, "x" * 10_000)
    assert len(row["error"][0].as_py()) == 2000


def test_transactions_have_unique_ids():
    assert Transaction("me", "feature").transaction_id != Transaction("me", "feature").transaction_id
