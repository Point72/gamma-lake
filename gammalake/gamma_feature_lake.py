"""
A feature store implementation based on DeltaLake and ray, which focuses on supporting a sortable index on a polars frame and efficient appends/reads.

Updates are 3-step process as follows:

Step 1: Use rows existing in both the existing index and input frame, rows in the index but missing from the input frame, and rows in the input frame which are not yet in the index.
  If there are no common rows between the index and input frame, this is strictly an append operation to the proper feature table(s) and an insertion into the index.
  In the case of overlaps, we merge data in-place into the existing DeltaTable using Delta Lake's merge (upsert) operation. No new table is created. We insert null values in the case
  where there is nonempty overlap, but rows present in the index which are not found in the input table. These operations can be performed in parallel across the set of DeltaTables.

    Processes step 1 of the Gamma Lake add operation for new input frames. Index modification happens outside this function, in parallel, and only once.

    We consider three types of tables to add to out Feature Store:
    - Rows in both the index table AND input frame
    - Rows in in the index table, but not the input frame
    - Rows in the input frame, but not in the index table

    When no rows from the input table are found in the index, we append rows into the provided delta table address,
    and perform updates to the feature/table metadata tables.

    .. code-block::

        Existing Table                    Input Table               Modified Existing Table
        +-------------------+           +-------------------+           +-------------------+
        | Index   | Value   |           | Index  | Value    |           | Index  | Value    |
        +-------------------+           +-------------------+           +-------------------+
        | I1      | V1      |           |        |          |           | I1     | V1       |
        | I2      | V2      |           |        |          |           | I2     | V2       |
        | I3      | V3      |           |        |          |           | I3     | V3       |
        | I4      | V4      |           |        |          |    =>     | I4     | V4       |
        | I5      | V5      |           |        |          |           | I5     | V5       |
        | I6      | V6      |           |        |          |           | I6     | V6       |
        +-------------------+           | I7     | V7'      |           | I7     | V7'      |
                                        | I8     | V8'      |           | I8     | V8'      |
                                        | I9     | V9'      |           | I9     | V9'      |
                                        +-------------------+           +-------------------+

    When we have input rows where at least some already exist in the input frame, we proceed as follows:

    If the input rows overlapping with the existing index are before the target table's "latest_update" timestamp,
    we merge the intersection and alignment rows directly into the existing DeltaTable. Overlapping rows are updated
    in-place; alignment rows (index rows not present in the input, with null feature values) are inserted. In the
    example below, assume that the existing index at the processing of the input table ranges from I1 to I6.


    .. code-block::

          Existing Table                     Input Table                  Modified Existing Table (new Delta version)
        +-------------------+           +-------------------+           +-------------------+
        | Index  | Value    |           | Index  | Value    |           | Index  | Value    |
        +-------------------+           +-------------------+           +-------------------+
        | I1     | V1       |           |        |          |           | I1     | V1       |
        | I2     | V2       |           |        |          |           | I2     | V2       |
        | I3     | V3       |           | I3     | V3'      |           | I3     | V3'      |
        | I4     | V4       |           |        |          |    =>     | I4     | null     |
        | I5     | V5       |           | I5     | V5'      |           | I5     | V5'      |
        | I6     | V6       |           |        |          |           | I6     | null     |
        +-------------------+           | I7     | V7'      |           | I7     | V7'      |
                                        | I8     | V8'      |           | I8     | V8'      |
                                        | I9     | V9'      |           | I9     | V9'      |
                                        +-------------------+           +-------------------+

    Otherwise, we simply append the new rows and "pad" missing index values between the latest_update timestamp of the existing table and input table. In the
    example below, assume the existing index at the processing of the input table ranges from I1 to I9.

    .. code-block::

        Existing Table                     Input Table               Modified Existing Table
        +-------------------+           +-------------------+           +-------------------+
        | Index  | Value    |           | Index  | Value    |           | Index  | Value    |
        +-------------------+           +-------------------+           +-------------------+
        | I1     | V1       |           |        |          |           | I1     | V1       |
        | I2     | V2       |           |        |          |           | I2     | V2       |
        | I3     | V3       |           |        |          |           | I3     | V3       |
        | I4     | V4       |           |        |          |    =>     | I4     | V4       |
        | I5     | V5       |           |        |          |           | I5     | V5       |
        | I6     | V6       |           |        |          |           | I6     | V6       |
        +-------------------+           | I7     | V7'      |           | I7     | V7'      |
                                        |        |          |           | I8     | null     |
                                        | I9     | V9'      |           | I9     | V9'      |
                                        +-------------------+           +-------------------+


Step 2: In the case of overlaps between the index and input table we must modify tables which are updated beyond the earliest rows in the intersection to preserve feature table alignment.
  This step identifies the qualified tables (all feature tables which contain no features found in the input table).
  This preserves index alignment across all our delta tables, and we can do this in parallel.

    In step 2, we append new index values (ie, with null feature values) to all applicable tables - tables which were not updated in step 1, which we can
    identify in parallel and perform updates to simultaneously.

    Updated index values for tables which hold no features in common with the input table will only update up to the latest_update preexisting in that table.

    Example:

    .. code-block::

           Feature Table 1                 Feature Table 2                  Input Table                    Feature Table 1                 Feature Table 2
        +-------------------+           +-------------------+           +-------------------+           +-------------------+           +-------------------+
        | Index  | Feature 1|           | Index  | Feature 2|           | Index  | Feature 2|           | Index  | Value    |           | Index  | Value    |
        +-------------------+           +-------------------+           +-------------------+           +-------------------+           +-------------------+
        | I1     | V1_1     |           | I1     | V1_2     |           | I1     |          |           | I1     | V1_1     |           | I1     | V1_2     |
        |        |          |           |        |          |           | I2     | V2'      |           | I2     | null     |           | I2     | V2'      |
        |        |          |           |        |          |           | I3     | V3'      |    ==>    | I3     | null     |           | I3     | V3'      |
        | I4     | V4_1     |           | I4     | V4_2     |           | I4     |          |           | I4     | V4_1     |           | I4     | V4_2     |
        |        |          |           |        |          |           | I5     | V5'      |           | I5     | null     |           | I5     | V5'      |
        | I6     | V6_1     |           | I6     | V6_2     |           | I6     |          |           | I6     | V6_1     |           | I6     | V6_2     |
        |        |          |           |        |          |           | I7     | V7'      |           |        |          |           | I7     | V7'      |
        +-------------------+           +-------------------+           +-------------------+           +-------------------+           +-------------------+

Step 3: Append the results of an anti-join between the new data and the index back into the master index. This can also be done in parallel.

Two overlap modes are supported via the ``overlap_mode`` parameter on :meth:`add_features`,
:meth:`add_targets`, :meth:`add_as_of_features`, and :meth:`add_sparse_features`:

- ``"copy"`` *(default, backwards-compatible)*: On overlap, a brand-new Delta table is created
  and ``feature_metadata`` records the new ``table_addr``.
- ``"merge"``: On overlap, the existing Delta table is updated in-place via a merge (upsert).
  All reads always use the latest physical version of the table.
"""

import io
import json
import os
import traceback
import uuid
from datetime import UTC, datetime
from functools import singledispatchmethod, wraps
from typing import Literal

import polars as pl
import pyarrow as pa
import ray
from ccflow import ArrowSchema, BaseModel
from ccflow.exttypes.polars import PolarsExpression
from deltalake import WriterProperties
from deltalake.writer.properties import Compression
from packaging import version
from pydantic import ConfigDict, Field, model_validator

from gammalake._telemetry import trace
from gammalake._topo import with_columns_topo
from gammalake._types import RayObjectReference
from gammalake.abstract import (
    BaseFeatureLake,
    Comparable,
    MissingFeaturesException,
    MissingOrMisregisteredSignalsException,
    UninitializedDeltaLakeException,
)
from gammalake.io import FrameIO, PolarsIO

__all__ = ("GammaFeatureLake", "align_feature_tables", "update_feature_tables", "update_index")

_DEFAULT_INDEX_SCHEMA = ArrowSchema.make(
    pa.schema(
        [
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("symbol", pa.string()),
        ]
    )
)


def _get_timestamp():
    return pl.lit(datetime.now(UTC))


def preprocess_df(feature_store, df):
    if len(df) == 0:
        raise ValueError("Provided dataframe has no rows!")
    if df.select(pl.struct(*feature_store.sort_keys).is_duplicated().any()).item():
        secondary_keys = [k for k in feature_store.sort_keys if k != feature_store.primary_sort_key]
        first_distinct_expr = pl.col(feature_store.primary_sort_key).is_first_distinct()
        if secondary_keys:
            first_distinct_expr = first_distinct_expr.over(*secondary_keys)
        df = df.lazy().filter(first_distinct_expr).collect(engine="streaming")
    return df.filter(pl.all_horizontal(pl.col(key).is_not_null() for key in feature_store.sort_keys))


def update_feature_tables(
    feature_store: "GammaFeatureLake",
    table_addr: str | None,
    input_length: int,
    inner_join_ref: RayObjectReference[pl.DataFrame],
    new_index_rows_ref: RayObjectReference[pl.DataFrame],
    missing_index_rows_ref: RayObjectReference[pl.DataFrame],
    input_comparable_max: Comparable,
    input_comparable_min: Comparable,
    columns: list[str],
    owner: str,
    signal_type: str,
    overlap_mode: str = "copy",
    feature_params: dict | None = None,
) -> None:
    """
    Processes step 1 of the Gamma Lake add operation for new input frames. Index modification happens outside this function, in parallel, and only once.

    This function processes the following three types of tables for a Gamma Lake feature store:
    - Rows in both the index table AND input frame
    - Rows in in the index table, but not the input frame
    - Rows in the input frame, but not in the index table

    When no rows from the input table are found in the index, we append rows into the provided delta table address,
    and perform updates to the feature/table metadata tables::

           Existing Table                    Input Table               Modified Existing Table
        +-------------------+           +-------------------+           +-------------------+
        | Index   | Value   |           | Index  | Value    |           | Index  | Value    |
        +-------------------+           +-------------------+           +-------------------+
        | I1      | V1      |           |        |          |           | I1     | V1       |
        | I2      | V2      |           |        |          |           | I2     | V2       |
        | I3      | V3      |           |        |          |           | I3     | V3       |
        | I4      | V4      |           |        |          |    =>     | I4     | V4       |
        | I5      | V5      |           |        |          |           | I5     | V5       |
        | I6      | V6      |           |        |          |           | I6     | V6       |
        +-------------------+           | I7     | V7'      |           | I7     | V7'      |
                                        | I8     | V8'      |           | I8     | V8'      |
                                        | I9     | V9'      |           | I9     | V9'      |
                                        +-------------------+           +-------------------+


    When we have input rows where at least some already exist in the input frame, we proceed as follows:

    If the input rows overlaping with the existing index are before the target table's "latest_update" timestamp,
    we filter the target table and append the intersection and missing rows (where applicable) to a new table. In the example below,
    assume that the existing index at the processing of the input table ranges from I1 to I6::

          Existing Table                     Input Table                  New Feature Table
        +-------------------+           +-------------------+           +-------------------+
        | Index  | Value    |           | Index  | Value    |           | Index  | Value    |
        +-------------------+           +-------------------+           +-------------------+
        | I1     | V1       |           |        |          |           | I1     | V1       |
        | I2     | V2       |           |        |          |           | I2     | V2       |
        | I3     | V3       |           | I3     | V3'      |           | I3     | V3'      |
        | I4     | V4       |           |        |          |    =>     | I4     | null     |
        | I5     | V5       |           | I5     | V5'      |           | I5     | V5'      |
        | I6     | V6       |           |        |          |           | I6     | null     |
        +-------------------+           | I7     | V7'      |           | I7     | V7'      |
                                        | I8     | V8'      |           | I8     | V8'      |
                                        | I9     | V9'      |           | I9     | V9'      |
                                        +-------------------+           +-------------------+

    Otherwise, we simply append the new rows and "pad" missing index values between the latest_update timestamp of the existing table and input table. In the
    example below, assume the existing index at the processing of the input table ranges from I1 to I9::

          Existing Table                     Input Table               Modified Existing Table
        +-------------------+           +-------------------+           +-------------------+
        | Index  | Value    |           | Index  | Value    |           | Index  | Value    |
        +-------------------+           +-------------------+           +-------------------+
        | I1     | V1       |           |        |          |           | I1     | V1       |
        | I2     | V2       |           |        |          |           | I2     | V2       |
        | I3     | V3       |           |        |          |           | I3     | V3       |
        | I4     | V4       |           |        |          |    =>     | I4     | V4       |
        | I5     | V5       |           |        |          |           | I5     | V5       |
        | I6     | V6       |           |        |          |           | I6     | V6       |
        +-------------------+           | I7     | V7'      |           | I7     | V7'      |
                                        |        |          |           | I8     | null     |
                                        | I9     | V9'      |           | I9     | V9'      |
                                        +-------------------+           +-------------------+

    """
    try:
        if table_addr is None:
            # These features have never been seen before. Add them to a new table, address the new/missing index rows where applicable, and return.
            # sparse_feature tables never receive null sentinel rows — only rows with actual values are stored.
            feature_df = feature_store._get_latest_feature_tables(columns).filter(pl.col("table_addr").is_null())
            table_addr, new_feature_metadata_row = feature_store._write_new_feature_table(feature_df, owner, signal_type, feature_params)
            rows_to_write = (
                pl.concat([inner_join_ref, new_index_rows_ref], how="diagonal")
                if signal_type == "sparse_feature"
                else pl.concat([inner_join_ref, new_index_rows_ref, missing_index_rows_ref], how="diagonal")
            )
            feature_store.io.write_delta(
                rows_to_write.lazy().sort(feature_store.sort_keys),
                feature_store.get_path(table_addr),
                mode="append",
                delta_write_options={
                    "writer_properties": WriterProperties(compression=feature_store.compression),
                    "target_file_size": feature_store.target_file_size,
                },
            )
            feature_store.io.write_delta(
                pl.DataFrame().with_columns(
                    table_addr=pl.lit(table_addr), last_updated=pl.lit(input_comparable_max), update_timestamp=_get_timestamp()
                ),
                feature_store.table_metadata,
                mode="append",
                delta_write_options={"writer_properties": WriterProperties(compression=feature_store.compression)},
            )
            # Write feature_metadata last so a failure in the data or table_metadata writes
            # does not leave a stale entry that would cause null last_updated on retry.
            feature_store.io.write_delta(
                new_feature_metadata_row,
                feature_store.feature_metadata,
                mode="append",
                delta_write_options={"writer_properties": WriterProperties(compression=feature_store.compression), "schema_mode": "merge"},
            )

            return

        feature_df = feature_store._get_latest_feature_tables(columns).filter(pl.col("table_addr") == table_addr)
        columns = feature_store.sort_keys + feature_df["feature_name"].to_list()
        last_updated = feature_store._get_last_updated(table_addr)

        if input_length == new_index_rows_ref.height:
            # All input rows are new to the index. Write them to the feature table directly.
            feature_store.io.write_delta(
                new_index_rows_ref.lazy().sort(feature_store.sort_keys),
                feature_store.get_path(table_addr),
                mode="append",
                delta_write_options={
                    "writer_properties": WriterProperties(compression=feature_store.compression),
                    "schema_mode": "merge",
                    "target_file_size": feature_store.target_file_size,
                },
            )
        else:
            if input_comparable_min <= last_updated:
                writer_props = WriterProperties(compression=feature_store.compression)
                pk = feature_store.primary_sort_key
                next_rank = feature_df["version"].max() + 1
                source_path = feature_store.get_path(table_addr)

                if overlap_mode == "merge":
                    overlap_df = pl.concat(
                        [
                            inner_join_ref,
                            new_index_rows_ref,
                            missing_index_rows_ref.filter((pl.col(pk) >= input_comparable_min) & (pl.col(pk) <= input_comparable_max)),
                        ],
                        how="diagonal",
                    ).select(columns)
                    feature_store.io.merge_delta(overlap_df, source_path, on=feature_store.sort_keys, writer_properties=writer_props)
                else:
                    # copy mode: freeze old rows into the current table, write everything to a new address
                    new_index_rows_slice = new_index_rows_ref.filter(pl.col(pk) <= last_updated)
                    if new_index_rows_slice.height > 0:
                        feature_store.io.write_delta(
                            new_index_rows_slice.lazy().sort(feature_store.sort_keys),
                            source_path,
                            mode="append",
                            delta_write_options={
                                "writer_properties": writer_props,
                                "schema_mode": "merge",
                                "target_file_size": feature_store.target_file_size,
                            },
                        )
                    old_features = feature_store.io.scan_delta(source_path).select(columns).filter(pl.col(pk) < input_comparable_min).collect()
                    alignment_rows = (
                        missing_index_rows_ref.filter(pl.col(pk) >= last_updated)
                        if (old_features.height > 0 and signal_type not in ("as_of_feature", "sparse_feature"))
                        else pl.DataFrame()
                    )
                    new_rows = pl.concat([inner_join_ref, new_index_rows_ref, alignment_rows], how="diagonal").select(columns)
                    table_addr = feature_store.gen_table_addr()
                    feature_store.io.write_delta(
                        pl.concat([old_features, new_rows], how="vertical_relaxed").lazy().sort(feature_store.sort_keys),
                        feature_store.get_path(table_addr),
                        mode="append",
                        delta_write_options={"writer_properties": writer_props, "target_file_size": feature_store.target_file_size},
                    )

                feature_store.io.write_delta(
                    feature_df.with_columns(
                        version=pl.lit(next_rank, dtype=pl.Int64),
                        table_addr=pl.lit(table_addr),
                        owner=pl.lit(owner),
                        signal_type=pl.lit(signal_type),
                        feature_params=pl.lit(json.dumps(feature_params)) if feature_params is not None else pl.col("feature_params"),
                    ),
                    feature_store.feature_metadata,
                    mode="append",
                    delta_write_options={"writer_properties": writer_props, "schema_mode": "merge"},
                )

            else:
                feature_store.io.write_delta(
                    pl.concat(
                        [inner_join_ref, new_index_rows_ref, missing_index_rows_ref.filter(pl.col(feature_store.primary_sort_key) > last_updated)],
                        how="diagonal",
                    )
                    .select(columns)
                    .lazy()
                    .sort(feature_store.sort_keys),
                    feature_store.get_path(table_addr),
                    mode="append",
                    delta_write_options={
                        "writer_properties": WriterProperties(compression=feature_store.compression),
                        "target_file_size": feature_store.target_file_size,
                    },
                )

        feature_store.io.write_delta(
            pl.DataFrame().with_columns(
                table_addr=pl.lit(table_addr),
                last_updated=pl.lit(input_comparable_max),
                update_timestamp=_get_timestamp(),
            ),
            feature_store.table_metadata,
            mode="append",
            delta_write_options={"writer_properties": WriterProperties(compression=feature_store.compression)},
        )

    except Exception as e:
        print(columns, e, traceback.format_exc())
        raise


def align_feature_tables(feature_store, new_index_rows_ref, row, input_comparable_min):
    """
    In step 2, we append new index values (ie, with null feature values) to all applicable tables - tables which were not updated in step 1, which we can
    identify in parallel and perform updates to simultaneously.

    Updated index values for tables which hold no features in common with the input table will only update up to the latest_update preexisting in that table.
    as_of_feature tables are excluded upstream in _get_tables_to_update and never reach this function.
    sparse_feature tables are likewise excluded upstream and never reach this function.

    Args:
        feature_store: the GammaFeatureLake object in question
        new_index_rows_ref: A reference to index rows which were previously not in the index
        row: a row from ``table_metadata_frame`` (fields: ``table_addr``, ``last_updated``)
        input_comparable_min: The minimal sorted value of the input data

    Returns: None
    """
    if input_comparable_min >= row["last_updated"]:
        return

    filtered_antijoin = new_index_rows_ref.select(feature_store.sort_keys).filter(pl.col(feature_store.primary_sort_key) <= row["last_updated"])

    if filtered_antijoin.height > 0:
        table_path = feature_store.get_path(row["table_addr"])
        feature_store.io.write_delta(
            filtered_antijoin.lazy().sort(feature_store.sort_keys),
            table_path,
            mode="append",
            delta_write_options={
                "writer_properties": WriterProperties(compression=feature_store.compression),
                "schema_mode": "merge",
                "target_file_size": feature_store.target_file_size,
            },
        )


def update_index(feature_store, new_index_rows_ref):
    """
    Step 3 is a single-time index update based on new index rows provided by the input table.
    """
    if new_index_rows_ref.height > 0:
        feature_store.io.write_delta(
            new_index_rows_ref.select(feature_store.sort_keys).lazy().sort(feature_store.sort_keys),
            feature_store.index,
            mode="append",
            delta_write_options={
                "writer_properties": WriterProperties(compression=feature_store.compression),
                "schema_mode": "merge",
                "target_file_size": feature_store.target_file_size,
            },
        )


def inner_join_compute(feature_store, df):
    """
    Computes an inner join between a feature store index and an input frame, and returns metadata about the input frame.
    """
    inner_join_ref = (
        feature_store.index_frame()
        .filter(pl.col(feature_store.primary_sort_key) >= df[feature_store.primary_sort_key].min())
        .collect()
        .join(df, how="inner", on=feature_store.sort_keys)
        .select(df.columns)
    )
    input_comparable_max = df[feature_store.primary_sort_key].max()
    input_comparable_min = df[feature_store.primary_sort_key].min()
    return inner_join_ref, input_comparable_max, input_comparable_min, df.height, df.columns


def new_index_rows(feature_store, df):
    """Returns rows in the input frame which are missing from the index."""
    new_index_rows = df.join(
        feature_store.index_frame()
        .filter(
            (pl.col(feature_store.primary_sort_key) >= df[feature_store.primary_sort_key].min())
            & (pl.col(feature_store.primary_sort_key) <= df[feature_store.primary_sort_key].max())
        )
        .collect(),
        how="anti",
        on=feature_store.sort_keys,
    )
    return new_index_rows, new_index_rows[feature_store.primary_sort_key].min()


def missing_input_rows(feature_store, df):
    """Returns rows in the index which are missing from the input frame."""
    return (
        feature_store.index_frame()
        .filter(pl.col(feature_store.primary_sort_key) <= df[feature_store.primary_sort_key].max())
        .collect()
        .join(df, how="anti", on=feature_store.sort_keys)
    )


def read_table(feature_store, table_addr, features, start, end, signal_type, debug=False, materialize=False) -> pl.LazyFrame | pl.DataFrame:
    """Reads a path for a feature store, and sorts it."""
    flt = pl.col(feature_store.primary_sort_key) >= start if start is not None else pl.lit(True)
    flt &= pl.col(feature_store.primary_sort_key) <= end if end is not None else pl.lit(True)
    table = feature_store.io.scan_delta(feature_store.get_path(table_addr)).filter(flt).select(feature_store.sort_keys + features)

    if signal_type == "as_of_feature":
        by_keys = [key for key in feature_store.sort_keys if key != feature_store.primary_sort_key]
        table = (
            feature_store.index_frame()
            .filter(flt)
            .sort(by_keys + [feature_store.primary_sort_key])
            .join_asof(
                table.sort(by_keys + [feature_store.primary_sort_key]),
                on=feature_store.primary_sort_key,
                by=by_keys,
                **json.loads(feature_store.feature_metadata_frame().filter(pl.col("table_addr") == table_addr).collect()["feature_params"].first()),
            )
        )

    elif signal_type == "sparse_feature":
        index_filter = pl.col(feature_store.primary_sort_key) >= start if start is not None else pl.lit(True)
        sparse_max = table.select(pl.col(feature_store.primary_sort_key).max()).collect().item()
        if sparse_max is not None:
            index_filter &= pl.col(feature_store.primary_sort_key) <= sparse_max
        left_lf = feature_store.index_frame().filter(index_filter).sort(feature_store.sort_keys)
        right_lf = table.sort(feature_store.sort_keys)
        if version.parse(pl.__version__) >= version.parse("1.37.0"):
            left_lf = left_lf.set_sorted(feature_store.primary_sort_key)
            right_lf = right_lf.set_sorted(feature_store.primary_sort_key)
        table = left_lf.join(right_lf, on=feature_store.sort_keys, how="left", coalesce=True).collect(engine="streaming").lazy()

    result = table.sort(feature_store.sort_keys).rename({key: f"{key}_{table_addr}" for key in feature_store.sort_keys})
    return result.collect() if materialize else result


def _latest_per_feature(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per feature_name — the row with the highest version (nulls last)."""
    return df.sort("version", descending=True, nulls_last=True).unique(subset=["feature_name"], keep="first")


class GammaFeatureLake(BaseFeatureLake, BaseModel):
    """
    A feature store implementation based on DeltaLake and ray, which focuses on supporting a sortable index on a polars frame and efficient appends/reads.

    Overlapping uploads merge in-place via Delta Lake's merge (upsert) operation. No new DeltaTable is created on overlap.
    """

    base_path: str = Field(
        description="A local or s3:// qualified path, identifying the root directory for storing the DeltaTables associated with this class."
    )
    feature_metadata_path: str = Field("feature_metadata", description="The name of the DeltaTable holding feature metadata.")
    table_metadata_path: str = Field("table_metadata", description="The name of the DeltaTable holding table metadata.")
    index_path: str = Field("index", description="The name of the DeltaTable holding the global index.")
    table_name_prefix: str = Field(
        "",
        description="Optional prefix prepended to every data table address generated by gen_table_addr(). "
        "Useful for integration tests to ensure each test's data tables are isolated and can be cleaned up by prefix.",
    )
    primary_sort_key: str = Field("timestamp", description="The name of the primary key in the index used for comparisons when inserting features.")
    compression: str = Field("zstd", description="The compression level of the write_delta methods in this class")
    target_file_size: int = Field(
        128 * 1024 * 1024,
        description="Target parquet file size in bytes when writing feature tables. Splitting large writes into "
        "multiple files of this size enables file-level min/max skipping on timestamp-filtered reads.",
    )
    run_on_ray_cluster: bool = Field(True, description="Toggles whether or not this class uses ray remote functions, or strictly local python calls")
    enable_runtime_computed_features: bool = Field(
        False,
        description="Enables runtime-computed features. Only enable this when feature metadata is trusted because Polars expressions may execute Python code.",
    )
    io: FrameIO = Field(default_factory=PolarsIO, description="An IO object, abstracting how/where polars frames are read from storage.")
    _is_initialized: bool = False
    _sort_keys: list[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check(self) -> "GammaFeatureLake":
        """
        Write the proper metadata holding DeltaTables if they do not exist upon creation of this object.
        """
        compression_levels = {i.name for i in Compression}
        if self.compression.upper() not in compression_levels:
            raise ValueError(f"Please provide a valid compression level: {compression_levels}\n Found: {self.compression}")

        self._is_initialized = self.io.is_deltatable(self.index)
        if self._is_initialized:
            self._sort_keys = self.io.scan_delta(self.index).collect_schema().names()
            if self.primary_sort_key not in self._sort_keys:
                raise ValueError(f"The provided primary sort key: {self.primary_sort_key} does not exist in the index keys: {self._sort_keys}")

        return self

    def ensure_deltalake_is_initialized(func):
        """A decorator that delegates to an overridable method, and verifies that the proper initialization steps took place"""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._is_initialized:
                raise UninitializedDeltaLakeException("Please call '.initialize(...)' on your feature store before using this method!")
            return func(self, *args, **kwargs)

        return wrapper

    @property
    def sort_keys(self) -> list[str]:
        """Returns the index table's columm schema."""
        return self._sort_keys

    @property
    def index(self) -> str:
        """Convenience method to find the correct path for this feature store's index table."""
        return os.path.join(self.base_path, self.index_path)

    @property
    def feature_metadata(self) -> str:
        """Convenience method to find the correct path for this feature store's feature metadata table"""
        return os.path.join(self.base_path, self.feature_metadata_path)

    @property
    def table_metadata(self) -> str:
        """Convenience method to find the correct path for this feature store's table metadata table"""
        return os.path.join(self.base_path, self.table_metadata_path)

    def _apply_filters_to_lazy_frame(self, df: pl.LazyFrame, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Apply optional filters to a lazyframe, and return it"""
        if start is not None:
            df = df.filter(pl.col(self.primary_sort_key) >= start)
        if end is not None:
            df = df.filter(pl.col(self.primary_sort_key) <= end)
        return df

    def table_metadata_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Convenience method to find the correct path for this feature store's table metadata table"""
        return self._apply_filters_to_lazy_frame(self.io.scan_delta(self.table_metadata), start, end)

    def feature_metadata_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Convenience method to find the correct path for this feature store's feature metadata table"""
        return self._apply_filters_to_lazy_frame(self.io.scan_delta(self.feature_metadata), start, end)

    def index_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Convenience method to find the correct path for this feature store's index table."""
        return self._apply_filters_to_lazy_frame(self.io.scan_delta(self.index), start, end)

    def get_path(self, addr) -> str:
        """Convenience method to join a feature store's base path to a provided table address"""
        return os.path.join(self.base_path, addr)

    def gen_table_addr(self) -> str:
        """Convenience method for generating new table addresses."""
        return f"{self.table_name_prefix}delta_{uuid.uuid4().hex[:8]}"

    def _ensure_runtime_computed_features_enabled(self) -> None:
        if not self.enable_runtime_computed_features:
            raise ValueError(
                "Runtime-computed features are disabled. Set enable_runtime_computed_features=True only when feature metadata is trusted."
            )

    def initialize(self, schema: ArrowSchema = _DEFAULT_INDEX_SCHEMA) -> "GammaFeatureLake":
        """
        Write the proper metadata holding DeltaTables with the provided index schema.
        """
        if self.primary_sort_key not in schema.schema.names:
            raise ValueError(f"The provided primary sort key: {self.primary_sort_key} does not exist in the schema keys: {schema.schema.names}")

        if not self.io.is_deltatable(self.index):
            self.io.write_delta(pl.from_arrow(pa.Table.from_pylist([], schema=schema.schema)), self.index)

        if not self.io.is_deltatable(self.feature_metadata):
            feature_metadata_frame = pl.DataFrame(
                [],
                schema={
                    "feature_name": pl.String,
                    "version": pl.Int64,
                    "table_addr": pl.String,
                    "owner": pl.String,
                    "signal_type": pl.String,
                    "feature_params": pl.String,
                },
            )
            self.io.write_delta(feature_metadata_frame, self.feature_metadata)

        if not self.io.is_deltatable(self.table_metadata):
            table_metadata_frame = pl.from_arrow(
                pa.Table.from_arrays(
                    [
                        pa.array([], type=pa.string()),
                        pa.array([], type=schema.schema[schema.schema.get_field_index(self.primary_sort_key)].type),
                        pa.array([], type=pa.timestamp("us", tz="UTC")),
                    ],
                    names=["table_addr", "last_updated", "update_timestamp"],
                )
            )
            self.io.write_delta(table_metadata_frame, self.table_metadata)

        self._is_initialized = True
        self._sort_keys = schema.schema.names
        return self

    def _get_correct_feature_tables(self, features: list[str]) -> pl.DataFrame:
        """Fetches the tables for the latest version of each feature provided in the input features list"""
        meta = self.feature_metadata_frame().collect()
        tables = _latest_per_feature(meta.filter(pl.col("feature_name").is_in(features)))

        # Verify that if runtime computed signal types are provided, we also ensure all required root signals are computed too.
        runtime_features = tables.filter(pl.col("signal_type") == "runtime_computed")
        if runtime_features.height > 0:
            self._ensure_runtime_computed_features_enabled()
            existing_root_feature_names = tables["feature_name"]
            missing_root_feature_names = []
            for expr in runtime_features["feature_params"]:
                for name in pl.Expr.deserialize(source=io.BytesIO(expr.encode()), format="json").meta.root_names():
                    if name not in existing_root_feature_names:
                        missing_root_feature_names.append(name)

            if len(missing_root_feature_names) > 0:
                missing_root_features = _latest_per_feature(meta.filter(pl.col("feature_name").is_in(missing_root_feature_names)))
                tables = pl.concat([tables, missing_root_features])

        if tables.height == 0:
            raise MissingFeaturesException(f"No FeatureTables exist for any of your input features: {features}")
        return tables

    def _load_features_from_tables(self, tables, start: Comparable | None = None, end: Comparable | None = None, debug: bool = False) -> pl.LazyFrame:
        frames = [
            read_table(self, table_addr, frame["feature_name"].to_list(), start, end, frame["signal_type"].first(), debug)
            for (table_addr,), frame in tables.filter(pl.col("signal_type") != "runtime_computed").group_by("table_addr")
        ]
        addresses = tables.filter(pl.col("signal_type") != "runtime_computed")["table_addr"].unique()
        root_features = (
            pl.concat(frames, how="horizontal")
            .lazy()
            .with_columns([pl.coalesce(pl.col([f"{key}_{addr}" for addr in addresses]).alias(key)) for key in self.sort_keys])
        )
        return self._compute_runtime_features(tables, root_features).select("*" if debug else tables["feature_name"].to_list() + self.sort_keys)

    def _load_features_from_tables_in_parallel(
        self, tables, start: Comparable | None = None, end: Comparable | None = None, debug: bool = False
    ) -> pl.LazyFrame:
        groups = [
            (table_addr, frame["feature_name"].to_list(), frame["signal_type"].first())
            for (table_addr,), frame in tables.filter(pl.col("signal_type") != "runtime_computed").group_by("table_addr")
        ]
        addresses = tables.filter(pl.col("signal_type") != "runtime_computed")["table_addr"].unique()
        table_addr_coalesce_statements = [pl.coalesce(pl.col([f"{key}_{addr}" for addr in addresses]).alias(key)) for key in self.sort_keys]

        table_load_refs = [
            self.switch(read_table)(self, table_addr, features, start, end, signal_type, debug, self.run_on_ray_cluster)
            for table_addr, features, signal_type in groups
        ]
        frames = []
        while table_load_refs:
            ready, table_load_refs = self.wait(table_load_refs)
            frames.append(self.get(ready[0]))

        root_features = pl.concat(frames, how="horizontal").lazy().with_columns(table_addr_coalesce_statements)
        return self._compute_runtime_features(tables, root_features).select("*" if debug else tables["feature_name"].to_list() + self.sort_keys)

    def _compute_runtime_features(self, feature_table, root_features):
        runtime_computed_features = feature_table.filter(pl.col("signal_type") == "runtime_computed")
        if runtime_computed_features.height > 0:
            self._ensure_runtime_computed_features_enabled()
        runtime_feature_expressions = [
            pl.Expr.deserialize(source=io.BytesIO(expr.encode()), format="json") for expr in runtime_computed_features["feature_params"]
        ]
        return with_columns_topo(root_features, runtime_feature_expressions)

    @singledispatchmethod
    @trace(always=True)
    @ensure_deltalake_is_initialized
    def read(self, *arg) -> pl.DataFrame:
        """Abstract base dispatch method for sequential reads of feature tables"""
        raise TypeError(
            "Invalid input types! Please pass a list of string feature names OR a filtered feature_metadata_table object, and optional start/end Comparables."
        )

    @read.register
    @trace(always=True)
    @ensure_deltalake_is_initialized
    def _(
        self,
        features: pl.DataFrame | None = None,
        start: Comparable | None = None,
        end: Comparable | None = None,
        materialized: bool = True,
    ) -> pl.DataFrame:
        duplicates = features.filter(features["feature_name"].is_duplicated())["feature_name"].unique().to_list()
        if duplicates:
            raise ValueError(
                f"feature_metadata DataFrame contains the same feature at multiple versions: {duplicates}. Make separate read() calls per version."
            )
        runtime_features = features.filter(pl.col("signal_type") == "runtime_computed")
        if runtime_features.height > 0:
            self._ensure_runtime_computed_features_enabled()
            missing_root_feature_names = [
                name
                for expr in runtime_features["feature_params"]
                for name in pl.Expr.deserialize(source=io.BytesIO(expr.encode()), format="json").meta.root_names()
                if name not in features["feature_name"]
            ]
            if missing_root_feature_names:
                features = pl.concat([features, self._get_correct_feature_tables(missing_root_feature_names)], how="diagonal")
        result = (
            self._load_features_from_tables(features, start, end)
            if not self.run_on_ray_cluster
            else self._load_features_from_tables_in_parallel(features, start, end)
        ).select(features["feature_name"].to_list() + self.sort_keys)
        return result.collect() if materialized else result

    @read.register
    @trace(always=True)
    @ensure_deltalake_is_initialized
    def _(
        self,
        features: list,
        targets: list | None = None,
        start: Comparable | None = None,
        end: Comparable | None = None,
        debug: bool = False,
        materialized: bool = True,
    ) -> pl.DataFrame:
        """
        Given a list of features, identifies the delta tables with the most up-to-date versions of those features,
        and reads/sorts/horizontally concatenates those tables.
        """
        targets = targets or []
        missing_targets = [target for target in targets if target not in self._get_target_column_names()]
        if len(missing_targets) > 0:
            raise MissingOrMisregisteredSignalsException(
                f"The following target values were not found to be registered as target signal types: {missing_targets}"
            )
        tables = self._get_correct_feature_tables(features + targets)
        if not self.run_on_ray_cluster:
            features = self._load_features_from_tables(tables, start, end, debug)
        else:
            features = self._load_features_from_tables_in_parallel(tables, start, end, debug)
        return features.collect() if materialized else features

    @ensure_deltalake_is_initialized
    def _get_latest_feature_tables(self, feature_columns):
        return _latest_per_feature(self.feature_metadata_frame().collect()).join(
            pl.DataFrame(pl.Series(list(set(feature_columns) - set(self.sort_keys))).alias("feature_name")), how="right", on="feature_name"
        )

    @ensure_deltalake_is_initialized
    def _write_new_feature_table(self, feature_df, owner, signal_type, feature_params) -> tuple[str, pl.DataFrame]:
        """Prepares a new table address and the corresponding feature_metadata row.

        Does NOT write to feature_metadata. Callers are responsible for writing the returned
        DataFrame to feature_metadata only after all data and table_metadata writes succeed,
        so that a failed feature write does not leave a stale feature_metadata entry.
        """
        table_addr = self.gen_table_addr()
        feature_df = feature_df.with_columns(
            table_addr=pl.lit(table_addr),
            owner=pl.lit(owner),
            signal_type=pl.lit(signal_type),
            feature_params=pl.lit(json.dumps(feature_params) if feature_params is not None else "{}"),
            version=pl.lit(0, dtype=pl.Int64),
        )
        return table_addr, feature_df

    @ensure_deltalake_is_initialized
    def _get_tables_to_update(self, feature_columns, min_value, metadata=None) -> pl.DataFrame:
        """
        We need to append antijoined rows between the input frame and index in the following cases:
          - A table has no overlap in columns with the input features AND its last_updated date is more recent than the oldest date in the antijoin.
          - A table has overlap in columns with the input features AND does not hold the latest versions of those features AND its last_updated date
            is more recent than the oldest date in the antijoin.

        as_of_feature tables are excluded: they are read via join_asof against the global index, which naturally
        carries forward the last known observation. Injecting null sentinels would corrupt that semantics.
        sparse_feature tables are excluded for the same reason: they are read via a left join against the index
        and must not receive null sentinel rows.
        """
        if min_value is None:
            return pl.DataFrame()
        do_not_update = self._get_latest_feature_tables(feature_columns).filter(pl.col("table_addr").is_not_null()) if metadata is None else metadata
        excluded_addrs = (
            self.feature_metadata_frame()
            .filter(pl.col("signal_type").is_in(["as_of_feature", "sparse_feature"]))
            .select("table_addr")
            .unique()
            .collect()
        )
        return (
            self.table_metadata_frame()
            .collect()
            .join(do_not_update, how="anti", on="table_addr")
            .join(excluded_addrs, how="anti", on="table_addr")
            .filter(pl.col("last_updated") >= min_value)
        )

    @ensure_deltalake_is_initialized
    def _get_target_column_names(self) -> pl.Series:
        """Returns the column names of uploaded target signals"""
        return self.feature_metadata_frame().filter(pl.col("signal_type") == "target").collect()["feature_name"].unique().to_list()

    @ensure_deltalake_is_initialized
    def _get_last_updated(self, table_addr):
        """Shorthand to find the last updated value for a given table address"""
        return (
            self.table_metadata_frame()
            .filter(pl.col("table_addr") == table_addr)
            .select(pl.col("last_updated").filter(pl.col("update_timestamp") == pl.col("update_timestamp").max()).first().alias("last_updated"))
            .collect()
            .item()
        )

    def switch(self, function, num_returns=1, **kwargs):
        """Returns a remote function OR local function depending on the run_on_ray_cluster parameter"""
        return function if not self.run_on_ray_cluster else ray.remote(num_returns=num_returns, **kwargs)(function).remote

    def get(self, potential_ref):
        """Wrap the getting of references (or actual objects) behind a switch"""
        return ray.get(potential_ref) if self.run_on_ray_cluster else potential_ref

    def wait(self, list_of_potential_refs, **kwargs):
        """Wrap waiting on ray references (or actual object) behind a switch"""
        if self.run_on_ray_cluster:
            return ray.wait(list_of_potential_refs, **kwargs)
        return [list_of_potential_refs.pop(0)], list_of_potential_refs

    @ensure_deltalake_is_initialized
    @trace(always=True)
    def _add(
        self,
        df: pl.DataFrame | ray.ObjectRef,
        signal_type,
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
        feature_params: dict | None = None,
        overlap_mode: str = "copy",
    ) -> list:
        """
        Onboards a new table of features to our FeatureStore.

        Args:
            overlap_mode: ``"copy"`` (default) creates a new DeltaTable on overlap, preserving the
                old table as an immutable snapshot.  ``"merge"`` upserts into the existing table in-place.
        """

        if isinstance(df, ray.ObjectRef) and not self.run_on_ray_cluster:
            raise ValueError("Invalid RayObjectRef input! This class is configured not to use remote functions.")

        df = self.switch(preprocess_df)(self, df)
        inner_join_ref, primary_sort_key_max, primary_sort_key_min, input_length, columns = self.switch(inner_join_compute, num_returns=5)(self, df)
        new_index_rows_ref, earliest_new_index_rows_ref = self.switch(new_index_rows, num_returns=2)(self, df)
        missing_index_rows_ref = self.switch(missing_input_rows)(self, df)
        feature_tables = self._get_latest_feature_tables(self.get(columns))
        tables_to_update = self._get_tables_to_update(
            self.get(columns),
            self.get(earliest_new_index_rows_ref),
            metadata if metadata is not None else feature_tables.filter(pl.col("table_addr").is_not_null()),
        )
        refs = []
        # Step 1: Append-only if no overlap; otherwise merge in-place into the existing DeltaTable (upsert).
        # New index rows are inserted; overlapping rows are updated. Alignment rows (index rows absent from the
        # input, with null feature values) are merged in to maintain index alignment across all tables.
        for (table_addr,), feature_df in feature_tables.group_by("table_addr"):
            refs.append(
                self.switch(update_feature_tables)(
                    feature_store=self,
                    table_addr=table_addr,
                    input_length=input_length,
                    inner_join_ref=inner_join_ref,
                    new_index_rows_ref=new_index_rows_ref,
                    missing_index_rows_ref=missing_index_rows_ref,
                    input_comparable_max=primary_sort_key_max,
                    input_comparable_min=primary_sort_key_min,
                    columns=columns,
                    owner=owner,
                    signal_type=signal_type,
                    feature_params=feature_params,
                    overlap_mode=overlap_mode,
                )
            )

        # Step 2: In the case of overlaps between the index and input table we must we must modify tables which are updated beyond the earliest rows in the intersection to preserve feature table alignment.
        # This step identifies the qualified tables (all feature tables which contain no features found in the input table, OR contain a non-latest version of a feature found in the input table).
        # This preserves index alignment across all our delta tables, and we can do this in parallel.
        for row in tables_to_update.iter_rows(named=True):
            refs.append(self.switch(align_feature_tables)(self, new_index_rows_ref, row, primary_sort_key_min))

        # Step 3: Append the results of an anti-join between the new data and the index back into the master index.
        # This can also be done in parallel.
        refs.append(self.switch(update_index)(self, new_index_rows_ref))
        return self.get(refs) if self.run_on_ray_cluster else refs

    @ensure_deltalake_is_initialized
    @trace(always=True)
    def add_targets(
        self,
        df: pl.DataFrame | ray.ObjectRef,
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
        overlap_mode: Literal["copy", "merge"] = "copy",
    ) -> list:
        """Calls _add for target signal_types"""
        return self._add(df=df, signal_type="target", owner=owner, metadata=metadata, overlap_mode=overlap_mode)

    @ensure_deltalake_is_initialized
    @trace(always=True)
    def add_features(
        self,
        df: pl.DataFrame | ray.ObjectRef,
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
        overlap_mode: Literal["copy", "merge"] = "copy",
    ) -> list:
        """Calls _add for feature signal_types"""
        return self._add(df=df, signal_type="feature", owner=owner, metadata=metadata, overlap_mode=overlap_mode)

    @ensure_deltalake_is_initialized
    @trace(always=True)
    def add_as_of_features(
        self,
        df: pl.DataFrame | ray.ObjectRef,
        params: dict,
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
        overlap_mode: Literal["copy", "merge"] = "copy",
    ) -> list:
        """Calls _add for as_of feature signal_types"""
        return self._add(df=df, signal_type="as_of_feature", owner=owner, metadata=metadata, feature_params=params, overlap_mode=overlap_mode)

    @ensure_deltalake_is_initialized
    @trace(always=True)
    def add_sparse_features(
        self,
        df: pl.DataFrame | ray.ObjectRef,
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
        overlap_mode: Literal["copy", "merge"] = "copy",
    ) -> list:
        """Add sparse features to this FeatureLake.

        Sparse features are stored without null sentinel rows — only rows with actual values
        are written to the physical Delta table.  At read time they are joined back to the
        global index via a left join using Polars' streaming sort-merge engine, so every index
        row is returned with ``null`` where the sparse feature has no observation.

        Use sparse features for data that is genuinely sparse (e.g. corporate actions, earnings
        events, issuer attributes) where storing a null for every missing index row would be
        wasteful.  For slowly-changing data where you want the last known value carried forward,
        use :meth:`add_as_of_features` instead.
        """
        return self._add(df=df, signal_type="sparse_feature", owner=owner, metadata=metadata, overlap_mode=overlap_mode)

    @ensure_deltalake_is_initialized
    @trace(always=True)
    def add_runtime_computed_features(self, exprs=list[PolarsExpression], owner: str = "missing_owner") -> None:
        """Add (and possibly increment the version) a serialized expression representing an adhoc runtime feature computation"""
        self._ensure_runtime_computed_features_enabled()
        for expr in exprs:
            if expr.meta.output_name() in expr.meta.root_names():
                raise ValueError(
                    f"Invalid input: the expression {expr} provides the same output name as a provided root name. Please alias your runtime features uniquely!"
                )
        all_output_names = {expr.meta.output_name() for expr in exprs}
        existing_feature_names = self.feature_metadata_frame().select("feature_name").collect()["feature_name"].unique()
        missing_root_names = []
        for expr in exprs:
            for name in expr.meta.root_names():
                if name not in existing_feature_names and name not in all_output_names:
                    missing_root_names.append(name)

        if len(missing_root_names) > 0:
            raise ValueError(
                f"Your expressions require the following root names which are not yet stored in (or computed by) this Gamma Lake: {missing_root_names}"
            )

        new_features = pl.DataFrame().with_columns(
            feature_name=pl.Series([expr.meta.output_name() for expr in exprs]),
            table_addr=pl.Series(["n/a"] * len(exprs)),
            owner=pl.Series([owner] * len(exprs)),
            signal_type=pl.Series(["runtime_computed"] * len(exprs)),
            feature_params=pl.Series([expr.meta.serialize(format="json") for expr in exprs]),
        )

        already_registered = [n for n in new_features["feature_name"].to_list() if n in existing_feature_names]
        if already_registered:
            raise ValueError(
                f"The following feature names are already registered and cannot be re-registered: {already_registered}. "
                "Use a unique alias for each runtime computed feature."
            )

        existing_versions = (
            self.feature_metadata_frame()
            .filter(pl.col("feature_name").is_in(new_features["feature_name"]))
            .group_by("feature_name")
            .agg(pl.col("version").max().alias("_max_version"))
            .collect()
        )
        new_features = (
            new_features.join(existing_versions, on="feature_name", how="left")
            .with_columns(version=(pl.col("_max_version").fill_null(-1) + 1).cast(pl.Int64))
            .drop("_max_version")
        )
        self.io.write_delta(new_features, self.feature_metadata, mode="append", delta_write_options={"schema_mode": "merge"})

    @staticmethod
    def merge(
        left: "GammaFeatureLake",
        right: "GammaFeatureLake",
        left_features: list | pl.DataFrame,
        right_features: list | pl.DataFrame,
        left_targets: list | None = None,
        right_targets: list | None = None,
        start: Comparable | None = None,
        end: Comparable | None = None,
        suffix: str = "_right",
    ) -> pl.DataFrame:
        """Read from two GammaFeatureLakes with the same sort keys and full-outer-join their outputs.

        Both lakes must share identical ``sort_keys`` (i.e. the same ``primary_sort_key`` and
        secondary key columns).  Features present in only one lake will appear with nulls in the
        rows sourced from the other lake.

        The join exploits the fact that both ``read()`` calls return frames that are already sorted
        on ``sort_keys``.  Rather than building a hash table (the default eager join strategy),
        we route through Polars' streaming execution engine.  On Polars ≥
        ``_STREAMING_MERGE_JOIN_MIN_VERSION`` the ``primary_sort_key`` column is declared sorted
        (via ``set_sorted``) so the optimizer promotes the join to an O(n+m) sort-merge join,
        which is typically **3–18× faster** on large frames.  On older Polars versions the same
        streaming engine is used but falls back to a hash join (correct, just without the
        sort-merge optimisation).

        Args:
            left: First GammaFeatureLake instance.
            right: Second GammaFeatureLake instance.  Must have the same ``sort_keys`` as *left*.
            left_features: Feature names to read from *left*, either as a list of strings or a
                pre-filtered ``feature_metadata`` DataFrame (e.g. to pin a specific version).
            right_features: Feature names to read from *right*, either as a list of strings or a
                pre-filtered ``feature_metadata`` DataFrame.
            left_targets: Target names to read from *left*.  Ignored when ``left_features`` is a
                DataFrame (targets are encoded in the metadata frame).  Defaults to ``[]``.
            right_targets: Target names to read from *right*.  Ignored when ``right_features`` is a
                DataFrame.  Defaults to ``[]``.
            start: Inclusive lower bound on ``primary_sort_key``.  Applied to both lakes.
            end: Inclusive upper bound on ``primary_sort_key``.  Applied to both lakes.
            suffix: Suffix appended to non-key columns whose names clash between the two lakes.

        Returns:
            A ``pl.DataFrame`` containing all rows from both lakes, sorted on ``sort_keys``.
            Rows present in only one lake have nulls in the columns sourced from the other.

        Raises:
            ValueError: If the two lakes have different ``sort_keys``.
        """
        if left.sort_keys != right.sort_keys:
            raise ValueError(f"Cannot merge two GammaFeatureLakes with different sort_keys: {left.sort_keys!r} vs {right.sort_keys!r}")

        def _read_lazy(lake: "GammaFeatureLake", features: list | pl.DataFrame, targets: list | None) -> pl.LazyFrame:
            if isinstance(features, pl.DataFrame):
                return lake.read(features, start=start, end=end, materialized=False)
            return lake.read(features, targets or [], start=start, end=end, materialized=False)

        on = left.sort_keys
        left_lf = _read_lazy(left, left_features, left_targets)
        right_lf = _read_lazy(right, right_features, right_targets)

        # Polars ≥ 1.37.0: hint that both frames are sorted so the streaming engine
        # uses an O(n+m) sort-merge join instead of a hash join.
        if version.parse(pl.__version__) >= version.parse("1.37.0"):
            left_lf = left_lf.set_sorted(left.primary_sort_key)
            right_lf = right_lf.set_sorted(right.primary_sort_key)

        return left_lf.join(right_lf, on=on, how="full", coalesce=True, suffix=suffix).collect(engine="streaming")
