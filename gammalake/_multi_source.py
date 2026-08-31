"""Lazy positional concatenation for aligned Polars sources."""

from collections.abc import Mapping, Sequence

import polars as pl
from polars_io_tools import FilterSpec, pushdown_combine

__all__ = ("scan_aligned_sources",)


def scan_aligned_sources(
    sources: Mapping[str, pl.LazyFrame],
    *,
    alignment_columns: Sequence[str],
) -> pl.LazyFrame:
    """Expose aligned lazy sources as one positionally concatenated LazyFrame.

    The first source owns shared alignment columns in the output. Other output
    columns must occur in exactly one source. Supported constraints on alignment
    columns are applied to every source before collection, and all predicates are
    applied to the combined result. Projection is restricted to columns needed
    from each source.

    Args:
        sources: Ordered names and aligned lazy sources.
        alignment_columns: Columns present with equal types in every source.

    Returns:
        A lazy outer scan preserving each source's row order.

    Raises:
        ValueError: If the source schemas do not satisfy the alignment contract.
        polars.exceptions.ShapeError: If filtered source heights differ.

    """
    if len(sources) < 2:
        raise ValueError("scan_aligned_sources requires at least two sources")

    alignment_columns = tuple(alignment_columns)
    if not alignment_columns or len(set(alignment_columns)) != len(alignment_columns):
        raise ValueError("alignment_columns must contain unique column names")

    source_items = tuple(sources.items())
    source_schemas = {name: frame.collect_schema() for name, frame in source_items}
    first_name, _ = source_items[0]
    first_schema = source_schemas[first_name]
    missing_from_first = set(alignment_columns) - set(first_schema)
    if missing_from_first:
        raise ValueError(f"source {first_name!r} is missing alignment columns: {sorted(missing_from_first)}")

    output_schema = dict(first_schema)
    output_columns_by_source = {first_name: tuple(first_schema)}
    for name, _ in source_items[1:]:
        schema = source_schemas[name]
        missing = set(alignment_columns) - set(schema)
        if missing:
            raise ValueError(f"source {name!r} is missing alignment columns: {sorted(missing)}")

        mismatched_types = [column for column in alignment_columns if schema[column] != first_schema[column]]
        if mismatched_types:
            raise ValueError(f"source {name!r} has incompatible alignment column types: {mismatched_types}")

        source_output_columns = tuple(column for column in schema if column not in alignment_columns)
        duplicates = set(source_output_columns) & set(output_schema)
        if duplicates:
            raise ValueError(f"source {name!r} has duplicate output columns: {sorted(duplicates)}")
        output_schema.update((column, schema[column]) for column in source_output_columns)
        output_columns_by_source[name] = source_output_columns

    marker_columns = tuple(f"__gammalake_alignment_{index}" for index in range(len(source_items)))
    if set(marker_columns) & set(output_schema):
        raise ValueError("source columns use reserved GammaLake alignment names")

    def validate_alignment(markers: dict[str, bool | None]) -> bool:
        if not all(markers.values()):
            raise pl.exceptions.ShapeError("aligned sources have mismatched heights")
        return True

    def combine(filtered_sources: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        pieces = [
            filtered_sources[name].select(output_columns_by_source[name]).with_columns(pl.lit(True).alias(marker))
            for marker, (name, _) in zip(marker_columns, source_items, strict=True)
        ]
        return (
            pl.concat(pieces, how="horizontal")
            .filter(pl.struct(marker_columns).map_elements(validate_alignment, return_dtype=pl.Boolean))
            .drop(marker_columns)
        )

    filter_specs = {column: FilterSpec() for column in alignment_columns}
    return pushdown_combine(
        sources={name: (source, filter_specs) for name, source in source_items},
        combine=combine,
    )
