"""Lazy canonical-key alignment for multiple Polars sources."""

from collections.abc import Callable, Mapping, Sequence

import polars as pl
from polars_io_tools import FilterSpec, pushdown_combine

__all__ = ("scan_aligned_sources",)


def scan_aligned_sources(
    sources: Mapping[str, pl.LazyFrame],
    *,
    alignment_columns: Sequence[str],
    postprocess: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    predicate_pushdown: bool = True,
) -> pl.LazyFrame:
    """Expose aligned lazy sources as one canonical-key LazyFrame.

    The first source owns the alignment columns and row set. Other output
    columns must occur in exactly one source. Supported alignment-column
    constraints are pushed into every source before they are joined to the
    canonical keys. All predicates are reapplied after optional postprocessing.

    Args:
        sources: Ordered names and aligned lazy sources.
        alignment_columns: Columns present with equal types in every source.
        postprocess: Optional transformation applied after source alignment and
            before the downstream predicate and projection.
        predicate_pushdown: Whether alignment-column predicates may be applied
            to the individual sources before collection.

    Returns:
        A lazy frame preserving the canonical first source's row order.

    Raises:
        ValueError: If the source schemas do not satisfy the alignment contract.
        polars.exceptions.ComputeError: If a source contains duplicate
            alignment keys.

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

    output_columns_by_source = {first_name: tuple(first_schema)}
    output_columns = set(first_schema)
    for name, _ in source_items[1:]:
        schema = source_schemas[name]
        missing = set(alignment_columns) - set(schema)
        if missing:
            raise ValueError(f"source {name!r} is missing alignment columns: {sorted(missing)}")

        mismatched_types = [column for column in alignment_columns if schema[column] != first_schema[column]]
        if mismatched_types:
            raise ValueError(f"source {name!r} has incompatible alignment column types: {mismatched_types}")

        source_output_columns = tuple(column for column in schema if column not in alignment_columns)
        duplicates = set(source_output_columns) & output_columns
        if duplicates:
            raise ValueError(f"source {name!r} has duplicate output columns: {sorted(duplicates)}")
        output_columns.update(source_output_columns)
        output_columns_by_source[name] = source_output_columns

    def combine(filtered_sources: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        result = filtered_sources[first_name].select(output_columns_by_source[first_name])
        for name, _ in source_items[1:]:
            result = result.join(
                filtered_sources[name].select(*alignment_columns, *output_columns_by_source[name]),
                on=alignment_columns,
                how="left",
                validate="1:1",
                coalesce=True,
                maintain_order="left",
            )
        return postprocess(result) if postprocess is not None else result

    filter_specs = (
        {column: FilterSpec() for column in alignment_columns if not source_schemas[first_name][column].is_temporal()} if predicate_pushdown else {}
    )
    return pushdown_combine(
        sources={name: (source, filter_specs) for name, source in source_items},
        combine=combine,
    )
