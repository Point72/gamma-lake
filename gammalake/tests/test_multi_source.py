import polars as pl
import pytest

from gammalake._multi_source import scan_aligned_sources


def _tracked_source(df: pl.DataFrame, calls: list[tuple[list[str] | None, pl.Expr | None]]) -> pl.LazyFrame:
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ):
        calls.append((with_columns, predicate))
        result = df.lazy()
        if predicate is not None:
            result = result.filter(predicate)
        if n_rows is not None:
            result = result.head(n_rows)
        if with_columns is not None:
            result = result.select(with_columns)
        yield result.collect()

    return pl.io.plugins.register_io_source(source_generator, schema=df.schema, is_pure=True)


def _sources(
    *,
    right_keys: list[int] | None = None,
) -> tuple[dict[str, pl.LazyFrame], dict[str, list[tuple[list[str] | None, pl.Expr | None]]]]:
    calls = {"index": [], "left": [], "right": []}
    sources = {
        "index": _tracked_source(pl.DataFrame({"key": [1, 2, 3]}), calls["index"]),
        "left": _tracked_source(pl.DataFrame({"key": [1, 2, 3], "left_value": [10, 20, 30]}), calls["left"]),
        "right": _tracked_source(
            pl.DataFrame({"key": right_keys or [1, 2, 3], "right_value": [100, 200, 300][: len(right_keys or [1, 2, 3])]}),
            calls["right"],
        ),
    }
    return sources, calls


def test_preserves_exact_positional_alignment():
    sources, _ = _sources()

    result = scan_aligned_sources(sources, alignment_columns=["key"]).collect()

    expected = pl.DataFrame({"key": [1, 2, 3], "left_value": [10, 20, 30], "right_value": [100, 200, 300]})
    assert result.equals(expected)


def test_propagates_filter_to_every_source():
    sources, calls = _sources()
    frame = scan_aligned_sources(sources, alignment_columns=["key"])
    query = frame.filter(pl.col("key").is_in([2, 3])).select("left_value", "right_value")

    plan = query.explain()
    result = query.collect()

    assert "PYTHON SCAN" in plan
    assert 'col("key")' in plan
    assert result.equals(pl.DataFrame({"left_value": [20, 30], "right_value": [200, 300]}))
    for source_calls in calls.values():
        assert len(source_calls) == 1
        assert source_calls[0][1] is not None
        assert source_calls[0][1].meta.root_names() == ["key"]


def test_projects_only_needed_source_columns():
    sources, calls = _sources()

    result = scan_aligned_sources(sources, alignment_columns=["key"]).select("right_value").collect()

    assert result.equals(pl.DataFrame({"right_value": [100, 200, 300]}))
    assert calls["index"][0][0] in (None, ["key"])
    assert calls["left"][0][0] == ["left_value"]
    assert calls["right"][0][0] == ["right_value"]


def test_returns_empty_filtered_result():
    sources, _ = _sources()

    result = scan_aligned_sources(sources, alignment_columns=["key"]).filter(pl.col("key") > 3).collect()

    assert result.is_empty()
    assert result.schema == {"key": pl.Int64, "left_value": pl.Int64, "right_value": pl.Int64}


def test_rejects_mismatched_filtered_heights():
    sources, _ = _sources(right_keys=[1, 2])

    with pytest.raises(pl.exceptions.ComputeError, match="aligned sources have mismatched heights"):
        scan_aligned_sources(sources, alignment_columns=["key"]).collect()


def test_propagates_source_errors():
    def broken_source(with_columns, predicate, n_rows, batch_size):
        raise RuntimeError("source collection failed")

    sources, _ = _sources()
    sources["right"] = pl.io.plugins.register_io_source(
        broken_source,
        schema={"key": pl.Int64, "right_value": pl.Int64},
        is_pure=True,
    )

    with pytest.raises(pl.exceptions.ComputeError, match="source collection failed"):
        scan_aligned_sources(sources, alignment_columns=["key"]).collect()


def test_is_safe_to_collect_repeatedly():
    sources, calls = _sources()
    query = scan_aligned_sources(sources, alignment_columns=["key"]).filter(pl.col("key") == 2)

    first = query.collect()
    second = query.collect()

    assert first.equals(second)
    assert first.equals(pl.DataFrame({"key": [2], "left_value": [20], "right_value": [200]}))
    assert all(len(source_calls) == 2 for source_calls in calls.values())
