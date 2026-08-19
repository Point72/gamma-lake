"""Topological ``with_columns`` helper."""

from graphlib import TopologicalSorter

import polars as pl

__all__ = ("with_columns_topo",)


def with_columns_topo(lf: pl.LazyFrame, exprs: list[pl.Expr]) -> pl.LazyFrame:
    """Apply expressions to a LazyFrame in topological order, batching independent expressions
    into the same `with_columns` call to encourage parallel execution.

    NOTE: only supports expressions with 1 output per expression.

    Args:
        lf: The input LazyFrame.
        exprs: Expressions to add; later expressions may depend on aliases defined by earlier ones.

    Returns:
        The LazyFrame with expressions applied in a dependency-safe order.

    """
    if not exprs:
        return lf

    name_to_expr: dict[str, pl.Expr] = {}
    feature_names: set[str] = set()

    for expr in exprs:
        if expr.meta.has_multiple_outputs():
            raise ValueError(f"with_columns_topo does not support multi-output expressions, received: {expr!s}")
        name = expr.meta.output_name()
        name_to_expr[name] = expr
        feature_names.add(name)

    dep_graph: dict[str, set[str]] = {}
    for name, expr in name_to_expr.items():
        upstreams = set(expr.meta.root_names())
        dep_graph[name] = upstreams.intersection(feature_names)

    ts = TopologicalSorter(dep_graph)
    ts.prepare()

    while ts:
        ready = ts.get_ready()
        lf = lf.with_columns([name_to_expr[name] for name in ready])
        ts.done(*ready)
    return lf
