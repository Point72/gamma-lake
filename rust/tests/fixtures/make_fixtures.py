"""Generate the Delta-lake fixtures and expected results used by the Rust tests.

``make fixtures`` runs this script with ``GammaFeatureLake`` imported from the pure-Python
engine of a release (``COMPAT_REF``, by default v0.2.0). The outputs are not committed.

Each lake under ``lakes/`` is written by ``GammaFeatureLake`` from a scripted write
history. ``queries.json`` lists the queries run by ``rust/tests/read_parity.rs``;
``expected/<name>.arrow`` holds the reference result of each successful query.

The reference is ``GammaFeatureLake.read`` itself, filtered by the query predicate and
with columns selected in request order (Python's own column order is not stable). Both
engines align feature tables to the index by row position, so lakes whose tables are not
row-aligned (see Point72/gamma-lake#45) read identically misaligned in both. Such queries
set ``validate_expect``: with ``validate=True`` the Rust core must fail with an integrity
error instead.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

from gammalake import GammaFeatureLake

ROOT = Path(__file__).resolve().parent
LAKES = ROOT / "lakes"
EXPECTED = ROOT / "expected"
SYMBOLS = ["A", "B"]


def ts(day: int) -> dt.datetime:
    return dt.datetime(2024, 1, day, tzinfo=dt.timezone.utc)


def frame(days, symbols, **features) -> pl.DataFrame:
    """Rows for every (day, symbol); feature value = scale * day + symbol offset."""
    rows = list(itertools.product(days, symbols))
    data = {
        "timestamp": [ts(d) for d, _ in rows],
        "symbol": [s for _, s in rows],
    }
    for name, scale in features.items():
        data[name] = [float(scale * d + (ord(s) - ord("A")) / 10) for d, s in rows]
    return pl.DataFrame(data).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))


def new_lake(name: str, **kwargs) -> GammaFeatureLake:
    path = LAKES / name
    shutil.rmtree(path, ignore_errors=True)
    counter = itertools.count()
    lake = GammaFeatureLake(base_path=str(path), **kwargs).initialize()
    # Deterministic table addresses keep regenerated fixtures diff-friendly.
    object.__setattr__(lake, "gen_table_addr", lambda: f"{lake.table_name_prefix}delta_{next(counter):08x}")
    return lake


def build_lakes() -> dict[str, GammaFeatureLake]:
    lakes = {}

    # basic: several tables, a target, tail rows absent from older tables, a new
    # secondary key inserted at existing timestamps (keys-only backfill rows).
    lake = new_lake("basic")
    lake.add_features(frame(range(1, 7), SYMBOLS, f1=1, f2=2))
    lake.add_targets(frame(range(1, 7), SYMBOLS, t1=-1))
    lake.add_features(frame(range(4, 10), SYMBOLS, f3=3))
    lake.add_features(frame([3, 6], ["C"], f4=4))
    lakes["basic"] = lake

    # misaligned: Python's all-new append skips index rows added by other tables.
    lake = new_lake("misaligned")
    lake.add_features(frame([1], ["A"], x=1))
    lake.add_features(frame(range(1, 5), ["A"], y=10))
    lake.add_features(frame([5], ["A"], x=1))
    lakes["misaligned"] = lake

    # overlap_copy: partial overwrite of one co-located feature (new version, new table).
    lake = new_lake("overlap_copy")
    lake.add_features(frame(range(1, 7), SYMBOLS, f1=1, f2=2))
    lake.add_features(frame(range(4, 10), SYMBOLS, f3=3))
    lake.add_features(frame([2, 3], ["A"], f1=100), overlap_mode="copy")
    lakes["overlap_copy"] = lake

    # overlap_merge: in-place upsert of one co-located feature.
    lake = new_lake("overlap_merge")
    lake.add_features(frame(range(1, 7), SYMBOLS, f1=1, f2=2))
    lake.add_features(frame([2, 3], ["A"], f1=100), overlap_mode="merge")
    lakes["overlap_merge"] = lake

    # files: one table, one file per append, for file-pruning checks.
    lake = new_lake("files")
    for day in range(1, 9):
        lake.add_features(frame([day], ["A"], f1=1))
    lakes["files"] = lake

    # joined: as-of features with each strategy, a tolerance and a late all-new write, and
    # a sparse feature, next to a dense one.
    lake = new_lake("joined")
    lake.add_features(frame(range(1, 9), SYMBOLS, f1=1))
    lake.add_as_of_features(frame([2, 5], ["A"], back=10), params={"strategy": "backward"})
    lake.add_as_of_features(frame([3, 6], SYMBOLS, fwd=20), params={"strategy": "forward", "tolerance": "1d"})
    lake.add_as_of_features(frame([4], ["B"], near=30), params={"strategy": "nearest", "allow_exact_matches": False})
    lake.add_sparse_features(frame([2, 7], ["B"], sp=40))
    lake.add_as_of_features(frame([10], ["A"], late=50), params={"strategy": "backward"})
    lakes["joined"] = lake

    # unsupported: a lake mixing every signal type; the Rust core rejects runtime-computed.
    lake = new_lake("unsupported", enable_runtime_computed_features=True)
    lake.add_features(frame(range(1, 5), SYMBOLS, f1=1))
    lake.add_as_of_features(frame([2], SYMBOLS, asof1=5), params={"strategy": "backward"})
    lake.add_sparse_features(frame([3], ["A"], sparse1=7))
    lake.add_runtime_computed_features([(pl.col("f1") + 1).alias("rt1")])
    lakes["unsupported"] = lake

    return lakes


# --------------------------------------------------------------------------- DNF


def literal(spec: dict):
    kind, value = spec["type"], spec["value"]
    if kind in ("timestamp_us_utc", "timestamp_ns_utc"):
        return pl.lit(dt.datetime.fromisoformat(value), dtype=pl.Datetime(kind.split("_")[1], "UTC"))
    if kind == "utf8":
        return pl.lit(value, dtype=pl.String)
    if kind == "int64":
        return pl.lit(value, dtype=pl.Int64)
    if kind == "float64":
        return pl.lit(value, dtype=pl.Float64)
    raise ValueError(kind)


def term_expr(term: dict) -> pl.Expr:
    column, op, values = pl.col(term["column"]), term["op"], term.get("values", [])
    if op == "is_null":
        return column.is_null()
    if op == "is_not_null":
        return column.is_not_null()
    if op in ("in", "not_in"):
        if not values:
            return pl.lit(False) if op == "in" else column.is_not_null()
        member = pl.any_horizontal([column == literal(v) for v in values])
        return member.fill_null(False) if op == "in" else (~member).fill_null(False) & column.is_not_null()
    lit = literal(values[0])
    return {"eq": column == lit, "not_eq": column != lit, "lt": column < lit, "lt_eq": column <= lit, "gt": column > lit, "gt_eq": column >= lit}[op]


def dnf_expr(dnf: list[list[dict]]) -> pl.Expr:
    if not dnf:
        return pl.lit(True)
    return pl.any_horizontal([pl.all_horizontal([term_expr(t) for t in conj]) for conj in dnf]).fill_null(False)


# ------------------------------------------------------------------------ oracle


def reference_read(lake: GammaFeatureLake, features: list[str], targets: list[str], dnf) -> pl.DataFrame:
    metadata = lake.feature_metadata_frame().collect()
    requested = list(dict.fromkeys(features + targets))
    resolved = set(lake._get_correct_feature_tables(requested, metadata).drop_nulls("table_addr")["feature_name"])
    names = [n for n in requested if n in resolved]
    window = primary_window(lake, dnf)
    if window is not None:
        # Python filters every table by start/end before aligning them, as Rust does with
        # any predicate. On lakes that are not row-aligned this differs from read-then-filter.
        return lake.read(features, targets, **window).select(names + lake.sort_keys)
    return lake.read(features, targets).filter(dnf_expr(dnf)).select(names + lake.sort_keys)


def primary_window(lake: GammaFeatureLake, dnf) -> dict | None:
    """``start``/``end`` for ``GammaFeatureLake.read`` if ``dnf`` is an inclusive range on the primary sort key."""
    bounds = {"gt_eq": "start", "lt_eq": "end"}
    if len(dnf) != 1 or not all(t["column"] == lake.primary_sort_key and t["op"] in bounds for t in dnf[0]):
        return None
    window = {bounds[t["op"]]: dt.datetime.fromisoformat(t["values"][0]["value"]) for t in dnf[0]}
    return window if len(window) == len(dnf[0]) else None


# ----------------------------------------------------------------------- queries

T = lambda day: {"type": "timestamp_us_utc", "value": ts(day).isoformat()}  # noqa: E731
S = lambda value: {"type": "utf8", "value": value}  # noqa: E731


def term(column, op, *values):
    return {"column": column, "op": op, "values": list(values)}


READS = [
    # name, lake, features, targets, dnf[, expected outcome with validate=True]
    ("basic_all", "basic", ["f1", "f2", "f3", "f4"], ["t1"], []),
    ("basic_order", "basic", ["f3", "f1"], [], []),
    ("basic_window", "basic", ["f1", "f3"], [], [[term("timestamp", "gt_eq", T(3)), term("timestamp", "lt_eq", T(7))]]),
    ("basic_or", "basic", ["f1", "f4"], [], [[term("symbol", "eq", S("C"))], [term("timestamp", "lt", T(2))]]),
    ("basic_in", "basic", ["f2"], [], [[term("symbol", "in", S("A"), S("C"))]]),
    ("basic_not_in", "basic", ["f2"], [], [[term("symbol", "not_in", S("A"))]]),
    ("basic_in_empty", "basic", ["f2"], [], [[term("symbol", "in")]]),
    ("basic_not_null", "basic", ["f1"], [], [[term("symbol", "is_not_null")]]),
    ("basic_empty_result", "basic", ["f1", "f3"], [], [[term("timestamp", "gt", T(28))]]),
    ("basic_unknown_dropped", "basic", ["f1", "nope"], [], []),
    ("basic_ns_literal", "basic", ["f1"], [], [[term("timestamp", "gt_eq", {"type": "timestamp_ns_utc", "value": ts(5).isoformat()})]]),
    ("misaligned", "misaligned", ["x", "y"], [], [], "error:integrity"),
    ("misaligned_window", "misaligned", ["x", "y"], [], [[term("timestamp", "gt_eq", T(5)), term("timestamp", "lt_eq", T(5))]]),
    ("overlap_copy", "overlap_copy", ["f1", "f2", "f3"], [], [], "error:integrity"),
    ("overlap_copy_window", "overlap_copy", ["f1", "f2"], [], [[term("timestamp", "gt_eq", T(2)), term("timestamp", "lt_eq", T(3))]], "error:integrity"),
    ("overlap_merge", "overlap_merge", ["f1", "f2"], [], []),
    ("files_all", "files", ["f1"], [], []),
    ("files_window", "files", ["f1"], [], [[term("timestamp", "gt_eq", T(7))]]),
    ("unsupported_plain", "unsupported", ["f1"], [], []),
    ("mixed_as_of", "unsupported", ["f1", "asof1"], [], []),
    ("mixed_sparse", "unsupported", ["sparse1"], [], []),
    ("joined_all", "joined", ["f1", "back", "fwd", "near", "sp", "late"], [], []),
    ("joined_window", "joined", ["back", "fwd", "near", "sp"], [], [[term("timestamp", "gt_eq", T(4)), term("timestamp", "lt_eq", T(6))]]),
    ("joined_eq", "joined", ["back", "fwd", "near", "sp"], [], [[term("timestamp", "eq", T(4))]]),
    ("joined_in", "joined", ["back", "fwd", "sp"], [], [[term("timestamp", "in", T(3), T(7)), term("symbol", "eq", S("A"))]]),
    ("joined_in_empty", "joined", ["back", "fwd"], [], [[term("timestamp", "in")], [term("symbol", "eq", S("B"))]]),
    ("joined_or", "joined", ["f1", "back", "fwd", "sp"], [], [[term("timestamp", "lt", T(3))], [term("symbol", "eq", S("B")), term("timestamp", "gt", T(6))]]),
    ("joined_not_eq", "joined", ["back", "fwd", "near"], [], [[term("timestamp", "not_eq", T(5))]]),
    ("joined_gt", "joined", ["back", "late"], [], [[term("timestamp", "gt", T(7))]]),
]

ERRORS = [
    ("err_missing", "basic", ["nope"], [], [], "missing_features"),
    ("err_target_not_target", "basic", ["f1"], ["f2"], [], "misregistered_targets"),
    ("err_feature_predicate", "basic", ["f1"], [], [[term("f1", "gt", {"type": "float64", "value": 1.0})]], "invalid_predicate"),
    ("err_unknown_column", "basic", ["f1"], [], [[term("nope", "eq", S("A"))]], "invalid_predicate"),
    ("err_lossy_literal", "basic", ["f1"], [], [[term("timestamp", "gt", {"type": "timestamp_ns_utc", "value": "2024-01-02T00:00:00.000000001+00:00"})]], "invalid_predicate"),
    ("err_bad_literal_type", "basic", ["f1"], [], [[term("symbol", "eq", {"type": "int64", "value": 1})]], "invalid_predicate"),
    ("err_runtime", "unsupported", ["rt1"], [], [], "unsupported"),
]

METADATA = [
    # name, lake, kind, dnf, sort_by
    ("meta_feature_metadata", "basic", "feature_metadata", [], ["feature_name", "version"]),
    ("meta_feature_metadata_filtered", "overlap_copy", "feature_metadata", [[term("feature_name", "eq", S("f1")), term("version", "gt_eq", {"type": "int64", "value": 1})]], ["feature_name", "version"]),
    ("meta_latest_features", "overlap_copy", "latest_features", [], ["feature_name"]),
    ("meta_table_metadata", "basic", "table_metadata", [], ["table_addr", "last_updated"]),
    ("meta_index", "basic", "index", [[term("symbol", "eq", S("C"))]], ["timestamp", "symbol"]),
]


def write_ipc(df: pl.DataFrame, path: Path) -> None:
    table = df.to_arrow()
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)


def main() -> None:
    shutil.rmtree(EXPECTED, ignore_errors=True)
    EXPECTED.mkdir(parents=True)
    LAKES.mkdir(exist_ok=True)
    lakes = build_lakes()
    queries = []

    for name, lake_name, features, targets, dnf, *validate_expect in READS:
        lake = lakes[lake_name]
        write_ipc(reference_read(lake, features, targets, dnf), EXPECTED / f"{name}.arrow")
        query = {"name": name, "kind": "read", "lake": lake_name, "features": features, "targets": targets, "dnf": dnf, "expect": "ok"}
        if validate_expect:
            query["validate_expect"] = validate_expect[0]
        queries.append(query)

    for name, lake_name, features, targets, dnf, error in ERRORS:
        queries.append({"name": name, "kind": "read", "lake": lake_name, "features": features, "targets": targets, "dnf": dnf, "expect": f"error:{error}"})

    for name, lake_name, kind, dnf, sort_by in METADATA:
        lake = lakes[lake_name]
        source = {
            "feature_metadata": lambda: lake.feature_metadata_frame().collect(),
            "latest_features": lambda: lake.feature_metadata_frame().collect().sort("version", descending=True, nulls_last=True).unique(subset=["feature_name"], keep="first"),
            "table_metadata": lambda: lake.table_metadata_frame().collect(),
            "index": lambda: lake.index_frame().collect(),
        }[kind]()
        write_ipc(source.filter(dnf_expr(dnf)).sort(sort_by), EXPECTED / f"{name}.arrow")
        queries.append({"name": name, "kind": kind, "lake": lake_name, "dnf": dnf, "sort_by": sort_by, "expect": "ok"})

    (ROOT / "queries.json").write_text(json.dumps(queries, indent=2) + "\n")
    print(f"wrote {len(queries)} queries")


if __name__ == "__main__":
    main()
