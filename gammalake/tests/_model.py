"""Frames keyed by ``(day, symbol)`` and a reference model of the reads they produce."""

import datetime as dt

import polars as pl

KEYS = ["timestamp", "symbol"]


def ts(day: int) -> dt.datetime:
    return dt.datetime(2024, 1, day, tzinfo=dt.UTC)


def keys(days, symbols) -> list[tuple[int, str]]:
    return [(d, s) for d in days for s in symbols]


def frame(rows, **scales) -> pl.DataFrame:
    """One row per ``(day, symbol)`` key; feature ``name`` holds ``scale * day``."""
    data = {"timestamp": [ts(d) for d, _ in rows], "symbol": [s for _, s in rows]}
    for name, scale in scales.items():
        data[name] = [float(scale * d) for d, _ in rows]
    schema = {"timestamp": pl.Datetime("us", "UTC"), "symbol": pl.String, **dict.fromkeys(scales, pl.Float64)}
    return pl.DataFrame(data, schema=schema)


class Model:
    """Expected reads.

    An input holding only new keys appends them. Otherwise it replaces its features over
    its primary key window, so keys of the window missing from the input read as null.
    """

    def __init__(self):
        self.index: set[tuple[int, str]] = set()
        self.values: dict[str, dict[tuple[int, str], float | None]] = {}

    def add(self, rows, **scales) -> None:
        given = set(rows)
        overlaps = bool(given & self.index)
        self.index |= given
        low, high = min(d for d, _ in rows), max(d for d, _ in rows)
        for name, scale in scales.items():
            column = self.values.setdefault(name, {})
            for key in self.index if overlaps else given:
                if low <= key[0] <= high:
                    column[key] = float(scale * key[0]) if key in given else None

    def read(self, features) -> pl.DataFrame:
        index = sorted(self.index)
        data = {name: [self.values[name].get(k) for k in index] for name in features}
        data |= {"timestamp": [ts(d) for d, _ in index], "symbol": [s for _, s in index]}
        return pl.DataFrame(data, schema={**dict.fromkeys(features, pl.Float64), "timestamp": pl.Datetime("us", "UTC"), "symbol": pl.String})


# Each step is (rows, overlap_mode, scales).
SCENARIOS = {
    "append": [
        (keys(range(1, 4), "ab"), "copy", {"f": 1}),
        (keys(range(4, 6), "ab"), "copy", {"f": 1}),
    ],
    "new_feature_on_new_keys": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        (keys(range(2, 6), "ab"), "copy", {"g": 2}),
    ],
    # Point72/gamma-lake#45: appending only new keys skips padding the tail of f.
    "all_new_keys_after_coverage": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        (keys(range(1, 6), "a"), "copy", {"g": 2}),
        (keys([6], "a"), "copy", {"f": 1}),
    ],
    # New keys inside the coverage of g are padded into it.
    "all_new_keys_inside_coverage": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        (keys(range(1, 4), "a"), "copy", {"g": 2}),
        ([(2, "b"), (3, "b")], "copy", {"f": 1}),
    ],
    # Point72/gamma-lake#46: copy mode drops the rows of f after the window.
    "overlap_copy": [
        (keys(range(1, 7), "a"), "copy", {"f": 1}),
        (keys(range(1, 7), "ab"), "copy", {"g": 2}),
        (keys(range(2, 4), "a"), "copy", {"f": 10}),
    ],
    "overlap_merge": [
        (keys(range(1, 5), "ab"), "copy", {"f": 1, "h": 3}),
        ([(2, "a"), (3, "a"), (5, "a")], "merge", {"f": 10}),
    ],
    # Tables that a merge left untouched are still padded by the next write.
    "write_after_overlap_merge": [
        (keys(range(1, 5), "ab"), "copy", {"f": 1, "h": 3}),
        (keys(range(2, 4), "a"), "merge", {"f": 10}),
        (keys(range(5, 6), "ab"), "copy", {"g": 1}),
    ],
    # A new key whose primary key equals the coverage of f must be padded into f.
    "new_key_at_coverage_end": [
        (keys(range(1, 4), "a"), "copy", {"f": 1}),
        ([(3, "b")], "copy", {"g": 1}),
        (keys([4], "a"), "copy", {"f": 1}),
    ],
}
