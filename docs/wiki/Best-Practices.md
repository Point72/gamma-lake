# Best Practices for Gamma Lake

______________________________________________________________________

## Prefer Appending New Versions Over In-Place Modification

Generally speaking, it is best to **append** new versions of features rather than trying to modify them in-place.
Generic Delta Lake operations (updates, merges, upserts) are supported, but they carry higher complexity and should be
used only when necessary.

The versioning model makes re-appending cheap and safe:

```python
# Add an updated version of your features — old version is preserved
lake.add_features(updated_feature_df, owner="my-team")

# You now have two versions — the old one is still readable
print(lake.feature_metadata_frame().collect())
```

By default, `read()` returns the **latest version** of each feature. To read an earlier version, filter the metadata
frame first:

```python
# Read only the original version (version == 0)
meta = lake.feature_metadata_frame().filter(pl.col("version") == 0).collect()
result = lake.read(meta)
```

______________________________________________________________________

## Batch Writes to Minimise Index Work

Each write of stored features or targets (`add_features()`, `add_targets()`, `add_as_of_features()` and
`add_sparse_features()`) scans the global index once and appends newly discovered index rows once. Every call also pays
fixed dispatch, metadata, and storage-commit overhead, so the shape and number of writes can matter more than the total
row count.

Prefer **wide-and-shallow** writes over **tall-and-skinny** writes. Here, wide means batching related value columns and
all secondary-key values for a primary-sort-key period; shallow means writing fewer complete periods at a time. Avoid
splitting the same period across many calls.

For backfills, build blocks containing every secondary-key value for each primary-sort-key period, include as many
complete periods as fit comfortably in memory, and write each block in one call. Do not repeatedly batch the same
period by entity. Keep unrelated owners or update cadences in separate feature groups even when batching writes.

If incremental weekly additions leave related features fragmented across many small feature-group tables, call
`consolidate_feature_groups()` after the features share an update cadence. Consolidation reduces the number of table
scans required by future reads without changing feature values.

______________________________________________________________________

## Use Owners to Organise Features

Every `add_*` call accepts an `owner` string. Use this to track which team, pipeline, or experiment produced a feature:

```python
lake.add_features(df_alpha, owner="alpha-team")
lake.add_features(df_beta,  owner="beta-experiment-2024-06")
```

You can then read only a specific team's features:

```python
alpha_meta = lake.feature_metadata_frame().filter(pl.col("owner") == "alpha-team").collect()
result = lake.read(alpha_meta)
```

______________________________________________________________________

## Keep Feature Groups Logically Coherent

New features written in the same `add_features()` call share a new Delta table (a *feature group*). Later writes to
those features append to that table or, when they overlap stored rows, follow the call's `overlap_mode`. Design your
feature groups around:

- **Update frequency** — features updated daily should be in the same group; intraday features in a separate group
- **Ownership** — features owned by the same team are natural candidates for the same group
- **Size** — very wide groups (thousands of columns) can be split for better read performance

Reads always select the minimum set of Delta tables required for the requested feature names, so smaller groups improve
selective read performance.

______________________________________________________________________

## Choose a Meaningful `primary_sort_key`

The `primary_sort_key` (default: `"timestamp"`) is used for range filtering in all reads. Ensure it is:

- **Present in your index schema** — the `initialize()` call will raise if it is missing
- **Monotonically useful for range queries** — dates and timestamps are natural choices
- **Consistent** — changing `primary_sort_key` on an already-initialised lake is not supported

______________________________________________________________________

## Use Date Ranges to Limit Read I/O

All read methods accept `start` and `end` parameters that are pushed down to the Delta table scan:

```python
from datetime import date

result = lake.read(["momentum", "vol_20d"], start=date(2023, 1, 1), end=date(2023, 12, 31))
```

Always provide date ranges when reading from large lakes, so that scans can skip files whose statistics fall outside
the range.

______________________________________________________________________

## Avoid Reading the Entire Index in Production

`lake.index_frame().collect()` returns the full master index. This can be very large in production. Use it for
diagnostics and small-scale inspection only. For range-bounded reads, use the `read()` API with `start`/`end` instead.

______________________________________________________________________

## Clean Up Temporary Lakes

If you are creating throwaway lakes for tests or experiments, always clean up the base directory afterwards:

```python
import tempfile, shutil

tmp = tempfile.mkdtemp()
try:
    lake = GammaFeatureLake(base_path=tmp)
    lake.initialize(schema)
    # ... work ...
finally:
    shutil.rmtree(tmp)
```

Or use `tempfile.TemporaryDirectory()` as a context manager, as the [quick start](https://github.com/Point72/gamma-lake#quick-start) does.

______________________________________________________________________

## Feature Type Reference

| Method                            | Use when                                                                  |
| --------------------------------- | ------------------------------------------------------------------------- |
| `add_features()`                  | Standard features updated on a regular schedule                           |
| `add_targets()`                   | Labels / target variables (stored separately, read together)              |
| `add_as_of_features()`            | Point-in-time safe features (e.g., fundamental data with publication lag) |
| `add_sparse_features()`           | Infrequently updated features (e.g., quarterly earnings)                  |
| `add_runtime_computed_features()` | Features derived from stored features at read time (no extra storage)     |
| `add_runtime_transforms()`        | Standard row-local transforms over stored features at read time           |

### Runtime transforms

Set `enable_runtime_computed_features=True` only when the lake's feature metadata is trusted. The opt-in is required both
to register runtime transforms and to read them.

`add_runtime_transforms(features, transforms, owner)` creates the cross-product of the supplied lists with deterministic
`<feature>__<transform>` names. Supported transforms are `abs`, `missing`, `nan`, `negative`, `nonzero`, `positive`,
`reciprocal`, `sign`, `signed_log1p`, and `squared`. Numeric transforms cast inputs and outputs to `Float64` and propagate
nulls; reciprocal also maps zero to null. `missing`, `nan`, and `nonzero` are boolean. `positive` and `negative` clamp to
the non-negative part of the value and its negation, respectively. `sign` returns -1, 0, or 1, and `signed_log1p` computes
`sign(x) * log1p(abs(x))`.
