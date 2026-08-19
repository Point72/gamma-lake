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
# Read only the original version (version == 1)
meta = lake.feature_metadata_frame().filter(pl.col("version") == 1).collect()
result = lake.read(meta)
```

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

Gamma Lake writes one Delta table per `add_features()` call. Features written in the same call share a Delta table
(a *feature group*). Design your feature groups around:

- **Update frequency** — features updated daily should be in the same group; intraday features in a separate group
- **Ownership** — features owned by the same team are natural candidates for the same group
- **Size** — very wide groups (thousands of columns) can be split for better read performance

Reads always select the minimum set of Delta tables required for the requested feature names, so smaller groups improve
selective read performance.

______________________________________________________________________

## Use `run_on_ray_cluster=False` for Local Development

When iterating locally — unit tests, notebooks, small experiments — set `run_on_ray_cluster=False` to skip Ray
dispatch overhead:

```python
lake = GammaFeatureLake(base_path="/tmp/my_lake", run_on_ray_cluster=False)
```

Switch back to `run_on_ray_cluster=True` (the default) for production or large-scale workloads where parallel writes
and reads provide significant speedups.

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
result = lake.read(["momentum", "vol_20d"], start="2023-01-01", end="2023-12-31")
```

Always provide date ranges when reading from large lakes to avoid scanning all partitions.

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
    lake = GammaFeatureLake(base_path=tmp, run_on_ray_cluster=False)
    lake.initialize(schema)
    # ... work ...
finally:
    shutil.rmtree(tmp)
```

Or use `tempfile.TemporaryDirectory()` as a context manager (see [`examples/quickstart.py`](https://github.com/Point72/gamma-lake/blob/main/examples/quickstart.py)).

______________________________________________________________________

## Feature Type Reference

| Method                            | Use when                                                                  |
| --------------------------------- | ------------------------------------------------------------------------- |
| `add_features()`                  | Standard features updated on a regular schedule                           |
| `add_targets()`                   | Labels / target variables (stored separately, read together)              |
| `add_as_of_features()`            | Point-in-time safe features (e.g., fundamental data with publication lag) |
| `add_sparse_features()`           | Infrequently updated features (e.g., quarterly earnings)                  |
| `add_runtime_computed_features()` | Features derived from stored features at read time (no extra storage)     |
