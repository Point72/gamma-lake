# How Gamma Lake Works

Feature store implementations are difficult: data representations and memory layouts can make the addition of new feature
columns or rows computationally expensive, versioning and partitioning schemes are things users often don't want to worry
about, and scaling laws can prevent efficient small-scale solutions from being useful at terabyte scale.

This document introduces `Gamma Lake`, a `DeltaTable`-backed implementation of a feature store which leverages Ray to
perform parallel updates and efficient read operations.

______________________________________________________________________

## The Index and Feature Groups

Assume that your features can be *indexed*. More specifically, we assume that your index uniquely represents a single
vector of features and that there exists a strict ordering on those tuple keys. In practice `(timestamp, symbol)`
satisfies this property and covers many common use cases for machine learning in finance.

A Gamma Lake object maintains two kinds of Delta tables:

- **Index table** — the master ordered index of all `(timestamp, symbol)` pairs ever seen
- **Feature group tables** — one Delta table per group of features, each sharing the same index structure

```
Index Table                        Feature Group F1         Feature Group F2
+-------------+--------+           +-------------+-------+  +-------------+-------+
| timestamp   | symbol |           | timestamp   | feat1 |  | timestamp   | feat3 |
+-------------+--------+           +-------------+-------+  +-------------+-------+
| 2024-01-01  | AAPL   |           | 2024-01-01  |  0.42 |  | 2024-01-01  |  1.1  |
| 2024-01-01  | GOOG   |           | 2024-01-01  |  0.31 |  | 2024-01-01  |  0.9  |
| 2024-01-02  | AAPL   |           | 2024-01-02  |  0.45 |  | 2024-01-02  |  1.2  |
| 2024-01-02  | GOOG   |           | 2024-01-02  |  0.28 |  +-------------+-------+
+-------------+--------+           +-------------+-------+
```

> **Note:** The physical layout of tables is not ordered in storage — the ordering shown here is conceptual, for clarity.

______________________________________________________________________

## Reading Features

When reading features from Gamma Lake:

1. Each relevant feature group's Delta table is queried with range filters applied to the primary sort key
1. Each table is sorted by the index key(s)
1. The sorted tables are **horizontally concatenated** using Polars

Because all feature groups are aligned to the master index, this concatenation is correct without any join. Groups with
fewer rows (not yet updated for the latest index values) simply produce `null` in the trailing rows:

```
Feature Group F1     Feature Group F2     Concatenated Result
(3 rows)             (2 rows)             (3 rows)
+-------+            +-------+            +-------+-------+
| feat1 |            | feat3 |            | feat1 | feat3 |
+-------+            +-------+            +-------+-------+
|  0.42 |            |  1.1  |     =>     |  0.42 |  1.1  |
|  0.31 |            |  0.9  |            |  0.31 |  0.9  |
|  0.45 |            (end)               |  0.45 |  null |
+-------+                                +-------+-------+
```

The `null` values for F2's missing row are implicit — they arise from Polars padding a shorter DataFrame during
horizontal concatenation, not from any actual null-storage in the Delta table for F2. This is a key efficiency win.

______________________________________________________________________

## Appending New Features (No Overlap)

When you call `add_features()` with rows whose index values are **not yet** in the master index, this is a pure append:

1. A new Delta table is written (or an existing group's table is appended)
1. The master index is extended with the new index values

Feature groups that were not updated do not need to be touched — their alignment is automatically maintained because
their latest row still maps to the same position in the master index.

______________________________________________________________________

## Appending New Features (Overlap with Existing Index)

A more complex case: the incoming data contains index values **already present** in the master index. Gamma Lake handles
this correctly:

**Case A — Appending beyond the feature group's latest row:**

If the new rows are all *after* the feature group's `latest_update` timestamp, Gamma Lake simply appends them. Any index
values between the group's latest row and the new rows are padded with `null` to maintain alignment:

```
Before:                              After add_features([I7, I9]):
Feature Group F3 (latest = I3)       Feature Group F3
+-------+                            +-------+
| feat3 |                            | feat3 |
+-------+                            +-------+
|  V1   |  (I1)                      |  V1   |  (I1)
|  V2   |  (I2)                      |  V2   |  (I2)
|  V3   |  (I3)                      |  V3   |  (I3)
+-------+                            | null  |  (I4 — padded)
                                     | null  |  (I5 — padded)
                                     | null  |  (I6 — padded)
                                     |  V7'  |  (I7 — new)
                                     | null  |  (I8 — padded)
                                     |  V9'  |  (I9 — new)
                                     +-------+
```

**Case B — Updating rows already stored in the feature group:**

If the incoming data overlaps with rows already in the feature group (index values *before* `latest_update`), Gamma Lake
uses Delta Lake's **merge (upsert)** operation to update those rows in-place. Other feature groups that share the same
overlapping index range but are not being updated receive `null`-padded rows to maintain alignment.

______________________________________________________________________

## Lazy Read Pushdown

Local reads use the bounded, sorted global index as the canonical source of sort keys and align feature-only values
from each requested feature group. For context-free reads, a downstream filter on sort keys is coordinated into the
index and every feature source before the final result is assembled. As-of joins retain right-hand carry-forward
context, while runtime-computed reads apply downstream filters after computation to preserve neighbour semantics. Ray
reads retain the materialized per-table path.

Projection-only aggregations remain correct, but the multi-source scan cannot yet prune an entire unused child source,
so operations such as `select(pl.len())` may scan more data than a direct single-source aggregation.

______________________________________________________________________

## Parallel Writes

When `run_on_ray_cluster=True` (the default), Gamma Lake dispatches updates to each feature group table as separate
`@ray.remote` functions. This means:

- All feature group table writes happen **in parallel**
- The master index is updated **once**, separately, after all table writes
- Large feature stores with many groups see near-linear write-time improvements

______________________________________________________________________

## Feature Metadata and Versioning

Every call to `add_features()` records:

| Column         | Description                                      |
| -------------- | ------------------------------------------------ |
| `feature_name` | The column name                                  |
| `version`      | Auto-incremented version number                  |
| `owner`        | The `owner` string you pass                      |
| `table_addr`   | Path to the underlying Delta table               |
| `signal_type`  | `"feature"`, `"target"`, `"as_of_feature"`, etc. |

Reading always returns the **latest version** by default. To read a specific version or owner, pass a filtered metadata
DataFrame:

```python
# Read only features owned by "team-a"
meta = lake.feature_metadata_frame().filter(pl.col("owner") == "team-a").collect()
result = lake.read(meta)
```

______________________________________________________________________

## Summary

| Concept                         | Implementation                                            |
| ------------------------------- | --------------------------------------------------------- |
| Master index                    | One Delta table tracking all `(timestamp, symbol)` pairs  |
| Feature group                   | One Delta table per logical group of features             |
| Append (no overlap)             | Write new rows to feature table; extend master index      |
| Append (overlap, beyond latest) | Append with null padding to maintain alignment            |
| Overlap within existing rows    | Delta Lake merge (upsert) in-place                        |
| Cross-group alignment           | Maintained implicitly; no join required at read time      |
| Parallel I/O                    | Ray `@ray.remote` per feature group                       |
| Versioning                      | Metadata table records owner + version per feature column |
