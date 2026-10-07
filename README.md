# Gamma Lake

High-performance feature store built on Delta Lake with optional Ray parallelism

[![Build Status](https://github.com/Point72/gamma-lake/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/Point72/gamma-lake/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/Point72/gamma-lake/branch/main/graph/badge.svg)](https://codecov.io/gh/Point72/gamma-lake)
[![License](https://img.shields.io/github/license/Point72/gamma-lake)](https://github.com/Point72/gamma-lake)
[![PyPI](https://img.shields.io/pypi/v/gamma-lake.svg)](https://pypi.python.org/pypi/gamma-lake)

## Overview

Gamma Lake is a **feature store** built on [Delta Lake](https://delta.io/) with optional [Ray](https://www.ray.io/) parallelism, designed for
efficient storage, versioning, and retrieval of time-indexed features backed by [Polars](https://pola.rs/) DataFrames.

It was built to solve the real-world pain points that teams encounter when using flat Parquet files for ML feature
storage: expensive column additions, no versioning, and painful cross-team collaboration.

## Why Gamma Lake?

The most common alternative — storing features in per-day Parquet files — breaks down quickly:

| Problem                                                  | Gamma Lake's answer                                                                                 |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Adding a new feature group rewrites *all* existing files | Gamma Lake writes one new Delta table per feature group — **O(1) cost regardless of existing data** |
| Multiple teams can't write features in parallel          | An opt-in Ray IO writer serializes index changes and pipelines independent feature writes           |
| No versioning or audit trail                             | Delta Lake's transaction log provides full time-travel and version history                          |
| Cross-group reads require expensive joins                | Gamma Lake maintains a master index; reads are aligned horizontal concatenations                    |
| Experimental features pollute production data            | Features carry `owner` and `version` metadata; `read()` accepts a filtered metadata frame           |

See [performance](https://github.com/Point72/gamma-lake/wiki/Performance) for benchmark data comparing Gamma Lake against per-day Parquet at scale.

## Key Features

- **Sortable index** — a master index table ensures all feature groups stay aligned for efficient range-filtered reads
- **Parallel reads and writes** — Ray remote functions parallelise across feature groups for large-scale workloads
- **Feature versioning** — every `add_features()` call records owner and version; read any historical version by
  filtering the metadata frame
- **Multiple signal types** — standard features, as-of features, sparse features, and runtime-computed features
- **Native Delta Lake storage** — ACID transactions, schema enforcement, and time-travel out of the box
- **Local or cloud** — `base_path` accepts a local directory, an on-prem object store path, or an `s3://` URI
- **Configurable compression** — `zstd` by default; pluggable via `compression` field

## Quick Start

```python
import tempfile
from datetime import datetime, timezone

import polars as pl
import pyarrow as pa
from ccflow import ArrowSchema

from gammalake import GammaFeatureLake

# 1. Create and initialise a feature lake
with tempfile.TemporaryDirectory() as tmp:
    lake = GammaFeatureLake(base_path=tmp, run_on_ray_cluster=False)
    lake.initialize(
        ArrowSchema.make(
            pa.schema([("timestamp", pa.timestamp("us", tz="UTC")), ("symbol", pa.large_string())])
        )
    )

    # 2. Build a toy feature DataFrame (timestamp × symbol × features)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame({
        "timestamp": pl.Series([now]).cast(pl.Datetime("us", "UTC")),
        "symbol":    ["AAPL"],
        "momentum":  [0.42],
        "vol_20d":   [0.18],
    })

    # 3. Write features
    lake.add_features(df, owner="my-team")

    # 4. Read features back
    result = lake.read(["momentum", "vol_20d"])
    print(result)

    # 5. Inspect metadata
    print(lake.feature_metadata_frame().collect())
```

Run the full annotated demo: [`examples/quickstart.py`](examples/quickstart.py)

## How Gamma Lake Works

### The index and feature groups

Gamma Lake assumes each feature row has a unique, strictly ordered tuple key. In finance, `(timestamp, symbol)` is a
common choice. A lake stores one master index Delta table and one Delta table per feature group:

![Gamma Lake index and feature groups](docs/assets/gammalake/20250728092757.png)

Dense feature groups persist rows in canonical index order over their covered range. Missing rows inside that range
are stored explicitly for alignment, while a group may remain shorter when the index grows only beyond its latest
update. Reads never need keyed joins: Polars pads any missing trailing values during horizontal concatenation. Delta
files are not assumed to be physically ordered; each source is sorted by the configured sort keys before assembly.

As-of and sparse feature tables are intentionally not padded. Their read paths first align them to the bounded master
index with as-of and left joins, respectively, before they enter the same positional assembly path.

### Reading features

A read:

1. Resolves the minimum set of feature-group tables containing the requested features.
1. Applies range and supported sort-key predicates to the master index and each source.
1. Sorts each source by the same key order.
1. Takes keys from the master index, drops duplicate keys from feature groups, and concatenates the feature values
   horizontally.

![Reading aligned feature groups using the index](docs/assets/gammalake/20250728093051.png)

This positional concatenation is the central Gamma Lake invariant: it avoids repeated key hashing and multi-way joins
as feature groups accumulate.

Feature groups do not need to end at the same index value:

![Partially updated feature groups](docs/assets/gammalake/20250728093638.png)

Polars pads shorter groups with implicit trailing nulls during horizontal concatenation:

![Concatenated feature groups with implicit nulls](docs/assets/gammalake/20250728093712.png)

The nulls are not stored in the shorter Delta tables. A useful mental model separates persisted values from the
implicit aligned tail:

![Persisted and implicit Gamma Lake values](docs/assets/gammalake/20250728093901.png)

### Adding new rows and feature groups

When incoming keys extend beyond the current range, Gamma Lake writes the feature values and appends the keys to the
master index once. Untouched dense groups may remain shorter because their missing trailing values are implicit.
Adding or updating a dense group writes explicit alignment rows for gaps inside that group's covered range.

For example, appending values that continue a partially updated group requires no other table changes:

![New values that continue an existing feature group](docs/assets/gammalake/20250728093926.png)

![Feature group after a straightforward append](docs/assets/gammalake/20250728094124.png)

If an append skips index values inside the group's covered range, Gamma Lake stores null alignment rows for those
internal gaps:

![New values with gaps in the covered range](docs/assets/gammalake/20250728094155.png)

![Feature group padded across internal gaps](docs/assets/gammalake/20250728094204.png)

Independent updates to multiple groups can run together:

![Simultaneous updates to multiple feature groups](docs/assets/gammalake/20250728094723.png)

![Feature groups after simultaneous updates](docs/assets/gammalake/20250728094834.png)

The resulting read remains positionally aligned:

![Concatenated result after simultaneous updates](docs/assets/gammalake/20250728094920.png)

New keys can also fall inside the existing master-index range:

![New index values inside the existing range](docs/assets/gammalake/20250728095001.png)

Groups covering that range receive explicit alignment rows where needed. Groups whose covered prefix ends before the
new key remain untouched:

![Feature groups after inserting a new index value](docs/assets/gammalake/20250728095036.png)

The read result combines stored values with implicit trailing nulls:

![Read result after inserting a new index value](docs/assets/gammalake/20250728095341.png)

![Stored and implicit null values after the update](docs/assets/gammalake/20250728100128.png)

Backfills can therefore be expensive. Adding a new secondary-key value across historical primary-key periods may
require alignment updates to many feature groups; adding it only for future periods does not.

If incoming rows overlap existing rows, `overlap_mode="copy"` creates a new versioned Delta table, while
`overlap_mode="merge"` performs an in-place Delta merge. Both paths preserve the alignment invariant.

Consider an update to a partially populated feature group:

![Partially populated feature group before an overlap](docs/assets/gammalake/20250728100359.png)

![Overlapping feature values](docs/assets/gammalake/20250728100441.png)

The default copy mode creates a new version while retaining the previous table:

![Versioned copy of an overlapping feature group](docs/assets/gammalake/20250728100529.png)

Merge mode instead updates the current physical table:

![Merged overlapping feature group](docs/assets/gammalake/20250728100701.png)

Default reads select the latest feature version. Copy mode preserves earlier physical tables that can be selected by
filtering the metadata frame. Merge mode reuses the current table and does not retain pre-merge values as a separate
feature version.

### Lazy predicate pushdown

Local lazy reads use `polars-io-tools` to coordinate supported sort-key predicates into the bounded index and every
feature source before positional concatenation. Runtime-computed reads defer downstream predicates until after
computation to preserve neighbour context. As-of reads retain the right-hand history required for carry-forward
semantics.

Projection-only aggregations remain correct, but a requested feature group that is later projected away must still
read one column to preserve its row count during horizontal concatenation.

### Parallel I/O

With `run_on_ray_cluster=True`, Gamma Lake dispatches feature-group operations independently through Ray. Group writes
and the single master-index update run in parallel. Metadata becomes visible only after the feature and index writes
succeed. Install Ray support with `pip install "gamma-lake[ray]"`.

### Coordinated concurrent writes (Ray only)

Set `coordinate_writes=True` together with `run_on_ray_cluster=True` to route concurrent
`add_features`, `add_targets`, `add_as_of_features`, `add_sparse_features`, and `add_index_rows`
calls through a shared `RayIOWriter` actor. The default remains uncoordinated.
Initialize the lake once, before starting concurrent producers:

```python
import ray

from gammalake import GammaFeatureLake, RayIOWriter

path = "/shared/features"
GammaFeatureLake(base_path=path).initialize()
ray.init()
lake = GammaFeatureLake(base_path=path, run_on_ray_cluster=True, coordinate_writes=True)

# Independent producers can call lake.add_features(...) as usual.
# When multiple frames are already available, submit them together:
writer = RayIOWriter.connect(lake)
ray.get(writer.submit_batch.remote(lake, [features_a, features_b], owner="my-team"))
```

Every call has its own index critical section: read the current index, classify input keys,
reserve destinations and feature versions, and append only missing keys. The writer then
schedules feature writes and dense-table padding and releases the index lane **without waiting
for either**. The next call can advance the index while previous feature writes are still running.
Calls with no new keys skip the index write.

```text
add A: [index A] ---> feature A ----------------> publish A
add B:          [index B] ---> feature B --> publish B
add C:                    [index C] ---> feature C ...
```

The writer tracks pending tables as well as published ones. A later call adding keys inside a
pending table's range schedules padding after that table's creation. Writes to the same physical
table are ordered using Ray task dependencies; independent tables proceed concurrently.
Copy-mode operations reserve their destination in advance and depend on both the source and
destination lanes. Each task uses its call's captured index relationships and reserved metadata,
not a later live index snapshot. Sparse/as-of tables do not receive padding observations.

Once a call's own feature and alignment tasks finish, it publishes table metadata followed by
feature metadata under a separate publication lock. This short metadata critical section never
holds the index lane or blocks independent data tasks. Calls may complete out of admission order.
`submit_batch` is only a convenience for submitting multiple calls: it does not combine their
indices or introduce a batch completion barrier.

All producers must use the **same Ray cluster, lake configuration, IO configuration, and shared
storage path**. The named actor uses the fixed `gammalake-writes` Ray namespace, so producers in
different application namespaces still coordinate. Do not mix coordinated and uncoordinated
writers or write directly to the underlying Delta tables. This is not a cross-cluster lock.
Storage must support safe Delta commits; the actor does not configure an S3 locking provider.

Index additions are visible before feature completion: **reads are not transactionally isolated
from writes**, and an error can leave committed index rows or feature data behind. Any write
failure stops further admission; already scheduled work may still complete, and unrelated calls
may already have published. Writes are not automatically retried. Quiesce all producers
and outstanding tasks, repair or restore the lake, then remove the failed actor with
`ray.kill(writer)` before reconnecting. Never kill/recreate the actor during live writes.
The actor survives a producer driver's exit, but does not restart after actor/cluster failure.
It does not provide durable request replay.
Cancelling a caller is not a rollback. Nonrecursive cancellation leaves admitted writes running;
recursive Ray cancellation can interrupt their tasks and requires the same recovery as a write failure.

Maintenance (`initialize`, consolidation, annotation, restoration, and runtime-feature registration)
must run offline using an uncoordinated instance. Explicit `metadata=` overrides are rejected in
coordinated mode. Successful coordinated calls return `[]` rather than individual task results.
Input keys are classified before their index commit, preserving new-key versus overlap semantics.
The writer keeps planned metadata in memory; out-of-band writes and maintenance invalidate it.
The current prototype does not impose queue or memory limits and does not persist its dependency graph.

### Metadata and versioning

Every feature write records its name, version, owner, table address, signal type, and optional feature parameters.
`read()` selects the latest version by default; pass a filtered metadata frame to select an owner or historical
version explicitly.

| Concept               | Implementation                                                    |
| --------------------- | ----------------------------------------------------------------- |
| Master index          | One Delta table containing every configured sort-key tuple        |
| Dense feature group   | Canonically ordered rows; missing trailing index values implicit  |
| As-of or sparse group | Sparse physical table aligned to the index during reads           |
| New index rows        | Extend the index once; preserve aligned prefixes                  |
| Overlapping rows      | Versioned copy or Delta merge                                     |
| Cross-group assembly  | Predicate-distributed positional horizontal concatenation         |
| Parallel I/O          | Ray task per feature group                                        |
| Versioning            | Append-only metadata records owner and version per feature column |

## Documentation

| Document                                                                    | Description                                                 |
| --------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [Architecture](#how-gamma-lake-works)                                       | Index design, feature groups, and append/merge semantics    |
| [Best practices](https://github.com/Point72/gamma-lake/wiki/Best-Practices) | Recommended versioning, update, and parallel-write patterns |
| [Performance](https://github.com/Point72/gamma-lake/wiki/Performance)       | Benchmark comparison against per-day Parquet files          |
| [docs/benchmarking.md](docs/benchmarking.md)                                | Running the portable ASV read/write benchmarks              |
| [examples/quickstart.py](examples/quickstart.py)                            | Runnable end-to-end demo                                    |

## Core API

```python
from gammalake import GammaFeatureLake

lake = GammaFeatureLake(base_path="s3://my-bucket/features")

# One-time setup
lake.initialize(schema)

# Writing
lake.add_features(df, owner="team-a")          # standard features
lake.add_targets(df, owner="team-a")           # target/label columns
lake.add_as_of_features(df, params, owner=...) # point-in-time safe features
lake.add_sparse_features(df, owner=...)        # sparse / infrequently-updated features

# Runtime features require an explicit opt-in because stored expressions are executable.
trusted_lake = GammaFeatureLake(base_path="s3://my-bucket/features", enable_runtime_computed_features=True)
trusted_lake.add_runtime_computed_features(exprs, ...)                     # arbitrary Polars expressions
trusted_lake.add_runtime_transforms(["feature_a"], ["abs", "reciprocal"])  # named row-local transforms

# Reading
lake.read(["feature_a", "feature_b"])                        # latest versions
lake.read(["feature_a"], start="2023-01-01", end="2024-01-01")  # date range
lake.read(lake.feature_metadata_frame()                      # specific owner/version
          .filter(pl.col("owner") == "team-a").collect())

# Metadata inspection
lake.feature_metadata_frame().collect()   # features, versions, owners
lake.table_metadata_frame().collect()     # Delta table details
lake.index_frame().collect()              # master index
```

## Installation

`gamma-lake` can be installed via [pip](https://pip.pypa.io) or [conda](https://docs.conda.io/en/latest/), the two primary package managers for the Python ecosystem.

To install `gamma-lake` via **pip**, run this command in your terminal:

```bash
pip install gamma-lake
```

This installs the local execution dependencies without Ray. To enable distributed execution:

```bash
pip install "gamma-lake[ray]"
```

To install `gamma-lake` via **conda**, run this command in your terminal:

```bash
conda install gamma-lake -c conda-forge
```

## Getting Started

See [our wiki!](https://github.com/Point72/gamma-lake/wiki)

## Development

Check out the [contribution guide](https://github.com/Point72/gamma-lake/wiki/Contribute) for more information.

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
