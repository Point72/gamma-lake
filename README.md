# Gamma Lake

High-performance feature store built on Delta Lake and Ray

[![Build Status](https://github.com/Point72/gamma-lake/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/Point72/gamma-lake/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/Point72/gamma-lake/branch/main/graph/badge.svg)](https://codecov.io/gh/Point72/gamma-lake)
[![License](https://img.shields.io/github/license/Point72/gamma-lake)](https://github.com/Point72/gamma-lake)
[![PyPI](https://img.shields.io/pypi/v/gamma-lake.svg)](https://pypi.python.org/pypi/gamma-lake)

## Overview

Gamma Lake is a **feature store** built on [Delta Lake](https://delta.io/) and [Ray](https://www.ray.io/), designed for
efficient storage, versioning, and retrieval of time-indexed features backed by [Polars](https://pola.rs/) DataFrames.

It was built to solve the real-world pain points that teams encounter when using flat Parquet files for ML feature
storage: expensive column additions, no versioning, and painful cross-team collaboration.

## Why Gamma Lake?

The most common alternative — storing features in per-day Parquet files — breaks down quickly:

| Problem                                                  | Gamma Lake's answer                                                                                 |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Adding a new feature group rewrites *all* existing files | Gamma Lake writes one new Delta table per feature group — **O(1) cost regardless of existing data** |
| Multiple teams can't write features in parallel          | Each feature group is an independent Delta table — concurrent writes don't conflict                 |
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
lake.add_runtime_computed_features(exprs, ...) # features computed at read time

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
