# Native Engine

Gamma Lake reads and writes lakes through a Rust core in `rust/`, built on delta-rs and DataFusion. The core is exposed
two ways:

- to Python through `GammaFeatureLake` and `gammalake.native.NativeLake`,
- to Rust as the `gamma_lake` crate.

Python exchanges Arrow record batch streams with Rust through the
[Arrow C stream interface](https://arrow.apache.org/docs/format/CStreamInterface.html).

## Using it from Python

`GammaFeatureLake` always uses the core:

```python
from datetime import date

from gammalake import GammaFeatureLake

lake = GammaFeatureLake(base_path="/data/features")
lake.add_features(df, owner="team-a")
lake.read(["feature_a", "feature_b"], start=date(2024, 1, 1), end=date(2024, 2, 1))
```

Stored features are read and written in Rust. Runtime-computed features are Polars expressions, so they are evaluated
in Python with Polars over columns the core reads; Rust callers see them in the metadata but cannot read them.
Requests the core cannot serve raise `UnsupportedError`.

`NativeLake` exposes the core directly. The `start`, `end` and `filters` arguments of its read methods are pushed into
the index and feature table scans. `scan()` returns a `LazyFrame`. Filters applied to it later that compare sort keys
with literals, test membership with `is_in`, use `is_between` or test for nulls, combined with `&`, `|` and `~`, reach the
scans, for example `pl.col("timestamp") >= start` or `pl.col("symbol").is_in(symbols)`. The rest, and filters on float
columns, run after alignment.
`scan_index()`, `scan_feature_metadata()` and `scan_table_metadata()` push the same filters, on any of their columns, into
their scans. `read_arrow()` returns a single-use Arrow C stream, which `pyarrow.RecordBatchReader.from_stream` and
`polars.DataFrame` accept.

## How reads work

Every dense feature table holds the same sort keys in the same order as the index, up to the last row it covers. A read
filters the index, scans each requested dense table with the same filter, and concatenates the batches horizontally by
position. Rows of the index past the end of a table read as nulls. Dense features need no join, which is what makes
gamma-lake reads fast. As-of and sparse features are stored with their own keys and are joined to the index.

Filters arrive as a predicate in disjunctive normal form (an OR of ANDs of column comparisons). They can only reference
sort keys, since those are the only columns shared by the index and every feature table. DataFusion executes the scan of
each table, applying the predicate and the projection to the Parquet files.

When `NativeLake` is opened with `validate=True`, a read also checks that the sort keys of every dense table it reads match
the index row by row, and fails on the first mismatch.

## How writes work

A write splits its input by the feature table that holds the latest version of each column:

- Keys after the end of everything stored append to the index and to the tables.
- New features get a new table, padded with null rows so that it lines up with the index.
- Input that overlaps stored rows replaces those features over the input's key range. In copy mode the features move to a
  new table; in merge mode the existing table is updated with a Delta merge.

The index and feature tables are written first and the metadata tables last, so a reader never sees a feature before its
data. Writes to one lake are serialized within a process; separate processes writing the same lake must coordinate.

## Files

Every table of the default backend is an unpartitioned Delta table of parquet files, compressed with zstd unless
`DeltaOptions.compression` selects another codec. Rows are in key order, except in files written by merges, and a row group holds up to
about a million rows.

## Threads

The `GAMMA_LAKE_THREADS` environment variable sets how parallel the core is. It must be a positive integer and is
read once per process, at the first lake call. It sets:

- the number of worker threads of the core's tokio runtime;
- how many sorts and other CPU-bound tasks run at once;
- DataFusion's `target_partitions` for reads, writes and merges;
- how many column groups a wide feature table is split into for reading.

When it is unset, empty or not a positive integer, the core uses the number of CPUs available to the process. It is a
parallelism setting, not a cap on threads: Delta and object-store I/O still runs on tokio's blocking pool, which is
left at its default size because Delta log replay waits there for file reads. To confine the whole process to N
cores, use CPU affinity (`taskset`) as well. Polars has its own pool, set with `POLARS_MAX_THREADS`.

## Memory

By default a read sorts each table it reads in memory, so its peak memory is about the size of the data it reads, and
more for tables stored out of key order. Writes hold their input in memory.

The `memory_limit` option (`NativeLake(memory_limit=...)`, `GammaFeatureLake(memory_limit=...)`,
`LakeOptions::memory_limit` in Rust) bounds the memory a read uses for sorting. The
index and each scan of a feature table get an equal share of the limit and are sorted one at a time with DataFusion's
external sort, which writes sorted runs to the system temporary directory (`TMPDIR`) beyond that share and merges them
back as the read streams. A scan decodes batches of fewer than `batch_size` rows when a batch would take more than an
eighth of its share, estimating 32 bytes for each value of a string, binary or nested column. Combined with a streaming
consumer, such as `NativeLake.scan()` followed by `sink_parquet`, this reads more data than fits in memory. The first
batch arrives once every scan is sorted.

The limit covers sorting only. While a scan is sorted, the Parquet reader holds the compressed data of the columns it
reads for a whole row group, and feature files hold row groups of up to a million rows. As-of and sparse feature
tables, the feature metadata and the output batches are also held outside the limit. A share too small to sort one
batch fails the read with a resources-exhausted error; for features with long strings or nested values, a smaller
`batch_size` avoids this.

## Performance

The ASV suites in `benchmarks/gammalake_benchmarks.py` time reads and writes. See
[docs/benchmarking.md](https://github.com/Point72/gamma-lake/blob/main/docs/benchmarking.md) for how to run them, and
[Performance](Performance) for results.

## Using it from Rust

`DeltaStore::new(base, DeltaOptions)` configures the default Delta backend. Pass that store with `LakeOptions` to
`gamma_lake::open` or `gamma_lake::create`. The crate also exposes `AddOptions` and the predicate types `Dnf`, `Term`
and `Op`. Reads return a `BatchStream`, which implements Arrow's `RecordBatchReader`. Run
`cargo doc --open` in `rust/` for the API documentation and an example.

The Rust code is split into three crates in one Cargo workspace:

| Crate              | Directory    | Contents                                                                         |
| :----------------- | :----------- | :------------------------------------------------------------------------------- |
| `gamma_lake_core`  | `rust/core`  | `Lake`, reads, writes, DataFusion queries, and the `TableStore` trait            |
| `gamma_lake_delta` | `rust/delta` | `DeltaStore`, which keeps tables as Delta tables on local disk or object storage |
| `gamma_lake`       | `rust`       | Both of the above, plus the `open` and `create` convenience functions            |

`gamma_lake_core` does not depend on delta-rs. A `Lake` reads and writes through any `TableStore`, such as the
in-memory `MemoryStore`, with `Lake::open_with_store` and `Lake::create_with_store`.

## Querying with DataFusion

`Lake::feature_table` returns a `FeatureTable`, a DataFusion `TableProvider` over the latest versions of some features:

```rust
use std::sync::Arc;

use gamma_lake::datafusion::prelude::SessionContext;
use gamma_lake::{DeltaOptions, DeltaStore, LakeOptions};

let store = DeltaStore::new("/data/features", DeltaOptions::default())?;
let lake = gamma_lake::open(store, LakeOptions::default())?;
let table = lake.feature_table(&["feature_a", "feature_b"], &[])?;
let ctx = SessionContext::new();
ctx.register_table("features", Arc::new(table))?;
let runtime = tokio::runtime::Runtime::new()?;
let batches = runtime.block_on(async {
    let frame = ctx.sql("SELECT * FROM features WHERE symbol = 'AAPL'").await?;
    frame.collect().await
})?;
```

The query runs on the library's own runtime, whatever executor polls it.

Python gets the same table through `feature_table()` on `GammaFeatureLake` or `NativeLake`, which datafusion-python's
`SessionContext.register_table` accepts through the `__datafusion_table_provider__` FFI.

The table is the aligned read: one row per index row, sorted by the sort keys. Rows a feature table holds past the end of
the index are dropped, or reported as an error when the lake validates reads. Comparisons, `IN` lists, `BETWEEN` and null
tests on sort keys, combined with `AND` and `OR`, are pushed into the read as inexact filters, and DataFusion still
applies them afterwards. Filters that would expand into more than 64 conjunctions are not pushed. Other filters,
including Python UDFs, run in DataFusion. The table declares its sort order (with nulls first), so DataFusion can skip
sorts and use merge joins on the sort keys.

The features and their tables are chosen when the table is created. Each query reads those tables as they are when it
runs, so appends, merges and new index rows are visible, but a new version written to a new table is not; create a new
table to see it. Dropping a query stops its read, although sorting and joining already started inside the read finish
first.
