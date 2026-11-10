# Gamma Lake Performance

**TL;DR:** Adding a feature group to Gamma Lake writes one new table, where wide Parquet files must all be rewritten, and
reads line feature groups up by position instead of joining them. It also provides versioning, time-travel, as-of
features, and runtime-computed features that are impractical with raw Parquet.

______________________________________________________________________

The ASV suites in `benchmarks/gammalake_benchmarks.py` time reads and writes on your own machine and storage; see
[docs/benchmarking.md](https://github.com/Point72/gamma-lake/blob/main/docs/benchmarking.md) for how to run them. The
core's parallelism is set with `GAMMA_LAKE_THREADS`; see [Native Engine](Native-Engine#threads).

______________________________________________________________________

## "I'll just use a collection of Parquet files instead."

Storing features in per-day Parquet files is a common starting point. The convenience of daily pipelines producing
daily files is intuitively obvious, and small reads are fast. However it introduces scaling problems:

1. What happens when you add a new feature group? Every existing file must be read, widened, and rewritten.
1. How can multiple teams contribute features in parallel without stepping on each other?
1. How do you version features, deprecate old ones, or perform time-travel queries?

______________________________________________________________________

## Write Cost: Adding a New Feature Group

When features are stored in a single wide Parquet table, adding a new set of features requires:

1. Reading the entire existing table into memory
1. Horizontally concatenating the new columns
1. Writing the full merged result back to disk — **for every existing file**

This means every feature addition pays a cost proportional to everything already stored. Gamma Lake avoids this
entirely: each feature group is a separate Delta table, so adding a new group writes **one new table** regardless of
how many groups already exist.

______________________________________________________________________

## Read Cost: Cross-Group Range Queries

Storing different feature groups in separate per-day files is one way to address the write-cost problem. But cross-group
reads then require joins on an identifying key, and those joins scale with the number of groups.

Gamma Lake maintains a master index that all feature groups are aligned to, so reads are horizontal concatenations —
no joins needed. The per-group scans run concurrently.

______________________________________________________________________

## What Else Does Gamma Lake Provide?

Beyond raw I/O performance, Gamma Lake addresses requirements that pure Parquet solutions struggle with:

| Capability                          | Parquet files | Gamma Lake |
| ----------------------------------- | :-----------: | :--------: |
| Add a group without rewriting files |      ❌       |     ✅     |
| Concurrent team writes              |      ❌       |     ✅     |
| Feature versioning                  |      ❌       |     ✅     |
| Time-travel queries                 |      ❌       |     ✅     |
| As-of (point-in-time safe) features |      ❌       |     ✅     |
| Runtime-computed features           |      ❌       |     ✅     |
| Sparse feature storage              |      ❌       |     ✅     |
| Granular owner/permission metadata  |      ❌       |     ✅     |
| ACID transactions                   |      ❌       |     ✅     |
