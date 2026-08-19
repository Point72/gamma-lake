# Gamma Lake Performance vs Collections of Parquet Files

**TL;DR:** Gamma Lake uses O(1) write cost per feature group addition (vs O(N) for wide Parquet rewrites) and
sub-10-second reads for datasets up to 250 GB using parallel Ray reads. It also provides versioning, time-travel,
as-of features, and runtime-computed features that are impractical with raw Parquet.

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

**Benchmark: cost to add 1,000 features to an existing feature store**
_(Ray cluster, 1,000 features/group × 500 symbols × 180 days)_

| Existing feature groups | Parquet rewrite cost | Gamma Lake add cost |
| ----------------------: | -------------------: | ------------------: |
|                       1 |                4.6 s |               6.4 s |
|                       2 |                9.1 s |               6.2 s |
|                       3 |               14.8 s |               5.9 s |
|                       5 |               29.8 s |               5.8 s |
|                       7 |               53.5 s |               6.3 s |
|                      10 |              101.3 s |               6.1 s |

Parquet cost grows ~22× from 1 to 10 groups. Gamma Lake cost stays flat at ~6 s.

______________________________________________________________________

## Read Cost: Cross-Group Range Queries

Storing different feature groups in separate per-day files is one way to address the write-cost problem. But cross-group
reads then require joins on an identifying key, and those joins scale with the number of groups.

Gamma Lake maintains a master index that all feature groups are aligned to, so reads are horizontal concatenations —
no joins needed. Ray parallelises the per-group scans.

**Benchmark: read latency by window size and number of feature groups**
_(2,000 symbols × 1,000 features/group, S3 backend)_

|         Groups |  1 day | 1 week | 1 month | 3 months | 6 months |
| -------------: | -----: | -----: | ------: | -------: | -------: |
|    **Parquet** |        |        |         |          |          |
|              1 | 0.12 s | 0.86 s |   2.8 s |    8.9 s |   17.4 s |
|              5 | 0.52 s | 3.88 s |  16.9 s |   53.1 s |  132.8 s |
|             10 | 1.26 s | 9.44 s |  39.3 s |    114 s |    228 s |
|             20 | 2.81 s | 18.5 s |  81.0 s |    242 s |    488 s |
| **Gamma Lake** |        |        |         |          |          |
|              1 | 1.50 s | 1.25 s |   2.7 s |    5.6 s |   10.2 s |
|              5 | 4.04 s | 3.94 s |  12.6 s |   32.8 s |   99.9 s |
|             10 | 8.26 s | 8.42 s |  24.9 s |   57.9 s |    110 s |
|             20 | 14.9 s | 15.0 s |  44.3 s |    109 s |    197 s |

**Key findings:**

- For 1-day reads with few groups, Parquet is faster (lower Delta log overhead)
- Gamma Lake pulls ahead at ≥ 1-week windows with ≥ 5 groups
- At 20 groups and 6-month windows, Gamma Lake is **2.5× faster** than Parquet

______________________________________________________________________

## Large Dataset Read Performance

**Benchmark: 1-day read latency at scale**
_(Ray cluster, parallel reads with `run_on_ray_cluster=True`)_

| Dataset size | Features | Symbols | Periods | Gamma Lake 1-day read |
| -----------: | -------: | ------: | ------: | --------------------: |
|       1.0 GB |    1,000 |     100 |   1,250 |                0.87 s |
|       4.9 GB |    1,000 |     500 |   1,250 |                1.56 s |
|       9.8 GB |   50,000 |     100 |     250 |                2.62 s |
|      19.6 GB |   10,000 |     100 |   2,500 |                4.33 s |
|      48.9 GB |   50,000 |     100 |   1,250 |                6.28 s |
|      97.9 GB |   10,000 |     500 |   2,500 |                7.01 s |
|     244.6 GB |   50,000 |     500 |   1,250 |                5.94 s |

Sub-10-second 1-day reads up to 250 GB. Larger windows scale proportionally.

______________________________________________________________________

## What Else Does Gamma Lake Provide?

Beyond raw I/O performance, Gamma Lake addresses requirements that pure Parquet solutions struggle with:

| Capability                          | Parquet files | Gamma Lake |
| ----------------------------------- | :-----------: | :--------: |
| O(1) feature group addition         |      ❌       |     ✅     |
| Concurrent team writes              |      ❌       |     ✅     |
| Feature versioning                  |      ❌       |     ✅     |
| Time-travel queries                 |      ❌       |     ✅     |
| As-of (point-in-time safe) features |      ❌       |     ✅     |
| Runtime-computed features           |      ❌       |     ✅     |
| Sparse feature storage              |      ❌       |     ✅     |
| Granular owner/permission metadata  |      ❌       |     ✅     |
| Parallel reads via Ray              |      ❌       |     ✅     |
| ACID transactions                   |      ❌       |     ✅     |
