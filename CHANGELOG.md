# Changelog

## Unreleased
* [feature] Read and write lakes through a Rust core; `gammalake.native.NativeLake` exposes it directly to Python.
* [feature] The `gamma_lake` Rust crate exposes the core: `Lake` reads and writes lakes over any `TableStore` backend, and `Lake::read_async` reads without blocking.
* [feature] The Rust `Error`, `Op`, `OverlapMode` and `SignalType` enums and the option structs (`LakeOptions`, `AddOptions`, `CreateOptions`, `DeltaOptions`) are `#[non_exhaustive]`.
* [breaking] `GammaFeatureLake` always uses the Rust core. The `run_on_ray_cluster` and `io` fields are removed, along with the `switch`, `wait` and `get` methods, `gammalake.io`, `gammalake.metadata`, `RayObjectReference`, the Python write-path functions (`align_feature_tables`, `update_feature_tables`, `update_index`, `write_metadata`) and the `ray` extra. See [Migrating from 0.2](https://github.com/Point72/gamma-lake/wiki/Migrating-from-0.2).
* [breaking] `add_*` methods return `None`, accept only Polars DataFrames and no longer take `metadata`; writes go to the latest version of each feature. Their third positional argument is now `overlap_mode`.
* [breaking] `GammaFeatureLake.model_copy(update=...)` validates the update. `DEFAULT_INDEX_SCHEMA` is removed; `initialize()` without a schema uses the same default.
* [breaking] Sort keys cannot be float, time, duration or struct columns. Unsigned integer sort keys are stored as signed integers of the same width, and dictionary sort keys as strings.
* [breaking] Datetimes are stored with microsecond precision, zoned datetimes in UTC; arrays are stored as lists and categoricals and enums as strings. Time and duration features raise `UnsupportedError`, as does writing a value that the cast to a feature's stored type would change.
* [breaking] A write that conflicts with other writers on each of 10 attempts raises `ConflictError`. Two writers adding the same new feature at once no longer both record it as version 0: the one that runs again writes to it as a later write would.
* [breaking] Errors that 0.2 raised from Polars or `deltalake` are `UnsupportedError`, `ValueError` or `RuntimeError`. A lake constructed or used before a process forks raises `RuntimeError` in the child; use the `spawn` or `forkserver` start method.
* [deprecation] `GammaFeatureLake` takes Delta settings as `delta_options=DeltaOptions(compression=..., target_file_size=..., storage_options=...)`; the top-level `compression` and `target_file_size` fields still work but warn, and are moved into `delta_options`. `DeltaOptions` rejects unknown fields.
* [feature] A write commits to a table only if no other writer changed it after the write read it. Otherwise it is rolled back and run again, up to 10 times in all, so concurrent writers adding different features all succeed. A failed write is rolled back and raises the exception it failed with, noting the step that failed; `PartialWriteError` is raised only when the rollback cannot undo every commit.
* [feature] Add the `lock` option: a per-table lock factory held around every commit.
* [feature] `add_*` methods take `allow_new_index_values=False` to reject, writing nothing, an input with keys not in the index.
* [feature] `base_path` accepts `az://`, `abfss://` and `gs://` URLs as well as local paths and `file://` and `s3://` URLs.
* [bugfix] Rollback undoes only the failed write's commits, preserving other writers' commits, and reverts no table unless it can revert them all.
* [feature] Store list and struct features; copy-mode overlaps widen numeric types as Polars does.
* [feature] With the `datafusion` extra, `GammaFeatureLake.feature_table` and `NativeLake.feature_table` return features as a DataFusion table for SQL, with filters on sort keys pushed into the scan.
* [feature] The `GAMMA_LAKE_THREADS` environment variable sets the core's parallelism.
* [cleanup] Lakes written by 0.2.0 read unchanged, and the core writes parquet files with the same settings as the `deltalake` writer.
* [cleanup] `deltalake` is no longer a runtime dependency.

## 0.2.0 (09-22-26)
* [cleanup] Remove unused validation types and stale status scaffolding.
* [feature] Use `polars-io-tools` for native LazyFrame Delta writes.
* [docs] Move the architecture deep dive and diagrams into the project README.
* [bugfix] Restore positional concatenation for persisted, aligned GammaLake sources.
* [bugfix] Preserve out-of-range context required by backward, forward, and nearest as-of reads.
* [feature] Add `GammaFeatureLake.add_runtime_transforms` for deterministic row-local runtime transforms with validation and opt-in runtime-computed-feature security.
* [feature] Push sort-key predicates into the global index and eligible feature scans using `polars-io-tools`, while preserving alignment and runtime/as-of context.
* [feature] Add strict aligned-source concatenation using `polars-io-tools` coordinated predicate pushdown.
* [cleanup] Modernize GammaLake Ruff hygiene, including independent `FeatureMetadata` version lists and `TypeError` for unsupported read inputs.
* [cleanup] Resolve `feature_metadata` once per `GammaFeatureLake.read` to avoid repeated Delta-table scans.
* [cleanup] Remove the unused `debug` parameter from `GammaFeatureLake.read`; reads now always return requested features plus sort keys.
* [cleanup] Restrict feature-table data-skipping statistics to `GammaFeatureLake` sort keys.
* [cleanup] Document efficient `GammaFeatureLake` writes and backfills.
* [feature] Add `GammaFeatureLake.add_index_rows` to extend the index without adding features.
