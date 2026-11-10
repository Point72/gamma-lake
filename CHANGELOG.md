# Changelog

## Unreleased
* [breaking] Remove the standalone C/C++ API, headers, shared/static libraries and native SDK packaging; Gamma Lake is distributed as a Rust crate and Python extension.
* [feature] Read and write lakes through a Rust core; `gammalake.native.NativeLake` exposes it directly to Python.
* [breaking] `GammaFeatureLake` always uses the Rust core. The `engine`, `run_on_ray_cluster` and `io` fields are removed, along with `gammalake.io` and the `ray` extra.
* [breaking] `add_*` methods return `None` and accept only Polars DataFrames.
* [breaking] `NativeUnsupportedError` and `NativeWriteError` are renamed `UnsupportedError` and `PartialWriteError`.
* [feature] A write commits to a table only if no other writer changed it after the write read it, and otherwise raises `ConflictError` (`GAMMA_LAKE_ERR_CONFLICT` in C), or `PartialWriteError` if part of the write had already committed.
* [feature] Store list and struct features; copy-mode overlaps widen numeric types as Polars does.
* [cleanup] `deltalake` is no longer a runtime dependency.
* [feature] The `GAMMA_LAKE_THREADS` environment variable sets the core's parallelism.
* [feature] `benchmarks/compare_engines.py` compares the core with the pure-Python engine by thread count and storage.
* [cleanup] The core writes parquet files with the same settings, row order, row groups and Arrow types as the Python `deltalake` writer.
* [feature] `gammalake/tests/test_compat.py` checks that lakes written by either engine read the same in both and hold files with the same settings; `make test-compat` runs it against v0.2.0.
* [breaking] `GammaFeatureLake` takes Delta settings as `delta_options=DeltaOptions(compression=..., target_file_size=..., storage_options=...)`; the top-level `compression`, `target_file_size` and `storage_options` fields are removed. `DeltaOptions` rejects unknown fields.
* [feature] With the `datafusion` extra, `GammaFeatureLake.feature_table` and `NativeLake.feature_table` return features as a DataFusion table for SQL, with filters on sort keys pushed into the scan.
* [breaking] Rust: storage backends implement an Arrow-only `TableStore`/`StoredTable` contract whose scans return `TableScan`; the core fails a read whose scan returns columns other than the requested ones of the table's schema. `Lake::read_async` reads without blocking.
* [feature] Add the `lock` option: a per-table lock factory held around every commit.

## 0.2.0 (09-22-26)
* [cleanup] Remove unused validation types and stale status scaffolding.
* [feature] Use `polars-io-tools` for native LazyFrame Delta writes.
* [docs] Move the architecture deep dive and diagrams into the project README.
* [bugfix] Restore positional concatenation for persisted, aligned GammaLake sources.
* [bugfix] Preserve out-of-range context required by backward, forward, and nearest as-of reads.
* [feature] Add `GammaFeatureLake.add_runtime_transforms` for deterministic row-local runtime transforms with validation and opt-in runtime-computed-feature security.
* [feature] Add strict aligned-source concatenation using `polars-io-tools`.
* [cleanup] Modernize GammaLake Ruff hygiene, including independent `FeatureMetadata` version lists and `TypeError` for unsupported read inputs.
* [cleanup] Resolve `feature_metadata` once per `GammaFeatureLake.read` to avoid repeated Delta-table scans.
* [cleanup] Remove the unused `debug` parameter from `GammaFeatureLake.read`; reads now always return requested features plus sort keys.
* [cleanup] Restrict feature-table data-skipping statistics to `GammaFeatureLake` sort keys.
* [cleanup] Document efficient `GammaFeatureLake` writes and backfills.
* [feature] Add `GammaFeatureLake.add_index_rows` to extend the index without adding features.
