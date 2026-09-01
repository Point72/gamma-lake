# Changelog

## Unreleased
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
