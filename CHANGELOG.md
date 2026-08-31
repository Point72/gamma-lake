# Changelog

## Unreleased
* [CCRT-7089] Add `GammaFeatureLake.add_runtime_transforms` for deterministic row-local runtime transforms with Float64 numeric semantics, validation, and opt-in runtime-computed-feature security.
* [CCRT-7088] Coordinate local sort-key predicate pushdown across GammaLake sources using `polars-io-tools`, while preserving alignment and runtime/as-of context.
* [CCRT-7087] Add strict aligned-source concatenation using `polars-io-tools` coordinated predicate pushdown.
* [CCRT-7086] Modernize GammaLake Ruff hygiene, including independent `FeatureMetadata` version lists and `TypeError` for unsupported read inputs.
* [CCRT-7085] Resolve `feature_metadata` once per `GammaFeatureLake.read` and pipe it through read helpers to avoid repeated Delta-table scans.
* [CCRT-7084] Breaking API cleanup: remove the unused `debug` parameter from `GammaFeatureLake.read`; reads now always return requested features plus sort keys.
* [CCRT-7083] Restrict feature-table data-skipping statistics to `GammaFeatureLake` sort keys.
* [CCRT-7082] Document efficient `GammaFeatureLake` writes and backfills.
* [CCRT-7081] Add `GammaFeatureLake.add_index_rows` to extend the index without adding features.
