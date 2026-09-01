# Changelog

## Unreleased
* Add strict aligned-source concatenation using `polars-io-tools` coordinated predicate pushdown.
* Modernize GammaLake Ruff hygiene, including independent `FeatureMetadata` version lists and `TypeError` for unsupported read inputs.
* Resolve `feature_metadata` once per `GammaFeatureLake.read` and pipe it through read helpers to avoid repeated Delta-table scans.
* Breaking API cleanup: remove the unused `debug` parameter from `GammaFeatureLake.read`; reads now always return requested features plus sort keys.
* Restrict feature-table data-skipping statistics to `GammaFeatureLake` sort keys.
* Document efficient `GammaFeatureLake` writes and backfills.
* Add `GammaFeatureLake.add_index_rows` to extend the index without adding features.
