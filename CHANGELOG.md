# Changelog

## Unreleased
* [CCRT-7086] Modernize GammaLake Ruff hygiene, including independent `FeatureMetadata` version lists and `TypeError` for unsupported read inputs.
* [CCRT-7085] Resolve `feature_metadata` once per `GammaFeatureLake.read` and pipe it through read helpers to avoid repeated Delta-table scans.
* [CCRT-7084] Breaking API cleanup: remove the unused `debug` parameter from `GammaFeatureLake.read`; reads now always return requested features plus sort keys.
* [CCRT-7083] Restrict feature-table data-skipping statistics to `GammaFeatureLake` sort keys.
* [CCRT-7082] Document efficient `GammaFeatureLake` writes and backfills.
* [CCRT-7081] Add `GammaFeatureLake.add_index_rows` to extend the index without adding features.
