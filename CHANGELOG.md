# Changelog

## Unreleased
* [CCRT-7084] Breaking API cleanup: remove the unused `debug` parameter from `GammaFeatureLake.read`; reads now always return requested features plus sort keys.
* [CCRT-7083] Restrict feature-table data-skipping statistics to `GammaFeatureLake` sort keys.
* [CCRT-7082] Document efficient `GammaFeatureLake` writes and backfills.
* [CCRT-7081] Add `GammaFeatureLake.add_index_rows` to extend the index without adding features.
