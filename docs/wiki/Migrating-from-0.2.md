# Migrating from 0.2

Gamma Lake 0.3 reads and writes lakes through a Rust core instead of Python and Ray. The Delta layout is unchanged: 0.2
and 0.3 open the same lakes, and `make test-compat` checks that each engine reads what the other wrote. 0.3 stores some
column types differently and rejects others; see [Column types](#column-types).

## Installation

Wheels contain the compiled core and are built for Linux (x86_64 and aarch64) and macOS (arm64). Other platforms
[build from source](Build-from-Source), which needs a Rust toolchain. `ray` and `deltalake` are no longer dependencies,
and the `ray` extra is gone.

## `GammaFeatureLake` fields

Passing a removed field raises a pydantic `ValidationError` (`Extra inputs are not permitted`). `GammaFeatureLake` is
frozen: set fields when constructing it, or use `model_copy(update=...)`. `compression` and
`target_file_size` still work, with a `DeprecationWarning`, and are moved into `delta_options`.

| 0.2                       | 0.3                                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `io=PolarsIO(...)`        | Removed; the core does all I/O. Object store options and credentials go in `delta_options=DeltaOptions(storage_options={...})` |
| `compression=...`         | Deprecated; `delta_options=DeltaOptions(compression=...)`                                                                      |
| `target_file_size=...`    | Deprecated; `delta_options=DeltaOptions(target_file_size=...)`                                                                 |
| `run_on_ray_cluster=True` | Removed; writes run on the core's threads                                                                                      |

`memory_limit` and `lock` are new; see [Native Engine](Native-Engine) and "Concurrent writers" in the README.

## Methods

| 0.2                                                      | 0.3                                                                                                   |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `add_features(df, metadata=...)` and the other `add_*`   | `metadata` is removed; writes go to the latest version of each feature. `comments` and `tags` are new |
| `add_features(df, owner, metadata, overlap_mode)`        | The third positional argument is now `overlap_mode`; pass arguments after `df` by keyword             |
| `add_*` and `add_index_rows` with a Ray object reference | Pass a `polars.DataFrame`                                                                             |
| `switch`, `wait` and `get`                               | Removed with Ray                                                                                      |

## Modules and functions

| 0.2                                                                                  | 0.3                                                            |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| `gammalake.io` (`FrameIO`, `PolarsIO`)                                               | Removed                                                        |
| `gammalake.metadata`                                                                 | Removed; the core creates and writes the metadata tables       |
| `DEFAULT_INDEX_SCHEMA`                                                               | Removed; `initialize()` without a schema uses the same default |
| `align_feature_tables`, `update_feature_tables`, `update_index` and `write_metadata` | Removed; they were steps of the Python write path              |
| `gammalake._types.RayObjectReference`                                                | Removed                                                        |

## Column types

The core stores Arrow types that Delta supports, and converts or rejects the rest:

| Column                                     | 0.3                                                                                                              |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Sort key: float, time, duration or struct  | Rejected by `initialize`                                                                                         |
| Sort key: unsigned integer                 | Stored as the signed integer of the same width                                                                   |
| Sort key: dictionary or categorical        | Stored as string                                                                                                 |
| Sort key or feature: `ns` or `ms` datetime | Stored as `us`; a value with sub-microsecond precision raises `UnsupportedError`                                 |
| Sort key or feature: zoned datetime        | Stored as UTC                                                                                                    |
| Feature: time or duration                  | Rejected with `UnsupportedError`                                                                                 |
| Feature: array                             | Stored as list                                                                                                   |
| Feature: categorical or enum               | Stored as string                                                                                                 |
| Feature written with a new type            | Cast to the stored type; a value the cast would change, such as `3.5` into an integer, raises `UnsupportedError` |

## Errors

A write that loses a race with another writer raises `ConflictError`. Its commits are rolled back, and tables it
created are left unreferenced. Every write commits to the shared metadata tables without retrying, so two writers
conflict even when they add different features; in 0.2 both succeeded. Retry the write that raised. A write that fails otherwise is
rolled back and, as in 0.2, raises the exception it failed with; a note names the step that failed. If the rollback cannot
undo every commit, because another writer has since rewritten a table, the write raises `PartialWriteError`, which names
the tables that may hold part of it. Requests the core cannot serve raise `UnsupportedError`. All three are importable from
`gammalake`. Errors that 0.2 raised from Polars or `deltalake` are now `UnsupportedError`, `ValueError` or
`RuntimeError`.

The core's threads do not survive `fork`, so a lake used before a process forks raises `RuntimeError` in the child.
Start child processes with the `spawn` or `forkserver` method.
