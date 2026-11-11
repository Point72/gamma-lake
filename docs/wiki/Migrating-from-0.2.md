# Migrating from 0.2

Gamma Lake 0.3 reads and writes lakes through a Rust core instead of Python and Ray. The lake format is unchanged: 0.2
and 0.3 read and write the same lakes, and `make test-compat` checks that each engine reads what the other wrote.

## Installation

Wheels contain the compiled core and are built for Linux (x86_64 and aarch64) and macOS (arm64). Other platforms
[build from source](Build-from-Source), which needs a Rust toolchain. `ray` and `deltalake` are no longer dependencies,
and the `ray` extra is gone.

## `GammaFeatureLake` fields

Passing a removed field raises a pydantic `ValidationError` (`Extra inputs are not permitted`).

| 0.2                       | 0.3                                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `io=PolarsIO(...)`        | Removed; the core does all I/O. Object store options and credentials go in `delta_options=DeltaOptions(storage_options={...})` |
| `compression=...`         | `delta_options=DeltaOptions(compression=...)`                                                                                  |
| `target_file_size=...`    | `delta_options=DeltaOptions(target_file_size=...)`                                                                             |
| `run_on_ray_cluster=True` | Removed; writes run on the core's threads                                                                                      |

`memory_limit` and `lock` are new; see [Native Engine](Native-Engine) and "Concurrent writers" in the README.

## Methods

| 0.2                                                      | 0.3                                                                                                   |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `add_features(df, metadata=...)` and the other `add_*`   | `metadata` is removed; writes go to the latest version of each feature. `comments` and `tags` are new |
| `add_*` and `add_index_rows` with a Ray object reference | Pass a `polars.DataFrame`                                                                             |
| `switch`, `wait` and `get`                               | Removed with Ray                                                                                      |

## Modules and functions

| 0.2                                                                                  | 0.3                                                      |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| `gammalake.io` (`FrameIO`, `PolarsIO`)                                               | Removed                                                  |
| `gammalake.metadata`                                                                 | Removed; the core creates and writes the metadata tables |
| `align_feature_tables`, `update_feature_tables`, `update_index` and `write_metadata` | Removed; they were steps of the Python write path        |
| `gammalake._types.RayObjectReference`                                                | Removed                                                  |

## Errors

A write that loses a race with another writer raises `ConflictError` and writes nothing. A write that fails after some of
its tables may have committed raises `PartialWriteError`, which names them. Requests the core cannot serve raise
`UnsupportedError`. All three are importable from `gammalake`.
