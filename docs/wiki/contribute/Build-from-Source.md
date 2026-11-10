`gamma-lake` is written in Python and Rust. While prebuilt wheels are provided for end users, it is also straightforward to build `gamma-lake` from either the Python [source distribution](https://packaging.python.org/en/latest/specifications/source-distribution-format/) or the GitHub repository.

- [Make commands](#make-commands)
- [Prerequisites](#prerequisites)
- [Clone](#clone)
- [Install Python dependencies](#install-python-dependencies)
- [Build](#build)
- [Lint and Autoformat](#lint-and-autoformat)
- [Testing](#testing)

## Make commands

As a convenience, `gamma-lake` uses a `Makefile` for commonly used commands. You can print the main available commands by running `make` with no arguments

```bash
> make

build                          build the library
clean                          clean the repository
fix                            run autofixers
install                        install library
lint                           run lints
test                           run the tests
```

## Prerequisites

`gamma-lake` has a few system-level dependencies which you can install from your machine package manager. Other package managers like `conda`, `nix`, etc, should also work fine.

## Clone

Clone the repo with:

```bash
git clone https://github.com/Point72/gamma-lake.git
cd gamma-lake
```

## Install Rust

Follow the instructions for [installing Rust](https://rustup.rs) for your system. The Rust core needs Rust 1.95 or later.

## Install Python dependencies

Python build and develop dependencies are specified in the `pyproject.toml`, but you can manually install them:

```bash
make requirements
```

Note that these dependencies would otherwise be installed normally as part of [PEP517](https://peps.python.org/pep-0517/) / [PEP518](https://peps.python.org/pep-0518/).

## Build

Build the python project in the usual manner:

```bash
make build
```

## Lint and Autoformat

`gamma-lake` has linting and auto formatting.

| Language | Linter      | Autoformatter | Description |
| :------- | :---------- | :------------ | :---------- |
| Python   | `ruff`      | `ruff`        | Style       |
| Python   | `ruff`      | `ruff`        | Imports     |
| Rust     | `clippy`    | `rustfmt`     | Style       |
| Markdown | `mdformat`  | `mdformat`    | Style       |
| Markdown | `codespell` |               | Spelling    |

Run all linters or formatters together:

```bash
make lint
make fix
```

The corresponding language-specific targets are `lint-py`, `lint-rs`, `lint-docs`,
`fix-py`, `fix-rs` and `fix-docs`.

## Testing

`gamma-lake` has both Python and Rust tests. The bulk of the functionality is tested in Python, which can be run via `pytest`. First, install the Python development dependencies with

```bash
make develop
```

**Python**

```bash
make test-py
```

**Rust**

```bash
make test-rs
```

runs the tests of every crate in the `rust/` workspace. To test one crate, such as the core without delta-rs, run
`cargo test -p gamma_lake_core` in `rust/`.

The parity tests in `rust/tests/read_parity.rs` read Delta lakes written by the pure-Python engine of v0.2.0 and compare
against its results. `make test-rs` first runs `make fixtures`, which extracts that engine with `git archive` and
generates the lakes under `rust/tests/fixtures`; they are not committed. Without them `cargo test` skips the parity
tests.

`make test` runs the Python and Rust tests.
