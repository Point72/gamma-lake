.PHONY: develop-py develop-rs develop
develop-py:
	uv pip install -e .[develop]

develop-rs:
	make -C rust develop

develop: develop-rs develop-py  ## setup project for development

.PHONY: requirements-py requirements-rs requirements
requirements-py:  ## install prerequisite python build requirements
	uv pip install -r pyproject.toml --extra develop

requirements-rs:  ## install prerequisite rust build requirements
	make -C rust requirements

requirements: requirements-rs requirements-py  ## setup project for development

.PHONY: build-py build-rs build
build-py:
	python -m build -w -n

build-rs:
	make -C rust build

build: build-rs build-py  ## build the project

.PHONY: install
install:  ## install python library
	uv pip install .

.PHONY: lint-py lint-rs lint-docs lint lints
lint-py:  ## run python linter with ruff
	python -m ruff check gammalake benchmarks
	python -m ruff format --check gammalake benchmarks

lint-rs:  ## run rust linter
	make -C rust lint
	cargo clippy --all-features
	cargo fmt -- --check

lint-docs:  ## lint docs with mdformat and codespell
	python -m mdformat --check README.md $(wildcard docs/src) docs/wiki/
	python -m codespell_lib README.md $(wildcard docs/src) docs/wiki/

lint: lint-rs lint-py lint-docs  ## run project linters

lints: lint

.PHONY: fix-py fix-rs fix-docs fix format
fix-py:  ## fix python formatting with ruff
	python -m ruff check --fix gammalake benchmarks
	python -m ruff format gammalake benchmarks

fix-rs:  ## fix rust formatting
	make -C rust fix
	cargo fmt

fix-docs:  ## autoformat docs with mdformat and codespell
	python -m mdformat README.md $(wildcard docs/src) docs/wiki/
	python -m codespell_lib --write README.md $(wildcard docs/src) docs/wiki/

fix: fix-rs fix-py fix-docs  ## run project autoformatters

format: fix

.PHONY: check-dist check-types checks check

check-dist:  ## check python sdist and wheel with check-dist
	check-dist -v

check-types:  ## check python types with ty
	python -m ty check

checks: check-dist

check: checks

.PHONY: test-py tests-py coverage-py
test-py:  ## run python tests
	python -m pytest -v gammalake/tests

tests-py: test-py

coverage-py:  ## run python tests and collect test coverage
	python -m pytest -v gammalake/tests --cov=gammalake --cov-report term-missing --cov-report xml

.PHONY: test-rs tests-rs coverage-rs
test-rs: fixtures  ## run rust tests
	make -C rust test

tests-rs: test-rs

coverage-rs: fixtures  ## run rust tests and collect test coverage
	make -C rust coverage

# The pure-Python engine of COMPAT_REF, extracted into OLD_SRC. CI checkouts are shallow, without tags.
COMPAT_REF ?= v0.2.0
OLD_SRC := $(CURDIR)/build/old-engine
define extract-old-engine
	git rev-parse -q --verify "$(COMPAT_REF)^{commit}" >/dev/null || git fetch --no-tags origin tag $(COMPAT_REF)
	rm -rf $(OLD_SRC) && mkdir -p $(OLD_SRC)
	git archive $(COMPAT_REF) gammalake | tar -x -C $(OLD_SRC)
endef

.PHONY: test-compat
test-compat:  ## compare reads and written files with the pure-Python engine of COMPAT_REF (default v0.2.0)
	$(extract-old-engine)
	GAMMA_LAKE_OLD_SRC=$(OLD_SRC) python -m pytest -v gammalake/tests/test_compat.py

FIXTURES := rust/tests/fixtures/queries.json
.PHONY: fixtures
fixtures: $(FIXTURES)  ## generate the Rust parity fixtures with the pure-Python engine of COMPAT_REF

$(FIXTURES): rust/tests/fixtures/make_fixtures.py
	$(extract-old-engine)
	PYTHONPATH=$(OLD_SRC) python rust/tests/fixtures/make_fixtures.py

.PHONY: test coverage tests
test: test-py test-rs  ## run all tests

coverage: coverage-py coverage-rs  ## run all tests and collect test coverage

tests: test

.PHONY: show-version patch minor major

show-version:  ## show current library version
	@bump-my-version show current_version

patch:  ## bump a patch version
	@bump-my-version bump patch

minor:  ## bump a minor version
	@bump-my-version bump minor

major:  ## bump a major version
	@bump-my-version bump major

.PHONY: dist-py-wheel dist-py-sdist dist-rs dist-check dist publish

dist-py-wheel:  ## build python wheel
	python -m cibuildwheel --output-dir dist

dist-py-sdist:  ## build python sdist
	python -m build --sdist -o dist

dist-rs:  ## build rust dists
	make -C rust dist

dist-check:  ## run python dist checker with twine
	python -m twine check dist/*

dist: clean dist-rs dist-py-wheel dist-py-sdist dist-check  ## build all dists

publish: dist  ## publish python assets

.PHONY: deep-clean clean

deep-clean: ## clean everything from the repository
	git clean -fdx

clean: ## clean the repository
	rm -rf .coverage coverage cover htmlcov logs build dist *.egg-info


.PHONY: help

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'

ASV_CONFIG := $(CURDIR)/benchmarks/asv.conf.json

.PHONY: benchmark-init
benchmark-init:  ## initialize ASV machine information
	python -m asv machine --config $(ASV_CONFIG) --verbose --yes

.PHONY: benchmark-check
benchmark-check:  ## check ASV benchmark discovery
	python -m asv check --python=same --config $(ASV_CONFIG)

.PHONY: benchmark
benchmark:  ## run ASV benchmarks against HEAD
	python -m asv run --python=same --config $(ASV_CONFIG) --verbose --set-commit-hash HEAD

.PHONY: benchmark-quick
benchmark-quick:  ## run each ASV benchmark once
	python -m asv run --quick --python=same --config $(ASV_CONFIG) --verbose --set-commit-hash HEAD

.PHONY: benchmark-ci
benchmark-ci:  ## run benchmarks for CI, allowing ASV timeout exit status
	python -m asv run --python=same --config $(ASV_CONFIG) --verbose --set-commit-hash HEAD || [ $$? -le 1 ]

.PHONY: benchmark-publish
benchmark-publish:  ## generate an HTML report from ASV results
	@if find $(CURDIR)/.asv/results -name '*.json' -print -quit 2>/dev/null | grep -q .; then \
		python -m asv publish --config $(ASV_CONFIG); \
	else \
		echo "No benchmark results found, skipping publish"; \
	fi

.PHONY: benchmark-view
benchmark-view: benchmark-publish  ## preview the ASV HTML report
	python -m asv preview --config $(ASV_CONFIG)

.PHONY: benchmark-clean
benchmark-clean:  ## remove generated benchmark data and reports
	rm -rf $(CURDIR)/.asv/benchmark-data $(CURDIR)/.asv/results $(CURDIR)/.asv/html
