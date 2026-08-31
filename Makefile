#########
# BUILD #
#########
.PHONY: develop build install

develop:  ## install dependencies and build library
	uv pip install -e .[develop]

requirements:  ## install prerequisite python build requirements
	python -m pip install --upgrade pip toml
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["build-system"]["requires"]))'`
	python -m pip install `python -c 'import toml; c = toml.load("pyproject.toml"); print(" ".join(c["project"]["optional-dependencies"]["develop"]))'`

build:  ## build the python library
	python -m build -n

install:  ## install library
	uv pip install .

#########
# LINTS #
#########
.PHONY: lint-py lint-docs fix-py fix-docs lint lints fix format

lint-py:  ## lint python with ruff
	python -m ruff check gammalake benchmarks
	python -m ruff format --check gammalake benchmarks

lint-docs:  ## lint docs with mdformat and codespell
	python -m mdformat --check README.md docs/wiki/
	python -m codespell_lib README.md docs/wiki/

fix-py:  ## autoformat python code with ruff
	python -m ruff check --fix gammalake benchmarks
	python -m ruff format gammalake benchmarks

fix-docs:  ## autoformat docs with mdformat and codespell
	python -m mdformat README.md docs/wiki/
	python -m codespell_lib --write README.md docs/wiki/

lint: lint-py lint-docs  ## run all linters
lints: lint
fix: fix-py fix-docs  ## run all autoformatters
format: fix

################
# Other Checks #
################
.PHONY: check-dist check-types checks check

check-dist:  ## check python sdist and wheel with check-dist
	check-dist -v

check-types:  ## check python types with ty
	ty check --python $$(which python)

checks: check-dist

# Alias
check: checks

#########
# TESTS #
#########
.PHONY: test coverage tests

test:  ## run python tests
	python -m pytest -v gammalake/tests

coverage:  ## run tests and collect test coverage
	python -m pytest -v gammalake/tests --cov=gammalake --cov-report term-missing --cov-report xml

# Alias
tests: test

###########
# VERSION #
###########
.PHONY: show-version patch minor major

show-version:  ## show current library version
	@bump-my-version show current_version

patch:  ## bump a patch version
	@bump-my-version bump patch

minor:  ## bump a minor version
	@bump-my-version bump minor

major:  ## bump a major version
	@bump-my-version bump major

########
# DIST #
########
.PHONY: dist dist-build dist-sdist dist-local-wheel publish

dist-build:  # build python dists
	python -m build -w -s

dist-check:  ## run python dist checker with twine
	python -m twine check dist/*

dist: clean dist-build dist-check  ## build all dists

publish: dist  ## publish python assets

#########
# CLEAN #
#########
.PHONY: deep-clean clean

deep-clean: ## clean everything from the repository
	git clean -fdx

clean: ## clean the repository
	rm -rf .coverage coverage cover htmlcov logs build dist *.egg-info

############################################################################################

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
