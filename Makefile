# Copyright (c) Hugo Huang. 2026.
.PHONY: docs docs-dev docs-clean format test release lint docstring amend binary clean mcpb
.DEFAULT_GOAL := format

SPHINX_BUILD ?= sphinx-build
DOCS_SOURCE_DIR := docs
DOCS_BUILD_ROOT := $(DOCS_SOURCE_DIR)/_build
DOCS_HTML_DIR := $(DOCS_BUILD_ROOT)/html
DOCS_EPUB_DIR := $(DOCS_BUILD_ROOT)/epub
DOCS_SINGLEHTML_DIR := $(DOCS_BUILD_ROOT)/singlehtml

# Shared doctree cache outside each builder output to avoid EPUB packager unknown-mimetype warnings.
DOCS_DOCTREES_DIR := $(DOCS_BUILD_ROOT)/doctrees

docs-dev: docs-clean
	uv run --group docs $(SPHINX_BUILD) -d $(DOCS_DOCTREES_DIR) -b html $(DOCS_SOURCE_DIR) $(DOCS_HTML_DIR)

docs: docs-dev
	@uv run --group docs python -c 'from webbrowser import open; from pathlib import Path; open(f"file://{Path.cwd().resolve()}/docs/_build/html/index.html")' || true
	uv run --group docs $(SPHINX_BUILD) -d $(DOCS_DOCTREES_DIR) -b epub $(DOCS_SOURCE_DIR) $(DOCS_EPUB_DIR)
	uv run --group docs $(SPHINX_BUILD) -d $(DOCS_DOCTREES_DIR) -b singlehtml $(DOCS_SOURCE_DIR) $(DOCS_SINGLEHTML_DIR)

docs-clean:
	rm -rf $(DOCS_SOURCE_DIR)/_build

format:
	uv run --group lint python -m ruff check --fix || true
	uv run --group lint python -m ruff format || true

lint:
	@uv run --group lint python -m mypy -p sb3_extra_buffers

test:
	@DISABLE_TEST_OBSERVATIONS_SAVE=1 uv run --group test python -m pytest tests/

docstring:
	@uv run --group lint python -m ruff check --fix --select D sb3_extra_buffers/

amend:
	git commit --amend --no-edit

clean:
	rm -rf build/
	rm -rf dist/

release: badge
	@$(if $(strip $(VERSION)),:,$(error VERSION is required, e.g. make release VERSION=1.2.3))
	printf '%s\n' "$(VERSION)" > sb3_extra_buffers/version.txt
	git add pyproject.toml sb3_extra_buffers/version.txt README*.md
	git commit -m "chore: bump version to $(VERSION)"
	git tag -a "$(VERSION)" -m "$(VERSION)"
	git push --tags
	git push

# Mutate badge with uuid to force refresh GitCode cache
badge:
	python scripts/mutate_badge.py
