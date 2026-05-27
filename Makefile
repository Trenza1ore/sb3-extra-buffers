# Copyright (c) Hugo Huang. 2026.
.PHONY: install, docs docs-dev docs-clean format rst-table test release lint docstring amend binary clean mcpb
.DEFAULT_GOAL := install

format:
	ruff check --fix || true
	ruff check --select I --fix || true
	ruff format || true

lint:
	@mypy -p sb3_extra_buffers

test:
	@pytest tests/

docstring:
	@pydocstyle sb3_extra_buffers/

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
