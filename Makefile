.PHONY: install test lint smoke data site

install:
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff format --check .
	uv run ruff check .

smoke:
	uv run specdec smoke

data:
	uv run specdec prepare-data

site:
	uv run specdec report --run $(RUN)
	uv run specdec demo --run $(RUN) --out site/index.html $(ARMS)
