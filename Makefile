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
	uv run specdec report --run runs/final/v2
	uv run specdec demo --run runs/final/v2 --out site/index.html \
		--arm 'target=Large model alone' \
		--arm 'fixed/distilled-8/k12=Fixed length 12' \
		--arm 'adaptive/distilled-8=Adaptive'
