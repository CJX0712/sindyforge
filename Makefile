.PHONY: install test lint fmt bench demo ci clean docker

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

bench:
	$(PY) -m sindyforge.cli run --out benchmark.json

demo:
	$(PY) examples/run_demo.py --mode default

ci: lint test
	$(PY) -m sindyforge.cli run --quick --out /tmp/bench-quick.json

docker:
	docker build -t sindyforge:0.1.0 .
	docker run --rm sindyforge:0.1.0

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
