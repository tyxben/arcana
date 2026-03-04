.PHONY: install lint format typecheck test test-cov clean all

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/arcana/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=arcana --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info

all: lint typecheck test
