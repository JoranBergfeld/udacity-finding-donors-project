.PHONY: lint format test compare notebook clean

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/ -v

compare:
	python scripts/compare_configs.py

notebook:
	jupyter nbconvert --to html --execute starter/finding_donors.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
