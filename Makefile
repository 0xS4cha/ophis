.PHONY: install install-editable test run clean docker-build docker-run

install:
	uv sync

install-editable:
	uv pip install -e .

test:
	uv run pytest

run:
	uv run ophis

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf dist build *.egg-info

docker-build:
	docker build -t ophis:latest .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data ophis:latest
