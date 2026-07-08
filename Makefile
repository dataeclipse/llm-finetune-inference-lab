.PHONY: dev test test-smoke lint typecheck fmt data train-sft train-dpo eval export serve bench up

dev:
	uv sync --group dev --extra train

test:
	uv run pytest tests -m "not smoke and not gpu" --cov=src/lab --cov-report=term-missing

test-smoke:
	uv run pytest tests -m smoke

lint:
	uv run ruff check src tests
	uv run black --check src tests

typecheck:
	uv run mypy

fmt:
	uv run ruff check --fix src tests
	uv run black src tests

data:
	uv run lab data prepare

train-sft:
	uv run lab train sft profile=colab_a100

train-dpo:
	uv run lab train dpo profile=colab_a100

eval:
	uv run lab eval run profile=colab_a100

export:
	uv run lab export awq && uv run lab export gguf

serve:
	uv run lab serve vllm

bench:
	uv run lab bench latency

up:
	docker compose -f docker-compose.gpu.yml up -d
