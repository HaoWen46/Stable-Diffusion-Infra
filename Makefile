.PHONY: install train-lora train-full serve test test-unit test-integration lint fmt help

CONFIG ?= training/config/lora.yaml

## Setup
install:
	uv sync --all-extras

## Training
train-lora:
	uv run torchrun --nproc_per_node=4 training/train.py --config training/config/lora.yaml

train-full:
	uv run torchrun --nproc_per_node=4 training/train.py --config training/config/full_unet.yaml

train-resume:
	uv run torchrun --nproc_per_node=4 training/train.py --config $(CONFIG) --resume $(RESUME)

## Inference
serve:
	docker compose -f docker/docker-compose.yml up --build

serve-down:
	docker compose -f docker/docker-compose.yml down

## Testing
test: test-unit

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	INTEGRATION_TESTS=1 uv run pytest tests/integration/ -v

## Code quality
lint:
	uv run ruff check .
	uv run mypy training/ inference/ artifacts/ monitoring/

fmt:
	uv run ruff format .

## Help
help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | awk -F: '{print $$1}' | sort
