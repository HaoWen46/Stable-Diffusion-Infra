# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-ready infrastructure for training and serving Stable Diffusion models on a single 4×4090 machine. See `PROJECT_SPEC.md` for the full specification.

## Intended Repository Layout

```
stable-diffusion-infra/
├── training/
│   ├── train.py                 # Entry point: DDP/FSDP launcher
│   ├── trainer.py               # Training loop, gradient accumulation, checkpointing
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── lora.py                  # LoRA injection and management
│   └── config/
│       ├── lora.yaml
│       └── full_unet.yaml
├── inference/
│   ├── api/
│   │   ├── server.py            # FastAPI REST server
│   │   ├── routes.py            # /generate, /health, /models endpoints
│   │   └── schemas.py           # Request/response Pydantic models
│   ├── worker/
│   │   ├── worker.py            # Per-GPU inference worker process
│   │   ├── pipeline.py          # SD pipeline wrapper with dynamic LoRA loading
│   │   └── queue.py             # Redis-backed request/result queue
│   └── manager.py               # Spawns/monitors 4 worker processes (one per GPU)
├── artifacts/
│   ├── registry.py              # Versioned model artifact management
│   └── storage.py               # Local filesystem + optional remote storage
├── monitoring/
│   ├── metrics.py               # Prometheus metrics (GPU util, queue depth, latency)
│   └── logging.py               # Structured logging config
├── docker/
│   ├── Dockerfile.training
│   ├── Dockerfile.inference
│   └── docker-compose.yml       # Redis + API + workers
├── tests/
│   ├── unit/
│   └── integration/
├── requirements-training.txt
├── requirements-inference.txt
└── Makefile
```

## Core Architecture

### Training Pipeline
- **Distributed strategy**: `torchrun` with DDP (default) or FSDP for full UNet fine-tuning; `--nproc_per_node=4`
- **LoRA**: injected into UNet attention layers via `peft`; rank/alpha configurable per YAML
- **Checkpointing**: save every N steps to `artifacts/checkpoints/<run_id>/`; resume via `--resume <checkpoint_path>`
- **Artifact versioning**: on training completion, promote checkpoint to `artifacts/models/<model_id>/` with a `metadata.json` (base model, LoRA config, training run ID, commit hash)

### Inference Architecture
- **API layer** (`inference/api/`): FastAPI server, accepts `/generate` POST, enqueues job to Redis, polls for result
- **Worker layer** (`inference/worker/`): 4 persistent worker processes, each pinned to one GPU via `CUDA_VISIBLE_DEVICES`; pull jobs from Redis queue; maintain a loaded pipeline in memory
- **Dynamic LoRA loading**: workers support hot-swapping LoRA weights without reloading the base model (track current loaded LoRA per worker, reload only on mismatch)
- **Queue**: Redis lists; job payload includes prompt, LoRA model ID, generation params; result stored as Redis key with TTL

### GPU Allocation
- Training: all 4 GPUs via DDP/FSDP (exclusive)
- Inference: 1 worker process per GPU (1:1 mapping); `manager.py` sets `CUDA_VISIBLE_DEVICES=<i>` before forking worker `i`

## Key Commands

> These are the intended commands once the codebase is built.

```bash
# Install dependencies (uses uv)
make install            # uv sync --all-extras

# Training
make train-lora
make train-full
# Internally: uv run torchrun --nproc_per_node=4 training/train.py --config <config>

# Resume training
uv run torchrun --nproc_per_node=4 training/train.py \
  --config training/config/lora.yaml \
  --resume artifacts/checkpoints/<run_id>/step_1000/

# Inference (local dev)
make serve              # docker compose up --build

# Run tests
make test                                              # unit tests
uv run pytest tests/unit/                             # unit tests only
uv run pytest tests/unit/ -k "test_registry"          # single test
make test-integration                                 # requires running stack

# Lint / format
make lint               # ruff check + mypy
make fmt                # ruff format
```

## Design Decisions & Constraints

- **No unnecessary abstraction**: avoid generic base classes or plugin registries unless reuse is demonstrated; keep training and inference fully separated packages
- **FSDP only for full UNet**: LoRA fine-tuning uses DDP (simpler, lower overhead); FSDP engaged when `training_mode: full_unet` in config
- **Redis as queue**: preferred over in-process queues for durability and multi-process isolation; single-machine deployment means no network overhead
- **Workers are persistent processes**: avoid cold-start latency; use `multiprocessing.Process` (not threads) so each worker owns its CUDA context cleanly
- **Docker images are split**: `Dockerfile.training` (heavy: includes training deps, may use NCCL) vs `Dockerfile.inference` (lighter: diffusers + FastAPI + redis-py only)
- **Artifact metadata.json** must be written atomically (write to `.tmp` then `rename`) to avoid corrupt reads during serving

## Monitoring

- Prometheus metrics exported at `:9090/metrics`: `sd_queue_depth`, `sd_inference_latency_seconds` (histogram), `sd_worker_gpu_util`
- Structured JSON logs via `structlog`; training logs include `step`, `loss`, `lr`, `gpu_id`
- Worker crashes are caught by `manager.py` watchdog, which respawns the worker on that GPU

## Tech Stack

| Layer | Library |
|---|---|
| Training distributed | `torchrun` (PyTorch DDP/FSDP) |
| LoRA | `peft` |
| Diffusion models | `diffusers`, `transformers` |
| Inference API | `FastAPI` + `uvicorn` |
| Queue | `Redis` (via `redis-py`) |
| Containerization | Docker + `docker-compose` |
| Metrics | `prometheus-client` |
| Logging | `structlog` |
| Lint/type check | `ruff`, `mypy` |
| Testing | `pytest` |
