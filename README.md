# Stable-Diffusion-Infra

Production-ready infrastructure for training and serving Stable Diffusion models on a single 4×RTX 4090 machine.

## What's Inside

| Component | Description |
|---|---|
| **Training** | DDP/FSDP distributed training via `torchrun`; LoRA or full UNet fine-tuning |
| **Inference** | FastAPI server + persistent GPU workers backed by a Redis job queue |
| **Artifact registry** | Versioned model/checkpoint management with atomic metadata writes |
| **Monitoring** | Prometheus metrics + structured JSON logs via `structlog` |
| **Z-Image-Turbo** | Lumina2 6B GGUF deployment; generates 1024×1024 in ~7 s on a single RTX 4090 |

## Requirements

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/)
- CUDA 12.4, 4× RTX 4090 (24 GB each)
- Redis (via Docker or conda: `conda install -c conda-forge redis-server`)

## Repository Layout

```
Stable-Diffusion-Infra/
├── training/
│   ├── train.py               # torchrun entry point (--nproc_per_node=4)
│   ├── trainer.py             # Training loop, gradient accumulation, checkpointing
│   ├── dataset.py             # ImageCaptionDataset + DistributedSampler
│   ├── lora.py                # peft LoRA injection / weight loading
│   └── config/
│       ├── lora.yaml
│       └── full_unet.yaml
├── inference/
│   ├── api/
│   │   ├── server.py          # FastAPI app + uvicorn entry point
│   │   ├── routes.py          # POST /generate, GET /health, GET /models
│   │   └── schemas.py         # Pydantic request/response models
│   ├── worker/
│   │   ├── worker.py          # Per-GPU worker process (blpop loop)
│   │   ├── pipeline.py        # ZImagePipeline wrapper + LoRA hot-swap
│   │   └── queue.py           # Redis-backed job/result queue
│   └── manager.py             # Spawns workers (one per GPU), watchdog-respawns on crash
├── artifacts/
│   ├── registry.py            # Promote checkpoint → versioned model; atomic metadata.json
│   └── storage.py             # Local filesystem + optional remote backend
├── monitoring/
│   ├── metrics.py             # Prometheus metrics (queue depth, latency, GPU util)
│   └── logging.py             # structlog structured JSON config
├── scripts/
│   ├── download_model.py      # Download Z-Image-Turbo GGUF from HuggingFace
│   ├── generate.py            # Single-image generation (no server needed)
│   ├── serve_local.sh         # Start Redis + workers + API without Docker
│   └── load_test.py           # Async concurrent load tester (aiohttp)
├── docker/
│   ├── Dockerfile.inference
│   ├── Dockerfile.training
│   └── docker-compose.yml     # Redis + API + workers
├── tests/
│   ├── unit/                  # pytest, no live stack required
│   └── integration/           # requires running inference stack
├── config/
│   ├── .env.example
│   └── .env                   # gitignored; copy from .env.example
├── models/z-image-turbo/      # GGUF file stored here (gitignored)
├── outputs/                   # Generated images
├── pyproject.toml
└── Makefile
```

## Quick Start

### 1. Install

```bash
cp config/.env.example config/.env   # edit HF_HOME, CUDA_VISIBLE_DEVICES, etc.
make install                          # uv sync --all-extras
```

### 2. Generate an image (no server needed)

```bash
make download-model          # ~5 GB download
make generate                # → outputs/seed42_steps9.png

# Custom prompt
make generate-custom PROMPT="a neon city at rain" STEPS=12 SEED=7
```

Benchmark (single RTX 4090): load 13.4 s · 13.35 GB VRAM · 9 steps → 7.3 s/image

### 3. Train

```bash
make train-lora                                                  # LoRA, 4 GPUs DDP
make train-full                                                  # full UNet, 4 GPUs FSDP
make train-resume RESUME=artifacts/checkpoints/<run_id>/step_1000/
```

Edit `training/config/lora.yaml` or `training/config/full_unet.yaml` before running.

### 4. Serve (REST API)

**With Docker:**
```bash
make serve        # docker compose up --build
make serve-down
```

**Without Docker (local dev):**
```bash
make serve-local  # starts Redis + 2 GPU workers + FastAPI on port 9000
```

```bash
curl -X POST http://localhost:9000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox in a snowy forest", "num_inference_steps": 9}'
```

Prometheus metrics at `:9090/metrics` (Docker) or configure separately for local.

### 5. Load test

```bash
make load-test                                         # 4 concurrent requests
uv run scripts/load_test.py --url http://localhost:9000 --n 8
```

Observed throughput: 4 requests → 26 s wall / 2.4× speedup · 8 requests → 39 s / 4.7× speedup

## Configuration

All runtime config lives in `config/.env` (gitignored; template at `config/.env.example`):

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | — | HuggingFace cache directory |
| `HF_TOKEN` | — | Optional, for gated models |
| `CUDA_VISIBLE_DEVICES` | `1` | GPU(s) to use for local generation |
| `GGUF_REPO` | `jayn7/Z-Image-Turbo-GGUF` | GGUF model repo |
| `GGUF_FILE` | `z_image_turbo-Q4_K_M.gguf` | Quantization variant |
| `BASE_MODEL` | `Tongyi-MAI/Z-Image-Turbo` | Base pipeline repo |
| `MODEL_DIR` | `./models/z-image-turbo` | Local GGUF storage path |

## Architecture

```
POST /generate
      │
      ▼
 FastAPI (inference/api/)
      │  enqueue job → Redis list "sd:jobs"
      │  poll result ← Redis key (5-min TTL)
      │
 ┌────┴────┐
 │  Worker │  × N  (inference/worker/)
 │  GPU 0  │       each: blpop → pipeline.generate() → rpush result
 │  GPU 1  │       LoRA hot-swap without reloading base model
 │  ...    │
 └─────────┘
```

Workers are persistent processes pinned to a single GPU via `CUDA_VISIBLE_DEVICES`. `manager.py` spawns them and respawns any that crash.

Training uses all 4 GPUs via `torchrun` (DDP for LoRA, FSDP for full UNet). Training and inference are fully separated packages with no shared state.

## Development

```bash
make test                                          # uv run pytest tests/unit/
uv run pytest tests/unit/ -k "test_registry"      # single test
make test-integration                             # requires live stack
make lint                                         # ruff check + mypy
make fmt                                          # ruff format
```
