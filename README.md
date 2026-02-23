# Stable Diffusion Infra

Production-ready infrastructure for training and serving Stable Diffusion models on a single 4×RTX 4090 machine.

## What's Inside

| Component | Description |
|---|---|
| **Training** | DDP/FSDP distributed training via `torchrun`; LoRA or full UNet fine-tuning |
| **Inference** | FastAPI server + 4 persistent GPU workers backed by a Redis job queue |
| **Artifact registry** | Versioned model/checkpoint management with atomic metadata writes |
| **Monitoring** | Prometheus metrics + structured JSON logs via `structlog` |
| **Z-Image-Turbo** | Lumina2 6B GGUF deployment; working, generates 1024×1024 in ~7s on a single 4090 |

## Requirements

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/)
- CUDA 12.4, 4× RTX 4090 (24 GB each)
- Docker + `docker compose` (for the inference stack)
- Redis (provided via Docker)

## Quick Start

### 1. Install

```bash
cp config/.env.example config/.env   # edit HF_HOME, CUDA_VISIBLE_DEVICES, etc.
make install                          # uv sync --all-extras
```

### 2. Generate an image (Z-Image-Turbo GGUF)

```bash
make download-model          # ~5 GB download
make generate                # outputs/seed42_steps9.png

# Custom prompt
make generate-custom PROMPT="a neon city at rain" STEPS=12 SEED=7
```

Benchmark (single RTX 4090): load 13.4 s · 13.35 GB VRAM · 9 steps → 7.3 s

### 3. Train

```bash
# LoRA fine-tuning (4 GPUs, DDP)
make train-lora

# Full UNet (4 GPUs, FSDP)
make train-full

# Resume from checkpoint
make train-resume RESUME=artifacts/checkpoints/<run_id>/step_1000/
```

Edit `training/config/lora.yaml` or `training/config/full_unet.yaml` before running.

### 4. Serve (REST API)

```bash
make serve        # docker compose up --build (Redis + API + 4 GPU workers)
make serve-down
```

API at `http://localhost:8000`. Prometheus metrics at `:9090/metrics`.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox in a snowy forest", "num_inference_steps": 9}'
```

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
inference/
├── api/           FastAPI server — receives POST /generate, enqueues to Redis
├── worker/        Per-GPU inference worker — blpop from Redis, run pipeline
└── manager.py     Spawns 4 workers (CUDA_VISIBLE_DEVICES=0..3), watchdog-respawns on crash

training/
├── train.py       torchrun entry point (--nproc_per_node=4)
├── trainer.py     Training loop, gradient accumulation, checkpointing
├── dataset.py     ImageCaptionDataset + DistributedSampler
└── lora.py        peft LoRA injection / weight loading

artifacts/
├── registry.py    Promote checkpoint → versioned model; atomic metadata.json write
└── storage.py     Local filesystem + optional S3 backend
```

Workers maintain a loaded pipeline in memory and hot-swap LoRA adapters without reloading the base model.

## Development

```bash
make test          # uv run pytest tests/unit/
make lint          # ruff check + mypy
make fmt           # ruff format
```

Single test: `uv run pytest tests/unit/ -k "test_registry"`

Integration tests (requires live stack): `make test-integration`
