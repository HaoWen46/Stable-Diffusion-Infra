#!/usr/bin/env python3
"""
Download Z-Image-Turbo GGUF transformer + base pipeline (text encoder, VAE, scheduler).

Usage:
    uv run scripts/download_model.py
    GGUF_FILE=z_image_turbo-Q4_K_S.gguf uv run scripts/download_model.py
"""
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

GGUF_REPO = os.environ.get("GGUF_REPO", "jayn7/Z-Image-Turbo-GGUF")
GGUF_FILE = os.environ.get("GGUF_FILE", "z_image_turbo-Q4_K_M.gguf")
BASE_MODEL = os.environ.get("BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/z-image-turbo"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def download_gguf() -> Path:
    print(f"[1/2] Downloading GGUF: {GGUF_REPO}/{GGUF_FILE}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
    )
    size_gb = Path(path).stat().st_size / 1e9
    print(f"    -> {path}  ({size_gb:.2f} GB)")
    return Path(path)


def download_base_pipeline() -> Path:
    """
    Fetch base pipeline config + components (text encoder, VAE, tokenizer, scheduler).
    We skip the transformer weights entirely since we use the GGUF version.
    """
    print(f"[2/2] Downloading base pipeline: {BASE_MODEL}")
    print("      (skipping transformer/*.safetensors — using GGUF instead)")
    hf_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_cache) / "hub"
    path = snapshot_download(
        repo_id=BASE_MODEL,
        cache_dir=str(cache_dir),
        token=HF_TOKEN,
        ignore_patterns=[
            # Skip full-precision transformer weights — we use GGUF
            "transformer/*.safetensors",
            "transformer/*.bin",
            # Skip assets (sample images, etc.)
            "assets/*",
        ],
    )
    print(f"    -> {path}")
    return Path(path)


if __name__ == "__main__":
    gguf_path = download_gguf()
    base_path = download_base_pipeline()
    print("\nAll done.")
    print(f"  GGUF:     {gguf_path}")
    print(f"  Pipeline: {base_path}")
    print(f"\nNext: make generate")
