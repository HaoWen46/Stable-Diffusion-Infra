#!/usr/bin/env python3
"""
Generate a test image using Z-Image-Turbo GGUF.

Usage:
    uv run scripts/generate.py
    uv run scripts/generate.py --prompt "a cyberpunk city at night" --steps 12 --seed 0
    uv run scripts/generate.py --out outputs/my_image.png
"""
import argparse
import os
import time
from pathlib import Path

import torch
from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="a serene mountain lake at golden hour, photorealistic")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gguf_file = Path(os.environ.get("MODEL_DIR", "models/z-image-turbo")) / os.environ.get(
        "GGUF_FILE", "z_image_turbo-Q4_K_M.gguf"
    )
    base_model = os.environ.get("BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")

    if not gguf_file.exists():
        print(f"ERROR: GGUF file not found: {gguf_file}")
        print("       Run: make download-model")
        raise SystemExit(1)

    print(f"Loading GGUF transformer from {gguf_file} ...")
    t0 = time.perf_counter()
    transformer = ZImageTransformer2DModel.from_single_file(
        str(gguf_file),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    )

    print(f"Loading pipeline from {base_model} ...")
    pipe = ZImagePipeline.from_pretrained(
        base_model,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    load_s = time.perf_counter() - t0
    print(f"Pipeline ready in {load_s:.1f}s")

    # Memory snapshot after loading
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM allocated: {mem:.2f} GB")

    print(f"\nGenerating: '{args.prompt}'")
    t1 = time.perf_counter()
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    gen_s = time.perf_counter() - t1
    print(f"Generation done in {gen_s:.1f}s")

    # Save
    if args.out is None:
        out_path = Path("outputs") / f"seed{args.seed}_steps{args.steps}.png"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
