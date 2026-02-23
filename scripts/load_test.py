#!/usr/bin/env python3
"""
Concurrent load test for the inference API.

Sends N requests in parallel and reports per-request latency + total wall time.

Usage:
    uv run scripts/load_test.py                        # 4 concurrent requests
    uv run scripts/load_test.py --n 8 --steps 4        # 8 requests, 4 steps each
    uv run scripts/load_test.py --url http://host:8000
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import aiohttp

PROMPTS = [
    "a serene mountain lake at golden hour, photorealistic",
    "a neon-lit cyberpunk street at night, rain reflections",
    "an ancient temple covered in jungle vines, dramatic lighting",
    "a cozy cabin in a winter forest, warm light in windows",
    "a futuristic city skyline at dusk, flying vehicles",
    "a close-up of a red fox in autumn leaves",
    "a watercolor painting of Paris in spring",
    "a macro photograph of a dewdrop on a flower petal",
]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request_id: int,
    payload: dict,
) -> dict:
    t0 = time.perf_counter()
    try:
        async with session.post(f"{url}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            elapsed = time.perf_counter() - t0
            if resp.status == 200:
                data = await resp.json()
                return {"id": request_id, "status": "ok", "elapsed_s": elapsed, "job_id": data["job_id"]}
            else:
                text = await resp.text()
                return {"id": request_id, "status": "error", "elapsed_s": elapsed, "detail": text[:200]}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {"id": request_id, "status": "exception", "elapsed_s": elapsed, "detail": str(exc)}


async def run(url: str, n: int, steps: int, width: int, height: int, save_images: bool) -> None:
    payloads = [
        {
            "prompt": PROMPTS[i % len(PROMPTS)],
            "num_inference_steps": steps,
            "guidance_scale": 0.0,
            "width": width,
            "height": height,
            "seed": i,
        }
        for i in range(n)
    ]

    print(f"Sending {n} concurrent requests to {url}/generate")
    print(f"  steps={steps}  size={width}×{height}")
    print()

    t_wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, i, p) for i, p in enumerate(payloads)]
        results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - t_wall_start

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]

    print("─" * 60)
    for r in sorted(results, key=lambda x: x["id"]):
        mark = "✓" if r["status"] == "ok" else "✗"
        detail = f"  job={r.get('job_id', '')[:8]}" if r["status"] == "ok" else f"  {r.get('detail', '')[:60]}"
        print(f"  [{mark}] request {r['id']:2d}  {r['elapsed_s']:6.1f}s{detail}")
    print("─" * 60)
    if ok:
        latencies = [r["elapsed_s"] for r in ok]
        print(f"  Succeeded : {len(ok)}/{n}")
        print(f"  Failed    : {len(errors)}/{n}")
        print(f"  Latency   : min={min(latencies):.1f}s  avg={sum(latencies)/len(latencies):.1f}s  max={max(latencies):.1f}s")
    print(f"  Wall time : {wall_time:.1f}s  (sequential estimate: {sum(r['elapsed_s'] for r in ok):.1f}s)")
    if len(ok) > 1:
        sequential_est = sum(r["elapsed_s"] for r in ok)
        speedup = sequential_est / wall_time
        print(f"  Parallelism speedup: {speedup:.2f}×")
    print()

    if errors:
        print("Errors:")
        for r in errors:
            print(f"  request {r['id']}: {r['status']} — {r.get('detail', '')}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=4, help="Number of concurrent requests")
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.url, args.n, args.steps, args.width, args.height, args.save_images))
