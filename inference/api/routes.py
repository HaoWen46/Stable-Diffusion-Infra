"""
API route handlers: /generate, /health, /models.
"""
from __future__ import annotations

import uuid
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from inference.api.schemas import GenerateRequest, GenerateResponse, JobStatus
from inference.worker.queue import JobQueue

router = APIRouter()
queue = JobQueue()

POLL_TIMEOUT_S = 60
POLL_INTERVAL_S = 0.5


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    job_id = str(uuid.uuid4())
    queue.enqueue(job_id, request.model_dump())

    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        result = queue.poll_result(job_id)
        if result is not None:
            if result.get("error"):
                raise HTTPException(status_code=500, detail=result["error"])
            return GenerateResponse(job_id=job_id, image_b64=result["image_b64"])
        time.sleep(POLL_INTERVAL_S)

    raise HTTPException(status_code=504, detail="Inference timed out")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "queue_depth": queue.depth()}


@router.get("/models")
async def list_models() -> dict:
    from artifacts.registry import ArtifactRegistry
    from pathlib import Path
    import os
    registry = ArtifactRegistry(Path(os.environ.get("ARTIFACT_DIR", "/artifacts")))
    return {"models": registry.list_models()}
