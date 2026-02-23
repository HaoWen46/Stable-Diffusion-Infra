"""
Pydantic request/response schemas for the inference API.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    lora_model_id: Optional[str] = None
    num_inference_steps: int = Field(default=9, ge=1, le=150)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=30.0)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    job_id: str
    image_b64: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    error: Optional[str] = None
