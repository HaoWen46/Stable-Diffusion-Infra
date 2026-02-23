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
    num_inference_steps: int = Field(default=20, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    job_id: str
    image_b64: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    error: Optional[str] = None
