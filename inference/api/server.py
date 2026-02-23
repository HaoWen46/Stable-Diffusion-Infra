"""
FastAPI REST server for image generation requests.
Enqueues jobs to Redis and polls for results.
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from inference.api.routes import router

app = FastAPI(title="Stable Diffusion Inference API", version="0.1.0")
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("inference.api.server:app", host="0.0.0.0", port=8000, reload=False)
