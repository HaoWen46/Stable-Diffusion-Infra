"""
Stable Diffusion pipeline wrapper with dynamic LoRA loading.
Maintains a loaded pipeline in memory; hot-swaps LoRA without reloading base model.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional

import torch
from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel

import structlog

log = structlog.get_logger()


class InferencePipeline:
    def __init__(self, gpu_id: int) -> None:
        self.gpu_id = gpu_id
        self.device = torch.device("cuda")  # CUDA_VISIBLE_DEVICES already set by manager
        self._current_lora: Optional[str] = None
        self._pipe = self._load_base_pipeline()

    def _load_base_pipeline(self) -> ZImagePipeline:
        gguf_file = Path(os.environ.get("MODEL_DIR", "models/z-image-turbo")) / os.environ.get(
            "GGUF_FILE", "z_image_turbo-Q4_K_M.gguf"
        )
        base_model = os.environ.get("BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")

        log.info("loading_gguf_transformer", gguf=str(gguf_file), gpu_id=self.gpu_id)
        transformer = ZImageTransformer2DModel.from_single_file(
            str(gguf_file),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16,
        )

        log.info("loading_pipeline", model=base_model, gpu_id=self.gpu_id)
        pipe = ZImagePipeline.from_pretrained(
            base_model,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        return pipe

    def _maybe_swap_lora(self, lora_model_id: Optional[str]) -> None:
        if lora_model_id == self._current_lora:
            return

        artifact_dir = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))

        if self._current_lora is not None:
            self._pipe.unload_lora_weights()

        if lora_model_id is not None:
            lora_path = artifact_dir / "models" / lora_model_id
            log.info("loading_lora", lora=lora_model_id, gpu_id=self.gpu_id)
            self._pipe.load_lora_weights(str(lora_path))

        self._current_lora = lora_model_id

    def generate(self, payload: dict) -> str:
        """Run inference and return base64-encoded PNG."""
        self._maybe_swap_lora(payload.get("lora_model_id"))

        generator = None
        if payload.get("seed") is not None:
            generator = torch.Generator(device=self.device).manual_seed(payload["seed"])

        result = self._pipe(
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt", ""),
            num_inference_steps=payload.get("num_inference_steps", 9),
            guidance_scale=payload.get("guidance_scale", 0.0),
            width=payload.get("width", 1024),
            height=payload.get("height", 1024),
            generator=generator,
        )
        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
