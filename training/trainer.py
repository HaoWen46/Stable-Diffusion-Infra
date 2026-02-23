"""
Core training loop: gradient accumulation, checkpointing, logging.
Supports DDP (LoRA) and FSDP (full UNet) depending on config.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import structlog

from training.dataset import build_dataloader
from training.lora import inject_lora
from artifacts.registry import ArtifactRegistry

log = structlog.get_logger()


class Trainer:
    def __init__(
        self,
        config_path: str,
        rank: int,
        world_size: int,
        resume_from: Optional[str] = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.config = self._load_config(config_path)
        self.resume_from = resume_from
        self.global_step = 0

        self.device = torch.device(f"cuda:{rank}")
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.dataloader = build_dataloader(self.config, rank, world_size)
        self.registry = ArtifactRegistry(Path(self.config["artifact_dir"]))

        if resume_from:
            self._load_checkpoint(resume_from)

    def _load_config(self, path: str) -> dict:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    def _build_model(self) -> nn.Module:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            self.config["base_model"], subfolder="unet"
        ).to(self.device)

        training_mode = self.config.get("training_mode", "lora")
        if training_mode == "lora":
            unet = inject_lora(unet, self.config["lora"])
            unet = DDP(unet, device_ids=[self.rank])
        elif training_mode == "full_unet":
            unet = FSDP(unet)
        else:
            raise ValueError(f"Unknown training_mode: {training_mode}")

        return unet

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-2),
        )

    def _save_checkpoint(self) -> None:
        if not self.is_main:
            return
        ckpt_dir = self.registry.checkpoint_path(
            self.config["run_id"], self.global_step
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.global_step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            ckpt_dir / "checkpoint.pt",
        )
        log.info("checkpoint_saved", step=self.global_step, path=str(ckpt_dir))

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(Path(path) / "checkpoint.pt", map_location=self.device)
        self.global_step = ckpt["step"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        log.info("checkpoint_loaded", step=self.global_step, path=path)

    def run(self) -> None:
        save_every = self.config.get("save_every_steps", 500)
        max_steps = self.config.get("max_steps", 10_000)
        grad_accum = self.config.get("gradient_accumulation_steps", 1)

        self.model.train()
        self.optimizer.zero_grad()

        for batch in self.dataloader:
            if self.global_step >= max_steps:
                break

            loss = self._training_step(batch)
            loss = loss / grad_accum
            loss.backward()

            if (self.global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.global_step += 1
            if self.is_main:
                log.info(
                    "train_step",
                    step=self.global_step,
                    loss=loss.item() * grad_accum,
                )

            if self.global_step % save_every == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        if self.is_main:
            self.registry.promote_checkpoint(
                self.config["run_id"],
                self.global_step,
                self.config,
            )

    def _training_step(self, batch: dict) -> torch.Tensor:
        # Placeholder: implement noise prediction loss
        raise NotImplementedError
