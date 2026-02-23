"""
Versioned model artifact management.
Checkpoint path: artifacts/checkpoints/<run_id>/step_<N>/
Promoted model path: artifacts/models/<model_id>/
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional


class ArtifactRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.checkpoints_dir = root / "checkpoints"
        self.models_dir = root / "models"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, run_id: str, step: int) -> Path:
        return self.checkpoints_dir / run_id / f"step_{step}"

    def promote_checkpoint(
        self,
        run_id: str,
        step: int,
        config: dict,
        model_id: Optional[str] = None,
    ) -> Path:
        """
        Promote a checkpoint to a versioned model artifact.
        Writes metadata.json atomically alongside model weights.
        """
        import subprocess
        if model_id is None:
            model_id = f"{run_id}-step{step}"

        src = self.checkpoint_path(run_id, step)
        dst = self.models_dir / model_id
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            commit_hash = "unknown"

        metadata = {
            "model_id": model_id,
            "run_id": run_id,
            "step": step,
            "base_model": config.get("base_model"),
            "training_mode": config.get("training_mode"),
            "lora_config": config.get("lora"),
            "commit_hash": commit_hash,
        }
        tmp = dst / "metadata.json.tmp"
        tmp.write_text(json.dumps(metadata, indent=2))
        tmp.rename(dst / "metadata.json")

        return dst

    def list_models(self) -> list[dict]:
        models = []
        for model_dir in sorted(self.models_dir.iterdir()):
            meta_file = model_dir / "metadata.json"
            if meta_file.exists():
                models.append(json.loads(meta_file.read_text()))
        return models

    def model_path(self, model_id: str) -> Path:
        path = self.models_dir / model_id
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        return path
