"""Unit tests for ArtifactRegistry."""
import json
import pytest
from pathlib import Path

from artifacts.registry import ArtifactRegistry


def test_checkpoint_path(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    path = registry.checkpoint_path("run-001", 500)
    assert path == tmp_path / "checkpoints" / "run-001" / "step_500"


def test_list_models_empty(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    assert registry.list_models() == []


def test_model_path_missing(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.model_path("nonexistent")


def test_list_models_with_metadata(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True)
    meta = {"model_id": "test-model", "run_id": "run-001", "step": 100}
    (model_dir / "metadata.json").write_text(json.dumps(meta))

    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["model_id"] == "test-model"
