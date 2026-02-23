"""Unit tests for API request/response schemas."""
import pytest
from pydantic import ValidationError

from inference.api.schemas import GenerateRequest


def test_generate_request_defaults():
    req = GenerateRequest(prompt="a dog")
    assert req.num_inference_steps == 20
    assert req.guidance_scale == 7.5
    assert req.width == 512
    assert req.height == 512
    assert req.lora_model_id is None
    assert req.seed is None


def test_generate_request_clamps_steps():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", num_inference_steps=0)

    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", num_inference_steps=200)


def test_generate_request_custom():
    req = GenerateRequest(
        prompt="a cat",
        negative_prompt="blurry",
        lora_model_id="my-lora-v1",
        num_inference_steps=50,
        seed=42,
    )
    assert req.lora_model_id == "my-lora-v1"
    assert req.seed == 42
