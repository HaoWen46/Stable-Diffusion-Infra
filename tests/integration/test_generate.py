"""
Integration test: POST /generate returns a valid base64 PNG.
Requires a running API server + Redis (use docker-compose or test fixtures).
Skip if INTEGRATION_TESTS env var is not set.
"""
import base64
import os
import pytest
import requests

BASE_URL = os.environ.get("API_URL", "http://localhost:8000")


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Set INTEGRATION_TESTS=1 to run integration tests",
)
def test_generate_returns_image():
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": "a red apple on a table", "num_inference_steps": 5},
        timeout=120,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "image_b64" in data
    # Verify it's valid base64-encoded data
    raw = base64.b64decode(data["image_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Set INTEGRATION_TESTS=1 to run integration tests",
)
def test_health():
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
