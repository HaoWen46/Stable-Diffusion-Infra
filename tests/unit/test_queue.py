"""Unit tests for JobQueue (mocked Redis)."""
import json
import pytest
from unittest.mock import MagicMock, patch


def test_enqueue_pushes_to_redis():
    with patch("inference.worker.queue.redis.Redis") as mock_redis_cls:
        mock_r = MagicMock()
        mock_redis_cls.return_value = mock_r

        from inference.worker.queue import JobQueue
        q = JobQueue()
        q.enqueue("job-1", {"prompt": "a cat"})

        mock_r.rpush.assert_called_once()
        args = mock_r.rpush.call_args[0]
        assert args[0] == "sd:jobs"
        msg = json.loads(args[1])
        assert msg["job_id"] == "job-1"


def test_poll_result_returns_none_on_miss():
    with patch("inference.worker.queue.redis.Redis") as mock_redis_cls:
        mock_r = MagicMock()
        mock_r.get.return_value = None
        mock_redis_cls.return_value = mock_r

        from inference.worker.queue import JobQueue
        q = JobQueue()
        assert q.poll_result("missing-job") is None


def test_store_and_poll_result():
    with patch("inference.worker.queue.redis.Redis") as mock_redis_cls:
        mock_r = MagicMock()
        stored = {}

        def fake_set(key, value, ex=None):
            stored[key] = value

        def fake_get(key):
            return stored.get(key)

        mock_r.set.side_effect = fake_set
        mock_r.get.side_effect = fake_get
        mock_redis_cls.return_value = mock_r

        from inference.worker.queue import JobQueue
        q = JobQueue()
        q.store_result("job-1", {"image_b64": "abc"})
        result = q.poll_result("job-1")
        assert result == {"image_b64": "abc"}
