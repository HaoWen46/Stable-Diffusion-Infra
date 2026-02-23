"""
Redis-backed job queue for routing inference requests to workers.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import redis

QUEUE_KEY = "sd:jobs"
RESULT_TTL_S = 300  # 5 minutes


class JobQueue:
    def __init__(self) -> None:
        self._redis = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            decode_responses=True,
        )

    def enqueue(self, job_id: str, payload: dict) -> None:
        message = json.dumps({"job_id": job_id, "payload": payload})
        self._redis.rpush(QUEUE_KEY, message)

    def dequeue(self, timeout: int = 1) -> Optional[tuple[str, dict]]:
        """Blocking pop with timeout. Returns (job_id, payload) or None."""
        result = self._redis.blpop(QUEUE_KEY, timeout=timeout)
        if result is None:
            return None
        _, raw = result
        msg = json.loads(raw)
        return msg["job_id"], msg["payload"]

    def store_result(self, job_id: str, result: dict) -> None:
        key = f"sd:result:{job_id}"
        self._redis.set(key, json.dumps(result), ex=RESULT_TTL_S)

    def poll_result(self, job_id: str) -> Optional[dict]:
        key = f"sd:result:{job_id}"
        raw = self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def depth(self) -> int:
        return self._redis.llen(QUEUE_KEY)
