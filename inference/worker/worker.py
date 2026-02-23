"""
Per-GPU inference worker process.
Each worker is pinned to one GPU via CUDA_VISIBLE_DEVICES set by manager.py.
"""
from __future__ import annotations

import os
import signal
import structlog

from inference.worker.pipeline import InferencePipeline
from inference.worker.queue import JobQueue

log = structlog.get_logger()


class Worker:
    def __init__(self, gpu_id: int) -> None:
        self.gpu_id = gpu_id
        self.pipeline = InferencePipeline(gpu_id=gpu_id)
        self.queue = JobQueue()
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, *_) -> None:
        log.info("worker_shutdown_signal", gpu_id=self.gpu_id)
        self._running = False

    def run(self) -> None:
        log.info("worker_started", gpu_id=self.gpu_id)
        while self._running:
            job = self.queue.dequeue(timeout=1)
            if job is None:
                continue

            job_id, payload = job
            log.info("job_received", job_id=job_id, gpu_id=self.gpu_id)

            try:
                image_b64 = self.pipeline.generate(payload)
                self.queue.store_result(job_id, {"image_b64": image_b64})
                log.info("job_done", job_id=job_id, gpu_id=self.gpu_id)
            except Exception as exc:
                log.error("job_failed", job_id=job_id, gpu_id=self.gpu_id, error=str(exc))
                self.queue.store_result(job_id, {"error": str(exc)})


def main() -> None:
    gpu_id = int(os.environ.get("WORKER_GPU_ID", "0"))
    Worker(gpu_id=gpu_id).run()


if __name__ == "__main__":
    main()
