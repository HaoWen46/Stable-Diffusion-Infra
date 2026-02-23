"""
Spawns and monitors 4 per-GPU worker processes.
Respawns crashed workers automatically.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import structlog

log = structlog.get_logger()

NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
RESTART_DELAY_S = 5


def _worker_target(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["WORKER_GPU_ID"] = str(gpu_id)
    from inference.worker.worker import main
    main()


class WorkerManager:
    def __init__(self, num_gpus: int = NUM_GPUS) -> None:
        self.num_gpus = num_gpus
        self._processes: dict[int, mp.Process] = {}

    def _spawn(self, gpu_id: int) -> mp.Process:
        p = mp.Process(target=_worker_target, args=(gpu_id,), daemon=True)
        p.start()
        log.info("worker_spawned", gpu_id=gpu_id, pid=p.pid)
        return p

    def run(self) -> None:
        mp.set_start_method("spawn", force=True)
        for i in range(self.num_gpus):
            self._processes[i] = self._spawn(i)

        while True:
            for gpu_id, proc in list(self._processes.items()):
                if not proc.is_alive():
                    log.warning("worker_died", gpu_id=gpu_id, exit_code=proc.exitcode)
                    time.sleep(RESTART_DELAY_S)
                    self._processes[gpu_id] = self._spawn(gpu_id)
            time.sleep(1)


if __name__ == "__main__":
    WorkerManager().run()
