"""
Prometheus metrics: GPU utilization, queue depth, inference latency.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

inference_latency = Histogram(
    "sd_inference_latency_seconds",
    "End-to-end inference latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

queue_depth = Gauge(
    "sd_queue_depth",
    "Number of pending inference jobs in the queue",
)

worker_gpu_util = Gauge(
    "sd_worker_gpu_util",
    "GPU utilization percentage",
    labelnames=["gpu_id"],
)

jobs_total = Counter(
    "sd_jobs_total",
    "Total inference jobs processed",
    labelnames=["status"],  # "success" | "error"
)


def start_metrics_server(port: int = 9090) -> None:
    start_http_server(port)


def update_gpu_metrics(gpu_id: int, utilization: float) -> None:
    worker_gpu_util.labels(gpu_id=str(gpu_id)).set(utilization)
