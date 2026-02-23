"""
Storage backend abstraction: local filesystem (default) + optional remote (S3, GCS).
Plugged in via STORAGE_BACKEND env var.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def upload(self, local_path: Path, remote_key: str) -> None: ...

    @abstractmethod
    def download(self, remote_key: str, local_path: Path) -> None: ...

    @abstractmethod
    def exists(self, remote_key: str) -> bool: ...


class LocalStorage(StorageBackend):
    """No-op storage — artifacts already live on local disk."""

    def upload(self, local_path: Path, remote_key: str) -> None:
        pass

    def download(self, remote_key: str, local_path: Path) -> None:
        pass

    def exists(self, remote_key: str) -> bool:
        return Path(remote_key).exists()


class S3Storage(StorageBackend):
    def __init__(self, bucket: str) -> None:
        import boto3
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def upload(self, local_path: Path, remote_key: str) -> None:
        self.s3.upload_file(str(local_path), self.bucket, remote_key)

    def download(self, remote_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, remote_key, str(local_path))

    def exists(self, remote_key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=remote_key)
            return True
        except Exception:
            return False


def get_storage() -> StorageBackend:
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if backend == "local":
        return LocalStorage()
    elif backend == "s3":
        return S3Storage(bucket=os.environ["S3_BUCKET"])
    else:
        raise ValueError(f"Unknown STORAGE_BACKEND: {backend}")
