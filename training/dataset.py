"""
Dataset loading and preprocessing for Stable Diffusion fine-tuning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image


class ImageCaptionDataset(Dataset):
    """Simple image-caption dataset from a directory of images + a captions file."""

    def __init__(self, data_dir: str, resolution: int = 512) -> None:
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        captions_file = self.data_dir / "captions.txt"
        samples = []
        with open(captions_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_name, caption = line.split("\t", 1)
                samples.append({"image": self.data_dir / img_name, "caption": caption})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "caption": sample["caption"],
        }


def build_dataloader(
    config: dict,
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset = ImageCaptionDataset(
        data_dir=config["data_dir"],
        resolution=config.get("resolution", 512),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        sampler=sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
