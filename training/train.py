"""
Entry point for distributed training.
Launch with: torchrun --nproc_per_node=4 training/train.py --config <config.yaml>
"""
import argparse
import os

import torch
import torch.distributed as dist

from training.trainer import Trainer
from training.dataset import build_dataloader
from training.lora import inject_lora
from artifacts.registry import ArtifactRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint directory to resume from")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def main() -> None:
    args = parse_args()
    rank, world_size = setup_distributed()

    trainer = Trainer(
        config_path=args.config,
        rank=rank,
        world_size=world_size,
        resume_from=args.resume,
    )
    trainer.run()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
