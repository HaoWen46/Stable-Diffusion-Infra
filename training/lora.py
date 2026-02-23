"""
LoRA injection into UNet attention layers via peft.
"""
from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


def inject_lora(unet: nn.Module, lora_config: dict) -> nn.Module:
    """
    Inject LoRA adapters into UNet attention projection layers.

    Args:
        unet: Base UNet2DConditionModel.
        lora_config: Dict with keys: rank, alpha, dropout, target_modules.

    Returns:
        peft-wrapped model with LoRA adapters; base weights frozen.
    """
    config = LoraConfig(
        r=lora_config.get("rank", 4),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get(
            "target_modules",
            ["to_q", "to_k", "to_v", "to_out.0"],
        ),
        bias="none",
    )
    model = get_peft_model(unet, config)
    model.print_trainable_parameters()
    return model


def load_lora_weights(unet: nn.Module, weights_path: str) -> nn.Module:
    """Load saved LoRA adapter weights into a base UNet (for inference hot-swap)."""
    from peft import PeftModel
    return PeftModel.from_pretrained(unet, weights_path)
