"""
Brainnet/Banis Training Module - Fine-tune DeepSeek-V3 on quant code datasets.

This module provides tools for:
- Filtering quant-relevant code from The Stack
- Chain-of-thought synthesis for finance reasoning
- MinHash deduplication for clean training data
- Sequence packing for efficient 128K context training
- LoRA/QLoRA for efficient fine-tuning of DeepSeek-V3

Target model: deepseek-ai/DeepSeek-V3-0324
- 685B MoE parameters (37B active)
- 128K context window
- Excellent at code and reasoning tasks

Usage:
    # Train with LoRA on RTX 4090 (limited due to model size)
    python -m brainnet.training.train_deepseek --config 4090
    
    # For inference-only, use the HF pipeline with 4-bit quantization
    # Training full DeepSeek-V3 requires significant compute (8x A100 minimum)
"""

from .config import (
    TrainingConfig,
    DatasetConfig,
    PackingConfig,
    DeduplicationConfig,
    SyntheticConfig,
    get_default_config,
    get_colab_config,
    get_4090_config,
)

__all__ = [
    "TrainingConfig",
    "DatasetConfig",
    "PackingConfig",
    "DeduplicationConfig",
    "SyntheticConfig",
    "get_default_config",
    "get_colab_config",
    "get_4090_config",
]

