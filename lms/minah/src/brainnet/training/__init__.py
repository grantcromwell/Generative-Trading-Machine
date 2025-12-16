"""
Brainnet Training Module - Fine-tune Phi-3.5 on quant code datasets.

Datasets:
- The Stack v1.2 (bigcode/the-stack) - Python/TS repos with quant libs
- AlphaFin CoT - Chain-of-thought annotations for trading tasks
- Evol-Instruct (SurgeGlobal) - Instruction-tuning amplification

Training approach:
- 50/50 mix: bigcode for code syntax, AlphaFin for finance reasoning
- MinHash deduplication for quality
- Quant keyword filtering (GAF, TA-Lib, Backtrader, RL)
- Sequence packing to 4K-16K tokens
- 10% synthetic pairs: "AlphaFin CoT on GAF → bigcode-style Python strat"
- Unsloth for efficient Phi-3.5 fine-tuning

Target: 85%+ HumanEval on trading functions, 75%+ on FinGPT-Benchmark

Quick Start:
    # 1. Install training deps
    pip install -r requirements-training.txt
    pip install unsloth
    
    # 2. Login to HuggingFace
    huggingface-cli login
    
    # 3. Accept The Stack terms
    # https://huggingface.co/datasets/bigcode/the-stack
    
    # 4. Run training
    python -m brainnet.training.train_phi35 --config 4090

References:
    - https://huggingface.co/datasets/bigcode/the-stack
    - https://huggingface.co/datasets/SurgeGlobal/Evol-Instruct
    - https://github.com/AlphaFin-proj/AlphaFin
"""

# Lazy imports to avoid dependency errors when training deps not installed
def __getattr__(name):
    """Lazy import training components."""
    if name == "TrainingConfig":
        from .config import TrainingConfig
        return TrainingConfig
    elif name == "QuantDatasetLoader":
        from .data_loader import QuantDatasetLoader
        return QuantDatasetLoader
    elif name == "QuantCodeFilter":
        from .filters import QuantCodeFilter
        return QuantCodeFilter
    elif name == "MinHashDeduplicator":
        from .dedup import MinHashDeduplicator
        return MinHashDeduplicator
    elif name == "SyntheticPairGenerator":
        from .synthetic import SyntheticPairGenerator
        return SyntheticPairGenerator
    elif name == "SequencePacker":
        from .packer import SequencePacker
        return SequencePacker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrainingConfig",
    "QuantDatasetLoader",
    "QuantCodeFilter",
    "MinHashDeduplicator",
    "SyntheticPairGenerator",
    "SequencePacker",
]

