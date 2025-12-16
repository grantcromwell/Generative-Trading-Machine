#!/usr/bin/env python3
"""
Train Minah on Quant Datasets - Example Script

Fine-tunes Phi-3.5-mini on:
- The Stack v1.2 (bigcode/the-stack) - 3TB Python code, quant-filtered
- AlphaFin CoT - Chain-of-thought finance reasoning
- Evol-Instruct (SurgeGlobal) - Instruction amplification
- Synthetic pairs - GAF + strategy code

Target: 85%+ HumanEval on trading functions, 75%+ FinGPT-Benchmark

Prerequisites:
    1. pip install -r requirements-training.txt
    2. pip install unsloth  # or: pip install 'unsloth[colab-new]'
    3. huggingface-cli login
    4. Accept The Stack terms: https://huggingface.co/datasets/bigcode/the-stack

Usage:
    # Quick test (100 steps)
    python train_quant_model.py --quick
    
    # Full training on 4090
    python train_quant_model.py --preset 4090
    
    # Colab Pro
    python train_quant_model.py --preset colab
    
    # Custom config
    python train_quant_model.py --epochs 2 --batch_size 4 --lr 2e-4

Cost estimates (1-2 epochs on 100k samples):
    - RTX 4090 (local): ~$40-60 electricity
    - Colab Pro (T4): ~$10-20
    - Colab Pro+ (A100): ~$20-30

References:
    - https://huggingface.co/datasets/bigcode/the-stack
    - https://huggingface.co/datasets/SurgeGlobal/Evol-Instruct
    - https://github.com/AlphaFin-proj/AlphaFin
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_prerequisites():
    """Check all prerequisites before training."""
    print("=" * 60)
    print("Checking prerequisites...")
    print("=" * 60)
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required, got {sys.version_info}")
    else:
        print("✓ Python version OK")
    
    # Check core packages
    packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'datasets': 'datasets',
        'trl': 'trl',
        'peft': 'peft',
    }
    
    for name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {name} installed")
        except ImportError:
            issues.append(f"Missing: {name} (pip install {name})")
    
    # Check Unsloth
    try:
        from unsloth import FastLanguageModel
        print("✓ Unsloth installed")
    except ImportError:
        print("⚠ Unsloth not installed (will use standard training)")
        print("  For 2x faster training: pip install unsloth")
    
    # Check HuggingFace login
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ HuggingFace logged in")
        else:
            issues.append("Not logged in to HuggingFace (huggingface-cli login)")
    except Exception:
        issues.append("HuggingFace Hub issue")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        else:
            issues.append("No CUDA GPU available")
    except Exception as e:
        issues.append(f"GPU check failed: {e}")
    
    print()
    
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        return False
    
    print("All prerequisites OK!")
    return True


def quick_data_test():
    """Quick test of data loading."""
    print("\n" + "=" * 60)
    print("Testing data loading...")
    print("=" * 60)
    
    from datasets import load_dataset
    
    # Test The Stack
    print("\n1. Testing The Stack access...")
    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        sample = next(iter(ds))
        print(f"   ✓ The Stack accessible")
        print(f"   Sample size: {sample.get('size', 'N/A')} bytes")
    except Exception as e:
        print(f"   ✗ The Stack failed: {e}")
        print("   → Accept terms at: https://huggingface.co/datasets/bigcode/the-stack")
        return False
    
    # Test Evol-Instruct
    print("\n2. Testing Evol-Instruct access...")
    try:
        ds = load_dataset(
            "SurgeGlobal/Evol-Instruct",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        sample = next(iter(ds))
        print(f"   ✓ Evol-Instruct accessible")
        print(f"   Keys: {list(sample.keys())}")
    except Exception as e:
        print(f"   ✗ Evol-Instruct failed: {e}")
    
    print("\nData access test complete!")
    return True


def run_training(args):
    """Run the actual training."""
    from brainnet.training.config import (
        TrainingConfig, 
        get_colab_config, 
        get_4090_config
    )
    from brainnet.training.train_phi35 import (
        load_model_unsloth,
        load_model_standard,
        prepare_dataset,
        train,
        export_to_gguf,
    )
    
    # Select config preset
    if args.preset == "colab":
        config = get_colab_config()
        print("Using Colab config (batch=2, seq=4K)")
    elif args.preset == "4090":
        config = get_4090_config()
        print("Using 4090 config (batch=4, seq=8K)")
    else:
        config = TrainingConfig()
        print("Using default config")
    
    # Apply overrides
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.output_dir:
        config.output_dir = args.output_dir
        
    # Quick mode: just 100 steps
    if args.quick:
        config.max_steps = 100
        config.logging_steps = 10
        config.save_steps = 50
        print("Quick mode: 100 steps only")
    
    print(f"\nTraining configuration:")
    print(f"  Base model: {config.base_model}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_r}")
    
    # Confirmation
    if not args.yes:
        response = input("\nProceed with training? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Load model
    print("\nLoading model...")
    try:
        model, tokenizer = load_model_unsloth(config)
    except Exception as e:
        print(f"Unsloth failed ({e}), using standard loading...")
        model, tokenizer = load_model_standard(config)
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(config, tokenizer)
    
    # Train
    print("\nStarting training...")
    output_dir, metrics = train(config, model, tokenizer, dataset)
    
    # Export
    if args.export_gguf:
        print("\nExporting to GGUF...")
        export_to_gguf(output_dir)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train Minah on quant datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test
    python train_quant_model.py --quick
    
    # Full training on 4090
    python train_quant_model.py --preset 4090 -y
    
    # Custom settings
    python train_quant_model.py --epochs 2 --batch_size 2 --lr 1e-4
        """
    )
    
    parser.add_argument(
        "--preset",
        choices=["default", "colab", "4090"],
        default="default",
        help="Configuration preset"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (100 steps)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Max training steps"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory"
    )
    
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="Export to GGUF after training"
    )
    
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites"
    )
    
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Test data access only"
    )
    
    args = parser.parse_args()
    
    # Prerequisites check
    if not check_prerequisites():
        if not args.check_only:
            print("\nFix the above issues before training.")
        sys.exit(1)
    
    if args.check_only:
        sys.exit(0)
    
    # Data test
    if args.test_data:
        quick_data_test()
        sys.exit(0)
    
    # Run training
    run_training(args)


if __name__ == "__main__":
    main()


