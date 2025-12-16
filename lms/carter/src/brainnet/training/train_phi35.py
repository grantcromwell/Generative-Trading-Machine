#!/usr/bin/env python3
"""
Unsloth Training Script for Phi-3.5 Quant Code Generation.

Fine-tunes microsoft/Phi-3.5-mini-instruct on:
- The Stack v1.2 (bigcode/the-stack) - Python quant code
- AlphaFin CoT - Chain-of-thought finance reasoning
- Evol-Instruct (SurgeGlobal) - Instruction amplification
- Synthetic pairs - GAF + strategy code

Target metrics:
- 85%+ HumanEval on trading functions (GAF encoders)
- 75%+ FinGPT-Benchmark on trading tasks
- 65% accuracy boost on internal evaluations

Hardware requirements:
- RTX 4090 (24GB): batch_size=4, seq_len=8K (~$40-60)
- Colab Pro T4/A100: batch_size=2, seq_len=4K (~$20-30)
- 1-2 epochs recommended

Usage:
    # Full training
    python train_phi35.py --config default
    
    # Quick test
    python train_phi35.py --config default --max_steps 100
    
    # Colab mode
    python train_phi35.py --config colab
    
    # 4090 optimized
    python train_phi35.py --config 4090

References:
- https://huggingface.co/datasets/bigcode/the-stack
- https://huggingface.co/datasets/SurgeGlobal/Evol-Instruct
- https://github.com/AlphaFin-proj/AlphaFin
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check and report on required dependencies."""
    deps = {
        'torch': False,
        'transformers': False,
        'datasets': False,
        'unsloth': False,
        'trl': False,
        'peft': False,
    }
    
    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            pass
            
    missing = [d for d, available in deps.items() if not available]
    
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.info("Install with:")
        logger.info("  pip install torch transformers datasets trl peft")
        logger.info("  pip install unsloth  # or: pip install 'unsloth[colab-new]'")
        return False
        
    return True


def setup_unsloth():
    """Setup Unsloth for efficient training."""
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        return FastLanguageModel, is_bfloat16_supported()
    except ImportError:
        logger.warning("Unsloth not available, falling back to standard training")
        return None, False


def load_model_unsloth(config: 'TrainingConfig'):
    """
    Load Phi-3.5 with Unsloth optimizations.
    
    4-bit quantization for memory efficiency,
    LoRA adapters for parameter-efficient fine-tuning.
    """
    FastLanguageModel, bf16_supported = setup_unsloth()
    
    if FastLanguageModel is None:
        raise RuntimeError("Unsloth required for efficient training")
    
    logger.info(f"Loading {config.base_model} with Unsloth...")
    logger.info(f"  4-bit: {config.load_in_4bit}")
    logger.info(f"  bf16 supported: {bf16_supported}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    logger.info("Model loaded with LoRA adapters")
    logger.info(f"  LoRA rank: {config.lora_r}")
    logger.info(f"  LoRA alpha: {config.lora_alpha}")
    logger.info(f"  Target modules: {config.target_modules}")
    
    return model, tokenizer


def load_model_standard(config: 'TrainingConfig'):
    """
    Fallback: Load model with standard transformers/peft.
    
    Use when Unsloth is not available.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    import torch
    
    logger.info(f"Loading {config.base_model} with standard transformers...")
    
    # Quantization config
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )
    else:
        bnb_config = None
        
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Add LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(config: 'TrainingConfig', tokenizer):
    """
    Prepare the mixed training dataset.
    
    Combines:
    - The Stack (Python) - 50%
    - AlphaFin CoT - 40%  
    - Synthetic pairs - 10%
    
    With deduplication and quant keyword filtering.
    """
    from .data_loader import QuantDatasetLoader
    from .filters import QuantCodeFilter
    from .dedup import MinHashDeduplicator
    from .synthetic import SyntheticPairGenerator
    from .packer import SequencePacker
    
    logger.info("Preparing training dataset...")
    
    # Initialize components
    loader = QuantDatasetLoader(config)
    code_filter = QuantCodeFilter(config)
    deduplicator = MinHashDeduplicator(config)
    synthetic_gen = SyntheticPairGenerator(config)
    
    # Collect samples
    samples = []
    
    # 1. Load The Stack (Python/TS quant code)
    logger.info("Loading The Stack...")
    try:
        stack_ds = loader.load_the_stack(languages=["python"])
        
        stack_count = 0
        for sample in stack_ds:
            if stack_count >= 100000:  # Limit for memory
                break
                
            content = sample.get('content', '')
            
            # Apply quant filter
            if not code_filter.should_include(content):
                continue
                
            # Deduplicate
            if not deduplicator.process(content):
                continue
                
            samples.append(format_for_training(content, "code"))
            stack_count += 1
            
            if stack_count % 10000 == 0:
                logger.info(f"  Processed {stack_count} Stack samples")
                
        logger.info(f"  Total Stack samples: {stack_count}")
        
    except Exception as e:
        logger.warning(f"Failed to load The Stack: {e}")
        logger.info("Continuing without The Stack data...")
    
    # 2. Load AlphaFin CoT (if available)
    logger.info("Loading AlphaFin CoT...")
    try:
        alphafin_ds = loader.load_alphafin()
        
        alphafin_count = 0
        for sample in alphafin_ds:
            if alphafin_count >= 50000:
                break
                
            content = loader._format_alphafin_sample(sample)
            
            if not deduplicator.process(content):
                continue
                
            samples.append(format_for_training(content, "cot"))
            alphafin_count += 1
            
        logger.info(f"  Total AlphaFin samples: {alphafin_count}")
        
    except Exception as e:
        logger.warning(f"Failed to load AlphaFin: {e}")
    
    # 3. Generate synthetic pairs
    logger.info("Generating synthetic pairs...")
    synthetic_count = 0
    target_synthetic = int(config.synthetic.num_pairs * 0.1)  # 10% of target
    
    for synthetic_sample in synthetic_gen.generate(target_synthetic):
        samples.append(format_for_training(
            synthetic_sample.to_training_format(),
            "synthetic"
        ))
        synthetic_count += 1
        
    logger.info(f"  Total synthetic samples: {synthetic_count}")
    
    # 4. Create HuggingFace dataset
    logger.info(f"Creating dataset with {len(samples)} total samples...")
    
    from datasets import Dataset
    
    dataset = Dataset.from_dict({"text": samples})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    logger.info(f"Dataset prepared: {len(dataset)} samples")
    logger.info(f"Filter stats: {code_filter.stats}")
    logger.info(f"Dedup stats: {deduplicator.stats}")
    
    return dataset


def format_for_training(content: str, source: str) -> str:
    """Format content for Phi-3.5 training."""
    
    if source == "code":
        return f"""<|user|>
Write Python code for the following quant trading task.

<|assistant|>
{content}<|end|>"""
    
    elif source == "cot":
        return f"""<|user|>
Solve this trading problem with step-by-step reasoning.

<|assistant|>
{content}<|end|>"""
    
    elif source == "synthetic":
        return f"""<|user|>
{content.split('### Implementation:')[0] if '### Implementation:' in content else 'Implement this trading strategy.'}

<|assistant|>
{content.split('### Implementation:')[1] if '### Implementation:' in content else content}<|end|>"""
    
    else:
        return f"{content}<|end|>"


def train(
    config: 'TrainingConfig',
    model,
    tokenizer,
    dataset,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Run Unsloth/SFT training.
    
    Uses:
    - SFTTrainer from trl for supervised fine-tuning
    - Gradient checkpointing for memory efficiency
    - Cosine LR schedule with warmup
    """
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    logger.info("Starting training...")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.effective_batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Max seq length: {config.max_seq_length}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler,
        optim=config.optimizer,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="no",  # No eval during training for speed
        save_total_limit=3,
        seed=config.seed,
        report_to="none",  # Disable wandb by default
        gradient_checkpointing=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=config.max_seq_length,
        packing=True,  # Enable sequence packing
    )
    
    # Train
    logger.info("Training started...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Log metrics
    metrics = train_result.metrics
    logger.info(f"Training completed!")
    logger.info(f"  Total steps: {metrics.get('global_step', 'N/A')}")
    logger.info(f"  Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Runtime: {metrics.get('train_runtime', 'N/A'):.2f}s")
    
    return output_dir, metrics


def export_to_gguf(model_dir: str, quantization: str = "q4_k_m"):
    """
    Export trained model to GGUF format for llama.cpp inference.
    
    Useful for deployment without Python dependencies.
    """
    try:
        from unsloth import FastLanguageModel
        
        logger.info(f"Exporting to GGUF with {quantization} quantization...")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=8192,
            dtype=None,
            load_in_4bit=True,
        )
        
        gguf_path = f"{model_dir}/model-{quantization}.gguf"
        
        model.save_pretrained_gguf(
            gguf_path,
            tokenizer,
            quantization_method=quantization,
        )
        
        logger.info(f"GGUF model saved to {gguf_path}")
        return gguf_path
        
    except Exception as e:
        logger.error(f"GGUF export failed: {e}")
        return None


def push_to_hub(model_dir: str, hub_model_id: str, private: bool = True):
    """Push trained model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        api.upload_folder(
            folder_path=model_dir,
            repo_id=hub_model_id,
            repo_type="model",
            private=private,
        )
        
        logger.info(f"Model pushed to hub: {hub_model_id}")
        
    except Exception as e:
        logger.error(f"Hub push failed: {e}")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Phi-3.5 on quant code datasets"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "colab", "4090"],
        help="Training configuration preset"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epochs)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="Export to GGUF after training"
    )
    
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hub model ID to push to"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Test setup without training"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Load config
    from .config import TrainingConfig, get_colab_config, get_4090_config
    
    if args.config == "colab":
        config = get_colab_config()
    elif args.config == "4090":
        config = get_4090_config()
    else:
        config = TrainingConfig()
    
    # Override config with CLI args
    if args.max_steps > 0:
        config.max_steps = args.max_steps
    if args.output_dir:
        config.output_dir = args.output_dir
    
    logger.info("=" * 60)
    logger.info("Brainnet/Minah Phi-3.5 Quant Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Max seq length: {config.max_seq_length}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Epochs: {config.num_epochs}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Would train with above config")
        logger.info("Dependencies: OK")
        return
    
    # Load model
    try:
        model, tokenizer = load_model_unsloth(config)
    except Exception as e:
        logger.warning(f"Unsloth failed: {e}, trying standard loading...")
        model, tokenizer = load_model_standard(config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Train
    output_dir, metrics = train(
        config,
        model,
        tokenizer,
        dataset,
        resume_from_checkpoint=args.resume,
    )
    
    # Export to GGUF if requested
    if args.export_gguf:
        export_to_gguf(output_dir)
    
    # Push to hub if requested
    if args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


