#!/usr/bin/env python3
"""
Training Script for DeepSeek-V3 Quant Code Generation.

Fine-tunes deepseek-ai/DeepSeek-V3-0324 on:
- The Stack v1.2 (bigcode/the-stack) - Python quant code
- AlphaFin CoT - Chain-of-thought finance reasoning
- Synthetic pairs - GAF + strategy code

Hardware requirements:
- DeepSeek-V3 is 685B MoE (37B active) - requires significant compute
- LoRA/QLoRA: RTX 4090 (24GB) can run with 4-bit + LoRA r=32

Usage:
    python train_deepseek.py --config 4090 --max_steps 100
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required dependencies."""
    deps = ['torch', 'transformers', 'datasets', 'peft', 'trl', 'accelerate', 'bitsandbytes']
    missing = []
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        logger.error(f"Missing: {missing}")
        logger.info("pip install torch transformers datasets trl peft accelerate bitsandbytes")
        return False
    return True


def load_model_with_lora(config):
    """Load DeepSeek-V3 with LoRA adapters."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    
    logger.info(f"Loading {config.base_model} with LoRA...")
    
    dtype = torch.bfloat16 if config.bf16 else torch.float16
    
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    if config.use_lora:
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


def format_for_training(content: str, source: str) -> str:
    """Format content for DeepSeek-V3 training."""
    if source == "code":
        return f"User: Write Python code for this quant trading task.\n\nAssistant: {content}"
    elif source == "cot":
        return f"User: Solve this trading problem with step-by-step reasoning.\n\nAssistant: {content}"
    else:
        return f"User: Implement this trading strategy.\n\nAssistant: {content}"


def train(config, model, tokenizer, dataset, resume_from_checkpoint=None):
    """Run SFT training."""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    logger.info("Starting training...")
    
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
        eval_strategy="no",
        save_total_limit=3,
        seed=config.seed,
        report_to="none",
        gradient_checkpointing=config.gradient_checkpointing,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=config.max_seq_length,
        packing=True,
    )
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    metrics = train_result.metrics
    logger.info(f"Training completed! Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    
    return output_dir, metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-V3 on quant code")
    parser.add_argument("--config", type=str, default="4090", choices=["default", "colab", "4090", "a100"])
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    from .config import TrainingConfig, get_colab_config, get_4090_config, get_a100_config
    
    if args.config == "colab":
        config = get_colab_config()
    elif args.config == "a100":
        config = get_a100_config()
    elif args.config == "4090":
        config = get_4090_config()
    else:
        config = TrainingConfig()
    
    if args.max_steps > 0:
        config.max_steps = args.max_steps
    if args.output_dir:
        config.output_dir = args.output_dir
    
    logger.info("=" * 60)
    logger.info("Brainnet/Banis DeepSeek-V3 Quant Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Max seq length: {config.max_seq_length}")
    logger.info(f"LoRA r: {config.lora_r}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Would train with above config")
        return
    
    model, tokenizer = load_model_with_lora(config)
    
    # For now, use a simple dataset
    from datasets import Dataset
    samples = [
        format_for_training("import pandas as pd\nimport numpy as np\n\ndef calculate_sharpe(returns):\n    return returns.mean() / returns.std() * np.sqrt(252)", "code"),
        format_for_training("Let's calculate the Sharpe ratio step by step:\n1. Compute mean returns\n2. Compute std of returns\n3. Annualize by sqrt(252)", "cot"),
    ]
    dataset = Dataset.from_dict({"text": samples})
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.max_seq_length, padding="max_length")
    
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    output_dir, metrics = train(config, model, tokenizer, dataset, resume_from_checkpoint=args.resume)
    
    logger.info("=" * 60)
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
