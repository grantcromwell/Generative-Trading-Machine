#!/usr/bin/env python3
"""
Brainnet/Minah - Phi-3.5 Quant Training Script for Google Colab

Copy this script to a Colab notebook and run cell by cell.
Or upload and run: `!python train_phi35_colab.py`

Requirements:
- Google Colab Pro/Pro+ with GPU (T4 minimum, A100 recommended)
- HuggingFace account with The Stack access

References:
- https://huggingface.co/datasets/bigcode/the-stack
- https://huggingface.co/datasets/SurgeGlobal/Evol-Instruct
"""

# ============================================
# CELL 1: Check GPU & Install Dependencies
# ============================================
# !nvidia-smi
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes
# !pip install -q datasets huggingface-hub datasketch

# ============================================
# CELL 2: Login to HuggingFace
# ============================================
# from huggingface_hub import login
# login()  # Enter your token

# ============================================
# CELL 3: Configuration
# ============================================
CONFIG = {
    # Model
    "base_model": "microsoft/Phi-3.5-mini-instruct",
    "max_seq_length": 4096,  # Use 8192 on A100
    "load_in_4bit": True,
    
    # LoRA
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    
    # Training
    "batch_size": 2,  # Use 4 on A100
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "warmup_ratio": 0.03,
    
    # Data
    "max_samples": 50000,
    "bigcode_ratio": 0.50,
    "synthetic_ratio": 0.10,
    
    # Quant keywords
    "quant_keywords": [
        "gaf", "gramian", "talib", "backtrader", "zipline",
        "sharpe", "sortino", "drawdown", "volatility",
        "signal", "indicator", "macd", "rsi", "bollinger",
        "backtest", "strategy", "portfolio", "position_size",
        "yfinance", "ccxt", "ohlcv", "candlestick",
        "gymnasium", "stable_baselines", "reward", "agent",
    ],
    
    # Output
    "output_dir": "./phi35-quant",
}

# ============================================
# CELL 4: Load Model with Unsloth
# ============================================
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["base_model"],
    max_seq_length=CONFIG["max_seq_length"],
    dtype=None,
    load_in_4bit=CONFIG["load_in_4bit"],
    trust_remote_code=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"Model loaded: {CONFIG['base_model']}")

# ============================================
# CELL 5: Load & Filter The Stack
# ============================================
from datasets import load_dataset
import re
import random

quant_patterns = [re.compile(rf'\b{kw}\b', re.IGNORECASE) 
                  for kw in CONFIG["quant_keywords"]]

def is_quant_code(content):
    matches = sum(1 for p in quant_patterns if p.search(content))
    return matches >= 2

def format_for_training(content):
    return f"""<|user|>
Write Python code for the following quant trading task.

<|assistant|>
{content}<|end|>"""

print("Loading The Stack (Python)...")
print("Note: Accept terms at https://huggingface.co/datasets/bigcode/the-stack")

stack_ds = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

samples = []
processed = 0
target = int(CONFIG["max_samples"] * CONFIG["bigcode_ratio"])

for sample in stack_ds:
    processed += 1
    content = sample.get('content', '')
    lines = content.split('\n')
    
    if len(lines) < 10 or len(lines) > 1000:
        continue
    
    if is_quant_code(content):
        samples.append(format_for_training(content))
        
    if len(samples) >= target:
        break
        
    if processed % 50000 == 0:
        print(f"  Processed {processed}, found {len(samples)} quant samples")

print(f"Collected {len(samples)} quant code samples")

# ============================================
# CELL 6: Generate Synthetic Samples
# ============================================
GAF_TEMPLATES = [
    '''from pyts.image import GramianAngularField
import numpy as np

def create_gaf_image(prices, image_size=64):
    gaf = GramianAngularField(image_size=image_size, method='summation')
    X = prices.reshape(1, -1)
    return gaf.fit_transform(X)[0]
''',
    '''import talib
import numpy as np

def calculate_rsi_signal(prices, period=14):
    rsi = talib.RSI(prices, timeperiod=period)
    return np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
''',
]

num_synthetic = int(CONFIG["max_samples"] * CONFIG["synthetic_ratio"])
synthetic_samples = []

for _ in range(num_synthetic):
    code = random.choice(GAF_TEMPLATES)
    sample = f"""<|user|>
Implement a trading signal/strategy.

<|assistant|>
Let me implement this step by step:
1. Import required libraries
2. Define the function
3. Apply proper logic

{code}<|end|>"""
    synthetic_samples.append(sample)

all_samples = samples + synthetic_samples
random.shuffle(all_samples)
print(f"Total samples: {len(all_samples)}")

# ============================================
# CELL 7: Create Dataset & Train
# ============================================
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

train_dataset = Dataset.from_dict({"text": all_samples})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    packing=True,
    args=TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        seed=42,
        report_to="none",
    ),
)

print("Training...")
trainer_stats = trainer.train()

print(f"\nTraining complete!")
print(f"Final loss: {trainer_stats.training_loss:.4f}")

# ============================================
# CELL 8: Save Model
# ============================================
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"Model saved to {CONFIG['output_dir']}")

# Export GGUF
model.save_pretrained_gguf(
    f"{CONFIG['output_dir']}/gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF exported!")

# ============================================
# CELL 9: Test
# ============================================
FastLanguageModel.for_inference(model)

test_prompt = """<|user|>
Write a function to convert OHLCV data to a Gramian Angular Field image.

<|assistant|>
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

