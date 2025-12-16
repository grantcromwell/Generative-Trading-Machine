#!/usr/bin/env python3
"""
Data Preparation Script for Mac

Prepares training data locally on Mac, then upload to Colab for training.

This script:
1. Downloads and filters The Stack for quant code
2. Generates synthetic CoT + code pairs
3. Deduplicates with MinHash
4. Saves to JSONL for upload to Colab

Usage:
    python prepare_data_mac.py --output data/quant_train.jsonl --max_samples 50000
    
Then upload quant_train.jsonl to Colab and use with the training script.

Prerequisites:
    pip install datasets huggingface-hub datasketch
    huggingface-cli login
    Accept The Stack terms: https://huggingface.co/datasets/bigcode/the-stack
"""

import argparse
import json
import re
import random
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_dependencies():
    """Check required packages."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfFolder
        print("✓ Dependencies OK")
        
        if HfFolder.get_token():
            print("✓ HuggingFace logged in")
        else:
            print("⚠ Not logged in to HuggingFace")
            print("  Run: huggingface-cli login")
            return False
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Run: pip install datasets huggingface-hub datasketch")
        return False


# Quant keywords for filtering
QUANT_KEYWORDS = [
    "gaf", "gramian", "gasf", "gadf", "mtf",
    "talib", "ta-lib", "backtrader", "zipline", "vectorbt",
    "sharpe", "sortino", "drawdown", "volatility",
    "signal", "indicator", "macd", "rsi", "bollinger", "ema", "sma",
    "backtest", "strategy", "portfolio", "position_size",
    "yfinance", "ccxt", "ohlcv", "candlestick",
    "gymnasium", "gym", "stable_baselines", "reward", "agent",
    "pyts", "tsfresh", "prophet",
]

QUANT_PATTERNS = [re.compile(rf'\b{kw}\b', re.IGNORECASE) for kw in QUANT_KEYWORDS]


def is_quant_code(content: str, min_matches: int = 2) -> bool:
    """Check if code contains quant-related keywords."""
    matches = sum(1 for p in QUANT_PATTERNS if p.search(content))
    return matches >= min_matches


def format_sample(content: str, source: str = "code") -> dict:
    """Format content for training."""
    if source == "code":
        text = f"""<|user|>
Write Python code for the following quant trading task.

<|assistant|>
{content}<|end|>"""
    else:
        text = f"""<|user|>
Implement a trading signal/strategy with step-by-step reasoning.

<|assistant|>
{content}<|end|>"""
    
    return {"text": text, "source": source}


def load_the_stack(max_samples: int):
    """Load and filter The Stack dataset."""
    from datasets import load_dataset
    
    print("\n📚 Loading The Stack (Python)...")
    print("   Accept terms at: https://huggingface.co/datasets/bigcode/the-stack")
    
    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"✗ Failed to load The Stack: {e}")
        return []
    
    samples = []
    processed = 0
    
    print(f"   Filtering for quant code (target: {max_samples})...")
    
    for sample in ds:
        processed += 1
        content = sample.get('content', '')
        
        # Skip short/long files
        lines = content.split('\n')
        if len(lines) < 10 or len(lines) > 1000:
            continue
        
        # Check for quant keywords
        if is_quant_code(content):
            samples.append(format_sample(content, "code"))
            
        if len(samples) >= max_samples:
            break
            
        if processed % 50000 == 0:
            print(f"   Processed {processed:,}, found {len(samples):,} quant samples")
    
    print(f"   ✓ Collected {len(samples):,} quant code samples")
    return samples


def generate_synthetic(num_samples: int):
    """Generate synthetic CoT + code samples."""
    print(f"\n🔧 Generating {num_samples:,} synthetic samples...")
    
    GAF_TEMPLATES = [
        '''from pyts.image import GramianAngularField
import numpy as np

def create_gaf_image(prices: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Convert price series to Gramian Angular Field image."""
    gaf = GramianAngularField(image_size=image_size, method='summation')
    X = prices.reshape(1, -1)
    return gaf.fit_transform(X)[0]
''',
        '''import talib
import numpy as np

def calculate_rsi_signal(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Generate RSI-based trading signals."""
    rsi = talib.RSI(prices, timeperiod=period)
    signals = np.zeros_like(prices)
    signals[rsi > 70] = -1  # Overbought - sell
    signals[rsi < 30] = 1   # Oversold - buy
    return signals
''',
        '''import numpy as np

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
''',
        '''import backtrader as bt

class MomentumStrategy(bt.Strategy):
    """Simple momentum strategy using RSI."""
    params = (('period', 14), ('oversold', 30), ('overbought', 70))
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.period)
        
    def next(self):
        if not self.position:
            if self.rsi[0] < self.params.oversold:
                self.buy()
        elif self.rsi[0] > self.params.overbought:
            self.sell()
''',
    ]
    
    samples = []
    for _ in range(num_samples):
        code = random.choice(GAF_TEMPLATES)
        
        cot_text = f"""Let me implement this step by step:

1. **Import required libraries**: We'll need numpy and the relevant quant library.
2. **Define the function**: With clear type hints and docstring.
3. **Implement the logic**: Following best practices for numerical stability.

```python
{code}
```

This implementation follows quant coding best practices with proper documentation."""
        
        samples.append(format_sample(cot_text, "synthetic"))
    
    print(f"   ✓ Generated {len(samples):,} synthetic samples")
    return samples


def deduplicate(samples: list, threshold: float = 0.7):
    """Remove near-duplicates using MinHash."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("   ⚠ datasketch not installed, skipping deduplication")
        return samples
    
    print(f"\n🔍 Deduplicating {len(samples):,} samples...")
    
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_samples = []
    
    for i, sample in enumerate(samples):
        # Create MinHash for sample
        m = MinHash(num_perm=128)
        for word in sample['text'].lower().split():
            m.update(word.encode('utf-8'))
        
        # Check for duplicates
        result = lsh.query(m)
        if not result:
            lsh.insert(f"s{i}", m)
            unique_samples.append(sample)
            
        if (i + 1) % 10000 == 0:
            print(f"   Processed {i+1:,}, unique: {len(unique_samples):,}")
    
    removed = len(samples) - len(unique_samples)
    print(f"   ✓ Removed {removed:,} duplicates, kept {len(unique_samples):,}")
    return unique_samples


def save_jsonl(samples: list, output_path: str):
    """Save samples to JSONL file."""
    print(f"\n💾 Saving to {output_path}...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"   ✓ Saved {len(samples):,} samples ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare quant training data on Mac")
    parser.add_argument("--output", "-o", default="data/quant_train.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--max_samples", "-n", type=int, default=50000,
                        help="Maximum samples to collect")
    parser.add_argument("--synthetic_ratio", type=float, default=0.1,
                        help="Ratio of synthetic samples (default: 0.1)")
    parser.add_argument("--skip_dedup", action="store_true",
                        help="Skip deduplication")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 Brainnet/Minah - Data Preparation for Quant Training")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Calculate sample targets
    code_target = int(args.max_samples * (1 - args.synthetic_ratio))
    synthetic_target = int(args.max_samples * args.synthetic_ratio)
    
    print(f"\nTargets: {code_target:,} code + {synthetic_target:,} synthetic")
    
    # Load The Stack
    code_samples = load_the_stack(code_target)
    
    # Generate synthetic
    synthetic_samples = generate_synthetic(synthetic_target)
    
    # Combine
    all_samples = code_samples + synthetic_samples
    random.shuffle(all_samples)
    
    # Deduplicate
    if not args.skip_dedup:
        all_samples = deduplicate(all_samples)
    
    # Save
    save_jsonl(all_samples, args.output)
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print(f"   Output: {args.output}")
    print("\nNext steps:")
    print("   1. Upload to Google Colab")
    print("   2. Run training with notebooks/train_phi35_colab.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

