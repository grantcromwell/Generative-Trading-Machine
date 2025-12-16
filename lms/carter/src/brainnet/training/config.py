"""
Training Configuration for Minah Phi-3.5 Fine-tuning.

Optimized for quant code generation with GAF encoders,
backtesting strategies, and signal pipelines.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    # The Stack v1.2 (bigcode)
    the_stack_name: str = "bigcode/the-stack"
    the_stack_languages: List[str] = field(default_factory=lambda: ["python", "typescript"])
    the_stack_split: str = "train"
    the_stack_streaming: bool = True  # Essential for 3TB dataset
    
    # AlphaFin CoT dataset
    alphafin_name: str = "AlphaFin-proj/alphafin-cot"  # Note: verify exact HF path
    alphafin_split: str = "train"
    
    # Evol-Instruct for amplification
    evol_instruct_name: str = "SurgeGlobal/Evol-Instruct"
    evol_instruct_split: str = "train"
    
    # Mix ratios
    bigcode_ratio: float = 0.50  # Code syntax
    alphafin_ratio: float = 0.40  # Finance reasoning
    synthetic_ratio: float = 0.10  # Generated CoT+code pairs
    
    # Filtering
    quant_keywords: List[str] = field(default_factory=lambda: [
        # GAF and imaging
        "gaf", "gramian", "gramian_angular", "gasf", "gadf", "mtf", "recurrence",
        # Technical analysis libs
        "ta-lib", "talib", "ta.lib", "TA_", "backtrader", "bt.", "zipline",
        "vectorbt", "pandas_ta", "finta", "pyti", "tulipy",
        # RL and ML for trading
        "stable_baselines", "sb3", "gym", "gymnasium", "env.step", "reward",
        "policy", "agent", "q_learning", "dqn", "ppo", "a2c",
        # Quantitative finance
        "sharpe", "sortino", "max_drawdown", "var", "cvar", "volatility",
        "alpha", "beta", "kelly", "position_size", "risk_parity",
        # Data and signals
        "ohlcv", "candlestick", "ticker", "yfinance", "ccxt", "binance",
        "signal", "indicator", "macd", "rsi", "bollinger", "ema", "sma",
        # Strategy patterns
        "backtest", "strategy", "portfolio", "execution", "slippage",
        "order", "fill", "pnl", "equity_curve", "trade_log",
        # Time series
        "pyts", "tsfresh", "tslearn", "prophet", "arima", "garch",
    ])
    
    # Quality thresholds
    min_line_count: int = 10
    max_line_count: int = 2000
    min_quant_keyword_matches: int = 2
    

@dataclass
class DeduplicationConfig:
    """MinHash deduplication settings."""
    
    num_perm: int = 128  # Number of permutations for MinHash
    threshold: float = 0.7  # Jaccard similarity threshold
    ngram_size: int = 5  # Character n-gram size
    seed: int = 42
    

@dataclass
class PackingConfig:
    """Sequence packing configuration for Phi-3.5."""
    
    # Phi-3.5-Mini context lengths
    min_seq_length: int = 4096
    max_seq_length: int = 8192  # Can go up to 16K but 8K is efficient
    target_seq_length: int = 8192
    
    # Special tokens
    pad_token: str = "<|pad|>"
    eos_token: str = "<|endoftext|>"
    bos_token: str = "<|startoftext|>"
    sep_token: str = "\n\n---\n\n"
    
    # Packing strategy
    pack_multiple_samples: bool = True
    truncate_long_samples: bool = True
    

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    
    num_pairs: int = 50_000  # Target 50k synthetic pairs
    
    # Templates for CoT + code generation
    cot_templates: List[str] = field(default_factory=lambda: [
        "Let's think step by step about implementing a {strategy_type} strategy:\n"
        "1. First, we need to {step1}\n"
        "2. Then, {step2}\n"
        "3. Finally, {step3}\n\n"
        "Here's the implementation:\n```python\n{code}\n```",
        
        "To build a GAF-based {signal_type} signal:\n"
        "- Convert OHLCV to Gramian Angular Field\n"
        "- Apply {transform} transformation\n"
        "- Extract features using {method}\n\n"
        "Code:\n```python\n{code}\n```",
        
        "Chain-of-thought for {task_type}:\n"
        "Reasoning: {reasoning}\n"
        "Approach: {approach}\n"
        "Implementation:\n```python\n{code}\n```",
    ])
    
    # Strategy types for generation
    strategy_types: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "breakout", "pairs_trading",
        "stat_arb", "market_making", "trend_following", "volatility",
    ])
    
    # Use local model for generation or templates
    use_llm_generation: bool = False  # Set True if LLM available
    

@dataclass
class TrainingConfig:
    """Main training configuration for Unsloth Phi-3.5 fine-tuning."""
    
    # Model
    base_model: str = "microsoft/Phi-3.5-mini-instruct"
    load_in_4bit: bool = True
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Training hyperparameters
    max_seq_length: int = 8192
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 16  # batch_size * grad_accum
    
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    num_epochs: int = 2  # 1-2 epochs recommended
    max_steps: int = -1  # -1 for full epochs
    
    # Optimization
    optimizer: str = "adamw_8bit"
    lr_scheduler: str = "cosine"
    fp16: bool = False
    bf16: bool = True  # Better for Phi-3.5
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    output_dir: str = "outputs/phi35-quant"
    hub_model_id: Optional[str] = None
    push_to_hub: bool = False
    
    # Dataset configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dedup: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    
    # Paths
    cache_dir: Path = field(default_factory=lambda: Path("~/.cache/brainnet").expanduser())
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_colab_config() -> TrainingConfig:
    """Optimized config for Google Colab Pro (T4/A100)."""
    config = TrainingConfig()
    config.batch_size = 2
    config.gradient_accumulation_steps = 8
    config.max_seq_length = 4096  # Reduced for Colab memory
    return config


def get_4090_config() -> TrainingConfig:
    """Optimized config for RTX 4090 (24GB VRAM)."""
    config = TrainingConfig()
    config.batch_size = 4
    config.gradient_accumulation_steps = 4
    config.max_seq_length = 8192
    return config


