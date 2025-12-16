"""
Synthetic Data Generator for Quant Training.

Generates tailored pairs blending:
- AlphaFin Chain-of-Thought reasoning
- bigcode-style Python strategies
- GAF encoder implementations

Target: 50k synthetic pairs for 10% of training data.
Pattern: "AlphaFin CoT on GAF → bigcode-style Python strat"

This amplifies training data for calibration and achieves
~65% accuracy boost on internal evaluations.
"""

import random
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass
import json

from .config import SyntheticConfig, TrainingConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Code Templates - GAF, TA-Lib, Backtrader patterns
# =============================================================================

GAF_TEMPLATES = [
    # Basic GAF implementation
    '''from pyts.image import GramianAngularField
import numpy as np

def create_gaf_image(prices: np.ndarray, image_size: int = 64) -> np.ndarray:
    """
    Convert price series to Gramian Angular Field image.
    
    GAF transforms time series into polar coordinates, then computes
    angular difference matrix for pattern recognition.
    """
    gaf = GramianAngularField(image_size=image_size, method='{method}')
    # Reshape for pyts: (n_samples, n_timestamps)
    X = prices.reshape(1, -1)
    return gaf.fit_transform(X)[0]
''',
    
    # GAF with multiple fields
    '''from pyts.image import GramianAngularField, MarkovTransitionField
import numpy as np

class MultiFieldEncoder:
    """Encode time series using multiple field transformations."""
    
    def __init__(self, image_size: int = 64):
        self.gasf = GramianAngularField(image_size=image_size, method='summation')
        self.gadf = GramianAngularField(image_size=image_size, method='difference')
        self.mtf = MarkovTransitionField(image_size=image_size, n_bins={n_bins})
        
    def encode(self, series: np.ndarray) -> np.ndarray:
        """Create stacked multi-channel image."""
        X = series.reshape(1, -1)
        gasf_img = self.gasf.fit_transform(X)[0]
        gadf_img = self.gadf.fit_transform(X)[0]
        mtf_img = self.mtf.fit_transform(X)[0]
        return np.stack([gasf_img, gadf_img, mtf_img], axis=0)
''',
    
    # GAF for ConvNet
    '''import torch
import numpy as np
from pyts.image import GramianAngularField

def prepare_gaf_for_convnet(
    ohlcv: np.ndarray,
    lookback: int = {lookback},
    image_size: int = 64
) -> torch.Tensor:
    """
    Prepare GAF tensor for ConvNeXt or CNN model.
    
    Creates multi-channel GAF from OHLCV data:
    - Channel 0: Close price GAF
    - Channel 1: Volume GAF  
    - Channel 2: Returns GAF
    """
    gaf = GramianAngularField(image_size=image_size)
    
    close = ohlcv[-lookback:, 3]  # Close prices
    volume = ohlcv[-lookback:, 4]  # Volume
    returns = np.diff(close) / close[:-1]
    returns = np.pad(returns, (1, 0), mode='edge')
    
    channels = []
    for series in [close, volume, returns]:
        X = series.reshape(1, -1)
        img = gaf.fit_transform(X)[0]
        channels.append(img)
        
    # Stack and convert to tensor
    tensor = torch.tensor(np.stack(channels), dtype=torch.float32)
    return tensor.unsqueeze(0)  # Add batch dimension
''',
]


STRATEGY_TEMPLATES = [
    # Momentum strategy
    '''import numpy as np
import talib

class {strategy_name}Strategy:
    """
    {strategy_type} strategy using {indicator} indicator.
    
    Entry: {entry_condition}
    Exit: {exit_condition}
    """
    
    def __init__(self, period: int = {period}):
        self.period = period
        self.position = 0
        
    def calculate_signal(self, prices: np.ndarray) -> int:
        """
        Generate trading signal.
        
        Returns:
            1 for long, -1 for short, 0 for flat
        """
        indicator = talib.{talib_func}(prices, timeperiod=self.period)
        
        if indicator[-1] > {threshold_high}:
            return 1  # Long signal
        elif indicator[-1] < {threshold_low}:
            return -1  # Short signal
        return 0  # No signal
        
    def compute_position_size(
        self, 
        capital: float,
        volatility: float,
        risk_pct: float = 0.02
    ) -> float:
        """Kelly-inspired position sizing with volatility scaling."""
        risk_amount = capital * risk_pct
        return risk_amount / (volatility * {vol_multiplier})
''',
    
    # Backtrader strategy
    '''import backtrader as bt

class {strategy_name}(bt.Strategy):
    """
    Backtrader {strategy_type} strategy.
    
    Uses {description}.
    """
    
    params = (
        ('period', {period}),
        ('threshold', {threshold}),
    )
    
    def __init__(self):
        self.indicator = bt.indicators.{bt_indicator}(
            self.data.close,
            period=self.params.period
        )
        self.order = None
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.indicator[0] > self.params.threshold:
                self.order = self.buy()
        else:
            if self.indicator[0] < -self.params.threshold:
                self.order = self.sell()
                
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
''',

    # RL trading environment
    '''import gymnasium as gym
import numpy as np
from gymnasium import spaces

class {env_name}(gym.Env):
    """
    Trading environment for {strategy_type} strategy.
    
    Observation: {obs_description}
    Action: {action_description}
    Reward: {reward_description}
    """
    
    def __init__(self, prices: np.ndarray, lookback: int = {lookback}):
        super().__init__()
        
        self.prices = prices
        self.lookback = lookback
        self.current_step = lookback
        
        # Observation space: normalized price features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=({obs_dim},), dtype=np.float32
        )
        
        # Action space: hold, buy, sell
        self.action_space = spaces.Discrete(3)
        
        self.position = 0
        self.entry_price = 0
        
    def _get_observation(self) -> np.ndarray:
        """Extract features from price history."""
        window = self.prices[self.current_step - self.lookback:self.current_step]
        returns = np.diff(window) / window[:-1]
        
        return np.array([
            returns.mean(),
            returns.std(),
            (window[-1] - window.mean()) / window.std(),
            self.position,
        ], dtype=np.float32)
        
    def step(self, action: int):
        current_price = self.prices[self.current_step]
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            reward = (current_price - self.entry_price) / self.entry_price
            self.position = 0
            
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        return self._get_observation(), reward, done, False, {{}}
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.position = 0
        return self._get_observation(), {{}}
''',
]


COT_TEMPLATES = [
    # GAF reasoning
    """### Instruction:
Implement a {strategy_type} trading signal using Gramian Angular Field (GAF) encoding.

### Chain of Thought:
Let me think through this step by step:

1. **Understanding GAF**: Gramian Angular Fields convert time series to images by:
   - Normalizing data to [-1, 1] range
   - Computing polar coordinates (angle φ = arccos(x))
   - Creating matrix of angular {gaf_operation}

2. **Why GAF for {strategy_type}**: 
   - Captures temporal patterns visually
   - CNN/ConvNet can extract features efficiently
   - Preserves temporal dependencies better than flat features

3. **Implementation approach**:
   - Use pyts library for GAF transformation
   - Create {num_channels}-channel image (GASF, GADF, MTF)
   - Feed to neural network for signal prediction

4. **Key considerations**:
   - Lookback window: {lookback} periods
   - Image size: {image_size}x{image_size} pixels
   - Normalize inputs for stable training

### Implementation:
{code}
""",

    # Strategy development reasoning
    """### Instruction:
Build a {strategy_type} strategy with proper risk management.

### Chain of Thought:
Let's break this down systematically:

1. **Strategy Logic**:
   - Entry signal: {entry_logic}
   - Exit signal: {exit_logic}
   - Time horizon: {time_horizon}

2. **Risk Management**:
   - Position sizing: Kelly criterion with {kelly_fraction} fraction
   - Max drawdown limit: {max_dd}%
   - Stop loss: {stop_loss}% from entry

3. **Implementation Steps**:
   a) Calculate technical indicators
   b) Generate entry/exit signals
   c) Apply position sizing
   d) Track performance metrics

4. **Backtesting Considerations**:
   - Account for slippage: {slippage} bps
   - Transaction costs: {tx_cost} bps
   - Avoid lookahead bias

### Implementation:
{code}
""",

    # RL trading reasoning
    """### Instruction:
Create a reinforcement learning trading agent for {market_type} market.

### Chain of Thought:
Designing an RL trading system requires careful consideration:

1. **Environment Design**:
   - State space: {state_description}
   - Action space: discrete (hold/buy/sell) or continuous sizing
   - Reward: {reward_type} (Sharpe, PnL, risk-adjusted)

2. **Algorithm Selection**:
   - PPO for stable training
   - Use actor-critic architecture
   - {algo_consideration}

3. **Feature Engineering**:
   - Technical indicators: {indicators}
   - Normalized returns
   - Position state
   - Market regime indicators

4. **Training Strategy**:
   - Episode length: {episode_length} steps
   - Multiple random seeds
   - Walk-forward validation

### Implementation:
{code}
""",
]


# =============================================================================
# Generator Class
# =============================================================================

@dataclass
class SyntheticSample:
    """A generated synthetic training sample."""
    instruction: str
    response: str
    cot_reasoning: str
    source: str = "synthetic"
    template_type: str = ""
    metadata: Dict[str, Any] = None
    
    def to_training_format(self) -> str:
        """Convert to training text format."""
        return f"{self.instruction}\n\n{self.cot_reasoning}\n\n{self.response}"


class SyntheticPairGenerator:
    """
    Generate synthetic training pairs for quant code.
    
    Creates diverse combinations of:
    - GAF implementations with varying parameters
    - Trading strategies with different indicators
    - RL environments for various market types
    - Chain-of-thought reasoning annotations
    
    Target: 50k high-quality pairs for 10% training mix.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.synthetic_config = self.config.synthetic
        
        # Random seed for reproducibility
        random.seed(self.config.seed)
        
        # Parameter pools for generation
        self._init_parameter_pools()
        
    def _init_parameter_pools(self):
        """Initialize parameter pools for template filling."""
        
        self.gaf_params = {
            'method': ['summation', 'difference'],
            'n_bins': [4, 5, 8, 10],
            'lookback': [20, 30, 50, 100],
            'image_size': [32, 64, 128],
        }
        
        self.strategy_params = {
            'strategy_name': [
                'Momentum', 'MeanReversion', 'Breakout', 'Volatility',
                'TrendFollowing', 'StatArb', 'DualMomentum', 'AdaptiveMA'
            ],
            'strategy_type': [
                'momentum', 'mean reversion', 'breakout', 'trend following',
                'statistical arbitrage', 'volatility', 'pairs trading'
            ],
            'period': [5, 10, 14, 20, 30, 50],
            'threshold': [0.5, 0.7, 1.0, 1.5, 2.0],
            'threshold_high': [70, 75, 80],
            'threshold_low': [20, 25, 30],
            'talib_func': ['RSI', 'MOM', 'ROC', 'WILLR', 'CCI'],
            'bt_indicator': [
                'RSI', 'Momentum', 'RateOfChange100', 'StochasticFast',
                'MACD', 'BollingerBands', 'ATR'
            ],
            'vol_multiplier': [1.5, 2.0, 2.5, 3.0],
            'indicator': ['RSI', 'MACD', 'Bollinger Bands', 'ATR', 'Momentum'],
            'entry_condition': [
                'RSI crosses above 30', 'price breaks above resistance',
                'MACD crosses above signal', 'volatility expands'
            ],
            'exit_condition': [
                'RSI crosses below 70', 'trailing stop hit',
                'profit target reached', 'time-based exit'
            ],
            'description': [
                'RSI oversold/overbought signals',
                'Bollinger Band mean reversion',
                'MACD crossover momentum',
                'ATR-based volatility breakout'
            ],
        }
        
        self.rl_params = {
            'env_name': [
                'TradingEnv', 'StockTradingEnv', 'CryptoTradingEnv',
                'FuturesTradingEnv', 'ForexTradingEnv'
            ],
            'lookback': [20, 30, 50],
            'obs_dim': [4, 8, 16],
            'obs_description': [
                'normalized returns and position',
                'technical indicators and market state',
                'multi-timeframe features'
            ],
            'action_description': [
                'discrete: hold/buy/sell',
                'continuous position sizing',
                'multi-asset allocation'
            ],
            'reward_description': [
                'risk-adjusted returns (Sharpe)',
                'PnL with drawdown penalty',
                'differential Sharpe ratio'
            ],
        }
        
        self.cot_params = {
            'gaf_operation': ['summation (GASF)', 'difference (GADF)'],
            'num_channels': [1, 3, 4],
            'image_size': [32, 64, 128],
            'entry_logic': [
                'RSI < 30 with volume spike',
                'price breaks 20-day high with momentum',
                'MACD bullish crossover confirmed by volume'
            ],
            'exit_logic': [
                'RSI > 70 or trailing stop',
                'profit target 2x risk',
                'time-based with momentum fade'
            ],
            'time_horizon': ['intraday', 'swing (1-5 days)', 'position (weeks)'],
            'kelly_fraction': ['0.25', '0.5', 'half-Kelly'],
            'max_dd': [10, 15, 20],
            'stop_loss': [1, 2, 3],
            'slippage': [5, 10, 20],
            'tx_cost': [5, 10, 20],
            'market_type': ['equity', 'crypto', 'futures', 'forex'],
            'state_description': [
                'OHLCV + technical indicators',
                'GAF image + position state',
                'multi-timeframe features'
            ],
            'reward_type': ['Sharpe ratio', 'risk-adjusted PnL', 'differential Sharpe'],
            'algo_consideration': [
                'Add entropy bonus for exploration',
                'Use reward shaping for sparse rewards',
                'Normalize observations for stability'
            ],
            'indicators': [
                'RSI, MACD, Bollinger %B',
                'ATR, momentum, volume profile',
                'moving averages, volatility regime'
            ],
            'episode_length': [252, 500, 1000],
        }
    
    def _fill_template(self, template: str, params: Dict[str, List]) -> str:
        """Fill template with random parameters."""
        result = template
        
        # Find all placeholders
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        for ph in set(placeholders):
            if ph in params:
                value = random.choice(params[ph])
                result = result.replace('{' + ph + '}', str(value))
                
        return result
    
    def generate_gaf_sample(self) -> SyntheticSample:
        """Generate a GAF-related code sample."""
        template = random.choice(GAF_TEMPLATES)
        code = self._fill_template(template, self.gaf_params)
        
        # Create instruction
        instructions = [
            "Implement a GAF encoder for time series data.",
            "Create a Gramian Angular Field transformation for price data.",
            "Build a multi-channel GAF encoder for ConvNeXt input.",
            "Write code to convert OHLCV data to GAF images.",
        ]
        
        return SyntheticSample(
            instruction=random.choice(instructions),
            response=code,
            cot_reasoning="Using pyts library for efficient GAF computation.",
            template_type="gaf",
            metadata={'params': self.gaf_params}
        )
    
    def generate_strategy_sample(self) -> SyntheticSample:
        """Generate a trading strategy code sample."""
        template = random.choice(STRATEGY_TEMPLATES)
        code = self._fill_template(template, self.strategy_params)
        
        instructions = [
            "Implement a momentum trading strategy with risk management.",
            "Create a Backtrader strategy for trend following.",
            "Build a mean reversion strategy using RSI.",
            "Write a trading strategy with proper position sizing.",
        ]
        
        return SyntheticSample(
            instruction=random.choice(instructions),
            response=code,
            cot_reasoning="Implementing with proper risk management and backtesting support.",
            template_type="strategy",
            metadata={'params': self.strategy_params}
        )
    
    def generate_rl_sample(self) -> SyntheticSample:
        """Generate an RL trading environment sample."""
        template = random.choice([t for t in STRATEGY_TEMPLATES if 'gymnasium' in t.lower() or 'gym.Env' in t])
        if not template:
            template = STRATEGY_TEMPLATES[-1]  # RL template is last
            
        code = self._fill_template(template, self.rl_params | self.strategy_params)
        
        instructions = [
            "Create a Gymnasium trading environment for RL agents.",
            "Implement a trading environment with proper observation space.",
            "Build an RL environment for algorithmic trading.",
        ]
        
        return SyntheticSample(
            instruction=random.choice(instructions),
            response=code,
            cot_reasoning="Designing environment with stable-baselines3 compatibility.",
            template_type="rl_env",
            metadata={'params': self.rl_params}
        )
    
    def generate_cot_sample(self) -> SyntheticSample:
        """Generate a full CoT + code sample."""
        # Combine all params
        all_params = {
            **self.gaf_params,
            **self.strategy_params,
            **self.rl_params,
            **self.cot_params,
        }
        
        # Pick a CoT template
        cot_template = random.choice(COT_TEMPLATES)
        
        # Pick matching code template
        if 'GAF' in cot_template or 'Gramian' in cot_template:
            code_template = random.choice(GAF_TEMPLATES)
        elif 'RL' in cot_template or 'reinforcement' in cot_template.lower():
            code_template = STRATEGY_TEMPLATES[-1]
        else:
            code_template = random.choice(STRATEGY_TEMPLATES[:2])
            
        # Generate code first
        code = self._fill_template(code_template, all_params)
        
        # Fill CoT template with code
        all_params['code'] = code
        full_sample = self._fill_template(cot_template, all_params)
        
        # Parse out instruction and response
        parts = full_sample.split('### Implementation:')
        if len(parts) == 2:
            instruction_and_cot = parts[0]
            response = parts[1].strip()
        else:
            instruction_and_cot = full_sample
            response = code
            
        return SyntheticSample(
            instruction=instruction_and_cot.split('### Chain of Thought:')[0].replace('### Instruction:', '').strip(),
            response=response,
            cot_reasoning=instruction_and_cot,
            template_type="cot_full",
            metadata={'params': self.cot_params}
        )
    
    def generate(self, num_samples: Optional[int] = None) -> Generator[SyntheticSample, None, None]:
        """
        Generate synthetic training samples.
        
        Args:
            num_samples: Number of samples to generate (default from config)
            
        Yields:
            SyntheticSample objects
        """
        num_samples = num_samples or self.synthetic_config.num_pairs
        
        # Distribution: 30% GAF, 30% strategies, 15% RL, 25% full CoT
        generators = [
            (0.30, self.generate_gaf_sample),
            (0.30, self.generate_strategy_sample),
            (0.15, self.generate_rl_sample),
            (0.25, self.generate_cot_sample),
        ]
        
        for i in range(num_samples):
            # Pick generator based on distribution
            r = random.random()
            cumulative = 0
            for prob, gen_func in generators:
                cumulative += prob
                if r < cumulative:
                    yield gen_func()
                    break
                    
            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} synthetic samples")
    
    def generate_to_file(
        self,
        output_path: str,
        num_samples: Optional[int] = None,
        format: str = 'jsonl'
    ) -> int:
        """
        Generate samples and write to file.
        
        Args:
            output_path: Output file path
            num_samples: Number of samples
            format: Output format ('jsonl' or 'json')
            
        Returns:
            Number of samples written
        """
        count = 0
        
        with open(output_path, 'w') as f:
            for sample in self.generate(num_samples):
                record = {
                    'instruction': sample.instruction,
                    'response': sample.response,
                    'text': sample.to_training_format(),
                    'source': sample.source,
                    'type': sample.template_type,
                }
                
                if format == 'jsonl':
                    f.write(json.dumps(record) + '\n')
                count += 1
                
        logger.info(f"Wrote {count} samples to {output_path}")
        return count


def demo_synthetic():
    """Demonstrate synthetic generation."""
    print("Synthetic Data Generation Demo")
    print("=" * 60)
    
    generator = SyntheticPairGenerator()
    
    print("\n1. GAF Sample:")
    sample = generator.generate_gaf_sample()
    print(f"Instruction: {sample.instruction}")
    print(f"Code preview: {sample.response[:200]}...")
    
    print("\n2. Strategy Sample:")
    sample = generator.generate_strategy_sample()
    print(f"Instruction: {sample.instruction}")
    print(f"Code preview: {sample.response[:200]}...")
    
    print("\n3. Full CoT Sample:")
    sample = generator.generate_cot_sample()
    print(f"Full text preview:\n{sample.to_training_format()[:500]}...")
    
    print("\n4. Generation stats:")
    samples = list(generator.generate(100))
    types = {}
    for s in samples:
        types[s.template_type] = types.get(s.template_type, 0) + 1
    print(f"Distribution: {types}")


if __name__ == "__main__":
    demo_synthetic()


