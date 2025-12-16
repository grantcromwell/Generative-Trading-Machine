"""
Quant Code Filtering Module.

Filters code samples for quant-relevant content:
- GAF/Gramian Angular Field implementations
- TA-Lib and technical analysis libraries
- Backtrader, Zipline strategy code
- RL for trading (stable-baselines, gym envs)
- Quantitative finance patterns

Based on keyword matching and code structure analysis.
"""

import re
import logging
from typing import Set, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

from .config import TrainingConfig, DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics from filtering run."""
    total_processed: int = 0
    passed_filter: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_no_keywords: int = 0
    rejected_low_quality: int = 0
    
    @property
    def pass_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.passed_filter / self.total_processed


class QuantCodeFilter:
    """
    Filter for quant-relevant code samples.
    
    Implements multi-stage filtering:
    1. Length filtering (min/max lines)
    2. Keyword matching (quant libraries, patterns)
    3. Code quality heuristics
    4. Language-specific checks
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.dataset_config = self.config.dataset
        
        # Compile keyword patterns for efficiency
        self._keyword_patterns = self._compile_patterns()
        
        # Stats tracking
        self.stats = FilterStats()
        
        # High-value patterns (boost priority)
        self._high_value_patterns = self._compile_high_value_patterns()
        
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile keyword regex patterns."""
        patterns = []
        for kw in self.dataset_config.quant_keywords:
            # Case-insensitive, word boundary matching
            try:
                pattern = re.compile(
                    rf'\b{re.escape(kw)}\b',
                    re.IGNORECASE
                )
                patterns.append(pattern)
            except re.error:
                logger.warning(f"Invalid regex pattern for keyword: {kw}")
        return patterns
    
    def _compile_high_value_patterns(self) -> List[re.Pattern]:
        """Compile patterns for high-value quant code."""
        high_value = [
            # GAF-specific patterns
            r'gramian[_\s]?angular',
            r'GramianAngularField',
            r'GASF|GADF|MTF',
            r'pyts\.image',
            
            # Strategy class patterns
            r'class\s+\w*Strategy\w*',
            r'class\s+\w*Backtest\w*',
            r'def\s+next\s*\(\s*self',  # Backtrader pattern
            
            # Trading function patterns
            r'def\s+(calculate_|compute_|get_)(signal|indicator|position)',
            r'def\s+on_(data|bar|tick|trade)',
            
            # RL trading patterns
            r'class\s+\w*TradingEnv\w*',
            r'env\.step\s*\(',
            r'self\.observation_space',
            
            # Quantitative patterns
            r'sharpe[_\s]?ratio',
            r'sortino[_\s]?ratio',
            r'max[_\s]?drawdown',
            r'position[_\s]?size',
        ]
        
        return [re.compile(p, re.IGNORECASE) for p in high_value]
    
    def should_include(self, content: str) -> bool:
        """
        Determine if content should be included in training set.
        
        Args:
            content: Code content to evaluate
            
        Returns:
            True if content passes all filters
        """
        self.stats.total_processed += 1
        
        # Stage 1: Length filtering
        lines = content.split('\n')
        line_count = len(lines)
        
        if line_count < self.dataset_config.min_line_count:
            self.stats.rejected_too_short += 1
            return False
            
        if line_count > self.dataset_config.max_line_count:
            self.stats.rejected_too_long += 1
            return False
        
        # Stage 2: Keyword matching
        keyword_score = self._calculate_keyword_score(content)
        
        if keyword_score < self.dataset_config.min_quant_keyword_matches:
            self.stats.rejected_no_keywords += 1
            return False
        
        # Stage 3: Code quality heuristics
        if not self._passes_quality_check(content):
            self.stats.rejected_low_quality += 1
            return False
        
        self.stats.passed_filter += 1
        return True
    
    def _calculate_keyword_score(self, content: str) -> int:
        """
        Calculate keyword match score.
        
        High-value patterns count double.
        """
        score = 0
        
        # Standard keyword matches
        for pattern in self._keyword_patterns:
            if pattern.search(content):
                score += 1
                
        # High-value pattern matches (bonus points)
        for pattern in self._high_value_patterns:
            if pattern.search(content):
                score += 2
                
        return score
    
    def _passes_quality_check(self, content: str) -> bool:
        """
        Basic code quality heuristics.
        
        Rejects:
        - Auto-generated boilerplate
        - Mostly comments/empty lines
        - Obfuscated/minified code
        - Non-code content
        """
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if not non_empty_lines:
            return False
            
        # Check code-to-comment ratio
        code_lines = [l for l in non_empty_lines if not l.strip().startswith('#')]
        if len(code_lines) < len(non_empty_lines) * 0.3:
            return False  # Too many comments
            
        # Check for auto-generated markers
        auto_gen_markers = [
            'auto-generated', 'autogenerated', 'do not edit',
            'generated by', 'this file was generated',
        ]
        first_lines = '\n'.join(lines[:10]).lower()
        if any(marker in first_lines for marker in auto_gen_markers):
            return False
            
        # Check average line length (detect minified code)
        avg_line_len = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        if avg_line_len > 200:  # Likely minified
            return False
            
        return True
    
    def get_priority_score(self, content: str) -> float:
        """
        Calculate priority score for ordering samples.
        
        Higher scores = more valuable for training.
        
        Returns:
            Float score from 0.0 to 1.0
        """
        score = 0.0
        max_score = 10.0
        
        # Keyword density
        keyword_matches = self._calculate_keyword_score(content)
        score += min(keyword_matches, 5)  # Cap at 5 points
        
        # High-value patterns bonus
        for pattern in self._high_value_patterns:
            if pattern.search(content):
                score += 0.5
                
        # Code structure indicators
        if 'class ' in content:
            score += 0.5
        if 'def ' in content:
            score += 0.5
        if 'import ' in content:
            score += 0.25
            
        # Docstring presence
        if '"""' in content or "'''" in content:
            score += 0.5
            
        # Type hints (modern Python)
        if ' -> ' in content or ': ' in content:
            score += 0.25
            
        return min(score / max_score, 1.0)
    
    def filter_batch(
        self,
        samples: List[str],
        return_scores: bool = False
    ) -> List[Tuple[str, float]] | List[str]:
        """
        Filter a batch of samples.
        
        Args:
            samples: List of code content strings
            return_scores: If True, return (content, score) tuples
            
        Returns:
            Filtered list of samples (with optional scores)
        """
        results = []
        
        for content in samples:
            if self.should_include(content):
                if return_scores:
                    score = self.get_priority_score(content)
                    results.append((content, score))
                else:
                    results.append(content)
                    
        if return_scores:
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            
        return results


class KeywordExtractor:
    """
    Extract and analyze quant-related keywords from code.
    
    Useful for:
    - Dataset analysis
    - Keyword frequency statistics
    - Identifying new relevant keywords
    """
    
    # Common quant library imports
    IMPORT_PATTERNS = {
        'talib': r'import\s+talib|from\s+talib',
        'backtrader': r'import\s+backtrader|from\s+backtrader',
        'zipline': r'import\s+zipline|from\s+zipline',
        'pandas_ta': r'import\s+pandas_ta|from\s+pandas_ta',
        'vectorbt': r'import\s+vectorbt|from\s+vectorbt',
        'pyts': r'import\s+pyts|from\s+pyts',
        'yfinance': r'import\s+yfinance|from\s+yfinance',
        'ccxt': r'import\s+ccxt|from\s+ccxt',
        'stable_baselines': r'import\s+stable_baselines|from\s+stable_baselines',
        'gymnasium': r'import\s+gymnasium|from\s+gymnasium',
        'gym': r'import\s+gym|from\s+gym',
    }
    
    def __init__(self):
        self._patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.IMPORT_PATTERNS.items()
        }
        
    def extract_imports(self, content: str) -> Set[str]:
        """Extract recognized quant library imports."""
        found = set()
        for name, pattern in self._patterns.items():
            if pattern.search(content):
                found.add(name)
        return found
    
    def analyze_content(self, content: str) -> dict:
        """
        Analyze content for quant indicators.
        
        Returns:
            Dict with analysis results
        """
        return {
            'imports': list(self.extract_imports(content)),
            'has_class': 'class ' in content,
            'has_function': 'def ' in content,
            'line_count': len(content.split('\n')),
            'has_docstring': '"""' in content or "'''" in content,
        }


def demo_filter():
    """Demonstrate filter functionality."""
    
    # Sample quant code
    good_sample = '''
import talib
import numpy as np
from pyts.image import GramianAngularField

class MomentumStrategy:
    """
    GAF-enhanced momentum strategy.
    Uses Gramian Angular Field for pattern recognition.
    """
    
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.gaf = GramianAngularField()
        
    def calculate_signal(self, prices):
        """Calculate trading signal from price data."""
        # Compute RSI
        rsi = talib.RSI(prices, timeperiod=14)
        
        # Generate GAF image
        gaf_image = self.gaf.fit_transform(prices.reshape(1, -1))
        
        # Compute Sharpe ratio
        returns = np.diff(prices) / prices[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return {
            'rsi': rsi[-1],
            'gaf': gaf_image,
            'sharpe': sharpe_ratio,
        }
'''

    bad_sample = '''
# Auto-generated file, do not edit
print("Hello World")
x = 1 + 2
'''

    filter_obj = QuantCodeFilter()
    
    print("Testing QuantCodeFilter...")
    print(f"\nGood sample passes: {filter_obj.should_include(good_sample)}")
    print(f"Good sample score: {filter_obj.get_priority_score(good_sample):.2f}")
    
    print(f"\nBad sample passes: {filter_obj.should_include(bad_sample)}")
    
    print(f"\nFilter stats: {filter_obj.stats}")


if __name__ == "__main__":
    demo_filter()


