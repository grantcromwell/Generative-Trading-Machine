"""
Coding Agent - Strategy code generation / hot-reloading
"""

import os
import sys
import importlib
import importlib.util
import tempfile
from typing import Optional, Callable
from pathlib import Path

from .base import BaseAgent


class CodingAgent(BaseAgent):
    """
    Coding agent that generates and hot-reloads trading strategy code.
    """

    def __init__(self, strategies_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.strategies_dir = Path(strategies_dir) if strategies_dir else Path(tempfile.gettempdir()) / "brainnet_strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_strategies: dict[str, Callable] = {}

    def generate_strategy_code(
        self,
        requirements: str,
        strategy_name: str = "generated_strategy",
    ) -> str:
        """
        Generate Python strategy code based on requirements.

        Args:
            requirements: Natural language description of strategy
            strategy_name: Name for the generated strategy

        Returns:
            Generated Python code as string
        """
        prompt = f"""Generate a Python trading strategy function based on these requirements:

{requirements}

Requirements for the code:
1. Function must be named `{strategy_name}`
2. Function signature: `def {strategy_name}(data: pd.DataFrame) -> str`
3. Must return exactly one of: 'long', 'short', or 'flat'
4. Input `data` is a pandas DataFrame with columns: Open, High, Low, Close, Volume
5. Include necessary imports at the top
6. Add docstring explaining the strategy logic
7. Handle edge cases (empty data, insufficient bars)

Generate ONLY the Python code, no explanations:"""

        code = self.llm.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )

        # Clean up the code
        code = self._clean_generated_code(code)
        return code

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code, removing markdown artifacts."""
        # Remove markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1]
        if "```" in code:
            code = code.split("```")[0]

        # Ensure proper imports
        required_imports = [
            "import pandas as pd",
            "import numpy as np",
        ]

        lines = code.strip().split('\n')
        existing_imports = [l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')]

        for imp in required_imports:
            if not any(imp in existing for existing in existing_imports):
                code = imp + '\n' + code

        return code.strip()

    def validate_strategy_code(self, code: str) -> tuple[bool, str]:
        """
        Validate generated strategy code.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Syntax check
            compile(code, '<string>', 'exec')

            # Check for required function
            if 'def ' not in code:
                return False, "No function definition found"

            # Check for return statement
            if 'return' not in code:
                return False, "No return statement found"

            # Check for valid return values
            valid_returns = ["'long'", "'short'", "'flat'", '"long"', '"short"', '"flat"']
            has_valid_return = any(ret in code for ret in valid_returns)
            if not has_valid_return:
                return False, "Must return 'long', 'short', or 'flat'"

            return True, ""

        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def save_strategy(self, code: str, strategy_name: str) -> Path:
        """
        Save strategy code to file.

        Args:
            code: Python code
            strategy_name: Name for the strategy file

        Returns:
            Path to saved file
        """
        filepath = self.strategies_dir / f"{strategy_name}.py"
        filepath.write_text(code)
        return filepath

    def hot_reload_strategy(
        self,
        code: str,
        strategy_name: str = "dynamic_strategy",
    ) -> Optional[Callable]:
        """
        Hot-reload a strategy from code string.

        Args:
            code: Python code containing the strategy
            strategy_name: Name of the strategy function

        Returns:
            The loaded strategy function, or None if loading failed
        """
        # Validate first
        is_valid, error = self.validate_strategy_code(code)
        if not is_valid:
            print(f"Strategy validation failed: {error}")
            return None

        # Save to file
        filepath = self.save_strategy(code, strategy_name)

        try:
            # Load module
            spec = importlib.util.spec_from_file_location(strategy_name, filepath)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[strategy_name] = module
            spec.loader.exec_module(module)

            # Get the strategy function
            if hasattr(module, strategy_name):
                strategy_func = getattr(module, strategy_name)
                self.loaded_strategies[strategy_name] = strategy_func
                return strategy_func

            # Try to find any callable that looks like a strategy
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith('_'):
                    self.loaded_strategies[strategy_name] = obj
                    return obj

            return None

        except Exception as e:
            print(f"Hot reload failed: {e}")
            return None

    def execute_strategy(
        self,
        strategy_name: str,
        data,
    ) -> Optional[str]:
        """
        Execute a loaded strategy.

        Args:
            strategy_name: Name of the strategy to execute
            data: DataFrame to pass to strategy

        Returns:
            Strategy decision ('long', 'short', 'flat') or None
        """
        if strategy_name not in self.loaded_strategies:
            return None

        try:
            result = self.loaded_strategies[strategy_name](data)
            if result in ['long', 'short', 'flat']:
                return result
            return 'flat'
        except Exception as e:
            print(f"Strategy execution failed: {e}")
            return None

    def improve_strategy(
        self,
        current_code: str,
        feedback: str,
        performance_metrics: Optional[dict] = None,
    ) -> str:
        """
        Improve an existing strategy based on feedback.

        Args:
            current_code: Current strategy code
            feedback: Natural language feedback
            performance_metrics: Optional dict with win_rate, sharpe, etc.

        Returns:
            Improved strategy code
        """
        metrics_section = ""
        if performance_metrics:
            metrics_section = f"""
Current performance metrics:
- Win rate: {performance_metrics.get('win_rate', 'N/A')}
- Sharpe ratio: {performance_metrics.get('sharpe', 'N/A')}
- Max drawdown: {performance_metrics.get('max_drawdown', 'N/A')}
"""

        prompt = f"""Improve this trading strategy based on feedback.

Current strategy code:
```python
{current_code}
```
{metrics_section}
Feedback for improvement:
{feedback}

Generate the improved strategy code. Keep the same function signature.
Return ONLY the Python code:"""

        improved_code = self.llm.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )

        return self._clean_generated_code(improved_code)

    def list_strategies(self) -> list[str]:
        """List all saved strategies."""
        return [f.stem for f in self.strategies_dir.glob("*.py")]
