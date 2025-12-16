"""
Router - Supervisor that triggers adaptation loops
"""

from typing import Optional
import pandas as pd

from .graph import build_graph, create_initial_state, TradingState
from brainnet.agents import CodingAgent
from brainnet.core import MemoryManager, load_config


class Router:
    """
    Supervisor router that manages the trading workflow
    and triggers adaptation loops when needed.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.graph = None
        self.memory = None
        self.coding_agent = None

    def _ensure_initialized(self):
        """Lazy initialization of components."""
        if self.graph is None:
            self.graph = build_graph()
        if self.memory is None:
            self.memory = MemoryManager(self.config)

    def trigger(
        self,
        data,
        symbol: str = "ES=F",
        confidence_threshold: Optional[float] = None,
        max_refinements: Optional[int] = None,
    ) -> dict:
        """
        Trigger the trading workflow.
        """
        self._ensure_initialized()

        threshold = confidence_threshold or self.config.get("confidence_threshold", 0.78)
        max_ref = max_refinements or self.config.get("max_refinements", 3)

        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict()
        else:
            data_dict = data

        initial_state = create_initial_state(
            data=data_dict,
            symbol=symbol,
            confidence_threshold=threshold,
            max_refinements=max_ref,
        )

        result = self.graph.invoke(initial_state)

        self._store_result(result)

        return {
            "decision": result.get("final_decision", "flat"),
            "confidence": result.get("confidence", 0.0),
            "analysis": result.get("analysis"),
            "refinements": result.get("refinement_count", 0),
            "error": result.get("error"),
        }

    def _store_result(self, result: dict):
        """Store trading result in memory."""
        try:
            self.memory.add({
                "decision": result.get("final_decision"),
                "confidence": result.get("confidence"),
                "refinements": result.get("refinement_count"),
                "pattern_scores": result.get("pattern_scores"),
            })
        except Exception:
            pass

    def adapt(self, feedback: str, performance: Optional[dict] = None) -> bool:
        """
        Trigger strategy adaptation based on feedback.
        """
        if self.coding_agent is None:
            self.coding_agent = CodingAgent()

        code = self.coding_agent.generate_strategy_code(feedback)
        strategy = self.coding_agent.hot_reload_strategy(code)

        return strategy is not None

    def get_memory_context(self, query: str) -> str:
        """Get relevant context from memory."""
        self._ensure_initialized()
        return self.memory.get_context(query)
