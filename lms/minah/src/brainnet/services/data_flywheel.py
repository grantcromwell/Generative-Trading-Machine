"""
Data Flywheel implementation for Minah to enable continuous data improvement.
This module handles the collection, refinement, and updating of trading data for model retraining.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from brainnet.core.config import get_config


class DataFlywheel:
    """
    Manages the data flywheel process for continuous improvement of trading data and models in Minah.
    Collects trading outcomes and feedback, refines data, and prepares it for model updates.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.data_store = []  # In-memory store for trading data and feedback
        self.feedback_store = []  # Store for user feedback

    def collect_trading_data(self, trading_result: Dict[str, any]):
        """Collect data from a trading analysis result for the flywheel."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": trading_result.get("symbol", "unknown"),
            "decision": trading_result.get("decision", "flat"),
            "confidence": trading_result.get("confidence", 0.0),
            "trend": trading_result.get("trend", ""),
            "momentum": trading_result.get("momentum", ""),
            "features": trading_result.get("features", {}),
            "refinements": trading_result.get("refinements", 0),
            # Derivatives data from Coinglass
            "funding_rate": trading_result.get("funding_rate"),
            "long_short_ratio": trading_result.get("long_short_ratio"),
            "funding_sentiment": trading_result.get("funding_sentiment"),
            "positioning_sentiment": trading_result.get("positioning_sentiment"),
        }
        self.data_store.append(record)
        return record

    def collect_feedback(self, feedback: str, symbol: str, decision: str):
        """Collect user feedback on a trading decision."""
        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "decision": decision,
            "feedback": feedback
        }
        self.feedback_store.append(feedback_record)
        return feedback_record

    def refine_data(self) -> List[Dict[str, any]]:
        """Refine collected data based on feedback and performance metrics for model retraining."""
        refined_data = []
        for trading_data in self.data_store:
            # Find related feedback
            related_feedback = [f for f in self.feedback_store if f["symbol"] == trading_data["symbol"] and f["decision"] == trading_data["decision"]]
            feedback_text = "; ".join([f["feedback"] for f in related_feedback]) if related_feedback else ""
            
            # Create refined record with feedback if available
            refined_record = {
                "symbol": trading_data["symbol"],
                "decision": trading_data["decision"],
                "confidence": trading_data["confidence"],
                "trend": trading_data["trend"],
                "momentum": trading_data["momentum"],
                "features": trading_data["features"],
                "timestamp": trading_data["timestamp"],
                "feedback": feedback_text,
                "quality_score": self._calculate_quality_score(trading_data, related_feedback)
            }
            refined_data.append(refined_record)
        return refined_data

    def _calculate_quality_score(self, trading_data: Dict[str, any], feedback: List[Dict[str, any]]) -> float:
        """Calculate a quality score for a trading data point based on confidence and feedback."""
        confidence = trading_data["confidence"]
        feedback_bonus = 0.1 if feedback else 0.0  # Bonus for having feedback
        negative_feedback_penalty = -0.2 if any("incorrect" in f["feedback"].lower() or "wrong" in f["feedback"].lower() for f in feedback) else 0.0
        return min(1.0, max(0.0, confidence + feedback_bonus + negative_feedback_penalty))

    def export_for_retraining(self, output_path: str):
        """Export refined data as a dataset for model retraining."""
        refined_data = self.refine_data()
        if not refined_data:
            return False
        
        df = pd.DataFrame(refined_data)
        df.to_csv(output_path, index=False)
        return True

    def clear_data(self):
        """Clear collected data after export or when resetting the flywheel."""
        self.data_store = []
        self.feedback_store = []
