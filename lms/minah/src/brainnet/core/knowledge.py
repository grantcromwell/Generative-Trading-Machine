"""
Knowledge base for storing structured trading knowledge
"""

import json
from datetime import datetime
from typing import Optional, Any
from pathlib import Path


class KnowledgeBase:
    """
    Knowledge base for storing structured trading knowledge,
    patterns, and market regime information.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("knowledge_base.json")
        self.knowledge: dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load knowledge from storage."""
        if self.storage_path.exists():
            try:
                self.knowledge = json.loads(self.storage_path.read_text())
            except json.JSONDecodeError:
                self.knowledge = {}

    def _save(self):
        """Save knowledge to storage."""
        self.storage_path.write_text(json.dumps(self.knowledge, indent=2, default=str))

    def add_knowledge(
        self,
        key: str,
        value: Any,
        category: str = "general",
        metadata: Optional[dict] = None,
    ):
        """
        Add knowledge to the base.

        Args:
            key: Unique identifier for this knowledge
            value: The knowledge content
            category: Category for organization
            metadata: Additional metadata
        """
        if category not in self.knowledge:
            self.knowledge[category] = {}

        self.knowledge[category][key] = {
            "value": value,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def get_knowledge(
        self,
        key: str,
        category: str = "general",
        default: Any = None,
    ) -> Any:
        """
        Retrieve knowledge by key.

        Args:
            key: Knowledge identifier
            category: Category to search in
            default: Default value if not found

        Returns:
            The knowledge value or default
        """
        if category in self.knowledge and key in self.knowledge[category]:
            return self.knowledge[category][key]["value"]
        return default

    def update_knowledge(
        self,
        key: str,
        value: Any,
        category: str = "general",
    ):
        """Update existing knowledge."""
        if category in self.knowledge and key in self.knowledge[category]:
            self.knowledge[category][key]["value"] = value
            self.knowledge[category][key]["updated_at"] = datetime.now().isoformat()
            self._save()
        else:
            self.add_knowledge(key, value, category)

    def delete_knowledge(self, key: str, category: str = "general") -> bool:
        """Delete knowledge by key."""
        if category in self.knowledge and key in self.knowledge[category]:
            del self.knowledge[category][key]
            self._save()
            return True
        return False

    def get_category(self, category: str) -> dict:
        """Get all knowledge in a category."""
        return self.knowledge.get(category, {})

    def list_categories(self) -> list[str]:
        """List all categories."""
        return list(self.knowledge.keys())

    def search(self, query: str) -> list[dict]:
        """
        Simple search across all knowledge.

        Args:
            query: Search string

        Returns:
            List of matching knowledge entries
        """
        results = []
        query_lower = query.lower()

        for category, items in self.knowledge.items():
            for key, entry in items.items():
                value_str = str(entry["value"]).lower()
                if query_lower in key.lower() or query_lower in value_str:
                    results.append({
                        "category": category,
                        "key": key,
                        "value": entry["value"],
                        "metadata": entry["metadata"],
                    })

        return results

    # Trading-specific knowledge methods

    def add_pattern(
        self,
        pattern_name: str,
        description: str,
        success_rate: float,
        market_conditions: list[str],
    ):
        """Add a trading pattern to knowledge base."""
        self.add_knowledge(
            key=pattern_name,
            value={
                "description": description,
                "success_rate": success_rate,
                "market_conditions": market_conditions,
            },
            category="patterns",
        )

    def get_pattern(self, pattern_name: str) -> Optional[dict]:
        """Get a trading pattern."""
        return self.get_knowledge(pattern_name, category="patterns")

    def add_market_regime(
        self,
        regime_name: str,
        characteristics: dict,
        recommended_strategies: list[str],
    ):
        """Add market regime information."""
        self.add_knowledge(
            key=regime_name,
            value={
                "characteristics": characteristics,
                "recommended_strategies": recommended_strategies,
            },
            category="market_regimes",
        )

    def get_market_regime(self, regime_name: str) -> Optional[dict]:
        """Get market regime information."""
        return self.get_knowledge(regime_name, category="market_regimes")

    def log_trade_outcome(
        self,
        trade_id: str,
        entry_price: float,
        exit_price: float,
        direction: str,
        pnl: float,
        pattern_used: Optional[str] = None,
        notes: str = "",
    ):
        """Log a trade outcome for learning."""
        self.add_knowledge(
            key=trade_id,
            value={
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": direction,
                "pnl": pnl,
                "pattern_used": pattern_used,
                "notes": notes,
            },
            category="trade_history",
        )

    def get_pattern_performance(self, pattern_name: str) -> dict:
        """Calculate performance statistics for a pattern."""
        trades = self.get_category("trade_history")
        pattern_trades = [
            t for t in trades.values()
            if t["value"].get("pattern_used") == pattern_name
        ]

        if not pattern_trades:
            return {"count": 0, "win_rate": 0, "avg_pnl": 0}

        wins = sum(1 for t in pattern_trades if t["value"]["pnl"] > 0)
        total_pnl = sum(t["value"]["pnl"] for t in pattern_trades)

        return {
            "count": len(pattern_trades),
            "win_rate": wins / len(pattern_trades),
            "avg_pnl": total_pnl / len(pattern_trades),
        }
