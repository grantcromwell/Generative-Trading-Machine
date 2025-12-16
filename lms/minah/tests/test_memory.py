"""Tests for memory module."""

import pytest
from unittest.mock import patch, MagicMock


class TestMemoryManager:
    @patch('brainnet.core.memory.MEM0_AVAILABLE', False)
    def test_sqlite_fallback(self):
        from brainnet.core.memory import MemoryManager
        config = {"memory_db": "sqlite", "sqlite_path": ":memory:"}
        manager = MemoryManager(config)
        assert manager.use_fallback

    @patch('brainnet.core.memory.MEM0_AVAILABLE', False)
    def test_add_and_search(self):
        from brainnet.core.memory import MemoryManager
        config = {"memory_db": "sqlite", "sqlite_path": ":memory:"}
        manager = MemoryManager(config)

        manager.add({"decision": "long", "confidence": 0.85})
        results = manager.search("long")
        assert len(results) > 0

    @patch('brainnet.core.memory.MEM0_AVAILABLE', False)
    def test_get_context(self):
        from brainnet.core.memory import MemoryManager
        config = {"memory_db": "sqlite", "sqlite_path": ":memory:"}
        manager = MemoryManager(config)

        manager.add("trade was successful")
        context = manager.get_context("trade")
        assert "trade" in context.lower()
