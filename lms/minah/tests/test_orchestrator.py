"""Tests for orchestrator module."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


class TestGraph:
    @patch('brainnet.orchestrator.graph.ResearchAgent')
    @patch('brainnet.orchestrator.graph.ReasoningAgent')
    def test_build_graph(self, mock_reason, mock_research):
        mock_research.return_value.research.return_value = {
            "analysis": "trend", "image": "b64", "scores": {}
        }
        mock_reason.return_value.compute_confidence.return_value = 0.85
        mock_reason.return_value.decide.return_value = "long"

        from brainnet.orchestrator.graph import build_graph
        graph = build_graph()
        assert graph is not None


class TestRouter:
    @patch('brainnet.orchestrator.router.build_graph')
    @patch('brainnet.orchestrator.router.MemoryManager')
    def test_trigger(self, mock_mem, mock_build):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_decision": "long",
            "confidence": 0.9,
            "analysis": {"analysis": "trend"},
            "refinement_count": 0,
        }
        mock_build.return_value = mock_graph

        from brainnet.orchestrator.router import Router
        router = Router()
        result = router.trigger({"Close": [1, 2, 3]})
        assert result["decision"] == "long"
