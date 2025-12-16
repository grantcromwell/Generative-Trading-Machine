"""Tests for agents module."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


class TestPhi35MiniClient:
    @patch('brainnet.agents.base.OpenAI')
    def test_local_backend(self, mock_openai):
        from brainnet.agents.base import Phi35MiniClient
        with patch.dict('os.environ', {'LLM_BACKEND': 'local'}):
            client = Phi35MiniClient()
            assert client.backend == 'local'

    @patch('brainnet.agents.base.OpenAI')
    def test_generate(self, mock_openai):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Test"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_resp

        from brainnet.agents.base import Phi35MiniClient
        with patch.dict('os.environ', {'LLM_BACKEND': 'local'}):
            client = Phi35MiniClient()
            result = client.generate([{"role": "user", "content": "Hi"}])
            assert result == "Test"


class TestResearchAgent:
    def test_generate_gaf(self):
        from brainnet.agents.research import ResearchAgent
        agent = ResearchAgent()
        series = np.sin(np.linspace(0, 10, 100))
        img = agent.generate_gaf_image(series)
        assert isinstance(img, str)
        assert len(img) > 0


class TestReasoningAgent:
    @patch('brainnet.agents.reasoning.ReasoningAgent.llm')
    def test_confidence(self, mock_llm):
        mock_llm.generate.return_value = "ERROR_PROBABILITY: 0.1"
        from brainnet.agents.reasoning import ReasoningAgent
        agent = ReasoningAgent()
        agent.llm = mock_llm
        conf = agent.compute_confidence("test")
        assert 0 <= conf <= 1


class TestCodingAgent:
    @patch('brainnet.agents.coding.CodingAgent.llm')
    def test_generate_code(self, mock_llm):
        mock_llm.generate.return_value = "def strategy(data): return 'long'"
        from brainnet.agents.coding import CodingAgent
        agent = CodingAgent()
        agent.llm = mock_llm
        code = agent.generate_strategy_code("momentum")
        assert "def" in code
