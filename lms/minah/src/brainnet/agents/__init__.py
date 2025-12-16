from .base import Phi35MiniClient, BaseAgent
from .research import ResearchAgent
from .reasoning import ReasoningAgent
from .coding import CodingAgent

# Conditional ConvNeXt import (requires torch)
# Graceful degradation when PyTorch is not installed
_CONVNEXT_AVAILABLE = False
ConvNeXtPredictor = None
ConvNeXtPrediction = None

try:
    from .convnext_predictor import ConvNeXtPredictor, ConvNeXtPrediction
    _CONVNEXT_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "Phi35MiniClient",
    "BaseAgent",
    "ResearchAgent",
    "ReasoningAgent",
    "CodingAgent",
    "ConvNeXtPredictor",
    "ConvNeXtPrediction",
    "_CONVNEXT_AVAILABLE",
]
