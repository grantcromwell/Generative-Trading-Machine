"""
Basic usage example for Brainnet trading system.
"""

import numpy as np
import pandas as pd
from brainnet.agents import ResearchAgent, ReasoningAgent
from brainnet.core import load_config, MemoryManager


def main():
    config = load_config()
    research_agent = ResearchAgent()
    reasoning_agent = ReasoningAgent()

    try:
        memory = MemoryManager(config)
    except Exception as e:
        print(f"Memory init failed: {e}")
        memory = None

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.DataFrame({'Close': prices})

    print("Running GAF analysis...")
    memory_context = memory.get_context("recent patterns") if memory else ""

    analysis_result = research_agent.research(data, memory_context)
    print(f"Analysis: {analysis_result['analysis'][:200]}...")

    print("\nComputing confidence...")
    confidence = reasoning_agent.compute_confidence(analysis_result['analysis'])
    print(f"Confidence: {confidence:.2f}")

    decision = reasoning_agent.decide(analysis_result['analysis'], confidence)
    print(f"Decision: {decision}")

    if memory:
        memory.add({"decision": decision, "confidence": confidence})
        print("\nStored in memory.")


if __name__ == "__main__":
    main()
