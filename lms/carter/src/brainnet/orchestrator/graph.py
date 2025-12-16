"""
LangGraph StateGraph with conditional routing for adaptive trading loops
"""

from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph import StateGraph, END

from brainnet.agents import ResearchAgent, ReasoningAgent
from brainnet.core import MemoryManager, load_config


class TradingState(TypedDict):
    """State container for the trading graph."""
    # Input data
    data: dict
    symbol: str

    # Analysis results
    analysis: Optional[dict]
    gaf_image: Optional[str]
    pattern_scores: Optional[dict]

    # Reasoning results
    confidence: float
    decision: str

    # Memory and context
    memory_context: str
    refinement_count: int

    # Configuration
    confidence_threshold: float
    max_refinements: int

    # Output
    final_decision: Optional[str]
    error: Optional[str]


def research_node(state: TradingState) -> dict:
    """
    Research node: Generate GAF and analyze patterns.

    This node:
    1. Creates GAF image from price data
    2. Analyzes patterns using Phi-3.5-Mini vision
    3. Extracts pattern scores
    """
    try:
        agent = ResearchAgent()

        # Run research with memory context
        result = agent.research(
            data=state["data"],
            memory_context=state.get("memory_context", ""),
        )

        return {
            "analysis": result,
            "gaf_image": result.get("image"),
            "pattern_scores": result.get("scores"),
        }

    except Exception as e:
        return {
            "error": f"Research failed: {str(e)}",
            "analysis": None,
        }


def reasoning_node(state: TradingState) -> dict:
    """
    Reasoning node: Compute confidence and make decision.

    This node:
    1. Evaluates analysis confidence using BSC model
    2. Makes trading decision if confidence is sufficient
    3. Requests refinement if confidence is too low
    """
    try:
        agent = ReasoningAgent(
            confidence_threshold=state.get("confidence_threshold", 0.78)
        )

        # Get analysis text
        analysis = state.get("analysis", {})
        analysis_text = analysis.get("analysis", "") if isinstance(analysis, dict) else str(analysis)

        if not analysis_text:
            return {
                "confidence": 0.0,
                "decision": "flat",
                "error": "No analysis available",
            }

        # Compute confidence
        confidence = agent.compute_confidence(analysis_text)

        # Make decision
        decision = agent.decide(
            analysis_text,
            confidence,
            memory_context=state.get("memory_context"),
        )

        return {
            "confidence": confidence,
            "decision": decision,
        }

    except Exception as e:
        return {
            "confidence": 0.0,
            "decision": "flat",
            "error": f"Reasoning failed: {str(e)}",
        }


def refinement_node(state: TradingState) -> dict:
    """
    Refinement node: Enrich context from memory for re-analysis.
    """
    try:
        config = load_config()
        memory = MemoryManager(config)

        # Get additional context
        analysis = state.get("analysis", {})
        analysis_text = analysis.get("analysis", "") if isinstance(analysis, dict) else str(analysis)

        additional_context = memory.get_context(
            f"pattern analysis: {analysis_text[:100]}"
        )

        current_context = state.get("memory_context", "")
        new_context = f"{current_context}\n\nAdditional context:\n{additional_context}"

        return {
            "memory_context": new_context,
            "refinement_count": state.get("refinement_count", 0) + 1,
        }

    except Exception as e:
        return {
            "refinement_count": state.get("refinement_count", 0) + 1,
            "error": f"Refinement failed: {str(e)}",
        }


def finalize_node(state: TradingState) -> dict:
    """
    Finalize node: Set final decision and prepare output.
    """
    decision = state.get("decision", "flat")

    # Normalize decision
    if decision in ["long", "short", "flat"]:
        final = decision
    elif "refine" in decision.lower():
        final = "flat"  # Default to flat if still refining
    else:
        final = "flat"

    return {
        "final_decision": final,
    }


def should_refine(state: TradingState) -> Literal["refine", "finalize"]:
    """
    Conditional edge: Determine if we should refine or finalize.

    Returns to research if:
    - Confidence is below threshold
    - We haven't exceeded max refinements
    """
    confidence = state.get("confidence", 0.0)
    threshold = state.get("confidence_threshold", 0.78)
    refinement_count = state.get("refinement_count", 0)
    max_refinements = state.get("max_refinements", 3)
    decision = state.get("decision", "")

    # Check if refinement is needed and allowed
    needs_refinement = (
        confidence < threshold or
        decision == "refine"
    )
    can_refine = refinement_count < max_refinements

    if needs_refinement and can_refine:
        return "refine"
    else:
        return "finalize"


def build_graph() -> StateGraph:
    """
    Build the LangGraph trading workflow.

    Flow:
    1. research -> reasoning
    2. reasoning -> (conditional) -> refine OR finalize
    3. refine -> research (loop back)
    4. finalize -> END

    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(TradingState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("refine", refinement_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("research")

    # Add edges
    workflow.add_edge("research", "reasoning")

    # Conditional edge from reasoning
    workflow.add_conditional_edges(
        "reasoning",
        should_refine,
        {
            "refine": "refine",
            "finalize": "finalize",
        }
    )

    # Refine loops back to research
    workflow.add_edge("refine", "research")

    # Finalize goes to END
    workflow.add_edge("finalize", END)

    # Compile and return
    return workflow.compile()


def create_initial_state(
    data,
    symbol: str = "ES=F",
    confidence_threshold: float = 0.78,
    max_refinements: int = 3,
) -> TradingState:
    """
    Create initial state for the trading graph.

    Args:
        data: Market data (DataFrame or dict)
        symbol: Trading symbol
        confidence_threshold: Minimum confidence for trading
        max_refinements: Maximum refinement iterations

    Returns:
        Initial TradingState
    """
    return TradingState(
        data=data if isinstance(data, dict) else data.to_dict(),
        symbol=symbol,
        analysis=None,
        gaf_image=None,
        pattern_scores=None,
        confidence=0.0,
        decision="",
        memory_context="",
        refinement_count=0,
        confidence_threshold=confidence_threshold,
        max_refinements=max_refinements,
        final_decision=None,
        error=None,
    )
