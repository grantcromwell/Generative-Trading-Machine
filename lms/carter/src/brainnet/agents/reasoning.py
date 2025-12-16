"""
Reasoning Agent - LLM-as-judge confidence intervals + decision logic
Uses Binary Symmetric Channel (BSC) model for calibrated confidence
"""

import re
import math
from typing import Optional

import numpy as np

from .base import BaseAgent


class ReasoningAgent(BaseAgent):
    """
    Reasoning agent that computes calibrated confidence intervals
    using LLM-as-a-judge approach with Binary Symmetric Channel model.
    """

    def __init__(self, confidence_threshold: float = 0.78, **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold

    def compute_confidence(self, analysis: str) -> float:
        """
        Compute calibrated confidence interval using BSC model.

        The Binary Symmetric Channel model treats the LLM's judgment
        as a noisy channel where:
        - ε = error probability (crossover probability)
        - Confidence = 1 - H(ε) where H is binary entropy

        Args:
            analysis: The analysis text to evaluate

        Returns:
            Confidence score between 0 and 1
        """
        # Ask LLM to self-evaluate using structured prompting
        evaluation_prompt = f"""You are evaluating the confidence of a trading analysis.

Analysis to evaluate:
{analysis}

Using the Binary Symmetric Channel (BSC) model for calibration:
1. Estimate the error probability (ε) - the chance this analysis is wrong
2. Consider: data quality, pattern clarity, market regime stability
3. Account for known biases in pattern recognition

Provide your evaluation in EXACTLY this format:
ERROR_PROBABILITY: [value between 0.0 and 0.5]
CONFIDENCE_LOW: [lower bound, 0.0-1.0]
CONFIDENCE_HIGH: [upper bound, 0.0-1.0]
REASONING: [brief explanation]"""

        response = self.llm.generate(
            [{"role": "user", "content": evaluation_prompt}],
            max_tokens=512,
            temperature=0.3,  # Lower temperature for more consistent evaluation
        )

        # Parse the response
        confidence = self._parse_confidence_response(response)
        return confidence

    def _parse_confidence_response(self, response: str) -> float:
        """Parse confidence values from LLM response."""
        try:
            # Extract error probability
            error_match = re.search(r'ERROR_PROBABILITY:\s*([\d.]+)', response)
            if error_match:
                epsilon = float(error_match.group(1))
                epsilon = min(0.5, max(0.0, epsilon))  # Clamp to valid range
                # BSC capacity-based confidence
                if epsilon == 0:
                    return 1.0
                elif epsilon >= 0.5:
                    return 0.5
                else:
                    # Confidence based on channel capacity: C = 1 - H(ε)
                    h_epsilon = -epsilon * math.log2(epsilon) - (1 - epsilon) * math.log2(1 - epsilon)
                    return 1 - h_epsilon

            # Fallback: try to extract confidence bounds directly
            low_match = re.search(r'CONFIDENCE_LOW:\s*([\d.]+)', response)
            high_match = re.search(r'CONFIDENCE_HIGH:\s*([\d.]+)', response)

            if low_match and high_match:
                low = float(low_match.group(1))
                high = float(high_match.group(1))
                return (low + high) / 2

            # Last resort: look for any confidence-like number
            numbers = re.findall(r'(\d+\.?\d*)', response)
            valid_numbers = [float(n) for n in numbers if 0 <= float(n) <= 1]
            if valid_numbers:
                return np.mean(valid_numbers)

            return 0.5  # Default uncertainty

        except Exception:
            return 0.5

    def decide(
        self,
        analysis: str,
        confidence: float,
        memory_context: Optional[str] = None,
    ) -> str:
        """
        Make a trading decision based on analysis and confidence.

        Args:
            analysis: The pattern analysis
            confidence: Calibrated confidence score
            memory_context: Optional context from memory

        Returns:
            Decision string: 'long', 'short', 'flat', or 'refine'
        """
        # If confidence is below threshold, request refinement
        if confidence < self.confidence_threshold:
            return "refine"

        # Build decision prompt
        context_section = ""
        if memory_context:
            context_section = f"""
Historical context from memory:
{memory_context}
"""

        decision_prompt = f"""Based on the following analysis, make a trading decision.

Analysis:
{analysis}

Confidence Score: {confidence:.3f} (threshold: {self.confidence_threshold})
{context_section}

Consider:
1. Pattern strength and clarity
2. Risk/reward ratio
3. Current market regime
4. Position sizing implications

Provide your decision in EXACTLY this format:
DECISION: [LONG/SHORT/FLAT]
CONVICTION: [1-10]
REASONING: [2-3 sentences explaining the decision]
RISK_FACTORS: [key risks to monitor]"""

        response = self.llm.generate(
            [{"role": "user", "content": decision_prompt}],
            max_tokens=512,
            temperature=0.4,
        )

        # Parse decision
        return self._parse_decision_response(response)

    def _parse_decision_response(self, response: str) -> str:
        """Parse decision from LLM response."""
        response_upper = response.upper()

        # Look for explicit decision
        decision_match = re.search(r'DECISION:\s*(LONG|SHORT|FLAT)', response_upper)
        if decision_match:
            return decision_match.group(1).lower()

        # Fallback: look for keywords
        if 'LONG' in response_upper and 'SHORT' not in response_upper:
            return 'long'
        elif 'SHORT' in response_upper and 'LONG' not in response_upper:
            return 'short'
        else:
            return 'flat'

    def evaluate_outcome(
        self,
        prediction: str,
        actual_return: float,
        holding_period: int,
    ) -> dict:
        """
        Evaluate the outcome of a prediction for learning.

        Args:
            prediction: The original prediction (long/short/flat)
            actual_return: Realized return over holding period
            holding_period: Number of bars held

        Returns:
            Evaluation dictionary for memory storage
        """
        # Determine if prediction was correct
        if prediction == 'long':
            correct = actual_return > 0
        elif prediction == 'short':
            correct = actual_return < 0
        else:  # flat
            correct = abs(actual_return) < 0.01  # Small move

        return {
            "prediction": prediction,
            "actual_return": actual_return,
            "holding_period": holding_period,
            "correct": correct,
            "magnitude": abs(actual_return),
        }

    def compute_ensemble_confidence(
        self,
        analyses: list[str],
        weights: Optional[list[float]] = None,
    ) -> float:
        """
        Compute ensemble confidence from multiple analyses.

        Args:
            analyses: List of analysis strings
            weights: Optional weights for each analysis

        Returns:
            Weighted average confidence
        """
        if not analyses:
            return 0.5

        if weights is None:
            weights = [1.0 / len(analyses)] * len(analyses)

        confidences = [self.compute_confidence(a) for a in analyses]
        return sum(c * w for c, w in zip(confidences, weights))
