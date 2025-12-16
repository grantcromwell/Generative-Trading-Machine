"""
Example of creating a custom agent.
"""

from brainnet.agents.base import BaseAgent


class SentimentAgent(BaseAgent):
    """Custom agent for market sentiment analysis."""

    def analyze_sentiment(self, headlines: list[str]) -> dict:
        text = "\n".join(f"- {h}" for h in headlines)
        prompt = f"""Analyze sentiment from these headlines:
{text}

Provide: sentiment (bullish/bearish/neutral), confidence (0-1), key themes."""

        response = self.llm.generate([{"role": "user", "content": prompt}])
        return {"response": response, "count": len(headlines)}

    def combine_with_technical(self, sentiment: dict, technical: str) -> str:
        prompt = f"""Given:
- Sentiment: {sentiment['response']}
- Technical: {technical}

Provide final recommendation (long/short/flat) with reasoning."""
        return self.llm.generate([{"role": "user", "content": prompt}])


def main():
    agent = SentimentAgent()
    headlines = [
        "Fed signals potential rate cut",
        "Tech earnings beat expectations",
        "Oil prices surge on supply concerns",
    ]

    sentiment = agent.analyze_sentiment(headlines)
    print("Sentiment:", sentiment['response'])

    decision = agent.combine_with_technical(sentiment, "RSI oversold, MACD bullish")
    print("\nDecision:", decision)


if __name__ == "__main__":
    main()
