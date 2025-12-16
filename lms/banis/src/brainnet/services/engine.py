"""
Trading Engine - Core analysis and trading loop logic
Powered by DeepSeek-V3 via HuggingFace Pipeline

Supports both LLM-based analysis and ConvNeXt neural pattern recognition.
"""

import time
from datetime import datetime
from typing import Optional

import yfinance as yf

from brainnet.agents import ResearchAgent, ReasoningAgent, _CONVNEXT_AVAILABLE
from brainnet.core import MemoryManager, load_config
from brainnet.services.coinglass import CoinglassClient


def run_single_analysis(symbol: str = "ES=F", interval: str = "5m") -> dict:
    """
    Run a single GAF analysis on a symbol using DeepSeek-V3.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"  BRAINNET/BANIS ANALYSIS: {symbol}")
    print(f"  Powered by DeepSeek-V3")
    print(f"{'='*60}")
    
    # Initialize agents and services
    config = load_config()
    research = ResearchAgent()
    reasoning = ReasoningAgent()
    coinglass = CoinglassClient(api_key=config.get("coinglass_api_key", ""))
    
    print(f"\n[1/5] Fetching {symbol} data...")
    data = yf.download(symbol, period="1d", interval=interval, progress=False)
    
    if data.empty:
        print("⚠ No data received")
        return {"symbol": symbol, "error": "No data", "decision": None, "confidence": 0}
    
    try:
        latest_price = float(data['Close'].iloc[-1].iloc[0]) if hasattr(data['Close'].iloc[-1], 'iloc') else float(data['Close'].iloc[-1])
    except:
        latest_price = float(data['Close'].iloc[-1])
    
    print(f"    {len(data)} bars | Latest: ${latest_price:.2f}")
    
    # Fetch derivatives data (funding rate, long/short ratio) for crypto
    derivatives = None
    funding_rate = None
    long_short_ratio = None
    if config.get("use_coinglass", True):
        print(f"\n[2/5] Fetching Derivatives Data (Coinglass)...")
        derivatives = coinglass.get_derivatives_metrics(symbol)
        if derivatives:
            if derivatives.funding_rate:
                funding_rate = derivatives.funding_rate.rate
                print(f"    Funding Rate: {funding_rate:.4%} ({derivatives.funding_sentiment})")
            if derivatives.long_short_ratio:
                long_short_ratio = derivatives.long_short_ratio.long_short_ratio
                print(f"    Long/Short Ratio: {long_short_ratio:.2f} ({derivatives.positioning_sentiment})")
        else:
            print(f"    ⚠ Derivatives data not available for {symbol}")
    else:
        print(f"\n[2/5] Skipping Derivatives Data (disabled)")
    
    print(f"\n[3/5] Running GAF Pattern Analysis...")
    analysis = research.research(data)
    
    print(f"\n    GAF Features:")
    for k, v in analysis['features'].items():
        indicator = ''
        if k == 'trend_score':
            indicator = ' ← UPTREND' if v > 0.1 else ' ← DOWNTREND' if v < -0.1 else ' ← NEUTRAL'
        elif k == 'momentum':
            indicator = ' ← STRENGTHENING' if v > 0 else ' ← WEAKENING'
        print(f"      {k:22s}: {v:+.4f}{indicator}")
    
    print(f"\n[4/5] Computing Confidence (BSC Model)...")
    confidence = reasoning.compute_confidence(analysis['analysis'])
    print(f"    Confidence: {confidence:.3f}")
    print(f"    Threshold:  0.650")
    
    refinements = 0
    memory = None
    try:
        memory = MemoryManager(config)
    except:
        pass
    
    memory_context = ""
    while confidence < 0.65 and refinements < 3:  # Banis threshold
        refinements += 1
        print(f"\n    ⟳ Refinement {refinements}/3")
        if memory:
            memory_context += " " + memory.get_context("pattern refinement")
        analysis = research.research(data, memory_context)
        confidence = reasoning.compute_confidence(analysis['analysis'])
        print(f"    New confidence: {confidence:.3f}")
    
    print(f"\n[5/5] Making Trading Decision...")
    # Prepare derivatives data for decision making
    derivatives_dict = None
    if derivatives:
        derivatives_dict = {
            'funding_rate': funding_rate,
            'long_short_ratio': long_short_ratio,
            'funding_sentiment': derivatives.funding_sentiment,
            'positioning_sentiment': derivatives.positioning_sentiment,
        }
    
    decision = reasoning.decide(analysis['analysis'], confidence, memory_context, derivatives_dict)
    
    if "long" in decision.lower():
        pos, emoji = "LONG", "📈"
    elif "short" in decision.lower():
        pos, emoji = "SHORT", "📉"
    elif "flat" in decision.lower():
        pos, emoji = "FLAT", "⏸"
    else:
        pos, emoji = "REFINE", "🔄"
    
    print(f"\n    {emoji} {pos} | Confidence: {confidence:.1%}")
    
    if memory:
        try:
            memory.add({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "decision": pos,
                "confidence": confidence,
                "refinements": refinements,
            })
            print("    ✓ Stored in memory")
        except:
            pass
    
    trend = "Bullish" if analysis['features']['trend_score'] > 0 else "Bearish"
    momentum = "Positive" if analysis['features']['momentum'] > 0 else "Negative"
    
    result = {
        "symbol": symbol,
        "price": latest_price,
        "decision": pos,
        "confidence": confidence,
        "trend": trend,
        "momentum": momentum,
        "features": analysis['features'],
        "refinements": refinements,
        # Derivatives data
        "funding_rate": funding_rate,
        "long_short_ratio": long_short_ratio,
        "funding_sentiment": derivatives.funding_sentiment if derivatives else None,
        "positioning_sentiment": derivatives.positioning_sentiment if derivatives else None,
    }
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Symbol:     {symbol}")
    print(f"  Price:      ${latest_price:.2f}")
    if funding_rate is not None:
        print(f"  Funding:    {funding_rate:.4%} ({derivatives.funding_sentiment})")
    if long_short_ratio is not None:
        print(f"  L/S Ratio:  {long_short_ratio:.2f} ({derivatives.positioning_sentiment})")
    print(f"  Trend:      {trend}")
    print(f"  Momentum:   {momentum}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Signal:     {pos}")
    print(f"{'='*60}\n")
    
    return result


def run_trading_loop(symbol: str = "ES=F", interval: str = "5m", delay: int = 300):
    """Continuous trading loop with periodic analysis."""
    config = load_config()
    
    try:
        memory = MemoryManager(config)
        print("✓ Memory initialized")
    except Exception as e:
        print(f"⚠ Memory failed: {e}")
        memory = None
    
    research = ResearchAgent()
    reasoning = ReasoningAgent()
    print("✓ Agents initialized (DeepSeek-V3)")
    
    iteration = 0
    
    while True:
        iteration += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*50}")
        print(f"Iteration {iteration} | {ts}")
        
        try:
            print(f"\n[1/5] Fetching {symbol} {interval} data...")
            data = yf.download(symbol, period="1d", interval=interval, progress=False)
            
            if data.empty:
                print("⚠ No data")
                time.sleep(60)
                continue
            
            try:
                latest = float(data['Close'].iloc[-1].iloc[0]) if hasattr(data['Close'].iloc[-1], 'iloc') else float(data['Close'].iloc[-1])
            except:
                latest = float(data['Close'].iloc[-1])
            
            print(f"    {len(data)} bars, latest: ${latest:.2f}")
            
            memory_context = ""
            if memory:
                print("\n[2/5] Getting memory context...")
                memory_context = memory.get_context("recent trades")
            
            print("\n[3/5] GAF analysis...")
            analysis = research.research(data, memory_context)
            print(f"    {analysis['analysis'][:100]}...")
            
            print("\n[4/5] Computing confidence...")
            confidence = reasoning.compute_confidence(analysis['analysis'])
            print(f"    Confidence: {confidence:.3f}")
            
            refinements = 0
            while confidence < 0.65 and refinements < 3:  # Banis threshold
                refinements += 1
                print(f"\n    ⟳ Refinement {refinements}/3")
                if memory:
                    memory_context += " " + memory.get_context("pattern refinement")
                analysis = research.research(data, memory_context)
                confidence = reasoning.compute_confidence(analysis['analysis'])
                print(f"    New confidence: {confidence:.3f}")
            
            print("\n[5/5] Decision...")
            decision = reasoning.decide(analysis['analysis'], confidence)
            
            if "long" in decision.lower():
                pos, emoji = "LONG", "📈"
            elif "short" in decision.lower():
                pos, emoji = "SHORT", "📉"
            else:
                pos, emoji = "FLAT", "⏸"
            
            print(f"\n    {emoji} {pos} | Conf: {confidence:.3f}")
            
            if memory:
                memory.add({
                    "timestamp": ts,
                    "symbol": symbol,
                    "decision": pos,
                    "confidence": confidence,
                    "refinements": refinements,
                })
                print("    ✓ Stored")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
        
        print(f"\nSleeping {delay}s...")
        time.sleep(delay)


def run_convnext_analysis(
    symbol: str = "ES=F",
    interval: str = "5m",
    combine_with_llm: bool = True,
    device: str = "auto",
) -> dict:
    """Run ConvNeXt-enhanced GAF analysis."""
    if not _CONVNEXT_AVAILABLE:
        raise ImportError("ConvNeXt requires PyTorch. Install with: pip install torch torchvision")
    
    from brainnet.agents import ConvNeXtPredictor
    
    print(f"\n{'='*60}")
    print(f"  BRAINNET/BANIS ConvNeXt ANALYSIS: {symbol}")
    print(f"  LLM: DeepSeek-V3 | CNN: ConvNeXt-Tiny")
    print(f"{'='*60}")
    
    research = ResearchAgent()
    convnext = ConvNeXtPredictor(device=device)
    
    print(f"\n    Model: ConvNeXt-Tiny (28M params)")
    print(f"    Device: {convnext.device}")
    
    print(f"\n[1/4] Fetching {symbol} data...")
    data = yf.download(symbol, period="1d", interval=interval, progress=False)
    
    if data.empty:
        return {"symbol": symbol, "error": "No data", "decision": None}
    
    try:
        latest_price = float(data['Close'].iloc[-1].iloc[0]) if hasattr(data['Close'].iloc[-1], 'iloc') else float(data['Close'].iloc[-1])
    except:
        latest_price = float(data['Close'].iloc[-1])
    
    print(f"    {len(data)} bars | Latest: ${latest_price:.2f}")
    
    close = data['Close'].values.flatten()[-100:]
    
    print(f"\n[2/4] Generating 3-channel GAF (224x224)...")
    gaf_rgb = research.generate_gaf_3channel(close, image_size=224)
    
    print(f"\n[3/4] Running ConvNeXt inference...")
    start = time.perf_counter()
    prediction = convnext.predict(gaf_rgb)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    print(f"    Inference time: {elapsed_ms:.1f}ms")
    print(f"\n    Predictions:")
    print(f"      Regime:     {prediction.regime:16s} ({prediction.regime_confidence:.1%})")
    print(f"      Direction:  {prediction.direction:16s} ({prediction.direction_confidence:.1%})")
    print(f"      Volatility: {prediction.volatility:16s} ({prediction.volatility_confidence:.1%})")
    
    result = {
        "symbol": symbol,
        "price": latest_price,
        "convnext": prediction.to_dict(),
        "inference_ms": elapsed_ms,
    }
    
    if combine_with_llm:
        print(f"\n[4/4] Running DeepSeek-V3 analysis for ensemble...")
        reasoning = ReasoningAgent()
        
        llm_analysis = research.research(data)
        confidence = reasoning.compute_confidence(llm_analysis['analysis'])
        
        result["llm"] = {
            "features": llm_analysis['features'],
            "confidence": confidence,
        }
        
        dir_scores = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        trend_sign = 1.0 if llm_analysis['features']['trend_score'] > 0 else -1.0
        
        total_weight = prediction.direction_confidence + confidence + 1e-8
        ensemble_score = (
            dir_scores[prediction.direction] * prediction.direction_confidence +
            trend_sign * confidence
        ) / total_weight
        
        if ensemble_score > 0.2:
            final_decision, emoji = "LONG", "📈"
        elif ensemble_score < -0.2:
            final_decision, emoji = "SHORT", "📉"
        else:
            final_decision, emoji = "FLAT", "⏸"
        
        result["ensemble_decision"] = final_decision
        result["ensemble_score"] = round(ensemble_score, 4)
        
        print(f"\n    {emoji} Ensemble Decision: {final_decision}")
    else:
        if prediction.direction == "bullish" and prediction.direction_confidence > 0.5:
            final_decision, emoji = "LONG", "📈"
        elif prediction.direction == "bearish" and prediction.direction_confidence > 0.5:
            final_decision, emoji = "SHORT", "📉"
        else:
            final_decision, emoji = "FLAT", "⏸"
        
        result["decision"] = final_decision
        print(f"\n    {emoji} Decision: {final_decision}")
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Symbol:     {symbol}")
    print(f"  Price:      ${latest_price:.2f}")
    print(f"  Regime:     {prediction.regime}")
    print(f"  Direction:  {prediction.direction}")
    print(f"{'='*60}\n")
    
    return result

