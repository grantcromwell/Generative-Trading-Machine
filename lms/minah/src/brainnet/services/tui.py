"""
Brainnet Terminal UI (TUI) - Blue-themed terminal interface

Features:
1. Market selection with pagination
2. Timeframe selection (1m, 5m, 15m, 1h, 4h, 1d)
3. Quick Score View: Score 0-100 (0=short, 100=long)
4. Detailed Stats: Press 2 for detailed analysis, 0 to return to menu

Performance optimizations:
- Cached model instances (ConvNeXt, ResearchAgent)
- Fast mode skips LLM calls for quick scoring
- Uses numpy vectorization for GAF features
"""

import sys
import os
import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# ============================================================================
# MODEL CACHE - Load once, use many times
# ============================================================================
_MODEL_CACHE = {
    "research_agent": None,
    "convnext": None,
    "reasoning_agent": None,
    "initialized": False,
}


def _get_cached_models(skip_llm: bool = True):
    """
    Get cached model instances for fast inference.
    
    First call loads models (~3-5s), subsequent calls are instant.
    """
    global _MODEL_CACHE
    
    if not _MODEL_CACHE["initialized"]:
        try:
            from brainnet.agents import ResearchAgent, _CONVNEXT_AVAILABLE
            
            # Cache ResearchAgent (lightweight)
            _MODEL_CACHE["research_agent"] = ResearchAgent()
            
            # Cache ConvNeXt (heavy - 28M params)
            if _CONVNEXT_AVAILABLE:
                from brainnet.agents import ConvNeXtPredictor
                _MODEL_CACHE["convnext"] = ConvNeXtPredictor(device="auto")
            
            # Only load ReasoningAgent if needed (requires LLM)
            if not skip_llm:
                from brainnet.agents import ReasoningAgent
                _MODEL_CACHE["reasoning_agent"] = ReasoningAgent()
            
            _MODEL_CACHE["initialized"] = True
        except Exception as e:
            print(f"Model cache init failed: {e}")
    
    return _MODEL_CACHE


# ANSI color codes for BLUE theme
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    
    # Blues
    BG_DARK = "\033[48;2;10;22;40m"
    BG_MEDIUM = "\033[48;2;26;39;68m"
    BG_HIGHLIGHT = "\033[48;2;42;63;95m"
    
    # Text colors
    CYAN = "\033[38;2;0;212;255m"
    WHITE = "\033[38;2;255;255;255m"
    GRAY = "\033[38;2;136;153;170m"
    RED = "\033[38;2;255;68;102m"
    GREEN = "\033[38;2;0;255;136m"
    YELLOW = "\033[38;2;255;212;0m"
    BLUE = "\033[38;2;100;149;237m"
    ORANGE = "\033[38;2;255;165;0m"
    MAGENTA = "\033[38;2;255;100;255m"


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    symbol: str
    interval: str
    score: int  # 0-100 (0=strong short, 50=neutral, 100=strong long)
    price: float
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    regime: str
    volatility: str
    trend_score: float
    momentum: float
    features: Dict[str, float]
    analysis_type: str  # "convnext", "llm", "ensemble"
    bars_analyzed: int


# ============================================================================
# MARKET & INTERVAL CONFIGURATION
# ============================================================================

# All available markets (paginated)
ALL_MARKETS = [
    # Page 1: Crypto
    ("BTC", "BTC-USD", "Bitcoin"),
    ("ETH", "ETH-USD", "Ethereum"),
    ("SOL", "SOL-USD", "Solana"),
    ("XRP", "XRP-USD", "Ripple"),
    # Page 2: Index Futures
    ("ES", "ES=F", "S&P 500 Futures"),
    ("NQ", "NQ=F", "Nasdaq Futures"),
    ("YM", "YM=F", "Dow Futures"),
    ("RTY", "RTY=F", "Russell 2000 Futures"),
    # Page 3: ETFs
    ("SPY", "SPY", "S&P 500 ETF"),
    ("QQQ", "QQQ", "Nasdaq 100 ETF"),
    ("IWM", "IWM", "Russell 2000 ETF"),
    ("DIA", "DIA", "Dow Jones ETF"),
    # Page 4: Commodities
    ("GC", "GC=F", "Gold Futures"),
    ("SI", "SI=F", "Silver Futures"),
    ("CL", "CL=F", "Crude Oil Futures"),
    ("NG", "NG=F", "Natural Gas Futures"),
]

MARKETS_PER_PAGE = 4

# Timeframe options with yfinance parameters
INTERVALS = [
    ("1", "1m", "1 Minute", "1d", 60),      # (key, interval, label, period, lookback)
    ("2", "5m", "5 Minutes", "1d", 100),
    ("3", "15m", "15 Minutes", "5d", 100),
    ("4", "1h", "1 Hour", "1mo", 100),
    ("5", "4h", "4 Hours", "3mo", 100),
    ("6", "1d", "1 Day", "1y", 100),
]


# ============================================================================
# KEYBOARD INPUT HANDLING
# ============================================================================

def get_key():
    """
    Get a single keypress including arrow keys.
    
    Returns:
        str: The key pressed ('LEFT', 'RIGHT', 'UP', 'DOWN', or the character)
    """
    if os.name == 'nt':
        # Windows
        import msvcrt
        key = msvcrt.getch()
        if key == b'\xe0':  # Arrow key prefix on Windows
            key = msvcrt.getch()
            if key == b'K':
                return 'LEFT'
            elif key == b'M':
                return 'RIGHT'
            elif key == b'H':
                return 'UP'
            elif key == b'P':
                return 'DOWN'
        return key.decode('utf-8', errors='ignore')
    else:
        # Unix/Mac
        import sys
        import tty
        import termios
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            
            if ch == '\x1b':  # Escape sequence
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'D':
                        return 'LEFT'
                    elif ch3 == 'C':
                        return 'RIGHT'
                    elif ch3 == 'A':
                        return 'UP'
                    elif ch3 == 'B':
                        return 'DOWN'
                return 'ESC'
            elif ch == '\r' or ch == '\n':
                return 'ENTER'
            elif ch == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            else:
                return ch.upper()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def get_input_with_arrows(valid_choices: list, allow_arrows: bool = True) -> str:
    """
    Get user input supporting both typed input and arrow keys.
    
    Args:
        valid_choices: List of valid single-key choices
        allow_arrows: Whether to return arrow key presses
    
    Returns:
        The key pressed or typed choice
    """
    buffer = ""
    
    while True:
        key = get_key()
        
        # Arrow keys
        if key in ['LEFT', 'RIGHT', 'UP', 'DOWN'] and allow_arrows:
            print()  # New line after prompt
            return key
        
        # Enter submits buffer
        if key == 'ENTER':
            print()  # New line after prompt
            return buffer.upper() if buffer else ""
        
        # Q to quit
        if key == 'Q':
            print(key)
            return 'Q'
        
        # Number keys (direct selection)
        if key in valid_choices:
            print(key)
            return key
        
        # Escape
        if key == 'ESC':
            print()
            return 'ESC'
        
        # Build buffer for multi-char input
        if key.isalnum():
            buffer += key
            print(key, end="", flush=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_score_color(score: int) -> str:
    """Get color based on score value."""
    c = Colors
    if score >= 70:
        return c.GREEN
    elif score >= 55:
        return c.CYAN
    elif score >= 45:
        return c.YELLOW
    elif score >= 30:
        return c.ORANGE
    else:
        return c.RED


def get_score_label(score: int) -> str:
    """Get human-readable label for score."""
    if score >= 80:
        return "STRONG LONG"
    elif score >= 65:
        return "LONG"
    elif score >= 55:
        return "LEAN LONG"
    elif score >= 45:
        return "NEUTRAL"
    elif score >= 35:
        return "LEAN SHORT"
    elif score >= 20:
        return "SHORT"
    else:
        return "STRONG SHORT"


def draw_score_bar(score: int, width: int = 40) -> str:
    """Draw a visual score bar."""
    c = Colors
    filled = int(score / 100 * width)
    
    bar = ""
    for i in range(width):
        if i < filled:
            if i < width * 0.3:
                bar += f"{c.RED}█{c.RESET}"
            elif i < width * 0.45:
                bar += f"{c.ORANGE}█{c.RESET}"
            elif i < width * 0.55:
                bar += f"{c.YELLOW}█{c.RESET}"
            elif i < width * 0.7:
                bar += f"{c.CYAN}█{c.RESET}"
            else:
                bar += f"{c.GREEN}█{c.RESET}"
        else:
            bar += f"{c.GRAY}░{c.RESET}"
    
    return bar


def draw_header():
    """Draw the main header."""
    c = Colors
    clear_screen()
    print()
    print(f"{c.CYAN}{c.BOLD}  ╔══════════════════════════════════════════════════════╗{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ║                    🧠 BRAINNET                        ║{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ║         Adaptive Quant Trading System                 ║{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ╚══════════════════════════════════════════════════════╝{c.RESET}")
    print()


# ============================================================================
# BACKEND ANALYSIS
# ============================================================================

def run_analysis(symbol: str, name: str, interval: str, period: str, 
                 lookback: int, fast_mode: bool = True) -> AnalysisResult:
    """
    Run market analysis and return score.
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        name: Display name (e.g., "Bitcoin")
        interval: yfinance interval (e.g., "5m", "1h", "1d")
        period: yfinance period (e.g., "1d", "1mo", "1y")
        lookback: Number of bars for GAF analysis
        fast_mode: Skip LLM calls for faster scoring (default True)
    
    Returns:
        AnalysisResult with score 0-100
    """
    c = Colors
    start_time = time.time()
    
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print(f"{c.WHITE}{c.BOLD}     Analyzing {name} ({symbol}) @ {interval}{c.RESET}")
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Get cached models
    cache = _get_cached_models(skip_llm=fast_mode)
    research = cache.get("research_agent")
    convnext = cache.get("convnext")
    
    # Import yfinance
    try:
        import yfinance as yf
        from brainnet.agents import _CONVNEXT_AVAILABLE
    except ImportError as e:
        print(f"{c.RED}  ✗ Import error: {e}{c.RESET}")
        return AnalysisResult(
            symbol=symbol, interval=interval, score=50, price=0.0, 
            direction="NEUTRAL", confidence=0.0, regime="unknown", 
            volatility="unknown", trend_score=0.0, momentum=0.0, 
            features={}, analysis_type="error", bars_analyzed=0
        )
    
    # Re-initialize research agent if not cached
    if research is None:
        from brainnet.agents import ResearchAgent
        research = ResearchAgent()
    
    # Step 1: Fetch data with specified interval
    step_start = time.time()
    print(f"{c.GRAY}     [1/3] Fetching {interval} data ({period})...{c.RESET}", end="", flush=True)
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError("No data received")
        
        # Get latest price
        try:
            price = float(data['Close'].iloc[-1].iloc[0]) if hasattr(data['Close'].iloc[-1], 'iloc') else float(data['Close'].iloc[-1])
        except:
            price = float(data['Close'].iloc[-1])
        
        bars_count = len(data)
        elapsed = time.time() - step_start
        print(f"\r{c.GREEN}     ✓ {bars_count} bars | ${price:,.2f} ({elapsed:.1f}s){c.RESET}")
    except Exception as e:
        print(f"\r{c.RED}     ✗ Data fetch failed: {e}{c.RESET}")
        return AnalysisResult(
            symbol=symbol, interval=interval, score=50, price=0.0, 
            direction="NEUTRAL", confidence=0.0, regime="unknown", 
            volatility="unknown", trend_score=0.0, momentum=0.0, 
            features={}, analysis_type="error", bars_analyzed=0
        )
    
    # Step 2: GAF features
    step_start = time.time()
    print(f"{c.GRAY}     [2/3] Extracting GAF features...{c.RESET}", end="", flush=True)
    try:
        # Use lookback parameter for GAF
        close = data['Close'].values.flatten()[-lookback:]
        features = research.extract_gaf_features(close)
        
        trend = features.get('trend_score', 0)
        momentum = features.get('momentum', 0)
        
        elapsed = time.time() - step_start
        print(f"\r{c.GREEN}     ✓ Trend: {trend:+.3f} | Mom: {momentum:+.3f} ({elapsed:.1f}s){c.RESET}")
    except Exception as e:
        print(f"\r{c.YELLOW}     ⚠ GAF warning: {e}{c.RESET}")
        features = {}
        trend = 0
        momentum = 0
        close = []
    
    # Step 3: ConvNeXt prediction
    regime = "unknown"
    volatility = "unknown"
    convnext_direction = "neutral"
    convnext_confidence = 0.0
    analysis_type = "gaf_features"
    
    if convnext is not None and len(close) >= 10:
        step_start = time.time()
        print(f"{c.GRAY}     [3/3] ConvNeXt inference...{c.RESET}", end="", flush=True)
        try:
            gaf_rgb = research.generate_gaf_3channel(close)
            prediction = convnext.predict(gaf_rgb)
            
            regime = prediction.regime
            volatility = prediction.volatility
            convnext_direction = prediction.direction
            convnext_confidence = prediction.direction_confidence
            analysis_type = "convnext"
            
            elapsed = time.time() - step_start
            print(f"\r{c.GREEN}     ✓ {convnext_direction.upper()} ({convnext_confidence:.0%}) | {regime} ({elapsed:.1f}s){c.RESET}")
        except Exception as e:
            print(f"\r{c.YELLOW}     ⚠ ConvNeXt: {e}{c.RESET}")
    else:
        print(f"{c.GRAY}     [3/3] ConvNeXt skipped (insufficient data){c.RESET}")
    
    # Calculate confidence
    if fast_mode:
        signal_strength = abs(trend) + abs(momentum) + convnext_confidence
        confidence = min(0.95, 0.4 + signal_strength * 0.3)
    else:
        try:
            reasoning = cache.get("reasoning_agent")
            if reasoning is None:
                from brainnet.agents import ReasoningAgent
                reasoning = ReasoningAgent()
            analysis = research.research(data)
            confidence = reasoning.compute_confidence(analysis.get('analysis', ''))
        except:
            confidence = 0.5
    
    # Calculate final score (0-100)
    base_score = 50 + (trend * 20)
    momentum_adj = momentum * 10
    direction_adj = 0
    if convnext_direction == "bullish":
        direction_adj = 15 * convnext_confidence
    elif convnext_direction == "bearish":
        direction_adj = -15 * convnext_confidence
    
    score = base_score + momentum_adj + direction_adj
    score = max(0, min(100, int(score)))
    
    if score >= 55:
        direction = "LONG"
    elif score <= 45:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    
    total_time = time.time() - start_time
    print(f"\n{c.GRAY}     Total: {total_time:.1f}s{c.RESET}")
    print()
    
    return AnalysisResult(
        symbol=symbol,
        interval=interval,
        score=score,
        price=price,
        direction=direction,
        confidence=confidence,
        regime=regime,
        volatility=volatility,
        trend_score=trend,
        momentum=momentum,
        features=features,
        analysis_type=analysis_type,
        bars_analyzed=len(close),
    )


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_score(result: AnalysisResult) -> str:
    """Display the score screen."""
    c = Colors
    clear_screen()
    draw_header()
    
    score_color = get_score_color(result.score)
    label = get_score_label(result.score)
    
    # Symbol, price, and interval
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print(f"{c.WHITE}{c.BOLD}     {result.symbol}{c.RESET}  {c.GRAY}│{c.RESET}  {c.WHITE}${result.price:,.2f}{c.RESET}  {c.GRAY}│{c.RESET}  {c.MAGENTA}{result.interval}{c.RESET}")
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Large score display
    print(f"{c.GRAY}     SHORT ◀────────────────────────────────────▶ LONG{c.RESET}")
    print(f"     {draw_score_bar(result.score, 50)}")
    print()
    
    # Score number
    print(f"     {score_color}{c.BOLD}")
    print(f"                    ┌─────────────┐")
    print(f"                    │    {result.score:3d}      │")
    print(f"                    └─────────────┘")
    print(f"{c.RESET}")
    
    # Label
    print(f"               {score_color}{c.BOLD}{label:^20}{c.RESET}")
    print()
    
    # Quick stats
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    
    if result.direction == "LONG":
        dir_indicator = f"{c.GREEN}▲ LONG{c.RESET}"
    elif result.direction == "SHORT":
        dir_indicator = f"{c.RED}▼ SHORT{c.RESET}"
    else:
        dir_indicator = f"{c.YELLOW}◆ NEUTRAL{c.RESET}"
    
    print(f"     {c.GRAY}Signal:{c.RESET}     {dir_indicator}     {c.GRAY}│{c.RESET}  {c.GRAY}Confidence:{c.RESET} {c.WHITE}{result.confidence:.0%}{c.RESET}")
    print(f"     {c.GRAY}Bars:{c.RESET}       {c.WHITE}{result.bars_analyzed}{c.RESET}          {c.GRAY}│{c.RESET}  {c.GRAY}Regime:{c.RESET}     {c.WHITE}{result.regime}{c.RESET}")
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Navigation options
    print(f"     {c.CYAN}[2]{c.RESET}  {c.WHITE}View Detailed Statistics{c.RESET}")
    print()
    print(f"     {c.CYAN}[0]{c.RESET}  {c.GRAY}Back to Menu{c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    print(f"{c.CYAN}{c.BOLD}  ▶ Select: {c.RESET}", end="", flush=True)
    
    try:
        choice = get_input_with_arrows(['2', '0'], allow_arrows=False)
        return choice
    except (KeyboardInterrupt, EOFError):
        return "0"


def display_detailed_stats(result: AnalysisResult) -> str:
    """Display detailed statistics screen."""
    c = Colors
    clear_screen()
    draw_header()
    
    score_color = get_score_color(result.score)
    
    print(f"{c.CYAN}  ═══════════════════════════════════════════════════════{c.RESET}")
    print(f"{c.WHITE}{c.BOLD}     DETAILED ANALYSIS: {result.symbol} @ {result.interval}{c.RESET}")
    print(f"{c.CYAN}  ═══════════════════════════════════════════════════════{c.RESET}")
    print()
    
    # Price & Score Summary
    print(f"     {c.GRAY}┌─────────────────────────────────────────────────┐{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Price:{c.RESET}        {c.CYAN}${result.price:>12,.2f}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Score:{c.RESET}        {score_color}{result.score:>12}{c.RESET} / 100              {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Signal:{c.RESET}       {score_color}{result.direction:>12}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Confidence:{c.RESET}   {c.CYAN}{result.confidence:>11.1%}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Interval:{c.RESET}     {c.MAGENTA}{result.interval:>12}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Bars:{c.RESET}         {c.WHITE}{result.bars_analyzed:>12}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}└─────────────────────────────────────────────────┘{c.RESET}")
    print()
    
    # Market Regime
    print(f"     {c.WHITE}{c.BOLD}MARKET REGIME{c.RESET}")
    print(f"     {c.CYAN}──────────────────────────────────────────────{c.RESET}")
    
    regime_color = c.WHITE
    if result.regime == "trending":
        regime_color = c.GREEN
    elif result.regime == "volatile":
        regime_color = c.RED
    elif result.regime == "mean_reverting":
        regime_color = c.YELLOW
        
    vol_color = c.WHITE
    if result.volatility in ["calm", "normal"]:
        vol_color = c.GREEN
    elif result.volatility == "elevated":
        vol_color = c.YELLOW
    elif result.volatility == "explosive":
        vol_color = c.RED
    
    print(f"     Regime:      {regime_color}{result.regime:15}{c.RESET}")
    print(f"     Volatility:  {vol_color}{result.volatility:15}{c.RESET}")
    print()
    
    # GAF Features
    print(f"     {c.WHITE}{c.BOLD}GAF PATTERN ANALYSIS{c.RESET}")
    print(f"     {c.CYAN}──────────────────────────────────────────────{c.RESET}")
    
    trend = result.trend_score
    trend_bar = "█" * max(1, int(abs(trend) * 10 + 1))
    if trend > 0:
        print(f"     Trend:       {c.GREEN}{trend:+.4f}  {trend_bar}{c.RESET}")
    else:
        print(f"     Trend:       {c.RED}{trend:+.4f}  {trend_bar}{c.RESET}")
    
    mom = result.momentum
    mom_bar = "█" * max(1, int(abs(mom) * 10 + 1))
    if mom > 0:
        print(f"     Momentum:    {c.GREEN}{mom:+.4f}  {mom_bar}{c.RESET}")
    else:
        print(f"     Momentum:    {c.RED}{mom:+.4f}  {mom_bar}{c.RESET}")
    print()
    
    # Raw features
    if result.features:
        print(f"     {c.WHITE}{c.BOLD}RAW FEATURES{c.RESET}")
        print(f"     {c.CYAN}──────────────────────────────────────────────{c.RESET}")
        
        for key, value in list(result.features.items())[:8]:
            if key in ['trend_score', 'momentum', 'recent_strength']:
                val_color = c.GREEN if value > 0 else c.RED
            else:
                val_color = c.WHITE
            print(f"     {c.GRAY}{key:22}{c.RESET} {val_color}{value:+.4f}{c.RESET}")
        print()
    
    print(f"     {c.GRAY}Analysis: {result.analysis_type.upper()}{c.RESET}")
    print()
    
    # Navigation
    print(f"     {c.CYAN}[1]{c.RESET}  {c.WHITE}Back to Score View{c.RESET}")
    print()
    print(f"     {c.CYAN}[0]{c.RESET}  {c.GRAY}Back to Menu{c.RESET}")
    print()
    print(f"{c.CYAN}  ═══════════════════════════════════════════════════════{c.RESET}")
    print()
    
    print(f"{c.CYAN}{c.BOLD}  ▶ Select: {c.RESET}", end="", flush=True)
    
    try:
        choice = get_input_with_arrows(['1', '0'], allow_arrows=False)
        return choice
    except (KeyboardInterrupt, EOFError):
        return "0"


def display_interval_menu(symbol: str, name: str) -> Optional[Tuple[str, str, int]]:
    """
    Display interval selection menu.
    
    Returns:
        Tuple of (interval, period, lookback) or None to go back
    """
    c = Colors
    clear_screen()
    draw_header()
    
    print(f"{c.GRAY}     Selected: {c.WHITE}{c.BOLD}{name}{c.RESET} {c.GRAY}({symbol}){c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    print(f"{c.WHITE}{c.BOLD}     SELECT TIMEFRAME:{c.RESET}")
    print()
    
    # Interval options
    for key, interval, label, period, lookback in INTERVALS:
        # Color code by timeframe type
        if interval in ["1m", "5m"]:
            color = c.CYAN  # Scalping
        elif interval in ["15m", "1h"]:
            color = c.GREEN  # Intraday
        else:
            color = c.MAGENTA  # Swing
        
        print(f"     {c.CYAN}[{c.WHITE}{c.BOLD}{key}{c.RESET}{c.CYAN}]{c.RESET}  {color}{interval:4}{c.RESET}  {c.GRAY}─  {label}{c.RESET}")
        print()
    
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    print(f"     {c.GRAY}[0]  Back to Market Selection{c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Timeframe descriptions
    print(f"     {c.GRAY}Tip: {c.CYAN}1m-5m{c.RESET} {c.GRAY}= Scalping | {c.GREEN}15m-1h{c.RESET} {c.GRAY}= Intraday | {c.MAGENTA}4h-1d{c.RESET} {c.GRAY}= Swing{c.RESET}")
    print()
    
    print(f"{c.CYAN}{c.BOLD}  ▶ Select: {c.RESET}", end="", flush=True)
    
    try:
        valid_keys = [str(i+1) for i in range(len(INTERVALS))] + ['0']
        choice = get_input_with_arrows(valid_keys, allow_arrows=False)
    except (KeyboardInterrupt, EOFError):
        return None
    
    if choice == "0" or choice == "ESC":
        return None
    
    # Find selected interval
    for key, interval, label, period, lookback in INTERVALS:
        if choice == key:
            return (interval, period, lookback)
    
    # Invalid choice - retry
    return display_interval_menu(symbol, name)


def display_menu(page: int = 0) -> Optional[Tuple[str, str, str, str, int]]:
    """
    Display main menu with pagination and interval selection.
    
    Args:
        page: Current page number (0-indexed)
    
    Returns:
        Tuple of (symbol, name, interval, period, lookback) or None if exit
    """
    c = Colors
    clear_screen()
    draw_header()
    
    # Calculate pagination
    total_pages = (len(ALL_MARKETS) + MARKETS_PER_PAGE - 1) // MARKETS_PER_PAGE
    start_idx = page * MARKETS_PER_PAGE
    end_idx = min(start_idx + MARKETS_PER_PAGE, len(ALL_MARKETS))
    current_markets = ALL_MARKETS[start_idx:end_idx]
    
    # Page categories
    page_titles = ["🪙 CRYPTO", "📊 INDEX FUTURES", "📈 ETFs", "🛢️ COMMODITIES"]
    page_title = page_titles[page] if page < len(page_titles) else f"PAGE {page + 1}"
    
    # Arrow indicators for pagination
    left_arrow = f"{c.CYAN}◀{c.RESET}" if page > 0 else f"{c.GRAY}◁{c.RESET}"
    right_arrow = f"{c.CYAN}▶{c.RESET}" if page < total_pages - 1 else f"{c.GRAY}▷{c.RESET}"
    
    print(f"{c.GRAY}     Powered by Phi-3.5-Mini • GAF • ConvNeXt{c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print(f"     {c.WHITE}{c.BOLD}{page_title}{c.RESET}              {left_arrow} {c.GRAY}Page {page + 1}/{total_pages}{c.RESET} {right_arrow}")
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    print(f"{c.WHITE}{c.BOLD}     SELECT A MARKET TO ANALYZE:{c.RESET}")
    print()
    
    # Market options for current page
    for i, (short, symbol, name) in enumerate(current_markets):
        key = str(i + 1)
        print(f"     {c.CYAN}[{c.WHITE}{c.BOLD}{key}{c.RESET}{c.CYAN}]{c.RESET}  {c.WHITE}{c.BOLD}{short:4}{c.RESET}  {c.GRAY}─  {name}{c.RESET}")
        print()
    
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    print(f"     {c.RED}[Q]{c.RESET}  {c.RED}EXIT{c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    print(f"     {c.GRAY}Use arrow keys ← → to navigate pages{c.RESET}")
    print()
    
    # Input prompt
    print(f"{c.CYAN}{c.BOLD}  ▶ Select: {c.RESET}", end="", flush=True)
    
    try:
        # Get input with arrow key support
        valid_keys = [str(i+1) for i in range(len(current_markets))]
        choice = get_input_with_arrows(valid_keys, allow_arrows=True)
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    
    # Handle navigation
    if choice == "Q" or choice == "EXIT" or choice == "ESC":
        return None
    
    # Arrow key pagination
    if choice == "LEFT" and page > 0:
        return display_menu(page - 1)
    if choice == "RIGHT" and page < total_pages - 1:
        return display_menu(page + 1)
    
    # Also support < > for pagination
    if choice in ["<", ",", "["] and page > 0:
        return display_menu(page - 1)
    if choice in [">", ".", "]"] and page < total_pages - 1:
        return display_menu(page + 1)
    
    # Find selected market
    for i, (short, symbol, name) in enumerate(current_markets):
        if choice == str(i + 1) or choice == short:
            # Go to interval selection
            interval_result = display_interval_menu(symbol, name)
            if interval_result is None:
                return display_menu(page)  # Back to market menu
            
            interval, period, lookback = interval_result
            return (symbol, name, interval, period, lookback)
    
    # Invalid or empty choice - stay on page
    return display_menu(page)


# ============================================================================
# MAIN TUI LOOP
# ============================================================================

def launch_tui() -> Optional[str]:
    """
    Launch the terminal UI with:
    1. Market selection (with pagination)
    2. Interval selection (1m, 5m, 15m, 1h, 4h, 1d)
    3. Quick Score View
    4. Detailed Statistics
    
    Returns:
        Selected symbol string or None if exited
    """
    while True:
        # Step 1: Select market and interval from menu
        selection = display_menu()
        
        if selection is None:
            clear_screen()
            c = Colors
            print(f"\n{c.CYAN}  👋 Goodbye!{c.RESET}\n")
            return None
        
        symbol, name, interval, period, lookback = selection
        
        # Step 2: Run analysis with selected interval
        result = run_analysis(symbol, name, interval, period, lookback)
        
        # Step 3: Score/Details navigation loop
        while True:
            choice = display_score(result)
            
            if choice == "2":
                # Detailed statistics
                while True:
                    detail_choice = display_detailed_stats(result)
                    
                    if detail_choice == "1":
                        break  # Back to score view
                    elif detail_choice == "0":
                        break  # Back to menu
                    else:
                        continue
                
                if detail_choice == "0":
                    break  # Back to menu
            
            elif choice == "0":
                break  # Back to menu
            
            else:
                continue  # Invalid, show score again


def launch_tui_interactive() -> Optional[str]:
    """Launch interactive TUI."""
    try:
        return launch_tui()
    except Exception as e:
        c = Colors
        print(f"\n{c.RED}  Error: {e}{c.RESET}\n")
        return None


# Legacy compatibility
def _launch_curses_tui() -> Optional[str]:
    """Deprecated: Use launch_tui() instead."""
    return launch_tui()


if __name__ == "__main__":
    selected = launch_tui()
    if selected:
        print(f"Final selection: {selected}")
