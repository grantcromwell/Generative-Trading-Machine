# Hybrid Adaptive Trading System

**A production-ready quantitative trading platform powered by Phi-3.5-Mini and ConvNeXt-Tiny**

This system is a production-ready multi-agent trading platform that combines:
- **Gramian Angular Field (GAF)** pattern recognition
- **ConvNeXt-Tiny** neural network for regime/direction/volatility prediction
- **LLM-as-a-judge** confidence calibration
- **pgvector** for high-performance vector memory
- **Interactive TUI** with multi-timeframe analysis

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Single LLM Architecture** | All agents use Phi-3.5-Mini (3.8B) exclusively—no model switching |
| **GAF Pattern Recognition** | Gramian Angular Field transforms time series into analyzable patterns |
| **ConvNeXt-Tiny Vision** | Neural pattern recognition with 3-channel GAF (GASF/GADF/heatmap) |
| **Dual Mode Analysis** | Vision mode (with images) or Text mode (numerical features) |
| **BSC Confidence Model** | Binary Symmetric Channel calibration for mathematically grounded confidence |
| **Adaptive Refinement** | Confidence < 0.78 triggers memory-enriched re-analysis loops |
| **Persistent Memory** | Mem0 + pgvector for high-performance vector similarity search |
| **Hot-Reload Strategies** | Generate and deploy trading strategies at runtime |
| **Multi-Backend Support** | Works with LM Studio, Ollama, Azure ML, or Hugging Face API |
| **LangGraph Orchestration** | Stateful multi-agent workflows with conditional routing |
| **31K Context Window** | Full utilization of Phi-3.5's extended context |

---

## Architecture

```
hybrid-trading-system/
├── lms/                      # Language Model Systems (monorepo)
│   ├── banis/                # Base implementation
│   │   ├── src/brainnet/
│   │   │   ├── agents/        # Research, Reasoning, Coding, ConvNeXt agents
│   │   │   ├── core/          # Config, Memory, Knowledge base
│   │   │   ├── orchestrator/  # LangGraph workflow orchestration
│   │   │   ├── services/      # CLI, TUI, Web GUI, Market data
│   │   │   └── training/     # Model training utilities
│   │   ├── examples/         # Usage examples
│   │   ├── tests/             # Unit and integration tests
│   │   └── pyproject.toml
│   ├── carter/                # Phi-3.5-Vision via NVIDIA NIM (ETHUSDT-PERP)
│   │   ├── src/brainnet/      # Same structure as banis
│   │   ├── examples/
│   │   ├── notebooks/         # Training notebooks
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── minah/                 # Phi-3.5-Mini local inference (ES=F futures)
│       ├── src/brainnet/      # Same structure as banis
│       ├── examples/
│       ├── notebooks/
│       ├── tests/
│       └── pyproject.toml
├── docs/                      # Documentation
│   └── data-flywheel/         # NVIDIA NeMo Data Flywheel blueprint
├── scripts/                    # Utility scripts
│   ├── brain                  # Brainnet CLI wrapper
│   └── carter                 # Carter CLI wrapper
├── shared/                     # Shared utilities
├── models/                     # Model artifacts storage
├── pyproject.toml             # Workspace root configuration
├── README.md                   # This file
└── .env.example                # Environment template
```

---

## Installation

### Prerequisites

- Python 3.10+
- One of the following LLM backends:
  - **LM Studio** (recommended) - Local inference with OpenAI-compatible API
  - **Ollama** - Local model serving
  - **Azure ML** - Cloud deployment
  - **Hugging Face** - Inference API

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd hybrid-trading-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### pgvector Setup (Required for Memory)

The system uses PostgreSQL with pgvector for vector similarity search:

```bash
# Option 1: Docker (Recommended)
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=trading_system \
  -e POSTGRES_DB=trading_system \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Option 2: Manual installation
# Install PostgreSQL 14+, then:
CREATE EXTENSION vector;

# Verify installation
psql -U postgres -d trading_system -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

Configure in `.env`:
```bash
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=trading_system
PGVECTOR_DB=trading_system
```

### LM Studio Setup (Recommended)

1. Download [LM Studio](https://lmstudio.ai/)
2. Download `phi-3.5-mini-instruct` model
3. Start the local server (default: `http://localhost:1234`)
4. Configure `.env`:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:1234
OLLAMA_MODEL=phi-3.5-mini-instruct
LLM_BACKEND=local
AGENT_MAX_TOKENS=31000
```

### For Vision Mode (Optional)

Load `phi-3.5-vision-instruct` in LM Studio and add:

```bash
USE_VISION=true
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `local` | Backend type: `local`, `azure`, `hf` |
| `OLLAMA_BASE_URL` | `http://localhost:1234` | LM Studio/Ollama server URL |
| `OLLAMA_MODEL` | `phi-3.5-mini-instruct` | Model identifier |
| `LLM_API_KEY` | - | API key (for Azure/HF) |
| `AGENT_MAX_TOKENS` | `31000` | Maximum tokens per request |
| `AGENT_TEMPERATURE` | `0.1` | Sampling temperature |
| `USE_VISION` | `false` | Enable vision mode for GAF images |
| `MEMORY_DB` | `pgvector` | Memory backend: `pgvector`, `sqlite` |
| `PGVECTOR_HOST` | `localhost` | PostgreSQL host |
| `PGVECTOR_PORT` | `5432` | PostgreSQL port |
| `PGVECTOR_USER` | `postgres` | PostgreSQL username |
| `PGVECTOR_PASSWORD` | - | PostgreSQL password |
| `PGVECTOR_DB` | `trading_system` | PostgreSQL database name |
| `CONFIDENCE_THRESHOLD` | `0.78` | Minimum confidence for trading |
| `MAX_REFINEMENTS` | `3` | Maximum refinement iterations |

### Example `.env`

```bash
# LM Studio Configuration
OLLAMA_BASE_URL=http://127.0.0.1:1234
OLLAMA_MODEL=phi-3.5-mini-instruct
LLM_BACKEND=local

# Model Settings
AGENT_MAX_TOKENS=31000
AGENT_TEMPERATURE=0.1
USE_VISION=false

# Memory (pgvector - PostgreSQL with vector extension)
MEMORY_DB=pgvector
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password
PGVECTOR_DB=trading_system

# Trading
CONFIDENCE_THRESHOLD=0.78
MAX_REFINEMENTS=3
```

---

## Quick Start

### Single Analysis

```python
from brainnet.agents import ResearchAgent, ReasoningAgent
import yfinance as yf

# Fetch market data
data = yf.download("ES=F", period="1d", interval="5m")

# Initialize agents
research = ResearchAgent()
reasoning = ReasoningAgent()

# Run GAF analysis
result = research.research(data)
print(f"Latest Price: {result['latest_price']:.2f}")
print(f"GAF Features: {result['features']}")
print(f"Analysis: {result['analysis'][:500]}...")

# Compute confidence and decide
confidence = reasoning.compute_confidence(result['analysis'])
decision = reasoning.decide(result['analysis'], confidence)

print(f"Confidence: {confidence:.3f}")
print(f"Decision: {decision}")
```

### Using the Orchestrator

```python
from brainnet.orchestrator import Router
import yfinance as yf

# Initialize router (handles full workflow)
router = Router()

# Fetch data
data = yf.download("ES=F", period="1d", interval="5m")

# Trigger adaptive analysis
result = router.trigger(data, symbol="ES=F")

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Refinements: {result['refinements']}")
```

### Interactive TUI (Recommended)

Launch the interactive terminal interface:

```bash
python -m brainnet.services.main gui
```

**Features:**
- **Paginated Markets**: Crypto, Futures, ETFs, Commodities (use arrow keys)
- **Multi-Timeframe**: 1m, 5m, 15m, 1h, 4h, 1d intervals
- **Score Display**: 0-100 score (0=short, 100=long) with visual bar
- **Detailed Stats**: Regime, volatility, GAF features, confidence

```
  ╔══════════════════════════════════════════════════════╗
  ║            HYBRID ADAPTIVE TRADING SYSTEM            ║
  ╚══════════════════════════════════════════════════════╝

     CRYPTO                 Page 1/4
  ─────────────────────────────────────────────────────

     [1]  BTC   -  Bitcoin
     [2]  ETH   -  Ethereum
     [3]  SOL   -  Solana
     [4]  XRP   -  Ripple

  ─────────────────────────────────────────────────────
     Use arrow keys to navigate pages

  > Select: 
```

### Command Line

```bash
# Launch interactive TUI
python -m brainnet.services.main gui

# Single analysis
python examples/gaf_trading_loop.py --symbol ES=F --single

# Continuous trading loop (every 5 minutes)
python examples/gaf_trading_loop.py --symbol ES=F --interval 5m --delay 300

# Start WebSocket server
python -m brainnet.services.main server --port 8000

# ConvNeXt-enhanced analysis (requires torch)
python -m brainnet.services.main convnext --symbol ES=F --interval 5m

# ConvNeXt-only mode (skip LLM)
python -m brainnet.services.main convnext --symbol BTC-USD --no-llm --device cuda

# ConvNeXt continuous loop
python -m brainnet.services.main convnext-loop --symbol ETH-USD --delay 300

# Show ConvNeXt model info
python -m brainnet.services.main convnext-info
```

---

## Core Components

### 1. Phi35MiniClient (`agents/base.py`)

Unified LLM client supporting multiple backends:

```python
from brainnet.agents.base import Phi35MiniClient

client = Phi35MiniClient()
response = client.generate([
    {"role": "user", "content": "Analyze this market pattern..."}
])
```

**Features:**
- Auto-detects LM Studio, Ollama, Azure, or HF backend
- Supports 31K token context window
- Vision mode for multimodal analysis
- OpenAI-compatible API

### 2. ResearchAgent (`agents/research.py`)

Generates and analyzes Gramian Angular Field patterns:

```python
from brainnet.agents import ResearchAgent

agent = ResearchAgent(use_vision=False)  # Text mode
result = agent.research(data)

# Returns:
# - analysis: LLM pattern analysis
# - image: Base64 GAF image
# - features: Numerical GAF features
# - scores: Extracted pattern scores (trend/cycle/burst)
```

**GAF Features Extracted:**
| Feature | Description |
|---------|-------------|
| `diagonal_mean/std` | Temporal autocorrelation |
| `trend_score` | Positive = uptrend, Negative = downtrend |
| `symmetry` | Low = cyclical, High = random |
| `momentum` | Recent vs historical strength |
| `recent_strength` | Pattern strength in recent data |

### 3. ReasoningAgent (`agents/reasoning.py`)

Calibrates confidence using Binary Symmetric Channel model:

```python
from brainnet.agents import ReasoningAgent

agent = ReasoningAgent(confidence_threshold=0.78)
confidence = agent.compute_confidence(analysis_text)
decision = agent.decide(analysis_text, confidence)
```

**Confidence Model:**
```
Confidence = 1 - H(ε)
```
Where `H(ε)` is binary entropy of error probability `ε`.

### 4. CodingAgent (`agents/coding.py`)

Generates and hot-reloads trading strategies:

```python
from brainnet.agents import CodingAgent

agent = CodingAgent()

# Generate strategy from natural language
code = agent.generate_strategy_code("RSI oversold with MACD bullish crossover")

# Hot-reload into runtime
strategy = agent.hot_reload_strategy(code, "rsi_macd")

# Execute
decision = agent.execute_strategy("rsi_macd", data)
```

### 5. ConvNeXtPredictor (`agents/convnext_predictor.py`)

Neural pattern recognition using ConvNeXt-Tiny:

```python
from brainnet.agents import ConvNeXtPredictor, ResearchAgent

# Initialize
research = ResearchAgent()
convnext = ConvNeXtPredictor(device="auto")  # auto-detects GPU

# Generate 3-channel GAF (224x224 RGB)
gaf_rgb = research.generate_gaf_3channel(price_series)

# Get predictions
prediction = convnext.predict(gaf_rgb)
print(f"Regime: {prediction.regime} ({prediction.regime_confidence:.1%})")
print(f"Direction: {prediction.direction} ({prediction.direction_confidence:.1%})")
print(f"Volatility: {prediction.volatility}")
```

**Output Classes:**
| Category | Classes |
|----------|---------|
| Regime | trending, mean_reverting, volatile, quiet |
| Direction | bullish, bearish, neutral |
| Volatility | calm, normal, elevated, explosive |

**3-Channel GAF:**
- Red: GASF (Gramian Angular Summation Field) - captures trend
- Green: GADF (Gramian Angular Difference Field) - captures cycles  
- Blue: Normalized price heatmap - captures correlations

### 6. MemoryManager (`core/memory.py`)

Persistent cross-session memory with Mem0 + pgvector:

**Why pgvector?**
- High-performance vector similarity search with HNSW indexes
- ACID compliance for trade data integrity
- Native PostgreSQL integration
- Scales to millions of vectors

```python
from brainnet.core import MemoryManager, load_config

memory = MemoryManager(load_config())

# Store trade outcome
memory.add({
    "decision": "long",
    "confidence": 0.85,
    "outcome": "win",
    "pnl": 150.0
})

# Retrieve context for refinement
context = memory.get_context("recent ES=F trades")
```

### 7. LangGraph Orchestrator (`orchestrator/graph.py`)

Stateful workflow with adaptive loops:

```python
from brainnet.orchestrator import build_graph, create_initial_state

graph = build_graph()
state = create_initial_state(data, symbol="ES=F")
result = graph.invoke(state)
```

**Workflow:**
```
research → reasoning → [confidence < 0.78?] → refine → research (loop)
                              ↓ (≥ 0.78)
                          finalize → decision
```

---

## How It Works

### 1. GAF Pattern Recognition

Time series data is transformed into a **Gramian Angular Field** matrix:

1. Normalize prices to [-1, 1]
2. Convert to polar coordinates (angle representation)
3. Compute angular summation/difference matrices
4. Extract statistical features or generate image

**Why GAF?**
- Preserves temporal dependencies
- Reveals hidden cyclical patterns
- Works with any time series length
- Enables both visual and numerical analysis

### 2. Dual-Mode Analysis

**Text Mode** (default):
- Extracts 13 numerical features from GAF matrix
- Sends features to Phi-3.5-Mini for interpretation
- Works with any text-only model

**Vision Mode** (requires vision model):
- Generates GAF image (PNG)
- Sends base64-encoded image to Phi-3.5-Vision
- Direct visual pattern recognition

### 3. BSC Confidence Calibration

The **Binary Symmetric Channel** model treats LLM judgment as a noisy channel:

```
P(correct) = 1 - ε
Channel Capacity = 1 - H(ε)
```

Where:
- `ε` = error probability (estimated by LLM self-evaluation)
- `H(ε)` = binary entropy = -ε·log₂(ε) - (1-ε)·log₂(1-ε)

This provides mathematically grounded confidence intervals rather than arbitrary scores.

### 4. Adaptive Refinement Loop

When confidence < 0.78:
1. Query Mem0 for relevant historical context
2. Enrich analysis prompt with memory
3. Re-run research with additional context
4. Recompute confidence
5. Repeat up to `MAX_REFINEMENTS` times

### 5. Memory-Augmented Learning

Mem0 stores:
- Trade outcomes (win/loss/flat)
- Pattern signatures from GAF analysis
- Strategy performance metrics
- Market regime observations

This enables the system to learn from past decisions and improve over time.

---

## API Reference

### Agents

| Class | Method | Description |
|-------|--------|-------------|
| `Phi35MiniClient` | `generate(messages, is_vision, max_tokens)` | Generate LLM response |
| `Phi35MiniClient` | `generate_with_image(prompt, image_base64)` | Vision analysis |
| `ResearchAgent` | `research(data, memory_context)` | Full GAF analysis |
| `ResearchAgent` | `generate_gaf_image(series)` | Create GAF image |
| `ResearchAgent` | `extract_gaf_features(series)` | Extract numerical features |
| `ReasoningAgent` | `compute_confidence(analysis)` | BSC confidence calibration |
| `ReasoningAgent` | `decide(analysis, confidence)` | Trading decision |
| `CodingAgent` | `generate_strategy_code(requirements)` | Generate strategy |
| `CodingAgent` | `hot_reload_strategy(code, name)` | Load strategy at runtime |
| `ConvNeXtPredictor` | `predict(gaf_image, return_features)` | Run ConvNeXt inference |
| `ConvNeXtPredictor` | `predict_batch(gaf_images)` | Batch inference |
| `ConvNeXtPredictor` | `extract_features(gaf_image)` | Get 768-dim features |
| `ResearchAgent` | `generate_gaf_3channel(series, image_size)` | Create RGB GAF image |

### Core

| Class | Method | Description |
|-------|--------|-------------|
| `MemoryManager` | `add(data)` | Store memory |
| `MemoryManager` | `search(query)` | Search memories |
| `MemoryManager` | `get_context(query)` | Get context string |
| `KnowledgeBase` | `add_pattern(name, desc, success_rate)` | Store pattern |
| `KnowledgeBase` | `log_trade_outcome(...)` | Log trade |

### Orchestrator

| Function/Class | Description |
|----------------|-------------|
| `build_graph()` | Create LangGraph workflow |
| `create_initial_state(data, symbol)` | Initialize state |
| `Router.trigger(data, symbol)` | Run full workflow |
| `Router.adapt(feedback)` | Trigger strategy adaptation |

---

## Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

### Custom Agent

```python
from brainnet.agents.base import BaseAgent

class SentimentAgent(BaseAgent):
    def analyze(self, headlines: list[str]) -> dict:
        prompt = f"Analyze sentiment: {headlines}"
        response = self.llm.generate([{"role": "user", "content": prompt}])
        return {"sentiment": response}
```

### Full Trading Loop

```bash
python examples/gaf_trading_loop.py --symbol ES=F --interval 5m
```

Output:
```
==================================================
Iteration 1 | 2024-01-15 09:30:00
==================================================

[1/5] Fetching ES=F 5m data...
    275 bars, latest: 6829.75

[2/5] Getting memory context...

[3/5] GAF analysis...
    TREND: 4/10, CYCLE: 2/10, BURST: 3/10

[4/5] Computing confidence...
    Confidence: 0.520

    > Refinement 1/3
    New confidence: 0.680

    > Refinement 2/3
    New confidence: 0.810

[5/5] Decision...

    LONG | Conf: 0.81
    Stored in memory
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage
pytest tests/ --cov=brainnet --cov-report=html
```

---

## Troubleshooting

### Connection Refused

```
openai.APIConnectionError: Connection error.
```

**Solution:** Ensure LM Studio/Ollama is running:
```bash
# Check if server is running
curl http://localhost:1234/v1/models
```

### Model Does Not Support Images

```
openai.BadRequestError: Model does not support images
```

**Solution:** Either:
1. Load `phi-3.5-vision-instruct` in LM Studio, OR
2. Use text mode (default): `USE_VISION=false`

### Module Not Found

```
ModuleNotFoundError: No module named 'brainnet'
```

**Solution:** Install in development mode:
```bash
pip install -e .
# or with uv:
uv pip install -e .
```

### Low Confidence Scores

If confidence is consistently low:
1. Check data quality (enough bars?)
2. Increase `MAX_REFINEMENTS`
3. Lower `CONFIDENCE_THRESHOLD` temporarily
4. Review GAF features for data issues

---

## Performance Optimization

1. **Local Inference**: LM Studio/Ollama provides 50-200ms latency compared to 500-2000ms for cloud APIs
2. **Batch Processing**: Process multiple symbols in parallel for improved throughput
3. **Memory Management**: Periodically clean old memories to maintain query performance
4. **Feature Caching**: Cache GAF features for repeated analysis to reduce computation overhead

---

## Roadmap

- [x] ConvNeXt-Tiny neural pattern recognition
- [x] 3-channel GAF (GASF/GADF/heatmap) generation
- [x] CNN + LLM ensemble predictions
- [ ] Multi-timeframe GAF fusion
- [ ] ConvNeXt fine-tuning pipeline
- [ ] LSTM head for multi-step forecasting
- [ ] Real-time WebSocket streaming
- [ ] Backtesting integration
- [ ] Portfolio-level risk management

---

## License

MIT

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing`)
7. Open a Pull Request

---

## Acknowledgments

- **Microsoft Phi-3.5** - Core language model for pattern analysis and decision-making
- **LangGraph** - Multi-agent workflow orchestration framework
- **Mem0** - Persistent memory layer for context retention
- **pyts** - Time series transformation library for GAF generation
- **yfinance** - Market data retrieval and processing
