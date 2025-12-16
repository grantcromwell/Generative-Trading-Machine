# Carter

**Phi-3.5-Vision-Instruct via NVIDIA NIM for ETHUSDT-PERP Scalping**

Carter is a fork of Brainnet/Minah optimized for live crypto scalping using Microsoft's Phi-3.5-Vision model hosted on NVIDIA's NIM API. It analyzes Gramian Angular Field (GAF) images natively—no local GPU required.

## Quick Start

```bash
# 1. Get NVIDIA API key at https://build.nvidia.com/
# 2. Set environment variable
export NVIDIA_API_KEY="nvapi-..."

# 3. Setup
cd lms/carter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Run
carter/source /Users/gdove/Desktop/Brainnet/.venv/bin/activate && cd /Users/gdove/Desktop/Brainnet/lms/carter/src && python -m brainnet.services.main info
```

## Model

| Spec | Value |
|------|-------|
| Model | `microsoft/phi-3.5-vision-instruct` |
| API | NVIDIA NIM |
| Endpoint | `https://integrate.api.nvidia.com/v1/chat/completions` |
| Vision | Native base64 PNG (inline img tags) |
| Local GPU | Not required |

## Environment Variables

```bash
# Required
NVIDIA_API_KEY=nvapi-...

# Optional
NVIDIA_MODEL=microsoft/phi-3.5-vision-instruct
NVIDIA_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
AGENT_MAX_TOKENS=1024

# Trading
TRADING_SYMBOL=ETHUSDT-PERP
CONFIDENCE_THRESHOLD=0.78

# Memory (pgvector)
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DB=brainnet
```

## Commands

```bash
carter              # Launch GUI selector
carter analyze      # Single analysis (ETHUSDT-PERP)
carter loop         # Continuous trading loop
carter info         # Show config
carter convnext     # ConvNeXt + Vision ensemble
carter server       # Start FastAPI server
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Carter                               │
├─────────────────────────────────────────────────────────────┤
│  Market Data  →  GAF Image  →  NVIDIA API  →  Signal        │
│  (ETHUSDT-PERP)   (base64 PNG)   (Phi-3.5-Vision)           │
├─────────────────────────────────────────────────────────────┤
│  Optional: ConvNeXt-Tiny ensemble for regime detection      │
└─────────────────────────────────────────────────────────────┘
```

## Key Differences from Minah

| Feature | Minah | Carter |
|---------|-------|--------|
| Model | Phi-3.5-Mini (local) | Phi-3.5-Vision (NVIDIA) |
| Backend | Ollama | NVIDIA NIM API |
| GPU | Required (6GB) | Not required |
| Vision | Optional | Default enabled |
| Symbol | ES=F (futures) | ETHUSDT-PERP (crypto) |

## GAF Analysis

Carter generates Gramian Angular Field images from OHLCV data and sends them directly to NVIDIA's Phi-3.5-Vision API:

- **GASF** (Red channel): Trend detection
- **GADF** (Green channel): Cycle detection  
- **Heatmap** (Blue channel): Price correlations

The model rates patterns on:
- Trend strength (1-10)
- Cycle presence (1-10)
- Volatility/burst (1-10)
- Scalping opportunity (1-10)

## Image Size Limit

NVIDIA's inline base64 limit is **180,000 characters**. GAF images are typically ~10-50KB encoded, well within limits. For larger images, use NVIDIA's assets API.

## API

```bash
# Start server
carter server --port 8000

# Endpoints
GET  /           # Status
GET  /health     # Health check
WS   /ws         # Real-time signals
```

## Example Usage

```python
from brainnet.agents import Phi35VisionClient
import base64

# Initialize client
client = Phi35VisionClient()

# Load GAF image
with open("gaf_pattern.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Analyze pattern
response = client.generate_with_image(
    prompt="Analyze this GAF pattern for ETHUSDT-PERP scalping opportunity.",
    image_base64=image_b64,
    temperature=0.20,
)
print(response)
```

## Requirements

- Python 3.10+
- NVIDIA API key (get at https://build.nvidia.com/)
- PostgreSQL + pgvector (optional, for memory)

## License

Same as parent Brainnet project.
