# Minah - Brainnet Trading Analysis System

Minah is a part of the Brainnet project, focused on trading analysis using AI models like Phi-3.5 for pattern recognition and decision-making in financial markets.

## Features
- **Trading Analysis**: Utilizes Phi-3.5-mini for analyzing financial data and making trading decisions.
- **Gramian Angular Field (GAF)**: Generates GAF images for visualizing price data patterns.
- **Coinglass Derivatives Data**: Fetches aggregated funding rates and long/short account ratios for crypto assets to enhance trading signals.
- **Data Flywheel**: Implements a continuous data improvement pipeline by collecting trading outcomes and feedback for model retraining.
- **Milvus Integration**: Uses a local Milvus setup (`milvus-lite`) for vector storage and similarity search of trading data embeddings.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure settings in `src/brainnet/core/config.py`, including NVIDIA API key for embedding generation.
3. Run analysis or server as needed (see usage below).

## Usage
- **Single Analysis**: Run a trading analysis on a specific symbol.
  ```bash
  python -m brainnet.services.main analyze --symbol ES=F
  ```
- **Continuous Loop**: Run a continuous trading loop.
  ```bash
  python -m brainnet.services.main loop --symbol ES=F --interval 5m --delay 300
  ```
- **API Server**: Start the FastAPI server for real-time trading signals.
  ```bash
  python -m brainnet.services.main server --host 0.0.0.0 --port 8000
  ```
- **GUI**: Launch the terminal GUI for interactive symbol selection.
  ```bash
  python -m brainnet.services.main gui
  ```
- **Banis Hybrid Analysis**: Launch Banis-style analysis with DeepSeek reasoning and 65% confidence threshold.
  ```bash
  python -m brainnet.services.main banis
  ```

## Integration Details
- **Banis Hybrid System**: Combines Minah's data sources (Coinglass derivatives, Milvus vectors) with Banis's DeepSeek-V3 BSC reasoning. Buy decisions only trigger when confidence ≥65%.
- **Coinglass API**: Provides aggregated funding rates and long/short account ratios for crypto derivatives markets. Set `COINGLASS_API_KEY` environment variable or configure in `config.py`. Get an API key at [coinglass.com/pricing](https://www.coinglass.com/pricing).
- **Data Flywheel**: Collects and refines trading data for continuous improvement of models.
- **Milvus**: Locally deployed vector database for storing and retrieving embeddings of trading patterns and analyses.
- **Embedding Model**: Uses Phi-3.5-vision-instruct via NVIDIA API for generating embeddings of trading data.

## Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `COINGLASS_API_KEY` | API key for Coinglass derivatives data | (empty) |
| `USE_COINGLASS` | Enable/disable Coinglass data fetching | `true` |
| `NVIDIA_API_KEY` | API key for NVIDIA embedding API | (empty) |
| `USE_MILVUS` | Enable/disable Milvus vector storage | `false` |
| `MILVUS_URI` | Milvus server URI | `http://localhost:19530` |

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

## License
[Add your license information here]
