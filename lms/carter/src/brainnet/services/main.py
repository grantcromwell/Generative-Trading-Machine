"""
Entrypoint CLI / FastAPI websocket runner

Carter fork: Phi-3.5-Vision-Instruct for ETHUSDT-PERP scalping.
Supports both LLM-based vision analysis and ConvNeXt neural pattern recognition.
"""

import json
import click
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from brainnet.orchestrator import Router
from brainnet.core import load_config, MemoryManager
from brainnet.services.tui import launch_tui, launch_tui_interactive
from brainnet.services.engine import run_single_analysis, run_trading_loop
from brainnet.agents import _CONVNEXT_AVAILABLE

app = FastAPI(title="Carter Trading API", version="0.1.0")

# Default symbol for ETHUSDT-PERP scalping
DEFAULT_SYMBOL = "ETHUSDT-PERP"


@app.get("/")
async def root():
    return {"status": "ok", "service": "carter", "model": "microsoft/phi-3.5-vision-instruct", "backend": "nvidia"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trading signals."""
    await websocket.accept()
    config = load_config()
    router = Router(config)
    memory = MemoryManager(config)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            result = router.trigger(data)
            memory.add({
                "trade_outcome": result["decision"],
                "confidence": result["confidence"],
            })
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass


@click.group()
def cli():
    """Carter CLI - Phi-3.5-Vision ETHUSDT-PERP Scalper"""
    pass


@cli.command()
@click.option("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol")
def run(symbol: str):
    """Run a single analysis."""
    import yfinance as yf

    router = Router()
    data = yf.download(symbol, period="1d", interval="5m", progress=False)

    if data.empty:
        click.echo("No data received")
        return

    result = router.trigger(data, symbol=symbol)
    click.echo(f"Decision: {result['decision']}")
    click.echo(f"Confidence: {result['confidence']:.3f}")


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
def server(host: str, port: int):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)


@cli.command()
def gui():
    """🧠 Launch the Carter terminal GUI selector."""
    selected_symbol = launch_tui_interactive()
    
    if selected_symbol:
        run_single_analysis(symbol=selected_symbol)
    else:
        click.echo("\n👋 Exited without selection.")


@cli.command()
@click.option("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol")
@click.option("--interval", default="5m", help="Data interval")
@click.option("--delay", default=300, help="Delay between iterations (seconds)")
def loop(symbol: str, interval: str, delay: int):
    """Run continuous trading loop."""
    run_trading_loop(symbol=symbol, interval=interval, delay=delay)


@cli.command()
@click.option("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol")
def analyze(symbol: str):
    """Run single analysis on a symbol."""
    run_single_analysis(symbol=symbol)


@cli.command()
@click.option("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol")
@click.option("--interval", default="5m", help="Data interval")
@click.option("--device", default="auto", type=click.Choice(["auto", "cuda", "mps", "cpu"]), 
              help="Compute device for ConvNeXt")
@click.option("--no-llm", is_flag=True, help="Skip LLM analysis (ConvNeXt only)")
def convnext(symbol: str, interval: str, device: str, no_llm: bool):
    """🧠 Run ConvNeXt-enhanced GAF analysis.
    
    Uses ConvNeXt-Tiny neural network to analyze 3-channel GAF images
    for regime detection, directional bias, and volatility forecasting.
    
    Requires PyTorch: pip install torch torchvision
    """
    if not _CONVNEXT_AVAILABLE:
        click.echo("❌ ConvNeXt requires PyTorch. Install with:")
        click.echo("   pip install torch torchvision")
        return
    
    from brainnet.services.engine import run_convnext_analysis
    
    result = run_convnext_analysis(
        symbol=symbol,
        interval=interval,
        combine_with_llm=not no_llm,
        device=device,
    )
    
    if "error" in result:
        click.echo(f"❌ Error: {result['error']}")


@cli.command()
@click.option("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol")
@click.option("--interval", default="5m", help="Data interval")
@click.option("--delay", default=300, help="Delay between iterations (seconds)")
@click.option("--device", default="auto", type=click.Choice(["auto", "cuda", "mps", "cpu"]),
              help="Compute device for ConvNeXt")
@click.option("--no-llm", is_flag=True, help="Skip LLM analysis (ConvNeXt only)")
def convnext_loop(symbol: str, interval: str, delay: int, device: str, no_llm: bool):
    """🔄 Run continuous ConvNeXt trading loop.
    
    Combines ConvNeXt neural pattern recognition with Phi-3.5-Vision analysis
    for ensemble trading decisions.
    
    Requires PyTorch: pip install torch torchvision
    """
    if not _CONVNEXT_AVAILABLE:
        click.echo("❌ ConvNeXt requires PyTorch. Install with:")
        click.echo("   pip install torch torchvision")
        return
    
    from brainnet.services.engine import run_convnext_trading_loop
    
    run_convnext_trading_loop(
        symbol=symbol,
        interval=interval,
        delay=delay,
        combine_with_llm=not no_llm,
        device=device,
    )


@cli.command()
def convnext_info():
    """ℹ️  Show ConvNeXt model information and availability."""
    click.echo("\n🧠 ConvNeXt-Tiny GAF Pattern Recognition")
    click.echo("=" * 50)
    
    if not _CONVNEXT_AVAILABLE:
        click.echo("\n❌ Status: NOT AVAILABLE")
        click.echo("\nPyTorch not installed. To enable ConvNeXt, run:")
        click.echo("   pip install torch torchvision")
        return
    
    click.echo("\n✅ Status: AVAILABLE")
    
    from brainnet.agents import ConvNeXtPredictor
    
    try:
        predictor = ConvNeXtPredictor(device="cpu")
        info = predictor.get_model_info()
        
        click.echo(f"\nModel: {info['model']}")
        click.echo(f"Parameters: {info['parameters']}")
        click.echo(f"Feature Dimension: {info['feature_dim']}")
        click.echo(f"Default Device: {info['device']}")
        
        click.echo(f"\nRegime Classes: {', '.join(info['regime_classes'])}")
        click.echo(f"Direction Classes: {', '.join(info['direction_classes'])}")
        click.echo(f"Volatility Classes: {', '.join(info['volatility_classes'])}")
        
        # Check GPU availability
        import torch
        click.echo(f"\nGPU Support:")
        click.echo(f"  CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}")
        if hasattr(torch.backends, 'mps'):
            click.echo(f"  MPS:  {'✅ Available' if torch.backends.mps.is_available() else '❌ Not available'}")
        
    except Exception as e:
        click.echo(f"\n⚠️  Error loading model: {e}")
    
    click.echo("")


@cli.command()
def info():
    """ℹ️  Show Carter configuration info."""
    import os
    click.echo("\n🧠 Carter - Phi-3.5-Vision via NVIDIA NIM")
    click.echo("=" * 50)
    click.echo(f"\nModel: microsoft/phi-3.5-vision-instruct")
    click.echo(f"Backend: NVIDIA NIM API")
    click.echo(f"Default Symbol: {DEFAULT_SYMBOL}")
    click.echo(f"Vision Mode: Enabled (native base64 PNG)")
    click.echo(f"Local GPU: Not required")
    
    # Check API key
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key:
        click.echo(f"\n✅ NVIDIA_API_KEY: Set ({api_key[:12]}...)")
    else:
        click.echo(f"\n❌ NVIDIA_API_KEY: Not set")
        click.echo("   Get your key at: https://build.nvidia.com/")
    click.echo("")


if __name__ == "__main__":
    cli()
