"""
Entrypoint CLI / FastAPI websocket runner
Powered by DeepSeek-V3 via HuggingFace Pipeline

Supports both LLM-based analysis and ConvNeXt neural pattern recognition.
"""

import json
import click
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from brainnet.orchestrator import Router
from brainnet.core import load_config, MemoryManager
from brainnet.services.engine import run_single_analysis, run_trading_loop
from brainnet.agents import _CONVNEXT_AVAILABLE

app = FastAPI(title="Brainnet/Banis Trading API", version="0.1.0", description="DeepSeek-V3 Powered")


@app.get("/")
async def root():
    return {"status": "ok", "service": "brainnet-banis", "llm": "deepseek-v3"}


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
    """Brainnet/Banis CLI - Powered by DeepSeek-V3"""
    pass


@cli.command()
@click.option("--symbol", default="ES=F", help="Trading symbol")
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
@click.option("--symbol", default="ES=F", help="Trading symbol")
def analyze(symbol: str):
    """Run single analysis on a symbol."""
    run_single_analysis(symbol=symbol)


@cli.command()
@click.option("--symbol", default="ES=F", help="Trading symbol")
@click.option("--interval", default="5m", help="Data interval")
@click.option("--delay", default=300, help="Delay between iterations (seconds)")
def loop(symbol: str, interval: str, delay: int):
    """Run continuous trading loop."""
    run_trading_loop(symbol=symbol, interval=interval, delay=delay)


@cli.command()
@click.option("--symbol", default="ES=F", help="Trading symbol")
@click.option("--interval", default="5m", help="Data interval")
@click.option("--device", default="auto", type=click.Choice(["auto", "cuda", "mps", "cpu"]))
@click.option("--no-llm", is_flag=True, help="Skip LLM analysis (ConvNeXt only)")
def convnext(symbol: str, interval: str, device: str, no_llm: bool):
    """🧠 Run ConvNeXt-enhanced GAF analysis."""
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
def info():
    """ℹ️  Show model information."""
    click.echo("\n🧠 BRAINNET/BANIS - DeepSeek-V3 Quant System")
    click.echo("=" * 50)
    click.echo("\n  LLM: DeepSeek-V3-0324 (685B MoE, 37B active)")
    click.echo("  Context: 128K tokens")
    click.echo("  Backend: HuggingFace Pipeline / API")
    click.echo("")
    
    if _CONVNEXT_AVAILABLE:
        click.echo("  ConvNeXt: ✅ Available")
    else:
        click.echo("  ConvNeXt: ❌ Not available (pip install torch torchvision)")
    
    click.echo("")


if __name__ == "__main__":
    cli()

