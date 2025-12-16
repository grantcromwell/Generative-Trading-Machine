"""
Complete GAF Trading Loop Example
Pulls live data, creates GAF image, analyzes with Phi-3.5-Mini,
computes calibrated confidence interval, decides position, stores in Mem0.

This example uses the engine module from brainnet.services.
"""

from brainnet.services.engine import run_single_analysis, run_trading_loop


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Brainnet GAF Trading Loop")
    parser.add_argument("--symbol", default="ES=F", help="Trading symbol (e.g., BTC-USD, ETH-USD, ES=F, NQ=F)")
    parser.add_argument("--interval", default="5m", help="Data interval")
    parser.add_argument("--delay", type=int, default=300, help="Delay between iterations (seconds)")
    parser.add_argument("--single", action="store_true", help="Run single analysis instead of loop")
    args = parser.parse_args()

    if args.single:
        result = run_single_analysis(args.symbol, args.interval)
        print(f"\nFinal Result: {result}")
    else:
        run_trading_loop(args.symbol, args.interval, args.delay)
