"""
Market data service for fetching live and historical data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class MarketDataService:
    """Service for fetching market data from various sources."""

    def __init__(self, default_symbol: str = "ES=F"):
        self.default_symbol = default_symbol
        self._cache: dict[str, pd.DataFrame] = {}

    def get_intraday(
        self,
        symbol: Optional[str] = None,
        interval: str = "5m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        Get intraday data.

        Args:
            symbol: Ticker symbol
            interval: Bar interval (1m, 5m, 15m, etc.)
            period: Lookback period

        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol or self.default_symbol
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        return data

    def get_daily(
        self,
        symbol: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
    ) -> pd.DataFrame:
        """Get daily data."""
        symbol = symbol or self.default_symbol

        if start and end:
            data = yf.download(symbol, start=start, end=end, progress=False)
        else:
            data = yf.download(symbol, period=period, progress=False)

        return data

    def get_latest_price(self, symbol: Optional[str] = None) -> float:
        """Get the latest price."""
        symbol = symbol or self.default_symbol
        ticker = yf.Ticker(symbol)
        return ticker.info.get("regularMarketPrice", 0.0)

    def get_quote(self, symbol: Optional[str] = None) -> dict:
        """Get full quote information."""
        symbol = symbol or self.default_symbol
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol,
            "price": info.get("regularMarketPrice"),
            "open": info.get("regularMarketOpen"),
            "high": info.get("regularMarketDayHigh"),
            "low": info.get("regularMarketDayLow"),
            "volume": info.get("regularMarketVolume"),
            "change": info.get("regularMarketChange"),
            "change_pct": info.get("regularMarketChangePercent"),
        }
