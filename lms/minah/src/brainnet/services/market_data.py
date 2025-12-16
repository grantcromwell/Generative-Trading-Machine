"""
Market data service for fetching live and historical data.

Integrates:
- yfinance for price data (OHLCV)
- Coinglass for derivatives data (funding rate, long/short ratio)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from brainnet.services.coinglass import CoinglassClient, DerivativesMetrics


# Symbol mapping from yfinance symbols to Coinglass symbols
SYMBOL_MAP = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "DOGE-USD": "DOGE",
    "XRP-USD": "XRP",
    "ADA-USD": "ADA",
    "AVAX-USD": "AVAX",
    "DOT-USD": "DOT",
    "LINK-USD": "LINK",
    "MATIC-USD": "MATIC",
    "LTC-USD": "LTC",
    "SHIB-USD": "SHIB",
    "TRX-USD": "TRX",
    "ATOM-USD": "ATOM",
    "UNI-USD": "UNI",
    # Futures symbols (not crypto - no Coinglass data)
    "ES=F": None,  # S&P 500 E-mini
    "NQ=F": None,  # Nasdaq E-mini
    "YM=F": None,  # Dow E-mini
    "GC=F": None,  # Gold
    "CL=F": None,  # Crude Oil
}


class MarketDataService:
    """Service for fetching market data from various sources."""

    def __init__(self, default_symbol: str = "ES=F", coinglass_api_key: Optional[str] = None):
        self.default_symbol = default_symbol
        self._cache: dict[str, pd.DataFrame] = {}
        self._coinglass = CoinglassClient(api_key=coinglass_api_key)
    
    def _get_coinglass_symbol(self, symbol: str) -> Optional[str]:
        """Map yfinance symbol to Coinglass symbol."""
        # Check direct mapping
        if symbol in SYMBOL_MAP:
            return SYMBOL_MAP[symbol]
        
        # Try to extract base symbol from crypto pairs
        if symbol.endswith("-USD"):
            return symbol.replace("-USD", "")
        
        # Unknown symbol - might be traditional futures
        return None

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
    
    def get_derivatives_metrics(self, symbol: Optional[str] = None) -> Optional[DerivativesMetrics]:
        """
        Get derivatives metrics (funding rate, long/short ratio) for a symbol.
        
        Only available for crypto symbols that Coinglass supports.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "ETH-USD")
            
        Returns:
            DerivativesMetrics with funding rate and long/short data, or None if unavailable
        """
        symbol = symbol or self.default_symbol
        coinglass_symbol = self._get_coinglass_symbol(symbol)
        
        if not coinglass_symbol:
            return None
        
        try:
            return self._coinglass.get_derivatives_metrics(coinglass_symbol)
        except Exception as e:
            print(f"Failed to fetch derivatives metrics: {e}")
            return None
    
    def get_funding_rate(self, symbol: Optional[str] = None) -> Optional[float]:
        """
        Get the current aggregated funding rate for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "ETH-USD")
            
        Returns:
            Current funding rate as a decimal (e.g., 0.01 = 1%), or None if unavailable
        """
        metrics = self.get_derivatives_metrics(symbol)
        if metrics and metrics.funding_rate:
            return metrics.funding_rate.rate
        return None
    
    def get_long_short_ratio(self, symbol: Optional[str] = None) -> Optional[float]:
        """
        Get the current long/short account ratio for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "ETH-USD")
            
        Returns:
            Long/short ratio (e.g., 2.0 means 2x more longs than shorts), or None if unavailable
        """
        metrics = self.get_derivatives_metrics(symbol)
        if metrics and metrics.long_short_ratio:
            return metrics.long_short_ratio.long_short_ratio
        return None
    
    def get_market_data_with_derivatives(
        self,
        symbol: Optional[str] = None,
        interval: str = "5m",
        period: str = "1d",
    ) -> Dict[str, Any]:
        """
        Get comprehensive market data including price data and derivatives metrics.
        
        Args:
            symbol: Trading symbol
            interval: Bar interval for price data
            period: Lookback period for price data
            
        Returns:
            Dictionary with price data and derivatives metrics
        """
        symbol = symbol or self.default_symbol
        
        # Get price data
        price_data = self.get_intraday(symbol, interval, period)
        
        # Get latest price
        try:
            if not price_data.empty:
                latest_price = float(price_data['Close'].iloc[-1].iloc[0]) if hasattr(price_data['Close'].iloc[-1], 'iloc') else float(price_data['Close'].iloc[-1])
            else:
                latest_price = self.get_latest_price(symbol)
        except Exception:
            latest_price = 0.0
        
        # Get derivatives metrics (if available for this symbol)
        derivatives = self.get_derivatives_metrics(symbol)
        
        result = {
            "symbol": symbol,
            "price": latest_price,
            "bars": len(price_data) if not price_data.empty else 0,
            "interval": interval,
            "period": period,
            "price_data": price_data,
        }
        
        # Add derivatives data if available
        if derivatives:
            result["derivatives"] = derivatives.to_dict()
            result["funding_rate"] = derivatives.funding_rate.rate if derivatives.funding_rate else None
            result["funding_sentiment"] = derivatives.funding_sentiment
            result["long_short_ratio"] = derivatives.long_short_ratio.long_short_ratio if derivatives.long_short_ratio else None
            result["positioning_sentiment"] = derivatives.positioning_sentiment
        else:
            result["derivatives"] = None
            result["funding_rate"] = None
            result["funding_sentiment"] = "unavailable"
            result["long_short_ratio"] = None
            result["positioning_sentiment"] = "unavailable"
        
        return result
