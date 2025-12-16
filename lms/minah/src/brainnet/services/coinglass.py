"""
Coinglass API client for fetching derivatives market data.

Provides access to:
- Aggregated funding rates across exchanges
- Long/short account ratios
- Open interest data
"""

import os
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass


# Coinglass API V4 base URL
COINGLASS_API_BASE = "https://open-api-v4.coinglass.com"


@dataclass
class FundingRateData:
    """Aggregated funding rate data."""
    symbol: str
    timestamp: datetime
    rate: float  # Current/latest funding rate
    rate_open: float  # Opening rate in period
    rate_high: float  # Highest rate in period
    rate_low: float  # Lowest rate in period
    rate_close: float  # Closing rate in period
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "rate": self.rate,
            "rate_open": self.rate_open,
            "rate_high": self.rate_high,
            "rate_low": self.rate_low,
            "rate_close": self.rate_close,
        }


@dataclass
class LongShortRatioData:
    """Long/short account ratio data."""
    symbol: str
    timestamp: datetime
    long_percent: float  # Percentage of accounts long
    short_percent: float  # Percentage of accounts short
    long_short_ratio: float  # Ratio of long to short accounts
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "long_percent": self.long_percent,
            "short_percent": self.short_percent,
            "long_short_ratio": self.long_short_ratio,
        }


@dataclass
class DerivativesMetrics:
    """Combined derivatives metrics for trading analysis."""
    symbol: str
    timestamp: datetime
    funding_rate: Optional[FundingRateData]
    long_short_ratio: Optional[LongShortRatioData]
    
    @property
    def funding_sentiment(self) -> str:
        """Interpret funding rate sentiment."""
        if not self.funding_rate:
            return "unknown"
        rate = self.funding_rate.rate
        if rate > 0.01:  # > 1% is very bullish sentiment (longs paying shorts)
            return "very_bullish"
        elif rate > 0.005:
            return "bullish"
        elif rate > -0.005:
            return "neutral"
        elif rate > -0.01:
            return "bearish"
        else:
            return "very_bearish"
    
    @property
    def positioning_sentiment(self) -> str:
        """Interpret long/short ratio sentiment."""
        if not self.long_short_ratio:
            return "unknown"
        ratio = self.long_short_ratio.long_short_ratio
        if ratio > 3.0:  # Very crowded long
            return "extremely_long"
        elif ratio > 2.0:
            return "heavily_long"
        elif ratio > 1.2:
            return "slightly_long"
        elif ratio > 0.8:
            return "neutral"
        elif ratio > 0.5:
            return "slightly_short"
        else:
            return "heavily_short"
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "funding_rate": self.funding_rate.to_dict() if self.funding_rate else None,
            "long_short_ratio": self.long_short_ratio.to_dict() if self.long_short_ratio else None,
            "funding_sentiment": self.funding_sentiment,
            "positioning_sentiment": self.positioning_sentiment,
        }


class CoinglassClient:
    """
    Client for Coinglass API V4.
    
    Fetches aggregated derivatives data including funding rates and long/short ratios.
    Requires a Coinglass API key (get one at https://www.coinglass.com/pricing).
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Coinglass client.
        
        Args:
            api_key: Coinglass API key. If not provided, reads from COINGLASS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("COINGLASS_API_KEY", "")
        self.base_url = COINGLASS_API_BASE
        self._session = requests.Session()
        
        if self.api_key:
            self._session.headers.update({
                "accept": "application/json",
                "CG-API-KEY": self.api_key,
            })
    
    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request to Coinglass."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "0":
                raise ValueError(f"Coinglass API error: {data.get('msg', 'Unknown error')}")
            
            return data
        except requests.RequestException as e:
            print(f"Coinglass API request failed: {e}")
            return {"code": "-1", "msg": str(e), "data": []}
    
    def get_funding_rate_history(
        self,
        symbol: str = "BTC",
        exchange: str = "Binance",
        interval: str = "h4",
        limit: int = 24,
    ) -> List[FundingRateData]:
        """
        Get historical funding rate data (OHLC format).
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            exchange: Exchange name (e.g., "Binance", "OKX", "Bybit")
            interval: Time interval (h1, h4, h8, h12, h24)
            limit: Number of data points to return
            
        Returns:
            List of FundingRateData objects
        """
        endpoint = "/api/futures/funding-rate/ohlc-history"
        params = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "limit": limit,
        }
        
        response = self._request(endpoint, params)
        data = response.get("data", [])
        
        results = []
        for item in data:
            try:
                results.append(FundingRateData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(item["time"] / 1000),
                    rate=float(item.get("close", 0)),
                    rate_open=float(item.get("open", 0)),
                    rate_high=float(item.get("high", 0)),
                    rate_low=float(item.get("low", 0)),
                    rate_close=float(item.get("close", 0)),
                ))
            except (KeyError, ValueError) as e:
                print(f"Error parsing funding rate data: {e}")
                continue
        
        return results
    
    def get_aggregated_funding_rate(
        self,
        symbol: str = "BTC",
    ) -> Optional[FundingRateData]:
        """
        Get the latest aggregated funding rate across exchanges.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            
        Returns:
            FundingRateData with the latest aggregated rate, or None if unavailable
        """
        # Use the exchange list endpoint to get aggregated data
        endpoint = "/api/futures/funding-rate/exchange-list"
        params = {"symbol": symbol}
        
        response = self._request(endpoint, params)
        data = response.get("data", [])
        
        if not data:
            return None
        
        # Calculate weighted average funding rate across exchanges
        total_oi = 0.0
        weighted_rate = 0.0
        
        for exchange_data in data:
            try:
                rate = float(exchange_data.get("rate", 0))
                oi = float(exchange_data.get("openInterest", 1))  # Use OI as weight
                weighted_rate += rate * oi
                total_oi += oi
            except (ValueError, TypeError):
                continue
        
        if total_oi > 0:
            avg_rate = weighted_rate / total_oi
        else:
            # Fallback to simple average
            rates = [float(d.get("rate", 0)) for d in data if d.get("rate")]
            avg_rate = sum(rates) / len(rates) if rates else 0.0
        
        return FundingRateData(
            symbol=symbol,
            timestamp=datetime.now(),
            rate=avg_rate,
            rate_open=avg_rate,
            rate_high=avg_rate,
            rate_low=avg_rate,
            rate_close=avg_rate,
        )
    
    def get_long_short_ratio(
        self,
        symbol: str = "BTC",
        interval: str = "h4",
        limit: int = 24,
    ) -> List[LongShortRatioData]:
        """
        Get global long/short account ratio history.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            interval: Time interval (m30, h1, h4, h12, h24)
            limit: Number of data points to return
            
        Returns:
            List of LongShortRatioData objects
        """
        endpoint = "/api/futures/global-long-short-account-ratio"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        
        response = self._request(endpoint, params)
        data = response.get("data", [])
        
        results = []
        for item in data:
            try:
                results.append(LongShortRatioData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(item["time"] / 1000),
                    long_percent=float(item.get("global_account_long_percent", 50)),
                    short_percent=float(item.get("global_account_short_percent", 50)),
                    long_short_ratio=float(item.get("global_account_long_short_ratio", 1.0)),
                ))
            except (KeyError, ValueError) as e:
                print(f"Error parsing long/short ratio data: {e}")
                continue
        
        return results
    
    def get_latest_long_short_ratio(
        self,
        symbol: str = "BTC",
    ) -> Optional[LongShortRatioData]:
        """
        Get the latest global long/short account ratio.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            
        Returns:
            LongShortRatioData with the latest ratio, or None if unavailable
        """
        history = self.get_long_short_ratio(symbol=symbol, interval="h1", limit=1)
        return history[0] if history else None
    
    def get_derivatives_metrics(
        self,
        symbol: str = "BTC",
    ) -> DerivativesMetrics:
        """
        Get combined derivatives metrics for a symbol.
        
        Fetches both aggregated funding rate and long/short ratio.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            
        Returns:
            DerivativesMetrics with funding rate and long/short data
        """
        funding_rate = self.get_aggregated_funding_rate(symbol)
        long_short = self.get_latest_long_short_ratio(symbol)
        
        return DerivativesMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            funding_rate=funding_rate,
            long_short_ratio=long_short,
        )
    
    def is_available(self) -> bool:
        """Check if Coinglass API is available and configured."""
        if not self.api_key:
            return False
        
        try:
            # Simple health check
            response = self._request("/api/futures/coins", {"limit": 1})
            return response.get("code") == "0"
        except Exception:
            return False


# Convenience function for quick access
def get_coinglass_metrics(symbol: str = "BTC", api_key: Optional[str] = None) -> DerivativesMetrics:
    """
    Quick helper to get derivatives metrics for a symbol.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        DerivativesMetrics with funding rate and long/short data
    """
    client = CoinglassClient(api_key=api_key)
    return client.get_derivatives_metrics(symbol)
