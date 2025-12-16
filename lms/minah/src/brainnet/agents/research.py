"""
Research Agent - GAF generation + vision analysis with Phi-3.5-Mini
Supports both vision mode (with images) and text mode (numerical features)

Also provides 3-channel GAF generation for ConvNeXt neural pattern recognition.
"""

import base64
import os
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from scipy.ndimage import zoom

from .base import BaseAgent


class ResearchAgent(BaseAgent):
    """
    Research agent that generates Gramian Angular Field (GAF) images
    from time series data and analyzes them using Phi-3.5-Mini.
    
    Supports two modes:
    - Vision mode: Sends GAF images to vision-capable model
    - Text mode: Extracts numerical features from GAF for text-only models
    """

    def __init__(self, use_vision: bool = None, **kwargs):
        super().__init__(**kwargs)
        self.gaf_summation = GramianAngularField(method='summation')
        self.gaf_difference = GramianAngularField(method='difference')
        # Auto-detect vision capability or use env var
        self.use_vision = use_vision if use_vision is not None else os.getenv("USE_VISION", "false").lower() == "true"

    def generate_gaf_image(
        self,
        series: np.ndarray,
        method: str = 'summation',
        image_size: int = 64,
        cmap: str = 'rainbow',
    ) -> str:
        """
        Generate a GAF image from time series data.

        Args:
            series: 1D numpy array of price/value data
            method: 'summation' or 'difference'
            image_size: Size of the output image
            cmap: Colormap for visualization

        Returns:
            Base64-encoded PNG image string
        """
        # Ensure series is 2D for pyts
        if series.ndim == 1:
            series = series.reshape(1, -1)

        # Normalize to [-1, 1] range
        series_min = series.min()
        series_max = series.max()
        if series_max - series_min > 0:
            series_normalized = 2 * (series - series_min) / (series_max - series_min) - 1
        else:
            series_normalized = series

        # Generate GAF
        gaf = self.gaf_summation if method == 'summation' else self.gaf_difference
        gaf_image = gaf.fit_transform(series_normalized)[0]

        # Create visualization
        fig, ax = plt.subplots(figsize=(image_size / 10, image_size / 10), dpi=100)
        ax.imshow(gaf_image, cmap=cmap, origin='lower')
        ax.axis('off')

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

        return image_base64

    def generate_dual_gaf(self, series: np.ndarray) -> Tuple[str, str]:
        """Generate both summation and difference GAF images."""
        gaf_sum = self.generate_gaf_image(series, method='summation')
        gaf_diff = self.generate_gaf_image(series, method='difference')
        return gaf_sum, gaf_diff

    def generate_gaf_3channel(
        self,
        series: np.ndarray,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Generate 3-channel GAF image for ConvNeXt analysis.
        
        Creates an RGB image where each channel captures different aspects
        of the time series structure:
        - Red channel: GASF (Gramian Angular Summation Field) - captures trend
        - Green channel: GADF (Gramian Angular Difference Field) - captures cycles
        - Blue channel: Normalized price heatmap - captures correlations
        
        This 3-channel representation is designed for ConvNeXt-Tiny input,
        which expects 224x224 RGB images pretrained on ImageNet.
        
        Args:
            series: 1D numpy array of price/value data
            image_size: Output image size (default 224 for ImageNet compatibility)
            
        Returns:
            RGB image as numpy array (H, W, 3) with values [0, 255] uint8
            
        Example:
            >>> agent = ResearchAgent()
            >>> series = np.sin(np.linspace(0, 10, 100))
            >>> gaf_rgb = agent.generate_gaf_3channel(series)
            >>> gaf_rgb.shape
            (224, 224, 3)
        """
        # Ensure 2D for pyts
        if series.ndim == 1:
            series = series.reshape(1, -1)
        
        # Normalize to [-1, 1] range for GAF computation
        series_min, series_max = series.min(), series.max()
        if series_max - series_min > 1e-8:
            series_norm = 2 * (series - series_min) / (series_max - series_min) - 1
        else:
            # Handle constant series
            series_norm = np.zeros_like(series)
        
        # Generate GAF matrices
        gaf_sum = self.gaf_summation.fit_transform(series_norm)[0]  # GASF
        gaf_diff = self.gaf_difference.fit_transform(series_norm)[0]  # GADF
        
        # Create price heatmap (outer product of normalized prices)
        # This captures price-level correlations across time
        price_norm = series_norm.flatten()
        heatmap = np.outer(price_norm, price_norm)
        
        # Normalize each channel to [0, 1]
        def normalize_channel(arr: np.ndarray) -> np.ndarray:
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min > 1e-8:
                return (arr - arr_min) / (arr_max - arr_min)
            return np.full_like(arr, 0.5)  # Mid-gray for constant
        
        red = normalize_channel(gaf_sum)
        green = normalize_channel(gaf_diff)
        blue = normalize_channel(heatmap)
        
        # Resize to target size using bilinear interpolation
        current_size = red.shape[0]
        if current_size != image_size:
            scale = image_size / current_size
            red = zoom(red, scale, order=1)  # Bilinear
            green = zoom(green, scale, order=1)
            blue = zoom(blue, scale, order=1)
            
            # Clamp after interpolation (can slightly exceed [0,1])
            red = np.clip(red, 0, 1)
            green = np.clip(green, 0, 1)
            blue = np.clip(blue, 0, 1)
        
        # Stack channels and convert to uint8
        rgb = np.stack([red, green, blue], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        
        return rgb

    def generate_gaf_3channel_base64(
        self,
        series: np.ndarray,
        image_size: int = 224,
    ) -> str:
        """
        Generate 3-channel GAF and return as base64 PNG string.
        
        Convenient for web transmission or storage.
        
        Args:
            series: 1D numpy array of price data
            image_size: Output image size
            
        Returns:
            Base64-encoded PNG image string
        """
        from PIL import Image
        
        rgb = self.generate_gaf_3channel(series, image_size)
        
        # Convert to PIL and encode
        img = Image.fromarray(rgb)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def extract_gaf_features(self, series: np.ndarray) -> dict:
        """
        Extract numerical features from GAF matrix for text-based analysis.
        
        Args:
            series: 1D numpy array of price data
            
        Returns:
            Dictionary of GAF-derived features
        """
        if series.ndim == 1:
            series = series.reshape(1, -1)
            
        # Normalize
        series_min = series.min()
        series_max = series.max()
        if series_max - series_min > 0:
            series_norm = 2 * (series - series_min) / (series_max - series_min) - 1
        else:
            series_norm = series
            
        # Generate GAF matrices
        gaf_sum = self.gaf_summation.fit_transform(series_norm)[0]
        gaf_diff = self.gaf_difference.fit_transform(series_norm)[0]
        
        # Extract features
        features = {
            # Diagonal features (temporal autocorrelation)
            "diagonal_mean": float(np.mean(np.diag(gaf_sum))),
            "diagonal_std": float(np.std(np.diag(gaf_sum))),
            
            # Quadrant analysis
            "upper_left_mean": float(np.mean(gaf_sum[:len(gaf_sum)//2, :len(gaf_sum)//2])),
            "lower_right_mean": float(np.mean(gaf_sum[len(gaf_sum)//2:, len(gaf_sum)//2:])),
            
            # Overall statistics
            "gaf_sum_mean": float(np.mean(gaf_sum)),
            "gaf_sum_std": float(np.std(gaf_sum)),
            "gaf_diff_mean": float(np.mean(gaf_diff)),
            "gaf_diff_std": float(np.std(gaf_diff)),
            
            # Trend indicators
            "trend_score": float(np.mean(gaf_sum) - np.mean(gaf_diff)),
            
            # Symmetry (cycles tend to be more symmetric)
            "symmetry": float(np.mean(np.abs(gaf_sum - gaf_sum.T))),
            
            # Corner analysis (recent vs old patterns)
            "recent_strength": float(np.mean(gaf_sum[-10:, -10:])),
            "historical_strength": float(np.mean(gaf_sum[:10, :10])),
        }
        
        # Derived scores
        features["momentum"] = features["recent_strength"] - features["historical_strength"]
        
        return features

    def analyze_pattern(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        gaf_features: Optional[dict] = None,
    ) -> str:
        """
        Analyze a GAF pattern using Phi-3.5-Mini.
        Uses vision mode if available, otherwise uses extracted features.

        Args:
            image_base64: Base64-encoded PNG image
            prompt: Custom analysis prompt
            gaf_features: Pre-extracted GAF features (for text mode)

        Returns:
            Analysis text from the model
        """
        base_prompt = """Analyze this Gramian Angular Field (GAF) representation of financial time series data.

Identify and rate the following patterns on a scale of 1-10:
1. TREND: Is there a clear directional trend? (1=no trend, 10=strong trend)
2. CYCLE: Are there cyclical patterns visible? (1=no cycles, 10=strong cycles)
3. BURST: Are there sudden bursts or volatility clusters? (1=stable, 10=high volatility)

Also identify:
- Primary pattern type (trend/mean-reversion/breakout/consolidation)
- Estimated phase in any cycle
- Trading bias (bullish/bearish/neutral)

Provide your analysis in a structured format."""

        if self.use_vision:
            # Vision mode - send image
            final_prompt = prompt or base_prompt
            return self.llm.generate_with_image(final_prompt, image_base64, max_tokens=1024)
        else:
            # Text mode - use extracted features
            if gaf_features is None:
                gaf_features = {}
            
            feature_text = "\n".join([f"- {k}: {v:.4f}" for k, v in gaf_features.items()])
            
            text_prompt = f"""{prompt or base_prompt}

GAF Feature Extraction Results:
{feature_text}

Based on these GAF-derived metrics:
- diagonal_mean/std: Temporal autocorrelation (high = trending)
- trend_score: Positive = uptrend, Negative = downtrend
- symmetry: Low = cyclical patterns, High = random
- momentum: Positive = strengthening, Negative = weakening
- recent_strength vs historical_strength: Pattern evolution

Analyze these features and provide your pattern assessment."""

            return self.llm.generate([{"role": "user", "content": text_prompt}], max_tokens=1024)

    def research(
        self,
        data: pd.DataFrame,
        memory_context: str = "",
        lookback: int = 100,
    ) -> dict:
        """
        Perform complete research analysis on market data.

        Args:
            data: DataFrame with 'Close' column
            memory_context: Context from previous analyses (from Mem0)
            lookback: Number of bars to analyze

        Returns:
            Dictionary with analysis results and GAF image
        """
        # Extract close prices
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                close = data['Close'].values.flatten()
            elif 'close' in data.columns:
                close = data['close'].values.flatten()
            else:
                close = data.iloc[:, 0].values.flatten()
        else:
            close = np.array(data).flatten()

        # Use last N bars
        series = close[-lookback:] if len(close) > lookback else close

        # Generate GAF image
        gaf_image = self.generate_gaf_image(series)
        
        # Extract GAF features for text mode
        gaf_features = self.extract_gaf_features(series)

        # Build analysis prompt with memory context
        base_prompt = """Analyze this Gramian Angular Field (GAF) representation of financial time series data.

Rate each pattern on a scale of 1-10:
- TREND strength (1=none, 10=very strong)
- CYCLE presence (1=none, 10=very clear)
- BURST/volatility (1=calm, 10=explosive)

Identify the primary market regime and provide a trading bias (bullish/bearish/neutral)."""

        if memory_context:
            prompt = f"""Previous context from memory:
{memory_context}

Given this context, {base_prompt}"""
        else:
            prompt = base_prompt

        # Analyze (uses vision or text mode based on self.use_vision)
        analysis = self.analyze_pattern(gaf_image, prompt, gaf_features)

        # Extract pattern scores (simple parsing)
        scores = self._extract_pattern_scores(analysis)

        return {
            "analysis": analysis,
            "image": gaf_image,
            "features": gaf_features,
            "scores": scores,
            "series_length": len(series),
            "latest_price": float(series[-1]) if len(series) > 0 else None,
        }

    def _extract_pattern_scores(self, analysis: str) -> dict:
        """Extract numerical scores from analysis text."""
        scores = {"trend": 5, "cycle": 5, "burst": 5}  # defaults

        analysis_lower = analysis.lower()

        # Simple extraction - look for patterns like "trend: 8" or "trend strength: 8/10"
        for pattern in ["trend", "cycle", "burst"]:
            import re
            # Match patterns like "trend: 8", "trend strength: 8/10", "trend = 8"
            matches = re.findall(rf'{pattern}[:\s=]+(\d+)', analysis_lower)
            if matches:
                try:
                    scores[pattern] = min(10, max(1, int(matches[0])))
                except ValueError:
                    pass

        return scores
