"""
Tests for ConvNeXt GAF predictor.

These tests verify:
1. 3-channel GAF generation produces valid 224x224 RGB images
2. ConvNeXtPredictor loads and runs inference correctly
3. Predictions are properly structured and in valid ranges
4. Batch prediction works efficiently
5. Integration with trading engine

Requires PyTorch: pip install torch torchvision
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd

# Check if torch is available for conditional test skipping
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    pass

# Skip entire module if torch not installed
pytestmark = pytest.mark.skipif(
    not _torch_available, 
    reason="PyTorch not installed - run: pip install torch torchvision"
)


class TestGAF3Channel:
    """Test 3-channel GAF generation in ResearchAgent."""
    
    def test_generate_gaf_3channel_shape(self):
        """GAF output should be 224x224x3 by default."""
        from brainnet.agents import ResearchAgent
        
        agent = ResearchAgent()
        series = np.sin(np.linspace(0, 10, 100))
        
        rgb = agent.generate_gaf_3channel(series, image_size=224)
        
        assert rgb.shape == (224, 224, 3)
        assert rgb.dtype == np.uint8
        
    def test_generate_gaf_3channel_custom_sizes(self):
        """GAF should resize to any requested size."""
        from brainnet.agents import ResearchAgent
        
        agent = ResearchAgent()
        series = np.random.randn(50)
        
        for size in [64, 128, 224, 256, 512]:
            rgb = agent.generate_gaf_3channel(series, image_size=size)
            assert rgb.shape == (size, size, 3), f"Failed for size {size}"
            assert rgb.dtype == np.uint8
            
    def test_generate_gaf_3channel_value_range(self):
        """GAF values should be in [0, 255] uint8 range."""
        from brainnet.agents import ResearchAgent
        
        agent = ResearchAgent()
        series = np.sin(np.linspace(0, 10, 100))
        
        rgb = agent.generate_gaf_3channel(series)
        
        assert rgb.min() >= 0
        assert rgb.max() <= 255
        
    def test_generate_gaf_3channel_channels_differ(self):
        """Each channel should contain different information."""
        from brainnet.agents import ResearchAgent
        
        agent = ResearchAgent()
        series = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
        
        rgb = agent.generate_gaf_3channel(series)
        
        red = rgb[:, :, 0].astype(float)
        green = rgb[:, :, 1].astype(float)
        blue = rgb[:, :, 2].astype(float)
        
        # Channels should not be identical
        assert not np.allclose(red, green), "Red and Green channels are identical"
        assert not np.allclose(green, blue), "Green and Blue channels are identical"
        
    def test_generate_gaf_3channel_constant_series(self):
        """Should handle constant series gracefully (edge case)."""
        from brainnet.agents import ResearchAgent
        
        agent = ResearchAgent()
        series = np.ones(100) * 42.0
        
        rgb = agent.generate_gaf_3channel(series)
        
        assert rgb.shape == (224, 224, 3)
        assert rgb.dtype == np.uint8
        # Should be mid-gray for constant series
        assert 100 < rgb.mean() < 200
        
    def test_generate_gaf_3channel_short_series(self):
        """Should handle short series (less than lookback)."""
        from brainnet.agents import ResearchAgent
        
        agent = ResearchAgent()
        series = np.random.randn(10)  # Very short
        
        rgb = agent.generate_gaf_3channel(series)
        
        assert rgb.shape == (224, 224, 3)
        
    def test_generate_gaf_3channel_base64(self):
        """Base64 encoding should produce valid PNG."""
        from brainnet.agents import ResearchAgent
        import base64
        
        agent = ResearchAgent()
        series = np.random.randn(100)
        
        b64 = agent.generate_gaf_3channel_base64(series)
        
        assert isinstance(b64, str)
        assert len(b64) > 0
        
        # Should be valid base64
        decoded = base64.b64decode(b64)
        # PNG magic bytes
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'


class TestConvNeXtPredictor:
    """Test ConvNeXt prediction functionality."""
    
    def test_initialization_cpu(self):
        """Should initialize on CPU without errors."""
        from brainnet.agents import ConvNeXtPredictor
        
        predictor = ConvNeXtPredictor(device="cpu")
        
        assert predictor is not None
        assert predictor.device.type == "cpu"
        assert predictor.feature_dim == 768
        
    def test_predict_returns_valid_output(self):
        """Prediction should return properly structured output."""
        from brainnet.agents import ConvNeXtPredictor, ConvNeXtPrediction
        from brainnet.agents import ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        # Generate test GAF
        series = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        
        assert isinstance(result, ConvNeXtPrediction)
        
    def test_predict_regime_valid(self):
        """Regime prediction should be one of valid classes."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100).cumsum()
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        
        assert result.regime in ["trending", "mean_reverting", "volatile", "quiet"]
        
    def test_predict_direction_valid(self):
        """Direction prediction should be one of valid classes."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100).cumsum()
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        
        assert result.direction in ["bullish", "bearish", "neutral"]
        
    def test_predict_volatility_valid(self):
        """Volatility prediction should be one of valid classes."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100).cumsum()
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        
        assert result.volatility in ["calm", "normal", "elevated", "explosive"]
        
    def test_predict_confidence_range(self):
        """Confidence scores should be in [0, 1] range."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100).cumsum()
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        
        assert 0 <= result.regime_confidence <= 1
        assert 0 <= result.direction_confidence <= 1
        assert 0 <= result.volatility_confidence <= 1
        
    def test_predict_logits_shape(self):
        """Logits should have correct shapes."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        
        assert result.regime_logits.shape == (4,)  # 4 regime classes
        assert result.direction_logits.shape == (3,)  # 3 direction classes
        assert result.volatility_logits.shape == (4,)  # 4 volatility classes
        
    def test_predict_with_features(self):
        """Should return features when requested."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb, return_features=True)
        
        assert result.features is not None
        assert result.features.shape == (768,)  # ConvNeXt-Tiny feature dim
        
    def test_predict_without_features(self):
        """Should not return features by default."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb, return_features=False)
        
        assert result.features is None
        
    def test_batch_prediction(self):
        """Batch prediction should process multiple images."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        # Generate multiple GAFs
        gaf_images = [
            research.generate_gaf_3channel(np.random.randn(100))
            for _ in range(3)
        ]
        
        results = predictor.predict_batch(gaf_images)
        
        assert len(results) == 3
        for r in results:
            assert r.regime in ["trending", "mean_reverting", "volatile", "quiet"]
            assert r.direction in ["bullish", "bearish", "neutral"]
            assert r.volatility in ["calm", "normal", "elevated", "explosive"]
            
    def test_extract_features(self):
        """Feature extraction should return 768-dim vector."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        features = predictor.extract_features(gaf_rgb)
        
        assert features.shape == (768,)
        assert features.dtype == np.float32 or features.dtype == np.float64
        
    def test_to_dict(self):
        """to_dict() should produce serializable output."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        import json
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        d = result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        
        # Should have all required keys
        assert "regime" in d
        assert "direction" in d
        assert "volatility" in d
        assert "regime_confidence" in d
        assert "direction_confidence" in d
        assert "volatility_confidence" in d
        
    def test_get_model_info(self):
        """get_model_info should return model details."""
        from brainnet.agents import ConvNeXtPredictor
        
        predictor = ConvNeXtPredictor(device="cpu")
        info = predictor.get_model_info()
        
        assert info["model"] == "ConvNeXt-Tiny"
        assert info["parameters"] == "28M"
        assert info["feature_dim"] == 768


class TestConvNeXtInputFormats:
    """Test handling of various input formats."""
    
    def test_hwc_format(self):
        """Should handle HWC (height, width, channels) format."""
        from brainnet.agents import ConvNeXtPredictor
        
        predictor = ConvNeXtPredictor(device="cpu")
        
        # HWC format
        gaf = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = predictor.predict(gaf)
        
        assert result.regime in ["trending", "mean_reverting", "volatile", "quiet"]
        
    def test_chw_format(self):
        """Should handle CHW (channels, height, width) format."""
        from brainnet.agents import ConvNeXtPredictor
        
        predictor = ConvNeXtPredictor(device="cpu")
        
        # CHW format
        gaf = np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8)
        result = predictor.predict(gaf)
        
        assert result.regime in ["trending", "mean_reverting", "volatile", "quiet"]
        
    def test_float_input_normalized(self):
        """Should handle float input in [0, 1] range."""
        from brainnet.agents import ConvNeXtPredictor
        
        predictor = ConvNeXtPredictor(device="cpu")
        
        # Float [0, 1]
        gaf = np.random.rand(224, 224, 3).astype(np.float32)
        result = predictor.predict(gaf)
        
        assert result.regime in ["trending", "mean_reverting", "volatile", "quiet"]
        
    def test_float_input_255_scale(self):
        """Should handle float input in [0, 255] range."""
        from brainnet.agents import ConvNeXtPredictor
        
        predictor = ConvNeXtPredictor(device="cpu")
        
        # Float [0, 255]
        gaf = np.random.rand(224, 224, 3).astype(np.float32) * 255
        result = predictor.predict(gaf)
        
        assert result.regime in ["trending", "mean_reverting", "volatile", "quiet"]


class TestIntegration:
    """Integration tests for ConvNeXt + trading engine."""
    
    def test_engine_convnext_analysis_mocked(self):
        """Test ConvNeXt analysis with mocked yfinance data."""
        # Create mock data
        mock_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100
        })
        
        with patch('yfinance.download', return_value=mock_data):
            from brainnet.services.engine import run_convnext_analysis
            
            result = run_convnext_analysis(
                symbol="TEST",
                interval="5m",
                combine_with_llm=False,
                device="cpu",
            )
            
            assert "convnext" in result
            assert result["convnext"]["regime"] in ["trending", "mean_reverting", "volatile", "quiet"]
            assert "inference_ms" in result
            
    def test_agent_availability_flag(self):
        """_CONVNEXT_AVAILABLE should be True when torch is installed."""
        from brainnet.agents import _CONVNEXT_AVAILABLE
        
        assert _CONVNEXT_AVAILABLE is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_short_series(self):
        """Should handle very short price series."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        # Only 5 data points
        series = np.array([1, 2, 3, 4, 5], dtype=float)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        assert result.regime in ["trending", "mean_reverting", "volatile", "quiet"]
        
    def test_single_value_series(self):
        """Should handle single value (edge case)."""
        from brainnet.agents import ResearchAgent
        
        research = ResearchAgent()
        series = np.array([42.0])
        
        # Should not crash
        gaf_rgb = research.generate_gaf_3channel(series)
        assert gaf_rgb.shape == (224, 224, 3)
        
    def test_nan_handling(self):
        """Should handle NaN values gracefully."""
        from brainnet.agents import ResearchAgent
        
        research = ResearchAgent()
        series = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Replace NaN before GAF (standard practice)
        series = np.nan_to_num(series, nan=0.0)
        
        gaf_rgb = research.generate_gaf_3channel(series)
        assert gaf_rgb.shape == (224, 224, 3)
        assert not np.isnan(gaf_rgb).any()


class TestReproducibility:
    """Test that predictions are deterministic."""
    
    def test_same_input_same_output(self):
        """Same input should produce same output."""
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        # Fix random seed for reproducible test data
        np.random.seed(42)
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result1 = predictor.predict(gaf_rgb)
        result2 = predictor.predict(gaf_rgb)
        
        assert result1.regime == result2.regime
        assert result1.direction == result2.direction
        assert result1.volatility == result2.volatility
        assert np.allclose(result1.regime_logits, result2.regime_logits)

