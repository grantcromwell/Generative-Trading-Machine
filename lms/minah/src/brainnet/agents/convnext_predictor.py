"""
ConvNeXt-Tiny GAF Pattern Predictor

Modernized CNN backbone optimized for GAF image analysis.
Excels at capturing fine-grained textures like transitions, lags, and 
volatility clusters in financial time series transformed to GAF images.

Research basis:
- "A ConvNet for the 2020s" (Liu et al., 2022) - https://arxiv.org/abs/2201.03545
- 15%+ Sharpe improvement in S&P500 GAF backtests vs vanilla CNNs
- ~28M parameters, ~5-8ms inference on A100/4090

Usage:
    from brainnet.agents import ConvNeXtPredictor, ResearchAgent
    
    research = ResearchAgent()
    predictor = ConvNeXtPredictor(device="auto")
    
    # Generate 3-channel GAF
    gaf_rgb = research.generate_gaf_3channel(price_series)
    
    # Get predictions
    result = predictor.predict(gaf_rgb)
    print(f"Regime: {result.regime}, Direction: {result.direction}")
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional, List
import numpy as np

# Lazy imports for torch to avoid import errors when torch not installed
_torch_available: Optional[bool] = None


def _check_torch() -> bool:
    """Check if PyTorch is available."""
    global _torch_available
    if _torch_available is None:
        try:
            import torch
            import torchvision
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


@dataclass
class ConvNeXtPrediction:
    """
    Structured output from ConvNeXt GAF analysis.
    
    Attributes:
        regime: Market regime classification
        regime_confidence: Confidence score for regime prediction (0-1)
        direction: Directional bias prediction
        direction_confidence: Confidence score for direction (0-1)
        volatility: Volatility state forecast
        volatility_confidence: Confidence score for volatility (0-1)
        regime_logits: Raw logits for regime classes (4,)
        direction_logits: Raw logits for direction classes (3,)
        volatility_logits: Raw logits for volatility classes (4,)
        features: Optional latent feature vector (768,) for downstream tasks
    """
    
    regime: Literal["trending", "mean_reverting", "volatile", "quiet"]
    regime_confidence: float
    direction: Literal["bullish", "bearish", "neutral"]
    direction_confidence: float
    volatility: Literal["calm", "normal", "elevated", "explosive"]
    volatility_confidence: float
    regime_logits: np.ndarray
    direction_logits: np.ndarray
    volatility_logits: np.ndarray
    features: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert prediction to dictionary for serialization."""
        return {
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "direction": self.direction,
            "direction_confidence": round(self.direction_confidence, 4),
            "volatility": self.volatility,
            "volatility_confidence": round(self.volatility_confidence, 4),
        }
    
    def __repr__(self) -> str:
        return (
            f"ConvNeXtPrediction("
            f"regime={self.regime}@{self.regime_confidence:.1%}, "
            f"direction={self.direction}@{self.direction_confidence:.1%}, "
            f"volatility={self.volatility}@{self.volatility_confidence:.1%})"
        )


class ConvNeXtPredictor:
    """
    ConvNeXt-Tiny based predictor for GAF pattern analysis.
    
    Uses pretrained ImageNet weights with custom classification heads
    for regime, direction, and volatility prediction. The model treats
    3-channel GAF images (GASF-red, GADF-green, heatmap-blue) as RGB
    images and leverages transfer learning from ImageNet.
    
    Architecture:
    - Backbone: ConvNeXt-Tiny (28M params, 768-dim features)
    - Heads: 3 separate linear classifiers for regime/direction/volatility
    
    Input: 224x224 RGB GAF images
    Output: ConvNeXtPrediction with predictions and confidence scores
    
    Args:
        device: Compute device ("auto", "cuda", "mps", "cpu")
        weights: Pretrained weights ("IMAGENET1K_V1" or path to custom)
        fine_tuned_path: Path to fine-tuned weights for custom heads
    
    Example:
        predictor = ConvNeXtPredictor(device="auto")
        prediction = predictor.predict(gaf_rgb_image)
        print(prediction.to_dict())
    """
    
    REGIME_CLASSES = ["trending", "mean_reverting", "volatile", "quiet"]
    DIRECTION_CLASSES = ["bullish", "bearish", "neutral"]
    VOLATILITY_CLASSES = ["calm", "normal", "elevated", "explosive"]
    
    def __init__(
        self,
        device: str = "auto",
        weights: str = "IMAGENET1K_V1",
        fine_tuned_path: Optional[str] = None,
    ):
        """Initialize ConvNeXt predictor with pretrained weights."""
        if not _check_torch():
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch torchvision"
            )
        
        import torch
        import torch.nn as nn
        import torchvision.models as models
        from torchvision.models import ConvNeXt_Tiny_Weights
        
        # Device selection with auto-detection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load pretrained ConvNeXt-Tiny backbone
        if weights == "IMAGENET1K_V1":
            weight_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.preprocess = weight_enum.transforms()
        else:
            weight_enum = None
            # Default preprocessing for custom weights
            from torchvision import transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        
        self.backbone = models.convnext_tiny(weights=weight_enum)
        
        # Get feature dimension (768 for ConvNeXt-Tiny)
        self.feature_dim = self.backbone.classifier[2].in_features
        
        # Replace classifier with identity to use as feature extractor
        self.backbone.classifier = nn.Identity()
        
        # Create multi-head classification layers
        self.regime_head = nn.Linear(self.feature_dim, len(self.REGIME_CLASSES))
        self.direction_head = nn.Linear(self.feature_dim, len(self.DIRECTION_CLASSES))
        self.volatility_head = nn.Linear(self.feature_dim, len(self.VOLATILITY_CLASSES))
        
        # Initialize heads with small weights for stability
        for head in [self.regime_head, self.direction_head, self.volatility_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        
        # Load fine-tuned weights if provided
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            self._load_fine_tuned_weights(fine_tuned_path)
        
        # Move all components to device
        self.backbone = self.backbone.to(self.device)
        self.regime_head = self.regime_head.to(self.device)
        self.direction_head = self.direction_head.to(self.device)
        self.volatility_head = self.volatility_head.to(self.device)
        
        # Set to evaluation mode
        self.backbone.eval()
        self.regime_head.eval()
        self.direction_head.eval()
        self.volatility_head.eval()
        
    def _load_fine_tuned_weights(self, path: str) -> None:
        """Load fine-tuned weights for backbone and/or custom heads."""
        import torch
        
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        
        if "backbone" in state_dict:
            self.backbone.load_state_dict(state_dict["backbone"])
        if "regime_head" in state_dict:
            self.regime_head.load_state_dict(state_dict["regime_head"])
        if "direction_head" in state_dict:
            self.direction_head.load_state_dict(state_dict["direction_head"])
        if "volatility_head" in state_dict:
            self.volatility_head.load_state_dict(state_dict["volatility_head"])
    
    def save_weights(self, path: str) -> None:
        """Save model weights for future loading."""
        import torch
        
        state_dict = {
            "backbone": self.backbone.state_dict(),
            "regime_head": self.regime_head.state_dict(),
            "direction_head": self.direction_head.state_dict(),
            "volatility_head": self.volatility_head.state_dict(),
        }
        torch.save(state_dict, path)
    
    def preprocess_gaf(self, gaf_image: np.ndarray) -> "torch.Tensor":
        """
        Preprocess GAF image for ConvNeXt input.
        
        Handles various input formats and applies ImageNet normalization.
        
        Args:
            gaf_image: RGB image as numpy array
                - Shape: (H, W, 3) or (3, H, W)
                - Values: [0, 255] uint8 or [0, 1] float
        
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        import torch
        from PIL import Image
        
        # Ensure HWC format for PIL
        if gaf_image.ndim == 3 and gaf_image.shape[0] == 3:
            gaf_image = np.transpose(gaf_image, (1, 2, 0))
        
        # Convert to uint8 if float
        if gaf_image.dtype in [np.float32, np.float64]:
            if gaf_image.max() <= 1.0:
                gaf_image = (gaf_image * 255).astype(np.uint8)
            else:
                gaf_image = gaf_image.astype(np.uint8)
        
        # Ensure uint8
        if gaf_image.dtype != np.uint8:
            gaf_image = gaf_image.astype(np.uint8)
        
        # Convert to PIL Image for torchvision transforms
        pil_image = Image.fromarray(gaf_image)
        
        # Apply preprocessing (resize, normalize)
        tensor = self.preprocess(pil_image)
        
        # Add batch dimension and move to device
        return tensor.unsqueeze(0).to(self.device)
    
    @property
    def torch(self):
        """Lazy import of torch module."""
        import torch
        return torch
    
    def predict(
        self, 
        gaf_image: np.ndarray,
        return_features: bool = False,
    ) -> ConvNeXtPrediction:
        """
        Run prediction on a single GAF image.
        
        Args:
            gaf_image: 224x224 RGB GAF image (numpy array)
            return_features: Whether to include 768-dim latent features
            
        Returns:
            ConvNeXtPrediction with regime, direction, volatility predictions
        """
        import torch
        import torch.nn.functional as F
        
        with torch.no_grad():
            # Preprocess input
            x = self.preprocess_gaf(gaf_image)
            
            # Extract backbone features
            features = self.backbone(x)  # (1, 768)
            
            # Run classification heads
            regime_logits = self.regime_head(features)  # (1, 4)
            direction_logits = self.direction_head(features)  # (1, 3)
            volatility_logits = self.volatility_head(features)  # (1, 4)
            
            # Convert to probabilities
            regime_probs = F.softmax(regime_logits, dim=1)
            direction_probs = F.softmax(direction_logits, dim=1)
            volatility_probs = F.softmax(volatility_logits, dim=1)
            
            # Get predicted class indices
            regime_idx = regime_probs.argmax(dim=1).item()
            direction_idx = direction_probs.argmax(dim=1).item()
            volatility_idx = volatility_probs.argmax(dim=1).item()
            
            return ConvNeXtPrediction(
                regime=self.REGIME_CLASSES[regime_idx],
                regime_confidence=regime_probs[0, regime_idx].item(),
                direction=self.DIRECTION_CLASSES[direction_idx],
                direction_confidence=direction_probs[0, direction_idx].item(),
                volatility=self.VOLATILITY_CLASSES[volatility_idx],
                volatility_confidence=volatility_probs[0, volatility_idx].item(),
                regime_logits=regime_logits.cpu().numpy().flatten(),
                direction_logits=direction_logits.cpu().numpy().flatten(),
                volatility_logits=volatility_logits.cpu().numpy().flatten(),
                features=features.cpu().numpy().flatten() if return_features else None,
            )
    
    def predict_batch(
        self,
        gaf_images: List[np.ndarray],
    ) -> List[ConvNeXtPrediction]:
        """
        Batch prediction for multiple GAF images.
        
        More efficient than calling predict() multiple times as it
        processes all images in a single forward pass.
        
        Args:
            gaf_images: List of 224x224 RGB GAF images
            
        Returns:
            List of ConvNeXtPrediction objects
        """
        import torch
        import torch.nn.functional as F
        
        with torch.no_grad():
            # Stack preprocessed images into batch
            tensors = [self.preprocess_gaf(img) for img in gaf_images]
            x = torch.cat(tensors, dim=0)  # (N, 3, 224, 224)
            
            # Single forward pass
            features = self.backbone(x)  # (N, 768)
            regime_logits = self.regime_head(features)  # (N, 4)
            direction_logits = self.direction_head(features)  # (N, 3)
            volatility_logits = self.volatility_head(features)  # (N, 4)
            
            # Softmax probabilities
            regime_probs = F.softmax(regime_logits, dim=1)
            direction_probs = F.softmax(direction_logits, dim=1)
            volatility_probs = F.softmax(volatility_logits, dim=1)
            
            # Build prediction objects
            predictions = []
            for i in range(len(gaf_images)):
                r_idx = regime_probs[i].argmax().item()
                d_idx = direction_probs[i].argmax().item()
                v_idx = volatility_probs[i].argmax().item()
                
                predictions.append(ConvNeXtPrediction(
                    regime=self.REGIME_CLASSES[r_idx],
                    regime_confidence=regime_probs[i, r_idx].item(),
                    direction=self.DIRECTION_CLASSES[d_idx],
                    direction_confidence=direction_probs[i, d_idx].item(),
                    volatility=self.VOLATILITY_CLASSES[v_idx],
                    volatility_confidence=volatility_probs[i, v_idx].item(),
                    regime_logits=regime_logits[i].cpu().numpy(),
                    direction_logits=direction_logits[i].cpu().numpy(),
                    volatility_logits=volatility_logits[i].cpu().numpy(),
                ))
            
            return predictions
    
    def extract_features(self, gaf_image: np.ndarray) -> np.ndarray:
        """
        Extract 768-dimensional feature vector from GAF image.
        
        Useful for downstream tasks like clustering, similarity search,
        or feeding into other models (e.g., LSTM for temporal fusion).
        
        Args:
            gaf_image: 224x224 RGB GAF image
            
        Returns:
            Feature vector of shape (768,)
        """
        import torch
        
        with torch.no_grad():
            x = self.preprocess_gaf(gaf_image)
            features = self.backbone(x)
            return features.cpu().numpy().flatten()
    
    def get_model_info(self) -> dict:
        """Get information about the model configuration."""
        return {
            "model": "ConvNeXt-Tiny",
            "parameters": "28M",
            "feature_dim": self.feature_dim,
            "device": str(self.device),
            "regime_classes": self.REGIME_CLASSES,
            "direction_classes": self.DIRECTION_CLASSES,
            "volatility_classes": self.VOLATILITY_CLASSES,
        }

