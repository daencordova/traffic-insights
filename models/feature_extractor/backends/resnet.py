"""ResNet backend for feature extraction using PyTorch.

This backend uses a pre-trained ResNet50 model for visual feature extraction.
Requires PyTorch for inference.

Example:
    >>> from models.feature_extractor.backends import ResNetBackend
    >>>
    >>> backend = ResNetBackend(device="auto")
    >>>
    >>> # Extract features from a cropped object
    >>> region = frame[100:200, 50:150]
    >>> features = backend.extract(region)
    >>>
    >>> if features is not None:
    ...     print(f"Extracted {len(features)} features")
    ...     print(f"Feature dimension: {backend.feature_dim}")
    >>>
    >>> # Warmup for faster first inference
    >>> backend.warmup()
"""

import numpy as np

try:
    import torch
    from torch import nn
    from torchvision import models, transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from models.feature_extractor.backends.base import FeatureBackend
from utils.logger import LoggerMixin


class ResNetBackend(FeatureBackend, LoggerMixin):
    """ResNet50 backend for feature extraction.

    Uses a ResNet50 model pre-trained on ImageNet as a visual
    feature extractor. Requires PyTorch for inference.

    Features:
        - Deep learning-based features
        - Pre-trained on ImageNet
        - High-quality feature representation
        - GPU acceleration support

    Attributes:
        device: Inference device ('cuda', 'mps', 'cpu').
        feature_dim: Dimension of the feature vector (2048).
        is_available: Whether the backend is available.

    Example:
        >>> backend = ResNetBackend(device="cuda")
        >>> features = backend.extract(cropped_region)
        >>> print(features.shape)  # (2048,)
    """

    FEATURE_DIM = 2048

    def __init__(self, device: str = "auto"):
        """Initializes the ResNet backend.

        Args:
            device: Inference device ('cuda', 'mps', 'cpu', 'auto').

        Example:
            >>> # Auto-select best device
            >>> backend = ResNetBackend(device="auto")
            >>>
            >>> # Force CPU
            >>> backend = ResNetBackend(device="cpu")
            >>>
            >>> # Use GPU
            >>> backend = ResNetBackend(device="cuda")
        """
        self._device = self._get_device(device)
        self._model = None
        self._transform = None
        self._available = False
        self._warmed_up = False

        self._initialize()

        self.logger.info(
            "ResNetBackend initialized",
            available=self._available,
            device=self._device,
            feature_dim=self.FEATURE_DIM,
        )

    def _get_device(self, device: str) -> str:
        """Determines the device to use.

        Args:
            device: Device preference.

        Returns:
            str: Selected device.

        Note:
            If 'auto', selects the best available device:
            CUDA > MPS > CPU.
        """
        if not TORCH_AVAILABLE:
            return "cpu"

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        return device

    def _initialize(self) -> None:
        """Initializes the ResNet model.

        Loads the pre-trained ResNet50 model and configures
        the necessary preprocessing transforms.
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available")
            self._available = False
            return

        try:
            self.logger.info("Loading pre-trained ResNet50...")

            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self._model = models.resnet50(weights=weights)

            self._model = nn.Sequential(*list(self._model.children())[:-1])

            self._model.to(self._device)
            self._model.eval()

            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            self._available = True
            self.logger.info(f"ResNet50 loaded on {self._device}")

        except Exception as e:
            self.logger.error(f"Error initializing ResNet: {e}")
            self._available = False

    def extract(self, region: np.ndarray) -> np.ndarray | None:
        """Extracts features using ResNet50.

        Args:
            region: Image region (cropped object patch).

        Returns:
            Optional[np.ndarray]: Feature vector of dimension 2048 or None.

        Example:
            >>> region = frame[100:200, 50:150]
            >>> features = backend.extract(region)
            >>> if features is not None:
            ...     print(f"Extracted {len(features)} features")
        """
        if not self._available or self._model is None:
            return None

        if region is None or region.size == 0:
            return None

        try:
            input_tensor = self._transform(region).unsqueeze(0).to(self._device)

            with torch.no_grad():
                features = self._model(input_tensor)

            features = features.cpu().numpy().flatten()

            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features

        except Exception as e:
            self.logger.debug(f"ResNet extraction error: {e}")
            return None

    def warmup(self) -> None:
        """Warms up the model to reduce initial latency.

        Performs dummy inference to load the model into memory
        and initialize CUDA/MPS kernels.

        Example:
            >>> backend.warmup()
            >>> # Subsequent calls will be faster
        """
        if not self._available or self._warmed_up:
            return

        self.logger.info("Warming up ResNet...")

        try:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)

            for _ in range(3):
                self.extract(dummy)

            self._warmed_up = True
            self.logger.info("ResNet warmed up")

        except Exception as e:
            self.logger.warning(f"Warmup error: {e}")

    @property
    def feature_dim(self) -> int:
        """Dimension of the feature vector."""
        return self.FEATURE_DIM

    @property
    def is_available(self) -> bool:
        """Whether the backend is available."""
        return self._available
