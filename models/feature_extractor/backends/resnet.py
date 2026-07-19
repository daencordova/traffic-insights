"""
Backend ResNet para extracción de features usando PyTorch.
"""

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .base import FeatureBackend
from utils.logger import LoggerMixin


class ResNetBackend(FeatureBackend, LoggerMixin):
    """
    Backend ResNet50 para extracción de features.

    Utiliza ResNet50 pre-entrenado en ImageNet como extractor
    de features visuales. Requiere PyTorch.

    Attributes:
        device: Dispositivo para inferencia ('cuda', 'mps', 'cpu')
        feature_dim: Dimensión del vector de features (2048)
        is_available: Si el backend está disponible
    """

    FEATURE_DIM = 2048

    def __init__(self, device: str = "auto"):
        """
        Inicializa el backend ResNet.

        Args:
            device: Dispositivo para inferencia ('cuda', 'mps', 'cpu', 'auto')
        """
        self._device = self._get_device(device)
        self._model = None
        self._transform = None
        self._available = False
        self._warmed_up = False

        self._initialize()

        self.logger.info(
            "ResNetBackend inicializado",
            available=self._available,
            device=self._device,
            feature_dim=self.FEATURE_DIM
        )

    def _get_device(self, device: str) -> str:
        """Determina el dispositivo a usar."""
        if not TORCH_AVAILABLE:
            return "cpu"

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        return device

    def _initialize(self) -> None:
        """Inicializa el modelo ResNet."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch no disponible")
            self._available = False
            return

        try:
            self.logger.info("Cargando ResNet50 pre-entrenado...")

            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self._model = models.resnet50(weights=weights)

            self._model = nn.Sequential(*list(self._model.children())[:-1])

            self._model.to(self._device)
            self._model.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            self._available = True
            self.logger.info(f"ResNet50 cargado en {self._device}")

        except Exception as e:
            self.logger.error(f"Error inicializando ResNet: {e}")
            self._available = False

    def extract(self, region: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae features usando ResNet50.

        Args:
            region: Región de imagen (recorte del objeto)

        Returns:
            Optional[np.ndarray]: Vector de features de dimensión 2048
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
            self.logger.debug(f"Error en extracción ResNet: {e}")
            return None

    def warmup(self) -> None:
        """Calienta el modelo para reducir latencia inicial."""
        if not self._available or self._warmed_up:
            return

        self.logger.info("🔥 Calentando ResNet...")

        try:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)

            for _ in range(3):
                self.extract(dummy)

            self._warmed_up = True
            self.logger.info("✅ ResNet calentado")

        except Exception as e:
            self.logger.warning(f"Error en warmup: {e}")

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    @property
    def is_available(self) -> bool:
        return self._available
