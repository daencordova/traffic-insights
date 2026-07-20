"""
Backend SIFT para extracción de features.

Utiliza SIFT (Scale-Invariant Feature Transform) para extraer
features locales de la imagen.
"""
from typing import Optional

import cv2
import numpy as np

from models.feature_extractor.backends.base import FeatureBackend
from utils.logger import LoggerMixin


class SIFTBackend(FeatureBackend, LoggerMixin):
    """
    Backend SIFT para extracción de features.

    Características:
    - Features locales invariantes a escala y rotación
    - Buena para objetos con textura
    - No requiere GPU

    Attributes:
        feature_dim: Dimensión del vector de features (128)
        is_available: Si el backend está disponible
    """

    FEATURE_DIM = 128

    def __init__(self, n_features: int = 128):
        """
        Inicializa el backend SIFT.

        Args:
            n_features: Número máximo de features a extraer
        """
        self.n_features = n_features
        self._sift = None
        self._available = False
        self._warmed_up = False

        self._initialize()

        self.logger.info(
            "SIFTBackend inicializado",
            available=self._available,
            n_features=n_features
        )

    def _initialize(self) -> None:
        """Inicializa el extractor SIFT."""
        try:
            self._sift = cv2.SIFT_create(nfeatures=self.n_features)
            self._available = True
            self.logger.info("SIFT inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando SIFT: {e}")
            self._available = False

    def extract(self, region: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae features usando SIFT.

        Args:
            region: Región de imagen (recorte del objeto)

        Returns:
            Optional[np.ndarray]: Vector de features de dimensión 128
        """
        if not self._available or self._sift is None:
            return None

        if region is None or region.size == 0:
            return None

        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            keypoints, descriptors = self._sift.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                return self._fallback_features(region)

            features = np.mean(descriptors, axis=0)

            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features.astype(np.float32)

        except Exception as e:
            self.logger.debug(f"Error en extracción SIFT: {e}")
            return self._fallback_features(region)

    def _fallback_features(self, region: np.ndarray) -> Optional[np.ndarray]:
        """
        Features de fallback usando histograma simple.

        Args:
            region: Región de imagen

        Returns:
            Optional[np.ndarray]: Vector de features de dimensión 128
        """
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if len(hist) < self.FEATURE_DIM:
                hist = np.pad(hist, (0, self.FEATURE_DIM - len(hist)))

            return hist[:self.FEATURE_DIM].astype(np.float32)

        except Exception:
            return None

    def warmup(self) -> None:
        """Calienta el backend."""
        if self._warmed_up or not self._available:
            return

        self.logger.info("🔥 Calentando SIFT...")
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            self.extract(dummy)
            self._warmed_up = True
            self.logger.info("✅ SIFT calentado")
        except Exception as e:
            self.logger.warning(f"Error en warmup: {e}")

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    @property
    def is_available(self) -> bool:
        return self._available
