"""
Backend basado en histogramas para extracción de features.

Este backend es rápido y no requiere GPU, ideal para CPU.
Combina histogramas de color, textura y momentos.
"""

from typing import Optional

import cv2
import numpy as np

from .base import FeatureBackend
from utils.logger import LoggerMixin


class HistogramBackend(FeatureBackend, LoggerMixin):
    """
    Backend basado en histogramas para extracción de features.

    Características:
    - Histogramas HSV y LAB
    - Histogramas de gradiente
    - Momentos de Hu
    - Estadísticas básicas
    - No requiere GPU

    Attributes:
        feature_dim: Dimensión del vector de features (2048)
        is_available: Siempre True (no requiere dependencias externas)
    """

    FEATURE_DIM = 2048

    def __init__(self):
        """Inicializa el backend de histogramas."""
        self._available = True
        self._warmed_up = False

        self.logger.info(
            "HistogramBackend inicializado",
            feature_dim=self.FEATURE_DIM
        )

    def extract(self, region: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae features usando histogramas.

        Args:
            region: Región de imagen (recorte del objeto)

        Returns:
            Optional[np.ndarray]: Vector de features de dimensión 2048
        """
        if region is None or region.size == 0:
            return None

        try:
            features = []

            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            hist_hsv = cv2.calcHist(
                [hsv], [0, 1], None,
                [8, 8], [0, 180, 0, 256]
            )
            hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            features.extend(hist_hsv[:64])

            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            hist_lab = cv2.calcHist(
                [lab], [0, 1, 2], None,
                [4, 4, 4], [0, 256, 0, 256, 0, 256]
            )
            hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
            features.extend(hist_lab[:32])

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobel_x, sobel_y)

            hist_mag, _ = np.histogram(
                magnitude.flatten(),
                bins=16,
                range=(0, 255)
            )
            hist_mag = cv2.normalize(
                hist_mag.astype(np.float32),
                hist_mag.astype(np.float32)
            ).flatten()
            features.extend(hist_mag[:16])

            angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)
            hist_angle, _ = np.histogram(
                angle.flatten(),
                bins=16,
                range=(0, 360)
            )
            hist_angle = cv2.normalize(
                hist_angle.astype(np.float32),
                hist_angle.astype(np.float32)
            ).flatten()
            features.extend(hist_angle[:8])

            moments = cv2.HuMoments(cv2.moments(gray)).flatten()
            features.extend(moments[:4])

            stats = [
                float(np.mean(gray)) / 255.0,
                float(np.std(gray)) / 255.0,
                float(np.median(gray)) / 255.0,
                float(np.min(gray)) / 255.0,
                float(np.max(gray)) / 255.0,
            ]
            features.extend(stats)

            h, w = region.shape[:2]
            area_ratio = (h * w) / (region.size / 3)
            aspect_ratio = w / h if h > 0 else 1.0
            features.extend([area_ratio, min(aspect_ratio, 5.0) / 5.0])

            features_array = np.array(features, dtype=np.float32)

            if len(features_array) > self.FEATURE_DIM:
                features_array = features_array[:self.FEATURE_DIM]
            elif len(features_array) < self.FEATURE_DIM:
                padding = self.FEATURE_DIM - len(features_array)
                features_array = np.pad(features_array, (0, padding))

            norm = np.linalg.norm(features_array)
            if norm > 0:
                features_array = features_array / norm

            return features_array

        except Exception as e:
            self.logger.debug(f"Error en extracción de histogramas: {e}")
            return None

    def warmup(self) -> None:
        """Calienta el backend."""
        if self._warmed_up:
            return

        self.logger.info("🔥 Calentando HistogramBackend...")
        try:
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            self.extract(dummy)
            self._warmed_up = True
            self.logger.info("✅ HistogramBackend calentado")
        except Exception as e:
            self.logger.warning(f"Error en warmup: {e}")

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    @property
    def is_available(self) -> bool:
        return self._available
