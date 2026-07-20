"""
Extractor de features principal.

Coordina los diferentes backends y proporciona una interfaz
unificada para la extracción de features.
"""

import time
from typing import Optional, Dict, Any, Tuple
from collections import deque

import numpy as np

from models.feature_extractor.cache import FeatureCache
from models.feature_extractor.validator import FeatureValidator
from models.feature_extractor.backends.base import FeatureBackend
from utils.logger import LoggerMixin


class FeatureExtractor(LoggerMixin):
    """
    Extractor de features para re-identificación.

    Coordina el backend, caché y validador para extraer
    features de regiones de imagen.

    Attributes:
        backend: Backend de extracción de features
        cache: Caché de features
        validator: Validador de calidad
        feature_dim: Dimensión del vector de features
    """

    def __init__(
        self,
        backend: FeatureBackend,
        cache_size: int = 500,
        feature_dim: int = 2048,
        max_age_seconds: float = 3.0
    ):
        """
        Inicializa el extractor de features.

        Args:
            backend: Backend de extracción
            cache_size: Tamaño del caché
            feature_dim: Dimensión del vector de features
            max_age_seconds: Edad máxima de las entradas de caché
        """
        self.backend = backend
        self.feature_dim = feature_dim
        self.cache = FeatureCache(
            max_size=cache_size,
            max_age_seconds=max_age_seconds
        )
        self.validator = FeatureValidator()

        self._metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cached_extractions": 0,
            "avg_extraction_time_ms": 0.0,
            "extraction_times": deque(maxlen=100),
        }

        try:
            self.backend.warmup()
        except Exception as e:
            self.logger.warning(f"Error en warmup: {e}")

        self.logger.info(
            "FeatureExtractor inicializado",
            backend=backend.name,
            backend_available=backend.is_available,
            feature_dim=feature_dim,
            cache_size=cache_size
        )

    def extract_features(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence: float = 0.5,
        force: bool = False
    ) -> Optional[np.ndarray]:
        """
        Extrae features de una región de imagen.

        Args:
            image: Imagen completa
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Confianza de la detección (0-1)
            force: Forzar extracción aunque la calidad sea baja

        Returns:
            Optional[np.ndarray]: Vector de features o None
        """
        start_time = time.perf_counter()
        self._metrics["total_extractions"] += 1

        if image is None or image.size == 0:
            self._metrics["failed_extractions"] += 1
            return None

        if not self.validator.validate_bbox(bbox, image.shape):
            self._metrics["failed_extractions"] += 1
            return None

        cache_key = self.cache.compute_key(image, bbox)
        cached = self.cache.get(cache_key)

        if cached is not None and not force:
            self._metrics["cached_extractions"] += 1
            self._metrics["successful_extractions"] += 1
            return cached

        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2]

        quality_score = self.validator.validate_region(region)

        if quality_score < 0.3 and not force:
            self._metrics["failed_extractions"] += 1
            self.logger.debug(
                "Región de baja calidad",
                quality=f"{quality_score:.2f}",
                bbox=bbox
            )
            return None

        features = self.backend.extract(region)

        if features is None:
            self._metrics["failed_extractions"] += 1
            return None

        if not force and quality_score >= 0.3:
            self.cache.put(cache_key, features, confidence, quality_score)

        self._metrics["successful_extractions"] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._metrics["extraction_times"].append(elapsed_ms)
        self._metrics["avg_extraction_time_ms"] = (
            sum(self._metrics["extraction_times"]) /
            len(self._metrics["extraction_times"])
        )

        return features

    def compare_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        method: str = "cosine"
    ) -> float:
        """
        Compara dos vectores de features.

        Args:
            features1: Primer vector
            features2: Segundo vector
            method: Método de comparación ('cosine', 'euclidean', 'dot')

        Returns:
            float: Similitud (0-1 para cosine, distancia para otros)
        """
        if features1 is None or features2 is None:
            return 0.0

        try:
            if method == "cosine":
                norm1 = np.linalg.norm(features1)
                norm2 = np.linalg.norm(features2)

                if norm1 == 0 or norm2 == 0:
                    return 0.0

                similarity = np.dot(features1, features2) / (norm1 * norm2)
                return max(0.0, min(1.0, similarity))

            elif method == "euclidean":
                dist = np.linalg.norm(features1 - features2)
                return 1.0 / (1.0 + dist)

            elif method == "dot":
                return np.dot(features1, features2)

            else:
                self.logger.warning(f"Método no soportado: {method}")
                return self.compare_features(features1, features2, "cosine")

        except Exception as e:
            self.logger.debug(f"Error comparando features: {e}")
            return 0.0

    def clear_cache(self) -> None:
        """Limpia el caché de features."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        return self.cache.get_stats()

    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento."""
        total = self._metrics["total_extractions"]
        success = self._metrics["successful_extractions"]

        return {
            **self._metrics,
            "success_rate": success / max(1, total),
            "backend": self.backend.name,
            "backend_available": self.backend.is_available,
            "cache": self.cache.get_stats(),
            "validator": self.validator.get_stats(),
            "feature_dim": self.feature_dim,
        }

    def reset_metrics(self) -> None:
        """Reinicia las métricas."""
        self._metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "cached_extractions": 0,
            "avg_extraction_time_ms": 0.0,
            "extraction_times": deque(maxlen=100),
        }
        self.validator.reset_stats()

    @property
    def is_available(self) -> bool:
        """Verifica si el extractor está disponible."""
        return self.backend.is_available

    @property
    def feature_dimension(self) -> int:
        """Dimensión del vector de features."""
        return self.feature_dim

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_cache()
