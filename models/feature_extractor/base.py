"""Extractor de features principal.

Coordina los diferentes backends y proporciona una interfaz
unificada para la extracción de features.
"""

from __future__ import annotations

from collections import deque
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants.values import MIN_REGION_QUALITY
from models.feature_extractor.cache import FeatureCache
from models.feature_extractor.validator import FeatureValidator
from utils.logger import LoggerMixin

if TYPE_CHECKING:
    from models.feature_extractor.backends.base import FeatureBackend


class FeatureExtractor(LoggerMixin):
    """Extractor de features para re-identificación.

    Coordina el backend, caché y validador para extraer
    features de regiones de imagen.

    Attributes:
        backend: Backend de extracción de features
        cache: Caché de features
        validator: Validador de calidad
        feature_dim: Dimensión del vector de features
    """

    __slots__ = (
        "backend",
        "feature_dim",
        "cache",
        "validator",
        "_metrics",
    )

    def __init__(
        self,
        backend: FeatureBackend,
        cache_size: int = 500,
        feature_dim: int = 2048,
        max_age_seconds: float = 3.0,
    ) -> None:
        """Inicializa el extractor de features.

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
            max_age_seconds=max_age_seconds,
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
            cache_size=cache_size,
        )

    def extract_features(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        confidence: float = 0.5,
        *,
        force: bool = False,
    ) -> np.ndarray | None:
        """Extrae features de una región de imagen.

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

        if not self._validate_input(image, bbox):
            self._metrics["failed_extractions"] += 1
            return None

        cached_result = self._check_cache(image, bbox, force)
        if cached_result is not None:
            return cached_result

        region = self._extract_region(image, bbox)
        quality_score = self.validator.validate_region(region)

        if not self._is_quality_sufficient(quality_score, force):
            self._metrics["failed_extractions"] += 1
            self.logger.debug(
                "Región de baja calidad",
                quality=f"{quality_score:.2f}",
                bbox=bbox,
            )
            return None

        features = self.backend.extract(region)

        if features is None:
            self._metrics["failed_extractions"] += 1
            return None

        self._cache_if_valid(bbox, features, confidence, quality_score, force)

        self._update_extraction_metrics(start_time)

        return features

    def _validate_input(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """Valida la entrada para extracción de features.

        Args:
            image: Imagen a validar
            bbox: Bounding box a validar

        Returns:
            bool: True si la entrada es válida
        """
        if image is None or image.size == 0:
            return False

        return self.validator.validate_bbox(bbox, image.shape)

    def _check_cache(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        force: bool,
    ) -> np.ndarray | None:
        """Verifica si los features están en caché.

        Args:
            image: Imagen completa
            bbox: Bounding box
            force: Si se fuerza la extracción

        Returns:
            Optional[np.ndarray]: Features cacheados o None
        """
        if force:
            return None

        cache_key = self.cache.compute_key(image, bbox)
        cached = self.cache.get(cache_key)

        if cached is not None:
            self._metrics["cached_extractions"] += 1
            self._metrics["successful_extractions"] += 1
            return cached

        return None

    def _extract_region(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Extrae la región de la imagen según el bounding box."""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    def _is_quality_sufficient(self, quality_score: float, force: bool) -> bool:
        """Verifica si la calidad de la región es suficiente.

        Args:
            quality_score: Puntuación de calidad (0-1)
            force: Si se fuerza la extracción

        Returns:
            bool: True si la calidad es suficiente o se fuerza
        """
        return force or quality_score >= MIN_REGION_QUALITY

    def _cache_if_valid(
        self,
        bbox: tuple[int, int, int, int],
        features: np.ndarray,
        confidence: float,
        quality_score: float,
        force: bool,
    ) -> None:
        """Almacena en caché si las condiciones son favorables.

        Args:
            bbox: Bounding box
            features: Features extraídos
            confidence: Confianza de la detección
            quality_score: Puntuación de calidad
            force: Si se forzó la extracción
        """
        if force or quality_score < MIN_REGION_QUALITY:
            return

        cache_key = self.cache.compute_key(bbox, features)
        self.cache.put(cache_key, features, confidence, quality_score)

    def _update_extraction_metrics(self, start_time: float) -> None:
        """Actualiza las métricas de extracción."""
        self._metrics["successful_extractions"] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._metrics["extraction_times"].append(elapsed_ms)
        self._metrics["avg_extraction_time_ms"] = sum(self._metrics["extraction_times"]) / len(
            self._metrics["extraction_times"]
        )

    def compare_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        method: str = "cosine",
    ) -> float:
        """Compara dos vectores de features.

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
                return self._cosine_similarity(features1, features2)
            if method == "euclidean":
                return self._euclidean_similarity(features1, features2)
            if method == "dot":
                return float(np.dot(features1, features2))
            self.logger.warning(f"Método no soportado: {method}")
            return self.compare_features(features1, features2, "cosine")

        except Exception as e:
            self.logger.debug(f"Error comparando features: {e}")
            return 0.0

    def _cosine_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calcula la similitud coseno entre dos vectores."""
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(f1, f2) / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))

    def _euclidean_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Calcula la similitud basada en distancia euclidiana."""
        dist = np.linalg.norm(f1 - f2)
        return 1.0 / (1.0 + dist)

    def clear_cache(self) -> None:
        """Limpia el caché de features."""
        self.cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del caché."""
        return self.cache.get_stats()

    def get_metrics(self) -> dict[str, Any]:
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

    def __enter__(self) -> FeatureExtractor:
        """Entra al context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sale del context manager y limpia el caché."""
        self.clear_cache()
