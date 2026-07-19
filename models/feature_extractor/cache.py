"""
Caché para features extraídos.

Implementa un caché LRU con gestión de memoria para almacenar
features y evitar extracciones redundantes.
"""

import time
import hashlib
from typing import Optional, Dict, Tuple, Any
from collections import OrderedDict

import cv2
import numpy as np

from utils.logger import LoggerMixin


class FeatureCacheEntry:
    """
    Entrada en el caché de features.

    Usa __slots__ para optimizar el uso de memoria.
    """

    __slots__ = ('features', 'timestamp', 'confidence', 'quality', 'access_count')

    def __init__(
        self,
        features: np.ndarray,
        confidence: float,
        quality: float
    ):
        self.features = features.copy()
        self.timestamp = time.time()
        self.confidence = confidence
        self.quality = quality
        self.access_count = 0

    def touch(self) -> None:
        """Actualiza el contador de acceso y timestamp."""
        self.access_count += 1
        self.timestamp = time.time()

    def is_valid(self, max_age: float = 3.0) -> bool:
        """Verifica si la entrada sigue siendo válida."""
        return (time.time() - self.timestamp) < max_age

    def get_score(self) -> float:
        """Calcula una puntuación para evicción."""
        access_score = min(1.0, self.access_count / 10.0)
        quality_score = min(1.0, self.quality)
        age_score = min(1.0, (time.time() - self.timestamp) / 30.0)

        return (
            0.2 * access_score +
            0.4 * quality_score +
            0.4 * age_score
        )


class FeatureCache(LoggerMixin):
    """
    Caché LRU para features extraídos.

    Características:
    - Política LRU (Least Recently Used)
    - Expiración por tiempo
    - Límite de tamaño
    - Estadísticas de uso

    Attributes:
        max_size: Número máximo de entradas
        max_age_seconds: Tiempo máximo de vida de una entrada
    """

    def __init__(
        self,
        max_size: int = 500,
        max_age_seconds: float = 3.0
    ):
        """
        Inicializa el caché de features.

        Args:
            max_size: Número máximo de entradas
            max_age_seconds: Edad máxima en segundos
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds

        self._cache: OrderedDict[str, FeatureCacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        self._last_cleanup = time.time()
        self._cleanup_interval = 5.0

        self.logger.info(
            "FeatureCache inicializado",
            max_size=max_size,
            max_age_seconds=max_age_seconds
        )

    def compute_key(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> str:
        """
        Calcula una clave única para la región.

        Args:
            image: Imagen completa
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            str: Hash MD5 de la región redimensionada
        """
        try:
            x1, y1, x2, y2 = bbox
            region = image[y1:y2, x1:x2]

            if region.size > 0:
                small = cv2.resize(region, (32, 32))
                return hashlib.md5(small.tobytes()).hexdigest()

        except Exception:
            pass

        return f"{int(time.time() * 1000)}"

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Obtiene features del caché.

        Args:
            key: Clave de la región

        Returns:
            Optional[np.ndarray]: Features o None
        """
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if not entry.is_valid(self.max_age_seconds):
            self._remove(key)
            self._misses += 1
            return None

        entry.touch()
        self._cache.move_to_end(key)
        self._hits += 1

        return entry.features

    def put(
        self,
        key: str,
        features: np.ndarray,
        confidence: float,
        quality: float
    ) -> None:
        """
        Almacena features en el caché.

        Args:
            key: Clave de la región
            features: Vector de features
            confidence: Confianza de la detección
            quality: Calidad de la región (0-1)
        """
        if features is None or quality < 0.3:
            return

        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        entry = FeatureCacheEntry(features, confidence, quality)
        self._cache[key] = entry

        self._periodic_cleanup()

    def _remove(self, key: str) -> None:
        """Elimina una entrada del caché."""
        if key in self._cache:
            del self._cache[key]
            self._evictions += 1

    def _evict_oldest(self) -> None:
        """Elimina la entrada menos útil."""
        if not self._cache:
            return

        scores = {}
        for key, entry in self._cache.items():
            scores[key] = entry.get_score()

        worst_key = min(scores, key=scores.get)
        self._remove(worst_key)

    def _periodic_cleanup(self) -> None:
        """Limpieza periódica de entradas expiradas."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time

        expired_keys = [
            key for key, entry in self._cache.items()
            if not entry.is_valid(self.max_age_seconds)
        ]

        for key in expired_keys:
            self._remove(key)

        if expired_keys:
            self.logger.debug(
                "Cleaned expired entries",
                count=len(expired_keys)
            )

    def clear(self) -> None:
        """Limpia todo el caché."""
        count = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self.logger.info("Cache cleared", entries=count)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        total_requests = self._hits + self._misses

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / max(1, total_requests),
            "max_age_seconds": self.max_age_seconds,
        }

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(1, total)

    def __len__(self) -> int:
        return len(self._cache)
