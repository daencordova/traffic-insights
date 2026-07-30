"""Sistema de caché para detecciones de objetos.

Implementa un caché LRU con gestión de memoria para almacenar
resultados de detección y evitar procesamiento redundante.
"""

from collections import OrderedDict
import hashlib
import time
from typing import Any

import cv2
import numpy as np

from core.constants.pipeline import (
    CACHE_CLEANUP_THRESHOLD,
    DEFAULT_CACHE_SIZE,
    MAX_CACHE_MEMORY_MB,
)
from utils.logger import LoggerMixin

DetectionList = list[dict[str, Any]]
FrameHash = str


class CacheEntry:
    """Entrada en el caché de detecciones.

    Attributes:
        detections: Lista de detecciones almacenadas.
        timestamp: Momento de creación de la entrada.
        size: Tamaño estimado en bytes de la entrada.
        access_count: Número de veces que se ha accedido a la entrada.
    """

    __slots__ = ("detections", "timestamp", "size", "access_count")

    def __init__(self, detections: DetectionList):
        self.detections = detections
        self.timestamp = time.time()
        self.size = len(detections) * 4 * 4
        self.access_count = 0

    def touch(self) -> None:
        """Actualiza el contador de acceso y timestamp.

        Este método se llama cada vez que se accede a la entrada,
        manteniendo actualizada su posición en el orden LRU.
        """
        self.access_count += 1
        self.timestamp = time.time()

    def is_valid(self, max_age: float = 3.0) -> bool:
        """Verifica si la entrada sigue siendo válida.

        Args:
            max_age: Edad máxima en segundos antes de expirar (default: 3.0).

        Returns:
            bool: True si la entrada es válida y no ha expirado.
        """
        return (time.time() - self.timestamp) < max_age


class DetectionCache(LoggerMixin):
    """Caché LRU para detecciones de objetos.

    Características:
        - Política LRU (Least Recently Used)
        - Límite de memoria configurable
        - Expiración por tiempo
        - Estadísticas de uso
        - Thread-safe

    Attributes:
        max_size: Número máximo de entradas en caché.
        max_age_seconds: Tiempo máximo de vida de una entrada.
        max_memory_mb: Memoria máxima permitida para el caché.
        cleanup_threshold: Umbral para limpieza automática.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_SIZE,
        max_age_seconds: float = 3.0,
        max_memory_mb: int = MAX_CACHE_MEMORY_MB,
        cleanup_threshold: float = CACHE_CLEANUP_THRESHOLD,
    ):
        """Inicializa el caché de detecciones.

        Args:
            max_size: Número máximo de entradas
            max_age_seconds: Edad máxima de una entrada en segundos
            max_memory_mb: Memoria máxima en MB
            cleanup_threshold: Umbral para limpieza automática
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold

        self._cache: OrderedDict[FrameHash, CacheEntry] = OrderedDict()
        self._memory_usage: float = 0.0

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 5.0

        self.logger.info(
            "DetectionCache inicializado",
            max_size=max_size,
            max_age_seconds=max_age_seconds,
            max_memory_mb=max_memory_mb,
        )

    def compute_key(self, frame: np.ndarray) -> str:
        """Calcula una clave única para el frame.

        Args:
            frame: Imagen a procesar.

        Returns:
            str: Hash MD5 del frame redimensionado a 32x32.

        Note:
            El redimensionamiento a 32x32 asegura que frames similares
            produzcan la misma clave, aumentando la tasa de aciertos.
        """
        try:
            small = cv2.resize(frame, (32, 32))
            return hashlib.md5(small.tobytes()).hexdigest()
        except Exception:
            return str(time.perf_counter())

    def get(self, key: str) -> DetectionList | None:
        """Obtiene detecciones del caché.

        Args:
            key: Clave del frame.

        Returns:
            Optional[DetectionList]: Detecciones cacheadas o None si no existen.

        Note:
            Esta operación actualiza el orden LRU de la entrada.
            Las entradas expiradas se eliminan automáticamente.
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

        self.logger.debug("Cache hit", key=key[:8], detections=len(entry.detections))

        return entry.detections

    def put(self, key: str, detections: DetectionList) -> None:
        """Almacena detecciones en el caché.

        Args:
            key: Clave del frame.
            detections: Lista de detecciones a almacenar.

        Note:
            Si el caché está lleno, se elimina la entrada más antigua (LRU).
            Si la memoria excede el límite, se realiza una limpieza agresiva.
        """
        if not detections:
            return

        entry_size = len(detections) * 4 * 4
        if self._memory_usage + entry_size > self.max_memory_mb * 1024 * 1024:
            self._cleanup(aggressive=True)

        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        entry = CacheEntry(detections)
        self._cache[key] = entry
        self._memory_usage += entry.size

        self._periodic_cleanup()

        self.logger.debug(
            "Cache put",
            key=key[:8],
            detections=len(detections),
            cache_size=len(self._cache),
            memory_mb=self._memory_usage / (1024 * 1024),
        )

    def _remove(self, key: str) -> None:
        """Elimina una entrada del caché.

        Args:
            key: Clave de la entrada a eliminar.

        Note:
            Este método actualiza el contador de evictions y libera memoria.
        """
        entry = self._cache.pop(key, None)
        if entry:
            self._memory_usage -= entry.size
            self._evictions += 1

    def _evict_oldest(self) -> None:
        """Elimina la entrada más antigua (LRU).

        Note:
            Se elimina la primera entrada del OrderedDict,
            que corresponde a la menos recientemente usada.
        """
        if not self._cache:
            return

        oldest_key = next(iter(self._cache))
        self._remove(oldest_key)
        self.logger.debug("Evicted oldest entry", key=oldest_key[:8], cache_size=len(self._cache))

    def _periodic_cleanup(self) -> None:
        """Limpieza periódica de entradas expiradas.

        Note:
            La limpieza se realiza cada `_cleanup_interval` segundos
            para evitar overhead en cada operación.
        """
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time

        expired_keys = [
            key for key, entry in self._cache.items() if not entry.is_valid(self.max_age_seconds)
        ]

        for key in expired_keys:
            self._remove(key)

        if expired_keys:
            self.logger.debug("Cleaned expired entries", count=len(expired_keys))

    def _cleanup(self, aggressive: bool = False) -> None:
        """Limpieza de caché.

        Args:
            aggressive: Si es True, limpia más entradas (50%).
                Si es False, limpia solo el 30% de las entradas.

        Note:
            La limpieza agresiva se usa cuando la memoria excede el límite.
        """
        if not self._cache:
            return

        if aggressive:
            keys_to_remove = list(self._cache.keys())[: len(self._cache) // 2]
            for key in keys_to_remove:
                self._remove(key)
            self.logger.debug("Aggressive cleanup", removed=len(keys_to_remove))
        else:
            keys_to_remove = list(self._cache.keys())[: int(len(self._cache) * 0.3)]
            for key in keys_to_remove:
                self._remove(key)
            self.logger.debug("Partial cleanup", removed=len(keys_to_remove))

    def clear(self) -> None:
        """Limpia todo el caché.

        Note:
            Reinicia todas las estadísticas y libera toda la memoria.
        """
        count = len(self._cache)
        self._cache.clear()
        self._memory_usage = 0
        self._hits = 0
        self._misses = 0
        self.logger.info("Cache cleared", entries=count)

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del caché.

        Returns:
            Dict[str, Any]: Estadísticas incluyendo:
                - size: Número actual de entradas
                - max_size: Tamaño máximo configurado
                - memory_usage_mb: Memoria usada en MB
                - max_memory_mb: Memoria máxima configurada
                - hits: Número de aciertos
                - misses: Número de fallos
                - evictions: Número de entradas eliminadas
                - hit_rate: Tasa de aciertos (0-1)
                - max_age_seconds: Edad máxima configurada

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        total_requests = self._hits + self._misses

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": self._memory_usage / (1024 * 1024),
            "max_memory_mb": self.max_memory_mb,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / max(1, total_requests),
            "max_age_seconds": self.max_age_seconds,
        }

    @property
    def size(self) -> int:
        """Número de entradas en el caché."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Tasa de aciertos del caché."""
        total = self._hits + self._misses
        return self._hits / max(1, total)

    def __len__(self) -> int:
        return len(self._cache)
