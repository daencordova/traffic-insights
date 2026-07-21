"""
Gestor de caché para features de re-identificación
"""

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from utils.logger import LoggerMixin
from utils.geometry import euclidean_distance


class FeatureEntry:
    """Entrada en el caché de features"""

    __slots__ = ('track_id', 'features', 'confidence', 'last_seen',
                 'last_position', 'access_count')

    def __init__(
        self,
        track_id: int,
        features: np.ndarray,
        confidence: float,
        last_seen: float,
        last_position: Tuple[float, float]
    ):
        self.track_id = track_id
        self.features = features.copy() if features is not None else None
        self.confidence = confidence
        self.last_seen = last_seen
        self.last_position = last_position
        self.access_count = 0

    def touch(self) -> None:
        """Actualiza el contador de acceso y timestamp."""
        self.access_count += 1
        self.last_seen = time.time()

    def is_valid(self, max_age_seconds: float = 30.0) -> bool:
        """Verifica si la entrada sigue siendo válida."""
        return (time.time() - self.last_seen) < max_age_seconds


class FeatureCacheManager(LoggerMixin):
    """
    Gestor de caché para features con política LRU (Least Recently Used)

    Características:
    - Límite de tamaño con LRU
    - Expiración por tiempo
    - Búsqueda eficiente de candidatos
    - Estadísticas de uso
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_age_seconds: float = 30.0,
        cleanup_interval: float = 10.0,
    ):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.cleanup_interval = cleanup_interval

        self._entries: Dict[int, FeatureEntry] = {}
        self._access_order: List[int] = []

        self._hits = 0
        self._misses = 0
        self._total_entries_added = 0
        self._total_entries_removed = 0

        self._last_cleanup = time.time()

        self.logger.info(
            "FeatureCacheManager inicializado",
            max_size=max_size,
            max_age_seconds=max_age_seconds
        )

    def add(self, track_id: int, features: np.ndarray, confidence: float = 0.5):
        """
        Añade un track al caché
        """
        if features is None or len(features) == 0:
            return

        if len(self._entries) >= self.max_size:
            self._evict_oldest()

        entry = FeatureEntry(
            track_id=track_id,
            features=features,
            confidence=confidence,
            last_seen=time.time(),
            last_position=(0, 0)
        )

        if track_id in self._entries:
            old_entry = self._entries[track_id]
            entry.access_count = old_entry.access_count + 1
            self._entries[track_id] = entry
            self._total_entries_added += 1
        else:
            self._entries[track_id] = entry
            self._access_order.append(track_id)
            self._total_entries_added += 1

        if time.time() - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = time.time()

    def get(self, track_id: int) -> Optional[FeatureEntry]:
        """
        Obtiene un track del caché

        Args:
            track_id: ID del track

        Returns:
            FeatureEntry o None si no existe
        """
        entry = self._entries.get(track_id)
        if entry is None:
            self._misses += 1
            return None

        if time.time() - entry.last_seen > self.max_age_seconds:
            self.remove(track_id)
            self._misses += 1
            return None

        entry.access_count += 1
        entry.last_seen = time.time()
        self._hits += 1

        if track_id in self._access_order:
            self._access_order.remove(track_id)
        self._access_order.append(track_id)

        return entry

    def remove(self, track_id: int) -> bool:
        """Elimina un track del caché"""
        if track_id in self._entries:
            del self._entries[track_id]
            if track_id in self._access_order:
                self._access_order.remove(track_id)
            self._total_entries_removed += 1
            return True
        return False

    def find_candidates(
        self,
        query_features: np.ndarray,
        query_position: Tuple[float, float],
        max_candidates: int = 5,
        similarity_threshold: float = 0.6,
        spatial_threshold: float = 100.0,
    ) -> List:
        """
        Encuentra los mejores candidatos para re-identificación

        Args:
            query_features: Features de la detección actual
            query_position: Posición de la detección actual
            max_candidates: Número máximo de candidatos a retornar
            similarity_threshold: Umbral mínimo de similitud
            spatial_threshold: Umbral máximo de distancia espacial

        Returns:
            Lista de ReIdentificationCandidate ordenados por score
        """
        from core.tracker.reidentifier import ReIdentificationCandidate

        if query_features is None or len(query_features) == 0:
            return []

        self._cleanup_expired()

        candidates = []

        for track_id, entry in self._entries.items():
            norm_query = np.linalg.norm(query_features)
            norm_entry = np.linalg.norm(entry.features)

            if norm_query == 0 or norm_entry == 0:
                continue

            similarity = np.dot(query_features, entry.features) / (norm_query * norm_entry)

            if similarity < similarity_threshold:
                continue

            spatial_distance = euclidean_distance(query_position, entry.last_position)
            if spatial_distance > spatial_threshold:
                continue

            spatial_score = 1.0 - min(1.0, spatial_distance / spatial_threshold)

            age_seconds = time.time() - entry.last_seen
            age_score = max(0, 1.0 - age_seconds / self.max_age_seconds)

            confidence_weight = 0.7 + 0.3 * entry.confidence

            combined_score = (
                0.5 * similarity +
                0.2 * spatial_score +
                0.2 * age_score +
                0.1 * confidence_weight
            )

            candidate = ReIdentificationCandidate(
                track_id=track_id,
                features=entry.features,
                confidence=entry.confidence,
                last_seen=entry.last_seen,
                similarity_score=similarity,
                spatial_score=spatial_score,
                combined_score=combined_score,
            )
            candidates.append(candidate)

        candidates.sort(key=lambda x: x.combined_score, reverse=True)

        return candidates[:max_candidates]

    def _evict_oldest(self):
        """Elimina el entry más antiguo según LRU"""
        if not self._access_order:
            return

        oldest_id = self._access_order[0]
        self.remove(oldest_id)
        self.logger.debug("Entry eliminado por LRU", track_id=oldest_id)

    def _cleanup_expired(self):
        """Limpia entradas expiradas"""
        current_time = time.time()
        expired_ids = []

        for track_id, entry in self._entries.items():
            if current_time - entry.last_seen > self.max_age_seconds:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            self.remove(track_id)

        if expired_ids:
            self.logger.debug("Entradas expiradas eliminadas", count=len(expired_ids))

    def clear(self):
        """Limpia todo el caché"""
        count = len(self._entries)
        self._entries.clear()
        self._access_order.clear()
        self._total_entries_removed += count
        self.logger.info("Caché limpiado", entries_removed=count)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        total_requests = self._hits + self._misses

        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, total_requests),
            "total_added": self._total_entries_added,
            "total_removed": self._total_entries_removed,
            "active_entries": len(self._entries),
        }

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(1, total)
