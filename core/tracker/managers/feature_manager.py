"""
Gestor de features para tracking.

Maneja la extracción, almacenamiento y comparación de features
para re-identificación y aprendizaje en línea.
"""

from typing import Optional, Dict, Any, List

import numpy as np

from models.feature_extractor import FeatureExtractor
from core.tracker.feature_cache import FeatureCacheManager
from utils.logger import LoggerMixin


class FeatureManager(LoggerMixin):
    """
    Gestor centralizado de features.

    Responsabilidades:
    - Extracción de features de detecciones
    - Almacenamiento en caché para re-identificación
    - Comparación de features para matching
    - Gestión de límites de memoria

    Attributes:
        feature_extractor: Extractor de features
        feature_cache: Caché de features
        use_features: Si las features están habilitadas
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        max_cache_size: int = 1000,
        max_age_seconds: float = 30.0,
        similarity_threshold: float = 0.6,
        spatial_threshold: float = 100.0
    ):
        self.feature_extractor = feature_extractor
        self.use_features = feature_extractor is not None

        self.feature_cache = FeatureCacheManager(
            max_size=max_cache_size,
            max_age_seconds=max_age_seconds
        )

        self.similarity_threshold = similarity_threshold
        self.spatial_threshold = spatial_threshold

        self._stats = {
            "features_extracted": 0,
            "features_cached": 0,
            "feature_comparisons": 0,
            "similar_matches": 0,
        }

        self.logger.info(
            "FeatureManager inicializado",
            use_features=self.use_features,
            cache_size=max_cache_size,
            max_age_seconds=max_age_seconds
        )

    def extract_features(
        self,
        frame: np.ndarray,
        bbox: tuple,
        confidence: float = 0.5
    ) -> Optional[np.ndarray]:
        """
        Extrae features de una región de imagen.

        Args:
            frame: Imagen completa
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Confianza de la detección

        Returns:
            Vector de features o None
        """
        if not self.use_features or self.feature_extractor is None:
            return None

        try:
            features = self.feature_extractor.extract_features(
                frame, bbox, confidence
            )
            if features is not None:
                self._stats["features_extracted"] += 1
            return features
        except Exception as e:
            self.logger.debug(
                "Error extrayendo features",
                error=str(e)
            )
            return None

    def cache_features(
        self,
        track_id: int,
        features: np.ndarray,
        confidence: float = 0.5
    ) -> None:
        """
        Almacena features en el caché.

        Args:
            track_id: ID del track
            features: Vector de features
            confidence: Confianza del track
        """
        if features is None:
            return

        self.feature_cache.add(track_id, features, confidence)
        self._stats["features_cached"] += 1

    def find_similar_tracks(
        self,
        features: np.ndarray,
        position: tuple,
        max_candidates: int = 5,
        exclude_tracks: Optional[List[int]] = None
    ) -> List[Any]:
        """
        Encuentra tracks similares para re-identificación.

        Args:
            features: Features de consulta
            position: Posición actual
            max_candidates: Número máximo de candidatos
            exclude_tracks: IDs de tracks a excluir

        Returns:
            Lista de candidatos ordenados por similitud
        """
        if features is None:
            return []

        exclude_tracks = exclude_tracks or []

        candidates = self.feature_cache.find_candidates(
            features,
            position,
            max_candidates,
            self.similarity_threshold,
            self.spatial_threshold
        )

        candidates = [
            c for c in candidates
            if c.track_id not in exclude_tracks
        ]

        self._stats["feature_comparisons"] += len(candidates)
        return candidates

    def compare_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        Compara dos vectores de features.

        Args:
            features1: Primer vector
            features2: Segundo vector

        Returns:
            Similaridad coseno (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0

        try:
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(features1, features2) / (norm1 * norm2)
            similarity = max(0.0, min(1.0, similarity))

            self._stats["feature_comparisons"] += 1
            if similarity > self.similarity_threshold:
                self._stats["similar_matches"] += 1

            return similarity
        except Exception as e:
            self.logger.debug(
                "Error comparando features",
                error=str(e)
            )
            return 0.0

    def clear_cache(self) -> None:
        """Limpia el caché de features."""
        self.feature_cache.clear()
        self.logger.info("Caché de features limpiado")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor."""
        return {
            **self._stats,
            "cache_size": self.feature_cache.size,
            "cache_hit_rate": self.feature_cache.hit_rate,
            "use_features": self.use_features,
            "similarity_threshold": self.similarity_threshold,
        }

    @property
    def is_available(self) -> bool:
        """Verifica si las features están disponibles."""
        return self.use_features and self.feature_extractor is not None
