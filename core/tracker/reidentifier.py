"""
Sistema de re-identificación robusto con persistencia de features
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np

from utils.logger import LoggerMixin
from models.feature_extractor import FeatureExtractor
from core.tracker.feature_cache import FeatureCacheManager
from core.constants import (
    REID_SIMILARITY_THRESHOLD,
    REID_SPATIAL_THRESHOLD,
    REID_MAX_AGE_SECONDS,
    REID_CACHE_SIZE,
    REID_MIN_FEATURES,
)

@dataclass
class ReIdentificationCandidate:
    """Candidato para re-identificación"""
    track_id: int
    features: np.ndarray
    confidence: float
    last_seen: float
    similarity_score: float = 0.0
    spatial_score: float = 0.0
    combined_score: float = 0.0


class ReIdentificationSystem(LoggerMixin):
    """
    Sistema avanzado de re-identificación

    Características:
    - Caché persistente de features
    - Matching temporal (considera cuándo se perdió el track)
    - Re-identificación progresiva (varias etapas)
    - Validación de calidad antes de re-identificar
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        max_cache_size: int = REID_CACHE_SIZE,
        max_age_seconds: float = REID_MAX_AGE_SECONDS,
        similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
        spatial_threshold: float = REID_SPATIAL_THRESHOLD,
        min_features_for_reid: int = REID_MIN_FEATURES,
    ):
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold
        self.spatial_threshold = spatial_threshold
        self.min_features_for_reid = min_features_for_reid
        self.max_age_seconds = max_age_seconds

        self.feature_cache = FeatureCacheManager(
            max_size=max_cache_size,
            max_age_seconds=max_age_seconds,
        )

        self.reid_history: List[Dict[str, Any]] = []
        self.max_history = 100

        self.stats = {
            "total_attempts": 0,
            "successful_reid": 0,
            "failed_reid": 0,
            "avg_confidence": 0.0,
            "avg_time_ms": 0.0,
        }

        self._recently_reid: Dict[int, float] = {}
        self._cooldown_seconds = 2.0

        self.logger.info(
            "ReIdentificationSystem inicializado",
            max_cache_size=max_cache_size,
            max_age_seconds=max_age_seconds,
            similarity_threshold=similarity_threshold,
            min_features_for_reid=min_features_for_reid,
        )

    def add_lost_track(self, track_id: int, features: np.ndarray, confidence: float):
        """
        Añade un track perdido al caché para posible re-identificación

        Args:
            track_id: ID del track perdido
            features: Vector de features del track
            confidence: Confianza del track
        """
        if features is None or len(features) == 0:
            return

        self.feature_cache.add(track_id, features, confidence)
        self.logger.debug(
            "Track añadido al caché de re-identificación",
            track_id=track_id,
            cache_size=self.feature_cache.size
        )

    def attempt_reidentification(
        self,
        detection: Dict[str, Any],
        frame: np.ndarray,
        current_tracks: Dict[int, Any],
        max_candidates: int = 5,
    ) -> Optional[int]:
        """
        Intenta re-identificar una detección con un track perdido

        Args:
            detection: Detección actual
            frame: Frame actual (para extraer features si es necesario)
            current_tracks: Tracks activos actuales (para evitar duplicados)
            max_candidates: Número máximo de candidatos a considerar

        Returns:
            track_id del track re-identificado o None
        """
        start_time = time.perf_counter()
        self.stats["total_attempts"] += 1

        det_centroid = detection.get("centroid")
        if det_centroid is None:
            return None

        det_features = detection.get("features")
        if det_features is None and self.feature_extractor:
            det_box = detection.get("box")
            if det_box:
                det_features = self.feature_extractor.extract_features(
                    frame, det_box, detection.get("confidence", 0.5)
                )
                if det_features is not None:
                    detection["features"] = det_features

        if det_features is None:
            self.logger.debug("No se pudieron obtener features para re-identificación")
            return None

        candidates = self.feature_cache.find_candidates(
            det_features,
            det_centroid,
            max_candidates,
            self.similarity_threshold,
            self.spatial_threshold,
        )

        if not candidates:
            self.stats["failed_reid"] += 1
            return None

        active_ids = set(current_tracks.keys())
        candidates = [c for c in candidates if c.track_id not in active_ids]

        if not candidates:
            self.stats["failed_reid"] += 1
            return None

        candidates = [
            c for c in candidates
            if c.track_id not in self._recently_reid
            or time.time() - self._recently_reid[c.track_id] > self._cooldown_seconds
        ]

        if not candidates:
            self.stats["failed_reid"] += 1
            return None

        best_candidate = candidates[0]

        if not self._validate_reidentification(best_candidate, detection):
            self.stats["failed_reid"] += 1
            return None

        track_id = best_candidate.track_id
        self._recently_reid[track_id] = time.time()
        self.stats["successful_reid"] += 1

        n = self.stats["successful_reid"] + self.stats["failed_reid"]
        self.stats["avg_confidence"] = (
            (self.stats["avg_confidence"] * (n - 1) + best_candidate.similarity_score) / n
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats["avg_time_ms"] = (
            (self.stats["avg_time_ms"] * (n - 1) + elapsed_ms) / n
        )

        self._add_history_entry(track_id, best_candidate, detection, elapsed_ms)

        self.logger.info(
            "Re-identificación exitosa",
            track_id=track_id,
            similarity=f"{best_candidate.similarity_score:.3f}",
            confidence=f"{best_candidate.confidence:.3f}",
            time_ms=f"{elapsed_ms:.1f}"
        )

        self.feature_cache.remove(track_id)

        return track_id

    def _validate_reidentification(
        self,
        candidate: ReIdentificationCandidate,
        detection: Dict[str, Any],
    ) -> bool:
        """
        Valida la calidad de una re-identificación

        Criterios:
        1. Similaridad de features suficiente
        2. Confianza de la detección adecuada
        3. Distancia espacial razonable
        4. No demasiado antigua (tiempo desde pérdida)
        """
        if candidate.similarity_score < self.similarity_threshold:
            self.logger.debug(
                "Re-identificación rechazada: baja similitud",
                similarity=candidate.similarity_score,
                threshold=self.similarity_threshold
            )
            return False

        det_confidence = detection.get("confidence", 0.0)
        if det_confidence < 0.3:
            self.logger.debug(
                "Re-identificación rechazada: baja confianza",
                confidence=det_confidence
            )
            return False

        if candidate.spatial_score < 0.3:
            self.logger.debug(
                "Re-identificación rechazada: gran distancia espacial",
                spatial_score=candidate.spatial_score
            )
            return False

        age_seconds = time.time() - candidate.last_seen
        if age_seconds > self.max_age_seconds:
            self.logger.debug(
                "Re-identificación rechazada: demasiado antiguo",
                age_seconds=age_seconds,
                max_age=self.max_age_seconds
            )
            return False

        return True

    def _add_history_entry(
        self,
        track_id: int,
        candidate: ReIdentificationCandidate,
        detection: Dict[str, Any],
        time_ms: float,
    ):
        """Añade entrada al historial de re-identificaciones"""
        entry = {
            "timestamp": time.time(),
            "track_id": track_id,
            "similarity": candidate.similarity_score,
            "confidence": candidate.confidence,
            "spatial_score": candidate.spatial_score,
            "combined_score": candidate.combined_score,
            "time_ms": time_ms,
            "detection": {
                "centroid": detection.get("centroid"),
                "confidence": detection.get("confidence", 0.0),
            }
        }

        self.reid_history.append(entry)
        if len(self.reid_history) > self.max_history:
            self.reid_history = self.reid_history[-self.max_history:]

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de re-identificación"""
        total = self.stats["successful_reid"] + self.stats["failed_reid"]

        return {
            **self.stats,
            "success_rate": self.stats["successful_reid"] / max(1, total),
            "cache_size": self.feature_cache.size,
            "cache_hit_rate": self.feature_cache.hit_rate,
            "history_size": len(self.reid_history),
            "recently_reid": len(self._recently_reid),
        }

    def clear_cache(self):
        """Limpia el caché de features"""
        self.feature_cache.clear()
        self._recently_reid.clear()
        self.logger.info("Caché de re-identificación limpiado")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_cache()
