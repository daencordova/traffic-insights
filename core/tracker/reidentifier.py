"""
Sistema de re-identificación robusto con persistencia de features.

Este módulo implementa un sistema de re-identificación que permite
recuperar objetos perdidos utilizando features visuales almacenados
en caché.

Características principales:
- Caché persistente de features con LRU
- Matching temporal (considera cuándo se perdió el track)
- Re-identificación progresiva (varias etapas)
- Validación de calidad antes de re-identificar
- Cooldown para evitar re-identificaciones repetidas
- Historial de re-identificaciones para depuración
"""

import time
from typing import Dict, List, Optional, Any, Tuple
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
    """Candidato para re-identificación."""
    track_id: int
    features: np.ndarray
    confidence: float
    last_seen: float
    similarity_score: float = 0.0
    spatial_score: float = 0.0
    combined_score: float = 0.0


class ReIDSystem(LoggerMixin):
    """
    Sistema avanzado de re-identificación.

    Este sistema permite recuperar objetos perdidos utilizando
    features visuales almacenados en caché durante el tracking.

    Características:
    - Caché persistente de features
    - Matching temporal (considera cuándo se perdió el track)
    - Re-identificación progresiva (varias etapas)
    - Validación de calidad antes de re-identificar
    - Cooldown para evitar re-identificaciones repetidas

    Attributes:
        feature_extractor: Extractor de features visuales.
        similarity_threshold: Umbral de similitud para re-identificación.
        spatial_threshold: Umbral de distancia espacial.
        min_features_for_reid: Mínimo de features para re-identificar.
        max_age_seconds: Edad máxima de un track para re-identificación.
        feature_cache: Caché de features.
        reid_history: Historial de re-identificaciones.
        _recently_reid: Tracks re-identificados recientemente (cooldown).
        _cooldown_seconds: Tiempo de cooldown entre re-identificaciones.

    Example:
        >>> reid = ReIDSystem()
        >>> reid.add_lost_track(5, features, 0.8)
        >>> track_id = reid.attempt_reidentification(detection, frame, current_tracks)
        >>> if track_id is not None:
        ...     print(f"Track {track_id} re-identificado")
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
        """
        Inicializa el sistema de re-identificación.

        Args:
            feature_extractor: Extractor de features visuales.
            max_cache_size: Tamaño máximo del caché de features.
            max_age_seconds: Edad máxima de un track en caché.
            similarity_threshold: Umbral de similitud (0-1).
            spatial_threshold: Umbral de distancia espacial (píxeles).
            min_features_for_reid: Mínimo de features requeridos.
        """
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
            "ReIDSystem inicializado",
            max_cache_size=max_cache_size,
            max_age_seconds=max_age_seconds,
            similarity_threshold=similarity_threshold,
            min_features_for_reid=min_features_for_reid,
        )

    def add_lost_track(self, track_id: int, features: np.ndarray, confidence: float) -> None:
        """
        Añade un track perdido al caché para posible re-identificación.

        Args:
            track_id: ID del track perdido.
            features: Vector de features del track.
            confidence: Confianza del track.
        """
        if features is None or len(features) == 0:
            return

        self.feature_cache.add(track_id, features, confidence)
        self.logger.debug(
            "Track añadido al caché de re-identificación",
            track_id=track_id,
            cache_size=self.feature_cache.size
        )

    def clear_cache(self) -> None:
        """Limpia el caché de features y el historial de re-identificaciones."""
        self.feature_cache.clear()
        self._recently_reid.clear()
        self.logger.info("Caché de re-identificación limpiado")

    def attempt_reidentification(
        self,
        detection: Dict[str, Any],
        frame: np.ndarray,
        current_tracks: Dict[int, Any],
        max_candidates: int = 5,
    ) -> Optional[int]:
        """
        Intenta re-identificar una detección con un track perdido.

        Args:
            detection: Detección actual.
            frame: Frame actual (para extraer features si es necesario).
            current_tracks: Tracks activos actuales (para evitar duplicados).
            max_candidates: Número máximo de candidatos a considerar.

        Returns:
            Optional[int]: ID del track re-identificado o None.
        """
        start_time = time.perf_counter()
        self.stats["total_attempts"] += 1

        det_centroid = detection.get("centroid")
        if det_centroid is None:
            return None

        det_features = self._get_features(detection, frame)
        if det_features is None:
            self.logger.debug("No se pudieron obtener features para re-identificación")
            return None

        candidates = self._find_candidates(det_features, det_centroid, max_candidates)

        if not candidates:
            self.stats["failed_reid"] += 1
            return None

        candidates = self._filter_candidates(candidates, current_tracks)

        if not candidates:
            self.stats["failed_reid"] += 1
            return None

        best_candidate = candidates[0]

        if not self._validate_reidentification(best_candidate, detection):
            self.stats["failed_reid"] += 1
            return None

        track_id = self._perform_reidentification(best_candidate, detection)

        if track_id is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_success(best_candidate, elapsed_ms)
            self._add_history_entry(track_id, best_candidate, detection, elapsed_ms)

            self.logger.info(
                "Re-identificación exitosa",
                track_id=track_id,
                similarity=f"{best_candidate.similarity_score:.3f}",
                confidence=f"{best_candidate.confidence:.3f}",
                time_ms=f"{elapsed_ms:.1f}"
            )

            return track_id

        return None

    def _get_features(
        self,
        detection: Dict[str, Any],
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Obtiene features de una detección.

        Args:
            detection: Detección actual.
            frame: Frame actual.

        Returns:
            Optional[np.ndarray]: Vector de features o None.
        """
        det_features = detection.get("features")

        if det_features is None and self.feature_extractor is not None:
            det_box = detection.get("box")
            if det_box:
                det_features = self.feature_extractor.extract_features(
                    frame, det_box, detection.get("confidence", 0.5)
                )
                if det_features is not None:
                    detection["features"] = det_features

        return det_features

    def _find_candidates(
        self,
        features: np.ndarray,
        centroid: Tuple[float, float],
        max_candidates: int
    ) -> List[ReIdentificationCandidate]:
        """
        Busca candidatos en el caché de features.

        Args:
            features: Features de la detección.
            centroid: Centroide de la detección.
            max_candidates: Número máximo de candidatos.

        Returns:
            List[ReIdentificationCandidate]: Candidatos encontrados.
        """
        return self.feature_cache.find_candidates(
            features,
            centroid,
            max_candidates,
            self.similarity_threshold,
            self.spatial_threshold,
        )

    def _filter_candidates(
        self,
        candidates: List[ReIdentificationCandidate],
        current_tracks: Dict[int, Any]
    ) -> List[ReIdentificationCandidate]:
        """
        Filtra candidatos según criterios adicionales.

        Args:
            candidates: Lista de candidatos.
            current_tracks: Tracks activos actuales.

        Returns:
            List[ReIdentificationCandidate]: Candidatos filtrados.
        """
        active_ids = set(current_tracks.keys())
        candidates = [c for c in candidates if c.track_id not in active_ids]

        if not candidates:
            return []

        current_time = time.time()
        candidates = [
            c for c in candidates
            if c.track_id not in self._recently_reid
            or current_time - self._recently_reid[c.track_id] > self._cooldown_seconds
        ]

        return candidates

    def _validate_reidentification(
        self,
        candidate: ReIdentificationCandidate,
        detection: Dict[str, Any]
    ) -> bool:
        """
        Valida la calidad de una re-identificación.

        Criterios:
        1. Similaridad de features suficiente
        2. Confianza de la detección adecuada
        3. Distancia espacial razonable
        4. No demasiado antigua (tiempo desde pérdida)

        Args:
            candidate: Candidato a validar.
            detection: Detección actual.

        Returns:
            bool: True si el candidato es válido.
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

    def _perform_reidentification(
        self,
        candidate: ReIdentificationCandidate,
        detection: Dict[str, Any]
    ) -> Optional[int]:
        """
        Ejecuta la re-identificación de un track.

        Args:
            candidate: Candidato seleccionado.
            detection: Detección actual.

        Returns:
            Optional[int]: ID del track re-identificado o None.
        """
        track_id = candidate.track_id

        self._recently_reid[track_id] = time.time()

        self.feature_cache.remove(track_id)

        return track_id

    def _record_success(self, candidate: ReIdentificationCandidate, time_ms: float) -> None:
        """
        Registra una re-identificación exitosa en estadísticas.

        Args:
            candidate: Candidato seleccionado.
            time_ms: Tiempo de ejecución en milisegundos.
        """
        self.stats["successful_reid"] += 1

        n = self.stats["successful_reid"] + self.stats["failed_reid"]
        self.stats["avg_confidence"] = (
            (self.stats["avg_confidence"] * (n - 1) + candidate.similarity_score) / n
        )

        self.stats["avg_time_ms"] = (
            (self.stats["avg_time_ms"] * (n - 1) + time_ms) / n
        )

    def _add_history_entry(
        self,
        track_id: int,
        candidate: ReIdentificationCandidate,
        detection: Dict[str, Any],
        time_ms: float
    ) -> None:
        """
        Añade entrada al historial de re-identificaciones.

        Args:
            track_id: ID del track re-identificado.
            candidate: Candidato seleccionado.
            detection: Detección actual.
            time_ms: Tiempo de ejecución en milisegundos.
        """
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
        """
        Obtiene estadísticas del sistema de re-identificación.

        Returns:
            Dict[str, Any]: Estadísticas del sistema.
        """
        total = self.stats["successful_reid"] + self.stats["failed_reid"]

        return {
            **self.stats,
            "success_rate": self.stats["successful_reid"] / max(1, total),
            "cache_size": self.feature_cache.size,
            "cache_hit_rate": self.feature_cache.hit_rate,
            "history_size": len(self.reid_history),
            "recently_reid": len(self._recently_reid),
        }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de re-identificaciones.

        Args:
            limit: Número máximo de entradas a retornar.

        Returns:
            List[Dict[str, Any]]: Historial de re-identificaciones.
        """
        return self.reid_history[-limit:] if self.reid_history else []

    def get_last_successful_reid(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene la última re-identificación exitosa.

        Returns:
            Optional[Dict[str, Any]]: Último evento exitoso.
        """
        for entry in reversed(self.reid_history):
            if entry.get("track_id") is not None:
                return entry
        return None

    def is_track_in_cooldown(self, track_id: int) -> bool:
        """
        Verifica si un track está en período de cooldown.

        Args:
            track_id: ID del track.

        Returns:
            bool: True si está en cooldown.
        """
        if track_id not in self._recently_reid:
            return False

        return time.time() - self._recently_reid[track_id] < self._cooldown_seconds

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - limpia caché."""
        self.clear_cache()
