"""
Detector de cambios de apariencia (concept drift).

Detecta cuándo la apariencia de un objeto ha cambiado significativamente.
"""

from typing import Dict, Any
from collections import deque
import numpy as np

from core.tracker.learning.statistics import FeatureStatistics


class ConceptDriftDetector:
    """
    Detector de cambios de apariencia.

    Responsabilidades:
    - Detectar cambios significativos en la apariencia
    - Mantener historial de similitudes
    - Calcular umbrales adaptativos

    Attributes:
        threshold: Umbral de similitud para detectar drift
        min_samples: Mínimo de muestras para detección
        _similarity_history: Historial de similitudes
        _stats: Estadísticas del detector
    """

    def __init__(
        self,
        threshold: float = 0.35,
        min_samples: int = 5,
        history_size: int = 50
    ):
        """
        Inicializa el detector de concept drift.

        Args:
            threshold: Umbral de similitud para detectar drift
            min_samples: Mínimo de muestras para detección
            history_size: Tamaño del historial de similitudes
        """
        self.threshold = threshold
        self.min_samples = min_samples
        self.history_size = history_size

        self._similarity_history: Dict[int, deque] = {}
        self._stats = {
            "total_drifts_detected": 0,
            "total_checks": 0,
            "avg_similarity": 0.0,
            "detection_rate": 0.0,
        }

    def detect_drift(
        self,
        track_id: int,
        stats: FeatureStatistics,
        features: np.ndarray
    ) -> bool:
        """
        Detecta si hay un cambio de apariencia.

        Args:
            track_id: ID del track
            stats: Estadísticas del track
            features: Nuevos features

        Returns:
            bool: True si se detectó drift
        """
        self._stats["total_checks"] += 1

        if stats.n_samples < self.min_samples:
            return False

        norm_mean = np.linalg.norm(stats.mean_features)
        norm_feat = np.linalg.norm(features)

        if norm_mean < 1e-8 or norm_feat < 1e-8:
            return False

        similarity = np.dot(stats.mean_features, features) / (norm_mean * norm_feat)
        similarity = max(0.0, min(1.0, similarity))

        if track_id not in self._similarity_history:
            self._similarity_history[track_id] = deque(maxlen=self.history_size)
        self._similarity_history[track_id].append(similarity)

        avg_similarity = np.mean(self._similarity_history[track_id])
        self._stats["avg_similarity"] = (
            (self._stats["avg_similarity"] * (self._stats["total_checks"] - 1) + avg_similarity) /
            self._stats["total_checks"]
        )

        if similarity < self.threshold and stats.n_samples > self.min_samples * 2:
            self._stats["total_drifts_detected"] += 1
            stats.concept_drift_detected = True
            return True

        return False

    def get_similarity_history(self, track_id: int) -> list:
        """
        Obtiene el historial de similitudes de un track.

        Args:
            track_id: ID del track

        Returns:
            list: Historial de similitudes
        """
        return list(self._similarity_history.get(track_id, []))

    def get_average_similarity(self, track_id: int) -> float:
        """
        Obtiene la similitud promedio de un track.

        Args:
            track_id: ID del track

        Returns:
            float: Similitud promedio
        """
        history = self._similarity_history.get(track_id, [])
        if not history:
            return 0.0
        return float(np.mean(list(history)))

    def get_drift_rate(self) -> float:
        """
        Obtiene la tasa de detección de drift.

        Returns:
            float: Tasa de detección (0-1)
        """
        total = self._stats["total_checks"]
        if total == 0:
            return 0.0
        return self._stats["total_drifts_detected"] / total

    def clear_track(self, track_id: int) -> None:
        """Elimina el historial de un track."""
        if track_id in self._similarity_history:
            del self._similarity_history[track_id]

    def clear_all(self) -> None:
        """Limpia todos los historiales."""
        self._similarity_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del detector."""
        return {
            **self._stats,
            "active_tracks": len(self._similarity_history),
            "drift_rate": self.get_drift_rate(),
            "threshold": self.threshold,
            "min_samples": self.min_samples,
        }

    def reset(self) -> None:
        """Reinicia el detector."""
        self.clear_all()
        self._stats = {
            "total_drifts_detected": 0,
            "total_checks": 0,
            "avg_similarity": 0.0,
            "detection_rate": 0.0,
        }
