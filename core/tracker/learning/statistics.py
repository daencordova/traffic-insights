"""
Estadísticas de features para aprendizaje en línea.

Mantiene las estadísticas de features por track.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import time

import numpy as np


@dataclass
class FeatureStatistics:
    """
    Estadísticas de features para aprendizaje en línea.

    Attributes:
        mean_features: Vector de features promedio
        covariance: Matriz de covarianza incremental (opcional)
        n_samples: Número total de muestras procesadas
        feature_history: Historial de features recientes
        timestamps: Timestamps de las actualizaciones
        confidence_history: Historial de confianzas
        last_update_time: Timestamp de la última actualización
        total_updates: Número total de actualizaciones
        concept_drift_detected: Flag de detección de cambio de concepto
        quality_score: Puntuación de calidad del modelo
    """
    mean_features: np.ndarray
    covariance: Optional[np.ndarray] = None
    n_samples: int = 0
    feature_history: deque = field(default_factory=lambda: deque(maxlen=50))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_update_time: float = field(default_factory=time.time)
    total_updates: int = 0
    concept_drift_detected: bool = False
    quality_score: float = 0.0

    def add_sample(
        self,
        features: np.ndarray,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Añade una nueva muestra a las estadísticas.

        Args:
            features: Vector de features
            confidence: Confianza de la muestra
            timestamp: Timestamp de la muestra (opcional)
        """
        if timestamp is None:
            timestamp = time.time()

        self.feature_history.append(features.copy())
        self.confidence_history.append(confidence)
        self.timestamps.append(timestamp)
        self.n_samples += 1
        self.total_updates += 1
        self.last_update_time = timestamp

    def get_average_feature(self) -> Optional[np.ndarray]:
        """
        Obtiene el feature promedio del historial.

        Returns:
            Optional[np.ndarray]: Feature promedio o None
        """
        if not self.feature_history:
            return None

        features_list = list(self.feature_history)
        avg_feature = np.mean(features_list, axis=0)
        norm = np.linalg.norm(avg_feature)
        if norm > 0:
            avg_feature = avg_feature / norm
        return avg_feature

    def get_average_confidence(self) -> float:
        """
        Obtiene la confianza promedio del historial.

        Returns:
            float: Confianza promedio
        """
        if not self.confidence_history:
            return 0.0
        return float(np.mean(list(self.confidence_history)))

    def get_history_length(self) -> int:
        """Obtiene la longitud del historial."""
        return len(self.feature_history)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "n_samples": self.n_samples,
            "total_updates": self.total_updates,
            "last_update_time": self.last_update_time,
            "concept_drift_detected": self.concept_drift_detected,
            "quality_score": self.quality_score,
            "history_length": self.get_history_length(),
            "avg_confidence": self.get_average_confidence(),
        }
