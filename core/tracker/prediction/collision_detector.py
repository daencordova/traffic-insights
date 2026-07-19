"""
Detector de colisiones para predicción de trayectorias.

Detecta posibles colisiones entre tracks basado en
sus predicciones de trayectoria.
"""

from typing import Dict, List, Tuple, Any

import numpy as np

from utils.geometry import euclidean_distance


class CollisionDetector:
    """
    Detector de colisiones.

    Responsabilidades:
    - Detectar posibles colisiones entre tracks
    - Mantener historial de riesgos
    - Calcular niveles de riesgo

    Attributes:
        _collision_history: Historial de riesgos por track
        _threshold: Umbral de distancia para considerar colisión
        _stats: Estadísticas del detector
    """

    def __init__(self, distance_threshold: float = 30.0, history_size: int = 10):
        """
        Inicializa el detector de colisiones.

        Args:
            distance_threshold: Umbral de distancia para colisión (px)
            history_size: Tamaño del historial de riesgos
        """
        self.distance_threshold = distance_threshold
        self.history_size = history_size

        self._collision_history: Dict[int, List[float]] = {}
        self._stats = {
            "total_checks": 0,
            "collisions_detected": 0,
            "high_risk_tracks": 0,
            "avg_risk": 0.0,
            "detection_time_ms": 0.0,
        }

    def detect_collisions(
        self,
        track_id: int,
        predictions: List[Tuple[float, float]],
        all_predictions: Dict[int, List[Tuple[float, float]]]
    ) -> float:
        """
        Detecta posibles colisiones para un track.

        Args:
            track_id: ID del track
            predictions: Predicciones del track
            all_predictions: Predicciones de todos los tracks

        Returns:
            float: Riesgo de colisión (0-1)
        """
        import time
        start_time = time.perf_counter()

        if len(predictions) < 2:
            return 0.0

        if track_id not in self._collision_history:
            self._collision_history[track_id] = []

        collision_count = 0
        total_checks = 0

        for other_id, other_preds in all_predictions.items():
            if other_id == track_id:
                continue

            if len(other_preds) < 2:
                continue

            horizon = min(len(predictions), len(other_preds))

            for i in range(horizon):
                dist = euclidean_distance(predictions[i], other_preds[i])
                if dist < self.distance_threshold:
                    collision_count += 1
                total_checks += 1

        if total_checks > 0:
            risk = collision_count / total_checks
        else:
            risk = 0.0

        self._collision_history[track_id].append(risk)
        if len(self._collision_history[track_id]) > self.history_size:
            self._collision_history[track_id] = self._collision_history[track_id][-self.history_size:]

        self._stats["total_checks"] += 1
        if risk > 0.3:
            self._stats["collisions_detected"] += 1

        avg_risk = np.mean(self._collision_history[track_id]) if self._collision_history[track_id] else 0
        self._stats["avg_risk"] = (
            (self._stats["avg_risk"] * (self._stats["total_checks"] - 1) + avg_risk) /
            self._stats["total_checks"]
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats["detection_time_ms"] = (
            (self._stats["detection_time_ms"] * (self._stats["total_checks"] - 1) + elapsed_ms) /
            self._stats["total_checks"]
        )

        return avg_risk

    def get_risk(self, track_id: int) -> float:
        """
        Obtiene el riesgo actual de un track.

        Args:
            track_id: ID del track

        Returns:
            float: Riesgo de colisión (0-1)
        """
        if track_id not in self._collision_history or not self._collision_history[track_id]:
            return 0.0
        return float(np.mean(self._collision_history[track_id]))

    def get_high_risk_tracks(self, threshold: float = 0.5) -> List[int]:
        """
        Obtiene tracks con alto riesgo de colisión.

        Args:
            threshold: Umbral de riesgo

        Returns:
            List[int]: IDs de tracks con alto riesgo
        """
        high_risk = []
        for track_id, history in self._collision_history.items():
            if history and np.mean(history) > threshold:
                high_risk.append(track_id)
        return high_risk

    def get_risk_level(self, risk: float) -> str:
        """
        Obtiene el nivel de riesgo.

        Args:
            risk: Valor de riesgo (0-1)

        Returns:
            str: Nivel de riesgo ('low', 'medium', 'high', 'critical')
        """
        if risk < 0.3:
            return "low"
        elif risk < 0.5:
            return "medium"
        elif risk < 0.7:
            return "high"
        else:
            return "critical"

    def clear_track(self, track_id: int) -> None:
        """Elimina el historial de un track."""
        if track_id in self._collision_history:
            del self._collision_history[track_id]

    def clear_all(self) -> None:
        """Limpia todos los historiales."""
        self._collision_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del detector."""
        return {
            **self._stats,
            "active_tracks": len(self._collision_history),
            "high_risk_count": len(self.get_high_risk_tracks()),
        }
