"""
Actualizador de tracks con lógica de predicción y corrección.

Maneja la actualización de tracks individuales incluyendo
la predicción de posición con Kalman y la corrección con
nuevas detecciones.
"""

from typing import Dict, Any

import numpy as np

from models.track_state import TrackState
from models.kalman import EnhancedKalmanFilter
from utils.logger import LoggerMixin
from utils.geometry import euclidean_distance


class TrackUpdater(LoggerMixin):
    """
    Actualizador de tracks con soporte para Kalman optimizado.

    Responsabilidades:
    - Predicción de posición usando filtro de Kalman
    - Corrección de estado con nuevas detecciones
    - Cálculo de métricas de movimiento (velocidad, aceleración)
    - Validación de consistencia de movimiento

    Attributes:
        use_kalman: Si se debe usar filtro de Kalman
        use_optimized_kalman: Si se debe usar la versión optimizada
        max_speed_change: Umbral máximo de cambio de velocidad
    """

    def __init__(
        self,
        use_kalman: bool = True,
        use_optimized_kalman: bool = False,
        max_speed_change: float = 50.0
    ):
        self.use_kalman = use_kalman
        self.use_optimized_kalman = use_optimized_kalman
        self.max_speed_change = max_speed_change

        self._stats = {
            "total_predictions": 0,
            "total_corrections": 0,
            "total_kalman_inits": 0,
            "kalman_errors": 0,
        }

        self.logger.info(
            "TrackUpdater inicializado",
            use_kalman=use_kalman,
            optimized=use_optimized_kalman
        )

    def predict_position(self, track: TrackState) -> None:
        """
        Predice la siguiente posición del track.

        Args:
            track: Track a predecir
        """
        if not self.use_kalman or track.kalman_filter is None:
            track.predicted_centroid = track.centroid
            return

        try:
            pred = track.kalman_filter.predict()
            track.predicted_centroid = (int(pred[0]), int(pred[1]))
            self._stats["total_predictions"] += 1
        except Exception as e:
            self._stats["kalman_errors"] += 1
            self.logger.debug(
                "Error en predicción Kalman",
                track_id=track.track_id,
                error=str(e)
            )
            track.predicted_centroid = track.centroid

    def correct_position(self, track: TrackState, detection: Dict[str, Any]) -> None:
        """
        Corrige la posición del track con una nueva detección.

        Args:
            track: Track a corregir
            detection: Nueva detección
        """
        if not self.use_kalman or track.kalman_filter is None:
            return

        try:
            measurement = np.array([
                detection.get("centroid", track.centroid)[0],
                detection.get("centroid", track.centroid)[1]
            ])
            track.kalman_filter.correct(measurement)
            self._stats["total_corrections"] += 1
        except Exception as e:
            self._stats["kalman_errors"] += 1
            self.logger.debug(
                "Error en corrección Kalman",
                track_id=track.track_id,
                error=str(e)
            )

    def init_kalman(self, track: TrackState) -> None:
        """
        Inicializa el filtro de Kalman para un track.

        Args:
            track: Track a inicializar
        """
        if not self.use_kalman:
            return

        try:
            if self.use_optimized_kalman:
                from models.kalman_optimized import OptimizedKalmanFilter
                kf = OptimizedKalmanFilter()
            else:
                kf = EnhancedKalmanFilter()

            kf.init(track.centroid[0], track.centroid[1])
            track.kalman_filter = kf
            self._stats["total_kalman_inits"] += 1
        except Exception as e:
            self._stats["kalman_errors"] += 1
            self.logger.warning(
                "Error inicializando Kalman",
                track_id=track.track_id,
                error=str(e)
            )
            track.kalman_filter = None

    def update_motion_metrics(self, track: TrackState) -> None:
        """
        Actualiza las métricas de movimiento del track.

        Args:
            track: Track a actualizar
        """
        if len(track.history) >= 2:
            prev = track.history[-2]
            curr = track.history[-1]

            track.velocity = (curr[0] - prev[0], curr[1] - prev[1])

            if len(track.history) >= 3:
                prev_prev = track.history[-3]
                prev_vel = (
                    prev[0] - prev_prev[0],
                    prev[1] - prev_prev[1]
                )
                track.acceleration = (
                    track.velocity[0] - prev_vel[0],
                    track.velocity[1] - prev_vel[1]
                )

    def validate_motion_consistency(
        self,
        track: TrackState,
        detection: Dict[str, Any]
    ) -> bool:
        """
        Valida que el movimiento sea consistente.

        Args:
            track: Track a validar
            detection: Nueva detección

        Returns:
            bool: True si el movimiento es consistente
        """
        if len(track.history) < 2:
            return True

        new_centroid = detection.get("centroid")
        if new_centroid is None:
            return True

        last_pos = track.history[-1]
        expected_speed = np.sqrt(
            track.velocity[0] ** 2 + track.velocity[1] ** 2
        )

        actual_speed = euclidean_distance(last_pos, new_centroid)

        if expected_speed > 0:
            speed_ratio = actual_speed / expected_speed
            if speed_ratio > 5.0:
                self.logger.debug(
                    "Movimiento inconsistente detectado",
                    track_id=track.track_id,
                    speed_ratio=speed_ratio
                )
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del actualizador."""
        return {
            **self._stats,
            "kalman_enabled": self.use_kalman,
            "kalman_optimized": self.use_optimized_kalman,
            "kalman_success_rate": (
                1 - self._stats["kalman_errors"] / max(1, self._stats["total_predictions"])
            ),
        }
