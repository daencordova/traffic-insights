"""
Detector de estados de trayectoria.

Detecta el estado de movimiento de un objeto basado en
su historial de posiciones y velocidades.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class TrajectoryState(Enum):
    """
    Estados posibles de la trayectoria de un objeto.
    """
    MOVING = "moving"
    STOPPED = "stopped"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    TURNING = "turning"
    ERRATIC = "erratic"
    UNKNOWN = "unknown"


class StateDetector:
    """
    Detector de estados de trayectoria.

    Responsabilidades:
    - Detectar el estado de movimiento de un objeto
    - Mantener historial de estados
    - Calcular métricas de movimiento

    Attributes:
        _state_history: Historial de estados por track
        _stats: Estadísticas del detector
    """

    def __init__(self):
        self._state_history: Dict[int, List[TrajectoryState]] = {}
        self._stats = {
            "total_detections": 0,
            "state_distribution": {state.value: 0 for state in TrajectoryState},
            "detection_time_ms": 0.0,
        }

    def detect_state(
        self,
        track_id: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> TrajectoryState:
        """
        Detecta el estado de trayectoria de un objeto.

        Args:
            track_id: ID del track
            positions: Array de posiciones [N, 2]
            velocities: Array de velocidades [N, 2]
            timestamps: Array de timestamps [N]

        Returns:
            TrajectoryState: Estado detectado
        """
        import time
        start_time = time.perf_counter()

        if len(positions) < 3 or len(velocities) < 2:
            state = TrajectoryState.UNKNOWN
        else:
            avg_speed = np.mean(np.linalg.norm(velocities, axis=1))

            if avg_speed < 0.5:
                state = TrajectoryState.STOPPED
            else:
                speed_changes = np.diff(np.linalg.norm(velocities, axis=1))
                avg_accel = np.mean(speed_changes) if len(speed_changes) > 0 else 0

                if avg_accel > 1.0:
                    state = TrajectoryState.ACCELERATING
                elif avg_accel < -1.0:
                    state = TrajectoryState.DECELERATING
                else:
                    if len(positions) > 3:
                        curvature = self._compute_curvature(positions)
                        if curvature > 0.3:
                            state = TrajectoryState.TURNING
                        else:
                            headings = self._compute_headings(positions)
                            if len(headings) > 1:
                                heading_changes = np.abs(np.diff(headings))
                                avg_change = np.mean(heading_changes)
                                if avg_change > 1.0:
                                    state = TrajectoryState.ERRATIC
                                else:
                                    state = TrajectoryState.MOVING
                            else:
                                state = TrajectoryState.MOVING
                    else:
                        state = TrajectoryState.MOVING

        if track_id not in self._state_history:
            self._state_history[track_id] = []
        self._state_history[track_id].append(state)

        if len(self._state_history[track_id]) > 50:
            self._state_history[track_id] = self._state_history[track_id][-50:]

        self._stats["total_detections"] += 1
        self._stats["state_distribution"][state.value] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats["detection_time_ms"] = (
            (self._stats["detection_time_ms"] * (self._stats["total_detections"] - 1) + elapsed_ms) /
            self._stats["total_detections"]
        )

        return state

    def _compute_curvature(self, positions: np.ndarray) -> float:
        """
        Calcula la curvatura de una trayectoria.

        Args:
            positions: Array de posiciones [N, 2]

        Returns:
            float: Curvatura promedio
        """
        if len(positions) < 3:
            return 0.0

        def circle_fit(x, y):
            A = np.column_stack([x, y, np.ones(len(x))])
            b = -(x**2 + y**2)
            c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
            radius = np.sqrt(c**2 + d**2 - e)
            return 1.0 / (radius + 1e-8)

        try:
            curvature = circle_fit(positions[:, 0], positions[:, 1])
            return float(curvature)
        except Exception:
            return 0.0

    def _compute_headings(self, positions: np.ndarray) -> np.ndarray:
        """
        Calcula los headings de una trayectoria.

        Args:
            positions: Array de posiciones [N, 2]

        Returns:
            np.ndarray: Headings en radianes
        """
        if len(positions) < 2:
            return np.array([])

        headings = []
        for i in range(1, len(positions)):
            dx = positions[i, 0] - positions[i-1, 0]
            dy = positions[i, 1] - positions[i-1, 1]
            heading = np.arctan2(dy, dx)
            headings.append(heading)

        return np.array(headings)

    def get_last_state(self, track_id: int) -> Optional[TrajectoryState]:
        """
        Obtiene el último estado detectado para un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[TrajectoryState]: Último estado o None
        """
        if track_id not in self._state_history or not self._state_history[track_id]:
            return None
        return self._state_history[track_id][-1]

    def get_state_history(self, track_id: int) -> List[TrajectoryState]:
        """
        Obtiene el historial de estados de un track.

        Args:
            track_id: ID del track

        Returns:
            List[TrajectoryState]: Historial de estados
        """
        return self._state_history.get(track_id, [])

    def get_most_common_state(self, track_id: int) -> Optional[TrajectoryState]:
        """
        Obtiene el estado más común de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[TrajectoryState]: Estado más común o None
        """
        history = self.get_state_history(track_id)
        if not history:
            return None

        from collections import Counter
        counter = Counter(history)
        return max(counter, key=counter.get)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del detector."""
        return {
            **self._stats,
            "active_tracks": len(self._state_history),
            "total_states": sum(len(h) for h in self._state_history.values()),
        }

    def clear_track(self, track_id: int) -> None:
        """Elimina el historial de un track."""
        if track_id in self._state_history:
            del self._state_history[track_id]

    def clear_all(self) -> None:
        """Limpia todos los historiales."""
        self._state_history.clear()
