"""
Gestión de historiales de trayectoria.

Mantiene el historial de posiciones, velocidades y otros datos
para cada track.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import numpy as np


class TrajectorySample:
    """
    Muestra de trayectoria para entrenamiento/predicción.

    Attributes:
        position: Posición (x, y)
        timestamp: Timestamp de la muestra
        velocity: Velocidad (vx, vy)
        acceleration: Aceleración (ax, ay)
        heading: Orientación en radianes
        confidence: Confianza de la muestra
        metadata: Metadatos adicionales
    """
    __slots__ = ('position', 'timestamp', 'velocity', 'acceleration',
                     'heading', 'confidence', 'metadata')

    def __init__(
        self,
        position: Tuple[float, float],
        timestamp: float,
        velocity: Tuple[float, float] = (0.0, 0.0),
        acceleration: Tuple[float, float] = (0.0, 0.0),
        heading: float = 0.0,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ):
        self.position = position
        self.timestamp = timestamp
        self.velocity = velocity
        self.acceleration = acceleration
        self.heading = heading
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "position": self.position,
            "timestamp": self.timestamp,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "heading": self.heading,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class TrajectoryHistory:
    """
    Gestor de historiales de trayectoria.

    Responsabilidades:
    - Almacenar historiales por track
    - Actualizar historiales con nuevas muestras
    - Proporcionar acceso a datos históricos
    - Calcular estadísticas básicas

    Attributes:
        history_length: Longitud máxima del historial
        _histories: Diccionario de historiales por track_id
        _stats: Estadísticas del gestor
    """

    def __init__(self, history_length: int = 30):
        """
        Inicializa el gestor de historiales.

        Args:
            history_length: Longitud máxima del historial
        """
        self.history_length = history_length
        self._histories: Dict[int, deque] = {}
        self._stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "total_samples": 0,
        }

    def update(
        self,
        track_id: int,
        position: Tuple[float, float],
        velocity: Optional[Tuple[float, float]] = None,
        acceleration: Optional[Tuple[float, float]] = None,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Actualiza el historial de un track.

        Args:
            track_id: ID del track
            position: Posición actual
            velocity: Velocidad actual (opcional)
            acceleration: Aceleración actual (opcional)
            confidence: Confianza de la muestra
            timestamp: Timestamp de la muestra
            metadata: Metadatos adicionales

        Returns:
            bool: True si se actualizó correctamente
        """
        if timestamp is None:
            timestamp = time.time()

        sample = TrajectorySample(
            position=position,
            timestamp=timestamp,
            velocity=velocity or (0.0, 0.0),
            acceleration=acceleration or (0.0, 0.0),
            confidence=confidence,
            metadata=metadata or {},
        )

        if track_id in self._histories and len(self._histories[track_id]) > 0:
            prev = self._histories[track_id][-1]
            dx = position[0] - prev.position[0]
            dy = position[1] - prev.position[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                sample.heading = np.arctan2(dy, dx)
            else:
                sample.heading = prev.heading

        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self.history_length)
            self._stats["total_tracks"] += 1

        self._histories[track_id].append(sample)
        self._stats["total_samples"] += 1
        self._stats["active_tracks"] = len(self._histories)

        return True

    def get_history(self, track_id: int) -> Optional[List[TrajectorySample]]:
        """
        Obtiene el historial completo de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[List[TrajectorySample]]: Historial del track o None
        """
        if track_id not in self._histories:
            return None
        return list(self._histories[track_id])

    def get_recent_history(
        self,
        track_id: int,
        n_samples: int = 10
    ) -> Optional[List[TrajectorySample]]:
        """
        Obtiene los últimos N samples de un track.

        Args:
            track_id: ID del track
            n_samples: Número de samples a obtener

        Returns:
            Optional[List[TrajectorySample]]: Samples recientes
        """
        history = self.get_history(track_id)
        if history is None:
            return None
        return history[-n_samples:]

    def get_positions(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        """
        Obtiene las posiciones del historial de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[List[Tuple[float, float]]]: Lista de posiciones
        """
        history = self.get_history(track_id)
        if history is None:
            return None
        return [s.position for s in history]

    def get_velocities(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        """
        Obtiene las velocidades del historial de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[List[Tuple[float, float]]]: Lista de velocidades
        """
        history = self.get_history(track_id)
        if history is None:
            return None
        return [s.velocity for s in history]

    def get_timestamps(self, track_id: int) -> Optional[List[float]]:
        """
        Obtiene los timestamps del historial de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[List[float]]: Lista de timestamps
        """
        history = self.get_history(track_id)
        if history is None:
            return None
        return [s.timestamp for s in history]

    def get_confidence(self, track_id: int) -> Optional[List[float]]:
        """
        Obtiene las confianzas del historial de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[List[float]]: Lista de confianzas
        """
        history = self.get_history(track_id)
        if history is None:
            return None
        return [s.confidence for s in history]

    def get_average_speed(self, track_id: int) -> float:
        """
        Calcula la velocidad promedio de un track.

        Args:
            track_id: ID del track

        Returns:
            float: Velocidad promedio
        """
        velocities = self.get_velocities(track_id)
        if not velocities:
            return 0.0

        speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities]
        return float(np.mean(speeds)) if speeds else 0.0

    def get_last_position(self, track_id: int) -> Optional[Tuple[float, float]]:
        """
        Obtiene la última posición de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[Tuple[float, float]]: Última posición o None
        """
        history = self.get_history(track_id)
        if not history:
            return None
        return history[-1].position

    def get_last_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """
        Obtiene la última velocidad de un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[Tuple[float, float]]: Última velocidad o None
        """
        history = self.get_history(track_id)
        if not history or len(history) < 2:
            return None

        p1 = history[-2].position
        p2 = history[-1].position
        dt = history[-1].timestamp - history[-2].timestamp

        if dt <= 0:
            return (0.0, 0.0)

        return ((p2[0] - p1[0]) / dt, (p2[1] - p1[1]) / dt)

    def get_history_length(self, track_id: int) -> int:
        """
        Obtiene la longitud del historial de un track.

        Args:
            track_id: ID del track

        Returns:
            int: Longitud del historial
        """
        if track_id not in self._histories:
            return 0
        return len(self._histories[track_id])

    def is_valid_for_prediction(self, track_id: int, min_samples: int = 5) -> bool:
        """
        Verifica si un track tiene suficientes muestras para predicción.

        Args:
            track_id: ID del track
            min_samples: Mínimo de muestras requeridas

        Returns:
            bool: True si tiene suficientes muestras
        """
        return self.get_history_length(track_id) >= min_samples

    def clear_track(self, track_id: int) -> None:
        """Elimina el historial de un track."""
        if track_id in self._histories:
            del self._histories[track_id]
            self._stats["active_tracks"] = len(self._histories)

    def clear_all(self) -> None:
        """Limpia todos los historiales."""
        self._histories.clear()
        self._stats["active_tracks"] = 0
        self._stats["total_samples"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor."""
        total_samples = sum(len(h) for h in self._histories.values())
        return {
            **self._stats,
            "total_samples": total_samples,
            "avg_history_length": total_samples / max(1, len(self._histories)),
            "tracks": list(self._histories.keys()),
        }

    def __len__(self) -> int:
        """Retorna el número de tracks activos."""
        return len(self._histories)

    def __contains__(self, track_id: int) -> bool:
        """Verifica si un track existe en el historial."""
        return track_id in self._histories
