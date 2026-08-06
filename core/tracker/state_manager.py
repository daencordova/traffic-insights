"""Gestor de estado de tracks para el sistema de tracking.

Este módulo proporciona una gestión centralizada del ciclo de vida de los tracks,
incluyendo creación, actualización, pérdida, recuperación y limpieza.

Características principales:
- Gestión de tracks activos y perdidos
- Límite de tracks activos configurable
- Recuperación de tracks perdidos
- Limpieza automática de tracks muertos
- Estadísticas de gestión
"""

import time
from typing import Any

import numpy as np

from core.constants.tracking import MAX_ACTIVE_TRACKS, MAX_LOST_TRACKS
from models.enums import TrackStatus
from models.track_state import TrackState
from utils.logger import LoggerMixin


class TrackStateManager(LoggerMixin):
    """Gestor centralizado del estado de todos los tracks.

    Esta clase maneja el ciclo de vida completo de los tracks:
    - Creación de nuevos tracks
    - Actualización de tracks existentes
    - Marcado de tracks como perdidos
    - Recuperación de tracks perdidos
    - Limpieza de tracks muertos
    - Estadísticas de gestión

    Attributes:
        active_tracks: Diccionario de tracks activos (ID -> TrackState)
        lost_tracks: Diccionario de tracks perdidos (ID -> TrackState)
        next_id: Siguiente ID disponible para nuevos tracks
        max_active_tracks: Límite máximo de tracks activos

    Example:
        >>> manager = TrackStateManager(max_active_tracks=50)
        >>> track = manager.create_track(detection, features)
        >>> manager.update_track(track.track_id, new_detection)
        >>> manager.mark_as_lost(track.track_id)
        >>> recovered = manager.recover_track(track.track_id)
    """

    def __init__(self, max_active_tracks: int = MAX_ACTIVE_TRACKS):
        """Inicializa el gestor de estado de tracks.

        Args:
            max_active_tracks: Número máximo de tracks activos permitidos.
                Si se excede, no se pueden crear nuevos tracks.
        """
        self.active_tracks: dict[int, TrackState] = {}
        self.lost_tracks: dict[int, TrackState] = {}
        self.next_id: int = 0
        self.max_active_tracks = max_active_tracks

        self._creation_timestamps: dict[int, float] = {}
        self._update_timestamps: dict[int, float] = {}

        self._stats = {
            "total_created": 0,
            "total_deleted": 0,
            "total_recovered": 0,
            "total_lost": 0,
            "creation_rate": 0.0,
            "recovery_rate": 0.0,
        }

        self._last_stats_update = time.time()
        self._stats_window = 60.0

        self.logger.info("TrackStateManager inicializado", max_active_tracks=max_active_tracks)

    def create_track(
        self,
        detection: dict[str, Any],
        features: np.ndarray | None = None,
        kalman_filter: Any | None = None,
    ) -> TrackState | None:
        """Crea un nuevo track a partir de una detección.

        Args:
            detection: Diccionario de detección con 'box' y 'centroid'.
            features: Features visuales para re-identificación (opcional).
            kalman_filter: Filtro de Kalman preconfigurado (opcional).

        Returns:
            Optional[TrackState]: Track creado o None si no se puede crear.

        Note:
            No se crea un track si se ha alcanzado el límite de tracks activos.
            El nuevo track se inicia en estado TENTATIVE.
        """
        if len(self.active_tracks) >= self.max_active_tracks:
            self.logger.debug(
                "Límite de tracks activos alcanzado",
                max=self.max_active_tracks,
                current=len(self.active_tracks),
            )
            return None

        try:
            track = TrackState(
                track_id=self.next_id,
                bbox=detection.get("box"),
                centroid=detection.get("centroid"),
                features=features,
                confidence=detection.get("confidence", 0.5),
                class_id=detection.get("class_id", -1),
                label=detection.get("label", "unknown"),
            )

            if kalman_filter:
                track.kalman_filter = kalman_filter

            self.active_tracks[self.next_id] = track
            self._creation_timestamps[self.next_id] = time.time()
            self._update_timestamps[self.next_id] = time.time()
            self.next_id += 1
            self._stats["total_created"] += 1

            self.logger.debug(
                "Track creado",
                track_id=track.track_id,
                confidence=track.confidence,
                active=len(self.active_tracks),
            )

            return track

        except Exception as e:
            self.logger.error(f"Error creando track: {e}")
            return None

    def update_track(
        self, track_id: int, detection: dict[str, Any], features: np.ndarray | None = None
    ) -> bool:
        """Actualiza un track existente con nueva detección.

        Args:
            track_id: ID del track a actualizar.
            detection: Nueva detección para actualizar.
            features: Nuevos features visuales (opcional).

        Returns:
            bool: True si se actualizó correctamente.
        """
        track = self.active_tracks.get(track_id)
        if track is None:
            return False

        try:
            track.update(detection, features)
            self._update_timestamps[track_id] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"Error actualizando track {track_id}: {e}")
            return False

    def mark_as_lost(self, track_id: int) -> bool:
        """Marca un track como perdido.

        Args:
            track_id: ID del track a marcar como perdido.

        Returns:
            bool: True si se marcó correctamente.

        Note:
            Si el track tiene historial insuficiente, se elimina directamente.
            Los tracks perdidos se almacenan en lost_tracks para posible
            re-identificación.
        """
        if track_id not in self.active_tracks:
            return False

        track = self.active_tracks.pop(track_id)
        track.mark_lost()

        if len(track.history) >= 2:
            self.lost_tracks[track_id] = track
            self._stats["total_lost"] += 1
            self.logger.debug(
                "Track marcado como perdido",
                track_id=track_id,
                age=track.age,
                lost_tracks=len(self.lost_tracks),
            )
        else:
            self.logger.debug("Track eliminado (historial insuficiente)", track_id=track_id)

        return True

    def recover_track(self, track_id: int) -> TrackState | None:
        """Recupera un track perdido.

        Args:
            track_id: ID del track a recuperar.

        Returns:
            Optional[TrackState]: Track recuperado o None si no se puede.

        Note:
            La recuperación solo es posible si hay espacio para tracks activos.
            El track recuperado vuelve al estado CONFIRMED.
        """
        if track_id not in self.lost_tracks:
            return None

        if len(self.active_tracks) >= self.max_active_tracks:
            self.logger.warning("No se puede recuperar track, límite alcanzado", track_id=track_id)
            return None

        track = self.lost_tracks.pop(track_id)
        self.active_tracks[track_id] = track
        self._stats["total_recovered"] += 1

        self.logger.info(
            "Track recuperado", track_id=track_id, active_tracks=len(self.active_tracks)
        )

        return track

    def remove_track(self, track_id: int, permanent: bool = False) -> bool:
        """Elimina un track.

        Args:
            track_id: ID del track a eliminar.
            permanent: Si es True, elimina también de lost_tracks.

        Returns:
            bool: True si se eliminó correctamente.
        """
        removed = False

        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
            removed = True
            self._stats["total_deleted"] += 1

        if permanent and track_id in self.lost_tracks:
            del self.lost_tracks[track_id]
            removed = True

        if removed:
            self.logger.debug("Track eliminado", track_id=track_id, permanent=permanent)

        return removed

    def cleanup_dead_tracks(self) -> int:
        """Elimina tracks muertos y gestiona el límite de lost_tracks.

        Returns:
            int: Número de tracks eliminados.

        Note:
            - Los tracks en estado DEAD se mueven a lost_tracks
            - Si lost_tracks supera el límite, se eliminan los más antiguos
        """
        removed = 0

        dead_tracks = [
            track_id
            for track_id, track in self.active_tracks.items()
            if track.status == TrackStatus.DEAD
        ]

        for track_id in dead_tracks:
            track = self.active_tracks.pop(track_id)
            self.lost_tracks[track_id] = track
            removed += 1

        if len(self.lost_tracks) > MAX_LOST_TRACKS:
            sorted_tracks = sorted(self.lost_tracks.items(), key=lambda x: x[1].age)
            to_remove = len(self.lost_tracks) - MAX_LOST_TRACKS
            for track_id, _ in sorted_tracks[:to_remove]:
                del self.lost_tracks[track_id]
                removed += 1

        if removed > 0:
            self.logger.debug(
                "Limpieza de tracks completada",
                removed=removed,
                active=len(self.active_tracks),
                lost=len(self.lost_tracks),
            )

        return removed

    def get_track(self, track_id: int) -> TrackState | None:
        """Obtiene un track activo por su ID.

        Args:
            track_id: ID del track.

        Returns:
            Optional[TrackState]: Track o None si no existe.
        """
        return self.active_tracks.get(track_id)

    def get_lost_track(self, track_id: int) -> TrackState | None:
        """Obtiene un track perdido por su ID.

        Args:
            track_id: ID del track.

        Returns:
            Optional[TrackState]: Track perdido o None si no existe.
        """
        return self.lost_tracks.get(track_id)

    def get_all_tracks(self) -> dict[int, TrackState]:
        """Obtiene todos los tracks activos.

        Returns:
            Dict[int, TrackState]: Diccionario de tracks activos.
        """
        return self.active_tracks

    def get_all_lost_tracks(self) -> dict[int, TrackState]:
        """Obtiene todos los tracks perdidos.

        Returns:
            Dict[int, TrackState]: Diccionario de tracks perdidos.
        """
        return self.lost_tracks

    def get_active_count(self) -> int:
        """Obtiene el número de tracks activos.

        Returns:
            int: Número de tracks activos.
        """
        return len(self.active_tracks)

    def get_lost_count(self) -> int:
        """Obtiene el número de tracks perdidos.

        Returns:
            int: Número de tracks perdidos.
        """
        return len(self.lost_tracks)

    def is_track_active(self, track_id: int) -> bool:
        """Verifica si un track está activo.

        Args:
            track_id: ID del track.

        Returns:
            bool: True si el track está activo.
        """
        return track_id in self.active_tracks

    def is_track_lost(self, track_id: int) -> bool:
        """Verifica si un track está perdido.

        Args:
            track_id: ID del track.

        Returns:
            bool: True si el track está perdido.
        """
        return track_id in self.lost_tracks

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas del gestor.

        Returns:
            Dict[str, Any]: Estadísticas incluyendo:
                - total_created: Total de tracks creados
                - total_deleted: Total de tracks eliminados
                - total_recovered: Total de tracks recuperados
                - total_lost: Total de tracks perdidos
                - creation_rate: Tasa de creación (tracks/segundo)
                - recovery_rate: Tasa de recuperación
                - active_tracks: Tracks activos actuales
                - lost_tracks: Tracks perdidos actuales
                - max_active_tracks: Límite máximo
                - next_id: Próximo ID disponible
                - avg_track_age: Edad promedio de tracks
        """
        current_time = time.time()

        if current_time - self._last_stats_update > 0:
            elapsed = current_time - self._last_stats_update
            self._stats["creation_rate"] = self._stats["total_created"] / max(1, elapsed)
            self._stats["recovery_rate"] = self._stats["total_recovered"] / max(1, elapsed)

        self._last_stats_update = current_time

        return {
            **self._stats,
            "active_tracks": len(self.active_tracks),
            "lost_tracks": len(self.lost_tracks),
            "max_active_tracks": self.max_active_tracks,
            "next_id": self.next_id,
            "avg_track_age": self._calculate_avg_age(),
        }

    def _calculate_avg_age(self) -> float:
        """Calcula la edad promedio de los tracks activos.

        Returns:
            float: Edad promedio en frames.
        """
        if not self.active_tracks:
            return 0.0

        total_age = sum(
            time.time() - self._creation_timestamps.get(tid, time.time())
            for tid in self.active_tracks
        )
        return total_age / len(self.active_tracks)

    def clear_all(self) -> None:
        """Elimina todos los tracks (activos y perdidos)."""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self._creation_timestamps.clear()
        self._update_timestamps.clear()
        self.next_id = 0
        self.logger.info("Todos los tracks eliminados")

    def reset(self) -> None:
        """Reinicia el gestor de estado."""
        self.clear_all()
        self._stats = {
            "total_created": 0,
            "total_deleted": 0,
            "total_recovered": 0,
            "total_lost": 0,
            "creation_rate": 0.0,
            "recovery_rate": 0.0,
        }
        self._last_stats_update = time.time()

    def __len__(self) -> int:
        """Retorna el número de tracks activos."""
        return len(self.active_tracks)

    def __contains__(self, track_id: int) -> bool:
        """Verifica si un track existe en el gestor."""
        return track_id in self.active_tracks or track_id in self.lost_tracks
