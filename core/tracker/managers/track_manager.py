"""
Gestor de ciclo de vida de tracks.

Este módulo maneja la creación, actualización, limpieza y recuperación
de tracks, separando la lógica de gestión de la lógica de tracking.
"""

from typing import Dict, Optional, Any

import numpy as np

from models.track_state import TrackState
from models.enums import TrackStatus
from core.constants import MAX_ACTIVE_TRACKS, MAX_LOST_TRACKS
from utils.logger import LoggerMixin


class TrackManager(LoggerMixin):
    """
    Gestiona el ciclo de vida completo de los tracks.

    Responsabilidades:
    - Creación de nuevos tracks desde detecciones
    - Actualización de tracks existentes
    - Marcado de tracks como perdidos
    - Recuperación de tracks perdidos
    - Limpieza de tracks muertos
    - Mantenimiento de límites de memoria

    Attributes:
        tracks: Diccionario de tracks activos {track_id: TrackState}
        lost_tracks: Diccionario de tracks perdidos {track_id: TrackState}
        next_id: Siguiente ID disponible para nuevos tracks
        max_active_tracks: Límite máximo de tracks activos
    """

    def __init__(self, max_active_tracks: int = MAX_ACTIVE_TRACKS):
        self.tracks: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, TrackState] = {}
        self.next_id: int = 0
        self.max_active_tracks = max_active_tracks

        self._stats = {
            "total_tracks_created": 0,
            "total_tracks_deleted": 0,
            "total_tracks_recovered": 0,
        }

        self.logger.info(
            "TrackManager inicializado",
            max_active_tracks=max_active_tracks
        )

    def create_track(
        self,
        detection: Dict[str, Any],
        features: Optional[np.ndarray] = None,
        kalman_filter: Optional[Any] = None
    ) -> Optional[TrackState]:
        """
        Crea un nuevo track a partir de una detección.

        Args:
            detection: Diccionario de detección con box, centroid, confidence
            features: Features visuales del objeto (opcional)
            kalman_filter: Filtro de Kalman para el track (opcional)

        Returns:
            TrackState: Track creado o None si no se puede crear
        """
        if len(self.tracks) >= self.max_active_tracks:
            self.logger.debug(
                "Límite de tracks activos alcanzado",
                max=self.max_active_tracks,
                current=len(self.tracks)
            )
            return None

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

        self.tracks[self.next_id] = track
        self.next_id += 1
        self._stats["total_tracks_created"] += 1

        self.logger.debug(
            "Track creado",
            track_id=track.track_id,
            confidence=track.confidence,
            active_tracks=len(self.tracks)
        )

        return track

    def update_track(
        self,
        track_id: int,
        detection: Dict[str, Any],
        features: Optional[np.ndarray] = None
    ) -> bool:
        """
        Actualiza un track existente con una nueva detección.

        Args:
            track_id: ID del track a actualizar
            detection: Nueva detección
            features: Nuevos features (opcional)

        Returns:
            bool: True si se actualizó correctamente
        """
        track = self.tracks.get(track_id)
        if track is None:
            return False

        track.update(detection, features)
        return True

    def mark_as_lost(self, track_id: int) -> bool:
        """
        Marca un track como perdido y lo mueve a lost_tracks.

        Args:
            track_id: ID del track a marcar

        Returns:
            bool: True si se marcó correctamente
        """
        if track_id not in self.tracks:
            return False

        track = self.tracks.pop(track_id)
        track.mark_lost()

        if len(track.history) >= 2:
            self.lost_tracks[track_id] = track
            self.logger.debug(
                "Track marcado como perdido",
                track_id=track_id,
                age=track.age,
                lost_tracks=len(self.lost_tracks)
            )

        return True

    def recover_track(self, track_id: int) -> Optional[TrackState]:
        """
        Recupera un track perdido.

        Args:
            track_id: ID del track a recuperar

        Returns:
            TrackState: Track recuperado o None
        """
        if track_id not in self.lost_tracks:
            return None

        track = self.lost_tracks.pop(track_id)

        if len(self.tracks) >= self.max_active_tracks:
            self.logger.warning(
                "No se puede recuperar track, límite alcanzado",
                track_id=track_id
            )
            self.lost_tracks[track_id] = track
            return None

        self.tracks[track_id] = track
        self._stats["total_tracks_recovered"] += 1

        self.logger.info(
            "Track recuperado",
            track_id=track_id,
            active_tracks=len(self.tracks)
        )

        return track

    def remove_track(self, track_id: int, permanent: bool = False) -> bool:
        """
        Elimina un track.

        Args:
            track_id: ID del track a eliminar
            permanent: Si es True, elimina también de lost_tracks

        Returns:
            bool: True si se eliminó correctamente
        """
        removed = False

        if track_id in self.tracks:
            del self.tracks[track_id]
            removed = True
            self._stats["total_tracks_deleted"] += 1

        if permanent and track_id in self.lost_tracks:
            del self.lost_tracks[track_id]
            removed = True

        if removed:
            self.logger.debug(
                "Track eliminado",
                track_id=track_id,
                permanent=permanent
            )

        return removed

    def cleanup_dead_tracks(self) -> int:
        """
        Elimina tracks muertos y gestiona el límite de lost_tracks.

        Returns:
            int: Número de tracks eliminados
        """
        removed = 0

        dead_tracks = []
        for track_id, track in self.tracks.items():
            if track.status == TrackStatus.DEAD:
                dead_tracks.append(track_id)

        for track_id in dead_tracks:
            if track_id in self.tracks:
                track = self.tracks.pop(track_id)
                self.lost_tracks[track_id] = track
                removed += 1

        if len(self.lost_tracks) > MAX_LOST_TRACKS:
            sorted_tracks = sorted(
                self.lost_tracks.items(),
                key=lambda x: x[1].age if x[1] else 0
            )
            to_remove = len(self.lost_tracks) - MAX_LOST_TRACKS
            for track_id, _ in sorted_tracks[:to_remove]:
                del self.lost_tracks[track_id]
                removed += 1

        if removed > 0:
            self.logger.debug(
                "Limpieza de tracks completada",
                removed=removed,
                active=len(self.tracks),
                lost=len(self.lost_tracks)
            )

        return removed

    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Obtiene un track por su ID."""
        return self.tracks.get(track_id)

    def get_lost_track(self, track_id: int) -> Optional[TrackState]:
        """Obtiene un track perdido por su ID."""
        return self.lost_tracks.get(track_id)

    def get_all_tracks(self) -> Dict[int, TrackState]:
        """Obtiene todos los tracks activos."""
        return self.tracks

    def get_all_lost_tracks(self) -> Dict[int, TrackState]:
        """Obtiene todos los tracks perdidos."""
        return self.lost_tracks

    def get_active_count(self) -> int:
        """Obtiene el número de tracks activos."""
        return len(self.tracks)

    def get_lost_count(self) -> int:
        """Obtiene el número de tracks perdidos."""
        return len(self.lost_tracks)

    def clear_all(self) -> None:
        """Elimina todos los tracks."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.next_id = 0
        self.logger.info("Todos los tracks eliminados")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor."""
        return {
            **self._stats,
            "active_tracks": len(self.tracks),
            "lost_tracks": len(self.lost_tracks),
            "max_active_tracks": self.max_active_tracks,
            "next_id": self.next_id,
        }

    def __len__(self) -> int:
        return len(self.tracks)
