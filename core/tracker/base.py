"""Tracker principal del sistema (fachada).

Este módulo proporciona la interfaz pública del sistema de tracking,
delegando toda la funcionalidad al TrackOrchestrator.

La clase MultiObjectTracker actúa como una fachada para mantener
compatibilidad con el código existente.
"""

from typing import Any

import numpy as np

from core.interfaces import ITracker
from core.tracker.orchestrator import TrackOrchestrator
from utils.logger import LoggerMixin


class MultiObjectTracker(ITracker, LoggerMixin):
    """Tracker principal del sistema (fachada).

    Esta clase actúa como fachada para el TrackOrchestrator,
    manteniendo la misma interfaz que la implementación anterior
    para garantizar compatibilidad con el código existente.

    Attributes:
        _orchestrator: Orquestador del sistema de tracking.

    Example:
        >>> tracker = MultiObjectTracker()
        >>> frame = cv2.imread("frame.jpg")
        >>> detections = detector.detect(frame)
        >>> tracks = tracker.update(detections, frame)
        >>> for track_id, track_data in tracks.items():
        ...     print(f"Track {track_id}: {track_data['centroid']}")
    """

    def __init__(self) -> None:
        """Inicializa el tracker (fachada)."""
        from config.manager import config_manager

        self.logger.info("Inicializando MultiObjectTracker (fachada)")

        config = config_manager.config.tracker
        self._orchestrator = TrackOrchestrator(config)

        self.logger.info("MultiObjectTracker inicializado")

    def update(
        self, detections: list[dict[str, Any]], frame: np.ndarray
    ) -> dict[int, dict[str, Any]]:
        """Actualiza el tracker con nuevas detecciones.

        Args:
            detections: Lista de detecciones del frame actual.
            frame: Imagen actual para extraer features y contexto.

        Returns:
            Dict[int, Dict[str, Any]]: Información de tracking actualizada.

        Raises:
            TrackingError: Si ocurre un error durante el tracking.
        """
        return self._orchestrator.update(detections, frame)

    def get_tracking_info(self) -> dict[int, dict[str, Any]]:
        """Retorna información de tracking actual.

        Returns:
            Dict[int, Dict[str, Any]]: Información de tracking actual.
        """
        return self._orchestrator.get_tracking_info()

    def get_stats(self) -> dict[str, Any]:
        """Retorna estadísticas del tracker.

        Returns:
            Dict[str, Any]: Estadísticas detalladas del tracker.
        """
        return self._orchestrator.get_stats()

    def get_track(self, track_id: int) -> Any:
        """Obtiene un track por su ID.

        Args:
            track_id: ID del track a obtener.

        Returns:
            Optional[Any]: TrackState del track o None si no existe.
        """
        return self._orchestrator.get_track(track_id)

    def reset(self) -> None:
        """Reinicia el tracker completamente."""
        self._orchestrator.reset()

    @property
    def track_manager(self):
        """Retorna el gestor de tracks (para compatibilidad)."""
        return self._orchestrator.state_manager

    @property
    def feature_manager(self):
        """Retorna el gestor de features (para compatibilidad)."""
        return self._orchestrator.feature_manager

    @property
    def reid_system(self):
        """Retorna el sistema de re-identificación (para compatibilidad)."""
        return self._orchestrator.reid_system

    @property
    def mht_integration(self):
        """Retorna el sistema MHT (para compatibilidad)."""
        return self._orchestrator.mht_integration

    @property
    def online_learner(self):
        """Retorna el sistema de aprendizaje en línea (para compatibilidad)."""
        return self._orchestrator.online_learner

    @property
    def sensor_fusion(self):
        """Retorna el sistema de fusión de sensores (para compatibilidad)."""
        return self._orchestrator.sensor_fusion

    @property
    def path_predictor(self):
        """Retorna el sistema de predicción de trayectoria (para compatibilidad)."""
        return self._orchestrator.path_predictor

    @property
    def state_machine(self):
        """Retorna la máquina de estados (para compatibilidad)."""
        return self._orchestrator.state_machine

    @property
    def track_updater(self):
        """Retorna el actualizador de tracks (para compatibilidad)."""
        return self._orchestrator.track_updater
