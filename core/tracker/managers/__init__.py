"""
Gestores de ciclo de vida y features para el sistema de tracking.

Este módulo proporciona los gestores principales que manejan:
- TrackManager: Ciclo de vida completo de tracks (creación, actualización, limpieza)
- FeatureManager: Extracción, almacenamiento y comparación de features

Los gestores son componentes fundamentales del tracker que encapsulan
la lógica de gestión de estado y recursos.
"""

from core.tracker.managers.feature_manager import FeatureManager
from core.tracker.managers.track_manager import TrackManager

__all__ = [
    "TrackManager",
    "FeatureManager",
]
