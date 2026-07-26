"""
Gestión de estado para el sistema de tracking.

Este módulo proporciona componentes para gestionar el estado de los tracks:
- TrackStateMachine: Máquina de estados (TENTATIVE → CONFIRMED → LOST → DEAD)
- TrackUpdater: Actualización de estado con filtro de Kalman

Las transiciones de estado se basan en hits, pérdidas y otras métricas
para mantener un seguimiento robusto.
"""

from core.tracker.state.state_machine import TrackStateMachine
from core.tracker.state.track_updater import TrackUpdater

__all__ = [
    "TrackStateMachine",
    "TrackUpdater",
]
