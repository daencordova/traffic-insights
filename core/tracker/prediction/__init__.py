"""
Módulo de predicción de trayectoria.

Proporciona componentes para la predicción de trayectorias:
- Historial de trayectorias
- Modelos de movimiento
- Selector de modelos
- Detector de estados
- Detector de colisiones
"""

from .history import TrajectoryHistory, TrajectorySample
from .motion_models import (
    MotionModel,
    LinearModel,
    CurvedModel,
    CyclicModel,
    PolynomialModel,
    AdaptiveModel,
    MotionModelFactory,
)
from .model_selector import ModelSelector
from .state_detector import StateDetector, TrajectoryState
from .collision_detector import CollisionDetector

__all__ = [
    "TrajectoryHistory",
    "TrajectorySample",
    "MotionModel",
    "LinearModel",
    "CurvedModel",
    "CyclicModel",
    "PolynomialModel",
    "AdaptiveModel",
    "MotionModelFactory",
    "ModelSelector",
    "StateDetector",
    "TrajectoryState",
    "CollisionDetector",
]
