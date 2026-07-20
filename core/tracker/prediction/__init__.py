"""
Módulo de predicción de trayectoria.

Proporciona componentes para la predicción de trayectorias:
- Historial de trayectorias
- Modelos de movimiento
- Selector de modelos
- Detector de estados
- Detector de colisiones
"""

from core.tracker.prediction.history import TrajectoryHistory, TrajectorySample
from core.tracker.prediction.motion_models import (
    MotionModel,
    LinearModel,
    CurvedModel,
    CyclicModel,
    PolynomialModel,
    AdaptiveModel,
    MotionModelFactory,
)
from core.tracker.prediction.model_selector import ModelSelector
from core.tracker.prediction.state_detector import StateDetector, TrajectoryState
from core.tracker.prediction.collision_detector import CollisionDetector

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
