"""
Selector de modelos de movimiento.

Selecciona el modelo más apropiado para cada track basado
en el historial de movimiento.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from core.tracker.prediction.motion_models import (
    MotionModel,
    LinearModel,
    MotionModelFactory,
)

class ModelSelector:
    """
    Selector de modelos de movimiento.

    Responsabilidades:
    - Evaluar diferentes modelos para un track
    - Seleccionar el modelo más apropiado
    - Mantener historial de selecciones

    Attributes:
        available_models: Lista de modelos disponibles
        _model_history: Historial de selecciones por track
        _stats: Estadísticas del selector
    """

    def __init__(self, models: Optional[List[str]] = None):
        """
        Inicializa el selector de modelos.

        Args:
            models: Lista de tipos de modelos a considerar
        """
        if models is None:
            models = ["linear", "curved", "polynomial", "adaptive"]

        self.available_models = []
        for model_type in models:
            try:
                model = MotionModelFactory.create(model_type)
                self.available_models.append(model)
            except Exception as e:
                pass

        if not self.available_models:
            self.available_models.append(LinearModel())

        self._model_history: Dict[int, str] = {}
        self._stats = {
            "total_selections": 0,
            "model_usage": {model.name: 0 for model in self.available_models},
            "selection_time_ms": 0.0,
        }

    def select_model(
        self,
        track_id: int,
        positions: np.ndarray,
        velocities: np.ndarray
    ) -> MotionModel:
        """
        Selecciona el modelo más apropiado para un track.

        Args:
            track_id: ID del track
            positions: Array de posiciones [N, 2]
            velocities: Array de velocidades [N, 2]

        Returns:
            MotionModel: Modelo seleccionado
        """
        import time
        start_time = time.perf_counter()

        if len(positions) < 5:
            model = LinearModel()
        else:
            scores = {}
            for model in self.available_models:
                try:
                    error = model.evaluate(positions, velocities)
                    scores[model.name] = 1.0 / (1.0 + error)
                except Exception:
                    scores[model.name] = 0.0

            best_model_name = max(scores, key=scores.get)
            model = self._get_model_by_name(best_model_name)

        self._model_history[track_id] = model.name
        self._stats["total_selections"] += 1
        self._stats["model_usage"][model.name] = (
            self._stats["model_usage"].get(model.name, 0) + 1
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats["selection_time_ms"] = (
            (self._stats["selection_time_ms"] * (self._stats["total_selections"] - 1) + elapsed_ms) /
            self._stats["total_selections"]
        )

        return model

    def _get_model_by_name(self, name: str) -> MotionModel:
        """Obtiene un modelo por su nombre."""
        for model in self.available_models:
            if model.name == name:
                return model
        return LinearModel()

    def get_last_model(self, track_id: int) -> Optional[MotionModel]:
        """
        Obtiene el último modelo seleccionado para un track.

        Args:
            track_id: ID del track

        Returns:
            Optional[MotionModel]: Modelo seleccionado o None
        """
        if track_id not in self._model_history:
            return None

        model_name = self._model_history[track_id]
        return self._get_model_by_name(model_name)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del selector."""
        return {
            **self._stats,
            "available_models": [m.name for m in self.available_models],
            "total_models": len(self.available_models),
        }

    def reset(self) -> None:
        """Reinicia las estadísticas."""
        self._model_history.clear()
        self._stats = {
            "total_selections": 0,
            "model_usage": {model.name: 0 for model in self.available_models},
            "selection_time_ms": 0.0,
        }
