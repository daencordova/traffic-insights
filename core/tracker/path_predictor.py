"""
Sistema de predicción de trayectoria para tracking avanzado.

Este módulo implementa un sistema completo de predicción de trayectoria
que permite anticipar el movimiento futuro de los objetos en seguimiento.

El sistema combina:
- Historial de trayectorias
- Múltiples modelos de movimiento (lineal, curvado, cíclico, polinomial)
- Selección adaptativa de modelos
- Detección de estados de movimiento
- Predicción de colisiones
- Cuantificación de incertidumbre
"""

import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np

from utils.logger import LoggerMixin
from core.tracker.prediction.history import TrajectoryHistory
from core.tracker.prediction.model_selector import ModelSelector
from core.tracker.prediction.state_detector import StateDetector, TrajectoryState
from core.tracker.prediction.collision_detector import CollisionDetector
from core.constants import (
    PATH_PREDICTION_HISTORY_LENGTH,
    PATH_PREDICTION_HORIZON,
    PATH_PREDICTION_STEPS,
    PATH_PREDICTION_MIN_SAMPLES,
    PATH_PREDICTION_UNCERTAINTY_THRESHOLD,
)

@dataclass
class TrajectoryPrediction:
    """
    Representa una predicción de trayectoria.

    Attributes:
        track_id: ID del track.
        positions: Lista de posiciones predichas (x, y).
        confidences: Confianza por cada predicción (0-1).
        timestamps: Timestamps de cada predicción.
        horizon_seconds: Horizonte de predicción en segundos.
        state: Estado de la trayectoria (moving, stopped, etc.).
        motion_model: Modelo de movimiento utilizado.
        predicted_velocity: Velocidad predicha (vx, vy).
        predicted_acceleration: Aceleración predicha (ax, ay).
        uncertainty: Incertidumbre de la predicción (0-1).
        collision_risk: Riesgo de colisión (0-1).
        trajectory_type: Tipo de trayectoria.
        metadata: Metadatos adicionales.

    Example:
        >>> prediction = predictor.update(track_id, position)
        >>> if prediction.collision_risk > 0.7:
        ...     print(f"⚠️ Alto riesgo de colisión para track {track_id}")
        >>> for pos in prediction.positions[:5]:
        ...     print(f"Posición predicha: {pos}")
    """
    track_id: int
    positions: List[Tuple[float, float]]
    confidences: List[float]
    timestamps: List[float]
    horizon_seconds: float
    state: TrajectoryState
    motion_model: str
    predicted_velocity: Tuple[float, float]
    predicted_acceleration: Tuple[float, float]
    uncertainty: float
    collision_risk: float
    trajectory_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PathPredictor(LoggerMixin):
    """
    Sistema avanzado de predicción de trayectoria.

    Este sistema orquesta todos los componentes de predicción para
    proporcionar estimaciones precisas del movimiento futuro de los objetos.

    Características:
        - Múltiples modelos de movimiento (lineal, curvado, cíclico, polinomial)
        - Selección adaptativa de modelos basada en el historial
        - Detección de estados de movimiento (detenido, acelerando, girando, etc.)
        - Predicción de colisiones entre objetos
        - Cuantificación de incertidumbre
        - Estadísticas de rendimiento

    Attributes:
        prediction_horizon: Horizonte de predicción en segundos.
        prediction_steps: Número de pasos de predicción.
        min_samples: Mínimo de muestras para predicción.
        uncertainty_threshold: Umbral de incertidumbre.
        history: Historial de trayectorias por track.
        model_selector: Selector de modelos de movimiento.
        state_detector: Detector de estados de movimiento.
        collision_detector: Detector de colisiones.

    Example:
        >>> predictor = PathPredictor(
        ...     history_length=30,
        ...     prediction_horizon=2.0,
        ...     prediction_steps=20
        ... )
        >>> for track_id, track in tracks.items():
        ...     prediction = predictor.update(
        ...         track_id,
        ...         track.centroid,
        ...         track.velocity
        ...     )
        ...     if prediction and prediction.collision_risk > 0.5:
        ...         print(f"⚠️ Colisión potencial para track {track_id}")
    """

    def __init__(
        self,
        history_length: int = PATH_PREDICTION_HISTORY_LENGTH,
        prediction_horizon: float = PATH_PREDICTION_HORIZON,
        prediction_steps: int = PATH_PREDICTION_STEPS,
        min_samples: int = PATH_PREDICTION_MIN_SAMPLES,
        motion_model: str = "adaptive",
        uncertainty_threshold: float = PATH_PREDICTION_UNCERTAINTY_THRESHOLD,
    ):
        """
        Inicializa el sistema de predicción de trayectoria.

        Args:
            history_length: Longitud del historial de posiciones.
            prediction_horizon: Horizonte de predicción en segundos.
            prediction_steps: Número de pasos de predicción.
            min_samples: Mínimo de muestras para predicción.
            motion_model: Modelo de movimiento por defecto.
            uncertainty_threshold: Umbral de incertidumbre (0-1).
            collision_threshold: Umbral de distancia para colisión en píxeles.
        """
        self.prediction_horizon = prediction_horizon
        self.prediction_steps = prediction_steps
        self.min_samples = min_samples
        self.uncertainty_threshold = uncertainty_threshold

        self.history = TrajectoryHistory(history_length)
        self.model_selector = ModelSelector([motion_model, "linear", "curved", "polynomial"])
        self.state_detector = StateDetector()
        self.collision_detector = CollisionDetector(collision_threshold)

        self._predictions: Dict[int, TrajectoryPrediction] = {}

        self._stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_prediction_time_ms": 0.0,
            "avg_uncertainty": 0.0,
            "avg_collision_risk": 0.0,
        }

        self.logger.info(
            "PathPredictor inicializado",
            history_length=history_length,
            prediction_horizon=prediction_horizon,
            prediction_steps=prediction_steps,
            motion_model=motion_model
        )

    def update(
        self,
        track_id: int,
        position: Tuple[float, float],
        velocity: Optional[Tuple[float, float]] = None,
        acceleration: Optional[Tuple[float, float]] = None,
        confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> Optional[TrajectoryPrediction]:
        """
        Actualiza el historial de un track y genera predicción.

        Args:
            track_id: ID del track.
            position: Posición actual (x, y).
            velocity: Velocidad actual (vx, vy) (opcional).
            acceleration: Aceleración actual (ax, ay) (opcional).
            confidence: Confianza de la observación (0-1).
            timestamp: Timestamp de la observación.

        Returns:
            Optional[TrajectoryPrediction]: Predicción generada o None.

        Note:
            Se necesita un mínimo de muestras (min_samples) para
            generar una predicción confiable.
        """
        self.history.update(
            track_id=track_id,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            confidence=confidence,
            timestamp=timestamp,
            metadata={"track_id": track_id}
        )

        if not self.history.is_valid_for_prediction(track_id, self.min_samples):
            self._stats["failed_predictions"] += 1
            return None

        import time
        start_time = time.perf_counter()

        prediction = self._predict_trajectory(track_id)
        if prediction is not None:
            self._predictions[track_id] = prediction
            self._stats["successful_predictions"] += 1
        else:
            self._stats["failed_predictions"] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats["avg_prediction_time_ms"] = (
            (self._stats["avg_prediction_time_ms"] * self._stats["total_predictions"] + elapsed_ms) /
            (self._stats["total_predictions"] + 1)
        )
        self._stats["total_predictions"] += 1

        return prediction

    def _predict_trajectory(self, track_id: int) -> Optional[TrajectoryPrediction]:
        """
        Predice la trayectoria futura de un track.

        Args:
            track_id: ID del track.

        Returns:
            Optional[TrajectoryPrediction]: Predicción generada.

        Note:
            El proceso de predicción incluye:
            1. Obtener historial del track
            2. Detectar estado de movimiento
            3. Seleccionar modelo apropiado
            4. Generar predicciones
            5. Calcular incertidumbre y riesgo de colisión
        """
        samples = self.history.get_history(track_id)
        if not samples or len(samples) < self.min_samples:
            return None

        positions = np.array([s.position for s in samples])
        velocities = np.array([s.velocity for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        confidences = np.array([s.confidence for s in samples])

        state = self.state_detector.detect_state(
            track_id, positions, velocities, timestamps
        )

        model = self.model_selector.select_model(
            track_id, positions, velocities
        )

        predictions, uncertainties = model.predict(
            positions,
            velocities,
            timestamps,
            self.prediction_horizon,
            self.prediction_steps
        )

        if not predictions:
            return None

        confidences_list = []
        for i in range(len(predictions)):
            base_conf = np.mean(confidences) * 0.8
            decay = 1.0 - (i / len(predictions)) * 0.5
            confidences_list.append(base_conf * decay)

        current_time = time.time()
        dt = self.prediction_horizon / self.prediction_steps
        future_timestamps = [current_time + (i + 1) * dt for i in range(len(predictions))]

        if len(predictions) > 2:
            pred_vel = (
                (predictions[-1][0] - predictions[0][0]) / self.prediction_horizon,
                (predictions[-1][1] - predictions[0][1]) / self.prediction_horizon
            )
            pred_acc = (
                pred_vel[0] / self.prediction_horizon,
                pred_vel[1] / self.prediction_horizon
            )
        else:
            pred_vel = (0.0, 0.0)
            pred_acc = (0.0, 0.0)

        avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.5
        self._stats["avg_uncertainty"] = (
            (self._stats["avg_uncertainty"] * self._stats["total_predictions"] + avg_uncertainty) /
            (self._stats["total_predictions"] + 1)
        )

        all_predictions = {
            tid: pred.positions
            for tid, pred in self._predictions.items()
        }
        all_predictions[track_id] = predictions

        collision_risk = self.collision_detector.detect_collisions(
            track_id,
            predictions,
            all_predictions
        )

        self._stats["avg_collision_risk"] = (
            (self._stats["avg_collision_risk"] * self._stats["total_predictions"] + collision_risk) /
            (self._stats["total_predictions"] + 1)
        )

        trajectory_type = self._classify_trajectory(positions)

        prediction = TrajectoryPrediction(
            track_id=track_id,
            positions=predictions,
            confidences=confidences_list,
            timestamps=future_timestamps,
            horizon_seconds=self.prediction_horizon,
            state=state,
            motion_model=model.name,
            predicted_velocity=pred_vel,
            predicted_acceleration=pred_acc,
            uncertainty=avg_uncertainty,
            collision_risk=collision_risk,
            trajectory_type=trajectory_type,
            metadata={
                "samples_used": len(samples),
                "avg_confidence": float(np.mean(confidences)),
                "timestamp": time.time(),
                "model_confidence": self.model_selector._stats["model_usage"].get(model.name, 0),
            }
        )

        return prediction

    def _classify_trajectory(self, positions: np.ndarray) -> str:
        """
        Clasifica el tipo de trayectoria.

        Args:
            positions: Array de posiciones [N, 2].

        Returns:
            str: Tipo de trayectoria ('straight', 'slightly_curved',
                'highly_curved', 'stationary', 'unknown').
        """
        if len(positions) < 3:
            return "unknown"

        total_distance = 0
        straight_distance = np.sqrt(
            (positions[-1][0] - positions[0][0])**2 +
            (positions[-1][1] - positions[0][1])**2
        )

        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)

        if total_distance < 1.0:
            return "stationary"

        straightness = straight_distance / (total_distance + 0.001)

        if straightness > 0.85:
            return "straight"
        elif straightness > 0.5:
            return "slightly_curved"
        else:
            return "highly_curved"

    def get_prediction(self, track_id: int) -> Optional[TrajectoryPrediction]:
        """
        Obtiene la última predicción de un track.

        Args:
            track_id: ID del track.

        Returns:
            Optional[TrajectoryPrediction]: Última predicción o None.
        """
        return self._predictions.get(track_id)

    def get_state(self, track_id: int) -> TrajectoryState:
        """
        Obtiene el estado de trayectoria de un track.

        Args:
            track_id: ID del track.

        Returns:
            TrajectoryState: Estado de la trayectoria.
        """
        return self.state_detector.get_last_state(track_id) or TrajectoryState.UNKNOWN

    def get_collision_risk(self, track_id: int) -> float:
        """
        Obtiene el riesgo de colisión de un track.

        Args:
            track_id: ID del track.

        Returns:
            float: Riesgo de colisión (0-1).
        """
        return self.collision_detector.get_risk(track_id)

    def get_high_risk_tracks(self, threshold: float = 0.5) -> List[int]:
        """
        Obtiene tracks con alto riesgo de colisión.

        Args:
            threshold: Umbral de riesgo (0-1).

        Returns:
            List[int]: Lista de IDs de tracks con alto riesgo.
        """
        return self.collision_detector.get_high_risk_tracks(threshold)

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema.

        Returns:
            Dict[str, Any]: Estadísticas del sistema.
        """
        return {
            **self._stats,
            "active_tracks": len(self.history),
            "total_tracks_ever": self.history._stats["total_tracks"],
            "history_stats": self.history.get_stats(),
            "model_selector_stats": self.model_selector.get_stats(),
            "state_detector_stats": self.state_detector.get_stats(),
            "collision_detector_stats": self.collision_detector.get_stats(),
            "predictions_with_risk": len([
                p for p in self._predictions.values()
                if p.collision_risk > 0.3
            ]),
        }

    def clear_track(self, track_id: int) -> None:
        """Elimina todas las predicciones de un track."""
        self.history.clear_track(track_id)
        self.state_detector.clear_track(track_id)
        self.collision_detector.clear_track(track_id)
        self._predictions.pop(track_id, None)

    def clear_all(self) -> None:
        """Limpia todas las predicciones e historiales."""
        self.history.clear_all()
        self.state_detector.clear_all()
        self.collision_detector.clear_all()
        self._predictions.clear()

    def reset(self) -> None:
        """Reinicia completamente el sistema de predicción."""
        self.clear_all()
        self._stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_prediction_time_ms": 0.0,
            "avg_uncertainty": 0.0,
            "avg_collision_risk": 0.0,
        }
        self.logger.info("PathPredictor reiniciado")

    def __len__(self) -> int:
        """Retorna el número de tracks activos."""
        return len(self.history)
