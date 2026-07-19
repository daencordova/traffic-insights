"""
Sistema de predicción de trayectoria para tracking avanzado.

Este módulo implementa un sistema de predicción de trayectoria que permite
anticipar el movimiento futuro de los vehículos, mejorando la robustez
del tracking y permitiendo análisis de tráfico predictivo.
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

from utils.logger import LoggerMixin
from utils.geometry import euclidean_distance


class TrajectoryState(Enum):
    """
    Estados posibles de la trayectoria de un objeto.
    """
    MOVING = "moving"
    STOPPED = "stopped"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    TURNING = "turning"
    ERRATIC = "erratic"
    UNKNOWN = "unknown"


class MotionModel(Enum):
    """
    Modelos de movimiento para predicción de trayectoria.
    """
    LINEAR = "linear"
    CURVED = "curved"
    CYCLIC = "cyclic"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    ADAPTIVE = "adaptive"


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
        state: Estado de la trayectoria.
        motion_model: Modelo de movimiento utilizado.
        predicted_velocity: Velocidad predicha (vx, vy).
        predicted_acceleration: Aceleración predicha (ax, ay).
        uncertainty: Incertidumbre de la predicción (0-1).
        collision_risk: Riesgo de colisión (0-1).
        trajectory_type: Tipo de trayectoria.
        metadata: Metadatos adicionales.
    """
    track_id: int
    positions: List[Tuple[float, float]]
    confidences: List[float]
    timestamps: List[float]
    horizon_seconds: float
    state: TrajectoryState
    motion_model: MotionModel
    predicted_velocity: Tuple[float, float]
    predicted_acceleration: Tuple[float, float]
    uncertainty: float
    collision_risk: float
    trajectory_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectorySample:
    """
    Muestra de trayectoria para entrenamiento/predicción.

    Attributes:
        position: Posición (x, y).
        timestamp: Timestamp de la muestra.
        velocity: Velocidad (vx, vy).
        acceleration: Aceleración (ax, ay).
        heading: Orientación en radianes.
        confidence: Confianza de la muestra.
    """
    position: Tuple[float, float]
    timestamp: float
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    confidence: float = 1.0


class PathPredictor(LoggerMixin):
    """
    Sistema avanzado de predicción de trayectoria.

    Este sistema predice la trayectoria futura de vehículos usando
    modelos de movimiento y análisis histórico.

    Características:
    - Modelos de movimiento lineal, curvilíneo, cíclico y adaptativo
    - Regresión polinomial para trayectorias curvas
    - Detección de estados (movimiento, detenido, acelerando, etc.)
    - Predicción de colisiones y conflictos de trayectoria
    - Historial de predicciones para análisis
    - Incertidumbre cuantitativa
    - Estadísticas de rendimiento

    Attributes:
        history_length: Longitud del historial de posiciones.
        prediction_horizon: Horizonte de predicción en segundos.
        prediction_steps: Número de pasos de predicción.
        min_samples: Mínimo de muestras para predicción.
        motion_model: Modelo de movimiento por defecto.
        uncertainty_threshold: Umbral de incertidumbre.
        _histories: Historial de trayectorias por track_id.
        _predictions: Últimas predicciones por track_id.
        _states: Estados de trayectoria por track_id.
        _stats: Estadísticas del sistema.
    """

    def __init__(
        self,
        history_length: int = 30,
        prediction_horizon: float = 2.0,
        prediction_steps: int = 20,
        min_samples: int = 5,
        motion_model: str = "adaptive",
        uncertainty_threshold: float = 0.7
    ) -> None:
        """
        Inicializa el sistema de predicción de trayectoria.

        Args:
            history_length: Longitud del historial de posiciones.
            prediction_horizon: Horizonte de predicción en segundos.
            prediction_steps: Número de pasos de predicción.
            min_samples: Mínimo de muestras para predicción.
            motion_model: Modelo de movimiento por defecto.
            uncertainty_threshold: Umbral de incertidumbre.
        """
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.prediction_steps = prediction_steps
        self.min_samples = min_samples
        self.uncertainty_threshold = uncertainty_threshold

        if isinstance(motion_model, str):
            self.default_model = MotionModel(motion_model)
        else:
            self.default_model = motion_model

        self._histories: Dict[int, deque] = {}
        self._predictions: Dict[int, TrajectoryPrediction] = {}
        self._states: Dict[int, TrajectoryState] = {}
        self._collision_history: Dict[int, List[float]] = {}
        self._model_weights: Dict[MotionModel, float] = {
            MotionModel.LINEAR: 1.0,
            MotionModel.CURVED: 1.0,
            MotionModel.CYCLIC: 0.5,
            MotionModel.QUADRATIC: 0.8,
            MotionModel.CUBIC: 0.6,
            MotionModel.ADAPTIVE: 1.0,
        }

        self._stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_prediction_time_ms": 0.0,
            "avg_uncertainty": 0.0,
            "state_distribution": {state.value: 0 for state in TrajectoryState},
            "model_usage": {model.value: 0 for model in MotionModel},
            "active_tracks": 0,
            "total_tracks": 0,
        }

        self._lock = None

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
        """
        if timestamp is None:
            timestamp = time.time()

        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self.history_length)
            self._stats["total_tracks"] += 1
            self._stats["active_tracks"] += 1

        sample = TrajectorySample(
            position=position,
            timestamp=timestamp,
            velocity=velocity or (0.0, 0.0),
            acceleration=acceleration or (0.0, 0.0),
            confidence=confidence
        )

        if len(self._histories[track_id]) > 0:
            prev = self._histories[track_id][-1]
            dx = position[0] - prev.position[0]
            dy = position[1] - prev.position[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                sample.heading = np.arctan2(dy, dx)
            else:
                sample.heading = prev.heading

        self._histories[track_id].append(sample)

        self._update_state(track_id)

        if len(self._histories[track_id]) >= self.min_samples:
            prediction = self._predict_trajectory(track_id)
            if prediction is not None:
                self._predictions[track_id] = prediction
                self._stats["successful_predictions"] += 1
                return prediction

        self._stats["failed_predictions"] += 1
        return None

    def _update_state(self, track_id: int) -> None:
        """
        Actualiza el estado de trayectoria de un track.

        Args:
            track_id: ID del track.
        """
        history = self._histories.get(track_id)
        if history is None or len(history) < self.min_samples:
            self._states[track_id] = TrajectoryState.UNKNOWN
            return

        samples = list(history)
        positions = np.array([s.position for s in samples])
        velocities = np.array([s.velocity for s in samples])

        avg_speed = np.mean(np.linalg.norm(velocities, axis=1))

        if avg_speed < 0.5:
            self._states[track_id] = TrajectoryState.STOPPED
            self._stats["state_distribution"][TrajectoryState.STOPPED.value] += 1
            return

        if len(velocities) > 2:
            speed_changes = np.diff(np.linalg.norm(velocities, axis=1))
            avg_accel = np.mean(speed_changes)

            if avg_accel > 1.0:
                self._states[track_id] = TrajectoryState.ACCELERATING
                self._stats["state_distribution"][TrajectoryState.ACCELERATING.value] += 1
                return
            elif avg_accel < -1.0:
                self._states[track_id] = TrajectoryState.DECELERATING
                self._stats["state_distribution"][TrajectoryState.DECELERATING.value] += 1
                return

        if len(positions) > 3:
            curvature = self._compute_curvature(positions)
            if curvature > 0.3:
                self._states[track_id] = TrajectoryState.TURNING
                self._stats["state_distribution"][TrajectoryState.TURNING.value] += 1
                return

        if len(positions) > 3:
            headings = [s.heading for s in samples]
            heading_changes = np.abs(np.diff(headings))
            avg_change = np.mean(heading_changes)

            if avg_change > 1.0:
                self._states[track_id] = TrajectoryState.ERRATIC
                self._stats["state_distribution"][TrajectoryState.ERRATIC.value] += 1
                return

        self._states[track_id] = TrajectoryState.MOVING
        self._stats["state_distribution"][TrajectoryState.MOVING.value] += 1

    def _compute_curvature(self, positions: np.ndarray) -> float:
        """
        Calcula la curvatura de una trayectoria.

        Args:
            positions: Array de posiciones (N x 2).

        Returns:
            float: Curvatura promedio.
        """
        if len(positions) < 3:
            return 0.0

        def circle_fit(x, y):
            A = np.column_stack([x, y, np.ones(len(x))])
            b = -(x**2 + y**2)
            c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
            radius = np.sqrt(c**2 + d**2 - e)
            return 1.0 / (radius + 1e-8)

        try:
            curvature = circle_fit(positions[:, 0], positions[:, 1])
            return float(curvature)
        except Exception:
            return 0.0

    def _predict_trajectory(self, track_id: int) -> Optional[TrajectoryPrediction]:
        """
        Predice la trayectoria futura de un track.

        Args:
            track_id: ID del track.

        Returns:
            Optional[TrajectoryPrediction]: Predicción generada.
        """
        start_time = time.perf_counter()

        history = self._histories.get(track_id)
        if history is None or len(history) < self.min_samples:
            return None

        samples = list(history)
        positions = np.array([s.position for s in samples])
        velocities = np.array([s.velocity for s in samples])
        timestamps = np.array([s.timestamp for s in samples])
        confidences = np.array([s.confidence for s in samples])

        state = self._states.get(track_id, TrajectoryState.UNKNOWN)

        model = self._select_motion_model(track_id, positions, velocities)

        if model == MotionModel.LINEAR:
            predictions, uncertainties = self._predict_linear(positions, velocities, timestamps)
        elif model == MotionModel.CURVED:
            predictions, uncertainties = self._predict_curved(positions, timestamps)
        elif model == MotionModel.CYCLIC:
            predictions, uncertainties = self._predict_cyclic(positions, timestamps)
        elif model == MotionModel.QUADRATIC:
            predictions, uncertainties = self._predict_polynomial(positions, timestamps, degree=2)
        elif model == MotionModel.CUBIC:
            predictions, uncertainties = self._predict_polynomial(positions, timestamps, degree=3)
        else:
            predictions, uncertainties = self._predict_adaptive(positions, velocities, timestamps)

        if predictions is None or len(predictions) == 0:
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

        collision_risk = self._compute_collision_risk(track_id, predictions)

        self._stats["model_usage"][model.value] += 1

        prediction = TrajectoryPrediction(
            track_id=track_id,
            positions=predictions,
            confidences=confidences_list,
            timestamps=future_timestamps,
            horizon_seconds=self.prediction_horizon,
            state=state,
            motion_model=model,
            predicted_velocity=pred_vel,
            predicted_acceleration=pred_acc,
            uncertainty=avg_uncertainty,
            collision_risk=collision_risk,
            trajectory_type=self._classify_trajectory(positions),
            metadata={
                "samples_used": len(samples),
                "avg_confidence": float(np.mean(confidences)),
                "timestamp": time.time(),
            }
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats["total_predictions"] += 1
        self._stats["avg_prediction_time_ms"] = (
            (self._stats["avg_prediction_time_ms"] * (self._stats["total_predictions"] - 1) + elapsed_ms) /
            self._stats["total_predictions"]
        )
        self._stats["avg_uncertainty"] = (
            (self._stats["avg_uncertainty"] * (self._stats["total_predictions"] - 1) + avg_uncertainty) /
            self._stats["total_predictions"]
        )

        return prediction

    def _select_motion_model(
        self,
        track_id: int,
        positions: np.ndarray,
        velocities: np.ndarray
    ) -> MotionModel:
        """
        Selecciona el modelo de movimiento más apropiado.

        Args:
            track_id: ID del track.
            positions: Array de posiciones.
            velocities: Array de velocidades.

        Returns:
            MotionModel: Modelo seleccionado.
        """
        if self.default_model != MotionModel.ADAPTIVE:
            return self.default_model

        if len(positions) < 5:
            return MotionModel.LINEAR

        scores = {}

        linear_error = self._evaluate_linear_model(positions)
        scores[MotionModel.LINEAR] = 1.0 / (1.0 + linear_error)

        curved_error = self._evaluate_curved_model(positions)
        scores[MotionModel.CURVED] = 1.0 / (1.0 + curved_error)

        quadratic_error = self._evaluate_polynomial_model(positions, degree=2)
        scores[MotionModel.QUADRATIC] = 1.0 / (1.0 + quadratic_error)

        cubic_error = self._evaluate_polynomial_model(positions, degree=3)
        scores[MotionModel.CUBIC] = 1.0 / (1.0 + cubic_error)

        best_model = max(scores, key=scores.get)

        best_score = scores[best_model]
        second_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

        if best_score - second_score < 0.1:
            return MotionModel.LINEAR

        return best_model

    def _evaluate_linear_model(self, positions: np.ndarray) -> float:
        """
        Evalúa el modelo lineal.

        Args:
            positions: Array de posiciones.

        Returns:
            float: Error del modelo.
        """
        if len(positions) < 3:
            return 1.0

        t = np.arange(len(positions))
        coeffs_x = np.polyfit(t, positions[:, 0], 1)
        coeffs_y = np.polyfit(t, positions[:, 1], 1)

        pred_x = np.polyval(coeffs_x, t)
        pred_y = np.polyval(coeffs_y, t)

        error = np.mean(np.sqrt((pred_x - positions[:, 0])**2 + (pred_y - positions[:, 1])**2))
        return float(error)

    def _evaluate_curved_model(self, positions: np.ndarray) -> float:
        """
        Evalúa el modelo curvilíneo.

        Args:
            positions: Array de posiciones.

        Returns:
            float: Error del modelo.
        """
        if len(positions) < 4:
            return 1.0

        curvature = self._compute_curvature(positions)
        return 1.0 / (1.0 + curvature)

    def _evaluate_polynomial_model(self, positions: np.ndarray, degree: int) -> float:
        """
        Evalúa el modelo polinomial.

        Args:
            positions: Array de posiciones.
            degree: Grado del polinomio.

        Returns:
            float: Error del modelo.
        """
        if len(positions) < degree + 2:
            return 1.0

        t = np.arange(len(positions))
        try:
            coeffs_x = np.polyfit(t, positions[:, 0], degree)
            coeffs_y = np.polyfit(t, positions[:, 1], degree)

            pred_x = np.polyval(coeffs_x, t)
            pred_y = np.polyval(coeffs_y, t)

            error = np.mean(np.sqrt((pred_x - positions[:, 0])**2 + (pred_y - positions[:, 1])**2))
            return float(error)
        except Exception:
            return 1.0

    def _predict_linear(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Predicción con modelo lineal.

        Args:
            positions: Array de posiciones.
            velocities: Array de velocidades.
            timestamps: Array de timestamps.

        Returns:
            Tuple[List[Tuple[float, float]], List[float]]: Predicciones e incertidumbres.
        """
        avg_velocity = np.mean(velocities, axis=0)

        if np.linalg.norm(avg_velocity) < 0.1:
            if len(positions) > 1:
                avg_velocity = (positions[-1] - positions[0]) / max(1, len(positions) - 1)
            else:
                avg_velocity = (0.0, 0.0)

        predictions = []
        uncertainties = []

        last_pos = positions[-1]
        dt = self.prediction_horizon / self.prediction_steps

        for i in range(self.prediction_steps):
            pred = (
                last_pos[0] + avg_velocity[0] * (i + 1) * dt,
                last_pos[1] + avg_velocity[1] * (i + 1) * dt
            )
            predictions.append(pred)

            uncertainty = 0.1 + 0.4 * (i / self.prediction_steps)
            uncertainties.append(uncertainty)

        return predictions, uncertainties

    def _predict_curved(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Predicción con modelo curvilíneo.

        Args:
            positions: Array de posiciones.
            timestamps: Array de timestamps.

        Returns:
            Tuple[List[Tuple[float, float]], List[float]]: Predicciones e incertidumbres.
        """
        if len(positions) < 4:
            return self._predict_linear(positions, np.zeros_like(positions), timestamps)

        t = np.arange(len(positions))

        try:
            spline_x = UnivariateSpline(t, positions[:, 0], s=0.5, k=3)
            spline_y = UnivariateSpline(t, positions[:, 1], s=0.5, k=3)

            predictions = []
            uncertainties = []

            for i in range(1, self.prediction_steps + 1):
                future_t = len(positions) + i * (self.prediction_horizon / self.prediction_steps)
                pred_x = float(spline_x(future_t))
                pred_y = float(spline_y(future_t))
                predictions.append((pred_x, pred_y))

                uncertainty = 0.15 + 0.5 * (i / self.prediction_steps)
                uncertainties.append(uncertainty)

            return predictions, uncertainties

        except Exception:
            return self._predict_linear(positions, np.zeros_like(positions), timestamps)

    def _predict_cyclic(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Predicción con modelo cíclico (para trayectorias circulares).

        Args:
            positions: Array de posiciones.
            timestamps: Array de timestamps.

        Returns:
            Tuple[List[Tuple[float, float]], List[float]]: Predicciones e incertidumbres.
        """
        if len(positions) < 5:
            return self._predict_linear(positions, np.zeros_like(positions), timestamps)

        try:
            x = positions[:, 0]
            y = positions[:, 1]

            A = np.column_stack([x, y, np.ones(len(x))])
            b = -(x**2 + y**2)
            c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]

            center_x = -c / 2
            center_y = -d / 2
            radius = np.sqrt(c**2 + d**2 - e)

            angles = np.arctan2(y - center_y, x - center_x)

            if len(angles) > 2:
                angle_velocity = np.mean(np.diff(np.unwrap(angles)))
            else:
                angle_velocity = 0.1

            predictions = []
            uncertainties = []

            dt = self.prediction_horizon / self.prediction_steps

            for i in range(self.prediction_steps):
                future_angle = angles[-1] + angle_velocity * (i + 1) * dt
                pred_x = center_x + radius * np.cos(future_angle)
                pred_y = center_y + radius * np.sin(future_angle)
                predictions.append((float(pred_x), float(pred_y)))

                uncertainty = 0.2 + 0.4 * (i / self.prediction_steps)
                uncertainties.append(uncertainty)

            return predictions, uncertainties

        except Exception:
            return self._predict_linear(positions, np.zeros_like(positions), timestamps)

    def _predict_polynomial(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray,
        degree: int
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Predicción con modelo polinomial.

        Args:
            positions: Array de posiciones.
            timestamps: Array de timestamps.
            degree: Grado del polinomio.

        Returns:
            Tuple[List[Tuple[float, float]], List[float]]: Predicciones e incertidumbres.
        """
        if len(positions) < degree + 2:
            return self._predict_linear(positions, np.zeros_like(positions), timestamps)

        t = np.arange(len(positions))

        try:
            coeffs_x = np.polyfit(t, positions[:, 0], degree)
            coeffs_y = np.polyfit(t, positions[:, 1], degree)

            predictions = []
            uncertainties = []

            for i in range(1, self.prediction_steps + 1):
                future_t = len(positions) + i * (self.prediction_horizon / self.prediction_steps)
                pred_x = float(np.polyval(coeffs_x, future_t))
                pred_y = float(np.polyval(coeffs_y, future_t))
                predictions.append((pred_x, pred_y))

                uncertainty = 0.1 + 0.5 * (i / self.prediction_steps)
                uncertainties.append(uncertainty)

            return predictions, uncertainties

        except Exception:
            return self._predict_linear(positions, np.zeros_like(positions), timestamps)

    def _predict_adaptive(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Predicción adaptativa combinando múltiples modelos.

        Args:
            positions: Array de posiciones.
            velocities: Array de velocidades.
            timestamps: Array de timestamps.

        Returns:
            Tuple[List[Tuple[float, float]], List[float]]: Predicciones e incertidumbres.
        """
        models = [
            (MotionModel.LINEAR, self._predict_linear(positions, velocities, timestamps)),
            (MotionModel.CURVED, self._predict_curved(positions, timestamps)),
            (MotionModel.QUADRATIC, self._predict_polynomial(positions, timestamps, 2)),
        ]

        predictions = []
        uncertainties = []

        for i in range(self.prediction_steps):
            weighted_pos = (0.0, 0.0)
            weighted_uncertainty = 0.0
            total_weight = 0.0

            for model, (preds, uncerts) in models:
                if len(preds) > i:
                    weight = self._model_weights.get(model, 1.0)
                    weighted_pos = (
                        weighted_pos[0] + weight * preds[i][0],
                        weighted_pos[1] + weight * preds[i][1]
                    )
                    weighted_uncertainty += weight * uncerts[i]
                    total_weight += weight

            if total_weight > 0:
                predictions.append((weighted_pos[0] / total_weight, weighted_pos[1] / total_weight))
                uncertainties.append(weighted_uncertainty / total_weight)
            else:
                predictions.append(positions[-1])
                uncertainties.append(0.5)

        return predictions, uncertainties

    def _compute_collision_risk(
        self,
        track_id: int,
        predictions: List[Tuple[float, float]]
    ) -> float:
        """
        Calcula el riesgo de colisión con otros tracks.

        Args:
            track_id: ID del track.
            predictions: Lista de posiciones predichas.

        Returns:
            float: Riesgo de colisión (0-1).
        """
        if len(predictions) < 2:
            return 0.0

        if track_id not in self._collision_history:
            self._collision_history[track_id] = []

        collision_count = 0
        total_checks = 0

        for other_id, other_pred in self._predictions.items():
            if other_id == track_id:
                continue

            if len(other_pred.positions) < 2:
                continue

            horizon = min(len(predictions), len(other_pred.positions))

            for i in range(horizon):
                dist = euclidean_distance(predictions[i], other_pred.positions[i])
                if dist < 30.0:
                    collision_count += 1
                total_checks += 1

        if total_checks > 0:
            risk = collision_count / total_checks
        else:
            risk = 0.0

        self._collision_history[track_id].append(risk)
        if len(self._collision_history[track_id]) > 10:
            self._collision_history[track_id] = self._collision_history[track_id][-10:]

        return float(np.mean(self._collision_history[track_id]))

    def _classify_trajectory(self, positions: np.ndarray) -> str:
        """
        Clasifica el tipo de trayectoria.

        Args:
            positions: Array de posiciones.

        Returns:
            str: Tipo de trayectoria.
        """
        if len(positions) < 3:
            return "unknown"

        total_distance = 0
        straight_distance = euclidean_distance(positions[0], positions[-1])

        for i in range(1, len(positions)):
            total_distance += euclidean_distance(positions[i-1], positions[i])

        if total_distance < 1.0:
            return "stationary"

        straightness = straight_distance / total_distance

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
        return self._states.get(track_id, TrajectoryState.UNKNOWN)

    def get_collision_risk(self, track_id: int) -> float:
        """
        Obtiene el riesgo de colisión de un track.

        Args:
            track_id: ID del track.

        Returns:
            float: Riesgo de colisión (0-1).
        """
        pred = self._predictions.get(track_id)
        if pred is None:
            return 0.0
        return pred.collision_risk

    def get_high_risk_tracks(self, threshold: float = 0.6) -> List[int]:
        """
        Obtiene tracks con alto riesgo de colisión.

        Args:
            threshold: Umbral de riesgo (0-1).

        Returns:
            List[int]: Lista de IDs de tracks con alto riesgo.
        """
        high_risk = []
        for track_id, pred in self._predictions.items():
            if pred.collision_risk > threshold:
                high_risk.append(track_id)
        return high_risk

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema.

        Returns:
            Dict[str, Any]: Estadísticas del sistema.
        """
        return {
            **self._stats,
            "active_tracks": len(self._histories),
            "total_tracks_ever": self._stats["total_tracks"],
            "predictions_with_risk": len([p for p in self._predictions.values() if p.collision_risk > 0.3]),
            "avg_collision_risk": float(np.mean([p.collision_risk for p in self._predictions.values()])) if self._predictions else 0.0,
            "state_distribution": self._stats["state_distribution"],
            "model_usage": self._stats["model_usage"],
        }

    def clear_track(self, track_id: int) -> None:
        """
        Elimina todas las predicciones de un track.

        Args:
            track_id: ID del track.
        """
        if track_id in self._histories:
            del self._histories[track_id]
        if track_id in self._predictions:
            del self._predictions[track_id]
        if track_id in self._states:
            del self._states[track_id]
        if track_id in self._collision_history:
            del self._collision_history[track_id]

        self._stats["active_tracks"] = len(self._histories)

    def clear_all(self) -> None:
        """
        Limpia todas las predicciones e historiales.
        """
        self._histories.clear()
        self._predictions.clear()
        self._states.clear()
        self._collision_history.clear()

        self._stats["active_tracks"] = 0
        self._stats["total_tracks"] = 0

        self.logger.info("PathPredictor limpiado")

    def reset(self) -> None:
        """
        Reinicia completamente el sistema de predicción.
        """
        self.clear_all()
        self._stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_prediction_time_ms": 0.0,
            "avg_uncertainty": 0.0,
            "state_distribution": {state.value: 0 for state in TrajectoryState},
            "model_usage": {model.value: 0 for model in MotionModel},
            "active_tracks": 0,
            "total_tracks": 0,
        }
        self.logger.info("PathPredictor reiniciado")

    def __len__(self) -> int:
        """Retorna el número de tracks activos."""
        return len(self._histories)
