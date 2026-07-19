"""
Modelos de movimiento para predicción de trayectoria.

Implementa diferentes modelos de movimiento: lineal, curvado,
cíclico, polinomial y adaptativo.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from scipy.interpolate import UnivariateSpline


class MotionModel(ABC):
    """Interfaz abstracta para modelos de movimiento."""

    @abstractmethod
    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        horizon: float,
        steps: int
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Predice posiciones futuras.

        Args:
            positions: Array de posiciones [N, 2]
            velocities: Array de velocidades [N, 2]
            timestamps: Array de timestamps [N]
            horizon: Horizonte de predicción en segundos
            steps: Número de pasos de predicción

        Returns:
            Tuple[List[Tuple[float, float]], List[float]]:
                (predicciones, incertidumbres)
        """
        pass

    @abstractmethod
    def evaluate(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        """
        Evalúa la precisión del modelo.

        Args:
            positions: Array de posiciones [N, 2]
            velocities: Array de velocidades [N, 2]

        Returns:
            float: Error del modelo
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del modelo."""
        pass


class LinearModel(MotionModel):
    """Modelo lineal (velocidad constante)."""

    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        horizon: float,
        steps: int
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        avg_velocity = np.mean(velocities, axis=0)

        if np.linalg.norm(avg_velocity) < 0.1 and len(positions) > 1:
            avg_velocity = (positions[-1] - positions[0]) / max(1, len(positions) - 1)

        predictions = []
        uncertainties = []
        last_pos = positions[-1]
        dt = horizon / steps

        for i in range(steps):
            pred = (
                last_pos[0] + avg_velocity[0] * (i + 1) * dt,
                last_pos[1] + avg_velocity[1] * (i + 1) * dt
            )
            predictions.append(pred)
            uncertainty = 0.1 + 0.4 * (i / steps)
            uncertainties.append(uncertainty)

        return predictions, uncertainties

    def evaluate(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        if len(positions) < 3:
            return 1.0

        t = np.arange(len(positions))
        coeffs_x = np.polyfit(t, positions[:, 0], 1)
        coeffs_y = np.polyfit(t, positions[:, 1], 1)

        pred_x = np.polyval(coeffs_x, t)
        pred_y = np.polyval(coeffs_y, t)

        error = np.mean(np.sqrt((pred_x - positions[:, 0])**2 + (pred_y - positions[:, 1])**2))
        return float(error)

    @property
    def name(self) -> str:
        return "linear"


class CurvedModel(MotionModel):
    """Modelo curvilíneo con splines."""

    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        horizon: float,
        steps: int
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        if len(positions) < 4:
            return LinearModel().predict(positions, velocities, timestamps, horizon, steps)

        t = np.arange(len(positions))
        try:
            spline_x = UnivariateSpline(t, positions[:, 0], s=0.5, k=3)
            spline_y = UnivariateSpline(t, positions[:, 1], s=0.5, k=3)

            predictions = []
            uncertainties = []

            for i in range(1, steps + 1):
                future_t = len(positions) + i * (horizon / steps)
                pred_x = float(spline_x(future_t))
                pred_y = float(spline_y(future_t))
                predictions.append((pred_x, pred_y))
                uncertainty = 0.15 + 0.5 * (i / steps)
                uncertainties.append(uncertainty)

            return predictions, uncertainties

        except Exception:
            return LinearModel().predict(positions, velocities, timestamps, horizon, steps)

    def evaluate(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        if len(positions) < 4:
            return 1.0

        def circle_fit(x, y):
            A = np.column_stack([x, y, np.ones(len(x))])
            b = -(x**2 + y**2)
            c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
            radius = np.sqrt(c**2 + d**2 - e)
            return 1.0 / (radius + 1e-8)

        try:
            curvature = circle_fit(positions[:, 0], positions[:, 1])
            return 1.0 / (1.0 + curvature)
        except Exception:
            return 0.5

    @property
    def name(self) -> str:
        return "curved"


class CyclicModel(MotionModel):
    """Modelo cíclico (para trayectorias circulares/elípticas)."""

    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        horizon: float,
        steps: int
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        if len(positions) < 5:
            return LinearModel().predict(positions, velocities, timestamps, horizon, steps)

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
            dt = horizon / steps

            for i in range(steps):
                future_angle = angles[-1] + angle_velocity * (i + 1) * dt
                pred_x = center_x + radius * np.cos(future_angle)
                pred_y = center_y + radius * np.sin(future_angle)
                predictions.append((float(pred_x), float(pred_y)))
                uncertainty = 0.2 + 0.4 * (i / steps)
                uncertainties.append(uncertainty)

            return predictions, uncertainties

        except Exception:
            return LinearModel().predict(positions, velocities, timestamps, horizon, steps)

    def evaluate(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        if len(positions) < 5:
            return 1.0

        try:
            x = positions[:, 0]
            y = positions[:, 1]

            A = np.column_stack([x, y, np.ones(len(x))])
            b = -(x**2 + y**2)
            c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]

            center_x = -c / 2
            center_y = -d / 2
            radius = np.sqrt(c**2 + d**2 - e)

            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            error = np.mean(np.abs(distances - radius))
            return 1.0 / (1.0 + error / radius)

        except Exception:
            return 0.5

    @property
    def name(self) -> str:
        return "cyclic"


class PolynomialModel(MotionModel):
    """Modelo polinomial (grado configurable)."""

    def __init__(self, degree: int = 2):
        self._degree = degree

    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        horizon: float,
        steps: int
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        if len(positions) < self._degree + 2:
            return LinearModel().predict(positions, velocities, timestamps, horizon, steps)

        t = np.arange(len(positions))

        try:
            coeffs_x = np.polyfit(t, positions[:, 0], self._degree)
            coeffs_y = np.polyfit(t, positions[:, 1], self._degree)

            predictions = []
            uncertainties = []

            for i in range(1, steps + 1):
                future_t = len(positions) + i * (horizon / steps)
                pred_x = float(np.polyval(coeffs_x, future_t))
                pred_y = float(np.polyval(coeffs_y, future_t))
                predictions.append((pred_x, pred_y))
                uncertainty = 0.1 + 0.5 * (i / steps)
                uncertainties.append(uncertainty)

            return predictions, uncertainties

        except Exception:
            return LinearModel().predict(positions, velocities, timestamps, horizon, steps)

    def evaluate(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        if len(positions) < self._degree + 2:
            return 1.0

        t = np.arange(len(positions))

        try:
            coeffs_x = np.polyfit(t, positions[:, 0], self._degree)
            coeffs_y = np.polyfit(t, positions[:, 1], self._degree)

            pred_x = np.polyval(coeffs_x, t)
            pred_y = np.polyval(coeffs_y, t)

            error = np.mean(np.sqrt((pred_x - positions[:, 0])**2 + (pred_y - positions[:, 1])**2))
            return float(error)

        except Exception:
            return 1.0

    @property
    def name(self) -> str:
        return f"polynomial_{self._degree}"


class AdaptiveModel(MotionModel):
    """Modelo adaptativo que combina múltiples modelos."""

    def __init__(self):
        self._models = [
            LinearModel(),
            CurvedModel(),
            CyclicModel(),
            PolynomialModel(degree=2),
            PolynomialModel(degree=3),
        ]
        self._weights = {model.name: 1.0 for model in self._models}

    def predict(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        horizon: float,
        steps: int
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        errors = {}
        for model in self._models:
            try:
                error = model.evaluate(positions, velocities)
                errors[model.name] = error
            except Exception:
                errors[model.name] = 1.0

        max_error = max(errors.values()) + 0.001
        for name, error in errors.items():
            self._weights[name] = 1.0 / (1.0 + error)

        total_weight = sum(self._weights.values())
        for name in self._weights:
            self._weights[name] /= total_weight

        all_predictions = []
        all_uncertainties = []

        for model in self._models:
            preds, uncerts = model.predict(positions, velocities, timestamps, horizon, steps)
            all_predictions.append((model.name, preds, uncerts))

        combined_predictions = []
        combined_uncertainties = []

        for i in range(steps):
            weighted_pos = (0.0, 0.0)
            weighted_uncertainty = 0.0
            total_weight = 0.0

            for name, preds, uncerts in all_predictions:
                if len(preds) > i:
                    weight = self._weights.get(name, 1.0)
                    weighted_pos = (
                        weighted_pos[0] + weight * preds[i][0],
                        weighted_pos[1] + weight * preds[i][1]
                    )
                    weighted_uncertainty += weight * uncerts[i]
                    total_weight += weight

            if total_weight > 0:
                combined_predictions.append(
                    (weighted_pos[0] / total_weight, weighted_pos[1] / total_weight)
                )
                combined_uncertainties.append(weighted_uncertainty / total_weight)
            else:
                combined_predictions.append(positions[-1])
                combined_uncertainties.append(0.5)

        return combined_predictions, combined_uncertainties

    def evaluate(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        total_error = 0.0
        total_weight = 0.0

        for model in self._models:
            weight = self._weights.get(model.name, 1.0)
            try:
                error = model.evaluate(positions, velocities)
                total_error += weight * error
                total_weight += weight
            except Exception:
                pass

        return total_error / (total_weight + 0.001)

    @property
    def name(self) -> str:
        return "adaptive"


class MotionModelFactory:
    """Fábrica de modelos de movimiento."""

    _models = {
        "linear": LinearModel,
        "curved": CurvedModel,
        "cyclic": CyclicModel,
        "polynomial": PolynomialModel,
        "adaptive": AdaptiveModel,
    }

    @classmethod
    def create(cls, model_type: str, **kwargs) -> MotionModel:
        """
        Crea un modelo de movimiento.

        Args:
            model_type: Tipo de modelo ('linear', 'curved', 'cyclic', 'polynomial', 'adaptive')
            **kwargs: Argumentos adicionales para el modelo

        Returns:
            MotionModel: Modelo de movimiento
        """
        if model_type == "polynomial":
            degree = kwargs.get("degree", 2)
            return PolynomialModel(degree=degree)

        model_class = cls._models.get(model_type)
        if model_class is None:
            raise ValueError(f"Modelo no soportado: {model_type}")

        return model_class(**kwargs)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Obtiene la lista de modelos disponibles."""
        return list(cls._models.keys())
