"""
Filtro de Kalman mejorado para tracking suave de objetos
"""

from typing import Dict, Any
import numpy as np

StateVector = np.ndarray
CovarianceMatrix = np.ndarray
MeasurementVector = np.ndarray


class EnhancedKalmanFilter:
    """Filtro de Kalman con modelo de aceleración constante adaptativo"""

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.03,
        measurement_noise: float = 0.1,
    ) -> None:
        self.dt: float = dt
        self.process_noise: float = process_noise
        self.measurement_noise: float = measurement_noise
        self._initialized: bool = False

        self.state: StateVector = np.zeros((6, 1), dtype=np.float32)
        self.covariance: CovarianceMatrix = np.eye(6, dtype=np.float32) * 0.1

        self._setup_matrices()

    def _setup_matrices(self) -> None:
        """Configura las matrices del filtro"""
        dt = self.dt

        self.F: CovarianceMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.H: CovarianceMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        self.Q: CovarianceMatrix = np.eye(6, dtype=np.float32) * self.process_noise
        self.R: CovarianceMatrix = np.eye(2, dtype=np.float32) * self.measurement_noise

    def init(self, x: float, y: float) -> None:
        """Inicializa el filtro con una posición"""
        self.state = np.array([[x], [y], [0], [0], [0], [0]], dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self._initialized = True

    def predict(self) -> np.ndarray:
        """Predice el siguiente estado"""
        if not self._initialized:
            return self.state[:2].flatten()

        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

        return self.state[:2].flatten()

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """Corrige el estado con una medición"""
        if not self._initialized:
            self.init(measurement[0], measurement[1])
            return self.state[:2].flatten()

        if measurement.shape != (2, 1):
            measurement = measurement.reshape(2, 1)

        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ self.covariance

        return self.state[:2].flatten()

    def get_position(self) -> np.ndarray:
        """Retorna la posición estimada actual"""
        return self.state[:2].flatten()

    def get_velocity(self) -> np.ndarray:
        """Retorna la velocidad estimada actual"""
        return self.state[2:4].flatten()

    def get_state(self) -> Dict[str, Any]:
        """Retorna el estado completo del filtro"""
        if not self._initialized:
            return {"initialized": False}

        state = self.state.flatten()
        return {
            "initialized": True,
            "position": (state[0], state[1]),
            "velocity": (state[2], state[3]),
            "acceleration": (state[4], state[5]),
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        """Reinicia el filtro"""
        self.state = np.zeros((6, 1), dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self._initialized = False
