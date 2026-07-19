"""
Filtro de Kalman optimizado con Numba para CPU.
"""

import numpy as np

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator


@jit(nopython=True, cache=True)
def kalman_predict(state: np.ndarray, covariance: np.ndarray, F: np.ndarray, Q: np.ndarray) -> tuple:
    """
    Predicción del filtro de Kalman optimizada.

    Args:
        state: Vector de estado [6]
        covariance: Matriz de covarianza [6, 6]
        F: Matriz de transición de estado [6, 6]
        Q: Matriz de ruido del proceso [6, 6]

    Returns:
        tuple: (nuevo_estado, nueva_covarianza)
    """
    state_pred = F @ state
    cov_pred = F @ covariance @ F.T + Q
    return state_pred, cov_pred


@jit(nopython=True, cache=True)
def kalman_correct(state: np.ndarray, covariance: np.ndarray, measurement: np.ndarray,
                   H: np.ndarray, R: np.ndarray) -> tuple:
    """
    Corrección del filtro de Kalman optimizada.

    Args:
        state: Vector de estado [6]
        covariance: Matriz de covarianza [6, 6]
        measurement: Medición [2]
        H: Matriz de observación [2, 6]
        R: Matriz de ruido de medición [2, 2]

    Returns:
        tuple: (nuevo_estado, nueva_covarianza)
    """
    S = H @ covariance @ H.T + R
    K = covariance @ H.T @ np.linalg.inv(S)

    y = measurement - H @ state
    state_corrected = state + K @ y
    cov_corrected = (np.eye(6) - K @ H) @ covariance

    return state_corrected, cov_corrected


class OptimizedKalmanFilter:
    """
    Filtro de Kalman optimizado para CPU con Numba.

    Características:
    - Operaciones vectorizadas con Numba
    - Memoria preasignada
    - Modelo de aceleración constante
    - Inicialización rápida
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 0.03, measurement_noise: float = 0.1):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        self.Q = np.array([
            [dt4/4, 0, dt3/2, 0, dt2/2, 0],
            [0, dt4/4, 0, dt3/2, 0, dt2/2],
            [dt3/2, 0, dt2, 0, dt, 0],
            [0, dt3/2, 0, dt2, 0, dt],
            [dt2/2, 0, dt, 0, 1, 0],
            [0, dt2/2, 0, dt, 0, 1],
        ], dtype=np.float32) * process_noise

        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        self.state = np.zeros(6, dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = False

    def init(self, x: float, y: float) -> None:
        """Inicializa el filtro con una posición."""
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Predice el siguiente estado."""
        if not self.initialized:
            return self.state[:2]

        self.state, self.covariance = kalman_predict(
            self.state, self.covariance, self.F, self.Q
        )

        return self.state[:2]

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """Corrige el estado con una medición."""
        if not self.initialized:
            self.init(measurement[0], measurement[1])
            return self.state[:2]

        if measurement.shape != (2,):
            measurement = measurement.flatten()[:2]

        self.state, self.covariance = kalman_correct(
            self.state, self.covariance, measurement, self.H, self.R
        )

        return self.state[:2]

    def get_position(self) -> np.ndarray:
        """Retorna la posición estimada."""
        return self.state[:2]

    def get_velocity(self) -> np.ndarray:
        """Retorna la velocidad estimada."""
        return self.state[2:4]

    def get_state(self) -> dict:
        """Retorna el estado completo."""
        if not self.initialized:
            return {"initialized": False}

        return {
            "initialized": True,
            "position": (self.state[0], self.state[1]),
            "velocity": (self.state[2], self.state[3]),
            "acceleration": (self.state[4], self.state[5]),
        }

    @property
    def is_initialized(self) -> bool:
        return self.initialized

    def reset(self) -> None:
        """Reinicia el filtro."""
        self.state = np.zeros(6, dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = False
