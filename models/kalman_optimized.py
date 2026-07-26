"""Filtro de Kalman optimizado con Numba para CPU."""

import numpy as np

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):  # noqa: ARG001
        """Decorador dummy para cuando Numba no está disponible."""

        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator


@jit(nopython=True, cache=True)
def kalman_predict(
    state: np.ndarray,
    covariance: np.ndarray,
    f_matrix: np.ndarray,
    q_matrix: np.ndarray,
) -> tuple:
    """Predicción del filtro de Kalman optimizada.

    Args:
        state: Vector de estado [6]
        covariance: Matriz de covarianza [6, 6]
        f_matrix: Matriz de transición de estado [6, 6]
        q_matrix: Matriz de ruido del proceso [6, 6]

    Returns:
        tuple: (nuevo_estado, nueva_covarianza)
    """
    state_pred = f_matrix @ state
    cov_pred = f_matrix @ covariance @ f_matrix.T + q_matrix
    return state_pred, cov_pred


@jit(nopython=True, cache=True)
def kalman_correct(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    h_matrix: np.ndarray,
    r_matrix: np.ndarray,
) -> tuple:
    """Corrección del filtro de Kalman optimizada.

    Args:
        state: Vector de estado [6]
        covariance: Matriz de covarianza [6, 6]
        measurement: Medición [2]
        h_matrix: Matriz de observación [2, 6]
        r_matrix: Matriz de ruido de medición [2, 2]

    Returns:
        tuple: (nuevo_estado, nueva_covarianza)
    """
    s_matrix = h_matrix @ covariance @ h_matrix.T + r_matrix
    kalman_gain = covariance @ h_matrix.T @ np.linalg.inv(s_matrix)

    y = measurement - h_matrix @ state
    state_corrected = state + kalman_gain @ y
    cov_corrected = (np.eye(6) - kalman_gain @ h_matrix) @ covariance

    return state_corrected, cov_corrected


class OptimizedKalmanFilter:
    """Filtro de Kalman optimizado para CPU con Numba.

    Características:
    - Operaciones vectorizadas con Numba
    - Memoria preasignada
    - Modelo de aceleración constante
    - Inicialización rápida
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.03,
        measurement_noise: float = 0.1,
    ) -> None:
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        self.f_matrix = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.h_matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        self.q_matrix = (
            np.array(
                [
                    [dt4 / 4, 0, dt3 / 2, 0, dt2 / 2, 0],
                    [0, dt4 / 4, 0, dt3 / 2, 0, dt2 / 2],
                    [dt3 / 2, 0, dt2, 0, dt, 0],
                    [0, dt3 / 2, 0, dt2, 0, dt],
                    [dt2 / 2, 0, dt, 0, 1, 0],
                    [0, dt2 / 2, 0, dt, 0, 1],
                ],
                dtype=np.float32,
            )
            * process_noise
        )

        self.r_matrix = np.eye(2, dtype=np.float32) * measurement_noise

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
            self.state,
            self.covariance,
            self.f_matrix,
            self.q_matrix,
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
            self.state,
            self.covariance,
            measurement,
            self.h_matrix,
            self.r_matrix,
        )

        return self.state[:2]

    def get_position(self) -> np.ndarray:
        """Retorna la posición estimada."""
        return self.state[:2]

    def get_velocity(self) -> np.ndarray:
        """Retorna la velocidad estimada."""
        return self.state[2:4]

    def get_state(self) -> dict:
        """Retorna el estado completo del filtro."""
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
        """Verifica si el filtro está inicializado."""
        return self.initialized

    def reset(self) -> None:
        """Reinicia el filtro."""
        self.state = np.zeros(6, dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = False
