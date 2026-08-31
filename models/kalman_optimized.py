"""Optimized Kalman filter with Numba for CPU.

This module provides an optimized Kalman filter implementation using
Numba JIT compilation for high-performance tracking on CPU.
"""

import numpy as np

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Dummy decorator for when Numba is not available."""

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
    """Optimized Kalman filter prediction step.

    This function performs the prediction step of the Kalman filter
    using Numba JIT compilation for maximum performance.

    Args:
        state: State vector [6] (x, y, vx, vy, ax, ay).
        covariance: Covariance matrix [6, 6].
        f_matrix: State transition matrix [6, 6].
        q_matrix: Process noise matrix [6, 6].

    Returns:
        tuple: (new_state, new_covariance)

    Example:
        >>> state = np.zeros(6)
        >>> cov = np.eye(6) * 0.1
        >>> state_pred, cov_pred = kalman_predict(state, cov, f, q)
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
    """Optimized Kalman filter correction step.

    This function performs the correction step of the Kalman filter
    using Numba JIT compilation for maximum performance.

    Args:
        state: State vector [6] (x, y, vx, vy, ax, ay).
        covariance: Covariance matrix [6, 6].
        measurement: Measurement vector [2] (x, y).
        h_matrix: Observation matrix [2, 6].
        r_matrix: Measurement noise matrix [2, 2].

    Returns:
        tuple: (new_state, new_covariance)

    Example:
        >>> state_corr, cov_corr = kalman_correct(state, cov, measurement, h, r)
    """
    s_matrix = h_matrix @ covariance @ h_matrix.T + r_matrix
    kalman_gain = covariance @ h_matrix.T @ np.linalg.inv(s_matrix)

    y = measurement - h_matrix @ state
    state_corrected = state + kalman_gain @ y
    cov_corrected = (np.eye(6) - kalman_gain @ h_matrix) @ covariance

    return state_corrected, cov_corrected


class OptimizedKalmanFilter:
    """CPU-optimized Kalman filter with Numba.

    This class implements a constant acceleration Kalman filter with
    Numba JIT optimization for high-performance tracking on CPU.

    Features:
        - Vectorized operations with Numba
        - Preallocated memory
        - Constant acceleration model
        - Fast initialization
        - 6D state: (x, y, vx, vy, ax, ay)

    Attributes:
        dt: Time step between updates.
        process_noise: Process noise covariance scaling.
        measurement_noise: Measurement noise covariance scaling.

    Example:
        >>> kf = OptimizedKalmanFilter(dt=1.0, process_noise=0.03)
        >>>
        >>> # Initialize with first measurement
        >>> kf.init(100.0, 200.0)
        >>>
        >>> # Predict next state
        >>> predicted_pos = kf.predict()
        >>> print(f"Predicted: {predicted_pos}")
        >>>
        >>> # Correct with new measurement
        >>> measured_pos = np.array([105.0, 202.0])
        >>> corrected_pos = kf.correct(measured_pos)
        >>> print(f"Corrected: {corrected_pos}")
        >>>
        >>> # Get velocity estimate
        >>> velocity = kf.get_velocity()
        >>> print(f"Velocity: {velocity}")
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.03,
        measurement_noise: float = 0.1,
    ) -> None:
        """Initializes the Kalman filter.

        Args:
            dt: Time step between updates (default: 1.0).
            process_noise: Process noise covariance scaling (default: 0.03).
            measurement_noise: Measurement noise covariance scaling (default: 0.1).

        Example:
            >>> # Standard configuration
            >>> kf = OptimizedKalmanFilter(dt=0.1, process_noise=0.01)
            >>>
            >>> # High noise environment
            >>> kf = OptimizedKalmanFilter(dt=0.1, process_noise=0.05, measurement_noise=0.2)
        """
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
        """Initializes the filter with a position.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Example:
            >>> kf.init(100.0, 200.0)
            >>> print(kf.is_initialized)  # True
        """
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Predicts the next state.

        Returns:
            np.ndarray: Predicted position [x, y].

        Note:
            If not initialized, returns the current state.

        Example:
            >>> predicted = kf.predict()
            >>> print(f"Predicted position: {predicted}")
        """
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
        """Corrects the state with a measurement.

        Args:
            measurement: Measurement vector [x, y].

        Returns:
            np.ndarray: Corrected position [x, y].

        Note:
            If not initialized, initializes the filter with the measurement.

        Example:
            >>> measurement = np.array([105.0, 202.0])
            >>> corrected = kf.correct(measurement)
            >>> print(f"Corrected position: {corrected}")
        """
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
        """Returns the estimated position.

        Returns:
            np.ndarray: Position [x, y].

        Example:
            >>> pos = kf.get_position()
            >>> print(f"Position: {pos}")
        """
        return self.state[:2]

    def get_velocity(self) -> np.ndarray:
        """Returns the estimated velocity.

        Returns:
            np.ndarray: Velocity [vx, vy].

        Example:
            >>> vel = kf.get_velocity()
            >>> print(f"Velocity: {vel}")
        """
        return self.state[2:4]

    def get_state(self) -> dict:
        """Returns the complete filter state.

        Returns:
            dict: State information including:
                - initialized: Whether filter is initialized
                - position: (x, y) position
                - velocity: (vx, vy) velocity
                - acceleration: (ax, ay) acceleration

        Example:
            >>> state = kf.get_state()
            >>> if state["initialized"]:
            ...     print(f"Position: {state['position']}")
            ...     print(f"Velocity: {state['velocity']}")
        """
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
        """Checks if the filter is initialized.

        Returns:
            bool: True if initialized.

        Example:
            >>> if kf.is_initialized:
            ...     kf.predict()
            ... else:
            ...     print("Filter not initialized")
        """
        return self.initialized

    def reset(self) -> None:
        """Resets the filter to its initial state.

        Example:
            >>> kf.reset()
            >>> print(kf.is_initialized)  # False
        """
        self.state = np.zeros(6, dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = False
