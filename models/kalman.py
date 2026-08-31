"""Enhanced Kalman filter for smooth object tracking.

This module provides an enhanced Kalman filter implementation with
adaptive constant acceleration model for smooth object tracking.
"""

from typing import Any

import numpy as np

StateVector = np.ndarray
CovarianceMatrix = np.ndarray
MeasurementVector = np.ndarray


class EnhancedKalmanFilter:
    """Enhanced Kalman filter with adaptive constant acceleration model.

    This Kalman filter implementation provides smooth tracking with
    a 6D state vector: (x, y, vx, vy, ax, ay). It uses a constant
    acceleration motion model for accurate prediction and correction.

    Features:
        - 6D state space (position, velocity, acceleration)
        - Constant acceleration motion model
        - Configurable process and measurement noise
        - State initialization and reset
        - Access to position, velocity, and acceleration estimates

    Attributes:
        dt: Time step between updates.
        process_noise: Process noise covariance scaling.
        measurement_noise: Measurement noise covariance scaling.

    Example:
        >>> kf = EnhancedKalmanFilter(dt=0.1, process_noise=0.01)
        >>>
        >>> # Initialize with first position
        >>> kf.init(100.0, 200.0)
        >>>
        >>> # Predict next state
        >>> predicted = kf.predict()
        >>> print(f"Predicted position: {predicted}")
        >>>
        >>> # Correct with measurement
        >>> measurement = np.array([105.0, 202.0])
        >>> corrected = kf.correct(measurement)
        >>> print(f"Corrected position: {corrected}")
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
        """Initializes the enhanced Kalman filter.

        Args:
            dt: Time step between updates (default: 1.0).
            process_noise: Process noise covariance scaling (default: 0.03).
            measurement_noise: Measurement noise covariance scaling (default: 0.1).

        Example:
            >>> # Standard configuration
            >>> kf = EnhancedKalmanFilter(dt=0.1, process_noise=0.01)
            >>>
            >>> # Smooth tracking with low noise
            >>> kf = EnhancedKalmanFilter(dt=0.1, process_noise=0.005, measurement_noise=0.05)
        """
        self.dt: float = dt
        self.process_noise: float = process_noise
        self.measurement_noise: float = measurement_noise
        self._initialized: bool = False

        self.state: StateVector = np.zeros((6, 1), dtype=np.float32)
        self.covariance: CovarianceMatrix = np.eye(6, dtype=np.float32) * 0.1

        self._setup_matrices()

    def _setup_matrices(self) -> None:
        """Sets up the filter matrices.

        This method initializes the state transition matrix (F),
        observation matrix (H), process noise matrix (Q), and
        measurement noise matrix (R).
        """
        dt = self.dt

        self.F: CovarianceMatrix = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt * dt, 0],
                [0, 1, 0, dt, 0, 0.5 * dt * dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.H: CovarianceMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        self.Q: CovarianceMatrix = np.eye(6, dtype=np.float32) * self.process_noise
        self.R: CovarianceMatrix = np.eye(2, dtype=np.float32) * self.measurement_noise

    def init(self, x: float, y: float) -> None:
        """Initializes the filter with a position.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Example:
            >>> kf.init(100.0, 200.0)
            >>> print(kf.is_initialized)  # True
        """
        self.state = np.array([[x], [y], [0], [0], [0], [0]], dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self._initialized = True

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
        if not self._initialized:
            return self.state[:2].flatten()

        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

        return self.state[:2].flatten()

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
        if not self._initialized:
            self.init(measurement[0], measurement[1])
            return self.state[:2].flatten()

        if measurement.shape != (2, 1):
            measurement = measurement.reshape(2, 1)

        s_matrix = self.H @ self.covariance @ self.H.T + self.R
        kalman_gain = self.covariance @ self.H.T @ np.linalg.inv(s_matrix)

        y = measurement - self.H @ self.state
        self.state = self.state + kalman_gain @ y
        self.covariance = (np.eye(6) - kalman_gain @ self.H) @ self.covariance

        return self.state[:2].flatten()

    def get_position(self) -> np.ndarray:
        """Returns the current estimated position.

        Returns:
            np.ndarray: Position [x, y].

        Example:
            >>> pos = kf.get_position()
            >>> print(f"Position: {pos}")
        """
        return self.state[:2].flatten()

    def get_velocity(self) -> np.ndarray:
        """Returns the current estimated velocity.

        Returns:
            np.ndarray: Velocity [vx, vy].

        Example:
            >>> vel = kf.get_velocity()
            >>> print(f"Velocity: {vel}")
        """
        return self.state[2:4].flatten()

    def get_state(self) -> dict[str, Any]:
        """Returns the complete filter state.

        Returns:
            dict[str, Any]: State information including:
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
        """Checks if the filter is initialized.

        Returns:
            bool: True if initialized.

        Example:
            >>> if kf.is_initialized:
            ...     kf.predict()
            ... else:
            ...     print("Filter not initialized")
        """
        return self._initialized

    def reset(self) -> None:
        """Resets the filter to its initial state.

        Example:
            >>> kf.reset()
            >>> print(kf.is_initialized)  # False
        """
        self.state = np.zeros((6, 1), dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 0.1
        self._initialized = False
