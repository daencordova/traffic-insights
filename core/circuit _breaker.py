"""Circuit Breaker system to prevent cascading failures.

This module implements the Circuit Breaker pattern to protect system
components that may fail temporarily (network connections, cameras, etc.)
preventing system degradation due to repeated failures.

The circuit breaker has three states:
    - CLOSED: Normal operation, requests are allowed
    - OPEN: Failing state, requests are blocked
    - HALF_OPEN: Testing recovery, limited requests are allowed

Features:
    - Three-state operation (CLOSED, OPEN, HALF_OPEN)
    - Configurable failure threshold
    - Recovery timeout for automatic state transitions
    - Thread-safe operations with locking
    - Usage statistics and health monitoring
    - Global registry for centralized management

Example:
    >>> breaker = CircuitBreaker("camera_connection", failure_threshold=3)
    >>>
    >>> if breaker.can_execute():
    ...     try:
    ...         result = camera.read()
    ...         breaker.record_success()
    ...     except Exception as e:
    ...         breaker.record_failure(e)
    ...         raise
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
import logging
import threading
from typing import Any, Optional


class CircuitState(Enum):
    """Circuit breaker states.

    CLOSED: Normal operation, all requests are allowed.
    OPEN: Circuit is open, requests are blocked to prevent cascading failures.
    HALF_OPEN: Testing recovery state, limited requests are allowed to check if the system has recovered.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker to protect components from cascading failures.

    This class implements the Circuit Breaker pattern with automatic
    state transitions and recovery mechanisms.

    Features:
        - Three states: CLOSED, OPEN, HALF_OPEN
        - Configurable failure threshold
        - Recovery timeout for automatic state changes
        - Thread-safe operations
        - Usage statistics and monitoring
        - State change callbacks

    Attributes:
        name: Unique identifier for this circuit breaker.
        failure_threshold: Number of consecutive failures to open the circuit.
        timeout_seconds: Time before attempting recovery (OPEN -> HALF_OPEN).
        half_open_max_attempts: Maximum attempts in HALF_OPEN before returning to OPEN.
        on_state_change: Callback function when state changes.

    Example:
        >>> breaker = CircuitBreaker("camera_connection", failure_threshold=3)
        >>>
        >>> if breaker.can_execute():
        ...     try:
        ...         result = camera.read()
        ...         breaker.record_success()
        ...     except Exception as e:
        ...         breaker.record_failure(e)
        ...         raise
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 30.0,
        half_open_max_attempts: int = 3,
        on_state_change: Callable[[str, str], None] | None = None,
    ):
        """Initializes the circuit breaker.

        Args:
            name: Unique identifier for the circuit breaker.
            failure_threshold: Number of consecutive failures to open the circuit.
            timeout_seconds: Time before attempting recovery (OPEN -> HALF_OPEN).
            half_open_max_attempts: Maximum attempts in HALF_OPEN before returning to OPEN.
            on_state_change: Callback when the state changes. Receives (name, new_state).

        Example:
            >>> def on_change(name, state):
            ...     print(f"Circuit {name} changed to {state}")
            >>> breaker = CircuitBreaker("db", on_state_change=on_change)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_attempts = half_open_max_attempts
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._last_state_change: datetime | None = datetime.now()
        self._half_open_attempts = 0
        self._total_failures = 0
        self._total_successes = 0

        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")

        self.logger.info(f"Circuit breaker '{name}' initialized (threshold: {failure_threshold})")

    def can_execute(self) -> bool:
        """Checks if the operation can be executed.

        Returns:
            bool: True if the operation is allowed, False otherwise.

        Note:
            In CLOSED state: Always returns True.
            In OPEN state: Returns True only if the timeout has expired (transitions to HALF_OPEN).
            In HALF_OPEN state: Returns True if attempts are below the maximum.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._is_timeout_expired():
                    self._transition_to(CircuitState.HALF_OPEN)
                    self.logger.info(
                        f"Circuit breaker '{self.name}' transitioned to HALF_OPEN (timeout expired)"
                    )
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_attempts < self.half_open_max_attempts:
                    self._half_open_attempts += 1
                    self.logger.debug(
                        f"Circuit breaker '{self.name}' allowing attempt {self._half_open_attempts}/{self.half_open_max_attempts} in HALF_OPEN"
                    )
                    return True
                self.logger.warning(
                    f"Circuit breaker '{self.name}' returned to OPEN (too many attempts in HALF_OPEN: {self._half_open_attempts})"
                )
                self._transition_to(CircuitState.OPEN)
                return False

            return False

    def record_success(self) -> None:
        """Records a successful operation.

        This method:
            - Increments success counters
            - If in HALF_OPEN state: Transitions to CLOSED (recovery successful)
            - Resets the failure count

        Note:
            A successful operation in HALF_OPEN state indicates the system has recovered.
        """
        with self._lock:
            self._success_count += 1
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self.logger.info(f"Circuit breaker '{self.name}' closed (recovery successful)")
                self._transition_to(CircuitState.CLOSED)
                self._half_open_attempts = 0

            self._failure_count = 0

    def record_failure(self, error: Exception | None = None) -> None:
        """Records a failed operation.

        This method:
            - Increments failure counters
            - Records the failure time
            - If in HALF_OPEN state: Transitions to OPEN (recovery failed)
            - If in CLOSED state and threshold exceeded: Transitions to OPEN

        Args:
            error: Exception that caused the failure (optional).
        """
        with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self.logger.warning(
                    f"Circuit breaker '{self.name}' returned to OPEN (recovery attempt failed)"
                )
                self._transition_to(CircuitState.OPEN)
                self._half_open_attempts = 0
                return

            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self.logger.warning(
                    f"Circuit breaker '{self.name}' opened ({self._failure_count} consecutive failures)"
                )
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transitions the circuit breaker to a new state.

        This method:
            - Updates the current state
            - Records the state change time
            - Resets counters when transitioning to CLOSED
            - Calls the state change callback if provided

        Args:
            new_state: The new circuit state to transition to.
        """
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_attempts = 0

        if self.on_state_change:
            self.on_state_change(self.name, new_state.value)

        self.logger.debug(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")

    def _is_timeout_expired(self) -> bool:
        """Checks if the recovery timeout has expired.

        Returns:
            bool: True if the timeout has expired, False otherwise.
        """
        if self._last_state_change is None:
            return True
        elapsed = (datetime.now() - self._last_state_change).total_seconds()
        return elapsed >= self.timeout_seconds

    def get_state(self) -> str:
        """Gets the current state as a string.

        Returns:
            str: Current state value ('closed', 'open', or 'half_open').
        """
        with self._lock:
            return self._state.value

    def get_stats(self) -> dict[str, Any]:
        """Gets statistics for the circuit breaker.

        Returns:
            dict: Dictionary containing all statistics and configuration.

        Example:
            >>> stats = breaker.get_stats()
            >>> print(f"State: {stats['state']}, Failures: {stats['failure_count']}")
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "half_open_attempts": self._half_open_attempts,
                "last_failure_time": self._last_failure_time.isoformat()
                if self._last_failure_time
                else None,
                "last_state_change": self._last_state_change.isoformat()
                if self._last_state_change
                else None,
                "timeout_seconds": self.timeout_seconds,
                "failure_threshold": self.failure_threshold,
            }

    def reset(self) -> None:
        """Resets the circuit breaker to the CLOSED state.

        This method manually resets the circuit breaker, clearing all
        failure counts and transitioning back to normal operation.

        Note:
            This is useful for manual recovery or testing scenarios.
        """
        with self._lock:
            self.logger.info(f"Circuit breaker '{self.name}' manually reset")
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._half_open_attempts = 0


class CircuitBreakerRegistry:
    """Global registry for circuit breakers.

    This singleton class provides centralized access to all circuit breakers,
    making it useful for monitoring, health checks, and management.

    Features:
        - Singleton pattern for global access
        - Register and retrieve circuit breakers by name
        - Aggregate statistics and health summaries
        - Batch operations (reset all, get all stats)

    Example:
        >>> registry = CircuitBreakerRegistry()
        >>> registry.register(CircuitBreaker("camera"))
        >>> registry.register(CircuitBreaker("database"))
        >>>
        >>> # Get health summary
        >>> health = registry.get_health_summary()
        >>> print(f"Healthy: {health['healthy']}, Open: {health['open']}")
        >>>
        >>> # Get all statistics
        >>> all_stats = registry.get_all_stats()
    """

    _instance: Optional["CircuitBreakerRegistry"] = None
    _breakers: dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, breaker: CircuitBreaker) -> None:
        """Registers a circuit breaker in the global registry.

        Args:
            breaker: Circuit breaker instance to register.

        Note:
            If a breaker with the same name already exists, it will be overwritten.
        """
        with self._lock:
            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Gets a circuit breaker by name.

        Args:
            name: Name of the circuit breaker to retrieve.

        Returns:
            CircuitBreaker | None: The circuit breaker if found, None otherwise.
        """
        with self._lock:
            return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Gets statistics for all registered circuit breakers.

        Returns:
            dict: Dictionary mapping breaker names to their statistics.
        """
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Resets all registered circuit breakers to the CLOSED state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_health_summary(self) -> dict[str, Any]:
        """Gets a health summary of all registered circuit breakers.

        Returns:
            dict: Health summary including:
                - total: Total number of circuit breakers
                - open: Number in OPEN state
                - half_open: Number in HALF_OPEN state
                - closed: Number in CLOSED state
                - open_names: Names of OPEN circuit breakers
                - half_open_names: Names of HALF_OPEN circuit breakers
                - healthy: Boolean indicating if all are CLOSED

        Example:
            >>> summary = registry.get_health_summary()
            >>> if not summary["healthy"]:
            ...     print(f"Failing breakers: {summary['open_names']}")
        """
        with self._lock:
            total = len(self._breakers)
            open_breakers = [
                name for name, breaker in self._breakers.items() if breaker.get_state() == "open"
            ]
            half_open_breakers = [
                name
                for name, breaker in self._breakers.items()
                if breaker.get_state() == "half_open"
            ]

            return {
                "total": total,
                "open": len(open_breakers),
                "half_open": len(half_open_breakers),
                "closed": total - len(open_breakers) - len(half_open_breakers),
                "open_names": open_breakers,
                "half_open_names": half_open_breakers,
                "healthy": len(open_breakers) == 0,
            }


circuit_breaker_registry = CircuitBreakerRegistry()
