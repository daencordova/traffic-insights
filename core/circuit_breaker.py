"""
Sistema Circuit Breaker para prevenir fallos en cascada.

Protege componentes que pueden fallar temporalmente (conexiones de red,
cámaras, etc.) evitando que el sistema se degrade por fallos repetitivos.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import logging
import threading


class CircuitState(Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker para proteger componentes de fallos en cascada.

    Características:
    - Tres estados: CLOSED, OPEN, HALF_OPEN
    - Umbral configurable de fallos
    - Timeout para recuperación
    - Thread-safe
    - Estadísticas de uso

    Ejemplo:
        breaker = CircuitBreaker("camera_connection", failure_threshold=3)

        if breaker.can_execute():
            try:
                result = camera.read()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 30.0,
        half_open_max_attempts: int = 3,
        on_state_change: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Args:
            name: Identificador único del circuit breaker.
            failure_threshold: Número de fallos consecutivos para abrir.
            timeout_seconds: Tiempo antes de intentar recuperación (OPEN -> HALF_OPEN).
            half_open_max_attempts: Intentos máximos en estado HALF_OPEN antes de volver a OPEN.
            on_state_change: Callback cuando cambia el estado.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_attempts = half_open_max_attempts
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: Optional[datetime] = datetime.now()
        self._half_open_attempts = 0
        self._total_failures = 0
        self._total_successes = 0

        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")

        self.logger.info(f"Circuit breaker '{name}' inicializado (umbral: {failure_threshold})")

    def can_execute(self) -> bool:
        """
        Verifica si se puede ejecutar la operación.

        Returns:
            bool: True si la operación está permitida.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._is_timeout_expired():
                    self._transition_to(CircuitState.HALF_OPEN)
                    self.logger.info(f"Circuit breaker '{self.name}' pasó a HALF_OPEN (timeout expirado)")
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_attempts < self.half_open_max_attempts:
                    self._half_open_attempts += 1
                    self.logger.debug(
                        f"Circuit breaker '{self.name}' permitiendo intento {self._half_open_attempts}/{self.half_open_max_attempts} en HALF_OPEN"
                    )
                    return True
                self.logger.warning(
                    f"Circuit breaker '{self.name}' volvió a OPEN (demasiados intentos en HALF_OPEN: {self._half_open_attempts})"
                )
                self._transition_to(CircuitState.OPEN)
                return False

            return False

    def record_success(self) -> None:
        """Registra una operación exitosa."""
        with self._lock:
            self._success_count += 1
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self.logger.info(f"Circuit breaker '{self.name}' cerró (recuperación exitosa)")
                self._transition_to(CircuitState.CLOSED)
                self._half_open_attempts = 0

            self._failure_count = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        Registra una operación fallida.

        Args:
            error: Excepción que causó el fallo (opcional).
        """
        with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self.logger.warning(
                    f"Circuit breaker '{self.name}' volvió a OPEN (falló en recuperación)"
                )
                self._transition_to(CircuitState.OPEN)
                self._half_open_attempts = 0
                return

            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self.logger.warning(
                    f"Circuit breaker '{self.name}' abierto ({self._failure_count} fallos consecutivos)"
                )
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Cambia el estado del circuit breaker."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_attempts = 0

        if self.on_state_change:
            self.on_state_change(self.name, new_state.value)

    def _is_timeout_expired(self) -> bool:
        """Verifica si el timeout de recuperación ha expirado."""
        if self._last_state_change is None:
            return True
        elapsed = (datetime.now() - self._last_state_change).total_seconds()
        return elapsed >= self.timeout_seconds

    def get_state(self) -> str:
        """Obtiene el estado actual como string."""
        with self._lock:
            return self._state.value

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del circuit breaker."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "half_open_attempts": self._half_open_attempts,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "last_state_change": self._last_state_change.isoformat() if self._last_state_change else None,
                "timeout_seconds": self.timeout_seconds,
                "failure_threshold": self.failure_threshold,
            }

    def reset(self) -> None:
        """Reinicia el circuit breaker a estado cerrado."""
        with self._lock:
            self.logger.info(f"Circuit breaker '{self.name}' reiniciado manualmente")
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._half_open_attempts = 0


class CircuitBreakerRegistry:
    """
    Registro global de circuit breakers para acceso centralizado.

    Útil para monitorear y gestionar todos los circuit breakers desde un solo lugar.
    """

    _instance: Optional['CircuitBreakerRegistry'] = None
    _breakers: Dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, breaker: CircuitBreaker) -> None:
        """Registra un circuit breaker."""
        with self._lock:
            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Obtiene un circuit breaker por nombre."""
        with self._lock:
            return self._breakers.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estadísticas de todos los circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> None:
        """Reinicia todos los circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_health_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de salud de todos los circuit breakers."""
        with self._lock:
            total = len(self._breakers)
            open_breakers = [
                name for name, breaker in self._breakers.items()
                if breaker.get_state() == "open"
            ]
            half_open_breakers = [
                name for name, breaker in self._breakers.items()
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
