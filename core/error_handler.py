"""
Manejador global de errores para el sistema.
Proporciona recuperación y logging consistente.
"""

import sys
import traceback
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from core.exceptions import VehicleCountingError
from core.circuit_breaker import circuit_breaker_registry


class GlobalErrorHandler:
    """
    Manejador global de errores que captura excepciones no manejadas
    y proporciona recuperación automática cuando es posible.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("error_handler")
        self._error_count = 0
        self._last_error_time: Optional[datetime] = None
        self._error_threshold = 10
        self._error_window = 60.0
        self._recovery_callbacks: Dict[str, Callable] = {}
        self._is_recovering = False

    def register_recovery(self, name: str, callback: Callable) -> None:
        """
        Registra un callback para recuperación automática.

        Args:
            name: Identificador del callback.
            callback: Función que intenta recuperar el sistema.
        """
        self._recovery_callbacks[name] = callback

    def handle_exception(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Maneja una excepción y decide si se puede recuperar.

        Args:
            exc: Excepción capturada.
            context: Contexto adicional para logging.

        Returns:
            bool: True si se pudo recuperar, False si es fatal.
        """
        self._update_error_stats()

        self._log_error(exc, context)

        if not isinstance(exc, VehicleCountingError):
            return self._handle_system_error(exc)

        if self._can_recover(exc):
            return self._attempt_recovery(exc)

        self.logger.critical(f"Error fatal no recuperable: {exc}")
        return False

    def _update_error_stats(self) -> None:
        """Actualiza estadísticas de errores."""
        current_time = datetime.now()

        if self._last_error_time:
            elapsed = (current_time - self._last_error_time).total_seconds()
            if elapsed > self._error_window:
                self._error_count = 0

        self._error_count += 1
        self._last_error_time = current_time

        if self._error_count > self._error_threshold:
            self.logger.error(
                f"Demasiados errores ({self._error_count} en {self._error_window}s). "
                "Posible degradación del sistema."
            )

    def _log_error(self, exc: Exception, context: Optional[Dict[str, Any]]) -> None:
        """Registra el error con contexto detallado."""
        error_type = type(exc).__name__
        error_msg = str(exc) if str(exc) else "No details available"

        log_data = {
            "error_type": error_type,
            "error_message": error_msg,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
        }

        if context:
            log_data["context"] = context

        if isinstance(exc, VehicleCountingError) and exc.details:
            log_data["details"] = exc.details

        self.logger.error(
            f"Error no manejado: {error_type}: {error_msg}",
            extra=log_data
        )

    def _can_recover(self, exc: Exception) -> bool:
        """Determina si un error es recuperable."""
        from core.exceptions import (
            CameraError, CaptureError, ConnectionError,
            TimeoutError, IOError
        )

        recoverable_types = (
            CameraError,
            CaptureError,
            ConnectionError,
            TimeoutError,
            IOError,
        )

        if isinstance(exc, recoverable_types):
            return True

        if isinstance(exc, VehicleCountingError):
            return "recoverable" in exc.details.get("metadata", "")

        return False

    def _attempt_recovery(self, exc: Exception) -> bool:
        """Intenta recuperar el sistema después de un error."""
        if self._is_recovering:
            self.logger.warning("Ya en proceso de recuperación")
            return False

        self._is_recovering = True

        try:
            self.logger.info("Intentando recuperación automática...")

            circuit_breaker_registry.reset_all()

            for name, callback in self._recovery_callbacks.items():
                try:
                    self.logger.info(f"Ejecutando recuperación: {name}")
                    callback()
                except Exception as e:
                    self.logger.error(f"Error en recuperación {name}: {e}")

            self.logger.info("Recuperación completada")
            return True

        finally:
            self._is_recovering = False

    def _handle_system_error(self, exc: Exception) -> bool:
        """Maneja errores del sistema (no del dominio)."""
        if isinstance(exc, (MemoryError, SystemError)):
            self.logger.critical(f"Error crítico del sistema: {exc}")
            return False

        self.logger.warning(f"Error del sistema, intentando recuperación: {exc}")
        return self._attempt_recovery(exc)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del manejador de errores."""
        return {
            "total_errors": self._error_count,
            "last_error": self._last_error_time.isoformat() if self._last_error_time else None,
            "error_rate": self._error_count / self._error_window if self._error_count > 0 else 0,
            "recovery_callbacks": list(self._recovery_callbacks.keys()),
            "is_recovering": self._is_recovering,
        }


global_error_handler = GlobalErrorHandler()


def setup_global_exception_handler():
    """
    Configura el manejador global de excepciones.
    Debe llamarse al inicio del programa.
    """
    def global_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        global_error_handler.handle_exception(exc_value, {
            "exc_type": exc_type.__name__,
            "traceback": traceback.format_tb(exc_traceback)
        })

    sys.excepthook = global_handler
