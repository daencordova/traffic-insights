"""Global error handler for the system. Provides recovery and consistent logging.

This module implements a comprehensive error handling system with:
- Global exception catching and logging
- Automatic recovery for recoverable errors
- Circuit breaker integration for cascading failure prevention
- Error statistics and monitoring
- Decorators for consistent exception handling
- Recovery callbacks for system restoration
"""

from collections.abc import Callable
from datetime import datetime
from functools import wraps
import logging
import sys
import time
import traceback
from typing import Any

from core.circuit_breaker import circuit_breaker_registry
from core.exceptions import (
    CameraError,
    CaptureError,
    ConnectionError,
    IOError,
    TimeoutError,
    VehicleCountingError,
)
from utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class GlobalErrorHandler(LoggerMixin):
    """Global error handler that catches unhandled exceptions and provides automatic recovery when possible.

    This class manages system-wide error handling with:
        - Error counting and rate limiting
        - Automatic recovery for recoverable errors
        - Circuit breaker integration
        - Recovery callbacks registration
        - Error statistics collection

    Attributes:
        _error_count: Number of errors in the current window.
        _last_error_time: Timestamp of the last error.
        _error_threshold: Maximum errors before warning.
        _error_window: Time window for error counting in seconds.
        _recovery_callbacks: Dictionary of recovery callbacks.
        _is_recovering: Flag indicating if recovery is in progress.

    Example:
        >>> handler = GlobalErrorHandler()
        >>> handler.register_recovery("camera", lambda: reconnect_camera())
        >>>
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     if not handler.handle_exception(e):
        ...         print("Fatal error, shutting down")
    """

    def __init__(self):
        """Initializes the global error handler."""
        self._error_count = 0
        self._last_error_time: datetime | None = None
        self._error_threshold = 10
        self._error_window = 60.0
        self._recovery_callbacks: dict[str, Callable] = {}
        self._is_recovering = False

        self.logger.info("GlobalErrorHandler initialized")

    def register_recovery(self, name: str, callback: Callable) -> None:
        """Registers a callback for automatic recovery.

        Args:
            name: Identifier for the callback.
            callback: Function that attempts to recover the system.

        Example:
            >>> handler.register_recovery("database_connection", lambda: reconnect_to_database())
        """
        self._recovery_callbacks[name] = callback

    def handle_exception(self, exc: Exception, context: dict[str, Any] | None = None) -> bool:
        """Handles an exception and decides if recovery is possible.

        This method processes exceptions by:
            1. Updating error statistics
            2. Logging the error with context
            3. Determining if the error is recoverable
            4. Attempting recovery if possible

        Args:
            exc: The exception to handle.
            context: Additional context about the error (optional).

        Returns:
            bool: True if the system can continue, False if fatal.

        Example:
            >>> try:
            ...     process_frame()
            ... except CameraError as e:
            ...     if not handler.handle_exception(e, {"frame_id": 123}):
            ...         shutdown_system()
        """
        self._update_error_stats()

        error_type = type(exc).__name__
        error_msg = str(exc) if str(exc) else "No details available"

        self.logger.error(
            f"Unhandled error: {error_type}: {error_msg}",
            exc_info=True,
            error_type=error_type,
            error_details=error_msg,
            context=context,
            **(
                exc.details
                if isinstance(exc, VehicleCountingError) and hasattr(exc, "details")
                else {}
            ),
        )

        if not isinstance(exc, VehicleCountingError):
            return self._handle_system_error(exc)

        if self._can_recover(exc):
            return self._attempt_recovery(exc)

        self.logger.critical(f"Fatal unrecoverable error: {exc}")
        return False

    def _update_error_stats(self) -> None:
        """Updates error statistics and checks for error thresholds."""
        current_time = datetime.now()

        if self._last_error_time:
            elapsed = (current_time - self._last_error_time).total_seconds()
            if elapsed > self._error_window:
                self._error_count = 0

        self._error_count += 1
        self._last_error_time = current_time

        if self._error_count > self._error_threshold:
            self.logger.error(
                f"Too many errors ({self._error_count} in {self._error_window}s). "
                "Possible system degradation."
            )

    def _log_error(self, exc: Exception, context: dict[str, Any] | None) -> None:
        """Logs the error with detailed context.

        Args:
            exc: The exception to log.
            context: Additional context about the error.
        """
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

        self.logger.error(f"Unhandled error: {error_type}: {error_msg}", extra=log_data)

    def _can_recover(self, exc: Exception) -> bool:
        """Determines if an error is recoverable.

        Args:
            exc: The exception to check.

        Returns:
            bool: True if the error is recoverable, False otherwise.
        """
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
        """Attempts to recover the system after an error.

        This method:
            1. Resets all circuit breakers
            2. Executes all registered recovery callbacks
            3. Logs recovery progress

        Args:
            exc: The exception that triggered recovery.

        Returns:
            bool: True if recovery was successful, False otherwise.
        """
        if self._is_recovering:
            self.logger.warning("Recovery already in progress")
            return False

        self._is_recovering = True

        try:
            self.logger.info("Attempting automatic recovery...")

            circuit_breaker_registry.reset_all()

            for name, callback in self._recovery_callbacks.items():
                try:
                    self.logger.info(f"Executing recovery: {name}")
                    callback()
                except Exception as e:
                    self.logger.error(f"Error in recovery {name}: {e}")

            self.logger.info("Recovery completed")
            return True

        finally:
            self._is_recovering = False

    def _handle_system_error(self, exc: Exception) -> bool:
        """Handles system errors (non-domain errors).

        Args:
            exc: The system exception to handle.

        Returns:
            bool: True if the system can continue, False if fatal.
        """
        if isinstance(exc, (MemoryError, SystemError)):
            self.logger.critical(f"Critical system error: {exc}")
            return False

        self.logger.warning(f"System error, attempting recovery: {exc}")
        return self._attempt_recovery(exc)

    def get_stats(self) -> dict[str, Any]:
        """Gets statistics for the error handler.

        Returns:
            dict: Error statistics including:
                - total_errors: Number of errors in current window
                - last_error: Timestamp of last error
                - error_rate: Errors per second in current window
                - recovery_callbacks: List of registered callbacks
                - is_recovering: Whether recovery is in progress

        Example:
            >>> stats = handler.get_stats()
            >>> print(f"Error rate: {stats['error_rate']:.2f} errors/sec")
        """
        return {
            "total_errors": self._error_count,
            "last_error": self._last_error_time.isoformat() if self._last_error_time else None,
            "error_rate": self._error_count / self._error_window if self._error_count > 0 else 0,
            "recovery_callbacks": list(self._recovery_callbacks.keys()),
            "is_recovering": self._is_recovering,
        }

    def attempt_recovery(self, error: Exception) -> bool:
        """Attempts to recover the system after an error.

        Args:
            error: Exception that caused the error.

        Returns:
            bool: True if recovery was successful.

        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     if handler.attempt_recovery(e):
            ...         print("System recovered")
            ...     else:
            ...         print("System cannot recover")
        """
        return self._attempt_recovery(error)


global_error_handler = GlobalErrorHandler()


def setup_global_exception_handler():
    """Sets up the global exception handler. Should be called at program startup.

    This function installs a global exception handler that catches all
    unhandled exceptions and routes them through the GlobalErrorHandler.

    Example:
        >>> setup_global_exception_handler()
        >>> # All unhandled exceptions will now be handled automatically
    """

    def global_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        global_error_handler.handle_exception(
            exc_value,
            {"exc_type": exc_type.__name__, "traceback": traceback.format_tb(exc_traceback)},
        )

    sys.excepthook = global_handler


def handle_exceptions(
    error_types: tuple[type[Exception], ...] = (Exception,),
    re_raise_as: type[VehicleCountingError] | None = None,
    log_level: str = "warning",
    default_return: object | None = None,
    raise_on_error: bool = False,
) -> Callable:
    """Decorator for consistent exception handling.

    This decorator provides a uniform way to handle exceptions across
    the application with configurable behavior.

    Args:
        error_types: Exception types to catch (default: all exceptions).
        re_raise_as: Convert caught exception to this type (optional).
        log_level: Log level for error messages ('debug', 'info', 'warning', 'error').
        default_return: Value to return on error (if not raising).
        raise_on_error: Whether to re-raise the exception.

    Returns:
        Callable: Decorated function with exception handling.

    Example:
        >>> @handle_exceptions(
        ...     error_types=(ConnectionError, TimeoutError), log_level="error", default_return=False
        ... )
        ... def connect_to_server():
        ...     return server.connect()
        >>>
        >>> result = connect_to_server()  # Returns False on error
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                log_func = getattr(logger, log_level)
                log_func(f"Error in {func.__name__}: {e}", exc_info=True)

                if re_raise_as:
                    raise re_raise_as(str(e)) from e
                if raise_on_error:
                    raise

                return default_return

        return wrapper

    return decorator


def recover_on_error(
    recovery_func: Callable | None = None, max_attempts: int = 3, delay: float = 0.5
) -> Callable:
    """Decorator that attempts recovery on error.

    This decorator retries a function with exponential-like behavior
    and optional recovery function execution between attempts.

    Args:
        recovery_func: Function to call before each retry (optional).
        max_attempts: Maximum number of retry attempts.
        delay: Delay in seconds between attempts.

    Returns:
        Callable: Decorated function with retry logic.

    Example:
        >>> @recover_on_error(
        ...     recovery_func=lambda: reconnect_to_camera(), max_attempts=5, delay=1.0
        ... )
        ... def read_from_camera():
        ...     return camera.read()
        >>>
        >>> frame = read_from_camera()  # Will retry up to 5 times
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if recovery_func:
                        try:
                            recovery_func(*args, **kwargs)
                        except Exception as rec_error:
                            logger.warning(f"Recovery failed: {rec_error}")

                    logger.warning(
                        f"Error in {func.__name__}, attempt {attempt + 1}/{max_attempts}: {e}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(delay)

            raise last_error

        return wrapper

    return decorator
