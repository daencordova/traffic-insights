"""Exception handling decorators.

This module provides decorators for consistent exception handling
and error recovery across the system.

Features:
    - Consistent exception handling with logging
    - Automatic retry with recovery on error
    - Configurable error conversion
    - Support for default return values

Example:
    >>> from utils.decorators import handle_exceptions, recover_on_error
    >>> from core.exceptions import VehicleCountingError
    >>>
    >>> @handle_exceptions(
    ...     error_types=(ValueError, KeyError),
    ...     re_raise_as=VehicleCountingError,
    ...     log_level="error",
    ...     default_return=False,
    ... )
    ... def process_detection(data):
    ...     # Will convert ValueError/KeyError to VehicleCountingError
    ...     return data["validated"]
    >>>
    >>> @recover_on_error(recovery_func=lambda: reconnect(), max_attempts=3, delay=0.5)
    ... def connect_to_source():
    ...     # Will retry up to 3 times with recovery
    ...     return source.connect()
"""

from collections.abc import Callable
from functools import wraps
import logging
import time

from core.exceptions import VehicleCountingError

logger = logging.getLogger(__name__)


def handle_exceptions(
    error_types: tuple[type[Exception], ...] = (Exception,),
    re_raise_as: type[VehicleCountingError] | None = None,
    log_level: str = "warning",
    default_return: object | None = None,
    raise_on_error: bool = False,
) -> Callable:
    """Decorator for consistent exception handling.

    This decorator provides a unified way to handle exceptions
    with configurable logging, conversion, and return behavior.

    Args:
        error_types: Exception types to catch (default: all exceptions).
        re_raise_as: Convert caught exception to this type (optional).
        log_level: Log level for error messages ('debug', 'info', 'warning', 'error').
        default_return: Value to return on error (if not raising).
        raise_on_error: Whether to re-raise the exception.

    Returns:
        Callable: Decorated function with exception handling.

    Example:
        >>> # Log and return False on error
        >>> @handle_exceptions(error_types=(ValueError,), log_level="error", default_return=False)
        ... def validate_data(data):
        ...     return data.is_valid()
        >>>
        >>> # Convert to domain exception
        >>> @handle_exceptions(
        ...     error_types=(IOError, ConnectionError),
        ...     re_raise_as=VehicleCountingError,
        ...     log_level="critical",
        ... )
        ... def load_config():
        ...     return config.load()
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

    This decorator wraps a function with retry logic, attempting
    to recover and re-execute the function on failure.

    Args:
        recovery_func: Function to call before each retry (optional).
        max_attempts: Maximum number of retry attempts.
        delay: Delay between attempts in seconds.

    Returns:
        Callable: Decorated function with recovery logic.

    Example:
        >>> # Simple retry with delay
        >>> @recover_on_error(max_attempts=5, delay=1.0)
        ... def read_from_camera():
        ...     return camera.read()
        >>>
        >>> # With recovery callback
        >>> @recover_on_error(recovery_func=lambda: reconnect_camera(), max_attempts=3, delay=0.5)
        ... def capture_frame():
        ...     return camera.capture()
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
