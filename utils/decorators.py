"""Utility decorators for error handling and performance.

This module provides decorators for common operations:
    - Retry on failure with exponential backoff
    - Suppress errors with default return values
    - Measure and log execution time
    - Configurable retry and suppression settings

Example:
    >>> from utils.decorators import (
    ...     retry_on_failure,
    ...     RetryConfig,
    ...     suppress_errors,
    ...     SuppressConfig,
    ...     time_operation,
    ... )
    >>>
    >>> # Retry on connection errors
    >>> config = RetryConfig(
    ...     exceptions=(ConnectionError, TimeoutError), max_attempts=5, delay=0.1, backoff=2.0
    ... )
    >>>
    >>> @retry_on_failure(config)
    ... def connect_to_camera():
    ...     # Attempt connection
    ...     pass
    >>>
    >>> # Suppress errors and return empty list
    >>> config = SuppressConfig(default_return=[])
    >>> @suppress_errors(config)
    ... def get_detections():
    ...     # May fail, returns [] on error
    ...     pass
    >>>
    >>> # Measure execution time
    >>> @time_operation(threshold_ms=50.0)
    ... def heavy_operation():
    ...     # Function with performance tracking
    ...     pass
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry decorator.

    Attributes:
        exceptions: Exception types to catch and retry.
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between attempts in seconds.
        backoff: Multiplicative backoff factor for delay.
        max_delay: Maximum delay between attempts.
        on_retry: Callback when a retry occurs (attempt, exception).
        on_failure: Callback when all attempts fail (exception).
        raise_original: Whether to raise the original exception.

    Example:
        >>> config = RetryConfig(
        ...     exceptions=(ConnectionError, TimeoutError),
        ...     max_attempts=5,
        ...     delay=0.5,
        ...     backoff=2.0,
        ...     max_delay=30.0,
        ... )
    """

    exceptions: tuple[type[Exception], ...] = (Exception,)
    max_attempts: int = 3
    delay: float = 0.5
    backoff: float = 2.0
    max_delay: float = 30.0
    on_retry: Callable[[int, Exception], None] | None = None
    on_failure: Callable[[Exception], None] | None = None
    raise_original: bool = True


@dataclass
class SuppressConfig:
    """Configuration for error suppression decorator.

    Attributes:
        exceptions: Exception types to suppress.
        default_return: Value to return on error.
        log_error: Whether to log the error.
        log_level: Log level for error messages.

    Example:
        >>> config = SuppressConfig(
        ...     exceptions=(ValueError, KeyError),
        ...     default_return=None,
        ...     log_error=True,
        ...     log_level="warning",
        ... )
    """

    exceptions: tuple[type[Exception], ...] = (Exception,)
    default_return: Any = None
    log_error: bool = True
    log_level: str = "warning"


def retry_on_failure(config: RetryConfig):
    """Decorator to retry a function on failure.

    This decorator wraps a function with retry logic, attempting
    to execute it multiple times with exponential backoff on failure.

    Args:
        config: Retry configuration.

    Returns:
        Configured decorator.

    Example:
        >>> config = RetryConfig(
        ...     exceptions=(ConnectionError, TimeoutError), max_attempts=5, delay=0.1, backoff=2.0
        ... )
        >>>
        >>> @retry_on_failure(config)
        ... def connect_to_camera():
        ...     # Will retry up to 5 times on connection errors
        ...     return camera.connect()
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _delay = config.delay
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except config.exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        error_msg = (
                            f"Failed after {config.max_attempts} attempts in {func.__name__}"
                        )
                        logger.error(f"{error_msg}: {e}")

                        if config.on_failure:
                            config.on_failure(e)

                        if config.raise_original:
                            raise
                        raise last_exception from e

                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed in {func.__name__}: {e}. "
                        f"Retrying in {_delay:.2f}s..."
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e)

                    time.sleep(_delay)
                    _delay = min(_delay * config.backoff, config.max_delay)

            return None

        return wrapper

    return decorator


def suppress_errors(config: SuppressConfig):
    """Decorator to suppress errors and return a default value.

    This decorator wraps a function and catches specified exceptions,
    returning a default value instead of propagating the error.

    Args:
        config: Suppression configuration.

    Returns:
        Configured decorator.

    Example:
        >>> config = SuppressConfig(
        ...     exceptions=(ValueError, KeyError), default_return=[], log_error=True
        ... )
        >>>
        >>> @suppress_errors(config)
        ... def get_detections():
        ...     # Returns [] on ValueError or KeyError
        ...     return detector.detect(frame)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except config.exceptions as e:
                if config.log_error:
                    log_func = getattr(logger, config.log_level, logger.warning)
                    log_func(
                        f"Suppressed error in {func.__name__}: {e}",
                        exc_info=config.log_level in ("debug", "error"),
                    )
                return config.default_return

        return wrapper

    return decorator


def time_operation(log_level: str = "debug", *, threshold_ms: float = 100.0):
    """Decorator to measure and log execution time.

    This decorator wraps a function and logs its execution time,
    with warnings for operations that exceed the specified threshold.

    Args:
        log_level: Log level for timing messages.
        threshold_ms: Threshold in milliseconds to log as warning.

    Returns:
        Configured decorator.

    Example:
        >>> @time_operation(threshold_ms=50.0)
        ... def heavy_operation():
        ...     # Takes time to execute
        ...     time.sleep(0.1)  # Will trigger warning
        >>>
        >>> @time_operation(log_level="info")
        ... def normal_operation():
        ...     # Logs at info level
        ...     pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000

                log_func = getattr(logger, log_level, logger.debug)
                if elapsed_ms > threshold_ms:
                    log_func = logger.warning

                log_func(f"{func.__name__} executed in {elapsed_ms:.2f}ms")

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{func.__name__} failed after {elapsed_ms:.2f}ms: {e}")
                raise

        return wrapper

    return decorator
