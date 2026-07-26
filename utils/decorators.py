"""Decoradores utilitarios para manejo de errores y rendimiento."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuración para el decorador de reintentos."""

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
    """Configuración para el decorador de supresión de errores."""

    exceptions: tuple[type[Exception], ...] = (Exception,)
    default_return: Any = None
    log_error: bool = True
    log_level: str = "warning"


def retry_on_failure(config: RetryConfig):
    """Decorador para reintentar una función en caso de fallo.

    Args:
        config: Configuración de reintentos.

    Returns:
        Decorador configurado.

    Example:
        config = RetryConfig(
            exceptions=(ConnectionError, TimeoutError),
            max_attempts=5,
            delay=0.1,
            backoff=2.0
        )

        @retry_on_failure(config)
        def connect_to_camera():
            # ...
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
                            f"Fallo después de {config.max_attempts} intentos en {func.__name__}"
                        )
                        logger.error(f"{error_msg}: {e}")

                        if config.on_failure:
                            config.on_failure(e)

                        if config.raise_original:
                            raise
                        raise last_exception from e

                    logger.warning(
                        f"Intento {attempt}/{config.max_attempts} falló en {func.__name__}: {e}. "
                        f"Reintentando en {_delay:.2f}s..."
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e)

                    time.sleep(_delay)
                    _delay = min(_delay * config.backoff, config.max_delay)

            return None

        return wrapper

    return decorator


def suppress_errors(config: SuppressConfig):
    """Decorador para suprimir errores y retornar un valor por defecto.

    Args:
        config: Configuración de supresión.

    Returns:
        Decorador configurado.

    Example:
        config = SuppressConfig(default_return=[])

        @suppress_errors(config)
        def get_detections():
            # ...
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
                        f"Error suprimido en {func.__name__}: {e}",
                        exc_info=config.log_level in ("debug", "error"),
                    )
                return config.default_return

        return wrapper

    return decorator


def time_operation(log_level: str = "debug", *, threshold_ms: float = 100.0):
    """Decorador para medir y loggear el tiempo de ejecución.

    Args:
        log_level: Nivel de logging para el mensaje.
        threshold_ms: Umbral para loggear como advertencia si supera.

    Example:
        @time_operation(threshold_ms=50.0)
        def heavy_operation():
            # ...
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

                log_func(f"{func.__name__} ejecutado en {elapsed_ms:.2f}ms")

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{func.__name__} falló después de {elapsed_ms:.2f}ms: {e}")
                raise

        return wrapper

    return decorator
