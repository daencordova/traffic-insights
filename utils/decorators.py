"""
Decoradores utilitarios para manejo de errores y rendimiento.
"""

import time
import logging
from functools import wraps
from typing import Type, Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)


def retry_on_failure(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    raise_original: bool = True,
):
    """
    Decorador para reintentar una función en caso de fallo.

    Args:
        exceptions: Tupla de excepciones que activan el reintento.
        max_attempts: Número máximo de intentos.
        delay: Delay inicial entre intentos (segundos).
        backoff: Factor de aumento del delay.
        max_delay: Delay máximo permitido.
        on_retry: Callback opcional que se ejecuta en cada reintento.
        on_failure: Callback opcional que se ejecuta tras todos los fallos.
        raise_original: Si levantar la excepción original o la última.

    Returns:
        Decorador configurado.

    Ejemplo:
        @retry_on_failure(
            exceptions=(ConnectionError, TimeoutError),
            max_attempts=5,
            delay=0.1,
            backoff=2.0
        )
        def connect_to_camera():
            # ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        error_msg = f"Fallo después de {max_attempts} intentos en {func.__name__}"
                        logger.error(f"{error_msg}: {e}")

                        if on_failure:
                            on_failure(e)

                        if raise_original:
                            raise
                        else:
                            raise last_exception

                    logger.warning(
                        f"Intento {attempt}/{max_attempts} falló en {func.__name__}: {e}. "
                        f"Reintentando en {_delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    time.sleep(_delay)
                    _delay = min(_delay * backoff, max_delay)

            return None
        return wrapper
    return decorator


def suppress_errors(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    default_return: Any = None,
    log_error: bool = True,
    log_level: str = "warning",
):
    """
    Decorador para suprimir errores y retornar un valor por defecto.

    Args:
        exceptions: Tupla de excepciones a suprimir.
        default_return: Valor a retornar en caso de error.
        log_error: Si registrar el error.
        log_level: Nivel de logging ('debug', 'info', 'warning', 'error').

    Ejemplo:
        @suppress_errors(default_return=[])
        def get_detections():
            # ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    log_func = getattr(logger, log_level, logger.warning)
                    log_func(
                        f"Error suprimido en {func.__name__}: {e}",
                        exc_info=log_level in ("debug", "error")
                    )
                return default_return
        return wrapper
    return decorator


def time_operation(log_level: str = "debug", threshold_ms: float = 100.0):
    """
    Decorador para medir y loggear el tiempo de ejecución.

    Args:
        log_level: Nivel de logging para el mensaje.
        threshold_ms: Umbral para loggear como advertencia si supera.

    Ejemplo:
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
