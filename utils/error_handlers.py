from collections.abc import Callable
from functools import wraps
import logging

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

    Args:
        error_types: Exception types to catch
        re_raise_as: Convert caught exception to this type
        log_level: Log level for error messages
        default_return: Value to return on error (if not raising)
        raise_on_error: Whether to re-raise the exception
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
    """Decorator that attempts recovery on error."""

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
