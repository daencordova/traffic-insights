"""Sistema unificado de logging estructurado con soporte para contexto y JSON.

Este módulo proporciona un sistema de logging avanzado que soporta:
- Contexto enriquecido para cada mensaje
- Formato JSON para integración con sistemas de monitoreo
- Múltiples niveles de logging
- Configuración centralizada
- Mixin para fácil integración en clases
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
import sys
import threading
from typing import Any

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_JSON_FORMAT = '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": %(message)s}'

COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET": "\033[0m",
}

RESERVED_LOG_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

_logging_initialized = False
_logging_lock = threading.Lock()


def setup_logging(
    level: str | int = "INFO",
    log_file: str | None = None,
    *,
    json_format: bool = False,
    colored: bool = True,
    force: bool = False,
) -> None:
    """Configura el sistema de logging global.

    Esta función debe llamarse una vez al inicio de la aplicación.
    Si se llama múltiples veces, solo se aplica la primera llamada a menos
    que se especifique force=True.

    Args:
        level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Ruta al archivo de log (opcional).
        json_format: Si usar formato JSON para los logs.
        colored: Si usar colores en la salida de consola.
        force: Forzar reconfiguración aunque ya esté inicializado.

    Example:
        >>> setup_logging(level="DEBUG", log_file="logs/app.log")
        >>> logging.info("Aplicación iniciada")
    """
    global _logging_initialized

    with _logging_lock:
        if _logging_initialized and not force:
            return

        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        if force:
            root_logger.handlers.clear()

        if json_format:
            base_formatter = logging.Formatter(DEFAULT_JSON_FORMAT, DEFAULT_DATE_FORMAT)
        else:
            base_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if colored:
            colored_formatter = _ColoredFormatter(
                DEFAULT_JSON_FORMAT if json_format else DEFAULT_LOG_FORMAT,
                datefmt=DEFAULT_DATE_FORMAT,
            )
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(base_formatter)

        root_logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(base_formatter)
            root_logger.addHandler(file_handler)

        _configure_third_party_logging()

        _logging_initialized = True

        logger = logging.getLogger(__name__)
        logger.info(
            f"📋 Logging configurado (nivel={logging.getLevelName(level)}, "
            f"json={json_format}, archivo={log_file or 'ninguno'})"
        )


def _configure_third_party_logging() -> None:
    """Configura niveles de logging para librerías de terceros."""
    third_party_loggers = {
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "PIL": logging.WARNING,
        "matplotlib": logging.WARNING,
        "numba": logging.WARNING,
        "onnxruntime": logging.WARNING,
        "torch": logging.WARNING,
        "ultralytics": logging.WARNING,
        "cv2": logging.WARNING,
        "asyncio": logging.WARNING,
    }

    for name, level in third_party_loggers.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)


def get_logging_status() -> dict[str, Any]:
    """Obtiene el estado actual del sistema de logging.

    Returns:
        dict: Estado del logging incluyendo nivel, handlers, etc.
    """
    root_logger = logging.getLogger()
    return {
        "initialized": _logging_initialized,
        "level": logging.getLevelName(root_logger.level),
        "handlers": [type(h).__name__ for h in root_logger.handlers],
        "effective_level": logging.getLevelName(root_logger.getEffectiveLevel()),
    }


class _ColoredFormatter(logging.Formatter):
    """Formatter que añade colores ANSI a los mensajes de consola."""

    def __init__(self, fmt: str = DEFAULT_LOG_FORMAT, datefmt: str = DEFAULT_DATE_FORMAT):
        """Inicializa el formatter con colores.

        Args:
            fmt: Formato del mensaje.
            datefmt: Formato de fecha.
        """
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Formatea el registro con colores."""
        levelname = record.levelname
        color = COLORS.get(levelname, COLORS["RESET"])
        reset = COLORS["RESET"]

        original_levelname = record.levelname
        record.levelname = f"{color}{original_levelname}{reset}"

        result = super().format(record)

        record.levelname = original_levelname

        return result


class StructuredLogger(logging.LoggerAdapter):
    """Logger estructurado con soporte para contexto y formato JSON.

    Este logger extiende la funcionalidad estándar de Python logging
    añadiendo contexto enriquecido y formato estructurado.

    Características:
        - Contexto enriquecido para cada mensaje
        - Formato JSON opcional
        - Múltiples niveles de logging
        - Soporte para extra fields

    Attributes:
        name: Nombre del logger.
        json_format: Si el formato debe ser JSON.
        context: Contexto actual para todos los mensajes.
        extra: Información extra adicional.

    Example:
        >>> logger = StructuredLogger("my_module")
        >>> logger.set_context(user_id=123)
        >>> logger.info("Procesando usuario", extra={"action": "login"})
    """

    def __init__(
        self,
        name: str,
        log_file: str | None = None,
        *,
        json_format: bool = False,
        level: int = logging.DEBUG,
    ) -> None:
        """Inicializa el logger estructurado.

        Args:
            name: Nombre identificador del logger.
            log_file: Ruta al archivo de log (opcional).
            json_format: Si se debe usar formato JSON para los mensajes.
            level: Nivel de logging para este logger.
        """
        self._name = name
        self._json_format = json_format
        self._context: dict[str, Any] = {}
        self._extra: dict[str, Any] = {}

        self._logger = logging.getLogger(name)

        if level != logging.DEBUG:
            self._logger.setLevel(level)

        if not self._logger.handlers and not _logging_initialized:
            setup_logging(level="INFO", log_file=log_file)

        super().__init__(self._logger, {})

    def set_context(self, **kwargs) -> None:
        """Establece contexto para los logs.

        El contexto se añade automáticamente a todos los mensajes subsiguientes.

        Args:
            **kwargs: Pares clave-valor para el contexto.

        Example:
            >>> logger.set_context(component="detector", frame_id=42)
            >>> logger.info("Frame procesado")
            >>> # [component=detector|frame_id=42] Frame procesado
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._context.update(filtered_kwargs)
        self._update_extra()

    def clear_context(self) -> None:
        """Limpia el contexto actual."""
        self._context.clear()
        self._update_extra()

    def set_extra(self, **kwargs) -> None:
        """Añade información extra al logger.

        Similar al contexto pero no se muestra en el mensaje principal,
        solo se incluye en formato JSON o en el extra del log.

        Args:
            **kwargs: Pares clave-valor para información extra.
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._extra.update(filtered_kwargs)
        self._update_extra()

    def clear_extra(self) -> None:
        """Limpia la información extra."""
        self._extra.clear()
        self._update_extra()

    def _update_extra(self) -> None:
        """Actualiza el extra del adapter con contexto y extra."""
        combined = {**self._context, **self._extra}
        self.extra = combined

    def _format_message(self, message: str, **kwargs) -> tuple[str, dict]:
        """Formatea el mensaje con contexto y kwargs adicionales.

        Args:
            message: Mensaje principal.
            **kwargs: Argumentos adicionales para incluir en el log.

        Returns:
            tuple: (mensaje_formateado, extra_dict)
        """
        extra = {}

        all_context = {**self._context, **kwargs}
        filtered_context = {k: v for k, v in all_context.items() if k not in RESERVED_LOG_ATTRS}

        if filtered_context and not self._json_format:
            context_str = " | ".join(f"{k}={v}" for k, v in filtered_context.items())
            formatted = f"[{context_str}] {message}"
        else:
            formatted = message

        if self._json_format:
            extra["context"] = filtered_context

        for key, value in kwargs.items():
            if key not in RESERVED_LOG_ATTRS and key not in extra:
                extra[key] = value

        return formatted, extra

    def log(
        self,
        level: int,
        msg: str,
        *args,
        exc_info=None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: dict | None = None,
        **kwargs,
    ) -> None:
        """Registra un mensaje con el nivel especificado."""
        formatted_msg, context_extra = self._format_message(msg, **kwargs)

        combined_extra = {**(extra or {}), **context_extra}
        combined_extra = {k: v for k, v in combined_extra.items() if k not in RESERVED_LOG_ATTRS}

        super().log(
            level,
            formatted_msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=combined_extra,
        )

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Registra un mensaje de depuración."""
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Registra un mensaje informativo."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Registra un mensaje de advertencia."""
        self.log(logging.WARNING, msg, *args, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:
        """Registra un mensaje de advertencia (alias de warning)."""
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Registra un mensaje de error."""
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Registra un mensaje de error crítico."""
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Registra una excepción con traceback."""
        self.error(msg, *args, exc_info=True, **kwargs)

    @property
    def name(self) -> str:
        """Nombre del logger."""
        return self._name

    @property
    def json_format(self) -> bool:
        """Si el formato es JSON."""
        return self._json_format

    def __repr__(self) -> str:
        return f"StructuredLogger(name='{self._name}', json={self._json_format})"


class LoggerMixin:
    """Mixin para agregar logging estructurado a clases.

    Proporciona un logger configurado automáticamente para cada clase
    que hereda de este mixin.

    Attributes:
        logger: Instancia de StructuredLogger para la clase.

    Example:
        >>> class MyClass(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("Procesando...")
    """

    _logger: StructuredLogger | None = None
    _log_context: dict[str, Any] = {}

    @property
    def logger(self) -> StructuredLogger:
        """Obtiene un logger estructurado para la clase.

        El logger se crea automáticamente con el nombre de la clase
        y se configura con contexto básico.

        Returns:
            StructuredLogger: Logger configurado para la clase.
        """
        if self._logger is None:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"

            log_file = None
            if hasattr(self, "config") and hasattr(self.config, "output"):
                logs_dir = getattr(self.config.output, "logs_dir", "data/logs/")
                if logs_dir:
                    log_file = str(Path(logs_dir) / f"{self.__class__.__name__.lower()}.log")

            self._logger = StructuredLogger(
                name=logger_name,
                log_file=log_file,
                json_format=False,
            )

            filtered_context = {
                k: v
                for k, v in {
                    "class_name": self.__class__.__name__,
                    "module": self.__class__.__module__,
                    **self._log_context,
                }.items()
                if k not in RESERVED_LOG_ATTRS
            }
            self._logger.set_context(**filtered_context)

        return self._logger

    def set_log_context(self, **kwargs) -> None:
        """Establece contexto adicional para los logs de la clase.

        Args:
            **kwargs: Pares clave-valor para el contexto.

        Example:
            >>> class MyTracker(LoggerMixin):
            ...     def track(self, frame_id):
            ...         self.set_log_context(frame_id=frame_id)
            ...         self.logger.info("Procesando frame")
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._log_context.update(filtered_kwargs)
        if self._logger:
            self._logger.set_context(**filtered_kwargs)

    def clear_log_context(self) -> None:
        """Limpia el contexto del logger de la clase."""
        self._log_context.clear()
        if self._logger:
            self._logger.clear_context()

    def log_error_with_context(
        self,
        error: Exception,
        message: str | None = None,
        **kwargs,
    ) -> None:
        """Registra un error con contexto completo.

        Args:
            error: La excepción capturada.
            message: Mensaje adicional (opcional).
            **kwargs: Contexto adicional.

        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     self.log_error_with_context(e, "Risky operation failed", operation="risky_op")
        """
        error_type = type(error).__name__
        error_msg = str(error) if str(error) else "No details available"

        log_message = f"{message}: " if message else ""
        log_message += f"{error_type}: {error_msg}"

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self.logger.error(
            log_message,
            error_type=error_type,
            error_details=error_msg,
            **filtered_kwargs,
        )


def get_logger(
    name: str,
    log_file: str | None = None,
    *,
    json_format: bool = False,
) -> StructuredLogger:
    """Obtiene un logger estructurado por nombre.

    Esta función mantiene compatibilidad con código existente
    que espera un logger estándar de Python.

    Args:
        name: Nombre del logger.
        log_file: Ruta al archivo de log (opcional).
        json_format: Si usar formato JSON.

    Returns:
        StructuredLogger: Logger estructurado.

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Mensaje con contexto")
    """
    return StructuredLogger(name, log_file=log_file, json_format=json_format)


def setup_logger(
    name: str = "vehicle_counter",
    log_file: str | None = None,
    level: int = logging.INFO,
    *,
    json_format: bool = False,
) -> logging.Logger:
    """Función de compatibilidad con código existente.

    Args:
        name: Nombre del logger.
        log_file: Ruta del archivo de log (opcional).
        level: Nivel de logging.
        json_format: Si usar formato JSON.

    Returns:
        logging.Logger: Logger configurado.
    """
    if not _logging_initialized:
        setup_logging(level=level, log_file=log_file, json_format=json_format)

    return logging.getLogger(name)


__all__ = [
    "setup_logging",
    "get_logging_status",
    "StructuredLogger",
    "LoggerMixin",
    "get_logger",
    "setup_logger",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
]
