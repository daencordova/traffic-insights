"""Sistema unificado de logging para el sistema de seguimiento de tráfico.

Este módulo proporciona una interfaz única y consistente para logging en
todo el proyecto, eliminando duplicación y asegurando formato uniforme.

Características:
    - Logger único por clase con contexto automático
    - Formato estructurado con colores en consola
    - Soporte para JSON en producción
    - Contexto enriquecido para cada mensaje
    - Niveles de logging configurables por módulo
    - Rotación de archivos de log
    - Múltiples salidas (consola, archivo, syslog)

Ejemplo de uso:
    >>> from utils.logger import get_logger
    >>>
    >>> class MyClass:
    ...     logger = get_logger_for_class(MyClass)
    ...
    ...     def process(self):
    ...         self.logger.info("Procesando...", extra={"frame_id": 42})
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import json
import logging
import logging.config
import logging.handlers
from pathlib import Path
import sys
import threading
from typing import Any, ClassVar, TypeVar

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_JSON_FORMAT = '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": %(message)s}'

LOG_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET": "\033[0m",
}

LOG_LEVEL_NAMES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
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

_logging_initialized: bool = False
_logging_lock: threading.Lock = threading.Lock()
_default_logger: StructuredLogger | None = None


class LoggingConfig:
    """Configuración global del sistema de logging."""

    _instance: ClassVar[LoggingConfig | None] = None

    def __new__(cls) -> LoggingConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.level: int = logging.INFO
        self.json_format: bool = False
        self.colored: bool = True
        self.log_file: Path | None = None
        self.max_bytes: int = 10 * 1024 * 1024
        self.backup_count: int = 5
        self.module_levels: dict[str, int] = {}
        self.third_party_level: int = logging.WARNING

    def update(self, **kwargs) -> None:
        """Actualiza la configuración."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if _logging_initialized:
            setup_logging(force=True)


config = LoggingConfig()


class ColoredFormatter(logging.Formatter):
    """Formatter que añade colores ANSI a los mensajes de consola."""

    def __init__(self, fmt: str = DEFAULT_LOG_FORMAT, datefmt: str = DEFAULT_DATE_FORMAT):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = LOG_COLORS.get(levelname, LOG_COLORS["RESET"])
        reset = LOG_COLORS["RESET"]

        original_levelname = record.levelname
        record.levelname = f"{color}{original_levelname}{reset}"

        result = super().format(record)
        record.levelname = original_levelname

        return result


class StructuredFormatter(logging.Formatter):
    """Formatter estructurado para producción."""

    def __init__(self, json_format: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._json_format = json_format

    def format(self, record: logging.LogRecord) -> str:
        if self._json_format:
            return self._format_json(record)
        return super().format(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Formatea el registro como JSON."""
        extra = {}
        for key, value in record.__dict__.items():
            if key not in RESERVED_LOG_ATTRS:
                extra[key] = value

        log_data = {
            "timestamp": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "extra": extra or None,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class StructuredLogger(logging.LoggerAdapter):
    """Logger estructurado con soporte para contexto.

    Attributes:
        name: Nombre del logger.
        json_format: Si el formato debe ser JSON.
        context: Contexto actual para todos los mensajes.
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
            json_format: Si se debe usar formato JSON.
            level: Nivel de logging para este logger.
        """
        self._name = name
        self._json_format = json_format
        self._context: dict[str, Any] = {}
        self._extra: dict[str, Any] = {}
        self._level = level

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = True

        if not _logging_initialized:
            setup_logging()

        super().__init__(self._logger, {})

    def set_context(self, **kwargs) -> None:
        """Establece contexto para los logs.

        Args:
            **kwargs: Pares clave-valor para el contexto.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._context.update(filtered)
        self._update_extra()

    def add_context(self, **kwargs) -> None:
        """Añade contexto sin eliminar el existente.

        Args:
            **kwargs: Pares clave-valor para añadir al contexto.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._context.update(filtered)
        self._update_extra()

    def clear_context(self) -> None:
        """Limpia el contexto actual."""
        self._context.clear()
        self._update_extra()

    def get_context(self) -> dict[str, Any]:
        """Obtiene una copia del contexto actual."""
        return self._context.copy()

    def set_extra(self, **kwargs) -> None:
        """Añade información extra al logger."""
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._extra.update(filtered)
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
        """Formatea el mensaje con contexto y kwargs adicionales."""
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
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.WARNING, msg, *args, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self.error(msg, *args, exc_info=True, **kwargs)

    def child(self, name: str) -> StructuredLogger:
        """Crea un logger hijo con el mismo contexto.

        Args:
            name: Nombre del logger hijo.

        Returns:
            StructuredLogger: Logger hijo con el contexto heredado.
        """
        child = StructuredLogger(
            name=f"{self._name}.{name}",
            json_format=self._json_format,
            level=self._level,
        )
        child.set_context(**self._context)
        return child

    def with_context(self, **kwargs) -> StructuredLogger:
        """Crea un logger con contexto adicional.

        Args:
            **kwargs: Contexto a añadir.

        Returns:
            StructuredLogger: Logger con el contexto añadido.
        """
        logger = StructuredLogger(
            name=self._name,
            json_format=self._json_format,
            level=self._level,
        )
        logger.set_context(**{**self._context, **kwargs})
        return logger

    @property
    def name(self) -> str:
        return self._name

    @property
    def json_format(self) -> bool:
        return self._json_format

    @property
    def level(self) -> int:
        return self._level

    def __repr__(self) -> str:
        return f"StructuredLogger(name='{self._name}', json={self._json_format})"


T = TypeVar("T", bound=type)


class LoggerMixin:
    """Mixin para agregar logging estructurado a clases.

    Ejemplo:
        >>> class MyClass(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("Procesando...")
    """

    _logger: StructuredLogger | None = None
    _log_context: dict[str, Any] = {}

    @property
    def logger(self) -> StructuredLogger:
        """Obtiene un logger estructurado para la clase.

        Returns:
            StructuredLogger: Logger configurado para la clase.
        """
        if self._logger is None:
            self._logger = get_logger_for_class(self.__class__)
            if self._log_context:
                self._logger.set_context(**self._log_context)
        return self._logger

    def set_log_context(self, **kwargs) -> None:
        """Establece contexto adicional para los logs de la clase.

        Args:
            **kwargs: Pares clave-valor para el contexto.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._log_context.update(filtered)
        if self._logger:
            self._logger.set_context(**filtered)

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
        """
        error_type = type(error).__name__
        error_msg = str(error) if str(error) else "No details available"

        log_msg = f"{message}: " if message else ""
        log_msg += f"{error_type}: {error_msg}"

        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self.logger.error(
            log_msg,
            error_type=error_type,
            error_details=error_msg,
            **filtered,
        )


def get_logger_for_class(cls: type) -> StructuredLogger:
    """Obtiene un logger estructurado para una clase.

    Args:
        cls: Clase para la cual obtener el logger.

    Returns:
        StructuredLogger: Logger configurado.

    Example:
        >>> class MyDetector:
        ...     logger = get_logger_for_class(MyDetector)
        ...
        ...     def detect(self):
        ...         self.logger.info("Detectando...")
    """
    module = cls.__module__
    name = cls.__name__
    full_name = f"{module}.{name}" if module != "__main__" else name

    return StructuredLogger(
        name=full_name,
        json_format=config.json_format,
        level=config.level,
    )


def get_logger(
    name: str,
    log_file: str | None = None,
    *,
    json_format: bool = False,
) -> StructuredLogger:
    """Obtiene un logger estructurado por nombre.

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
    return StructuredLogger(
        name=name,
        log_file=log_file,
        json_format=json_format or config.json_format,
        level=config.level,
    )


def get_default_logger() -> StructuredLogger:
    """Obtiene el logger por defecto.

    Returns:
        StructuredLogger: Logger por defecto.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger("system")
    return _default_logger


def setup_logging(
    level: str | int = "INFO",
    log_file: str | None = None,
    *,
    json_format: bool = False,
    colored: bool = True,
    force: bool = False,
    module_levels: dict[str, str | int] | None = None,
    third_party_level: str | int = "WARNING",
) -> None:
    """Configura el sistema de logging global.

    Esta función debe llamarse una vez al inicio de la aplicación.

    Args:
        level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Ruta al archivo de log (opcional).
        json_format: Si usar formato JSON para los logs.
        colored: Si usar colores en la salida de consola.
        force: Forzar reconfiguración aunque ya esté inicializado.
        module_levels: Niveles específicos por módulo.
        third_party_level: Nivel para librerías de terceros.

    Example:
        >>> setup_logging(
        ...     level="DEBUG",
        ...     log_file="logs/app.log",
        ...     module_levels={"core.detector": "DEBUG", "utils": "WARNING"},
        ... )
    """
    global _logging_initialized

    with _logging_lock:
        if _logging_initialized and not force:
            return

        if isinstance(level, str):
            config.level = LOG_LEVEL_NAMES.get(level.upper(), logging.INFO)
        else:
            config.level = level

        config.json_format = json_format
        config.colored = colored

        if log_file:
            config.log_file = Path(log_file)

        if module_levels:
            for mod, lvl in module_levels.items():
                if isinstance(lvl, str):
                    config.module_levels[mod] = LOG_LEVEL_NAMES.get(lvl.upper(), logging.INFO)
                else:
                    config.module_levels[mod] = lvl

        if isinstance(third_party_level, str):
            config.third_party_level = LOG_LEVEL_NAMES.get(
                third_party_level.upper(), logging.WARNING
            )
        else:
            config.third_party_level = third_party_level

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        if force:
            root_logger.handlers.clear()

        if json_format:
            formatter = StructuredFormatter(json_format=True, datefmt=DEFAULT_DATE_FORMAT)
        else:
            formatter = StructuredFormatter(
                fmt=DEFAULT_LOG_FORMAT,
                datefmt=DEFAULT_DATE_FORMAT,
            )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)

        if colored:
            colored_formatter = ColoredFormatter(
                DEFAULT_JSON_FORMAT if json_format else DEFAULT_LOG_FORMAT,
                datefmt=DEFAULT_DATE_FORMAT,
            )
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)

        if log_file:
            _setup_file_handler(log_file, formatter)

        _configure_third_party_logging()

        for mod, lvl in config.module_levels.items():
            logging.getLogger(mod).setLevel(lvl)

        _logging_initialized = True

        logger = logging.getLogger(__name__)
        logger.info(
            f"📋 Logging configurado (nivel={logging.getLevelName(config.level)}, "
            f"json={json_format}, archivo={log_file or 'ninguno'})"
        )


def _setup_file_handler(log_file: str, formatter: logging.Formatter) -> None:
    """Configura el handler de archivo."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def _configure_third_party_logging() -> None:
    """Configura niveles de logging para librerías de terceros."""
    third_party_loggers = {
        "urllib3": config.third_party_level,
        "requests": config.third_party_level,
        "PIL": config.third_party_level,
        "matplotlib": config.third_party_level,
        "numba": config.third_party_level,
        "onnxruntime": config.third_party_level,
        "torch": config.third_party_level,
        "ultralytics": config.third_party_level,
        "cv2": config.third_party_level,
        "asyncio": config.third_party_level,
    }

    for name, level in third_party_loggers.items():
        logging.getLogger(name).setLevel(level)


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


def get_logging_status() -> dict[str, Any]:
    """Obtiene el estado actual del sistema de logging.

    Returns:
        dict: Estado del logging.
    """
    root_logger = logging.getLogger()
    return {
        "initialized": _logging_initialized,
        "level": logging.getLevelName(root_logger.level),
        "handlers": [type(h).__name__ for h in root_logger.handlers],
        "effective_level": logging.getLevelName(root_logger.getEffectiveLevel()),
        "json_format": config.json_format,
        "log_file": str(config.log_file) if config.log_file else None,
        "module_levels": config.module_levels,
    }


def set_module_level(module: str, level: str | int) -> None:
    """Establece el nivel de logging para un módulo específico.

    Args:
        module: Nombre del módulo.
        level: Nivel de logging.
    """
    if isinstance(level, str):
        level = LOG_LEVEL_NAMES.get(level.upper(), logging.INFO)
    config.module_levels[module] = level
    logging.getLogger(module).setLevel(level)


@contextmanager
def temporary_log_level(
    level: str | int, logger_name: str | None = None
) -> Generator[None, None, None]:
    """Context manager para cambiar temporalmente el nivel de logging.

    Args:
        level: Nivel temporal.
        logger_name: Nombre del logger (None para root).

    Example:
        >>> with temporary_log_level("DEBUG", "core.detector"):
        ...     # Detector logs en DEBUG temporalmente
        ...     detector.detect(frame)
    """
    if isinstance(level, str):
        level = LOG_LEVEL_NAMES.get(level.upper(), logging.INFO)

    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    original_level = logger.level
    logger.setLevel(level)

    try:
        yield
    finally:
        logger.setLevel(original_level)


@contextmanager
def log_context(**kwargs) -> Generator[StructuredLogger, None, None]:
    """Context manager para logging con contexto temporal.

    Args:
        **kwargs: Contexto temporal.

    Yields:
        StructuredLogger: Logger con el contexto temporal.

    Example:
        >>> with log_context(frame_id=42) as logger:
        ...     logger.info("Procesando frame")
    """
    logger = get_default_logger()
    original_context = logger.get_context()
    logger.set_context(**kwargs)

    try:
        yield logger
    finally:
        logger.clear_context()
        logger.set_context(**original_context)


__all__ = [
    "StructuredLogger",
    "LoggerMixin",
    "LoggingConfig",
    "config",
    "get_logger",
    "get_logger_for_class",
    "get_default_logger",
    "setup_logging",
    "set_module_level",
    "get_logging_status",
    "temporary_log_level",
    "log_context",
    "LOG_LEVEL_NAMES",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
    "setup_logger",
]
