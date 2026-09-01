"""Unified logging system for the traffic tracking system.

This module provides a single, consistent interface for logging across
the entire project, eliminating duplication and ensuring uniform format.

Features:
    - Unique logger per class with automatic context
    - Structured format with console colors
    - JSON format support for production
    - Enriched context for each message
    - Configurable logging levels per module
    - Log file rotation
    - Multiple outputs (console, file, syslog)

Example:
    >>> from utils.logger import get_logger
    >>>
    >>> class MyClass:
    ...     logger = get_logger_for_class(MyClass)
    ...
    ...     def process(self):
    ...         self.logger.info("Processing...", extra={"frame_id": 42})
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
    """Global logging system configuration.

    This singleton class holds all logging configuration settings.

    Example:
        >>> config = LoggingConfig()
        >>> config.level = logging.DEBUG
        >>> config.json_format = True
        >>> config.update(log_file="logs/app.log")
    """

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
        """Updates the configuration.

        Args:
            **kwargs: Configuration key-value pairs to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if _logging_initialized:
            setup_logging(force=True)


config = LoggingConfig()


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to console messages."""

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
    """Structured formatter for production use."""

    def __init__(self, json_format: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._json_format = json_format

    def format(self, record: logging.LogRecord) -> str:
        if self._json_format:
            return self._format_json(record)
        return super().format(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Formats the record as JSON."""
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
    """Structured logger with context support.

    This logger extends the standard logging with structured context
    and additional formatting options.

    Attributes:
        name: Logger name.
        json_format: Whether to use JSON format.
        context: Current context for all messages.

    Example:
        >>> logger = StructuredLogger("my_module")
        >>> logger.set_context(frame_id=42)
        >>> logger.info("Processing frame")
        >>> # Output: [frame_id=42] Processing frame
        >>>
        >>> logger.child("detector").info("Detecting objects")
        >>> # Output: [frame_id=42] Detecting objects
    """

    def __init__(
        self,
        name: str,
        log_file: str | None = None,
        *,
        json_format: bool = False,
        level: int = logging.DEBUG,
    ) -> None:
        """Initializes the structured logger.

        Args:
            name: Identifying name for the logger.
            log_file: Path to log file (optional).
            json_format: Whether to use JSON format.
            level: Logging level for this logger.
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
        """Sets context for logs.

        Args:
            **kwargs: Key-value pairs for context.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._context.update(filtered)
        self._update_extra()

    def add_context(self, **kwargs) -> None:
        """Adds context without removing existing context.

        Args:
            **kwargs: Key-value pairs to add to context.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._context.update(filtered)
        self._update_extra()

    def clear_context(self) -> None:
        """Clears the current context."""
        self._context.clear()
        self._update_extra()

    def get_context(self) -> dict[str, Any]:
        """Gets a copy of the current context.

        Returns:
            dict[str, Any]: Copy of the current context.
        """
        return self._context.copy()

    def set_extra(self, **kwargs) -> None:
        """Adds extra information to the logger.

        Args:
            **kwargs: Key-value pairs for extra data.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._extra.update(filtered)
        self._update_extra()

    def clear_extra(self) -> None:
        """Clears the extra information."""
        self._extra.clear()
        self._update_extra()

    def _update_extra(self) -> None:
        """Updates the adapter extra with context and extra data."""
        combined = {**self._context, **self._extra}
        self.extra = combined

    def _format_message(self, message: str, **kwargs) -> tuple[str, dict]:
        """Formats the message with context and additional kwargs."""
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
        """Logs a message with the specified level."""
        reserved_keys = {"exc_info", "stack_info", "stacklevel", "extra"}
        context_kwargs = {k: v for k, v in kwargs.items() if k not in reserved_keys}

        formatted_msg, context_extra = self._format_message(msg, **context_kwargs)

        combined_extra = {**(extra or {}), **context_extra}
        combined_extra = {k: v for k, v in combined_extra.items() if k not in RESERVED_LOG_ATTRS}

        if not isinstance(level, int):
            level = LOG_LEVEL_NAMES.get(str(level).upper(), logging.INFO)

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
        kwargs.pop("level", None)
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        kwargs.pop("level", None)
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        kwargs.pop("level", None)
        self.log(logging.WARNING, msg, *args, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:
        kwargs.pop("level", None)
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        kwargs.pop("level", None)
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        kwargs.pop("level", None)
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        kwargs.pop("level", None)
        self.error(msg, *args, exc_info=True, **kwargs)

    def child(self, name: str) -> StructuredLogger:
        """Creates a child logger with the same context.

        Args:
            name: Name of the child logger.

        Returns:
            StructuredLogger: Child logger with inherited context.
        """
        child = StructuredLogger(
            name=f"{self._name}.{name}",
            json_format=self._json_format,
            level=self._level,
        )
        child.set_context(**self._context)
        return child

    def with_context(self, **kwargs) -> StructuredLogger:
        """Creates a logger with additional context.

        Args:
            **kwargs: Context to add.

        Returns:
            StructuredLogger: Logger with the added context.
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
        """Logger name."""
        return self._name

    @property
    def json_format(self) -> bool:
        """Whether JSON format is enabled."""
        return self._json_format

    @property
    def level(self) -> int:
        """Current logging level."""
        return self._level

    def __repr__(self) -> str:
        return f"StructuredLogger(name='{self._name}', json={self._json_format})"


T = TypeVar("T", bound=type)


class LoggerMixin:
    """Mixin to add structured logging to classes.

    Example:
        >>> class MyClass(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("Processing...")
    """

    _logger: StructuredLogger | None = None
    _log_context: dict[str, Any] = {}

    @property
    def logger(self) -> StructuredLogger:
        """Gets a structured logger for the class.

        Returns:
            StructuredLogger: Logger configured for the class.
        """
        if self._logger is None:
            self._logger = get_logger_for_class(self.__class__)
            if self._log_context:
                self._logger.set_context(**self._log_context)
        return self._logger

    def set_log_context(self, **kwargs) -> None:
        """Sets additional context for the class logs.

        Args:
            **kwargs: Key-value pairs for context.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in RESERVED_LOG_ATTRS}
        self._log_context.update(filtered)
        if self._logger:
            self._logger.set_context(**filtered)

    def clear_log_context(self) -> None:
        """Clears the class logger context."""
        self._log_context.clear()
        if self._logger:
            self._logger.clear_context()

    def log_error_with_context(
        self,
        error: Exception,
        message: str | None = None,
        **kwargs,
    ) -> None:
        """Logs an error with full context.

        Args:
            error: The captured exception.
            message: Additional message (optional).
            **kwargs: Additional context.
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
    """Gets a structured logger for a class.

    Args:
        cls: Class to get the logger for.

    Returns:
        StructuredLogger: Configured logger.

    Example:
        >>> class MyDetector:
        ...     logger = get_logger_for_class(MyDetector)
        ...
        ...     def detect(self):
        ...         self.logger.info("Detecting...")
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
    """Gets a structured logger by name.

    Args:
        name: Logger name.
        log_file: Path to log file (optional).
        json_format: Whether to use JSON format.

    Returns:
        StructuredLogger: Structured logger.

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Message with context")
    """
    return StructuredLogger(
        name=name,
        log_file=log_file,
        json_format=json_format or config.json_format,
        level=config.level,
    )


def get_default_logger() -> StructuredLogger:
    """Gets the default logger.

    Returns:
        StructuredLogger: Default logger.
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
    """Configures the global logging system.

    This function should be called once at application startup.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Path to log file (optional).
        json_format: Whether to use JSON format for logs.
        colored: Whether to use colors in console output.
        force: Force reconfiguration even if already initialized.
        module_levels: Specific levels per module.
        third_party_level: Level for third-party libraries.

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
            f"Logging configured (level={logging.getLevelName(config.level)}, "
            f"json={json_format}, file={log_file or 'none'})"
        )


def _setup_file_handler(log_file: str, formatter: logging.Formatter) -> None:
    """Configures the file handler."""
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
    """Configures logging levels for third-party libraries."""
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
    """Compatibility function for existing code.

    Args:
        name: Logger name.
        log_file: Path to log file (optional).
        level: Logging level.
        json_format: Whether to use JSON format.

    Returns:
        logging.Logger: Configured logger.
    """
    if not _logging_initialized:
        setup_logging(level=level, log_file=log_file, json_format=json_format)

    return logging.getLogger(name)


def get_logging_status() -> dict[str, Any]:
    """Gets the current status of the logging system.

    Returns:
        dict: Logging status.

    Example:
        >>> status = get_logging_status()
        >>> print(f"Initialized: {status['initialized']}")
        >>> print(f"Level: {status['level']}")
        >>> print(f"Handlers: {status['handlers']}")
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
    """Sets the logging level for a specific module.

    Args:
        module: Module name.
        level: Logging level.

    Example:
        >>> set_module_level("core.detector", "DEBUG")
        >>> # Detector logs will now show DEBUG messages
    """
    if isinstance(level, str):
        level = LOG_LEVEL_NAMES.get(level.upper(), logging.INFO)
    config.module_levels[module] = level
    logging.getLogger(module).setLevel(level)


@contextmanager
def temporary_log_level(
    level: str | int, logger_name: str | None = None
) -> Generator[None, None, None]:
    """Context manager to temporarily change logging level.

    Args:
        level: Temporary level.
        logger_name: Logger name (None for root).

    Example:
        >>> with temporary_log_level("DEBUG", "core.detector"):
        ...     # Detector logs temporarily in DEBUG
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
    """Context manager for logging with temporary context.

    Args:
        **kwargs: Temporary context.

    Yields:
        StructuredLogger: Logger with the temporary context.

    Example:
        >>> with log_context(frame_id=42) as logger:
        ...     logger.info("Processing frame")
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
