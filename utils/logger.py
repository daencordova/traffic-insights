"""
Sistema de logging estructurado con niveles y formato consistente.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class StructuredLogger:
    """
    Logger estructurado con soporte para formato JSON y contexto.

    Características:
    - Contexto enriquecido para cada mensaje
    - Formato JSON opcional para integración con sistemas de monitoreo
    - Múltiples niveles de logging
    - Separación de logs por módulo
    """

    def __init__(
        self,
        name: str = "vehicle_counter",
        log_file: Optional[str] = None,
        json_format: bool = False
    ):
        self.name = name
        self.json_format = json_format
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:
            self._setup_handlers(log_file)

        self.context: Dict[str, Any] = {}
        self._extra_info: Dict[str, Any] = {}

    def _setup_handlers(self, log_file: Optional[str] = None):
        """Configura los handlers del logger."""
        self.logger.setLevel(logging.DEBUG)

        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def set_context(self, **kwargs):
        """Establece contexto para los logs."""
        self.context.update(kwargs)

    def clear_context(self):
        """Limpia el contexto."""
        self.context.clear()

    def add_extra(self, **kwargs):
        """Añade información extra al logger."""
        self._extra_info.update(kwargs)

    def _format_message(self, message: str, **kwargs) -> str:
        """Formatea el mensaje con contexto y kwargs adicionales."""
        parts = []

        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[{context_str}]")

        parts.append(message)

        if kwargs or self._extra_info:
            all_kwargs = {**self._extra_info, **kwargs}
            if all_kwargs:
                kwargs_str = " | ".join(f"{k}={v}" for k, v in all_kwargs.items())
                parts.append(f"({kwargs_str})")

        if self.json_format:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "context": self.context,
                "extra": kwargs
            }
            return json.dumps(log_data)

        return " ".join(parts)

    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message(message, **kwargs))

    def critical(self, message: str, **kwargs):
        self.logger.critical(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs):
        """Log de excepción con traceback."""
        self.logger.exception(self._format_message(message, **kwargs))


class LoggerMixin:
    """
    Mixin para agregar logging estructurado a clases.

    Proporciona:
    - Un logger configurado automáticamente para cada clase
    - Métodos de logging consistentes
    - Contexto enriquecido automático
    """

    @property
    def logger(self) -> StructuredLogger:
        """Obtiene un logger estructurado para la clase."""
        if not hasattr(self, "_structured_logger"):
            log_file = None

            if hasattr(self, "config") and hasattr(self.config, "output"):
                logs_dir = getattr(self.config.output, "logs_dir", "data/logs/")
                if logs_dir:
                    log_file = Path(logs_dir) / f"{self.__class__.__name__.lower()}.log"
                    log_file = str(log_file)

            self._structured_logger = StructuredLogger(
                name=self.__class__.__name__,
                log_file=log_file,
            )
            self._structured_logger.set_context(
                class_name=self.__class__.__name__,
                module=self.__class__.__module__
            )
        return self._structured_logger

    def set_log_context(self, **kwargs):
        """Establece contexto adicional para los logs."""
        self.logger.set_context(**kwargs)

    def clear_log_context(self):
        """Limpia el contexto del logger."""
        self.logger.clear_context()

    def log_error_with_context(self, error: Exception, message: str = None, **kwargs):
        """
        Registra un error con contexto completo.

        Args:
            error: La excepción capturada
            message: Mensaje adicional (opcional)
            **kwargs: Contexto adicional
        """
        error_type = type(error).__name__
        error_msg = str(error) if str(error) else "No details available"

        log_message = f"{message}: " if message else ""
        log_message += f"{error_type}: {error_msg}"

        self.logger.error(
            log_message,
            error_type=error_type,
            error_details=error_msg,
            **kwargs
        )


def setup_logger(
    name: str = "vehicle_counter",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configura y retorna un logger estándar (compatibilidad con código existente).

    Args:
        name: Nombre del logger
        log_file: Ruta del archivo de log (opcional)
        level: Nivel de logging
        json_format: Si usar formato JSON

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        if json_format:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "name": "%(name)s", '
                '"level": "%(levelname)s", "message": %(message)s}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
