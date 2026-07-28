"""Constantes del sistema: configuración, memoria, logging, timeouts."""

from typing import Final

# CODIFICACIÓN Y ARCHIVOS
DEFAULT_ENCODING: Final[str] = "utf-8"
"""Codificación por defecto para archivos."""

DEFAULT_CONFIG_FILENAME: Final[str] = "config.yaml"
"""Nombre del archivo de configuración por defecto."""

DEFAULT_LOG_FILENAME: Final[str] = "system.log"
"""Nombre del archivo de log por defecto."""

DEFAULT_CONFIG_PATH: Final[str] = "config.yaml"
"""Ruta por defecto del archivo de configuración."""

DEFAULT_DATA_DIR: Final[str] = "data/"
"""Directorio por defecto para datos."""

SCREENSHOTS_DIR: Final[str] = "data/screenshots/"
"""Directorio para capturas de pantalla."""

EXPORTS_DIR: Final[str] = "data/exports/"
"""Directorio para exportaciones de datos."""

LOGS_DIR: Final[str] = "data/logs/"
"""Directorio para archivos de log."""

# MEMORIA DEL SISTEMA
MEMORY_WARNING_THRESHOLD: Final[float] = 70.0
"""Umbral de memoria para advertencia (%)."""

MEMORY_CRITICAL_THRESHOLD: Final[float] = 80.0
"""Umbral de memoria para estado crítico (%)."""

MEMORY_LIMIT_MB: Final[int] = 2048
"""Límite de memoria del sistema en MB."""

MEMORY_MINIMUM_AVAILABLE_MB: Final[int] = 500
"""Memoria mínima disponible en MB para operar."""

# DISPOSITIVOS
PREFERRED_DEVICE_ORDER: Final[list[str]] = ["cuda", "mps", "cpu"]
"""Orden de preferencia para dispositivos de inferencia."""

# LOGGING
LOG_LEVELS: Final[dict[str, int]] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
"""Mapeo de niveles de logging a valores numéricos."""

LOG_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Formato de timestamp para logs."""

# VALIDACIONES
VALID_IMGSZ: Final[list[int]] = [320, 416, 512, 640, 768, 832, 1024]
"""Tamaños de imagen válidos para el modelo (múltiplos de 32)."""

VALID_FPS_RANGE: Final[tuple[float, float]] = (1.0, 120.0)
"""Rango válido de FPS."""

VALID_CONFIDENCE_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
"""Rango válido de confianza."""

VALID_IOU_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
"""Rango válido de IoU."""

# TIEMPOS
SECONDS_PER_MINUTE: Final[int] = 60
"""Segundos por minuto."""

SECONDS_PER_HOUR: Final[int] = 3600
"""Segundos por hora."""

MILLISECONDS_PER_SECOND: Final[int] = 1000
"""Milisegundos por segundo."""

BYTES_PER_MB: Final[int] = 1024 * 1024
"""Bytes por megabyte."""

BYTES_PER_KB: Final[int] = 1024
"""Bytes por kilobyte."""

# SLEEP Y TIMEOUTS
DEFAULT_SLEEP_SHORT: Final[float] = 0.001
"""Sleep corto para loops de alta frecuencia (1ms)."""

DEFAULT_SLEEP_MEDIUM: Final[float] = 0.01
"""Sleep medio para loops de frecuencia media (10ms)."""

DEFAULT_SLEEP_LONG: Final[float] = 0.1
"""Sleep largo para operaciones de baja frecuencia (100ms)."""

DEFAULT_TIMEOUT_SHORT: Final[float] = 0.05
"""Timeout corto (50ms)."""

DEFAULT_TIMEOUT_MEDIUM: Final[float] = 0.5
"""Timeout medio (500ms)."""

DEFAULT_TIMEOUT_LONG: Final[float] = 5.0
"""Timeout largo (5s)."""

DEFAULT_TIMEOUT_VERY_LONG: Final[float] = 30.0
"""Timeout muy largo (30s)."""

# ERRORES Y RECUPERACIÓN
MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Máximo de errores consecutivos antes de acción."""

ERROR_RECOVERY_COOLDOWN: Final[float] = 1.0
"""Cooldown para recuperación de errores."""

ERROR_WINDOW_SECONDS: Final[float] = 60.0
"""Ventana de tiempo para conteo de errores."""

MAX_ERRORS_IN_WINDOW: Final[int] = 10
"""Máximo de errores en ventana de tiempo."""

COOLDOWN_RECOVERY: Final[float] = 3.0
"""Cooldown para recuperación de tracks."""

COOLDOWN_REIDENTIFICATION: Final[float] = 2.0
"""Cooldown para re-identificación."""

COOLDOWN_RENDER_ERROR: Final[float] = 1.0
"""Cooldown para errores de renderizado."""
