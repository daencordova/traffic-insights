"""Constantes de pipeline: rendimiento, buffer, captura, health checks."""

from typing import Final

# RENDIMIENTO
TARGET_FPS: Final[int] = 30
"""FPS objetivo del sistema."""

MIN_ACCEPTABLE_FPS: Final[int] = 15
"""FPS mínimo aceptable para rendimiento aceptable."""

CRITICAL_FPS: Final[int] = 5
"""FPS crítico por debajo del cual el sistema es inestable."""

MEMORY_CHECK_INTERVAL: Final[int] = 30
"""Intervalo en segundos para verificar memoria."""

GC_INTERVAL: Final[int] = 60
"""Intervalo en segundos para ejecutar garbage collection."""

CLEANUP_INTERVAL: Final[int] = 50
"""Intervalo en frames para limpiar tracks muertos."""

PERFORMANCE_MONITOR_INTERVAL: Final[int] = 60
"""Intervalo en segundos para monitoreo de rendimiento."""

# PROCESAMIENTO DE FRAMES
MAX_FRAME_SKIP: Final[int] = 2
"""Número máximo de frames a saltar en control de flujo."""

MIN_FRAME_SKIP: Final[int] = 1
"""Número mínimo de frames a saltar."""

PROCESS_EVERY_N_FRAMES: Final[int] = 1
"""Procesar cada N frames (1 = todos)."""

# BUFFER
BUFFER_SIZE_CPU: Final[int] = 20
"""Tamaño de buffer en modo CPU."""

BUFFER_SIZE_GPU: Final[int] = 30
"""Tamaño de buffer en modo GPU."""

MAX_WORKERS_CPU: Final[int] = 4
"""Máximo de workers en modo CPU."""

MAX_WORKERS_GPU: Final[int] = 8
"""Máximo de workers en modo GPU."""

MIN_WORKERS_CPU: Final[int] = 2
"""Mínimo de workers en modo CPU."""

MAX_BUFFER_SIZE_CPU: Final[int] = 20
"""Tamaño máximo del buffer en modo CPU."""

MAX_BUFFER_SIZE_GPU: Final[int] = 30
"""Tamaño máximo del buffer en modo GPU."""

BUFFER_DROP_THRESHOLD: Final[float] = 0.8
"""Umbral de ocupación para comenzar a descartar frames."""

BUFFER_RECOVERY_THRESHOLD: Final[float] = 0.3
"""Umbral de ocupación para recuperar frames."""

BUFFER_SKIP_MAX: Final[int] = 2
"""Máximo de frames a saltar."""

BUFFER_SKIP_CONSECUTIVE_LIMIT: Final[int] = 5
"""Límite de saltos consecutivos."""

# CAPTURA
CAPTURE_MIN_FPS_CPU: Final[float] = 5.0
"""FPS mínimo de captura en modo CPU."""

CAPTURE_MAX_FPS_CPU: Final[float] = 15.0
"""FPS máximo de captura en modo CPU."""

CAPTURE_TARGET_FPS_CPU: Final[float] = 8.0
"""FPS objetivo de captura en modo CPU."""

CAPTURE_TARGET_FPS_GPU: Final[float] = 30.0
"""FPS objetivo de captura en modo GPU."""

CAPTURE_DEFAULT_INTERVAL_CPU: Final[float] = 1.0 / 8.0
"""Intervalo de captura por defecto en CPU."""

CAPTURE_DEFAULT_INTERVAL_GPU: Final[float] = 1.0 / 30.0
"""Intervalo de captura por defecto en GPU."""

CAPTURE_BUFFER_MIN_SIZE: Final[int] = 1
"""Tamaño mínimo del buffer de captura."""

CAPTURE_BUFFER_MAX_SIZE: Final[int] = 10
"""Tamaño máximo del buffer de captura."""

CAPTURE_BUFFER_DEFAULT_SIZE: Final[int] = 1
"""Tamaño por defecto del buffer de captura."""

CAPTURE_CV2_BUFFER_SIZE: Final[int] = 1
"""Tamaño de buffer de OpenCV (CV_CAP_PROP_BUFFERSIZE)."""

CAPTURE_FOURCC_MJPG: Final[int] = 0x47504A4D  # 'MJPG' en hexadecimal
"""Código FOURCC para formato MJPG."""

CAPTURE_RECONNECT_DELAY: Final[float] = 1.0
"""Delay de reconexión en segundos."""

CAPTURE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Máximo de errores consecutivos antes de reconectar."""

CAPTURE_RECONNECT_ATTEMPTS: Final[int] = 5
"""Número de intentos de reconexión."""

# PIPELINE
PIPELINE_MAX_RECONNECT_ATTEMPTS: Final[int] = 3
"""Máximo de intentos de reconexión del pipeline."""

PIPELINE_RECONNECT_DELAY: Final[float] = 0.1
"""Delay de reconexión del pipeline."""

PIPELINE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Máximo de errores consecutivos del pipeline."""

PIPELINE_DEFAULT_FRAME_TIMEOUT: Final[int] = 100
"""Timeout por defecto para frames."""

PIPELINE_MAX_RENDER_QUEUE_RATIO: Final[float] = 0.33
"""Ratio máximo de cola de renderizado."""

PIPELINE_STATS_INTERVAL_DEFAULT: Final[float] = 5.0
"""Intervalo por defecto para estadísticas del pipeline."""

# HEALTH CHECKS
HEALTH_CHECK_INTERVAL: Final[float] = 10.0
"""Intervalo de health checks en segundos."""

HEALTH_BUFFER_CRITICAL: Final[float] = 0.85
"""Umbral crítico de ocupación de buffer."""

HEALTH_BUFFER_WARNING: Final[float] = 0.7
"""Umbral de advertencia de ocupación de buffer."""

HEALTH_QUEUE_CRITICAL: Final[int] = 30
"""Tamaño crítico de cola."""

HEALTH_QUEUE_WARNING: Final[int] = 15
"""Tamaño de advertencia de cola."""

HEALTH_FPS_CRITICAL: Final[float] = 3.0
"""FPS crítico para health check."""

HEALTH_FPS_WARNING: Final[float] = 8.0
"""FPS de advertencia para health check."""

HEALTH_DROP_RATE_CRITICAL: Final[float] = 0.3
"""Tasa de drop crítica."""

HEALTH_DROP_RATE_WARNING: Final[float] = 0.1
"""Tasa de drop de advertencia."""

# BATCH PROCESSING
DEFAULT_BATCH_SIZE: Final[int] = 4
"""Tamaño de lote por defecto."""

MAX_BATCH_SIZE: Final[int] = 8
"""Tamaño máximo de lote."""

MIN_BATCH_SIZE: Final[int] = 2
"""Tamaño mínimo de lote."""

BATCH_TIMEOUT: Final[float] = 0.01
"""Timeout para procesamiento por lotes."""

# THREAD POOL
THREAD_POOL_MIN_WORKERS: Final[int] = 2
"""Mínimo de workers en thread pool."""

THREAD_POOL_MAX_WORKERS: Final[int] = 8
"""Máximo de workers en thread pool."""

THREAD_POOL_IDLE_TIMEOUT: Final[float] = 30.0
"""Timeout para workers inactivos."""

THREAD_POOL_MAX_QUEUE_SIZE: Final[int] = 100
"""Tamaño máximo de cola de thread pool."""

THREAD_POOL_MAX_HISTORY: Final[int] = 1000
"""Máximo histórico de tareas en thread pool."""

# MONITOREO
MONITOR_SAMPLE_INTERVAL: Final[float] = 5.0
"""Intervalo de muestreo para monitoreo."""

MONITOR_ALERT_THRESHOLD_MB: Final[float] = 100.0
"""Umbral de alerta de memoria en MB."""

MONITOR_MAX_SAMPLES: Final[int] = 60
"""Máximo de muestras de monitoreo."""

MONITOR_MEMORY_CRITICAL_MB: Final[int] = 2000
"""Memoria crítica en MB para forzar GC."""

# CACHÉ
DEFAULT_CACHE_SIZE: Final[int] = 16
"""Tamaño por defecto del caché de detecciones."""

MAX_CACHE_SIZE: Final[int] = 64
"""Tamaño máximo del caché de detecciones."""

MIN_CACHE_SIZE: Final[int] = 4
"""Tamaño mínimo del caché de detecciones."""

MAX_CACHE_MEMORY_MB: Final[int] = 250
"""Memoria máxima del caché en MB."""

CACHE_CLEANUP_THRESHOLD: Final[float] = 0.6
"""Umbral de ocupación para limpiar caché."""

CACHE_ENTRY_SIZE_ESTIMATE: Final[int] = 16
"""Tamaño estimado de una entrada de caché en bytes."""

# RENDERIZADO
RENDER_ERROR_COOLDOWN: Final[float] = 1.0
"""Cooldown en segundos entre errores de renderizado."""

MAX_RENDER_TIMES: Final[int] = 100
"""Máximo de tiempos de renderizado almacenados."""

# RENDIMIENTO
MAX_INFERENCE_TIMES: Final[int] = 100
"""Número máximo de tiempos de inferencia almacenados."""
MAX_BATCH_TIMES: Final[int] = 50
"""Número máximo de tiempos de batch almacenados."""
MAX_PROCESSING_TIMES: Final[int] = 100
"""Número máximo de tiempos de procesamiento almacenados."""

# PROCESAMIENTO POR LOTES
BATCH_DEFAULT_SIZE: Final[int] = 4
"""Tamaño de lote por defecto."""
BATCH_MAX_SIZE: Final[int] = 8
"""Tamaño máximo de lote."""
BATCH_MIN_SIZE: Final[int] = 2
"""Tamaño mínimo de lote."""
