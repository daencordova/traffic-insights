"""
Constantes centralizadas del sistema de seguimiento de tráfico.

Este módulo contiene todas las constantes globales utilizadas en el sistema,
organizadas por categorías para facilitar su mantenimiento y uso.

Las constantes están definidas como Final para prevenir modificaciones
accidentales y mejorar la claridad del código.

Categorías:
- Detección de objetos
- Tracking y seguimiento
- Rendimiento y FPS
- Procesamiento de frames
- Caché y memoria
- Colores y visualización
- Dashboard y UI
- Archivos y directorios
- Validaciones
- Logging
- Análisis y congestión
- Exportación
- Captura y buffer
- Pipeline y health checks
- Features avanzados (MHT, Re-ID, Sensor Fusion)
"""

from typing import Dict, Tuple, List, Final

# DETECCIÓN DE OBJETOS
MIN_BOX_SIZE: Final[int] = 10
"""Tamaño mínimo de un bounding box en píxeles."""
MAX_BOX_SIZE: Final[int] = 10000
"""Tamaño máximo de un bounding box en píxeles."""
MIN_DETECTION_AREA: Final[int] = 500
"""Área mínima de una detección en píxeles cuadrados."""
MAX_DETECTION_AREA: Final[int] = 100000
"""Área máxima de una detección en píxeles cuadrados."""
MIN_DETECTION_CONFIDENCE: Final[float] = 0.0
"""Confianza mínima permitida para una detección."""
MAX_DETECTION_CONFIDENCE: Final[float] = 1.0
"""Confianza máxima permitida para una detección."""

# TRACKING Y SEGUIMIENTO
MAX_ACTIVE_TRACKS: Final[int] = 50
"""Número máximo de tracks activos simultáneamente."""
MAX_LOST_TRACKS: Final[int] = 50
"""Número máximo de tracks perdidos almacenados."""
MAX_TRACK_HISTORY: Final[int] = 15
"""Longitud máxima del historial de posiciones por track."""
MIN_HITS_TO_CONFIRM: Final[int] = 3
"""Número mínimo de detecciones para confirmar un track."""
MAX_FRAMES_MISSED: Final[int] = 30
"""Número máximo de frames perdidos antes de eliminar un track."""

IOU_THRESHOLD: Final[float] = 0.3
"""Umbral de IoU para matching entre detecciones y tracks."""
FEATURE_THRESHOLD: Final[float] = 0.5
"""Umbral de similitud de features para re-identificación."""
MAX_MATCH_DISTANCE: Final[float] = 50.0
"""Distancia máxima para matching espacial."""
MIN_MOTION_DISTANCE: Final[float] = 5.0
"""Distancia mínima para considerar movimiento."""

# RENDIMIENTO Y FPS
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

# PROCESAMIENTO DE FRAMES
MAX_FRAME_SKIP: Final[int] = 2
"""Número máximo de frames a saltar en control de flujo."""
MIN_FRAME_SKIP: Final[int] = 1
"""Número mínimo de frames a saltar."""
PROCESS_EVERY_N_FRAMES: Final[int] = 1
"""Procesar cada N frames (1 = todos)."""

# CACHÉ Y MEMORIA
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

# COLORES Y VISUALIZACIÓN
COLORS: Final[Dict[str, Tuple[int, int, int]]] = {
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "CYAN": (255, 255, 0),
    "MAGENTA": (255, 0, 255),
    "ORANGE": (0, 165, 255),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "GRAY": (128, 128, 128),
    "DARK_GRAY": (64, 64, 64),
    "LIGHT_GRAY": (192, 192, 192),
}
"""Diccionario de colores predefinidos en formato BGR."""

DETECTION_COLORS: Final[List[Tuple[int, int, int]]] = [
    (0, 255, 0),      # Verde
    (255, 165, 0),    # Naranja
    (255, 0, 0),      # Rojo
    (255, 255, 0),    # Amarillo
    (0, 255, 255),    # Cian
    (255, 0, 255),    # Magenta
    (0, 128, 255),    # Azul claro
    (128, 0, 255),    # Púrpura
    (255, 128, 0),    # Naranja oscuro
    (0, 255, 128),    # Verde claro
]
"""Paleta de colores para detecciones y tracks."""

# DASHBOARD Y UI
DASHBOARD_WIDTH: Final[int] = 220
"""Ancho del dashboard en píxeles."""
DASHBOARD_HEIGHT: Final[int] = 120
"""Alto del dashboard en píxeles."""
DASHBOARD_ALPHA: Final[float] = 0.7
"""Opacidad del dashboard (0 = transparente, 1 = opaco)."""
FONT_SCALE: Final[float] = 0.5
"""Escala de fuente para textos en UI."""
LINE_THICKNESS: Final[int] = 2
"""Grosor de líneas en UI."""
POINT_RADIUS: Final[int] = 4
"""Radio de puntos en UI."""
TRAIL_POINTS: Final[int] = 15
"""Número de puntos en la trayectoria (trail)."""

# Dimensiones de ventana
MIN_WINDOW_WIDTH: Final[int] = 320
"""Ancho mínimo de la ventana de visualización."""
MIN_WINDOW_HEIGHT: Final[int] = 240
"""Alto mínimo de la ventana de visualización."""
MAX_WINDOW_WIDTH: Final[int] = 1920
"""Ancho máximo de la ventana de visualización."""
MAX_WINDOW_HEIGHT: Final[int] = 1080
"""Alto máximo de la ventana de visualización."""

# MEMORIA DEL SISTEMA
MEMORY_WARNING_THRESHOLD: Final[float] = 70.0
"""Umbral de memoria para advertencia (%)."""
MEMORY_CRITICAL_THRESHOLD: Final[float] = 80.0
"""Umbral de memoria para estado crítico (%)."""
MEMORY_LIMIT_MB: Final[int] = 2048
"""Límite de memoria del sistema en MB."""

# DISPOSITIVOS
PREFERRED_DEVICE_ORDER: Final[List[str]] = ["cuda", "mps", "cpu"]
"""Orden de preferencia para dispositivos de inferencia."""

# ARCHIVOS Y DIRECTORIOS
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

# VALIDACIONES
VALID_IMGSZ: Final[List[int]] = [320, 416, 512, 640, 768, 832, 1024]
"""Tamaños de imagen válidos para el modelo (múltiplos de 32)."""
VALID_FPS_RANGE: Final[Tuple[float, float]] = (1.0, 120.0)
"""Rango válido de FPS."""
VALID_CONFIDENCE_RANGE: Final[Tuple[float, float]] = (0.0, 1.0)
"""Rango válido de confianza."""
VALID_IOU_RANGE: Final[Tuple[float, float]] = (0.0, 1.0)
"""Rango válido de IoU."""

# LOGGING
LOG_LEVELS: Final[Dict[str, int]] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
"""Mapeo de niveles de logging a valores numéricos."""
LOG_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Formato de timestamp para logs."""

# ANÁLISIS Y CONGESTIÓN
CONGESTION_LOW: Final[float] = 0.3
"""Umbral bajo de congestión."""
CONGESTION_MEDIUM: Final[float] = 0.6
"""Umbral medio de congestión."""
CONGESTION_HIGH: Final[float] = 0.8
"""Umbral alto de congestión."""

ANALYSIS_WINDOW_SECONDS: Final[int] = 60
"""Ventana de análisis en segundos."""
PREDICTION_HORIZON_SECONDS: Final[int] = 300
"""Horizonte de predicción en segundos."""
PREDICTION_SAMPLES: Final[int] = 100
"""Número de muestras para predicción."""

# EXPORTACIÓN
SUPPORTED_EXPORT_FORMATS: Final[List[str]] = ["json", "csv", "both"]
"""Formatos de exportación soportados."""
SUPPORTED_IMAGE_FORMATS: Final[List[str]] = ["jpg", "png", "bmp", "tiff"]
"""Formatos de imagen soportados."""
AUTO_SAVE_INTERVAL_SECONDS: Final[int] = 300
"""Intervalo de auto-guardado en segundos."""

# CONSTANTES DE SISTEMA
DEFAULT_ENCODING: Final[str] = "utf-8"
"""Codificación por defecto."""
DEFAULT_CONFIG_FILENAME: Final[str] = "config.yaml"
"""Nombre del archivo de configuración por defecto."""
DEFAULT_LOG_FILENAME: Final[str] = "system.log"
"""Nombre del archivo de log por defecto."""

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

# CONSTANTES DE IMAGEN
DEFAULT_IMAGE_WIDTH: Final[int] = 640
"""Ancho de imagen por defecto."""
DEFAULT_IMAGE_HEIGHT: Final[int] = 480
"""Alto de imagen por defecto."""
DEFAULT_LANE_WIDTH: Final[int] = 40
"""Ancho de carril por defecto."""
DEFAULT_BUFFER_ZONE: Final[int] = 15
"""Zona de buffer por defecto."""

# CONSTANTES DE FUENTE Y RENDERIZADO
DEFAULT_FONT: Final[int] = 0
"""Fuente por defecto (OpenCV)."""
DEFAULT_FONT_SCALE: Final[float] = 0.5
"""Escala de fuente por defecto."""
DEFAULT_FONT_THICKNESS: Final[int] = 2
"""Grosor de fuente por defecto."""
DEFAULT_LINE_THICKNESS: Final[int] = 2
"""Grosor de línea por defecto."""

# CONSTANTES DE VENTANA
WINDOW_NAME: Final[str] = "Vehicle Counting System"
"""Nombre de la ventana de visualización."""
DEFAULT_WINDOW_WIDTH: Final[int] = 1280
"""Ancho de ventana por defecto."""
DEFAULT_WINDOW_HEIGHT: Final[int] = 720
"""Alto de ventana por defecto."""

# CONSTANTES DE CAPTURA
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
CAPTURE_RECONNECT_DELAY: Final[float] = 1.0
"""Delay de reconexión en segundos."""
CAPTURE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5
"""Máximo de errores consecutivos antes de reconectar."""

# CONSTANTES DE BUFFER
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
BUFFER_DROP_THRESHOLD: Final[float] = 0.8
"""Umbral de ocupación para comenzar a descartar frames."""
BUFFER_RECOVERY_THRESHOLD: Final[float] = 0.3
"""Umbral de ocupación para recuperar frames."""
BUFFER_SKIP_MAX: Final[int] = 2
"""Máximo de frames a saltar."""
BUFFER_SKIP_CONSECUTIVE_LIMIT: Final[int] = 5
"""Límite de saltos consecutivos."""

# CONSTANTES DE FRAME
MIN_FRAME_DIMENSION: Final[int] = 10
"""Dimensión mínima de un frame."""
MIN_FRAME_WIDTH: Final[int] = 10
"""Ancho mínimo de un frame."""
MIN_FRAME_HEIGHT: Final[int] = 10
"""Alto mínimo de un frame."""
DEFAULT_FRAME_WIDTH: Final[int] = 640
"""Ancho de frame por defecto."""
DEFAULT_FRAME_HEIGHT: Final[int] = 480
"""Alto de frame por defecto."""
DEFAULT_FRAME_CHANNELS: Final[int] = 3
"""Canales de frame por defecto."""

# CONSTANTES DE RENDERIZADO
DEFAULT_RENDER_WIDTH: Final[int] = 640
"""Ancho de renderizado por defecto."""
DEFAULT_RENDER_HEIGHT: Final[int] = 480
"""Alto de renderizado por defecto."""
DEFAULT_RENDER_CHANNELS: Final[int] = 3
"""Canales de renderizado por defecto."""
RENDER_ERROR_COOLDOWN: Final[float] = 1.0
"""Cooldown en segundos entre errores de renderizado."""
MAX_RENDER_TIMES: Final[int] = 100
"""Máximo de tiempos de renderizado almacenados."""

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

# CONSTANTES DE TRACK (VISUALIZACIÓN)
TRACK_ARROW_LENGTH_MIN: Final[int] = 10
"""Longitud mínima de flecha de track."""
TRACK_ARROW_LENGTH_MAX: Final[int] = 30
"""Longitud máxima de flecha de track."""

TRACK_CIRCLE_RADIUS: Final[int] = 6
"""Radio del círculo de track."""
TRACK_CONFIDENCE_RADIUS_MIN: Final[int] = 2
"""Radio mínimo de confianza de track."""
TRACK_CONFIDENCE_RADIUS_MAX: Final[int] = 6
"""Radio máximo de confianza de track."""

TRACK_TRAIL_THICKNESS_MIN: Final[int] = 1
"""Grosor mínimo de trail de track."""
TRACK_TRAIL_THICKNESS_MAX: Final[int] = 2
"""Grosor máximo de trail de track."""

TRACK_BBOX_THICKNESS_MIN: Final[int] = 1
"""Grosor mínimo de bbox de track."""
TRACK_BBOX_THICKNESS_MAX: Final[int] = 2
"""Grosor máximo de bbox de track."""

PREDICTION_POINT_RADIUS_MIN: Final[int] = 2
"""Radio mínimo de punto de predicción."""
PREDICTION_POINT_RADIUS_MAX: Final[int] = 5
"""Radio máximo de punto de predicción."""

# CONSTANTES DE ESTADOS
STATUS_COLORS: Final[Dict[str, Tuple[Tuple[int, int, int], str, str]]] = {
    "confirmed": ((0, 255, 0), "✅", "OK"),
    "lost": ((0, 255, 255), "⚠️", "Lost"),
    "tentative": ((255, 255, 0), "⏳", "New"),
    "dead": ((128, 128, 128), "💀", "Dead"),
}
"""Colores, iconos y textos para estados de tracks."""

PREDICTION_STATE_COLORS: Final[Dict[str, Tuple[int, int, int]]] = {
    "stopped": (0, 0, 255),
    "accelerating": (0, 255, 255),
    "decelerating": (0, 165, 255),
    "turning": (255, 0, 255),
    "erratic": (255, 0, 0),
    "moving": (255, 255, 0),
    "unknown": (255, 255, 0),
}
"""Colores para estados de predicción."""

# CONSTANTES DE APRENDIZAJE EN LÍNEA
ONLINE_LEARNING_DEFAULT_LR: Final[float] = 0.05
"""Tasa de aprendizaje por defecto."""
ONLINE_LEARNING_MIN_SAMPLES: Final[int] = 5
"""Mínimo de muestras para aprendizaje."""
ONLINE_LEARNING_DRIFT_THRESHOLD: Final[float] = 0.35
"""Umbral de drift de concepto."""
ONLINE_LEARNING_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de aprendizaje."""

# CONSTANTES DE FUSIÓN DE SENSORES
SENSOR_FUSION_VISUAL_WEIGHT: Final[float] = 0.7
"""Peso del sensor visual."""
SENSOR_FUSION_DEPTH_WEIGHT: Final[float] = 0.5
"""Peso del sensor de profundidad."""
SENSOR_FUSION_THERMAL_WEIGHT: Final[float] = 0.4
"""Peso del sensor térmico."""
SENSOR_FUSION_MOTION_WEIGHT: Final[float] = 0.3
"""Peso del sensor de movimiento."""
SENSOR_FUSION_MIN_OBSERVATIONS: Final[int] = 2
"""Mínimo de observaciones para fusión."""
SENSOR_FUSION_MAX_HISTORY: Final[int] = 50
"""Máximo histórico de fusión."""
SENSOR_FUSION_PARTICLE_COUNT: Final[int] = 500
"""Número de partículas para filtro de partículas."""

# CONSTANTES DE PREDICCIÓN DE TRAYECTORIA
PATH_PREDICTION_HISTORY_LENGTH: Final[int] = 30
"""Longitud de histórico para predicción."""
PATH_PREDICTION_HORIZON: Final[float] = 2.0
"""Horizonte de predicción en segundos."""
PATH_PREDICTION_STEPS: Final[int] = 20
"""Número de pasos de predicción."""
PATH_PREDICTION_MIN_SAMPLES: Final[int] = 5
"""Mínimo de muestras para predicción."""
PATH_PREDICTION_UNCERTAINTY_THRESHOLD: Final[float] = 0.7
"""Umbral de incertidumbre para predicción."""

# CONSTANTES DE MHT (Multi-Hypothesis Tracking)
MHT_MAX_DEPTH: Final[int] = 5
"""Profundidad máxima del árbol MHT."""
MHT_PRUNING_THRESHOLD: Final[float] = 0.01
"""Umbral de poda de hipótesis MHT."""
MHT_MAX_HYPOTHESES: Final[int] = 3
"""Máximo de hipótesis por track."""

# CONSTANTES DE RE-IDENTIFICACIÓN
REID_SIMILARITY_THRESHOLD: Final[float] = 0.6
"""Umbral de similitud para re-identificación."""
REID_SPATIAL_THRESHOLD: Final[float] = 100.0
"""Umbral espacial para re-identificación."""
REID_MAX_AGE_SECONDS: Final[float] = 30.0
"""Edad máxima para re-identificación."""
REID_CACHE_SIZE: Final[int] = 1000
"""Tamaño de caché de re-identificación."""
REID_MIN_FEATURES: Final[int] = 3
"""Mínimo de features para re-identificación."""

# CONSTANTES DE VALIDACIÓN DE TRACKS
TRACK_VALIDATION_MIN_CONFIDENCE: Final[float] = 0.3
"""Confianza mínima para validación de tracks."""
TRACK_VALIDATION_MAX_SPEED_CHANGE: Final[float] = 50.0
"""Cambio máximo de velocidad para validación."""
TRACK_VALIDATION_IOU_THRESHOLD: Final[float] = 0.3
"""Umbral de IoU para validación."""
TRACK_VALIDATION_FEATURE_THRESHOLD: Final[float] = 0.6
"""Umbral de features para validación."""
TRACK_VALIDATION_MOTION_THRESHOLD: Final[float] = 0.7
"""Umbral de movimiento para validación."""
TRACK_VALIDATION_SHAPE_THRESHOLD: Final[float] = 0.5
"""Umbral de forma para validación."""
