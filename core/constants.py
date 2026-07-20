"""
Constantes centralizadas del sistema
"""

from typing import Dict, Tuple, List, Final

# DETECCIÓN
MIN_BOX_SIZE: Final[int] = 10
MAX_BOX_SIZE: Final[int] = 10000
MIN_DETECTION_AREA: Final[int] = 500
MAX_DETECTION_AREA: Final[int] = 100000
MIN_DETECTION_CONFIDENCE: Final[float] = 0.0
MAX_DETECTION_CONFIDENCE: Final[float] = 1.0

# TRACKING
MAX_ACTIVE_TRACKS: Final[int] = 50
MAX_LOST_TRACKS: Final[int] = 50
MAX_TRACK_HISTORY: Final[int] = 15
MIN_HITS_TO_CONFIRM: Final[int] = 3
MAX_FRAMES_MISSED: Final[int] = 30

IOU_THRESHOLD: Final[float] = 0.3
FEATURE_THRESHOLD: Final[float] = 0.5
MAX_MATCH_DISTANCE: Final[float] = 50.0
MIN_MOTION_DISTANCE: Final[float] = 5.0

# RENDIMIENTO Y FPS
TARGET_FPS: Final[int] = 30
MIN_ACCEPTABLE_FPS: Final[int] = 15
CRITICAL_FPS: Final[int] = 5
MEMORY_CHECK_INTERVAL: Final[int] = 30
GC_INTERVAL: Final[int] = 60
CLEANUP_INTERVAL: Final[int] = 50

# PROCESAMIENTO DE FRAMES
MAX_FRAME_SKIP: Final[int] = 2
MIN_FRAME_SKIP: Final[int] = 1
PROCESS_EVERY_N_FRAMES: Final[int] = 1

# CACHÉ
DEFAULT_CACHE_SIZE: Final[int] = 16
MAX_CACHE_SIZE: Final[int] = 64
MIN_CACHE_SIZE: Final[int] = 4
MAX_CACHE_MEMORY_MB: Final[int] = 250
CACHE_CLEANUP_THRESHOLD: Final[float] = 0.6
CACHE_ENTRY_SIZE_ESTIMATE: Final[int] = 16

# COLORES
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

DETECTION_COLORS: Final[List[Tuple[int, int, int]]] = [
    (0, 255, 0),
    (255, 165, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
]

# VISUALIZACIÓN Y DASHBOARD
DASHBOARD_WIDTH: Final[int] = 220
DASHBOARD_HEIGHT: Final[int] = 120
DASHBOARD_ALPHA: Final[float] = 0.7
FONT_SCALE: Final[float] = 0.5
LINE_THICKNESS: Final[int] = 2
POINT_RADIUS: Final[int] = 4
TRAIL_POINTS: Final[int] = 15

# Dimensiones de ventana
MIN_WINDOW_WIDTH: Final[int] = 320
MIN_WINDOW_HEIGHT: Final[int] = 240
MAX_WINDOW_WIDTH: Final[int] = 1920
MAX_WINDOW_HEIGHT: Final[int] = 1080

# MEMORIA
MEMORY_WARNING_THRESHOLD: Final[float] = 70.0
MEMORY_CRITICAL_THRESHOLD: Final[float] = 80.0
MEMORY_LIMIT_MB: Final[int] = 2048

# DISPOSITIVOS
PREFERRED_DEVICE_ORDER: Final[List[str]] = ["cuda", "mps", "cpu"]

# ARCHIVOS Y DIRECTORIOS
DEFAULT_CONFIG_PATH: Final[str] = "config.yaml"
DEFAULT_DATA_DIR: Final[str] = "data/"
SCREENSHOTS_DIR: Final[str] = "data/screenshots/"
EXPORTS_DIR: Final[str] = "data/exports/"
LOGS_DIR: Final[str] = "data/logs/"

# VALIDACIONES
VALID_IMGSZ: Final[List[int]] = [320, 416, 512, 640, 768, 832, 1024]
VALID_FPS_RANGE: Final[Tuple[float, float]] = (1.0, 120.0)
VALID_CONFIDENCE_RANGE: Final[Tuple[float, float]] = (0.0, 1.0)
VALID_IOU_RANGE: Final[Tuple[float, float]] = (0.0, 1.0)

# LOGGING
LOG_LEVELS: Final[Dict[str, int]] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
LOG_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# ANÁLISIS Y CONGESTIÓN
CONGESTION_LOW: Final[float] = 0.3
CONGESTION_MEDIUM: Final[float] = 0.6
CONGESTION_HIGH: Final[float] = 0.8

ANALYSIS_WINDOW_SECONDS: Final[int] = 60
PREDICTION_HORIZON_SECONDS: Final[int] = 300
PREDICTION_SAMPLES: Final[int] = 100

# EXPORTACIÓN
SUPPORTED_EXPORT_FORMATS: Final[List[str]] = ["json", "csv", "both"]
SUPPORTED_IMAGE_FORMATS: Final[List[str]] = ["jpg", "png", "bmp", "tiff"]
AUTO_SAVE_INTERVAL_SECONDS: Final[int] = 300

# CONSTANTES DE SISTEMA
DEFAULT_ENCODING: Final[str] = "utf-8"
DEFAULT_CONFIG_FILENAME: Final[str] = "config.yaml"
DEFAULT_LOG_FILENAME: Final[str] = "system.log"

SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3600
MILLISECONDS_PER_SECOND: Final[int] = 1000
BYTES_PER_MB: Final[int] = 1024 * 1024
BYTES_PER_KB: Final[int] = 1024

# CONSTANTES DE IMAGEN
DEFAULT_IMAGE_WIDTH: Final[int] = 640
DEFAULT_IMAGE_HEIGHT: Final[int] = 480
DEFAULT_LANE_WIDTH: Final[int] = 40
DEFAULT_BUFFER_ZONE: Final[int] = 15

# CONSTANTES DE FUENTE Y RENDERIZADO
DEFAULT_FONT: Final[int] = 0
DEFAULT_FONT_SCALE: Final[float] = 0.5
DEFAULT_FONT_THICKNESS: Final[int] = 2
DEFAULT_LINE_THICKNESS: Final[int] = 2

# CONSTANTES DE VENTANA
WINDOW_NAME: Final[str] = "Vehicle Counting System"
DEFAULT_WINDOW_WIDTH: Final[int] = 1280
DEFAULT_WINDOW_HEIGHT: Final[int] = 720

# CONSTANTES DE CAPTURA
CAPTURE_MIN_FPS_CPU: Final[float] = 5.0
CAPTURE_MAX_FPS_CPU: Final[float] = 15.0
CAPTURE_TARGET_FPS_CPU: Final[float] = 8.0
CAPTURE_TARGET_FPS_GPU: Final[float] = 30.0
CAPTURE_DEFAULT_INTERVAL_CPU: Final[float] = 1.0 / 8.0
CAPTURE_DEFAULT_INTERVAL_GPU: Final[float] = 1.0 / 30.0
CAPTURE_RECONNECT_DELAY: Final[float] = 1.0
CAPTURE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5

# CONSTANTES DE BUFFER
BUFFER_SIZE_CPU: Final[int] = 20
BUFFER_SIZE_GPU: Final[int] = 30
MAX_WORKERS_CPU: Final[int] = 4
MAX_WORKERS_GPU: Final[int] = 8
MIN_WORKERS_CPU: Final[int] = 2
BUFFER_DROP_THRESHOLD: Final[float] = 0.8
BUFFER_RECOVERY_THRESHOLD: Final[float] = 0.3
BUFFER_SKIP_MAX: Final[int] = 2
BUFFER_SKIP_CONSECUTIVE_LIMIT: Final[int] = 5

# CONSTANTES DE FRAME
MIN_FRAME_DIMENSION: Final[int] = 10
MIN_FRAME_WIDTH: Final[int] = 10
MIN_FRAME_HEIGHT: Final[int] = 10
DEFAULT_FRAME_WIDTH: Final[int] = 640
DEFAULT_FRAME_HEIGHT: Final[int] = 480
DEFAULT_FRAME_CHANNELS: Final[int] = 3

# CONSTANTES DE RENDERIZADO
DEFAULT_RENDER_WIDTH: Final[int] = 640
DEFAULT_RENDER_HEIGHT: Final[int] = 480
DEFAULT_RENDER_CHANNELS: Final[int] = 3
RENDER_ERROR_COOLDOWN: Final[float] = 1.0
MAX_RENDER_TIMES: Final[int] = 100

# CONSTANTES DE HEALTH CHECK
HEALTH_CHECK_INTERVAL: Final[float] = 10.0
HEALTH_BUFFER_CRITICAL: Final[float] = 0.85
HEALTH_BUFFER_WARNING: Final[float] = 0.7
HEALTH_QUEUE_CRITICAL: Final[int] = 30
HEALTH_QUEUE_WARNING: Final[int] = 15
HEALTH_FPS_CRITICAL: Final[float] = 3.0
HEALTH_FPS_WARNING: Final[float] = 8.0
HEALTH_DROP_RATE_CRITICAL: Final[float] = 0.3
HEALTH_DROP_RATE_WARNING: Final[float] = 0.1

# CONSTANTES DE PIPELINE
PIPELINE_MAX_RECONNECT_ATTEMPTS: Final[int] = 3
PIPELINE_RECONNECT_DELAY: Final[float] = 0.1
PIPELINE_MAX_CONSECUTIVE_ERRORS: Final[int] = 5
PIPELINE_DEFAULT_FRAME_TIMEOUT: Final[int] = 100
PIPELINE_MAX_RENDER_QUEUE_RATIO: Final[float] = 0.33

# CONSTANTES DE TRACK (ARROWS, CÍRCULOS, ETC.)
TRACK_ARROW_LENGTH_MIN: Final[int] = 10
TRACK_ARROW_LENGTH_MAX: Final[int] = 30

TRACK_CIRCLE_RADIUS: Final[int] = 6
TRACK_CONFIDENCE_RADIUS_MIN: Final[int] = 2
TRACK_CONFIDENCE_RADIUS_MAX: Final[int] = 6

TRACK_TRAIL_THICKNESS_MIN: Final[int] = 1
TRACK_TRAIL_THICKNESS_MAX: Final[int] = 2

TRACK_BBOX_THICKNESS_MIN: Final[int] = 1
TRACK_BBOX_THICKNESS_MAX: Final[int] = 2

PREDICTION_POINT_RADIUS_MIN: Final[int] = 2
PREDICTION_POINT_RADIUS_MAX: Final[int] = 5

# CONSTANTES DE ESTADOS (para máquina de estados)
STATUS_COLORS: Final[Dict[str, Tuple[Tuple[int, int, int], str, str]]] = {
    "confirmed": ((0, 255, 0), "✅", "OK"),
    "lost": ((0, 255, 255), "⚠️", "Lost"),
    "tentative": ((255, 255, 0), "⏳", "New"),
    "dead": ((128, 128, 128), "💀", "Dead"),
}

PREDICTION_STATE_COLORS: Final[Dict[str, Tuple[int, int, int]]] = {
    "stopped": (0, 0, 255),
    "accelerating": (0, 255, 255),
    "decelerating": (0, 165, 255),
    "turning": (255, 0, 255),
    "erratic": (255, 0, 0),
    "moving": (255, 255, 0),
    "unknown": (255, 255, 0),
}

# CONSTANTES DE APRENDIZAJE EN LÍNEA
ONLINE_LEARNING_DEFAULT_LR: Final[float] = 0.05
ONLINE_LEARNING_MIN_SAMPLES: Final[int] = 5
ONLINE_LEARNING_DRIFT_THRESHOLD: Final[float] = 0.35
ONLINE_LEARNING_MAX_HISTORY: Final[int] = 50

# CONSTANTES DE FUSIÓN DE SENSORES
SENSOR_FUSION_VISUAL_WEIGHT: Final[float] = 0.7
SENSOR_FUSION_DEPTH_WEIGHT: Final[float] = 0.5
SENSOR_FUSION_THERMAL_WEIGHT: Final[float] = 0.4
SENSOR_FUSION_MOTION_WEIGHT: Final[float] = 0.3
SENSOR_FUSION_MIN_OBSERVATIONS: Final[int] = 2
SENSOR_FUSION_MAX_HISTORY: Final[int] = 50
SENSOR_FUSION_PARTICLE_COUNT: Final[int] = 500

# CONSTANTES DE PREDICCIÓN DE TRAYECTORIA
PATH_PREDICTION_HISTORY_LENGTH: Final[int] = 30
PATH_PREDICTION_HORIZON: Final[float] = 2.0
PATH_PREDICTION_STEPS: Final[int] = 20
PATH_PREDICTION_MIN_SAMPLES: Final[int] = 5
PATH_PREDICTION_UNCERTAINTY_THRESHOLD: Final[float] = 0.7

# CONSTANTES DE MHT (Multi-Hypothesis Tracking)
MHT_MAX_DEPTH: Final[int] = 5
MHT_PRUNING_THRESHOLD: Final[float] = 0.01
MHT_MAX_HYPOTHESES: Final[int] = 3

# CONSTANTES DE RE-IDENTIFICACIÓN
REID_SIMILARITY_THRESHOLD: Final[float] = 0.6
REID_SPATIAL_THRESHOLD: Final[float] = 100.0
REID_MAX_AGE_SECONDS: Final[float] = 30.0
REID_CACHE_SIZE: Final[int] = 1000
REID_MIN_FEATURES: Final[int] = 3

# CONSTANTES DE VALIDACIÓN DE TRACKS
TRACK_VALIDATION_MIN_CONFIDENCE: Final[float] = 0.3
TRACK_VALIDATION_MAX_SPEED_CHANGE: Final[float] = 50.0
TRACK_VALIDATION_IOU_THRESHOLD: Final[float] = 0.3
TRACK_VALIDATION_FEATURE_THRESHOLD: Final[float] = 0.6
TRACK_VALIDATION_MOTION_THRESHOLD: Final[float] = 0.7
TRACK_VALIDATION_SHAPE_THRESHOLD: Final[float] = 0.5
