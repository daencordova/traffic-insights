"""Constantes de visión: detección, colores, imágenes, renderizado."""

from typing import Final

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

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.35
"""Umbral de confianza por defecto para detecciones."""

DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
"""Umbral de IoU por defecto para NMS."""

# COLORES
COLORS: Final[dict[str, tuple[int, int, int]]] = {
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

DETECTION_COLORS: Final[list[tuple[int, int, int]]] = [
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
"""Paleta de colores para detecciones y tracks."""

# IMAGEN
DEFAULT_IMAGE_WIDTH: Final[int] = 640
"""Ancho de imagen por defecto."""

DEFAULT_IMAGE_HEIGHT: Final[int] = 480
"""Alto de imagen por defecto."""

DEFAULT_LANE_WIDTH: Final[int] = 40
"""Ancho de carril por defecto."""

DEFAULT_BUFFER_ZONE: Final[int] = 15
"""Zona de buffer por defecto."""

# FRAME
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

# RENDERIZADO
DEFAULT_RENDER_WIDTH: Final[int] = 640
"""Ancho de renderizado por defecto."""

DEFAULT_RENDER_HEIGHT: Final[int] = 480
"""Alto de renderizado por defecto."""

DEFAULT_RENDER_CHANNELS: Final[int] = 3
"""Canales de renderizado por defecto."""

DEFAULT_FONT: Final[int] = 0
"""Fuente por defecto (OpenCV)."""

DEFAULT_FONT_SCALE: Final[float] = 0.5
"""Escala de fuente por defecto."""

DEFAULT_FONT_THICKNESS: Final[int] = 2
"""Grosor de fuente por defecto."""

DEFAULT_LINE_THICKNESS: Final[int] = 2
"""Grosor de línea por defecto."""

# FORMATOS SOPORTADOS
SUPPORTED_IMAGE_FORMATS: Final[list[str]] = ["jpg", "png", "bmp", "tiff"]
"""Formatos de imagen soportados."""

# PROCESAMIENTO DE IMÁGENES
IMAGE_RESIZE_DEFAULT: Final[tuple[int, int]] = (32, 32)
"""Tamaño por defecto para redimensionar imágenes."""

IMAGE_PREPROCESS_DENOISE_STRENGTH: Final[int] = 5
"""Fuerza de reducción de ruido."""

IMAGE_EQUALIZE_HISTOGRAM: Final[bool] = True
"""Si aplicar ecualización de histograma."""

IMAGE_ENHANCE_CONTRAST: Final[bool] = True
"""Si mejorar contraste."""

# VENTANA
WINDOW_NAME: Final[str] = "Vehicle Counting System"
"""Nombre de la ventana de visualización."""

DEFAULT_WINDOW_WIDTH: Final[int] = 1280
"""Ancho de ventana por defecto."""

DEFAULT_WINDOW_HEIGHT: Final[int] = 720
"""Alto de ventana por defecto."""

# VALIDACIÓN DE BBOX
MAX_BBOX_DIMENSION: Final[int] = 4
"""Número máximo de elementos en un bounding box."""

MAX_BRIGHTNESS: Final[int] = 240
"""Brillo máximo permitido antes de considerar sobreexpuesto."""

MIN_BBOX_SIZE: Final[int] = 10
"""Tamaño mínimo de un bounding box en píxeles."""
