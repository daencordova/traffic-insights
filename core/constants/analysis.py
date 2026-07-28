"""Constantes de análisis: congestión, exportación, ventanas."""

from typing import Final

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
SUPPORTED_EXPORT_FORMATS: Final[list[str]] = ["json", "csv", "both"]
"""Formatos de exportación soportados."""

AUTO_SAVE_INTERVAL_SECONDS: Final[int] = 300
"""Intervalo de auto-guardado en segundos."""

# VENTANAS
MIN_WINDOW_WIDTH: Final[int] = 320
"""Ancho mínimo de la ventana de visualización."""

MIN_WINDOW_HEIGHT: Final[int] = 240
"""Alto mínimo de la ventana de visualización."""

MAX_WINDOW_WIDTH: Final[int] = 1920
"""Ancho máximo de la ventana de visualización."""

MAX_WINDOW_HEIGHT: Final[int] = 1080
"""Alto máximo de la ventana de visualización."""

# VALIDACIÓN DE CONFIGURACIÓN
MINIMUM_BUFFER_MEMORY_MB: Final[int] = 500
"""Memoria mínima para buffer."""

MAX_RECOMMENDED_FPS: Final[int] = 60
"""FPS máximo recomendado."""
