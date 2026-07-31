"""Constantes de visualización: dashboard, UI, controles de usuario."""

from typing import Final

# DASHBOARD
DASHBOARD_WIDTH: Final[int] = 220
"""Ancho del dashboard en píxeles."""

DASHBOARD_HEIGHT: Final[int] = 120
"""Alto del dashboard en píxeles."""

DASHBOARD_ALPHA: Final[float] = 0.7
"""Opacidad del dashboard (0 = transparente, 1 = opaco)."""

# FUENTE Y UI
FONT_SCALE: Final[float] = 0.5
"""Escala de fuente para textos en UI."""

LINE_THICKNESS: Final[int] = 2
"""Grosor de líneas en UI."""

POINT_RADIUS: Final[int] = 4
"""Radio de puntos en UI."""

TRAIL_POINTS: Final[int] = 15
"""Número de puntos en la trayectoria (trail)."""

# TRACK VISUALIZATION
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

# CONTROLES DE USUARIO
CONTROL_KEY_QUIT: Final[int] = ord("q")
"""Tecla para salir."""

CONTROL_KEY_ESCAPE: Final[int] = 27
"""Tecla ESC para salir."""

CONTROL_KEY_PAUSE: Final[int] = ord(" ")
"""Tecla ESPACIO para pausar/reanudar."""

CONTROL_KEY_SCREENSHOT: Final[int] = ord("s")
"""Tecla S para captura de pantalla."""

CONTROL_KEY_RESET: Final[int] = ord("r")
"""Tecla R para reiniciar."""

CONTROL_KEY_HELP: Final[int] = ord("h")
"""Tecla H para ayuda."""

# COLORES Y VISUALIZACIÓN
HUE_SEGMENTS: Final[int] = 6
"""Número de segmentos para la rueda de colores HSV."""

HUE_CYCLE: Final[int] = 360
"""Ciclo completo de la rueda de colores HSV."""

SATURATION: Final[int] = 200
"""Saturación por defecto para colores generados (0-255)."""

VALUE: Final[int] = 200
"""Valor/brillo por defecto para colores generados (0-255)."""

MAX_COLOR_INDEX: Final[int] = 255
"""Valor máximo para índices de color."""

COLOR_CHANNEL_MAX: Final[int] = 255
"""Valor máximo para un canal de color."""
