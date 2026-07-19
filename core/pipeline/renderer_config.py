"""
Configuración para el renderizador de frames.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2


@dataclass(frozen=True)
class RendererConfig:
    """
    Configuración inmutable del renderizador.

    Attributes:
        default_width: Ancho por defecto del frame
        default_height: Alto por defecto del frame
        font: Fuente OpenCV
        font_scale: Escala de la fuente
        font_thickness: Grosor de la fuente
        info_color: Color de información (BGR)
        help_color: Color de ayuda (BGR)
        error_color: Color de error (BGR)
        status_color_running: Color para estado RUNNING (BGR)
        status_color_paused: Color para estado PAUSED (BGR)
        status_color_stopped: Color para estado STOPPED (BGR)
        status_color_error: Color para estado ERROR (BGR)
        show_controls_help: Mostrar ayuda de controles
        show_system_info: Mostrar la información del sistema
        show_velocity_vectors: Mostrar vectores de velocidad
        show_trails: Mostrar trayectorias
        show_track_arrows: Mostrar flechas de dirección
        show_track_speed: Mostrar velocidad
        show_track_confidence: Mostrar confianza
        track_circle_style: Estilo de círculo para tracks
    """

    default_width: int = 640
    default_height: int = 480
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.38
    font_thickness: int = 1
    info_color: Tuple[int, int, int] = (200, 200, 200)
    help_color: Tuple[int, int, int] = (128, 128, 128)
    error_color: Tuple[int, int, int] = (0, 0, 255)
    status_color_running: Tuple[int, int, int] = (0, 255, 0)
    status_color_paused: Tuple[int, int, int] = (0, 255, 255)
    status_color_stopped: Tuple[int, int, int] = (0, 0, 255)
    status_color_error: Tuple[int, int, int] = (0, 0, 255)
    show_controls_help: bool = True
    show_system_info: bool = True
    show_velocity_vectors: bool = True
    show_trails: bool = True
    show_track_arrows: bool = True
    show_track_speed: bool = True
    show_track_confidence: bool = True
    track_circle_style: str = "solid"

    @classmethod
    def from_global_config(cls, config) -> "RendererConfig":
        """Crea una configuración desde la configuración global del sistema."""
        if config is None:
            return cls()

        visualization = config.visualization
        return cls(
            show_controls_help=visualization.show_controls_help,
            show_system_info=visualization.show_system_info,
            show_velocity_vectors=visualization.show_velocity_vectors,
            show_trails=visualization.show_trails,
            show_track_arrows=getattr(visualization, "show_track_arrows", True),
            show_track_speed=getattr(visualization, "show_track_speed", True),
            show_track_confidence=getattr(visualization, "show_track_confidence", True),
            track_circle_style=getattr(visualization, "track_circle_style", "solid"),
        )
