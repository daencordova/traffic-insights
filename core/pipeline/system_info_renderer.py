"""
Renderizador de la información del sistema.
"""

from typing import Tuple

import cv2
import numpy as np

from core.pipeline.renderer_config import RendererConfig
from core.pipeline.system_info import SystemInfo, SystemInfoCollector
from core.pipeline.text_utils import TextMetricsCache


class SystemInfoRenderer:
    """
    Renderiza la información del sistema (estado, FPS, CPU, memoria, etc.)
    en la parte inferior del frame.
    """

    __slots__ = ("_config", "_system_info_collector", "_text_cache", "_pipeline_status")

    def __init__(
        self,
        config: RendererConfig,
        system_info_collector: SystemInfoCollector,
        text_cache: TextMetricsCache,
    ):
        self._config = config
        self._system_info_collector = system_info_collector
        self._text_cache = text_cache
        self._pipeline_status = "RUNNING"

    def render(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Renderiza la información del sistema en el frame.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        fps = kwargs.get("fps", 0.0)
        processing_time_ms = kwargs.get("processing_time_ms", 0.0)
        tracks = kwargs.get("tracks", {})

        status = self._get_pipeline_status()
        self._system_info_collector.set_status(status.value if hasattr(status, "value") else status)

        info = self._system_info_collector.get_info(
            fps=fps,
            processing_time_ms=processing_time_ms,
            active_tracks=len(tracks),
        )

        self._draw_status_line(frame, status, info)

        return frame

    def _draw_status_line(self, frame: np.ndarray, status, info: SystemInfo) -> None:
        """
        Dibuja la línea de estado e información del sistema.
        """
        try:
            h, w = frame.shape[:2]
            x = 10
            y = h - 15

            status_text = f"{status.value if hasattr(status, 'value') else status}"
            status_color = self._get_status_color(status, info)

            info_text = (
                f"FPS: {info.fps:.1f} | "
                f"CPU: {info.cpu_percent:.0f}% | "
                f"MEMORIA: {info.memory_used_mb:.0f} MB | "
                f"DETECCIONES: {info.active_tracks} | "
                f"TIEMPO: {info.processing_time_ms:.1f} MS | Q: SALIR | ESPACIO: PAUSA | H: AYUDA"
            )

            cv2.putText(
                frame,
                f"{status_text} | ",
                (x, y),
                self._config.font,
                self._config.font_scale,
                status_color,
                self._config.font_thickness,
                cv2.LINE_AA,
            )

            (status_width, _), _ = cv2.getTextSize(
                f"{status_text} | ",
                self._config.font,
                self._config.font_scale,
                self._config.font_thickness,
            )

            cv2.putText(
                frame,
                info_text,
                (x + status_width, y),
                self._config.font,
                self._config.font_scale,
                self._config.info_color,
                self._config.font_thickness,
                cv2.LINE_AA,
            )

        except Exception as e:
            pass

    def _get_status_color(self, status, info: SystemInfo) -> Tuple[int, int, int]:
        """
        Obtiene el color para el estado del pipeline.
        """
        status_value = status.value if hasattr(status, "value") else status

        if status_value == "STOPPED":
            return self._config.status_color_stopped
        elif status_value == "PAUSED":
            return self._config.status_color_paused
        elif status_value == "ERROR":
            return self._config.status_color_error

        return self._system_info_collector.get_color_for_performance(
            info.fps, info.cpu_percent
        )

    def _get_pipeline_status(self):
        """
        Obtiene el estado actual del pipeline.
        """
        return self._pipeline_status

    def set_pipeline_status(self, status) -> None:
        """Establece el estado del pipeline."""
        self._pipeline_status = status
