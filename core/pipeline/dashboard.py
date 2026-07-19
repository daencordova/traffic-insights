"""
Renderizador del dashboard con estadísticas en pantalla.

Muestra información clave del sistema como:
- Total de vehículos contados
- Tracks activos
- FPS y tiempo de procesamiento
- Conteos por línea
- Estado del sistema
"""

from typing import Dict, Any

import cv2
import numpy as np

from utils.logger import LoggerMixin
from core.constants import (
    COLORS,
    DASHBOARD_WIDTH,
    DASHBOARD_HEIGHT,
    DASHBOARD_ALPHA,
    TARGET_FPS,
    MIN_ACCEPTABLE_FPS
)


class DashboardRenderer(LoggerMixin):
    """
    Renderizador del dashboard de estadísticas.

    Responsabilidades:
    - Dibujar el panel de estadísticas
    - Mostrar conteos de vehículos
    - Mostrar métricas de rendimiento
    - Actualizar información en tiempo real

    Attributes:
        config: Configuración del sistema
        position: Posición del dashboard (top-left, top-right, etc.)
        width: Ancho del dashboard
        height: Alto del dashboard
    """

    def __init__(self, config=None):
        self.config = config or config
        self.position = self.config.visualization.dashboard_position
        self.width = DASHBOARD_WIDTH
        self.height = DASHBOARD_HEIGHT
        self._default_frame_size = (480, 640)

        self.logger.info(
            "DashboardRenderer inicializado",
            position=self.position,
            width=self.width,
            height=self.height
        )

    def render(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any],
        fps: float = 0.0,
        processing_time_ms: float = 0.0,
        frame_number: int = 0
    ) -> np.ndarray:
        """
        Renderiza el dashboard en el frame.

        Args:
            frame: Frame base
            stats: Estadísticas del sistema
            fps: FPS actual
            processing_time_ms: Tiempo de procesamiento en ms
            frame_number: Número de frame

        Returns:
            np.ndarray: Frame con dashboard (siempre retorna un array válido)
        """
        if frame is None:
            self.logger.warning("Dashboard: frame es None, creando frame por defecto")
            h, w = self._default_frame_size
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        if not isinstance(frame, np.ndarray):
            self.logger.warning(f"Dashboard: frame no es numpy array: {type(frame)}")
            h, w = self._default_frame_size
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        if frame.size == 0:
            self.logger.warning("Dashboard: frame está vacío")
            h, w = self._default_frame_size
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) != 3:
            self.logger.warning(f"Dashboard: frame shape inválido: {frame.shape}")
            h, w = self._default_frame_size
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        try:
            h, w = frame.shape[:2]

            x, y = self._get_position((h, w))

            dashboard_w = min(self.width, w - 20)
            dashboard_h = min(self.height, h - 20)

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x, y),
                (x + dashboard_w, y + dashboard_h),
                COLORS["BLACK"],
                -1
            )
            cv2.addWeighted(overlay, DASHBOARD_ALPHA, frame, 1 - DASHBOARD_ALPHA, 0, frame)

            border_color = self._get_performance_color(fps)
            cv2.rectangle(
                frame,
                (x, y),
                (x + dashboard_w, y + dashboard_h),
                border_color,
                2
            )

            y_offset = y + 25

            info_items = [
                ("🚗 Total", f"{stats.get('total', 0):>6}"),
                ("🎯 Activos", f"{stats.get('active_objects', 0):>6}"),
                ("⚡ FPS", f"{fps:>6.1f}"),
                ("⏱️ Tiempo", f"{processing_time_ms:>5.1f}ms"),
            ]

            for label, value in info_items:
                cv2.putText(
                    frame,
                    label,
                    (x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS["LIGHT_GRAY"],
                    1
                )
                cv2.putText(
                    frame,
                    value,
                    (x + dashboard_w - 80, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS["WHITE"],
                    1
                )
                y_offset += 22

            line_counts = stats.get("line_counts", {})
            if len(line_counts) <= 4 and line_counts:
                y_offset = y + dashboard_h - 10
                for idx, (line_id, count) in enumerate(line_counts.items()):
                    if idx >= 4:
                        break

                    line_name = self._get_line_name(line_id)
                    color = self._get_line_color(idx)

                    x_pos = x + 10 + idx * 55

                    if x_pos + 50 < x + dashboard_w:
                        text = f"{line_name}:{count}"
                        cv2.putText(
                            frame,
                            text,
                            (x_pos, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1
                        )

        except Exception as e:
            self.logger.warning(f"Error en renderizado de dashboard: {e}")
            pass

        return frame

    def _get_position(self, frame_shape: tuple) -> tuple:
        """Calcula la posición del dashboard según la configuración."""
        h, w = frame_shape[:2]

        positions = {
            "top-left": (10, 10),
            "top-right": (w - self.width - 10, 10),
            "bottom-left": (10, h - self.height - 10),
            "bottom-right": (w - self.width - 10, h - self.height - 10),
        }

        pos = positions.get(self.position, (10, 10))

        pos_x = max(0, pos[0])
        pos_y = max(0, pos[1])

        pos_x = min(pos_x, w - self.width - 10)
        pos_y = min(pos_y, h - self.height - 10)

        return (pos_x, pos_y)

    def _get_performance_color(self, fps: float) -> tuple:
        """Obtiene el color del borde según el rendimiento."""
        if fps >= TARGET_FPS:
            return COLORS["GREEN"]
        elif fps >= MIN_ACCEPTABLE_FPS:
            return COLORS["YELLOW"]
        else:
            return COLORS["RED"]

    def _get_line_name(self, line_id: str) -> str:
        """Obtiene el nombre de una línea de conteo."""
        for line in self.config.counting_lines:
            if line.get("id") == line_id:
                return line.get("name", line_id[:2])
        return line_id[:2]

    def _get_line_color(self, idx: int) -> tuple:
        """Obtiene el color para una línea de conteo."""
        colors = [
            (0, 255, 0),
            (255, 165, 0),
            (255, 0, 0),
            (255, 255, 0),
        ]
        return colors[idx % len(colors)]
