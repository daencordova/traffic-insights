"""
Servicio de control y eventos de usuario.

Responsable de:
- Manejar eventos de teclado
- Gestionar pausa/reanudación
- Capturar pantallas
- Reiniciar el sistema
- Mostrar ayuda
"""

from typing import Optional, Callable

import numpy as np

from core.pipeline.controls import ControlHandler
from utils.logger import LoggerMixin


class ControlService(LoggerMixin):
    """
    Servicio especializado en control y eventos de usuario.
    """

    def __init__(
        self,
        config,
        controls: Optional[ControlHandler] = None,
        on_pause_toggle: Optional[Callable] = None,
        on_reset: Optional[Callable] = None,
        on_screenshot: Optional[Callable] = None,
    ):
        self.config = config
        self.controls = controls or ControlHandler(config)
        self.on_pause_toggle = on_pause_toggle
        self.on_reset = on_reset
        self.on_screenshot = on_screenshot

        self._last_frame: Optional[np.ndarray] = None

        self.controls.register_callback("on_pause_toggle", self._handle_pause_toggle)
        self.controls.register_callback("on_reset", self._handle_reset)
        self.controls.register_callback("on_screenshot", self._handle_screenshot)

        self.logger.info("ControlService inicializado")

    def handle_key(self, key: int) -> bool:
        """
        Maneja una tecla presionada.

        Args:
            key: Código de la tecla

        Returns:
            bool: True si el sistema debe continuar ejecutándose
        """
        return self.controls.process_key(key)

    def set_last_frame(self, frame: np.ndarray) -> None:
        """Establece el último frame para capturas."""
        self._last_frame = frame
        self.controls.set_last_frame(frame)

    def _handle_pause_toggle(self, is_paused: bool) -> None:
        """Maneja el toggle de pausa."""
        if self.on_pause_toggle:
            self.on_pause_toggle(is_paused)

    def _handle_reset(self) -> None:
        """Maneja el reinicio del sistema."""
        if self.on_reset:
            self.on_reset()

    def _handle_screenshot(self, filepath: str) -> None:
        """Maneja la captura de pantalla."""
        if self.on_screenshot:
            self.on_screenshot(filepath)

    def save_screenshot(self, frame: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Guarda una captura de pantalla.

        Args:
            frame: Frame a guardar (opcional)

        Returns:
            Optional[str]: Ruta del archivo guardado
        """
        if frame is None:
            frame = self._last_frame

        if frame is None:
            self.logger.warning("No hay frame para capturar")
            return None

        return self.controls.save_screenshot(frame)

    def is_paused(self) -> bool:
        """Verifica si el sistema está pausado."""
        return self.controls.is_paused

    def is_running(self) -> bool:
        """Verifica si el sistema está en ejecución."""
        return self.controls.is_running

    def toggle_pause(self) -> None:
        """Alterna el estado de pausa."""
        self.controls.toggle_pause()

    def reset(self) -> None:
        """Reinicia el sistema."""
        self.controls.reset_system()

    def show_help(self) -> None:
        """Muestra la ayuda."""
        self.controls.show_help()

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        return {
            'is_paused': self.is_paused(),
            'is_running': self.is_running(),
            'has_last_frame': self._last_frame is not None,
        }
