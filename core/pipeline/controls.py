"""
Manejador de controles de teclado para el pipeline.

Gestiona los eventos de teclado y las acciones correspondientes:
- Pausar/Reanudar
- Capturar pantalla
- Reiniciar contadores
- Mostrar ayuda
- Salir
"""

from typing import Optional, Callable, Dict
import os

import cv2

from utils.logger import LoggerMixin
from utils.helpers import ensure_directory_exists, get_timestamp_filename


class ControlHandler(LoggerMixin):
    """
    Manejador de controles de teclado.

    Responsabilidades:
    - Procesar eventos de teclado
    - Ejecutar acciones asociadas
    - Mantener estado de pausa
    - Gestionar capturas de pantalla

    Attributes:
        config: Configuración del sistema
        is_paused: Estado de pausa
        callbacks: Diccionario de callbacks para acciones
    """

    def __init__(self, config=None):
        self.config = config or config
        self.is_paused = False
        self.is_running = True

        self.callbacks: Dict[str, Callable] = {}

        self.logger.info("ControlHandler inicializado")

    def process_key(self, key: int) -> bool:
        """
        Procesa una tecla presionada.

        Args:
            key: Código de tecla (valor de cv2.waitKey)

        Returns:
            bool: True si el sistema debe continuar ejecutándose
        """
        if key == ord('q') or key == 27:
            self.is_running = False
            self.logger.info("Tecla de salida presionada")
            return False

        elif key == ord(' '):
            self.toggle_pause()
            status = "pausado" if self.is_paused else "reanudado"
            self.logger.info(f"Sistema {status}")
            print(f"{'⏸️' if self.is_paused else '▶️'} Sistema {status}")

        elif key == ord('s'):
            self.save_screenshot()

        elif key == ord('r'):
            self.reset_system()

        elif key == ord('h'):
            self.show_help()

        return True

    def toggle_pause(self) -> None:
        """Alterna el estado de pausa."""
        self.is_paused = not self.is_paused
        if "on_pause_toggle" in self.callbacks:
            self.callbacks["on_pause_toggle"](self.is_paused)

    def save_screenshot(self, frame: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Guarda una captura de pantalla.

        Args:
            frame: Frame a guardar (si es None, usa el último frame)

        Returns:
            Optional[str]: Ruta del archivo guardado o None
        """
        if frame is None:
            self.logger.warning("No hay frame para capturar")
            return None

        try:
            filename = get_timestamp_filename("capture", "jpg")
            filepath = os.path.join(self.config.output.screenshots_dir, filename)
            ensure_directory_exists(self.config.output.screenshots_dir)

            cv2.imwrite(filepath, frame)
            self.logger.info(f"Captura guardada: {filepath}")
            print(f"📸 Captura guardada: {filepath}")

            if "on_screenshot" in self.callbacks:
                self.callbacks["on_screenshot"](filepath)

            return filepath

        except Exception as e:
            self.logger.error(f"Error guardando captura: {e}")
            return None

    def reset_system(self) -> None:
        """Reinicia los contadores y el tracker."""
        if "on_reset" in self.callbacks:
            self.callbacks["on_reset"]()
            print("🔄 Sistema reiniciado")
            self.logger.info("Sistema reiniciado")

    def show_help(self) -> None:
        """Muestra la ayuda en consola."""
        print("""
        ═══════════════════════════════════════════════════
        🎮 CONTROLES DEL SISTEMA
        ═══════════════════════════════════════════════════
        q / ESC  → Salir
        SPACE    → Pausar/Reanudar
        s        → Captura de pantalla
        r        → Reiniciar contadores
        h        → Esta ayuda
        ═══════════════════════════════════════════════════
        """)

    def register_callback(self, action: str, callback: Callable) -> None:
        """
        Registra un callback para una acción.

        Args:
            action: Nombre de la acción
            callback: Función a ejecutar
        """
        self.callbacks[action] = callback
        self.logger.debug(f"Callback registrado: {action}")

    def unregister_callback(self, action: str) -> None:
        """Elimina un callback registrado."""
        if action in self.callbacks:
            del self.callbacks[action]
            self.logger.debug(f"Callback eliminado: {action}")

    def set_last_frame(self, frame: np.ndarray) -> None:
        """Establece el último frame para capturas."""
        self._last_frame = frame
