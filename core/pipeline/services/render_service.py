"""
Servicio de renderizado y visualización.

Responsable de:
- Renderizar overlays en los frames
- Mostrar frames en ventana
- Manejar eventos de teclado
- Gestionar la cola de renderizado
"""

import time
import threading
from typing import Optional, Callable, List

import cv2
import numpy as np

from core.pipeline.renderer import FrameRenderer
from core.pipeline.controls import ControlHandler
from core.pipeline.services.processing_service import ProcessingResult
from utils.logger import LoggerMixin
from core.constants import WINDOW_NAME


class RenderService(LoggerMixin):
    """
    Servicio especializado en renderizado y visualización.
    """

    def __init__(
        self,
        config,
        renderer: Optional[FrameRenderer] = None,
        controls: Optional[ControlHandler] = None,
        max_queue_size: int = 3,
        on_key_pressed: Optional[Callable] = None,
    ):
        self.config = config
        self.renderer = renderer or FrameRenderer(config)
        self.controls = controls or ControlHandler(config)
        self.max_queue_size = max_queue_size
        self.on_key_pressed = on_key_pressed

        self._render_queue: List[ProcessingResult] = []
        self._queue_lock = threading.Lock()
        self._last_valid_frame: Optional[np.ndarray] = None
        self._running = False
        self._paused = False

        self._thread: Optional[threading.Thread] = None
        self._frames_rendered = 0
        self._frames_dropped = 0
        self._errors = 0

        self.logger.info(
            "RenderService inicializado",
            max_queue_size=max_queue_size
        )

    def start(self) -> None:
        """Inicia el servicio de renderizado."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._render_loop,
            name="RenderService",
            daemon=True
        )
        self._thread.start()

        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, self.config.camera.width, self.config.camera.height)
        except Exception as e:
            self.logger.warning(f"Error creando ventana: {e}")

        self.logger.info("Servicio de renderizado iniciado")

    def stop(self) -> None:
        """Detiene el servicio de renderizado."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.logger.info("Servicio de renderizado detenido")

    def pause(self) -> None:
        """Pausa el renderizado."""
        self._paused = True
        self.logger.debug("Renderizado pausado")

    def resume(self) -> None:
        """Reanuda el renderizado."""
        self._paused = False
        self.logger.debug("Renderizado reanudado")

    def enqueue_frame(self, result: ProcessingResult) -> None:
        """
        Encola un frame para renderizado.

        Args:
            result: Resultado del procesamiento
        """
        if not self._running or self._paused:
            return

        with self._queue_lock:
            self._render_queue.append(result)

            while len(self._render_queue) > self.max_queue_size:
                dropped = self._render_queue.pop(0)
                self._frames_dropped += 1
                self.logger.debug(
                    f"Frame {dropped.frame_number} descartado de cola de renderizado"
                )

    def _render_loop(self) -> None:
        """Bucle principal de renderizado."""
        self.logger.info("Bucle de renderizado iniciado")

        while self._running and self.controls.is_running:
            try:
                if self._paused:
                    self._render_pause_frame()
                    time.sleep(0.01)
                    continue

                result = self._get_next_frame()
                if result is None:
                    time.sleep(0.001)
                    continue

                self._display_frame(result)

            except Exception as e:
                self._errors += 1
                self.logger.error(f"Error en renderizado: {e}", exc_info=True)
                time.sleep(0.01)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.logger.info("Bucle de renderizado terminado")

    def _get_next_frame(self) -> Optional[ProcessingResult]:
        """Obtiene el siguiente frame de la cola."""
        with self._queue_lock:
            if self._render_queue:
                return self._render_queue.pop(0)
        return None

    def _display_frame(self, result: ProcessingResult) -> None:
        """
        Muestra un frame en la ventana.

        Args:
            result: Resultado del procesamiento
        """
        if result.processed_frame is None or result.processed_frame.size == 0:
            return

        try:
            rendered_frame = self.renderer.render(
                result.processed_frame,
                result.tracks,
                result.stats,
                fps=0.0,
                processing_time_ms=result.processing_time_ms,
                frame_number=result.frame_number
            )

            cv2.imshow(WINDOW_NAME, rendered_frame)
            self._last_valid_frame = rendered_frame
            self._frames_rendered += 1

            key = cv2.waitKey(1) & 0xFF
            if key:
                self.controls.process_key(key)
                if self.on_key_pressed:
                    self.on_key_pressed(key)

        except Exception as e:
            self._errors += 1
            self.logger.error(f"Error mostrando frame: {e}")

    def _render_pause_frame(self) -> None:
        """Muestra un frame de pausa."""
        if self._last_valid_frame is not None:
            try:
                pause_frame = self._last_valid_frame.copy()
                h, w = pause_frame.shape[:2]

                overlay = pause_frame.copy()
                cv2.rectangle(
                    overlay,
                    (w//4, h//3),
                    (3*w//4, 2*h//3),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.5, pause_frame, 0.5, 0, pause_frame)

                cv2.putText(
                    pause_frame,
                    "PAUSADO",
                    (w//2 - 80, h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )
                cv2.putText(
                    pause_frame,
                    "Presiona ESPACIO para reanudar",
                    (w//2 - 120, h//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(WINDOW_NAME, pause_frame)

                key = cv2.waitKey(50) & 0xFF
                if key:
                    self.controls.process_key(key)
                    if self.on_key_pressed:
                        self.on_key_pressed(key)

            except Exception as e:
                self.logger.debug(f"Error mostrando pausa: {e}")

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        return {
            'frames_rendered': self._frames_rendered,
            'frames_dropped': self._frames_dropped,
            'errors': self._errors,
            'queue_size': len(self._render_queue),
            'max_queue_size': self.max_queue_size,
            'is_running': self._running,
            'is_paused': self._paused,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused
