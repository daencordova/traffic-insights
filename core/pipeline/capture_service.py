"""
Servicio de captura de video.

Maneja la conexión a la fuente de video, la lectura de frames
y el almacenamiento en el buffer circular.
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass

import cv2
import numpy as np

from core.frame_buffer import CircularFrameBuffer, FrameMetadata
from core.capture.reconnector import Reconnector
from core.validators import validate_frame
from utils.logger import LoggerMixin
from core.constants import CAPTURE_MAX_CONSECUTIVE_ERRORS


@dataclass
class CaptureConfig:
    """Configuración del servicio de captura."""
    source: str = "0"
    width: int = 640
    height: int = 480
    buffer_size: int = 30
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0


class CaptureService(LoggerMixin):
    """
    Servicio de captura de video.

    Responsabilidades:
    - Conectar y reconectar a la fuente de video
    - Leer frames de la fuente
    - Almacenar frames en el buffer circular
    - Monitorear la salud de la fuente

    Attributes:
        config: Configuración de captura
        buffer: Buffer circular para frames
        controller: Controlador del pipeline
        on_frame_captured: Callback opcional al capturar un frame
        on_frame_dropped: Callback opcional al descartar un frame
    """

    def __init__(
        self,
        config: CaptureConfig,
        buffer: CircularFrameBuffer,
        controller,
        on_frame_captured: Optional[Callable[[int], None]] = None,
        on_frame_dropped: Optional[Callable[[int], None]] = None,
    ):
        self.config = config
        self.buffer = buffer
        self.controller = controller

        self.on_frame_captured = on_frame_captured
        self.on_frame_dropped = on_frame_dropped

        self._reconnector = Reconnector(
            max_attempts=config.reconnect_attempts,
            delay=config.reconnect_delay
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = CAPTURE_MAX_CONSECUTIVE_ERRORS

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self.logger.info(
            "CaptureService inicializado",
            source=config.source,
            buffer_size=config.buffer_size
        )

    def start(self) -> None:
        """Inicia el hilo de captura."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CaptureService",
            daemon=True
        )
        self._thread.start()
        self.logger.info("Hilo de captura iniciado")

    def stop(self) -> None:
        """Detiene el hilo de captura."""
        self._running = False
        self.controller.stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        self.logger.info("Hilo de captura detenido")

    def _capture_loop(self) -> None:
        """Bucle principal de captura."""
        self.logger.info(f"Capturando desde: {self.config.source}")

        while self._running and not self.controller.stop_event.is_set():
            try:
                if not self.controller.can_capture_frame():
                    time.sleep(0.001)
                    continue

                if self._cap is None or not self._cap.isOpened():
                    if not self._connect():
                        time.sleep(self.config.reconnect_delay)
                        continue

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self._handle_read_error()
                    continue

                if not self._validate_frame(frame):
                    self.logger.debug("Frame inválido, saltando...")
                    continue

                buffer_usage = self._get_buffer_usage()
                if not self.controller.should_process_frame(buffer_usage):
                    if self.on_frame_dropped:
                        self.on_frame_dropped(self.controller.frame_count)
                    continue

                self._store_frame(frame)

                self.controller.increment_frame_count()
                self.controller.update_fps()

            except Exception as e:
                self.logger.error(f"Error en captura: {e}", exc_info=True)
                self._consecutive_errors += 1
                if self._consecutive_errors > self._max_consecutive_errors:
                    self.controller.mark_error()
                    break
                time.sleep(self.config.reconnect_delay)

        self.logger.info("Bucle de captura terminado")

    def _connect(self) -> bool:
        """
        Conecta a la fuente de video.

        Returns:
            bool: True si la conexión fue exitosa.
        """
        try:
            self._cap = self._reconnector.connect(
                self.config.source,
                self.config
            )

            if self._cap and self._cap.isOpened():
                self._consecutive_errors = 0
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error conectando: {e}")
            return False

    def _handle_read_error(self) -> None:
        """Maneja errores de lectura de frames."""
        self.logger.debug("Error leyendo frame, intentando recuperar...")

        if self._cap:
            self._cap.release()
            self._cap = None

        self._consecutive_errors += 1

        if self._consecutive_errors > self._max_consecutive_errors:
            self.controller.mark_error()

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Valida que el frame sea válido."""
        return validate_frame(frame, min_width=10, min_height=10)

    def _get_buffer_usage(self) -> float:
        """Obtiene el uso del buffer."""
        max_size = self.buffer.max_size or 1
        return self.buffer.count / max_size

    def _store_frame(self, frame: np.ndarray) -> None:
        """
        Almacena un frame en el buffer.

        Args:
            frame: Frame a almacenar.
        """
        metadata = FrameMetadata(
            timestamp=time.time(),
            frame_number=self.controller.frame_count,
            source_fps=0.0,
            capture_time_ms=0.0,
        )

        if not self.buffer.put(frame, metadata):
            if self.on_frame_dropped:
                self.on_frame_dropped(self.controller.frame_count)

        if self.on_frame_captured:
            self.on_frame_captured(self.controller.frame_count)

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio de captura."""
        return {
            "connected": self._cap is not None and self._cap.isOpened(),
            "consecutive_errors": self._consecutive_errors,
            "is_running": self._running,
        }
