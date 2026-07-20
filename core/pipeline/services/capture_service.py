"""
Servicio de captura de video.
"""

import time
import threading
from typing import Optional, Callable

import cv2
import numpy as np

from core.capture.reconnector import Reconnector
from core.frame_buffer import CircularFrameBuffer, FrameMetadata
from core.validators import validate_frame
from utils.logger import LoggerMixin


class CaptureService(LoggerMixin):
    """
    Servicio especializado en captura de video.

    Responsabilidades:
    - Conectar y reconectar a la fuente
    - Leer frames de la fuente
    - Almacenar frames en el buffer
    - Monitorear la salud de la conexión
    """

    def __init__(
        self,
        config,
        buffer: Optional[CircularFrameBuffer] = None,
        on_frame_captured: Optional[Callable] = None,
        on_frame_dropped: Optional[Callable] = None
    ):
        self.config = config
        self.buffer = buffer or self._create_buffer()
        self.on_frame_captured = on_frame_captured
        self.on_frame_dropped = on_frame_dropped

        self._reconnector = Reconnector(
            max_attempts=config.camera.reconnect_attempts,
            delay=config.camera.reconnect_delay
        )
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False

        self._stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'reconnections': 0,
            'errors': 0,
            'fps': 0.0,
            'buffer_usage': 0.0,
        }

        self.logger.info(
            "CaptureService inicializado",
            source=config.camera.source
        )

    def _create_buffer(self) -> CircularFrameBuffer:
        """Crea el buffer circular."""
        frame_shape = (
            self.config.camera.height,
            self.config.camera.width,
            3
        )
        return CircularFrameBuffer(
            max_size=self.config.camera.buffer_size,
            frame_shape=frame_shape,
            drop_policy="oldest"
        )

    def start(self, source: Optional[str] = None) -> None:
        """Inicia el servicio de captura."""
        if self._running:
            return

        self._running = True
        self._source = source or self.config.camera.source

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CaptureService",
            daemon=True
        )
        self._thread.start()
        self.logger.info("Servicio de captura iniciado")

    def stop(self) -> None:
        """Detiene el servicio de captura."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        self.logger.info("Servicio de captura detenido")

    def pause(self) -> None:
        """Pausa la captura."""
        self._paused = True
        self.logger.debug("Captura pausada")

    def resume(self) -> None:
        """Reanuda la captura."""
        self._paused = False
        self.logger.debug("Captura reanudada")

    def reconnect(self) -> bool:
        """Reconecta a la fuente de video."""
        self.logger.info("Intentando reconexión...")
        if self._cap:
            self._cap.release()
            self._cap = None

        return self._connect()

    def _capture_loop(self) -> None:
        """Bucle principal de captura."""
        self.logger.info(f"Capturando desde: {self._source}")

        while self._running:
            try:
                if self._paused:
                    time.sleep(0.01)
                    continue

                if not self._ensure_connected():
                    time.sleep(0.5)
                    continue

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self._handle_read_error()
                    continue

                if not self._validate_frame(frame):
                    self.logger.debug("Frame inválido, saltando...")
                    continue

                self._process_frame(frame)

            except Exception as e:
                self._stats['errors'] += 1
                self.logger.error(f"Error en captura: {e}", exc_info=True)
                time.sleep(0.1)

        self.logger.info("Bucle de captura terminado")

    def _ensure_connected(self) -> bool:
        """Asegura que la conexión esté activa."""
        if self._cap and self._cap.isOpened():
            return True

        return self._connect()

    def _connect(self) -> bool:
        """Conecta a la fuente de video."""
        try:
            self._cap = self._reconnector.connect(
                self._source,
                self.config.camera
            )

            if self._cap and self._cap.isOpened():
                self._stats['reconnections'] += 1
                self._stats['errors'] = 0
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error conectando: {e}")
            return False

    def _handle_read_error(self) -> None:
        """Maneja errores de lectura."""
        self.logger.warning("Error leyendo frame, intentando recuperar...")
        self._stats['errors'] += 1

        if self._stats['errors'] > 5:
            self.logger.warning("Demasiados errores, reconectando...")
            self._cap = None
            self._stats['errors'] = 0

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Valida la integridad del frame."""
        return validate_frame(frame, min_width=10, min_height=10)

    def _process_frame(self, frame: np.ndarray) -> None:
        """Procesa y almacena el frame."""
        metadata = FrameMetadata(
            timestamp=time.time(),
            frame_number=self._stats['frames_captured'],
            source_fps=self._stats['fps'],
            capture_time_ms=0.0
        )

        if not self.buffer.put(frame, metadata):
            self._stats['frames_dropped'] += 1
            if self.on_frame_dropped:
                self.on_frame_dropped(self._stats['frames_captured'])
            return

        self._stats['frames_captured'] += 1
        self._update_fps()

        if self.on_frame_captured:
            self.on_frame_captured(frame, metadata)

    def _update_fps(self) -> None:
        """Actualiza el FPS de captura."""
        pass

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        return {
            **self._stats,
            'buffer_size': len(self.buffer),
            'buffer_usage': self.buffer.count / self.buffer.max_size,
            'is_running': self._running,
            'is_paused': self._paused,
            'is_connected': self._cap is not None and self._cap.isOpened(),
        }
