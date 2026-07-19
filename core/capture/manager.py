"""
Gestor de captura de video con reconexión automática y control de flujo.

Extraído de AsyncVehicleCountingPipeline para mejorar la estructura del código.
"""

import time
import threading
from typing import Optional, Callable

import numpy as np

from core.capture.reconnector import Reconnector
from core.frame_buffer import CircularFrameBuffer, FrameMetadata
from core.validators import validate_frame
from core.constants import (
    CAPTURE_MIN_FPS_CPU,
    CAPTURE_MAX_FPS_CPU,
    CAPTURE_TARGET_FPS_CPU,
    CAPTURE_TARGET_FPS_GPU,
    CAPTURE_DEFAULT_INTERVAL_CPU,
    CAPTURE_DEFAULT_INTERVAL_GPU,
    BUFFER_SKIP_MAX,
    BUFFER_SKIP_CONSECUTIVE_LIMIT,
    BUFFER_DROP_THRESHOLD,
    BUFFER_RECOVERY_THRESHOLD,
    CAPTURE_MAX_CONSECUTIVE_ERRORS,
    MIN_FRAME_DIMENSION,
)
from utils.logger import LoggerMixin


class CaptureManager(LoggerMixin):
    """
    Gestiona la captura de frames desde una fuente de video.

    Responsabilidades:
    - Conexión y reconexión automática
    - Control de flujo con buffer
    - Monitoreo de FPS
    - Gestión de pausa/continuación

    Attributes:
        config: Configuración del sistema.
        buffer: Buffer circular para frames.
        stop_event: Evento para detener la captura.
        pause_event: Evento para pausar la captura.
        fps_target: FPS objetivo.
        is_cpu_mode: Si está en modo CPU.
    """

    def __init__(
        self,
        config,
        buffer: CircularFrameBuffer,
        stop_event: threading.Event,
        pause_event: threading.Event,
        is_cpu_mode: bool = False,
        capture_interval: Optional[float] = None,
    ):
        self.config = config
        self.buffer = buffer
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.is_cpu_mode = is_cpu_mode

        if capture_interval is None:
            capture_interval = CAPTURE_DEFAULT_INTERVAL_CPU if is_cpu_mode else CAPTURE_DEFAULT_INTERVAL_GPU

        self._capture_interval = capture_interval
        self._last_capture_time = time.time()
        self._frame_count = 0
        self._dropped_count = 0
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

        self._reconnector = Reconnector(
            max_attempts=config.camera.reconnect_attempts,
            delay=config.camera.reconnect_delay
        )

        self._flow_control_enabled = True
        self._frame_skip_counter = 0
        self._max_frame_skip = BUFFER_SKIP_MAX
        self._consecutive_skips = 0

        self._min_capture_fps = CAPTURE_MIN_FPS_CPU
        self._max_capture_fps = CAPTURE_MAX_FPS_CPU if is_cpu_mode else CAPTURE_TARGET_FPS_GPU
        self._capture_fps_target = CAPTURE_TARGET_FPS_CPU if is_cpu_mode else CAPTURE_TARGET_FPS_GPU

        self._on_frame_dropped: Optional[Callable[[int], None]] = None
        self._on_frame_captured: Optional[Callable[[int], None]] = None

        self._max_consecutive_errors = CAPTURE_MAX_CONSECUTIVE_ERRORS

        self.logger.info(
            "CaptureManager inicializado",
            is_cpu_mode=is_cpu_mode,
            capture_fps_target=self._capture_fps_target,
            capture_interval=self._capture_interval
        )

    def run(self, source: Optional[str] = None) -> None:
        """
        Bucle principal de captura.

        Args:
            source: Fuente de video (opcional).
        """
        source = source or self.config.camera.source
        self.logger.info(f"Iniciando captura desde: {source}")

        cap = None
        consecutive_errors = 0

        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - self._last_capture_time < self._capture_interval:
                    time.sleep(0.001)
                    continue
                self._last_capture_time = current_time

                if self.pause_event.is_set():
                    time.sleep(0.01)
                    continue

                if cap is None or not cap.isOpened():
                    cap = self._reconnector.connect(source, self.config.camera)
                    if cap is None:
                        self.logger.warning("No se pudo conectar, reintentando...")
                        time.sleep(self._reconnector.delay)
                        consecutive_errors += 1
                        if consecutive_errors > self._max_consecutive_errors:
                            self._add_health_issue("Fallo de conexión persistente a la fuente")
                            consecutive_errors = 0
                        continue

                    consecutive_errors = 0
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        self.logger.warning("Frame de prueba falló, reconectando...")
                        cap.release()
                        cap = None
                        continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    self.logger.warning("Error leyendo frame, reconectando...")
                    cap.release()
                    cap = None
                    continue

                if not validate_frame(frame, min_width=MIN_FRAME_DIMENSION, min_height=MIN_FRAME_DIMENSION):
                    self.logger.debug("Frame inválido, saltando...")
                    continue

                if not self._apply_flow_control():
                    continue

                self._store_frame(frame)

            except Exception as e:
                self.logger.error(f"Error en captura: {e}", exc_info=True)
                if cap:
                    cap.release()
                    cap = None
                time.sleep(self._reconnector.delay)

        if cap:
            cap.release()

        self.logger.info("Bucle de captura terminado")

    def _apply_flow_control(self) -> bool:
        """
        Aplica control de flujo basado en el estado del buffer.

        Returns:
            bool: True si el frame debe ser procesado.
        """
        if not self._flow_control_enabled:
            return True

        buffer_usage = self.buffer.count / self.buffer.max_size if self.buffer.max_size > 0 else 0

        if buffer_usage > BUFFER_DROP_THRESHOLD:
            self._frame_skip_counter += 1
            if self._frame_skip_counter < self._max_frame_skip:
                self._dropped_count += 1
                self._consecutive_skips += 1
                if self._consecutive_skips > BUFFER_SKIP_CONSECUTIVE_LIMIT:
                    self._add_health_issue(f"Buffer crítico: {buffer_usage*100:.1f}%")
                    if self.is_cpu_mode:
                        self._capture_fps_target = max(
                            self._min_capture_fps,
                            self._capture_fps_target * 0.9
                        )
                        self._capture_interval = 1.0 / self._capture_fps_target
                if self._on_frame_dropped:
                    self._on_frame_dropped(self._frame_count)
                return False
            else:
                self._frame_skip_counter = 0
                self._consecutive_skips = max(0, self._consecutive_skips - 2)

        elif buffer_usage < BUFFER_RECOVERY_THRESHOLD:
            self._frame_skip_counter = 0
            self._consecutive_skips = max(0, self._consecutive_skips - 2)
            if self._capture_fps_target < self._max_capture_fps:
                self._capture_fps_target = min(
                    self._max_capture_fps,
                    self._capture_fps_target + 0.5
                )
                self._capture_interval = 1.0 / self._capture_fps_target

        elif buffer_usage < 0.6:
            self._consecutive_skips = max(0, self._consecutive_skips - 1)

        return True

    def _store_frame(self, frame: np.ndarray) -> None:
        """
        Almacena un frame en el buffer.

        Args:
            frame: Frame a almacenar.
        """
        metadata = FrameMetadata(
            timestamp=time.time(),
            frame_number=self._frame_count,
            source_fps=self._current_fps,
            capture_time_ms=0.0,
        )

        if not self.buffer.put(frame, metadata):
            self._dropped_count += 1
            if self._on_frame_dropped:
                self._on_frame_dropped(self._frame_count)
            self.logger.debug(f"Frame {self._frame_count} descartado (buffer lleno)")
            return

        self._frame_count += 1
        if self._on_frame_captured:
            self._on_frame_captured(self._frame_count)

        self._update_fps()

    def _update_fps(self) -> None:
        """Actualiza el contador de FPS."""
        self._fps_counter += 1
        if time.time() - self._fps_timer >= 1.0:
            self._current_fps = self._fps_counter
            self._fps_counter = 0
            self._fps_timer = time.time()

    def _add_health_issue(self, issue: str) -> None:
        """Registra un problema de salud del sistema."""
        timestamp = time.strftime("%H:%M:%S")
        self.logger.warning(f"[{timestamp}] {issue}")

    @property
    def frame_count(self) -> int:
        """Número de frames capturados."""
        return self._frame_count

    @property
    def dropped_count(self) -> int:
        """Número de frames descartados."""
        return self._dropped_count

    @property
    def current_fps(self) -> float:
        """FPS actual de captura."""
        return self._current_fps

    def set_on_frame_dropped(self, callback: Callable[[int], None]) -> None:
        """Establece callback para frames descartados."""
        self._on_frame_dropped = callback

    def set_on_frame_captured(self, callback: Callable[[int], None]) -> None:
        """Establece callback para frames capturados."""
        self._on_frame_captured = callback

    def get_stats(self) -> dict:
        """Obtiene estadísticas del capturador."""
        return {
            "frames_captured": self._frame_count,
            "frames_dropped": self._dropped_count,
            "current_fps": self._current_fps,
            "capture_fps_target": self._capture_fps_target,
            "capture_interval": self._capture_interval,
            "is_paused": self.pause_event.is_set(),
            "is_running": not self.stop_event.is_set(),
        }

    def stop(self) -> None:
        """Detiene la captura."""
        self.stop_event.set()
        self.logger.info("Captura detenida")

    def pause(self) -> None:
        """Pausa la captura."""
        self.pause_event.set()
        self.logger.info("Captura pausada")

    def resume(self) -> None:
        """Reanuda la captura."""
        self.pause_event.clear()
        self.logger.info("Captura reanudada")
