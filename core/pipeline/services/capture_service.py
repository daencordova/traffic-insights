"""Servicio de captura con circuit breaker y manejo robusto de errores.

Proporciona un servicio especializado en la captura de video con
reconexión automática, control de flujo y monitoreo de salud.
"""

from collections.abc import Callable
import logging
import threading
import time

import cv2
import numpy as np

from core.capture.reconnector import Reconnector
from core.circuit_breaker import CircuitBreaker, circuit_breaker_registry
from core.exceptions import CameraError
from core.frame_buffer import FrameBuffer, FrameMetadata
from core.validators import validate_frame
from utils.decorators import RetryConfig, retry_on_failure
from utils.logger import LoggerMixin


class CaptureService(LoggerMixin):
    """Servicio especializado en captura de video.

    Responsabilidades:
        - Conectar y reconectar a la fuente
        - Leer frames de la fuente
        - Almacenar frames en el buffer
        - Monitorear la salud de la conexión
        - Control de flujo basado en uso del buffer

    Attributes:
        config: Configuración del sistema.
        buffer: Buffer circular para frames.
        on_frame_captured: Callback para frames capturados.
        on_frame_dropped: Callback para frames descartados.

    Example:
        >>> service = CaptureService(config)
        >>> service.start("0")
        >>> frame, metadata = service.buffer.get()
        >>> service.stop()
    """

    def __init__(
        self,
        config,
        buffer: FrameBuffer | None = None,
        on_frame_captured: Callable | None = None,
        on_frame_dropped: Callable | None = None,
    ):
        self.config = config
        self.buffer = buffer or self._create_buffer()
        self.on_frame_captured = on_frame_captured
        self.on_frame_dropped = on_frame_dropped

        self._circuit_breaker = CircuitBreaker(
            name="capture_connection",
            failure_threshold=3,
            timeout_seconds=5.0,
            on_state_change=self._on_breaker_state_change,
        )
        circuit_breaker_registry.register(self._circuit_breaker)

        self._reconnector = Reconnector(
            max_attempts=config.camera.reconnect_attempts, delay=config.camera.reconnect_delay
        )
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False

        self._stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "reconnections": 0,
            "errors": 0,
            "fps": 0.0,
            "buffer_usage": 0.0,
            "breaker_state": "closed",
        }

        self.logger.info(
            "CaptureService inicializado con circuit breaker",
            source=config.camera.source,
            breaker_name=self._circuit_breaker.name,
        )

    def _create_buffer(self) -> FrameBuffer:
        """Crea el buffer circular."""
        frame_shape = (self.config.camera.height, self.config.camera.width, 3)
        return FrameBuffer(
            max_size=self.config.camera.buffer_size, frame_shape=frame_shape, drop_policy="oldest"
        )

    def start(self, source: str | None = None) -> None:
        """Inicia el servicio de captura.

        Args:
            source: Fuente de video (opcional). Si es None, usa la configuración.

        Note:
            Inicia un thread dedicado para la captura continua.
            El thread se ejecuta en segundo plano y es daemon.
        """
        if self._running:
            return

        self._running = True
        self._source = source or self.config.camera.source

        self._thread = threading.Thread(
            target=self._capture_loop, name="CaptureService", daemon=True
        )
        self._thread.start()
        self.logger.info("Servicio de captura iniciado")

    def stop(self) -> None:
        """Detiene el servicio de captura.

        Note:
            Espera a que el thread de captura termine (timeout 2s).
            Libera el recurso de captura (VideoCapture).
        """
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        self.logger.info("Servicio de captura detenido")

    def pause(self) -> None:
        """Pausa la captura.

        Note:
            Los frames no se leen mientras está pausado.
            El buffer mantiene los frames existentes.
        """
        self._paused = True
        self.logger.debug("Captura pausada")

    def resume(self) -> None:
        """Reanuda la captura."""
        self._paused = False
        self.logger.debug("Captura reanudada")

    def reconnect(self) -> bool:
        """Reconecta a la fuente de video.

        Returns:
            bool: True si la reconexión fue exitosa.

        Note:
            Libera la captura actual y crea una nueva.
            Aplica reintentos según configuración.
        """
        self.logger.info("Intentando reconexión...")
        if self._cap:
            self._cap.release()
            self._cap = None

        return self._connect()

    def _capture_loop(self) -> None:
        """Bucle principal de captura con control de estado.

        Note:
            Se ejecuta en un thread separado.
            Maneja:
            - Reconexión automática
            - Control de flujo
            - Circuit breaker
            - Frame skipping
        """
        self.logger.info(f"Iniciando bucle de captura desde: {self._source}")

        consecutive_errors = 0
        max_consecutive_errors = 5

        while self._running:
            try:
                if self._paused:
                    time.sleep(0.01)
                    continue

                if not self._circuit_breaker.can_execute():
                    time.sleep(0.5)
                    continue

                if not self._ensure_connected():
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        self._circuit_breaker.record_failure(
                            CameraError("Demasiados errores consecutivos")
                        )
                        consecutive_errors = 0
                    continue

                ret, frame = self._read_frame()
                if not ret or frame is None:
                    consecutive_errors += 1
                    self._handle_read_error()
                    if consecutive_errors > max_consecutive_errors:
                        self._circuit_breaker.record_failure(
                            CameraError("Demasiados errores de lectura")
                        )
                        consecutive_errors = 0
                    continue

                if not self._validate_frame(frame):
                    self.logger.debug("Frame inválido, saltando...")
                    continue

                self._process_frame(frame)
                consecutive_errors = 0
                self._circuit_breaker.record_success()

            except Exception as e:
                self._stats["errors"] += 1
                self.logger.error(f"Error en bucle de captura: {e}", exc_info=True)
                time.sleep(0.1)

        self.logger.info("Bucle de captura terminado")

    def _ensure_connected(self) -> bool:
        """Asegura que la conexión esté activa.

        Returns:
            bool: True si la conexión está activa.

        Note:
            Si la conexión está caída, intenta reconectar.
        """
        if self._cap and self._cap.isOpened():
            return True

        return self._connect()

    retry_config = RetryConfig(
        exceptions=(CameraError, ConnectionError, TimeoutError),
        max_attempts=3,
        delay=0.5,
        backoff=2.0,
        on_retry=lambda attempt, e: logging.warning(f"Reintentando conexión {attempt}: {e}"),
    )

    @retry_on_failure(retry_config)
    def _connect(self) -> bool:
        """Conecta a la fuente de video con reintentos.

        Returns:
            bool: True si la conexión fue exitosa.

        Raises:
            CameraError: Si no se puede conectar después de reintentos.
        """
        try:
            self._cap = self._reconnector.connect(self._source, self.config.camera)

            if self._cap and self._cap.isOpened():
                self._stats["reconnections"] += 1
                self._stats["errors"] = 0
                self._circuit_breaker.record_success()
                self.logger.info("Conexión exitosa a la fuente")
                return True

            raise CameraError(f"No se pudo conectar a la fuente: {self._source}")

        except Exception as e:
            self._circuit_breaker.record_failure(e)
            self.logger.error(f"Error conectando: {e}")
            raise CameraError(f"Fallo en conexión: {e}") from e

    def _read_frame(self) -> tuple:
        """Lee un frame con manejo de errores.

        Returns:
            tuple: (ret, frame) donde ret es booleano.

        Note:
            Maneja errores de OpenCV y otros fallos de lectura.
        """
        try:
            return self._cap.read()
        except cv2.error as e:
            self.logger.error(f"Error de OpenCV: {e}")
            return False, None
        except Exception as e:
            self.logger.error(f"Error leyendo frame: {e}")
            return False, None

    def _handle_read_error(self) -> None:
        """Maneja errores de lectura con recuperación.

        Note:
            Incrementa contador de errores y reconecta si es necesario.
        """
        self.logger.warning("Error leyendo frame, intentando recuperar...")
        self._stats["errors"] += 1

        if self._stats["errors"] > 5:
            self.logger.warning("Demasiados errores, reconectando...")
            self._cap = None
            self._stats["errors"] = 0

    def _on_breaker_state_change(self, name: str, new_state: str) -> None:
        """Callback cuando cambia el estado del circuit breaker.

        Args:
            name: Nombre del circuit breaker.
            new_state: Nuevo estado ('closed', 'open', 'half_open').

        Note:
            Registra el cambio y puede tomar acciones correctivas.
        """
        self._stats["breaker_state"] = new_state
        self.logger.warning(f"Circuit breaker '{name}' cambió a estado: {new_state}")

        if new_state == "open":
            self.logger.error("Conexión bloqueada por circuit breaker. Intentando recuperación...")

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Valida la integridad del frame.

        Args:
            frame: Frame a validar.

        Returns:
            bool: True si el frame es válido.

        Note:
            Verifica que no sea None, tenga tamaño > 0 y dimensiones mínimas.
        """
        return validate_frame(frame, min_width=10, min_height=10)

    def _process_frame(self, frame: np.ndarray) -> None:
        """Procesa y almacena el frame.

        Args:
            frame: Frame capturado.

        Note:
            Crea metadatos y almacena en el buffer.
            Actualiza estadísticas y FPS.
        """
        metadata = FrameMetadata(
            timestamp=time.time(),
            frame_number=self._stats["frames_captured"],
            source_fps=self._stats["fps"],
            capture_time_ms=0.0,
        )

        if not self.buffer.put(frame, metadata):
            self._stats["frames_dropped"] += 1
            if self.on_frame_dropped:
                self.on_frame_dropped(self._stats["frames_captured"])
            return

        self._stats["frames_captured"] += 1
        self._update_fps()

        if self.on_frame_captured:
            self.on_frame_captured(frame, metadata)

    def _update_fps(self) -> None:
        """Actualiza el FPS de captura.

        Note:
            Calcula el FPS basado en el tiempo entre frames.
        """

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio.

        Returns:
            Dict: Estadísticas incluyendo:
                - frames_captured: Frames capturados
                - frames_dropped: Frames descartados
                - reconnections: Número de reconexiones
                - errors: Número de errores
                - fps: FPS actual
                - buffer_usage: Uso del buffer (0-1)
                - breaker_state: Estado del circuit breaker
                - is_running: Si está en ejecución
                - is_paused: Si está pausado
                - is_connected: Si hay conexión activa

        Example:
            >>> stats = service.get_stats()
            >>> print(f"FPS: {stats['fps']:.1f}")
        """
        return {
            **self._stats,
            "buffer_size": len(self.buffer),
            "buffer_usage": self.buffer.count / self.buffer.max_size,
            "is_running": self._running,
            "is_paused": self._paused,
            "is_connected": self._cap is not None and self._cap.isOpened(),
        }
