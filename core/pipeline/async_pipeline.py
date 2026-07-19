"""
Pipeline asíncrono.

Orquesta los servicios de captura, procesamiento y renderizado.
"""

import time
from typing import Optional, Callable

from core.frame_buffer import CircularFrameBuffer
from core.pipeline.controller import PipelineController, FlowControlConfig
from core.pipeline.capture_service import CaptureService, CaptureConfig
from core.pipeline.processing_service import ProcessingService, ProcessingResult
from core.pipeline.render_service import RenderService
from core.pipeline.system_info import set_system_status
from utils.logger import LoggerMixin
from utils.helpers import get_memory_usage, force_garbage_collection
from core.constants import MEMORY_CHECK_INTERVAL, GC_INTERVAL


class AsyncVehicleCountingPipeline(LoggerMixin):
    """
    Pipeline asíncrono de seguimiento de tráfico.

    Orquesta los servicios de captura, procesamiento y renderizado
    en un flujo asíncrono optimizado para CPU.

    Arquitectura:
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Capture    │───▶│  Processing  │───▶│   Render    │
    │   Service    │    │   Service    │    │   Service   │
    └──────────────┘    └──────────────┘    └──────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
                       ┌──────────────┐
                       │  Controller  │
                       └──────────────┘
    """

    def __init__(
        self,
        detector=None,
        tracker=None,
        counter=None,
        buffer_size: int = 30,
        num_workers: int = 4,
        enable_batch_processing: bool = False,
        batch_size: int = 4,
        render_callback: Optional[Callable] = None,
    ):
        """
        Inicializa el pipeline asíncrono.

        Args:
            detector: Detector de objetos (opcional)
            tracker: Tracker de objetos (opcional)
            counter: Contador de vehículos (opcional)
            buffer_size: Tamaño del buffer de frames
            num_workers: Número de workers para procesamiento
            enable_batch_processing: Habilitar procesamiento por lotes
            batch_size: Tamaño del lote
            render_callback: Callback para renderizado personalizado
        """
        from config.manager import config_manager

        self.config = config_manager.config
        self.logger.info("Inicializando AsyncVehicleCountingPipeline")

        is_cpu = self.config.model.device == "cpu"
        if is_cpu:
            buffer_size = min(buffer_size, 20)
            num_workers = min(num_workers, 4)
            enable_batch_processing = True
            batch_size = min(batch_size, 4)
            self.logger.info(
                "Modo CPU - ajustando límites",
                workers=num_workers,
                buffer=buffer_size,
                batch=batch_size
            )

        flow_config = FlowControlConfig(
            drop_threshold=0.8,
            recovery_threshold=0.3,
            max_frame_skip=2,
            consecutive_skip_limit=5,
            min_capture_fps=5.0,
            max_capture_fps=15.0 if is_cpu else 30.0,
        )
        self.controller = PipelineController(
            flow_config=flow_config,
            is_cpu_mode=is_cpu
        )

        frame_shape = (self.config.camera.height, self.config.camera.width, 3)
        self._buffer = CircularFrameBuffer(
            max_size=buffer_size,
            frame_shape=frame_shape,
            drop_policy="oldest"
        )

        self._init_services(
            detector=detector,
            tracker=tracker,
            counter=counter,
            num_workers=num_workers,
            enable_batch=enable_batch_processing,
            batch_size=batch_size,
            render_callback=render_callback,
            is_cpu=is_cpu
        )

        self._last_memory_check = time.time()
        self._last_gc_time = time.time()
        self._gc_interval = GC_INTERVAL

        self._stats = {}
        self._start_time: Optional[float] = None

        self.on_frame_processed: Optional[Callable[[ProcessingResult], None]] = None
        self.on_frame_dropped: Optional[Callable[[int], None]] = None

        self.logger.info(
            "Pipeline asíncrono inicializado",
            buffer_size=buffer_size,
            num_workers=num_workers,
            batch_processing=enable_batch_processing,
            cpu_mode=is_cpu
        )

    def _init_services(
        self,
        detector=None,
        tracker=None,
        counter=None,
        num_workers: int = 4,
        enable_batch: bool = False,
        batch_size: int = 4,
        render_callback: Optional[Callable] = None,
        is_cpu: bool = True
    ) -> None:
        """Inicializa los servicios del pipeline."""

        capture_config = CaptureConfig(
            source=self.config.camera.source,
            width=self.config.camera.width,
            height=self.config.camera.height,
            buffer_size=self._buffer.max_size,
            reconnect_attempts=self.config.camera.reconnect_attempts,
            reconnect_delay=self.config.camera.reconnect_delay,
        )

        self._capture_service = CaptureService(
            config=capture_config,
            buffer=self._buffer,
            controller=self.controller,
            on_frame_captured=self._on_frame_captured,
            on_frame_dropped=self._on_frame_dropped
        )

        self._processing_service = ProcessingService(
            config=self.config,
            detector=detector,
            tracker=tracker,
            counter=counter,
            enable_batch=enable_batch,
            batch_size=batch_size,
            on_frame_processed=self._on_frame_processed
        )

        self._render_service = RenderService(
            config=self.config,
            renderer=None,
            controls=None,
            max_queue_size=min(3, self._buffer.max_size // 3),
            on_key_pressed=self._on_key_pressed
        )

        original_on_frame = self._processing_service.on_frame_processed

        def wrapper(result: ProcessingResult):
            self._render_service.queue_frame(result)
            if original_on_frame:
                original_on_frame(result)

        self._processing_service.on_frame_processed = wrapper

    def start(self, source: Optional[str] = None) -> None:
        """
        Inicia el pipeline.

        Args:
            source: Fuente de video (opcional)
        """
        if self.controller.state.value == "RUNNING":
            self.logger.warning("Pipeline ya está en ejecución")
            return

        self.logger.info("Iniciando pipeline asíncrono")
        self.controller.start()
        self._start_time = time.time()

        if source:
            self._capture_service.config.source = source

        self._capture_service.start()
        self._render_service.start()

        self.logger.info("Pipeline iniciado")

    def stop(self) -> None:
        """Detiene el pipeline."""
        if self.controller.state.value == "STOPPED":
            return

        self.logger.info("Deteniendo pipeline...")
        self.controller.stop()

        self._capture_service.stop()
        self._render_service.stop()

        self.controller.mark_stopped()
        set_system_status("STOPPED")

        self.logger.info("Pipeline detenido")
        self._log_performance_stats()

    def pause(self) -> None:
        """Pausa el pipeline."""
        if self.controller.state.value == "RUNNING":
            self.controller.pause()
            set_system_status("PAUSED")
            self.logger.info("Pipeline pausado")

    def resume(self) -> None:
        """Reanuda el pipeline."""
        if self.controller.state.value == "PAUSED":
            self.controller.resume()
            set_system_status("RUNNING")
            self.logger.info("Pipeline reanudado")

    def _on_frame_captured(self, frame_number: int) -> None:
        """Callback cuando se captura un frame."""
        pass

    def _on_frame_dropped(self, frame_number: int) -> None:
        """Callback cuando se descarta un frame."""
        if self.on_frame_dropped:
            self.on_frame_dropped(frame_number)

    def _on_frame_processed(self, result: ProcessingResult) -> None:
        """Callback cuando se procesa un frame."""
        if self.on_frame_processed:
            self.on_frame_processed(result)

    def _on_key_pressed(self, key: int) -> None:
        """Callback cuando se presiona una tecla."""
        if key == ord('q') or key == 27:
            self.stop()

    def _check_memory(self) -> None:
        """Verifica el uso de memoria."""
        current_time = time.time()

        if current_time - self._last_memory_check >= MEMORY_CHECK_INTERVAL:
            self._last_memory_check = current_time

            mem = get_memory_usage()
            mem_percent = mem.get("percent", 0)

            if mem_percent > 70:
                self.logger.warning(f"Memoria alta: {mem_percent:.1f}%, limpiando...")
                force_garbage_collection()

            if mem_percent > 80:
                self.logger.warning("Memoria crítica, limpieza agresiva")
                import gc
                gc.collect()

        if current_time - self._last_gc_time >= self._gc_interval:
            self._last_gc_time = current_time
            import gc
            gc.collect()

    def get_stats(self) -> dict:
        """Obtiene estadísticas del pipeline."""
        return {
            "state": self.controller.state.name,
            "runtime_seconds": time.time() - self._start_time if self._start_time else 0,
            "fps": self.controller.current_fps,
            "frame_count": self.controller.frame_count,
            "dropped_count": self.controller.dropped_count,
            "buffer": self._buffer.get_stats(),
            "capture": self._capture_service.get_stats(),
            "processing": self._processing_service.get_stats(),
            "render": self._render_service.get_stats(),
            "memory": get_memory_usage(),
            "health": self.controller.get_health_status(),
        }

    def _log_performance_stats(self) -> None:
        """Registra estadísticas de rendimiento."""
        runtime = time.time() - self._start_time if self._start_time else 0
        self.logger.info(
            "Rendimiento final",
            runtime_seconds=f"{runtime:.1f}",
            frames=self.controller.frame_count,
            fps=self.controller.current_fps,
            dropped=self.controller.dropped_count
        )

    @property
    def is_running(self) -> bool:
        return self.controller.is_running

    @property
    def is_paused(self) -> bool:
        return self.controller.is_paused

    @property
    def fps(self) -> float:
        return self.controller.current_fps

    @property
    def total_frames(self) -> int:
        return self.controller.frame_count

    @property
    def dropped_frames(self) -> int:
        return self.controller.dropped_count

    @property
    def is_healthy(self) -> bool:
        return self.controller.get_health_status()["healthy"]

    @property
    def is_cpu_mode(self) -> bool:
        return self.controller.is_cpu_mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
